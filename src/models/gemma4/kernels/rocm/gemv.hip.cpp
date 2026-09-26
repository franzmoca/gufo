// Autoregressive decode projections for Gemma 4's K-quant weights.
//
// y[m] = sum_k x[k] * W[m][k]; the weights stay in their GGUF blocks and the
// activations stay FP32. A lane's unit of work is one 16-byte vector of
// quantized values -- 32 values of a Q4_K/Q5_K super-block (with their Q5_K
// high bits) or 64 values of a Q6_K super-block -- dequantized once and
// contracted in a fixed FMA order. Lanes holding the same task of different
// rows read the same activations, which the hardware broadcasts; each row
// reduces over its task lanes with a fixed xor tree.
#include "src/models/gemma4/kernels/rocm/gemv.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstdint>

namespace gufo::models::gemma4::rocm {
namespace {

constexpr int kWave = 32;
constexpr int kWavesPerBlock = 8;

__device__ __forceinline__ float Half(const std::uint8_t* p) {
  return __half2float(*reinterpret_cast<const __half*>(p));
}

/// ggml get_scale_min_k4 over the 12 scale bytes read as three aligned
/// words (s[0..3], s[4..7], s[8..11]): 6-bit scale and min of 32-value
/// sub-block j, extracted with shifts instead of byte indexing.
__device__ __forceinline__ void ScaleMin(const std::uint8_t* s, int j,
                                         float* sc, float* m) {
  const auto* w = reinterpret_cast<const std::uint32_t*>(s);
  const std::uint32_t shift = 8U * static_cast<std::uint32_t>(j & 3);
  const std::uint32_t a = (w[0] >> shift) & 0xFFU;
  const std::uint32_t b = (w[1] >> shift) & 0xFFU;
  if (j < 4) {
    *sc = static_cast<float>(a & 63U);
    *m = static_cast<float>(b & 63U);
  } else {
    const std::uint32_t c = (w[2] >> shift) & 0xFFU;
    *sc = static_cast<float>((c & 0xFU) | ((a >> 6) << 4));
    *m = static_cast<float>((c >> 4) | ((b >> 6) << 4));
  }
}

template<int B>
struct Accumulators {
  float v[B];
};

/// Dequantizes 16 values packed per byte (optional high bits) as
/// scale * v + offset, once per task however many activation rows follow.
__device__ __forceinline__ void Dequant16(const std::uint8_t* q, int shift,
                                          const std::uint8_t* high, int hshift,
                                          int hmask, float scale, float offset,
                                          float* out) {
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    int v = (q[i] >> shift) & 0xF;
    if (high != nullptr) {
      v |= ((high[i] >> hshift) & hmask) << 4;
    }
    out[i] = __builtin_fmaf(scale, static_cast<float>(v), offset);
  }
}

/// acc += sum_i w[i] * x[i] as one FMA chain over n 64-byte-aligned values.
template<int N>
__device__ __forceinline__ float Accumulate(float acc, const float* w,
                                            const float* x) {
#pragma unroll
  for (int i = 0; i < N; i += 4) {
    const float4 f = *reinterpret_cast<const float4*>(x + i);
    acc = __builtin_fmaf(w[i], f.x, acc);
    acc = __builtin_fmaf(w[i + 1], f.y, acc);
    acc = __builtin_fmaf(w[i + 2], f.z, acc);
    acc = __builtin_fmaf(w[i + 3], f.w, acc);
  }
  return acc;
}

// Q4_K: 8 tasks per 256-value super-block. Task t covers qs bytes
// [32 p + 16 h, +16) with p = t / 2, h = t % 2: low nibbles are values
// 64 p + 16 h + i of sub-block 2 p, high nibbles 64 p + 32 + 16 h + i of
// sub-block 2 p + 1. Q5_K adds the fifth bit of value 16 h + i of pair p
// from bit 2 p (low) or 2 p + 1 (high) of qh[16 h + i].
template<int B, bool kFiveBit>
__device__ __forceinline__ void TaskQ45K(const std::uint8_t* block, int t,
                                         const float* const* x, std::size_t k0,
                                         Accumulators<B>& acc) {
  const int p = t >> 1;
  const int h = t & 1;
  const float d = Half(block);
  const float dmin = Half(block + 2);
  float sc0, m0, sc1, m1;
  ScaleMin(block + 4, 2 * p, &sc0, &m0);
  ScaleMin(block + 4, 2 * p + 1, &sc1, &m1);
  const int qs = kFiveBit ? 48 : 16;
  const uint4 raw_q =
      *reinterpret_cast<const uint4*>(block + qs + 32 * p + 16 * h);
  const auto* q = reinterpret_cast<const std::uint8_t*>(&raw_q);
  uint4 raw_h{};
  const std::uint8_t* qh = nullptr;
  if constexpr (kFiveBit) {
    raw_h = *reinterpret_cast<const uint4*>(block + 16 + 16 * h);
    qh = reinterpret_cast<const std::uint8_t*>(&raw_h);
  }
  float w[32];
  Dequant16(q, 0, qh, 2 * p, 1, d * sc0, -(dmin * m0), w);
  Dequant16(q, 4, qh, 2 * p + 1, 1, d * sc1, -(dmin * m1), w + 16);
  const std::size_t v0 = k0 + 64 * p + 16 * h;
#pragma unroll
  for (int b = 0; b < B; ++b) {
    float a = Accumulate<16>(acc.v[b], w, x[b] + v0);
    acc.v[b] = Accumulate<16>(a, w + 16, x[b] + v0 + 32);
  }
}

// Q6_K: 4 tasks per super-block. Task t = 2 n + h covers ql bytes
// [64 n + 16 h, +16) and [64 n + 32 + 16 h, +16) and qh bytes
// [32 n + 16 h, +16): values 128 n + 16 h + i + {0, 32, 64, 96} with signed
// scales sc[8 n + h + {0, 2, 4, 6}] and an offset of 32.
template<int B>
__device__ __forceinline__ void TaskQ6K(const std::uint8_t* block, int t,
                                        const float* const* x, std::size_t k0,
                                        Accumulators<B>& acc) {
  const int n = t >> 1;
  const int h = t & 1;
  const float d = Half(block + 208);
  const auto* sc =
      reinterpret_cast<const std::int8_t*>(block + 192) + 8 * n + h;
  const uint4 raw_a = *reinterpret_cast<const uint4*>(block + 64 * n + 16 * h);
  const uint4 raw_b =
      *reinterpret_cast<const uint4*>(block + 64 * n + 32 + 16 * h);
  const uint4 raw_h =
      *reinterpret_cast<const uint4*>(block + 128 + 32 * n + 16 * h);
  const auto* ql_a = reinterpret_cast<const std::uint8_t*>(&raw_a);
  const auto* ql_b = reinterpret_cast<const std::uint8_t*>(&raw_b);
  const auto* qh = reinterpret_cast<const std::uint8_t*>(&raw_h);
  float w[64];
  const float s0 = d * static_cast<float>(sc[0]);
  const float s1 = d * static_cast<float>(sc[2]);
  const float s2 = d * static_cast<float>(sc[4]);
  const float s3 = d * static_cast<float>(sc[6]);
  // (q - 32) * s = s * q - 32 s.
  Dequant16(ql_a, 0, qh, 0, 3, s0, -32.0F * s0, w);
  Dequant16(ql_b, 0, qh, 2, 3, s1, -32.0F * s1, w + 16);
  Dequant16(ql_a, 4, qh, 4, 3, s2, -32.0F * s2, w + 32);
  Dequant16(ql_b, 4, qh, 6, 3, s3, -32.0F * s3, w + 48);
  const std::size_t v0 = k0 + 128 * n + 16 * h;
#pragma unroll
  for (int b = 0; b < B; ++b) {
    const float* xb = x[b] + v0;
    float a = Accumulate<16>(acc.v[b], w, xb);
    a = Accumulate<16>(a, w + 16, xb + 32);
    a = Accumulate<16>(a, w + 32, xb + 64);
    acc.v[b] = Accumulate<16>(a, w + 48, xb + 96);
  }
}

template<GemvFormat F>
struct FormatTraits;
template<>
struct FormatTraits<GemvFormat::kQ4_K> {
  static constexpr int kBlockBytes = 144;
  static constexpr int kTasks = 8;
};
template<>
struct FormatTraits<GemvFormat::kQ5_K> {
  static constexpr int kBlockBytes = 176;
  static constexpr int kTasks = 8;
};
template<>
struct FormatTraits<GemvFormat::kQ6_K> {
  static constexpr int kBlockBytes = 210;
  static constexpr int kTasks = 4;
};

/// Lanes sharing a task position read the same activations, so the wave
/// covers 32 / kTasks rows at once: lane = row_in_wave * kTasks + task. Each
/// row's partials reduce over its kTasks lanes with a fixed xor tree.
template<GemvFormat F, int B>
__global__ void __launch_bounds__(kWave* kWavesPerBlock)
    KQuantGemvKernel(const std::uint8_t* __restrict__ w,
                     const float* __restrict__ x, float* __restrict__ y,
                     std::uint32_t m, std::uint32_t k) {
  using T = FormatTraits<F>;
  constexpr int kRowsPerWave = kWave / T::kTasks;
  const int lane = threadIdx.x % kWave;
  const int wave = threadIdx.x / kWave;
  const int t = lane % T::kTasks;
  const std::uint32_t row =
      (blockIdx.x * kWavesPerBlock + wave) * kRowsPerWave + lane / T::kTasks;
  const std::uint32_t blocks_per_row = k / 256;
  const std::size_t row_bytes =
      static_cast<std::size_t>(blocks_per_row) * T::kBlockBytes;
  const float* xs[B];
#pragma unroll
  for (int b = 0; b < B; ++b) {
    xs[b] = x + static_cast<std::size_t>(b) * k;
  }
  Accumulators<B> acc;
#pragma unroll
  for (int b = 0; b < B; ++b) {
    acc.v[b] = 0.0F;
  }
  if (row < m) {
    const std::uint8_t* base = w + row * row_bytes;
#pragma unroll 2
    for (std::uint32_t blk = 0; blk < blocks_per_row; ++blk) {
      const std::uint8_t* block = base + blk * T::kBlockBytes;
      const std::size_t k0 = static_cast<std::size_t>(blk) * 256;
      if constexpr (F == GemvFormat::kQ4_K) {
        TaskQ45K<B, false>(block, t, xs, k0, acc);
      } else if constexpr (F == GemvFormat::kQ5_K) {
        TaskQ45K<B, true>(block, t, xs, k0, acc);
      } else {
        TaskQ6K<B>(block, t, xs, k0, acc);
      }
    }
  }
#pragma unroll
  for (int b = 0; b < B; ++b) {
    float v = acc.v[b];
#pragma unroll
    for (int offset = T::kTasks / 2; offset > 0; offset >>= 1) {
      v += __shfl_xor(v, offset, kWave);
    }
    if (t == 0 && row < m) {
      y[static_cast<std::size_t>(b) * m + row] = v;
    }
  }
}

template<GemvFormat F, int B>
void Launch(const void* w, const float* x, float* y, std::uint32_t m,
            std::uint32_t k, hipStream_t stream) {
  constexpr int kRowsPerBlock =
      (kWave / FormatTraits<F>::kTasks) * kWavesPerBlock;
  const unsigned blocks = (m + kRowsPerBlock - 1) / kRowsPerBlock;
  KQuantGemvKernel<F, B><<<blocks, kWave * kWavesPerBlock, 0, stream>>>(
      static_cast<const std::uint8_t*>(w), x, y, m, k);
}

}  // namespace

bool LaunchKQuantGemv(GemvFormat format, const void* w, const float* x,
                      float* y, std::uint32_t m, std::uint32_t k,
                      hipStream_t stream) {
  if (k % 256 != 0) {
    return false;
  }
  switch (format) {
    case GemvFormat::kQ4_K:
      Launch<GemvFormat::kQ4_K, 1>(w, x, y, m, k, stream);
      return true;
    case GemvFormat::kQ5_K:
      Launch<GemvFormat::kQ5_K, 1>(w, x, y, m, k, stream);
      return true;
    case GemvFormat::kQ6_K:
      Launch<GemvFormat::kQ6_K, 1>(w, x, y, m, k, stream);
      return true;
  }
  return false;
}

}  // namespace gufo::models::gemma4::rocm
