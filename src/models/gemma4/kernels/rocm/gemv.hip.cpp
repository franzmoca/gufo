// Decode and verification projections for Gemma 4's K-quant weights.
//
// Y[b][m] = sum_k X[b][k] * W[m][k] for up to kMaxGemvRows activation rows
// sharing every weight load. The weights stay in their GGUF blocks; the
// activations stay FP32. A lane's unit of work is one 16-byte vector of
// quantized values -- 32 values of a Q4_K/Q5_K super-block (with their
// Q5_K high bits) or 64 values of a Q6_K super-block -- so each lane issues
// wide loads and unpacks nibbles with word operations. Per row, tasks are
// strided over the 32 lanes of one wave in a fixed order and reduced with a
// fixed xor tree, so every activation row's result is independent of how
// many rows share the launch: verification reproduces decode bit for bit.
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

/// ggml get_scale_min_k4: 6-bit scale and min of 32-value sub-block j.
__device__ __forceinline__ void ScaleMin(const std::uint8_t* s, int j,
                                         float* sc, float* m) {
  if (j < 4) {
    *sc = static_cast<float>(s[j] & 63);
    *m = static_cast<float>(s[j + 4] & 63);
  } else {
    *sc = static_cast<float>((s[j + 4] & 0xF) | ((s[j - 4] >> 6) << 4));
    *m = static_cast<float>((s[j + 4] >> 4) | ((s[j] >> 6) << 4));
  }
}

/// Dot of 16 unsigned 4/5/6-bit values packed per byte with 16 activations,
/// and the activations' sum (for the block minimum).
struct Partial {
  float dot;
  float sum;
};

__device__ __forceinline__ Partial Dot16(const std::uint8_t* q, const float* x,
                                         int shift, int mask,
                                         const std::uint8_t* high, int hshift,
                                         int hmask, int hleft) {
  float dot = 0.0F;
  float sum = 0.0F;
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    int v = (q[i] >> shift) & mask;
    if (high != nullptr) {
      v |= ((high[i] >> hshift) & hmask) << hleft;
    }
    dot = __builtin_fmaf(static_cast<float>(v), x[i], dot);
    sum = __builtin_fmaf(1.0F, x[i], sum);
  }
  return {dot, sum};
}

/// acc + s0 * lo.dot - m0 * lo.sum + s1 * hi.dot - m1 * hi.sum as an explicit
/// FMA chain, so every width rounds identically.
__device__ __forceinline__ float Combine(float acc, float s0, Partial lo,
                                         float m0, float s1, Partial hi,
                                         float m1) {
  acc = __builtin_fmaf(s0, lo.dot, acc);
  acc = __builtin_fmaf(-m0, lo.sum, acc);
  acc = __builtin_fmaf(s1, hi.dot, acc);
  return __builtin_fmaf(-m1, hi.sum, acc);
}

template<int B>
struct Accumulators {
  float v[B];
};

// Q4_K: 8 tasks per 256-value super-block. Task t covers qs bytes
// [32 p + 16 h, +16) with p = t / 2, h = t % 2: low nibbles are values
// 64 p + 16 h + i of sub-block 2 p, high nibbles 64 p + 32 + 16 h + i of
// sub-block 2 p + 1.
template<int B>
__device__ __forceinline__ void TaskQ4K(const std::uint8_t* block, int t,
                                        const float* const* x, std::size_t k0,
                                        Accumulators<B>& acc) {
  const int p = t >> 1;
  const int h = t & 1;
  const float d = Half(block);
  const float dmin = Half(block + 2);
  float sc0, m0, sc1, m1;
  ScaleMin(block + 4, 2 * p, &sc0, &m0);
  ScaleMin(block + 4, 2 * p + 1, &sc1, &m1);
  const uint4 raw =
      *reinterpret_cast<const uint4*>(block + 16 + 32 * p + 16 * h);
  const auto* q = reinterpret_cast<const std::uint8_t*>(&raw);
  const std::size_t v0 = k0 + 64 * p + 16 * h;
#pragma unroll
  for (int b = 0; b < B; ++b) {
    const Partial lo = Dot16(q, x[b] + v0, 0, 0xF, nullptr, 0, 0, 0);
    const Partial hi = Dot16(q, x[b] + v0 + 32, 4, 0xF, nullptr, 0, 0, 0);
    acc.v[b] =
        Combine(acc.v[b], d * sc0, lo, dmin * m0, d * sc1, hi, dmin * m1);
  }
}

// Q5_K: as Q4_K with the fifth bit of value 16 h + i of pair p in bit 2 p
// (low nibble) or 2 p + 1 (high nibble) of qh[16 h + i].
template<int B>
__device__ __forceinline__ void TaskQ5K(const std::uint8_t* block, int t,
                                        const float* const* x, std::size_t k0,
                                        Accumulators<B>& acc) {
  const int p = t >> 1;
  const int h = t & 1;
  const float d = Half(block);
  const float dmin = Half(block + 2);
  float sc0, m0, sc1, m1;
  ScaleMin(block + 4, 2 * p, &sc0, &m0);
  ScaleMin(block + 4, 2 * p + 1, &sc1, &m1);
  const uint4 raw_q =
      *reinterpret_cast<const uint4*>(block + 48 + 32 * p + 16 * h);
  const uint4 raw_h = *reinterpret_cast<const uint4*>(block + 16 + 16 * h);
  const auto* q = reinterpret_cast<const std::uint8_t*>(&raw_q);
  const auto* qh = reinterpret_cast<const std::uint8_t*>(&raw_h);
  const std::size_t v0 = k0 + 64 * p + 16 * h;
#pragma unroll
  for (int b = 0; b < B; ++b) {
    const Partial lo = Dot16(q, x[b] + v0, 0, 0xF, qh, 2 * p, 1, 4);
    const Partial hi = Dot16(q, x[b] + v0 + 32, 4, 0xF, qh, 2 * p + 1, 1, 4);
    acc.v[b] =
        Combine(acc.v[b], d * sc0, lo, dmin * m0, d * sc1, hi, dmin * m1);
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
  const std::size_t v0 = k0 + 128 * n + 16 * h;
  const float s0 = d * static_cast<float>(sc[0]);
  const float s1 = d * static_cast<float>(sc[2]);
  const float s2 = d * static_cast<float>(sc[4]);
  const float s3 = d * static_cast<float>(sc[6]);
#pragma unroll
  for (int b = 0; b < B; ++b) {
    const float* xb = x[b] + v0;
    const Partial p0 = Dot16(ql_a, xb, 0, 0xF, qh, 0, 3, 4);
    const Partial p1 = Dot16(ql_b, xb + 32, 0, 0xF, qh, 2, 3, 4);
    const Partial p2 = Dot16(ql_a, xb + 64, 4, 0xF, qh, 4, 3, 4);
    const Partial p3 = Dot16(ql_b, xb + 96, 4, 0xF, qh, 6, 3, 4);
    // q - 32 folds the offset into the activation sum.
    float a = acc.v[b];
    a = __builtin_fmaf(s0, __builtin_fmaf(-32.0F, p0.sum, p0.dot), a);
    a = __builtin_fmaf(s1, __builtin_fmaf(-32.0F, p1.sum, p1.dot), a);
    a = __builtin_fmaf(s2, __builtin_fmaf(-32.0F, p2.sum, p2.dot), a);
    a = __builtin_fmaf(s3, __builtin_fmaf(-32.0F, p3.sum, p3.dot), a);
    acc.v[b] = a;
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
        TaskQ4K<B>(block, t, xs, k0, acc);
      } else if constexpr (F == GemvFormat::kQ5_K) {
        TaskQ5K<B>(block, t, xs, k0, acc);
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

template<GemvFormat F>
void LaunchRows(const void* w, const float* x, float* y, std::uint32_t rows,
                std::uint32_t m, std::uint32_t k, hipStream_t stream) {
  switch (rows) {
    case 1:
      Launch<F, 1>(w, x, y, m, k, stream);
      return;
    case 2:
      Launch<F, 2>(w, x, y, m, k, stream);
      return;
    case 3:
      Launch<F, 3>(w, x, y, m, k, stream);
      return;
    case 4:
      Launch<F, 4>(w, x, y, m, k, stream);
      return;
    case 5:
      Launch<F, 5>(w, x, y, m, k, stream);
      return;
    case 6:
      Launch<F, 6>(w, x, y, m, k, stream);
      return;
    case 7:
      Launch<F, 7>(w, x, y, m, k, stream);
      return;
    default:
      Launch<F, 8>(w, x, y, m, k, stream);
      return;
  }
}

}  // namespace

bool LaunchKQuantGemv(GemvFormat format, const void* w, const float* x,
                      float* y, std::uint32_t rows, std::uint32_t m,
                      std::uint32_t k, hipStream_t stream) {
  if (rows == 0 || rows > kMaxGemvRows || k % 256 != 0) {
    return false;
  }
  switch (format) {
    case GemvFormat::kQ4_K:
      LaunchRows<GemvFormat::kQ4_K>(w, x, y, rows, m, k, stream);
      return true;
    case GemvFormat::kQ5_K:
      LaunchRows<GemvFormat::kQ5_K>(w, x, y, rows, m, k, stream);
      return true;
    case GemvFormat::kQ6_K:
      LaunchRows<GemvFormat::kQ6_K>(w, x, y, rows, m, k, stream);
      return true;
  }
  return false;
}

}  // namespace gufo::models::gemma4::rocm
