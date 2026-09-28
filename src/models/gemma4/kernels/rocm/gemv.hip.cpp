// Autoregressive decode projections for Gemma 4's K-quant and Q4_0 weights.
//
// y[m] = sum_k x[k] * W[m][k]; the weights stay in their GGUF blocks and the
// activations stay FP32. A lane's unit of work is one 16-byte vector of
// quantized values -- 32 values of a Q4_K/Q5_K super-block (with their Q5_K
// high bits) or 64 values of a Q6_K super-block -- dequantized once and
// contracted in a fixed FMA order. Lanes holding the same task of different
// rows read the same activations, which the hardware broadcasts; each row
// reduces over its task lanes with a fixed xor tree. The eight waves of a
// workgroup split one row group's reduction into slices, summed in wave
// order, which keeps enough loads in flight for short outputs.
#include "src/models/gemma4/kernels/rocm/gemv.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstdint>

namespace gufo::models::gemma4::rocm {
namespace {

constexpr int kWave = 32;
constexpr int kWavesPerBlock = 8;

constexpr int kSlices = kWavesPerBlock;

__device__ __forceinline__ float Half(const std::uint8_t* p) {
  return __half2float(*reinterpret_cast<const __half*>(p));
}

__device__ __forceinline__ float HalfBits(std::uint32_t bits) {
  const auto h = static_cast<unsigned short>(bits & 0xFFFFU);
  return __half2float(*reinterpret_cast<const __half*>(&h));
}

/// ggml get_scale_min_k4 over the 12 scale bytes held as three words
/// (s[0..3], s[4..7], s[8..11]): 6-bit scale and min of 32-value sub-block
/// j, extracted with shifts instead of byte indexing.
__device__ __forceinline__ void ScaleMin(std::uint32_t w0, std::uint32_t w1,
                                         std::uint32_t w2, int j, float* sc,
                                         float* m) {
  const std::uint32_t shift = 8U * static_cast<std::uint32_t>(j & 3);
  const std::uint32_t a = (w0 >> shift) & 0xFFU;
  const std::uint32_t b = (w1 >> shift) & 0xFFU;
  const std::uint32_t c = (w2 >> shift) & 0xFFU;
  const bool low = j < 4;
  *sc = static_cast<float>(low ? (a & 63U) : ((c & 0xFU) | ((a >> 6) << 4)));
  *m = static_cast<float>(low ? (b & 63U) : ((c >> 4) | ((b >> 6) << 4)));
}

/// Dequantizes 16 values packed per byte (optional high bits) as
/// scale * v + offset.
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
// from bit 2 p (low) or 2 p + 1 (high) of qh[16 h + i]. The 16-byte header
// (d, dmin, 12 scale bytes) is one aligned load.
template<bool kFiveBit>
__device__ __forceinline__ float TaskQ45K(const std::uint8_t* block, int t,
                                          const float* x, float acc) {
  const int p = t >> 1;
  const int h = t & 1;
  const uint4 header = *reinterpret_cast<const uint4*>(block);
  const float d = HalfBits(header.x);
  const float dmin = HalfBits(header.x >> 16);
  float sc0, m0, sc1, m1;
  ScaleMin(header.y, header.z, header.w, 2 * p, &sc0, &m0);
  ScaleMin(header.y, header.z, header.w, 2 * p + 1, &sc1, &m1);
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
  const float* xv = x + 64 * p + 16 * h;
  acc = Accumulate<16>(acc, w, xv);
  return Accumulate<16>(acc, w + 16, xv + 32);
}

// Q6_K: 4 tasks per super-block. Task t = 2 n + h covers ql bytes
// [64 n + 16 h, +16) and [64 n + 32 + 16 h, +16) and qh bytes
// [32 n + 16 h, +16): values 128 n + 16 h + i + {0, 32, 64, 96} with signed
// scales sc[8 n + h + {0, 2, 4, 6}] and an offset of 32.
__device__ __forceinline__ float TaskQ6K(const std::uint8_t* block, int t,
                                         const float* x, float acc) {
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
  const float* xv = x + 128 * n + 16 * h;
  acc = Accumulate<16>(acc, w, xv);
  acc = Accumulate<16>(acc, w + 16, xv + 32);
  acc = Accumulate<16>(acc, w + 32, xv + 64);
  return Accumulate<16>(acc, w + 48, xv + 96);
}

// Q4_0: eight 32-value blocks form a 256-value, 144-byte group, and task t
// is block t: values 32 t + i (low nibbles) and 32 t + 16 + i (high) with
// (q - 8) d = d q - 8 d. Blocks are 18 bytes, so only every other block is
// word aligned; each lane loads the five words enclosing its block and
// selects the payload with byte alignment.
__device__ __forceinline__ float TaskQ40(const std::uint8_t* group, int t,
                                         const float* x, float acc) {
  const std::uint8_t* block = group + 18 * t;
  const bool odd = (t & 1) != 0;
  const auto* words =
      reinterpret_cast<const std::uint32_t*>(block - (odd ? 2 : 0));
  std::uint32_t r[5];
#pragma unroll
  for (int i = 0; i < 5; ++i) {
    r[i] = words[i];
  }
  const float d = HalfBits(odd ? r[0] >> 16 : r[0]);
  std::uint32_t qs[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    qs[i] = odd ? r[i + 1] : __builtin_amdgcn_alignbyte(r[i + 1], r[i], 2);
  }
  const auto* q = reinterpret_cast<const std::uint8_t*>(qs);
  float w[32];
  Dequant16(q, 0, nullptr, 0, 0, d, -8.0F * d, w);
  Dequant16(q, 4, nullptr, 0, 0, d, -8.0F * d, w + 16);
  const float* xv = x + 32 * t;
  acc = Accumulate<16>(acc, w, xv);
  return Accumulate<16>(acc, w + 16, xv + 16);
}

template<GemvFormat F>
struct FormatTraits;
template<>
struct FormatTraits<GemvFormat::kQ4_0> {
  static constexpr int kBlockBytes = 144;
  static constexpr int kTasks = 8;
};
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

template<GemvFormat F>
__device__ __forceinline__ float Task(const std::uint8_t* block, int t,
                                      const float* x, float acc) {
  if constexpr (F == GemvFormat::kQ4_0) {
    return TaskQ40(block, t, x, acc);
  } else if constexpr (F == GemvFormat::kQ4_K) {
    return TaskQ45K<false>(block, t, x, acc);
  } else if constexpr (F == GemvFormat::kQ5_K) {
    return TaskQ45K<true>(block, t, x, acc);
  } else {
    return TaskQ6K(block, t, x, acc);
  }
}

/// Lanes sharing a task position read the same activations, so the wave
/// covers 32 / kTasks rows at once: lane = row_in_wave * kTasks + task. The
/// workgroup's waves take the super-blocks of those rows in kSlices slices:
/// interleaved (wave-adjacent blocks) for short rows, contiguous runs for
/// long ones.
template<GemvFormat F, bool kInterleave>
__global__ void __launch_bounds__(kWave* kWavesPerBlock)
    KQuantGemvKernel(const std::uint8_t* __restrict__ w,
                     const float* __restrict__ x, float* __restrict__ y,
                     std::uint32_t m, std::uint32_t k) {
  using T = FormatTraits<F>;
  constexpr int kRowsPerWave = kWave / T::kTasks;
  __shared__ float partial[kSlices][kWave];
  const int lane = threadIdx.x % kWave;
  const int slice = threadIdx.x / kWave;
  const int t = lane % T::kTasks;
  const std::uint32_t row = blockIdx.x * kRowsPerWave + lane / T::kTasks;
  const std::uint32_t blocks_per_row = k / 256;
  float acc = 0.0F;
  if (row < m) {
    const std::uint8_t* base =
        w + static_cast<std::size_t>(row) * blocks_per_row * T::kBlockBytes;
    if constexpr (kInterleave) {
#pragma unroll 2
      for (std::uint32_t blk = slice; blk < blocks_per_row; blk += kSlices) {
        acc = Task<F>(base + blk * T::kBlockBytes, t, x + blk * 256, acc);
      }
    } else {
      const std::uint32_t per = (blocks_per_row + kSlices - 1) / kSlices;
      const std::uint32_t end = min(blocks_per_row, (slice + 1) * per);
#pragma unroll 2
      for (std::uint32_t blk = slice * per; blk < end; ++blk) {
        acc = Task<F>(base + blk * T::kBlockBytes, t, x + blk * 256, acc);
      }
    }
  }
#pragma unroll
  for (int offset = T::kTasks / 2; offset > 0; offset >>= 1) {
    acc += __shfl_xor(acc, offset, kWave);
  }
  partial[slice][lane] = acc;
  __syncthreads();
  if (slice == 0 && t == 0 && row < m) {
    float sum = partial[0][lane];
#pragma unroll
    for (int s = 1; s < kSlices; ++s) {
      sum += partial[s][lane];
    }
    y[row] = sum;
  }
}

/// Short rows (a few super-blocks, like the drafter's 1024-wide vocabulary
/// head) leave most slices idle; each wave takes whole rows instead.
template<GemvFormat F>
__global__ void __launch_bounds__(kWave* kWavesPerBlock)
    KQuantGemvRowsKernel(const std::uint8_t* __restrict__ w,
                         const float* __restrict__ x, float* __restrict__ y,
                         std::uint32_t m, std::uint32_t k) {
  using T = FormatTraits<F>;
  constexpr int kRowsPerWave = kWave / T::kTasks;
  const int lane = threadIdx.x % kWave;
  const int t = lane % T::kTasks;
  const std::uint32_t row =
      (blockIdx.x * kWavesPerBlock + threadIdx.x / kWave) * kRowsPerWave +
      lane / T::kTasks;
  const std::uint32_t blocks_per_row = k / 256;
  float acc = 0.0F;
  if (row < m) {
    const std::uint8_t* base =
        w + static_cast<std::size_t>(row) * blocks_per_row * T::kBlockBytes;
#pragma unroll 4
    for (std::uint32_t blk = 0; blk < blocks_per_row; ++blk) {
      acc = Task<F>(base + blk * T::kBlockBytes, t, x + blk * 256, acc);
    }
  }
#pragma unroll
  for (int offset = T::kTasks / 2; offset > 0; offset >>= 1) {
    acc += __shfl_xor(acc, offset, kWave);
  }
  if (t == 0 && row < m) {
    y[row] = acc;
  }
}

template<GemvFormat F>
void Launch(const void* w, const float* x, float* y, std::uint32_t m,
            std::uint32_t k, hipStream_t stream) {
  constexpr int kRowsPerBlock = kWave / FormatTraits<F>::kTasks;
  const auto* weights = static_cast<const std::uint8_t*>(w);
  if (k <= 1024) {
    constexpr int kRows = kRowsPerBlock * kWavesPerBlock;
    KQuantGemvRowsKernel<F>
        <<<(m + kRows - 1) / kRows, kWave * kWavesPerBlock, 0, stream>>>(
            weights, x, y, m, k);
    return;
  }
  const unsigned blocks = (m + kRowsPerBlock - 1) / kRowsPerBlock;
  // Measured on gfx1151: interleaving wins up to K = 8192, contiguous
  // slices on the 21504-long FFN down reduction.
  if (k <= 8192) {
    KQuantGemvKernel<F, true>
        <<<blocks, kWave * kWavesPerBlock, 0, stream>>>(weights, x, y, m, k);
  } else {
    KQuantGemvKernel<F, false>
        <<<blocks, kWave * kWavesPerBlock, 0, stream>>>(weights, x, y, m, k);
  }
}

constexpr int kRepackThreads = 256;

/// ggml's make_qkx2_quants(32, 15, x, weights, L, &min, Laux, -1, 0.1, 20):
/// the weighted least-squares (scale, min) over 21 candidate grids.
__device__ float FitScaleMin(const float* x, const float* weights,
                             std::uint8_t* levels, float* the_min) {
  constexpr int kN = 32;
  constexpr float kMax = 15.0F;
  float lo = x[0];
  float hi = x[0];
  float sum_w = weights[0];
  float sum_x = sum_w * x[0];
  for (int i = 1; i < kN; ++i) {
    lo = fminf(lo, x[i]);
    hi = fmaxf(hi, x[i]);
    sum_w += weights[i];
    sum_x += weights[i] * x[i];
  }
  lo = fminf(lo, 0.0F);
  if (hi == lo) {
    for (int i = 0; i < kN; ++i) {
      levels[i] = 0;
    }
    *the_min = -lo;
    return 0.0F;
  }
  float iscale = kMax / (hi - lo);
  float scale = 1.0F / iscale;
  float best = 0.0F;
  for (int i = 0; i < kN; ++i) {
    const int l = min(15, max(0, __float2int_rn(iscale * (x[i] - lo))));
    levels[i] = static_cast<std::uint8_t>(l);
    const float diff = scale * static_cast<float>(l) + lo - x[i];
    best += weights[i] * diff * diff;
  }
  std::uint8_t trial[kN];
  for (int step = 0; step <= 20; ++step) {
    iscale = (-1.0F + 0.1F * static_cast<float>(step) + kMax) / (hi - lo);
    float sum_l = 0.0F;
    float sum_l2 = 0.0F;
    float sum_xl = 0.0F;
    for (int i = 0; i < kN; ++i) {
      const int l = min(15, max(0, __float2int_rn(iscale * (x[i] - lo))));
      trial[i] = static_cast<std::uint8_t>(l);
      const float w = weights[i];
      sum_l += w * static_cast<float>(l);
      sum_l2 += w * static_cast<float>(l * l);
      sum_xl += w * static_cast<float>(l) * x[i];
    }
    const float det = sum_w * sum_l2 - sum_l * sum_l;
    if (det > 0.0F) {
      float this_scale = (sum_w * sum_xl - sum_x * sum_l) / det;
      float this_min = (sum_l2 * sum_x - sum_l * sum_xl) / det;
      if (this_min > 0.0F) {
        this_min = 0.0F;
        this_scale = sum_xl / sum_l2;
      }
      float error = 0.0F;
      for (int i = 0; i < kN; ++i) {
        const float diff =
            this_scale * static_cast<float>(trial[i]) + this_min - x[i];
        error += weights[i] * diff * diff;
      }
      if (error < best) {
        for (int i = 0; i < kN; ++i) {
          levels[i] = trial[i];
        }
        best = error;
        scale = this_scale;
        lo = this_min;
      }
    }
  }
  *the_min = -lo;
  return scale;
}

/// One thread per 32-value sub-block (one Q8_0 block); the eight threads of
/// a super-block share their fits through shared memory, as
/// quantize_row_q4_K_ref does within a block.
__global__ void __launch_bounds__(kRepackThreads)
    RepackQ8_0AsQ4KKernel(const std::uint8_t* __restrict__ src,
                          std::uint8_t* __restrict__ dst, std::uint32_t subs) {
  __shared__ float scales[kRepackThreads];
  __shared__ float mins[kRepackThreads];
  __shared__ std::uint8_t levels[kRepackThreads][32];
  const std::uint32_t sub = blockIdx.x * kRepackThreads + threadIdx.x;
  const bool live = sub < subs;
  float x[32];
  float weights[32];
  float sum_x2 = 0.0F;
  const std::uint8_t* block = src + static_cast<std::size_t>(sub) * 34;
  const float d8 = live ? Half(block) : 0.0F;
  for (int i = 0; i < 32; ++i) {
    x[i] = live
               ? d8 * static_cast<float>(static_cast<std::int8_t>(block[2 + i]))
               : 0.0F;
    sum_x2 += x[i] * x[i];
  }
  const float average = sqrtf(sum_x2 / 32.0F);
  for (int i = 0; i < 32; ++i) {
    weights[i] = average + fabsf(x[i]);
  }
  float the_min = 0.0F;
  scales[threadIdx.x] = FitScaleMin(x, weights, levels[threadIdx.x], &the_min);
  mins[threadIdx.x] = the_min;
  __syncthreads();
  const int first = threadIdx.x & ~7;
  const int j = threadIdx.x & 7;
  float max_scale = 0.0F;
  float max_min = 0.0F;
  for (int s = 0; s < 8; ++s) {
    max_scale = fmaxf(max_scale, scales[first + s]);
    max_min = fmaxf(max_min, mins[first + s]);
  }
  const float inv_scale = max_scale > 0.0F ? 63.0F / max_scale : 0.0F;
  const float inv_min = max_min > 0.0F ? 63.0F / max_min : 0.0F;
  const auto level = [&](float v, float inv) {
    return static_cast<std::uint8_t>(min(63, __float2int_rn(inv * v)));
  };
  const __half d = __float2half(max_scale / 63.0F);
  const __half dmin = __float2half(max_min / 63.0F);
  std::uint8_t* out = dst + static_cast<std::size_t>(sub / 8) * 144;
  if (live && j == 0) {
    std::uint8_t packed[12] = {};
    for (int s = 0; s < 8; ++s) {
      const std::uint8_t ls = level(scales[first + s], inv_scale);
      const std::uint8_t lm = level(mins[first + s], inv_min);
      if (s < 4) {
        packed[s] = ls;
        packed[s + 4] = lm;
      } else {
        packed[s + 4] =
            static_cast<std::uint8_t>((ls & 0xF) | ((lm & 0xF) << 4));
        packed[s - 4] |= static_cast<std::uint8_t>((ls >> 4) << 6);
        packed[s] |= static_cast<std::uint8_t>((lm >> 4) << 6);
      }
    }
    *reinterpret_cast<__half*>(out) = d;
    *reinterpret_cast<__half*>(out + 2) = dmin;
    for (int i = 0; i < 12; ++i) {
      out[4 + i] = packed[i];
    }
  }
  // Requantize against the rounded 6-bit scale and min.
  const float step = __half2float(d) *
                     static_cast<float>(level(scales[threadIdx.x], inv_scale));
  const float offset = __half2float(dmin) *
                       static_cast<float>(level(mins[threadIdx.x], inv_min));
  if (step != 0.0F) {
    for (int i = 0; i < 32; ++i) {
      levels[threadIdx.x][i] = static_cast<std::uint8_t>(
          min(15, max(0, __float2int_rn((x[i] + offset) / step))));
    }
  }
  __syncthreads();
  // Sub-blocks 2c and 2c+1 share 32 quant bytes (low and high nibbles).
  if (live && (j & 1) == 0) {
    std::uint8_t* qs = out + 16 + (j / 2) * 32;
    for (int i = 0; i < 32; ++i) {
      qs[i] = static_cast<std::uint8_t>(levels[threadIdx.x][i] |
                                        (levels[threadIdx.x + 1][i] << 4));
    }
  }
}

}  // namespace

bool LaunchKQuantGemv(GemvFormat format, const void* w, const float* x,
                      float* y, std::uint32_t m, std::uint32_t k,
                      hipStream_t stream) {
  if (k % 256 != 0) {
    return false;
  }
  switch (format) {
    case GemvFormat::kQ4_0:
      Launch<GemvFormat::kQ4_0>(w, x, y, m, k, stream);
      return true;
    case GemvFormat::kQ4_K:
      Launch<GemvFormat::kQ4_K>(w, x, y, m, k, stream);
      return true;
    case GemvFormat::kQ5_K:
      Launch<GemvFormat::kQ5_K>(w, x, y, m, k, stream);
      return true;
    case GemvFormat::kQ6_K:
      Launch<GemvFormat::kQ6_K>(w, x, y, m, k, stream);
      return true;
  }
  return false;
}

void RepackQ8_0AsQ4K(const void* src, void* dst, std::uint32_t rows,
                     std::uint32_t cols, hipStream_t stream) {
  const std::uint32_t subs = rows * (cols / 32);
  RepackQ8_0AsQ4KKernel<<<(subs + kRepackThreads - 1) / kRepackThreads,
                          kRepackThreads, 0, stream>>>(
      static_cast<const std::uint8_t*>(src), static_cast<std::uint8_t*>(dst),
      subs);
}

}  // namespace gufo::models::gemma4::rocm
