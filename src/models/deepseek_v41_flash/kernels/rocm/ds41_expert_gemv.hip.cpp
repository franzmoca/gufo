/* Single-token routed-expert projections for the V4.1 Flash IQ1_S, IQ1_M,
 * IQ2_XXS and IQ2_S weights with FP32 activations. Each slot multiplies its
 * own expert, ids[slot]; a wave covers 4 rows and a lane takes one 32-value
 * sub-block (task t) of each 256-value block, summed over the 8 tasks at the
 * end.
 *
 * IQ1 weights are (n - 1 + delta) * d * s with a grid nibble n in {0, 1, 2},
 * so a group of eight is d s (sum n_i x_i + (delta - 1) sum x_i): one convert
 * and one FMA per weight instead of decoding each weight to a float first.
 * The 8 KiB IQ1 codebook is copied to LDS, where the per-lane lookups do not
 * queue on the texture path behind the weight and activation loads. */
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>

#include "ds41_gemv_tasks.hpp"
#include "ds41_v41.h"
#include "mmq/ggml-common.h"

namespace {

using ds41_gemv_tasks::Accumulate;
using ds41_gemv_tasks::HalfBits;

constexpr int kWave = 32;
constexpr int kWaves = 8;

/* A task's 16-bit words of one block (IQ blocks are only 2-byte aligned
 * inside a row), loaded ahead of the decode. */
struct Raw {
  uint32_t r[7];
};

/* sum_i (n_i + c) x_i over one 8-value IQ1 grid group: the nibbles move to
 * bytes (elements 0-3 low, 4-7 high) so each converts in one instruction. */
__device__ __forceinline__ float Iq1Dot(uint32_t grid, float c, const float* x) {
  const uint32_t lo = grid & 0x0F0F0F0F;
  const uint32_t hi = (grid >> 4) & 0x0F0F0F0F;
  const float4 a = *reinterpret_cast<const float4*>(x);
  const float4 b = *reinterpret_cast<const float4*>(x + 4);
  float dot = static_cast<float>(lo & 0xFF) * a.x;
  dot = __builtin_fmaf(static_cast<float>((lo >> 8) & 0xFF), a.y, dot);
  dot = __builtin_fmaf(static_cast<float>((lo >> 16) & 0xFF), a.z, dot);
  dot = __builtin_fmaf(static_cast<float>(lo >> 24), a.w, dot);
  dot = __builtin_fmaf(static_cast<float>(hi & 0xFF), b.x, dot);
  dot = __builtin_fmaf(static_cast<float>((hi >> 8) & 0xFF), b.y, dot);
  dot = __builtin_fmaf(static_cast<float>((hi >> 16) & 0xFF), b.z, dot);
  dot = __builtin_fmaf(static_cast<float>(hi >> 24), b.w, dot);
  const float sum = ((a.x + a.y) + (a.z + a.w)) + ((b.x + b.y) + (b.z + b.w));
  return __builtin_fmaf(c, sum, dot);
}

/* Eight grid magnitudes (one per byte) with the sign bits of `signs`. */
__device__ __forceinline__ void Iq2Group(uint64_t grid, uint32_t signs, float scale, float* w) {
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const float v = static_cast<float>((grid >> (8 * i)) & 0xFF) * scale;
    w[i] = (signs >> i) & 1 ? -v : v;
  }
}

template <int kType>
struct Format;

/* IQ1_S (50 bytes): d, qs[32], qh[8] as u16; sub-block t uses qs[4 t..],
 * qh[t] (3 high index bits per group, 3-bit scale, delta sign). */
template <>
struct Format<19> {
  static constexpr int kBlockBytes = 50;
  static constexpr bool kIq1Grid = true;
  static __device__ __forceinline__ Raw Load(const uint8_t* b, int t) {
    const auto* h = reinterpret_cast<const uint16_t*>(b);
    return {{h[0], h[1 + 2 * t], h[2 + 2 * t], h[17 + t]}};
  }
  static __device__ __forceinline__ float Dot(const Raw& a, int, const float* x, const uint32_t* grid) {
    const float d = HalfBits(a.r[0]);
    const uint32_t qs = a.r[1] | (a.r[2] << 16);
    const uint32_t qh = a.r[3];
    const float scale = d * static_cast<float>(2 * ((qh >> 12) & 7) + 1);
    const float c = (qh & 0x8000 ? -IQ1S_DELTA : IQ1S_DELTA) - 1.0f;
    float dot = 0.0f;
#pragma unroll
    for (int l = 0; l < 4; ++l)
      dot += Iq1Dot(grid[((qs >> (8 * l)) & 0xFF) | (((qh >> (3 * l)) & 7) << 8)], c, x + 8 * l);
    return scale * dot;
  }
};

/* IQ1_M (56 bytes): qs[32], qh[16], four u16 scale words whose top nibbles
 * hold d. Groups 0-1 of sub-block t use its low 3-bit scale, 2-3 the high. */
template <>
struct Format<29> {
  static constexpr int kBlockBytes = 56;
  static constexpr bool kIq1Grid = true;
  static __device__ __forceinline__ Raw Load(const uint8_t* b, int t) {
    const auto* h = reinterpret_cast<const uint16_t*>(b);
    return {{h[2 * t], h[2 * t + 1], h[16 + t], h[24], h[25], h[26], h[27]}};
  }
  static __device__ __forceinline__ float Dot(const Raw& a, int t, const float* x, const uint32_t* grid) {
    const uint32_t qs = a.r[0] | (a.r[1] << 16);
    const uint32_t sc0 = a.r[3], sc1 = a.r[4], sc2 = a.r[5], sc3 = a.r[6];
    const float d = HalfBits((sc0 >> 12) | ((sc1 >> 8) & 0x00F0) | ((sc2 >> 4) & 0x0F00) | (sc3 & 0xF000));
    const uint32_t word = a.r[3 + (t >> 1)] >> (6 * (t & 1));
    const float s1 = d * static_cast<float>(2 * (word & 7) + 1);
    const float s2 = d * static_cast<float>(2 * ((word >> 3) & 7) + 1);
    const uint32_t qh0 = a.r[2] & 0xFF, qh1 = a.r[2] >> 8;
    const uint32_t high[4] = {(qh0 << 8) & 0x700, (qh0 << 4) & 0x700, (qh1 << 8) & 0x700, (qh1 << 4) & 0x700};
    const uint32_t flag[4] = {qh0 & 0x08, qh0 & 0x80, qh1 & 0x08, qh1 & 0x80};
    float dot[4];
#pragma unroll
    for (int l = 0; l < 4; ++l)
      dot[l] = Iq1Dot(grid[((qs >> (8 * l)) & 0xFF) | high[l]], (flag[l] ? -IQ1M_DELTA : IQ1M_DELTA) - 1.0f,
                      x + 8 * l);
    return __builtin_fmaf(s1, dot[0] + dot[1], s2 * (dot[2] + dot[3]));
  }
};

/* IQ2_XXS (66 bytes): d, then per sub-block four grid bytes and a u32 with
 * four 7-bit sign indices and a 4-bit scale. */
template <>
struct Format<16> {
  static constexpr int kBlockBytes = 66;
  static constexpr bool kIq1Grid = false;
  static __device__ __forceinline__ Raw Load(const uint8_t* b, int t) {
    const auto* h = reinterpret_cast<const uint16_t*>(b);
    const uint16_t* g = h + 1 + 4 * t;
    return {{h[0], g[0], g[1], g[2], g[3]}};
  }
  static __device__ __forceinline__ float Dot(const Raw& a, int, const float* x, const uint32_t*) {
    const float d = HalfBits(a.r[0]);
    const uint32_t qs = a.r[1] | (a.r[2] << 16);
    const uint32_t aux = a.r[3] | (a.r[4] << 16);
    const float scale = d * (0.5f + static_cast<float>(aux >> 28)) * 0.25f;
    float w[32];
#pragma unroll
    for (int l = 0; l < 4; ++l)
      Iq2Group(iq2xxs_grid[(qs >> (8 * l)) & 0xFF], ksigns_iq2xs[(aux >> (7 * l)) & 127], scale, w + 8 * l);
    return Accumulate<32>(0.0f, w, x);
  }
};

/* IQ2_S (82 bytes): d, qs[32], signs[32], qh[8], scales[8]. */
template <>
struct Format<22> {
  static constexpr int kBlockBytes = 82;
  static constexpr bool kIq1Grid = false;
  static __device__ __forceinline__ Raw Load(const uint8_t* b, int t) {
    const auto* h = reinterpret_cast<const uint16_t*>(b);
    return {{h[0], h[1 + 2 * t], h[2 + 2 * t], h[17 + 2 * t], h[18 + 2 * t], h[33 + (t >> 1)], h[37 + (t >> 1)]}};
  }
  static __device__ __forceinline__ float Dot(const Raw& a, int t, const float* x, const uint32_t*) {
    const float d = HalfBits(a.r[0]);
    const uint32_t qs = a.r[1] | (a.r[2] << 16);
    const uint32_t signs = a.r[3] | (a.r[4] << 16);
    const uint32_t qh = (a.r[5] >> (8 * (t & 1))) & 0xFF;
    const uint32_t sc = (a.r[6] >> (8 * (t & 1))) & 0xFF;
    const float low = d * (0.5f + static_cast<float>(sc & 0xF)) * 0.25f;
    const float high = d * (0.5f + static_cast<float>(sc >> 4)) * 0.25f;
    float w[32];
#pragma unroll
    for (int l = 0; l < 4; ++l)
      Iq2Group(iq2s_grid[((qs >> (8 * l)) & 0xFF) | ((qh << (8 - 2 * l)) & 0x300)], (signs >> (8 * l)) & 0xFF,
               l < 2 ? low : high, w + 8 * l);
    return Accumulate<32>(0.0f, w, x);
  }
};

/* Grid (row tiles of 32, slots). The raw words of U blocks are loaded before
 * any of them is decoded. */
template <int kType, int U>
__global__ void __launch_bounds__(kWave* kWaves)
    expert_gemv_kernel(const uint8_t* __restrict__ w, const float* __restrict__ x, float* __restrict__ y, uint32_t m,
                       uint32_t k, const int32_t* __restrict__ ids, uint64_t expert_bytes, uint32_t x_slot) {
  using F = Format<kType>;
  constexpr uint32_t kRows = 4 * kWaves;
  __shared__ uint32_t grid[F::kIq1Grid ? NGRID_IQ1S : 1];
  const int32_t id = ids[blockIdx.y];
  y += (uint64_t)blockIdx.y * m;
  if (id < 0) {  // dropped slot
    for (uint32_t r = threadIdx.x; r < kRows; r += blockDim.x)
      if (blockIdx.x * kRows + r < m) y[blockIdx.x * kRows + r] = 0.0f;
    return;
  }
  if constexpr (F::kIq1Grid) {
    for (uint32_t i = threadIdx.x; i < NGRID_IQ1S; i += blockDim.x) grid[i] = iq1s_grid_gpu[i];
    __syncthreads();
  }
  const int lane = threadIdx.x % kWave;
  const int wave = threadIdx.x / kWave;
  const int t = lane % 8;
  const uint32_t row = blockIdx.x * kRows + wave * 4 + lane / 8;
  if (row >= m) return;
  const float* xt = x + (uint64_t)blockIdx.y * x_slot + 32 * t;
  const uint32_t blocks_per_row = k / 256;
  const uint8_t* base = w + (uint64_t)id * expert_bytes + (uint64_t)row * blocks_per_row * F::kBlockBytes;
  float acc = 0.0f;
  uint32_t blk = 0;
  for (; blk + U <= blocks_per_row; blk += U) {
    Raw raw[U];
#pragma unroll
    for (int u = 0; u < U; ++u) raw[u] = F::Load(base + (uint64_t)(blk + u) * F::kBlockBytes, t);
#pragma unroll
    for (int u = 0; u < U; ++u) acc += F::Dot(raw[u], t, xt + (blk + u) * 256, grid);
  }
  for (; blk < blocks_per_row; ++blk)
    acc += F::Dot(F::Load(base + (uint64_t)blk * F::kBlockBytes, t), t, xt + blk * 256, grid);
#pragma unroll
  for (int offset = 4; offset > 0; offset >>= 1) acc += __shfl_xor(acc, offset, kWave);
  if (t == 0) y[row] = acc;
}

template <int kType, int U>
int launch(const void* w, uint64_t expert_bytes, const int32_t* ids, uint32_t slots, const float* x,
           uint32_t x_slot, float* y, uint32_t m, uint32_t k) {
  constexpr uint32_t kRows = 4 * kWaves;
  expert_gemv_kernel<kType, U><<<dim3((m + kRows - 1) / kRows, slots), kWave * kWaves>>>(
      static_cast<const uint8_t*>(w), x, y, m, k, ids, expert_bytes, x_slot);
  const hipError_t err = hipGetLastError();
  if (err == hipSuccess) return 0;
  fprintf(stderr, "ds41: expert gemv launch failed: %s\n", hipGetErrorString(err));
  return 1;
}

}  // namespace

extern "C" int ds41_v41_gemv_experts(int type, const void* w, uint64_t expert_bytes, const int32_t* ids,
                                     uint32_t slots, const float* x, uint32_t x_slot, float* y, uint32_t m,
                                     uint32_t k) {
  if (!ids || slots == 0 || m == 0 || k % 256 != 0) return 1;
  /* Blocks loaded ahead: 2 measured best for IQ1_S, 1 for the others. */
  switch (type) {
  case 19:
    return launch<19, 2>(w, expert_bytes, ids, slots, x, x_slot, y, m, k);
  case 29:
    return launch<29, 1>(w, expert_bytes, ids, slots, x, x_slot, y, m, k);
  case 16:
    return launch<16, 1>(w, expert_bytes, ids, slots, x, x_slot, y, m, k);
  case 22:
    return launch<22, 1>(w, expert_bytes, ids, slots, x, x_slot, y, m, k);
  default:
    return 1;
  }
}
