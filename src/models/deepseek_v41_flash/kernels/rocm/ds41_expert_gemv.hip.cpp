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
  /* Fixed order under -ffast-math: every kernel inlining this computes the
   * same sums, so grouped and per-slot launches agree bit for bit. */
#pragma clang fp reassociate(off)
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
  /* acc + scale * dot as one explicit FMA, so no kernel's contraction choice
   * changes the result. */
  static __device__ __forceinline__ float Acc(float acc, const Raw& a, int, const float* x, const uint32_t* grid) {
#pragma clang fp reassociate(off)
    const float d = HalfBits(a.r[0]);
    const uint32_t qs = a.r[1] | (a.r[2] << 16);
    const uint32_t qh = a.r[3];
    const float scale = d * static_cast<float>(2 * ((qh >> 12) & 7) + 1);
    const float c = (qh & 0x8000 ? -IQ1S_DELTA : IQ1S_DELTA) - 1.0f;
    float dot = 0.0f;
#pragma unroll
    for (int l = 0; l < 4; ++l)
      dot += Iq1Dot(grid[((qs >> (8 * l)) & 0xFF) | (((qh >> (3 * l)) & 7) << 8)], c, x + 8 * l);
    return __builtin_fmaf(scale, dot, acc);
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
  static __device__ __forceinline__ float Acc(float acc, const Raw& a, int t, const float* x, const uint32_t* grid) {
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
    return acc + __builtin_fmaf(s1, dot[0] + dot[1], s2 * (dot[2] + dot[3]));
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
  static __device__ __forceinline__ float Acc(float acc, const Raw& a, int, const float* x, const uint32_t*) {
    const float d = HalfBits(a.r[0]);
    const uint32_t qs = a.r[1] | (a.r[2] << 16);
    const uint32_t aux = a.r[3] | (a.r[4] << 16);
    const float scale = d * (0.5f + static_cast<float>(aux >> 28)) * 0.25f;
    float w[32];
#pragma unroll
    for (int l = 0; l < 4; ++l)
      Iq2Group(iq2xxs_grid[(qs >> (8 * l)) & 0xFF], ksigns_iq2xs[(aux >> (7 * l)) & 127], scale, w + 8 * l);
    return acc + Accumulate<32>(0.0f, w, x);
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
  static __device__ __forceinline__ float Acc(float acc, const Raw& a, int t, const float* x, const uint32_t*) {
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
    return acc + Accumulate<32>(0.0f, w, x);
  }
};

/* Grid (row tiles of 32, slots). The raw words of U blocks are loaded before
 * any of them is decoded. */
/* Row sums of one expert matrix; with kSwiGLU, a wave takes the same rows
 * of the gate (w) and up (w_up) matrices and stores
 * route_w[slot] * silu(min(gate, limit)) * clamp(up, +-limit). */
template <int kType, int U, bool kSwiGLU>
__global__ void __launch_bounds__(kWave* kWaves)
    expert_gemv_kernel(const uint8_t* __restrict__ w, const uint8_t* __restrict__ w_up,
                       const float* __restrict__ route_w, float limit, const float* __restrict__ x,
                       float* __restrict__ y, uint32_t m, uint32_t k, const int32_t* __restrict__ ids,
                       uint64_t expert_bytes, uint32_t x_slot, uint32_t slots_per_x) {
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
  const float* xt = x + (uint64_t)(blockIdx.y / slots_per_x) * x_slot + 32 * t;
  const uint32_t blocks_per_row = k / 256;
  const uint64_t offset = (uint64_t)id * expert_bytes + (uint64_t)row * blocks_per_row * F::kBlockBytes;
  const uint8_t* base = w + offset;
  const uint8_t* base_up = kSwiGLU ? w_up + offset : nullptr;
  float acc = 0.0f;
  float acc_up = 0.0f;
  uint32_t blk = 0;
  for (; blk + U <= blocks_per_row; blk += U) {
    Raw raw[U];
    Raw raw_up[kSwiGLU ? U : 1];
#pragma unroll
    for (int u = 0; u < U; ++u) {
      raw[u] = F::Load(base + (uint64_t)(blk + u) * F::kBlockBytes, t);
      if constexpr (kSwiGLU) raw_up[u] = F::Load(base_up + (uint64_t)(blk + u) * F::kBlockBytes, t);
    }
#pragma unroll
    for (int u = 0; u < U; ++u) {
      acc = F::Acc(acc, raw[u], t, xt + (blk + u) * 256, grid);
      if constexpr (kSwiGLU) acc_up = F::Acc(acc_up, raw_up[u], t, xt + (blk + u) * 256, grid);
    }
  }
  for (; blk < blocks_per_row; ++blk) {
    acc = F::Acc(acc, F::Load(base + (uint64_t)blk * F::kBlockBytes, t), t, xt + blk * 256, grid);
    if constexpr (kSwiGLU)
      acc_up = F::Acc(acc_up, F::Load(base_up + (uint64_t)blk * F::kBlockBytes, t), t, xt + blk * 256, grid);
  }
#pragma unroll
  for (int offset = 4; offset > 0; offset >>= 1) {
    acc += __shfl_xor(acc, offset, kWave);
    if constexpr (kSwiGLU) acc_up += __shfl_xor(acc_up, offset, kWave);
  }
  if (t != 0) return;
  if constexpr (kSwiGLU) {
    float g = acc;
    float u = acc_up;
    if (limit > 0.0f) {
      u = fminf(fmaxf(u, -limit), limit);
      g = fminf(g, limit);
    }
    const float h = g / (1.0f + expf(-g)) * u;
    y[row] = route_w ? route_w[blockIdx.y] * h : h;
  } else {
    y[row] = acc;
  }
}

/* ds41_v41_expert_plan: one 64-thread block groups the slots of `ids` by
 * expert. plan[0] is the group count; group g, at plan + 1 + g * kPlanGroup,
 * holds the expert id, the slot count and the slots in ascending order. */
constexpr int kPlanCols = DS41_V41_EXPERT_PLAN_COLS;
constexpr int kPlanGroup = 2 + kPlanCols;

__global__ void expert_plan_kernel(const int32_t* __restrict__ ids, uint32_t slots, int32_t* __restrict__ plan) {
  __shared__ int32_t sid[64];
  __shared__ int32_t group_of[64];
  const uint32_t s = threadIdx.x;
  if (s < slots) sid[s] = ids[s];
  __syncthreads();
  int32_t id = 0, col = 0, count = 0, group = 0;
  bool leader = false;
  if (s < slots) {
    id = sid[s];
    leader = true;
    for (uint32_t o = 0; o < slots; ++o) {
      const bool same = sid[o] == id;
      count += same;
      if (o < s && same) {
        ++col;
        leader = false;
      }
    }
    /* Groups are numbered by their first slot. */
    for (uint32_t o = 0; o < s; ++o) {
      bool first = true;
      for (uint32_t p = 0; p < o && first; ++p) first = sid[p] != sid[o];
      group += first;
    }
    if (leader) group_of[s] = group;
  }
  __syncthreads();
  if (s >= slots) return;
  if (!leader) {
    for (uint32_t o = 0; o < s; ++o)
      if (sid[o] == id) {
        group = group_of[o];
        break;
      }
  }
  int32_t* g = plan + 1 + group * kPlanGroup;
  if (leader) {
    g[0] = id;
    g[1] = count;
  }
  if (col < kPlanCols) g[2 + col] = (int32_t)s;
  if (s == slots - 1) {
    int32_t groups = 0;
    for (uint32_t o = 0; o < slots; ++o) {
      bool first = true;
      for (uint32_t p = 0; p < o && first; ++p) first = sid[p] != sid[o];
      groups += first;
    }
    plan[0] = groups;
  }
}

/* expert_gemv_kernel over the groups of a plan: block (row tile, group)
 * decodes its expert's weights once for the group's up to kCols slots, slot
 * s reading x + (s / slots_per_x) * x_slot and writing row s of y. Every
 * slot sees the arithmetic of expert_gemv_kernel. */
template <int kType, int U, bool kSwiGLU, int kCols>
__global__ void __launch_bounds__(kWave* kWaves)
    expert_group_kernel(const uint8_t* __restrict__ w, const uint8_t* __restrict__ w_up,
                        const float* __restrict__ route_w, float limit, const float* __restrict__ x,
                        float* __restrict__ y, uint32_t m, uint32_t k, const int32_t* __restrict__ plan,
                        uint64_t expert_bytes, uint32_t x_slot, uint32_t slots_per_x) {
  using F = Format<kType>;
  constexpr uint32_t kRows = 4 * kWaves;
  __shared__ uint32_t grid[F::kIq1Grid ? NGRID_IQ1S : 1];
  if ((int32_t)blockIdx.y >= plan[0]) return;
  const int32_t* g = plan + 1 + blockIdx.y * kPlanGroup;
  const int32_t id = g[0];
  const int cols = g[1] < kCols ? g[1] : kCols;
  int32_t slot[kCols];
#pragma unroll
  for (int c = 0; c < kCols; ++c) slot[c] = c < cols ? g[2 + c] : 0;
  if (id < 0) {  // dropped slots
    for (int c = 0; c < cols; ++c)
      for (uint32_t r = threadIdx.x; r < kRows; r += blockDim.x)
        if (blockIdx.x * kRows + r < m) y[(uint64_t)slot[c] * m + blockIdx.x * kRows + r] = 0.0f;
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
  const float* xt[kCols];
#pragma unroll
  for (int c = 0; c < kCols; ++c) xt[c] = x + (uint64_t)(slot[c] / slots_per_x) * x_slot + 32 * t;
  const uint32_t blocks_per_row = k / 256;
  const uint64_t offset = (uint64_t)id * expert_bytes + (uint64_t)row * blocks_per_row * F::kBlockBytes;
  const uint8_t* base = w + offset;
  const uint8_t* base_up = kSwiGLU ? w_up + offset : nullptr;
  float acc[kCols] = {};
  float acc_up[kCols] = {};
  uint32_t blk = 0;
  for (; blk + U <= blocks_per_row; blk += U) {
    Raw raw[U];
    Raw raw_up[kSwiGLU ? U : 1];
#pragma unroll
    for (int u = 0; u < U; ++u) {
      raw[u] = F::Load(base + (uint64_t)(blk + u) * F::kBlockBytes, t);
      if constexpr (kSwiGLU) raw_up[u] = F::Load(base_up + (uint64_t)(blk + u) * F::kBlockBytes, t);
    }
#pragma unroll
    for (int u = 0; u < U; ++u) {
#pragma unroll
      for (int c = 0; c < kCols; ++c) {
        if (c >= cols) break;
        acc[c] = F::Acc(acc[c], raw[u], t, xt[c] + (blk + u) * 256, grid);
        if constexpr (kSwiGLU) acc_up[c] = F::Acc(acc_up[c], raw_up[u], t, xt[c] + (blk + u) * 256, grid);
      }
    }
  }
  for (; blk < blocks_per_row; ++blk) {
    const Raw raw = F::Load(base + (uint64_t)blk * F::kBlockBytes, t);
    Raw raw_up{};
    if constexpr (kSwiGLU) raw_up = F::Load(base_up + (uint64_t)blk * F::kBlockBytes, t);
#pragma unroll
    for (int c = 0; c < kCols; ++c) {
      if (c >= cols) break;
      acc[c] = F::Acc(acc[c], raw, t, xt[c] + blk * 256, grid);
      if constexpr (kSwiGLU) acc_up[c] = F::Acc(acc_up[c], raw_up, t, xt[c] + blk * 256, grid);
    }
  }
#pragma unroll
  for (int c = 0; c < kCols; ++c) {
#pragma unroll
    for (int offset = 4; offset > 0; offset >>= 1) {
      acc[c] += __shfl_xor(acc[c], offset, kWave);
      if constexpr (kSwiGLU) acc_up[c] += __shfl_xor(acc_up[c], offset, kWave);
    }
  }
  if (t != 0) return;
#pragma unroll
  for (int c = 0; c < kCols; ++c) {
    if (c >= cols) break;
    float* yr = y + (uint64_t)slot[c] * m;
    if constexpr (kSwiGLU) {
      float gv = acc[c];
      float uv = acc_up[c];
      if (limit > 0.0f) {
        uv = fminf(fmaxf(uv, -limit), limit);
        gv = fminf(gv, limit);
      }
      const float h = gv / (1.0f + expf(-gv)) * uv;
      yr[row] = route_w ? route_w[slot[c]] * h : h;
    } else {
      yr[row] = acc[c];
    }
  }
}

template <int kType, int U, int kCols>
int launch_group(const void* w, const void* w_up, const float* route_w, float limit, uint64_t expert_bytes,
                 const int32_t* plan, uint32_t groups, const float* x, uint32_t x_slot, uint32_t slots_per_x,
                 float* y, uint32_t m, uint32_t k) {
  constexpr uint32_t kRows = 4 * kWaves;
  const dim3 blocks((m + kRows - 1) / kRows, groups);
  if (w_up)
    expert_group_kernel<kType, U, true, kCols><<<blocks, kWave * kWaves>>>(
        static_cast<const uint8_t*>(w), static_cast<const uint8_t*>(w_up), route_w, limit, x, y, m, k, plan,
        expert_bytes, x_slot, slots_per_x);
  else
    expert_group_kernel<kType, U, false, kCols><<<blocks, kWave * kWaves>>>(
        static_cast<const uint8_t*>(w), nullptr, nullptr, 0.0f, x, y, m, k, plan, expert_bytes, x_slot,
        slots_per_x);
  const hipError_t err = hipGetLastError();
  if (err == hipSuccess) return 0;
  fprintf(stderr, "ds41: grouped expert gemv launch failed: %s\n", hipGetErrorString(err));
  return 1;
}

template <int kType, int U>
int dispatch_group(const void* w, const void* w_up, const float* route_w, float limit, uint64_t expert_bytes,
                   const int32_t* plan, uint32_t groups, uint32_t cols, const float* x, uint32_t x_slot,
                   uint32_t slots_per_x, float* y, uint32_t m, uint32_t k) {
  if (cols <= 2)
    return launch_group<kType, U, 2>(w, w_up, route_w, limit, expert_bytes, plan, groups, x, x_slot, slots_per_x, y,
                                     m, k);
  if (cols <= 4)
    return launch_group<kType, U, 4>(w, w_up, route_w, limit, expert_bytes, plan, groups, x, x_slot, slots_per_x, y,
                                     m, k);
  return launch_group<kType, U, kPlanCols>(w, w_up, route_w, limit, expert_bytes, plan, groups, x, x_slot,
                                           slots_per_x, y, m, k);
}

template <int kType, int U>
int launch(const void* w, const void* w_up, const float* route_w, float limit, uint64_t expert_bytes,
           const int32_t* ids, uint32_t slots, const float* x, uint32_t x_slot, uint32_t slots_per_x, float* y,
           uint32_t m, uint32_t k) {
  constexpr uint32_t kRows = 4 * kWaves;
  const dim3 blocks((m + kRows - 1) / kRows, slots);
  if (w_up)
    expert_gemv_kernel<kType, U, true><<<blocks, kWave * kWaves>>>(static_cast<const uint8_t*>(w),
                                                                    static_cast<const uint8_t*>(w_up), route_w,
                                                                    limit, x, y, m, k, ids, expert_bytes, x_slot,
                                                                    slots_per_x);
  else
    expert_gemv_kernel<kType, U, false><<<blocks, kWave * kWaves>>>(
        static_cast<const uint8_t*>(w), nullptr, nullptr, 0.0f, x, y, m, k, ids, expert_bytes, x_slot, slots_per_x);
  const hipError_t err = hipGetLastError();
  if (err == hipSuccess) return 0;
  fprintf(stderr, "ds41: expert gemv launch failed: %s\n", hipGetErrorString(err));
  return 1;
}

}  // namespace

namespace {

int dispatch(int type, const void* w, const void* w_up, const float* route_w, float limit, uint64_t expert_bytes,
             const int32_t* ids, uint32_t slots, const float* x, uint32_t x_slot, uint32_t slots_per_x, float* y,
             uint32_t m, uint32_t k) {
  if (!ids || slots == 0 || slots_per_x == 0 || m == 0 || k % 256 != 0) return 1;
  /* Blocks loaded ahead: 2 measured best for IQ1_S, 1 for the others. */
  switch (type) {
  case 19:
    return launch<19, 2>(w, w_up, route_w, limit, expert_bytes, ids, slots, x, x_slot, slots_per_x, y, m, k);
  case 29:
    return launch<29, 1>(w, w_up, route_w, limit, expert_bytes, ids, slots, x, x_slot, slots_per_x, y, m, k);
  case 16:
    return launch<16, 1>(w, w_up, route_w, limit, expert_bytes, ids, slots, x, x_slot, slots_per_x, y, m, k);
  case 22:
    return launch<22, 1>(w, w_up, route_w, limit, expert_bytes, ids, slots, x, x_slot, slots_per_x, y, m, k);
  default:
    return 1;
  }
}

}  // namespace

extern "C" int ds41_v41_gemv_experts(int type, const void* w, uint64_t expert_bytes, const int32_t* ids,
                                     uint32_t slots, const float* x, uint32_t x_slot, uint32_t slots_per_x, float* y,
                                     uint32_t m, uint32_t k) {
  return dispatch(type, w, nullptr, nullptr, 0.0f, expert_bytes, ids, slots, x, x_slot, slots_per_x, y, m, k);
}

extern "C" int ds41_v41_gemv_experts_swiglu(int type, const void* gate, const void* up, uint64_t expert_bytes,
                                            const int32_t* ids, uint32_t slots, const float* x, uint32_t x_slot,
                                            uint32_t slots_per_x, const float* route_w, float limit, float* y,
                                            uint32_t m, uint32_t k) {
  if (!gate || !up) return 1;
  return dispatch(type, gate, up, route_w, limit, expert_bytes, ids, slots, x, x_slot, slots_per_x, y, m, k);
}

extern "C" int ds41_v41_expert_plan(const int32_t* ids, uint32_t slots, int32_t* plan) {
  if (!ids || !plan || slots == 0 || slots > DS41_V41_EXPERT_PLAN_SLOTS) return 1;
  expert_plan_kernel<<<1, 64>>>(ids, slots, plan);
  const hipError_t err = hipGetLastError();
  if (err == hipSuccess) return 0;
  fprintf(stderr, "ds41: expert plan launch failed: %s\n", hipGetErrorString(err));
  return 1;
}

extern "C" int ds41_v41_gemv_experts_grouped(int type, const void* w, const void* w_up, const float* route_w,
                                             float limit, uint64_t expert_bytes, const int32_t* plan, uint32_t slots,
                                             uint32_t cols, const float* x, uint32_t x_slot, uint32_t slots_per_x,
                                             float* y, uint32_t m, uint32_t k) {
  if (!plan || slots == 0 || slots > DS41_V41_EXPERT_PLAN_SLOTS || cols == 0 || cols > kPlanCols ||
      slots_per_x == 0 || m == 0 || k % 256 != 0)
    return 1;
  switch (type) {
  case 19:
    return dispatch_group<19, 2>(w, w_up, route_w, limit, expert_bytes, plan, slots, cols, x, x_slot, slots_per_x, y,
                                 m, k);
  case 29:
    return dispatch_group<29, 1>(w, w_up, route_w, limit, expert_bytes, plan, slots, cols, x, x_slot, slots_per_x, y,
                                 m, k);
  case 16:
    return dispatch_group<16, 1>(w, w_up, route_w, limit, expert_bytes, plan, slots, cols, x, x_slot, slots_per_x, y,
                                 m, k);
  case 22:
    return dispatch_group<22, 1>(w, w_up, route_w, limit, expert_bytes, plan, slots, cols, x, x_slot, slots_per_x, y,
                                 m, k);
  default:
    return 1;
  }
}
