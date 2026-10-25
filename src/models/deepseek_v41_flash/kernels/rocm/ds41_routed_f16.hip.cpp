/* Batched routed-expert projections for IQ1_S weights on the FP16 matrix
 * cores. The activations are rounded to FP16 (11-bit mantissa, against the
 * 8-bit block quantization of MMQ's Q8_1 inputs) and each weight decodes to
 * FP16 as d s (n - 1 + delta) with its scale folded in, so the K loop issues
 * no per-block scale arithmetic beside the matrix-core products; the
 * products accumulate in FP32.
 *
 * Work is listed by the host: `rows` holds the slots grouped by expert
 * (expert e owns rows[bounds[e] .. bounds[e + 1])), and each tile entry
 * (expert | index << 16) names 64 of them. A workgroup takes one tile and
 * 128 (gate/up pair) or 256 (single) weight rows; a wave holds 16 rows of
 * each pair projection, or 32 rows, as the B operand (lane = weight row), so
 * one decoded weight fragment feeds every live 16-slot tile and the stores
 * of a token's rows are contiguous. */
#include <hip/hip_runtime.h>

#include <cstdint>

#include "ds41_v41.h"
#include "mmq/ggml-common.h"

namespace {

typedef _Float16 half16 __attribute__((ext_vector_type(16)));
typedef float float8 __attribute__((ext_vector_type(8)));

constexpr int kSlots = 64;              // slots per tile
constexpr int kStageK = 128;            // activation values per LDS stage
constexpr int kActPitch = kStageK + 8;  // halves; rows land on distinct banks
constexpr int kBlockBytes = 50;         // IQ1_S: d, qs[32], qh[8]

/* The 50 bytes of one IQ1_S block as 13 words starting at the block (the
 * last one holds qh[7] in its low half). Blocks are 2-byte aligned; the
 * loads are aligned words, the 13th of which never crosses a page. */
struct Block {
  uint32_t w[13];
};

__device__ __forceinline__ Block load_raw(const uint8_t* p) {
  const auto* a = reinterpret_cast<const uint32_t*>(reinterpret_cast<uintptr_t>(p) & ~uintptr_t{3});
  Block b;
#pragma unroll
  for (int i = 0; i < 3; ++i) {
    typedef uint32_t u32x4 __attribute__((ext_vector_type(4)));
    const u32x4 v = reinterpret_cast<const u32x4*>(a)[i];
    b.w[4 * i] = v.x;
    b.w[4 * i + 1] = v.y;
    b.w[4 * i + 2] = v.z;
    b.w[4 * i + 3] = v.w;
  }
  b.w[12] = a[12];
  return b;
}

__device__ __forceinline__ Block realign(const Block& raw, uint32_t shift) {
  Block b;
#pragma unroll
  for (int i = 0; i < 12; ++i) b.w[i] = __builtin_amdgcn_alignbyte(raw.w[i + 1], raw.w[i], shift);
  b.w[12] = raw.w[12] >> (8 * shift);
  return b;
}

/* Sub-block ib of a realigned block: its four qs bytes and qh word. */
__device__ __forceinline__ uint32_t block_qs(const Block& b, int ib) {
  return __builtin_amdgcn_alignbyte(b.w[ib + 1], b.w[ib], 2);
}
__device__ __forceinline__ uint32_t block_qh(const Block& b, int ib) {
  const int byte = 34 + 2 * ib;
  return (b.w[byte / 4] >> (8 * (byte % 4))) & 0xffffu;
}

/* (scale, bias) of sub-block ib as a half2: weight = scale n + bias. */
__device__ __forceinline__ uint32_t block_affine(const Block& b, uint32_t qh) {
  const float d = static_cast<float>(__builtin_bit_cast(_Float16, static_cast<uint16_t>(b.w[0] & 0xffffu)));
  const float scale = d * static_cast<float>(2 * ((qh >> 12) & 7) + 1);
  const float delta = (qh & 0x8000u) ? -1.0f - IQ1S_DELTA : -1.0f + IQ1S_DELTA;
  const _Float16 s = static_cast<_Float16>(scale), o = static_cast<_Float16>(scale * delta);
  return static_cast<uint32_t>(__builtin_bit_cast(uint16_t, s)) |
         static_cast<uint32_t>(__builtin_bit_cast(uint16_t, o)) << 16;
}

typedef _Float16 half2_t __attribute__((ext_vector_type(2)));

/* The codebook in LDS as the high bytes of the FP16 values 0, 1 and 2 (the
 * low bytes are zero), one byte per weight in element order. */
__device__ __forceinline__ uint2 grid_high_bytes(uint32_t grid) {
  uint32_t out[2] = {0, 0};
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const uint32_t n = (grid >> (8 * (i % 4) + 4 * (i / 4))) & 0xfu;
    out[i / 4] |= (n == 0 ? 0x00u : n == 1 ? 0x3cu : 0x40u) << (8 * (i % 4));
  }
  return make_uint2(out[0], out[1]);
}

/* Eight weights of one grid group as four half2: the FP16 value n is
 * assembled from its high byte, then the affine. */
__device__ __forceinline__ void decode8(uint2 bytes, uint32_t affine, uint32_t* out) {
  const half2_t s = __builtin_bit_cast(half2_t, affine);
  const half2_t scale = {s.x, s.x}, bias = {s.y, s.y};
  const uint32_t p[4] = {
      __builtin_amdgcn_perm(bytes.x, 0u, 0x050c040cu), __builtin_amdgcn_perm(bytes.x, 0u, 0x070c060cu),
      __builtin_amdgcn_perm(bytes.y, 0u, 0x050c040cu), __builtin_amdgcn_perm(bytes.y, 0u, 0x070c060cu)};
#pragma unroll
  for (int i = 0; i < 4; ++i) out[i] = __builtin_bit_cast(uint32_t, __builtin_bit_cast(half2_t, p[i]) * scale + bias);
}

__device__ __forceinline__ float finite_or_zero(float v) { return __builtin_isfinite(v) ? v : 0.0f; }

template <bool kPair, bool kHalfOut>
__global__ __launch_bounds__(256) void routed_iq1s_f16_kernel(
    const uint8_t* __restrict__ w0, const uint8_t* __restrict__ w1, uint64_t expert_bytes,
    const uint16_t* __restrict__ x, uint32_t x_div, const int32_t* __restrict__ rows,
    const int32_t* __restrict__ bounds, const int32_t* __restrict__ tiles, const float* __restrict__ route_w,
    float limit, void* __restrict__ y, uint32_t m, uint32_t k) {
  __shared__ uint2 grid[NGRID_IQ1S];
  __shared__ __attribute__((aligned(16))) _Float16 act[kSlots * kActPitch];
  __shared__ int32_t slot_of[kSlots];
  const int tile = tiles[blockIdx.y];
  const int expert = tile & 0xffff;
  const int t0 = (tile >> 16) * kSlots;
  const int begin = bounds[expert] + t0;
  const int count = min(kSlots, bounds[expert + 1] - begin);
  const int live = (count + 15) / 16;
  const int tid = threadIdx.x, wave = tid / 32, lane = tid % 32;
  for (int i = tid; i < NGRID_IQ1S; i += 256) grid[i] = grid_high_bytes(iq1s_grid_gpu[i]);
  if (tid < kSlots) slot_of[tid] = tid < count ? rows[begin + tid] : -1;

  constexpr int kWaveRows = kPair ? 16 : 32;
  const uint32_t r_wave = blockIdx.x * 8 * kWaveRows + wave * kWaveRows;
  const uint64_t row_bytes = (uint64_t)k / 256 * kBlockBytes;
  const uint8_t* wp[2];
  uint32_t row[2];
#pragma unroll
  for (int p = 0; p < 2; ++p) {
    row[p] = r_wave + (kPair ? 0 : 16 * p) + lane % 16;
    const uint8_t* base = (kPair && p ? w1 : w0) + (uint64_t)expert * expert_bytes;
    wp[p] = base + (uint64_t)(row[p] < m ? row[p] : m - 1) * row_bytes;
  }

  float8 acc[2][4];
#pragma unroll
  for (int p = 0; p < 2; ++p)
#pragma unroll
    for (int t = 0; t < 4; ++t) acc[p][t] = float8{};

  /* Activation staging: a quarter of the stage (32 values) of one slot per
   * thread, consecutive slots in consecutive lanes. */
  const int st = tid % kSlots, sq = tid / kSlots;
  __syncthreads();
  const int32_t s_slot = slot_of[st];
  const uint16_t* xs = s_slot >= 0 ? x + (uint64_t)(s_slot / (int)x_div) * k + 32 * sq : nullptr;
  _Float16* as = act + st * kActPitch + 32 * sq;

  /* Weights: half-wave h loads the blocks of projection h (gate or up, or
   * the second 16 rows) two blocks ahead; the halves swap their words across
   * lanes. Ordinary loads: the 50-byte blocks share cache lines with the
   * next block, which nontemporal loads fetched again from DRAM. */
  const int h = lane / 16;
  const uint8_t* own = h ? wp[1] : wp[0];
  const uint32_t blocks = k / 256;
  Block ring[2];
  ring[0] = load_raw(own);
  if (blocks > 1) ring[1] = load_raw(own + kBlockBytes);
  const auto step = [&](uint32_t sb, Block& slot) {
    const Block mine = slot;
    if (sb + 2 < blocks) slot = load_raw(own + (uint64_t)(sb + 2) * kBlockBytes);
    Block blk[2];
#pragma unroll
    for (int i = 0; i < 13; ++i) {
      const uint32_t other = __builtin_amdgcn_permlanex16(mine.w[i], mine.w[i], 0x76543210u, 0xfedcba98u, false, false);
      blk[0].w[i] = h ? other : mine.w[i];
      blk[1].w[i] = h ? mine.w[i] : other;
    }
#pragma unroll
    for (int p = 0; p < 2; ++p)
      blk[p] = realign(blk[p], (reinterpret_cast<uintptr_t>(wp[p]) + sb * kBlockBytes) & 3);
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      const uint32_t k0 = sb * 256 + half * kStageK;
      __syncthreads();
      if (xs) {
        const uint4* src = reinterpret_cast<const uint4*>(xs + k0);
        uint4 v[4];
#pragma unroll
        for (int c = 0; c < 4; ++c) v[c] = src[c];
#pragma unroll
        for (int c = 0; c < 4; ++c) reinterpret_cast<uint4*>(as)[c] = v[c];
      }
      __syncthreads();
#pragma unroll
      for (int jj = 0; jj < 4; ++jj) {
        const int ib = half * 4 + jj;
        uint32_t qs[2], qh[2], affine[2];
#pragma unroll
        for (int p = 0; p < 2; ++p) {
          qs[p] = block_qs(blk[p], ib);
          qh[p] = block_qh(blk[p], ib);
          affine[p] = block_affine(blk[p], qh[p]);
        }
#pragma unroll
        for (int e2 = 0; e2 < 2; ++e2) {
          /* Each half-wave decodes one of the two groups (half h group
           * 2 e2 + h) and takes the other from the opposite half, so
           * half 1 holds the K16 step with its two groups swapped; its
           * activation fragment is read swapped to match. */
          half16 b[2];
#pragma unroll
          for (int p = 0; p < 2; ++p) {
            uint32_t words[8];
            const int l = 2 * e2 + h;
            const uint32_t idx = ((qs[p] >> (8 * l)) & 0xffu) | (((qh[p] >> (3 * l)) & 7u) << 8);
            decode8(grid[idx], affine[p], words);
#pragma unroll
            for (int i = 0; i < 4; ++i)
              words[4 + i] = __builtin_amdgcn_permlanex16(words[i], words[i], 0x76543210u, 0xfedcba98u, false, false);
            b[p] = __builtin_bit_cast(half16, words);
          }
          const int kk = jj * 32 + e2 * 16;
#pragma unroll
          for (int t = 0; t < 4; ++t) {
            if (t < live) {
              const _Float16* ar = act + (t * 16 + lane % 16) * kActPitch + kk;
              typedef _Float16 half8 __attribute__((ext_vector_type(8)));
              const half8 a0 = *reinterpret_cast<const half8*>(ar + 8 * h);
              const half8 a1 = *reinterpret_cast<const half8*>(ar + 8 - 8 * h);
              const half16 a = __builtin_shufflevector(a0, a1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
#pragma unroll
              for (int p = 0; p < 2; ++p)
                acc[p][t] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b[p], acc[p][t]);
            }
          }
        }
      }
    }
  };
  for (uint32_t sb = 0; sb + 1 < blocks; sb += 2) {
    step(sb, ring[0]);
    step(sb + 1, ring[1]);
  }
  if (blocks % 2) step(blocks - 1, ring[0]);

  /* Accumulator element e of a lane is (slot 2 e + lane / 16, row lane % 16). */
#pragma unroll
  for (int t = 0; t < 4; ++t) {
    if (t >= live) continue;
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      const int32_t slot = slot_of[t * 16 + 2 * e + lane / 16];
      if (slot < 0) continue;
      if constexpr (kPair) {
        if (row[0] >= m) continue;
        float g = finite_or_zero(acc[0][t][e]), u = finite_or_zero(acc[1][t][e]);
        if (limit > 1.0e-6f) {
          g = fminf(g, limit);
          u = fminf(fmaxf(u, -limit), limit);
        }
        const float v = g / (1.0f + __expf(-g)) * u * route_w[slot];
        if constexpr (kHalfOut)
          static_cast<_Float16*>(y)[(uint64_t)slot * m + row[0]] = static_cast<_Float16>(v);
        else
          static_cast<float*>(y)[(uint64_t)slot * m + row[0]] = v;
      } else {
#pragma unroll
        for (int p = 0; p < 2; ++p)
          if (row[p] < m) static_cast<float*>(y)[(uint64_t)slot * m + row[p]] = finite_or_zero(acc[p][t][e]);
      }
    }
  }
}

}  // namespace

extern "C" int ds41_v41_routed_f16(int type, const void* w, const void* w_up, uint64_t expert_bytes, const uint16_t* x,
                                   uint32_t x_div, const int32_t* rows, const int32_t* bounds, const int32_t* tiles,
                                   uint32_t n_tiles, const float* route_w, float limit, void* y, int y_half,
                                   uint32_t m, uint32_t k) {
  if (type != 19 || k % 256 != 0 || n_tiles == 0 || x_div == 0) return 1;
  const auto* a = static_cast<const uint8_t*>(w);
  if (w_up && y_half) {
    routed_iq1s_f16_kernel<true, true><<<dim3((m + 127) / 128, n_tiles), 256>>>(
        a, static_cast<const uint8_t*>(w_up), expert_bytes, x, x_div, rows, bounds, tiles, route_w, limit, y, m, k);
  } else if (w_up) {
    routed_iq1s_f16_kernel<true, false><<<dim3((m + 127) / 128, n_tiles), 256>>>(
        a, static_cast<const uint8_t*>(w_up), expert_bytes, x, x_div, rows, bounds, tiles, route_w, limit, y, m, k);
  } else {
    routed_iq1s_f16_kernel<false, false><<<dim3((m + 255) / 256, n_tiles), 256>>>(
        a, nullptr, expert_bytes, x, x_div, rows, bounds, tiles, nullptr, 0.0f, y, m, k);
  }
  return hipGetLastError() == hipSuccess ? 0 : 1;
}

__global__ void to_half_kernel(const float* __restrict__ x, _Float16* __restrict__ y, uint64_t n) {
  const uint64_t i = ((uint64_t)blockIdx.x * blockDim.x + threadIdx.x) * 4;
  if (i + 4 <= n) {
    const float4 v = *reinterpret_cast<const float4*>(x + i);
    typedef _Float16 half4_t __attribute__((ext_vector_type(4)));
    *reinterpret_cast<half4_t*>(y + i) = half4_t{(_Float16)v.x, (_Float16)v.y, (_Float16)v.z, (_Float16)v.w};
  } else {
    for (uint64_t j = i; j < n; ++j) y[j] = (_Float16)x[j];
  }
}

extern "C" int ds41_v41_to_half(const float* x, uint16_t* y, uint64_t n) {
  if (n == 0) return 0;
  to_half_kernel<<<(unsigned)((n + 1023) / 1024), 256>>>(x, reinterpret_cast<_Float16*>(y), n);
  return hipGetLastError() == hipSuccess ? 0 : 1;
}

extern "C" uint32_t ds41_v41_routed_plan(const int32_t* ids, uint32_t slots, uint32_t n_experts, int32_t* rows,
                                         int32_t* bounds, int32_t* tiles) {
  for (uint32_t e = 0; e <= n_experts; ++e) bounds[e] = 0;
  for (uint32_t s = 0; s < slots; ++s)
    if (ids[s] >= 0 && (uint32_t)ids[s] < n_experts) ++bounds[ids[s] + 1];
  for (uint32_t e = 0; e < n_experts; ++e) bounds[e + 1] += bounds[e];
  uint32_t n_tiles = 0;
  for (uint32_t e = 0; e < n_experts; ++e)
    for (int32_t t = 0; t * kSlots < bounds[e + 1] - bounds[e]; ++t)
      tiles[n_tiles++] = (int32_t)(e | (uint32_t)t << 16);
  /* Fill each bucket in slot order, using the next bucket's start as a cursor. */
  for (uint32_t s = 0; s < slots; ++s)
    if (ids[s] >= 0 && (uint32_t)ids[s] < n_experts) rows[bounds[ids[s]]++] = (int32_t)s;
  for (uint32_t e = n_experts; e > 0; --e) bounds[e] = bounds[e - 1];
  bounds[0] = 0;
  return n_tiles;
}
