/* Batched routed-expert projections for IQ1_S, IQ1_M, IQ2_XXS and IQ2_XS weights on
 * the FP16 matrix cores. The activations are rounded to FP16 (11-bit
 * mantissa, against the 8-bit block quantization of MMQ's Q8_1 inputs) and
 * each weight decodes to FP16 with its sub-block scale folded in, so the K
 * loop issues no per-block scale arithmetic beside the matrix-core products;
 * the products accumulate in FP32.
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
constexpr int kStageK = 64;             // activation values per LDS stage
constexpr int kStages = 256 / kStageK;  // stages per weight block
constexpr int kVec = kStageK / 32;      // 16-byte words staged per thread
constexpr int kActPitch = kStageK + 8;  // halves; rows land on distinct banks

typedef _Float16 half2_t __attribute__((ext_vector_type(2)));
typedef uint32_t u32x4 __attribute__((ext_vector_type(4)));

/* The words of one block from the aligned word at or below it: kWords words,
 * enough for a 2-byte aligned block, the last of which never crosses a page
 * the block does not reach. */
template <int kWords>
struct Raw {
  uint32_t w[kWords];
};

template <int kWords>
__device__ __forceinline__ Raw<kWords> load_raw(const uint8_t* p) {
  const auto* a = reinterpret_cast<const uint32_t*>(reinterpret_cast<uintptr_t>(p) & ~uintptr_t{3});
  Raw<kWords> b;
#pragma unroll
  for (int i = 0; i < kWords / 4; ++i) {
    const u32x4 v = reinterpret_cast<const u32x4*>(a)[i];
    b.w[4 * i] = v.x;
    b.w[4 * i + 1] = v.y;
    b.w[4 * i + 2] = v.z;
    b.w[4 * i + 3] = v.w;
  }
#pragma unroll
  for (int i = kWords / 4 * 4; i < kWords; ++i) b.w[i] = a[i];
  return b;
}

/* Words starting at the block itself, `shift` bytes past the aligned word. */
template <int kWords>
__device__ __forceinline__ Raw<kWords> realign(const Raw<kWords>& raw, uint32_t shift) {
  Raw<kWords> b;
#pragma unroll
  for (int i = 0; i + 1 < kWords; ++i) b.w[i] = __builtin_amdgcn_alignbyte(raw.w[i + 1], raw.w[i], shift);
  b.w[kWords - 1] = raw.w[kWords - 1] >> (8 * shift);
  return b;
}

__device__ __forceinline__ uint32_t pack_half2(float lo, float hi) {
  return __builtin_bit_cast(uint32_t, half2_t{static_cast<_Float16>(lo), static_cast<_Float16>(hi)});
}

__device__ __forceinline__ float half_bits(uint32_t bits) {
  return static_cast<float>(__builtin_bit_cast(_Float16, static_cast<uint16_t>(bits & 0xffffu)));
}

/* High bytes of the FP16 values of the IQ1 codebook (0, 1 and 2; the low
 * bytes are zero), one byte per weight in element order. */
__device__ __forceinline__ uint2 iq1_high_bytes(uint32_t grid) {
  uint32_t out[2] = {0, 0};
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const uint32_t n = (grid >> (8 * (i % 4) + 4 * (i / 4))) & 0xfu;
    out[i / 4] |= (n == 0 ? 0x00u : n == 1 ? 0x3cu : 0x40u) << (8 * (i % 4));
  }
  return make_uint2(out[0], out[1]);
}

/* Eight IQ1 weights as four half2: the codebook values n from their high
 * bytes, then scale n + bias. */
__device__ __forceinline__ void iq1_decode8(uint2 bytes, uint32_t affine, uint32_t* out) {
  const half2_t s = __builtin_bit_cast(half2_t, affine);
  const half2_t scale = {s.x, s.x}, bias = {s.y, s.y};
  const uint32_t p[4] = {
      __builtin_amdgcn_perm(bytes.x, 0u, 0x050c040cu), __builtin_amdgcn_perm(bytes.x, 0u, 0x070c060cu),
      __builtin_amdgcn_perm(bytes.y, 0u, 0x050c040cu), __builtin_amdgcn_perm(bytes.y, 0u, 0x070c060cu)};
#pragma unroll
  for (int i = 0; i < 4; ++i) out[i] = __builtin_bit_cast(uint32_t, __builtin_bit_cast(half2_t, p[i]) * scale + bias);
}

/* Weight formats. Each gives its block size, the words loaded per block
 * (kAligned: blocks start on words), the LDS tables, the fields of one
 * 32-value sub-block (Sub) and the eight FP16 weights of group l of it. */
template <int kType>
struct Format;

/* IQ1_S (50 bytes): d, qs[32], qh[8] as u16. Group l of sub-block ib takes
 * codebook entry qs[4 ib + l] | ((qh[ib] >> 3 l) & 7) << 8; the sub-block
 * scale d (2 ((qh >> 12) & 7) + 1) and delta -1 +- 1/8 (qh bit 15) apply to
 * all four groups. */
template <>
struct Format<19> {
  static constexpr int kBytes = 50, kWords = 13;
  static constexpr bool kAligned = false;
  struct Tables {
    uint2 grid[NGRID_IQ1S];
  };
  static __device__ void fill(Tables& t, int tid) {
    for (int i = tid; i < NGRID_IQ1S; i += 256) t.grid[i] = iq1_high_bytes(iq1s_grid_gpu[i]);
  }
  struct Sub {
    uint32_t qs, qh, affine;
  };
  static __device__ __forceinline__ Sub sub(const Raw<kWords>& b, int ib) {
    const int byte = 34 + 2 * ib;
    const uint32_t qh = (b.w[byte / 4] >> (8 * (byte % 4))) & 0xffffu;
    const float scale = half_bits(b.w[0]) * static_cast<float>(2 * ((qh >> 12) & 7) + 1);
    const float delta = (qh & 0x8000u) ? -1.0f - IQ1S_DELTA : -1.0f + IQ1S_DELTA;
    return {__builtin_amdgcn_alignbyte(b.w[ib + 1], b.w[ib], 2), qh, pack_half2(scale, scale * delta)};
  }
  static __device__ __forceinline__ void decode(const Sub& s, int l, const Tables& t, uint32_t* out) {
    const uint32_t idx = ((s.qs >> (8 * l)) & 0xffu) | (((s.qh >> (3 * l)) & 7u) << 8);
    iq1_decode8(t.grid[idx], s.affine, out);
  }
};

/* IQ1_M (56 bytes): qs[32], qh[16], scales[8]. Group l of sub-block ib takes
 * codebook entry qs[4 ib + l] | (nibble & 7) << 8 with nibble l % 2 of
 * qh[2 ib + l / 2], whose bit 3 picks the delta; each 16 values (groups
 * 2e, 2e + 1) have a 3-bit scale, and the block's f16 d is spread over the
 * top nibbles of the four scale words. */
template <>
struct Format<29> {
  static constexpr int kBytes = 56, kWords = 14;
  static constexpr bool kAligned = true;
  using Tables = Format<19>::Tables;
  static __device__ void fill(Tables& t, int tid) { Format<19>::fill(t, tid); }
  struct Sub {
    uint32_t qs, qh, sc;
    float d;
  };
  static __device__ __forceinline__ Sub sub(const Raw<kWords>& b, int ib) {
    const uint32_t s01 = b.w[12], s23 = b.w[13];
    const uint32_t d =
        ((s01 >> 12) & 0x000fu) | ((s01 >> 24) & 0x00f0u) | ((s23 >> 4) & 0x0f00u) | ((s23 >> 16) & 0xf000u);
    const uint32_t sc = (b.w[12 + ib / 4] >> (16 * ((ib / 2) % 2))) >> (6 * (ib % 2));
    return {b.w[ib], (b.w[8 + ib / 2] >> (16 * (ib % 2))) & 0xffffu, sc, half_bits(d)};
  }
  static __device__ __forceinline__ void decode(const Sub& s, int l, const Tables& t, uint32_t* out) {
    const uint32_t nibble = (s.qh >> (4 * l)) & 0xfu;
    const uint32_t idx = ((s.qs >> (8 * l)) & 0xffu) | ((nibble & 7u) << 8);
    const float scale = s.d * static_cast<float>(2 * ((s.sc >> (3 * (l / 2))) & 7) + 1);
    const float delta = (nibble & 8u) ? -1.0f - IQ1M_DELTA : -1.0f + IQ1M_DELTA;
    iq1_decode8(t.grid[idx], pack_half2(scale, scale * delta), out);
  }
};

/* IQ2_XXS (66 bytes): d, qs[32] as u16. Sub-block ib holds four codebook
 * bytes (group l's eight magnitudes) and a word of four 7-bit sign indices
 * plus a 4-bit scale: weight = d (1/2 + s) / 4 * +-magnitude. The LDS tables
 * hold each codebook entry as eight FP16 magnitudes and each sign index as
 * its eight FP16 sign bits. */
template <>
struct Format<16> {
  static constexpr int kBytes = 66, kWords = 17;
  static constexpr bool kAligned = false;
  struct Tables {
    uint4 grid[256];
    uint4 signs[128];
  };
  static __device__ void fill(Tables& t, int tid) {
    for (int i = tid; i < 256 + 128; i += 256) {
      uint32_t w[4];
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        if (i < 256) {
          const uint64_t g = iq2xxs_grid[i];
          w[j] = pack_half2(static_cast<float>((g >> (16 * j)) & 0xffu),
                            static_cast<float>((g >> (16 * j + 8)) & 0xffu));
        } else {
          const uint32_t signs = ksigns_iq2xs[i - 256];
          w[j] = ((signs >> (2 * j)) & 1u) << 15 | ((signs >> (2 * j + 1)) & 1u) << 31;
        }
      }
      (i < 256 ? t.grid[i] : t.signs[i - 256]) = make_uint4(w[0], w[1], w[2], w[3]);
    }
  }
  struct Sub {
    uint32_t idx, aux;
    uint32_t scale;  // half2 (s, s)
  };
  static __device__ __forceinline__ Sub sub(const Raw<kWords>& b, int ib) {
    const uint32_t aux = __builtin_amdgcn_alignbyte(b.w[2 * ib + 2], b.w[2 * ib + 1], 2);
    const float scale = half_bits(b.w[0]) * (0.5f + static_cast<float>(aux >> 28)) * 0.25f;
    return {__builtin_amdgcn_alignbyte(b.w[2 * ib + 1], b.w[2 * ib], 2), aux, pack_half2(scale, scale)};
  }
  static __device__ __forceinline__ void decode(const Sub& s, int l, const Tables& t, uint32_t* out) {
    const uint4 g = t.grid[(s.idx >> (8 * l)) & 0xffu];
    const uint4 sg = t.signs[(s.aux >> (7 * l)) & 127u];
    const half2_t scale = __builtin_bit_cast(half2_t, s.scale);
    const uint32_t gw[4] = {g.x ^ sg.x, g.y ^ sg.y, g.z ^ sg.z, g.w ^ sg.w};
#pragma unroll
    for (int i = 0; i < 4; ++i) out[i] = __builtin_bit_cast(uint32_t, __builtin_bit_cast(half2_t, gw[i]) * scale);
  }
};

/* IQ2_XS (74 bytes): d, qs[32] as u16 and eight scale bytes. Each
 * 32-value sub-block has four 9-bit codebook indices and four 7-bit sign
 * indices. The low and high scale nibbles apply to two groups each. */
template <>
struct Format<17> {
  static constexpr int kBytes = 74, kWords = 19;
  static constexpr bool kAligned = false;
  struct Tables {
    uint4 grid[512];
    uint4 signs[128];
  };
  static __device__ void fill(Tables& t, int tid) {
    for (int i = tid; i < 512 + 128; i += 256) {
      uint32_t w[4];
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        if (i < 512) {
          const uint64_t g = iq2xs_grid[i];
          w[j] = pack_half2(static_cast<float>((g >> (16 * j)) & 0xffu),
                            static_cast<float>((g >> (16 * j + 8)) & 0xffu));
        } else {
          const uint32_t signs = ksigns_iq2xs[i - 512];
          w[j] = ((signs >> (2 * j)) & 1u) << 15 | ((signs >> (2 * j + 1)) & 1u) << 31;
        }
      }
      (i < 512 ? t.grid[i] : t.signs[i - 512]) = make_uint4(w[0], w[1], w[2], w[3]);
    }
  }
  struct Sub {
    uint32_t qs[2], scale[2];
  };
  static __device__ __forceinline__ Sub sub(const Raw<kWords>& b, int ib) {
    const uint32_t sc = (b.w[(66 + ib) / 4] >> (8 * ((66 + ib) % 4))) & 0xffu;
    const float d = half_bits(b.w[0]) * 0.25f;
    const float lo = d * (0.5f + static_cast<float>(sc & 15u));
    const float hi = d * (0.5f + static_cast<float>(sc >> 4));
    return {{__builtin_amdgcn_alignbyte(b.w[2 * ib + 1], b.w[2 * ib], 2),
             __builtin_amdgcn_alignbyte(b.w[2 * ib + 2], b.w[2 * ib + 1], 2)},
            {pack_half2(lo, lo), pack_half2(hi, hi)}};
  }
  static __device__ __forceinline__ void decode(const Sub& s, int l, const Tables& t, uint32_t* out) {
    const uint32_t q = (s.qs[l / 2] >> (16 * (l % 2))) & 0xffffu;
    const uint4 g = t.grid[q & 511u];
    const uint4 sg = t.signs[q >> 9];
    const half2_t scale = __builtin_bit_cast(half2_t, s.scale[l / 2]);
    const uint32_t gw[4] = {g.x ^ sg.x, g.y ^ sg.y, g.z ^ sg.z, g.w ^ sg.w};
#pragma unroll
    for (int i = 0; i < 4; ++i) out[i] = __builtin_bit_cast(uint32_t, __builtin_bit_cast(half2_t, gw[i]) * scale);
  }
};

/* Lane L's word from lane L ^ 16, without the destination's old value
 * (fetch-inactive and bound-control set; every lane is active here), so no
 * register is copied for it before each permute. */
__device__ __forceinline__ uint32_t other_half(uint32_t v) {
  return __builtin_amdgcn_permlanex16(v, v, 0x76543210u, 0xfedcba98u, true, true);
}

__device__ __forceinline__ float finite_or_zero(float v) { return __builtin_isfinite(v) ? v : 0.0f; }

template <int kType, bool kPair, bool kHalfOut>
__global__ __launch_bounds__(256) void routed_f16_kernel(
    const uint8_t* __restrict__ w0, const uint8_t* __restrict__ w1,
    const uint8_t* const* __restrict__ w0_table, const uint8_t* const* __restrict__ w1_table, uint64_t expert_bytes,
    const uint16_t* __restrict__ x, uint32_t x_div, const int32_t* __restrict__ rows,
    const int32_t* __restrict__ bounds, const int32_t* __restrict__ tiles, const float* __restrict__ route_w,
    float limit, void* __restrict__ y, uint32_t m, uint32_t k) {
  using F = Format<kType>;
  using Block = Raw<F::kWords>;
  __shared__ typename F::Tables tables;
  __shared__ __attribute__((aligned(16))) _Float16 act[2][kSlots * kActPitch];
  __shared__ int32_t slot_of[kSlots];
  const int tile = tiles[blockIdx.y];
  if (tile < 0) return;  // unused entries in a bounded device-planned grid
  const int expert = tile & 0xffff;
  const int t0 = (tile >> 16) * kSlots;
  const int begin = bounds[expert] + t0;
  const int count = min(kSlots, bounds[expert + 1] - begin);
  const int live = (count + 15) / 16;
  const int tid = threadIdx.x, wave = tid / 32, lane = tid % 32;
  F::fill(tables, tid);
  if (tid < kSlots) slot_of[tid] = tid < count ? rows[begin + tid] : -1;

  constexpr int kWaveRows = kPair ? 16 : 32;
  const uint32_t r_wave = blockIdx.x * 8 * kWaveRows + wave * kWaveRows;
  const uint64_t row_bytes = (uint64_t)k / 256 * F::kBytes;
  const uint8_t* wp[2];
  uint32_t row[2];
#pragma unroll
  for (int p = 0; p < 2; ++p) {
    row[p] = r_wave + (kPair ? 0 : 16 * p) + lane % 16;
    const uint8_t* base;
    if (kPair && p)
      base = w1_table ? w1_table[expert] : w1 + (uint64_t)expert * expert_bytes;
    else
      base = w0_table ? w0_table[expert] : w0 + (uint64_t)expert * expert_bytes;
    wp[p] = base + (uint64_t)(row[p] < m ? row[p] : m - 1) * row_bytes;
  }

  float8 acc[2][4];
#pragma unroll
  for (int p = 0; p < 2; ++p)
#pragma unroll
    for (int t = 0; t < 4; ++t) acc[p][t] = float8{};

  /* Activation staging: a quarter of the stage (32 values) of one slot per
   * thread, consecutive slots in consecutive lanes. Stage `half` of each
   * block lives in buffer `half`; the next stage is loaded before the
   * current one's products and stored after them, so one barrier per stage
   * separates the writes from both buffers' reads. */
  const int st = tid % kSlots, sq = tid / kSlots;
  __syncthreads();
  const int32_t s_slot = slot_of[st];
  const uint16_t* xs = s_slot >= 0 ? x + (uint64_t)(s_slot / (int)x_div) * k + kStageK / 4 * sq : nullptr;
  const int as = st * kActPitch + kStageK / 4 * sq;
  u32x4 next[kVec];
  if (xs) {
#pragma unroll
    for (int c = 0; c < kVec; ++c) reinterpret_cast<u32x4*>(act[0] + as)[c] = reinterpret_cast<const u32x4*>(xs)[c];
  }
  __syncthreads();

  /* Weights: half-wave h loads the blocks of projection h (gate or up, or
   * the second 16 rows) two blocks ahead; the halves swap their words across
   * lanes. Ordinary loads: the 50-byte blocks share cache lines with the
   * next block, which nontemporal loads fetched again from DRAM. */
  const int h = lane / 16;
  const uint8_t* own = h ? wp[1] : wp[0];
  const uint32_t blocks = k / 256;
  Block ring[2];
  ring[0] = load_raw<F::kWords>(own);
  if (blocks > 1) ring[1] = load_raw<F::kWords>(own + F::kBytes);
  /* One copy of the block body (the kernel's code stays within the
   * instruction cache); the ring shifts by register moves. */
  for (uint32_t sb = 0; sb < blocks; ++sb) {
    const Block mine = ring[0];
    ring[0] = ring[1];
    if (sb + 2 < blocks) ring[1] = load_raw<F::kWords>(own + (uint64_t)(sb + 2) * F::kBytes);
    Block blk[2];
#pragma unroll
    for (int i = 0; i < F::kWords; ++i) {
      const uint32_t other = other_half(mine.w[i]);
      blk[0].w[i] = h ? other : mine.w[i];
      blk[1].w[i] = h ? mine.w[i] : other;
    }
    if constexpr (!F::kAligned) {
#pragma unroll
      for (int p = 0; p < 2; ++p)
        blk[p] = realign(blk[p], (reinterpret_cast<uintptr_t>(wp[p]) + sb * F::kBytes) & 3);
    }
#pragma unroll
    for (int stage = 0; stage < kStages; ++stage) {
      const uint32_t k_next = sb * 256 + (stage + 1) * kStageK;
      const bool more = xs && k_next < k;
      if (more) {
#pragma unroll
        for (int c = 0; c < kVec; ++c) next[c] = reinterpret_cast<const u32x4*>(xs + k_next)[c];
      }
      const _Float16* cur = act[stage % 2];
#pragma unroll
      for (int jj = 0; jj < kStageK / 32; ++jj) {
        const int ib = stage * (kStageK / 32) + jj;
        typename F::Sub sub[2];
#pragma unroll
        for (int p = 0; p < 2; ++p) sub[p] = F::sub(blk[p], ib);
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
            F::decode(sub[p], 2 * e2 + h, tables, words);
#pragma unroll
            for (int i = 0; i < 4; ++i)
              words[4 + i] = other_half(words[i]);
            b[p] = __builtin_bit_cast(half16, words);
          }
          const int kk = jj * 32 + e2 * 16;
#pragma unroll
          for (int t = 0; t < 4; ++t) {
            if (t < live) {
              const _Float16* ar = cur + (t * 16 + lane % 16) * kActPitch + kk;
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
      if (more) {
#pragma unroll
        for (int c = 0; c < kVec; ++c) reinterpret_cast<u32x4*>(act[1 - stage % 2] + as)[c] = next[c];
      }
      __syncthreads();
    }
  }

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

template <int kType>
void launch_routed(const uint8_t* w, const uint8_t* w_up, const uint8_t* const* w_table,
                   const uint8_t* const* w_up_table, uint64_t expert_bytes, const uint16_t* x, uint32_t x_div,
                   const int32_t* rows, const int32_t* bounds, const int32_t* tiles, uint32_t n_tiles,
                   const float* route_w, float limit, void* y, int y_half, uint32_t m, uint32_t k) {
  if (w_up && y_half)
    routed_f16_kernel<kType, true, true><<<dim3((m + 127) / 128, n_tiles), 256>>>(
        w, w_up, w_table, w_up_table, expert_bytes, x, x_div, rows, bounds, tiles, route_w, limit, y, m, k);
  else if (w_up)
    routed_f16_kernel<kType, true, false><<<dim3((m + 127) / 128, n_tiles), 256>>>(
        w, w_up, w_table, w_up_table, expert_bytes, x, x_div, rows, bounds, tiles, route_w, limit, y, m, k);
  else
    routed_f16_kernel<kType, false, false><<<dim3((m + 255) / 256, n_tiles), 256>>>(
        w, nullptr, w_table, w_up_table, expert_bytes, x, x_div, rows, bounds, tiles, nullptr, 0.0f, y, m, k);
}

}  // namespace

extern "C" int ds41_v41_routed_f16(int type, const void* w, const void* w_up, uint64_t expert_bytes, const uint16_t* x,
                                   uint32_t x_div, const int32_t* rows, const int32_t* bounds, const int32_t* tiles,
                                   uint32_t n_tiles, const float* route_w, float limit, void* y, int y_half,
                                   uint32_t m, uint32_t k, const void* const* w_table,
                                   const void* const* w_up_table) {
  if (k % 256 != 0 || n_tiles == 0 || x_div == 0) return 1;
  const auto* a = static_cast<const uint8_t*>(w);
  const auto* b = static_cast<const uint8_t*>(w_up);
  const auto* ta = reinterpret_cast<const uint8_t* const*>(w_table);
  const auto* tb = reinterpret_cast<const uint8_t* const*>(w_up_table);
  switch (type) {
  case 19:
    launch_routed<19>(a, b, ta, tb, expert_bytes, x, x_div, rows, bounds, tiles, n_tiles, route_w, limit, y, y_half,
                      m, k);
    break;
  case 29:
    launch_routed<29>(a, b, ta, tb, expert_bytes, x, x_div, rows, bounds, tiles, n_tiles, route_w, limit, y, y_half,
                      m, k);
    break;
  case 16:
    launch_routed<16>(a, b, ta, tb, expert_bytes, x, x_div, rows, bounds, tiles, n_tiles, route_w, limit, y, y_half,
                      m, k);
    break;
  case 17:
    launch_routed<17>(a, b, ta, tb, expert_bytes, x, x_div, rows, bounds, tiles, n_tiles, route_w, limit, y, y_half,
                      m, k);
    break;
  default:
    return 1;
  }
  return hipGetLastError() == hipSuccess ? 0 : 1;
}

namespace {

// Count expert buckets without allocating or downloading route IDs.
__global__ void routed_counts_kernel(const int32_t* ids, uint32_t slots, int32_t* bounds) {
  const int expert = (int)blockIdx.x;
  const uint32_t tid = threadIdx.x;
  uint32_t count = 0;
  for (uint32_t i = tid; i < slots; i += 256) count += ids[i] == expert;
  for (int offset = 16; offset > 0; offset /= 2) count += __shfl_down(count, offset, 32);
  __shared__ uint32_t waves[8];
  if ((tid & 31) == 0) waves[tid / 32] = count;
  __syncthreads();
  if (tid == 0) {
    count = 0;
    for (int w = 0; w < 8; w++) count += waves[w];
    bounds[expert + 1] = (int32_t)count;
  }
}

// One thread per expert; preserve the host plan's expert/64-slot tile order.
__global__ void routed_bounds_kernel(int32_t* bounds, int32_t* tiles, uint32_t experts, uint32_t tile_cap) {
  __shared__ uint32_t counts[512];
  const uint32_t tid = threadIdx.x;
  counts[tid] = tid < experts ? (uint32_t)bounds[tid + 1] : 0;
  for (uint32_t i = tid; i < tile_cap; i += 512) tiles[i] = -1;
  __syncthreads();
  if (tid >= experts) return;
  uint32_t first = 0, tile = 0;
  for (uint32_t e = 0; e < tid; e++) {
    first += counts[e];
    tile += (counts[e] + kSlots - 1) / kSlots;
  }
  bounds[tid] = (int32_t)first;
  if (tid + 1 == experts) bounds[experts] = (int32_t)(first + counts[tid]);
  for (uint32_t t = 0; t * kSlots < counts[tid]; t++) tiles[tile + t] = (int32_t)(tid | (t << 16));
}

// Contiguous slot ranges per lane, with a block prefix of their counts,
// retain ascending slot order within every expert (no unordered atomic append).
__global__ void routed_rows_kernel(const int32_t* ids, uint32_t slots, const int32_t* bounds, int32_t* rows) {
  const int expert = (int)blockIdx.x;
  const uint32_t tid = threadIdx.x, lane = tid & 31, wave = tid / 32;
  const uint32_t per = (slots + 255) / 256;
  const uint32_t begin = min(tid * per, slots), end = min(begin + per, slots);
  uint32_t count = 0;
  for (uint32_t i = begin; i < end; i++) count += ids[i] == expert;
  uint32_t prefix = count;
  for (int offset = 1; offset < 32; offset *= 2) {
    const uint32_t other = __shfl_up(prefix, offset, 32);
    if (lane >= (uint32_t)offset) prefix += other;
  }
  __shared__ uint32_t waves[8];
  if (lane == 31) waves[wave] = prefix;
  __syncthreads();
  uint32_t cursor = (uint32_t)bounds[expert] + prefix - count;
  for (uint32_t w = 0; w < wave; w++) cursor += waves[w];
  for (uint32_t i = begin; i < end; i++)
    if (ids[i] == expert) rows[cursor++] = (int32_t)i;
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

}  // namespace

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

extern "C" int ds41_v41_routed_plan_device(const int32_t* ids, uint32_t slots, uint32_t n_experts, int32_t* rows,
                                           int32_t* bounds, int32_t* tiles) {
  if (slots == 0 || n_experts == 0 || n_experts > 512) return 1;
  routed_counts_kernel<<<n_experts, 256>>>(ids, slots, bounds);
  routed_bounds_kernel<<<1, 512>>>(bounds, tiles, n_experts, slots / kSlots + n_experts);
  routed_rows_kernel<<<n_experts, 256>>>(ids, slots, bounds, rows);
  return hipGetLastError() == hipSuccess ? 0 : 1;
}
