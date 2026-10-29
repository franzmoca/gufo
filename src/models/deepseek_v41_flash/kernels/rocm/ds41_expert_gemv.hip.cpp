/* Single-token routed-expert projections for the V4.1 Flash IQ1_S, IQ1_M,
 * IQ2_XXS, IQ2_XS and IQ2_S weights with FP32 activations. Each slot multiplies its
 * own expert, ids[slot]; a wave covers 4 rows and a lane takes one 32-value
 * sub-block (task t) of each 256-value block, summed over the 8 tasks at the
 * end.
 *
 * IQ1 weights are (n - 1 + delta) * d * s with a grid nibble n in {0, 1, 2},
 * so a group of eight is d s (sum n_i x_i + (delta - 1) sum x_i): one convert
 * and one FMA per weight instead of decoding each weight to a float first.
 * The 8 KiB IQ1 codebook is copied to LDS, where the per-lane lookups do not
 * queue on the texture path behind the weight and activation loads; the IQ2
 * grids and sign tables are staged the same way, as u32 word pairs.
 *
 * Each task's payload is loaded as whole words (memcpy from the 2-byte
 * aligned block) instead of per-16-bit loads, and the IQ2 decode keeps each
 * grid entry in its u32 halves (byte extract + convert; casting the u64
 * entry emits a quarter-rate ldexp) with signs applied by flipping the
 * float sign bit. The fused gate/up pass loads x once per block into
 * registers and feeds both accumulations, halving activation loads. With
 * these the kernels decode at 194-200 GB/s, the dense GEMV's level. */
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

/* Iq1Dot over a register-resident slice of x (same FMA chain, scalar reads). */
__device__ __forceinline__ float Iq1DotR(uint32_t grid, float c, const float (&xr)[32], int base) {
#pragma clang fp reassociate(off)
  const uint32_t lo = grid & 0x0F0F0F0F;
  const uint32_t hi = (grid >> 4) & 0x0F0F0F0F;
  float dot = static_cast<float>(lo & 0xFF) * xr[base];
  dot = __builtin_fmaf(static_cast<float>((lo >> 8) & 0xFF), xr[base + 1], dot);
  dot = __builtin_fmaf(static_cast<float>((lo >> 16) & 0xFF), xr[base + 2], dot);
  dot = __builtin_fmaf(static_cast<float>(lo >> 24), xr[base + 3], dot);
  dot = __builtin_fmaf(static_cast<float>(hi & 0xFF), xr[base + 4], dot);
  dot = __builtin_fmaf(static_cast<float>((hi >> 8) & 0xFF), xr[base + 5], dot);
  dot = __builtin_fmaf(static_cast<float>((hi >> 16) & 0xFF), xr[base + 6], dot);
  dot = __builtin_fmaf(static_cast<float>(hi >> 24), xr[base + 7], dot);
  const float sum = ((xr[base] + xr[base + 1]) + (xr[base + 2] + xr[base + 3])) +
                    ((xr[base + 4] + xr[base + 5]) + (xr[base + 6] + xr[base + 7]));
  return __builtin_fmaf(c, sum, dot);
}

/* Accumulate<32> over a register-resident x slice (same FMA chain). */
__device__ __forceinline__ float AccumulateR(float acc, const float (&w)[32], const float (&xr)[32]) {
#pragma unroll
  for (int i = 0; i < 32; i += 4) {
    acc = __builtin_fmaf(w[i], xr[i], acc);
    acc = __builtin_fmaf(w[i + 1], xr[i + 1], acc);
    acc = __builtin_fmaf(w[i + 2], xr[i + 2], acc);
    acc = __builtin_fmaf(w[i + 3], xr[i + 3], acc);
  }
  return acc;
}

/* Eight grid magnitudes (one per byte) with the sign bits of `signs`. The
 * entry arrives as its u32 halves (never cast the u64 entry to float: the
 * conversion lowers to a quarter-rate ldexp); -v is the float with its sign
 * bit flipped, one xor instead of a compare and select. */
__device__ __forceinline__ void Iq2Group(uint32_t glo, uint32_t ghi, uint32_t signs, float scale, float* w) {
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const uint32_t b = (((i < 4) ? glo : ghi) >> (8 * (i & 3))) & 0xFFu;
    const float v = static_cast<float>(b) * scale;
    w[i] = __uint_as_float(__float_as_uint(v) ^ (((signs >> i) & 1u) << 31));
  }
}

/* The IQ2 grids staged in LDS as lo/hi u32 word pairs, with the signs. */
struct Iq2Tab {
  const uint32_t* words;
  const uint8_t* signs;
};

template <int kType>
struct Format;

/* IQ1_S (50 bytes): d, qs[32], qh[8] as u16; sub-block t uses qs[4 t..],
 * qh[t] (3 high index bits per group, 3-bit scale, delta sign). The task's
 * qs arrives as one word, the 8-byte scale tail as one u64. */
template <>
struct Format<19> {
  static constexpr int kBlockBytes = 50;
  static constexpr bool kIq1Grid = true;
  static constexpr int kIq2Words = 0;
  static constexpr bool kXReg = true;
  struct Raw {
    uint32_t d, qs, qh;
  };
  static __device__ __forceinline__ Raw Load(const uint8_t* b, int t) {
    Raw r{};
    __builtin_memcpy(&r.d, b, 2);
    __builtin_memcpy(&r.qs, b + 2 + 4 * t, 4);
    r.qh = reinterpret_cast<const uint16_t*>(b)[17 + t];
    return r;
  }
  /* acc + scale * dot as one explicit FMA, so no kernel's contraction choice
   * changes the result. */
  static __device__ __forceinline__ float Acc(float acc, const Raw& a, int, const float* x, const uint32_t* grid,
                                              const Iq2Tab&) {
#pragma clang fp reassociate(off)
    const float d = HalfBits(a.d);
    const uint32_t qs = a.qs;
    const uint32_t qh = a.qh;
    const float scale = d * static_cast<float>(2 * ((qh >> 12) & 7) + 1);
    const float c = (qh & 0x8000 ? -IQ1S_DELTA : IQ1S_DELTA) - 1.0f;
    float dot = 0.0f;
#pragma unroll
    for (int l = 0; l < 4; ++l)
      dot += Iq1Dot(grid[((qs >> (8 * l)) & 0xFF) | (((qh >> (3 * l)) & 7) << 8)], c, x + 8 * l);
    return __builtin_fmaf(scale, dot, acc);
  }
  static __device__ __forceinline__ float AccR(float acc, const Raw& a, int, const float (&xr)[32],
                                               const uint32_t* grid, const Iq2Tab&) {
#pragma clang fp reassociate(off)
    const float d = HalfBits(a.d);
    const uint32_t qs = a.qs;
    const uint32_t qh = a.qh;
    const float scale = d * static_cast<float>(2 * ((qh >> 12) & 7) + 1);
    const float c = (qh & 0x8000 ? -IQ1S_DELTA : IQ1S_DELTA) - 1.0f;
    float dot = 0.0f;
#pragma unroll
    for (int l = 0; l < 4; ++l)
      dot += Iq1DotR(grid[((qs >> (8 * l)) & 0xFF) | (((qh >> (3 * l)) & 7) << 8)], c, xr, 8 * l);
    return __builtin_fmaf(scale, dot, acc);
  }
};

/* IQ1_M (56 bytes): qs[32], qh[16], four u16 scale words whose top nibbles
 * hold d. Groups 0-1 of sub-block t use its low 3-bit scale, 2-3 the high.
 * The task's qs arrives as one word and the scale tail as one u64. */
template <>
struct Format<29> {
  static constexpr int kBlockBytes = 56;
  static constexpr bool kIq1Grid = true;
  static constexpr int kIq2Words = 0;
  static constexpr bool kXReg = false;
  struct Raw {
    uint32_t qs, qh, s0, s1;
  };
  static __device__ __forceinline__ Raw Load(const uint8_t* b, int t) {
    Raw r{};
    __builtin_memcpy(&r.qs, b + 4 * t, 4);
    r.qh = reinterpret_cast<const uint16_t*>(b)[16 + t];
    uint64_t s;
    __builtin_memcpy(&s, b + 48, 8);
    r.s0 = static_cast<uint32_t>(s);
    r.s1 = static_cast<uint32_t>(s >> 32);
    return r;
  }
  static __device__ __forceinline__ float Acc(float acc, const Raw& a, int t, const float* x, const uint32_t* grid,
                                              const Iq2Tab&) {
    const uint32_t qs = a.qs;
    const uint32_t sc0 = a.s0 & 0xFFFF, sc1 = a.s0 >> 16, sc2 = a.s1 & 0xFFFF, sc3 = a.s1 >> 16;
    const float d = HalfBits((sc0 >> 12) | ((sc1 >> 8) & 0x00F0) | ((sc2 >> 4) & 0x0F00) | (sc3 & 0xF000));
    const uint32_t sw[4] = {sc0, sc1, sc2, sc3};
    const uint32_t word = sw[t >> 1] >> (6 * (t & 1));
    const float s1 = d * static_cast<float>(2 * (word & 7) + 1);
    const float s2 = d * static_cast<float>(2 * ((word >> 3) & 7) + 1);
    const uint32_t qh0 = a.qh & 0xFF, qh1 = a.qh >> 8;
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
 * four 7-bit sign indices and a 4-bit scale; the task's eight bytes arrive
 * as one word pair. */
template <>
struct Format<16> {
  static constexpr int kBlockBytes = 66;
  static constexpr bool kIq1Grid = false;
  static constexpr int kIq2Words = 512;
  static constexpr bool kXReg = true;
  struct Raw {
    uint32_t d, qs, aux;
  };
  static __device__ __forceinline__ Raw Load(const uint8_t* b, int t) {
    Raw r{};
    __builtin_memcpy(&r.d, b, 2);
    uint64_t v;
    __builtin_memcpy(&v, b + 2 + 8 * t, 8);
    r.qs = static_cast<uint32_t>(v);
    r.aux = static_cast<uint32_t>(v >> 32);
    return r;
  }
  template <bool kReg>
  static __device__ __forceinline__ float AccT(float acc, const Raw& a, int, const float* x, const float (&xr)[32],
                                               const uint32_t*, const Iq2Tab& tab) {
    const float d = HalfBits(a.d);
    const uint32_t qs = a.qs, aux = a.aux;
    const float scale = d * (0.5f + static_cast<float>(aux >> 28)) * 0.25f;
    float w[32];
#pragma unroll
    for (int l = 0; l < 4; ++l) {
      const uint32_t idx = (qs >> (8 * l)) & 0xFF;
      Iq2Group(tab.words[2 * idx], tab.words[2 * idx + 1], tab.signs[(aux >> (7 * l)) & 127], scale, w + 8 * l);
    }
    if constexpr (kReg)
      return acc + AccumulateR(0.0f, w, xr);
    return acc + Accumulate<32>(0.0f, w, x);
  }
  static __device__ __forceinline__ float Acc(float acc, const Raw& a, int t, const float* x, const uint32_t* g,
                                              const Iq2Tab& tab) {
    float xr[32] = {};
    return AccT<false>(acc, a, t, x, xr, g, tab);
  }
  static __device__ __forceinline__ float AccR(float acc, const Raw& a, int t, const float (&xr)[32],
                                               const uint32_t* g, const Iq2Tab& tab) {
    return AccT<true>(acc, a, t, nullptr, xr, g, tab);
  }
};

/* IQ2_XS (74 bytes): d, 32 grid/sign words, eight pairs of scales; the
 * task's four q words arrive as one word pair. */
template <>
struct Format<17> {
  static constexpr int kBlockBytes = 74;
  static constexpr bool kIq1Grid = false;
  static constexpr int kIq2Words = 1024;
  static constexpr bool kXReg = true;
  struct Raw {
    uint32_t d, s0, s1, sc;
  };
  static __device__ __forceinline__ Raw Load(const uint8_t* b, int t) {
    Raw r{};
    __builtin_memcpy(&r.d, b, 2);
    __builtin_memcpy(&r.sc, b + 66 + 2 * (t >> 1), 2);
    uint64_t v;
    __builtin_memcpy(&v, b + 2 + 8 * t, 8);
    r.s0 = static_cast<uint32_t>(v);
    r.s1 = static_cast<uint32_t>(v >> 32);
    return r;
  }
  template <bool kReg>
  static __device__ __forceinline__ float AccT(float acc, const Raw& a, int t, const float* x, const float (&xr)[32],
                                               const uint32_t*, const Iq2Tab& tab) {
    const float d = HalfBits(a.d);
    const uint32_t sc = (a.sc >> (8 * (t & 1))) & 0xFF;
    const float low = d * (0.5f + static_cast<float>(sc & 0xF)) * 0.25f;
    const float high = d * (0.5f + static_cast<float>(sc >> 4)) * 0.25f;
    float w[32];
#pragma unroll
    for (int l = 0; l < 4; ++l) {
      const uint32_t q = ((l < 2 ? a.s0 : a.s1) >> (16 * (l & 1))) & 0xFFFF;
      Iq2Group(tab.words[2 * (q & 0x1FF)], tab.words[2 * (q & 0x1FF) + 1], tab.signs[q >> 9], l < 2 ? low : high,
               w + 8 * l);
    }
    if constexpr (kReg)
      return acc + AccumulateR(0.0f, w, xr);
    return acc + Accumulate<32>(0.0f, w, x);
  }
  static __device__ __forceinline__ float Acc(float acc, const Raw& a, int t, const float* x, const uint32_t* g,
                                              const Iq2Tab& tab) {
    float xr[32] = {};
    return AccT<false>(acc, a, t, x, xr, g, tab);
  }
  static __device__ __forceinline__ float AccR(float acc, const Raw& a, int t, const float (&xr)[32],
                                               const uint32_t* g, const Iq2Tab& tab) {
    return AccT<true>(acc, a, t, nullptr, xr, g, tab);
  }
};

/* IQ2_S (82 bytes): d, qs[32], signs[32], qh[8], scales[8]; the task's qs
 * and signs arrive as one word each, qh and the scale pair as u16s. */
template <>
struct Format<22> {
  static constexpr int kBlockBytes = 82;
  static constexpr bool kIq1Grid = false;
  static constexpr int kIq2Words = 2048;
  static constexpr bool kXReg = true;
  struct Raw {
    uint32_t d, qs, signs, qh, sc;
  };
  static __device__ __forceinline__ Raw Load(const uint8_t* b, int t) {
    Raw r{};
    __builtin_memcpy(&r.d, b, 2);
    __builtin_memcpy(&r.qs, b + 2 + 4 * t, 4);
    __builtin_memcpy(&r.signs, b + 34 + 4 * t, 4);
    __builtin_memcpy(&r.qh, b + 66 + 2 * (t >> 1), 2);
    __builtin_memcpy(&r.sc, b + 74 + 2 * (t >> 1), 2);
    return r;
  }
  template <bool kReg>
  static __device__ __forceinline__ float AccT(float acc, const Raw& a, int t, const float* x, const float (&xr)[32],
                                               const uint32_t*, const Iq2Tab& tab) {
    const float d = HalfBits(a.d);
    const uint32_t qh = (a.qh >> (8 * (t & 1))) & 0xFF;
    const uint32_t sc = (a.sc >> (8 * (t & 1))) & 0xFF;
    const float low = d * (0.5f + static_cast<float>(sc & 0xF)) * 0.25f;
    const float high = d * (0.5f + static_cast<float>(sc >> 4)) * 0.25f;
    float w[32];
#pragma unroll
    for (int l = 0; l < 4; ++l) {
      const uint32_t idx = ((a.qs >> (8 * l)) & 0xFF) | ((qh << (8 - 2 * l)) & 0x300);
      Iq2Group(tab.words[2 * idx], tab.words[2 * idx + 1], (a.signs >> (8 * l)) & 0xFF, l < 2 ? low : high, w + 8 * l);
    }
    if constexpr (kReg)
      return acc + AccumulateR(0.0f, w, xr);
    return acc + Accumulate<32>(0.0f, w, x);
  }
  static __device__ __forceinline__ float Acc(float acc, const Raw& a, int t, const float* x, const uint32_t* g,
                                              const Iq2Tab& tab) {
    float xr[32] = {};
    return AccT<false>(acc, a, t, x, xr, g, tab);
  }
  static __device__ __forceinline__ float AccR(float acc, const Raw& a, int t, const float (&xr)[32],
                                               const uint32_t* g, const Iq2Tab& tab) {
    return AccT<true>(acc, a, t, nullptr, xr, g, tab);
  }
};

/* Q4_K (144 bytes, 16-byte aligned blocks): the dense GEMV's task split
 * (DecodeQ45K/DotQ45K), one header and one 16-byte payload load per task.
 * The task's 32 values live in two 16-value slices 32 apart, so the fused
 * gate/up pass keeps the plain two-read accumulation. Serves the DSpark
 * drafter's routed experts (FP32 activations against mmvq's Q8_1). */
template <>
struct Format<12> {
  static constexpr int kBlockBytes = 144;
  static constexpr bool kIq1Grid = false;
  static constexpr int kIq2Words = 0;
  static constexpr bool kXReg = false;
  struct Raw {
    uint4 header, q;
  };
  static __device__ __forceinline__ Raw Load(const uint8_t* b, int t) {
    const int p = t >> 1, h = t & 1;
    return {*reinterpret_cast<const uint4*>(b), *reinterpret_cast<const uint4*>(b + 16 + 32 * p + 16 * h)};
  }
  static __device__ __forceinline__ float Acc(float acc, const Raw& a, int t, const float* x, const uint32_t*,
                                              const Iq2Tab&) {
    float w[32];
    ds41_gemv_tasks::DecodeQ45K<false>({a.header, a.q, {}}, t, w);
    /* The kernel hands the task's 32-float slice; DotQ45K offsets by the
     * task itself, so pass the block base. */
    return acc + ds41_gemv_tasks::DotQ45K(0.0f, w, t, x - 32 * t);
  }
  static __device__ __forceinline__ float AccR(float acc, const Raw& a, int t, const float (&xr)[32],
                                               const uint32_t* grid, const Iq2Tab& tab) {
    return Acc(acc, a, t, xr, grid, tab);
  }
};

/* Q2_K (84 bytes): d, dmin, 16 scale/min bytes (one per 16-value sub-block)
 * and 64 qs bytes (two bits per value). Task t holds sub-blocks 2t and 2t+1
 * contiguously: one scale word and one 8-byte payload load. Weight =
 * d sc v - dmin mn with v in {0..3}. */
template <>
struct Format<10> {
  static constexpr int kBlockBytes = 84;
  static constexpr bool kIq1Grid = false;
  static constexpr int kIq2Words = 0;
  static constexpr bool kXReg = false;
  struct Raw {
    uint32_t dm, sc;
    uint32_t qs[8];
  };
  static __device__ __forceinline__ Raw Load(const uint8_t* b, int t) {
    Raw r{};
    __builtin_memcpy(&r.dm, b + 80, 4);
    __builtin_memcpy(&r.sc, b + 2 * t, 2);
    __builtin_memcpy(r.qs, b + 16 + 32 * (t >> 2), 32);
    return r;
  }
  static __device__ __forceinline__ float Acc(float acc, const Raw& a, int t, const float* x,
                                              const uint32_t*, const Iq2Tab&) {
    const float d = HalfBits(a.dm & 0xFFFFu), dmin = HalfBits(a.dm >> 16);
    const uint32_t shift = 2u * (t & 3);
    float w[32];
#pragma unroll
    for (int i = 0; i < 32; ++i) {
      const uint32_t byte = (a.qs[i >> 2] >> (8 * (i & 3))) & 0xFFu;
      const uint32_t sb = (a.sc >> (8 * (i >> 4))) & 0xFFu;
      const float dsc = d * static_cast<float>(sb & 0xFu);
      const float off = dmin * static_cast<float>(sb >> 4);
      w[i] = dsc * static_cast<float>((byte >> shift) & 3u) - off;
    }
    return acc + Accumulate<32>(0.0f, w, x);
  }
  static __device__ __forceinline__ float AccR(float acc, const Raw& a, int t, const float (&xr)[32],
                                               const uint32_t* grid, const Iq2Tab& tab) {
    return Acc(acc, a, t, xr, grid, tab);
  }
};

/* The global u64 grid table of one IQ2 format, staged to LDS by the kernels. */
template <int kType>
__device__ __forceinline__ const uint64_t* Iq2GridTable() {
  if constexpr (kType == 16)
    return iq2xxs_grid;
  else if constexpr (kType == 17)
    return iq2xs_grid;
  else
    return iq2s_grid;
}

/* Grid (row tiles of 32, slots). The raw words of U blocks are loaded before
 * any of them is decoded. */
/* Row sums of one expert matrix; with kSwiGLU, a wave takes the same rows
 * of the gate (w) and up (w_up) matrices and stores
 * route_w[slot] * silu(min(gate, limit)) * clamp(up, +-limit). */
template <int kType, int U, bool kSwiGLU, bool kTable, bool kPair = false>
__global__ void __launch_bounds__(kWave* kWaves)
    expert_gemv_kernel(const uint8_t* __restrict__ w, const uint8_t* __restrict__ w_up,
                       const uint8_t* const* __restrict__ w_table, const uint8_t* const* __restrict__ w_up_table,
                       const float* __restrict__ route_w, float limit, const float* __restrict__ x,
                       float* __restrict__ y, uint32_t m, uint32_t k, const int32_t* __restrict__ ids,
                       uint64_t expert_bytes, uint32_t x_slot, uint32_t slots_per_x, int zero_dropped) {
  using F = Format<kType>;
  /* kPair: a lane group takes rows r and r + 4 of one matrix through the
   * fused pass's second stream, so x loads once for both. */
  constexpr bool kTwo = kSwiGLU || kPair;
  constexpr uint32_t kRows = (kPair ? 8 : 4) * kWaves;
  __shared__ uint32_t grid[F::kIq1Grid ? NGRID_IQ1S : 1];
  __shared__ uint32_t iq2_words[F::kIq2Words ? F::kIq2Words : 1];
  __shared__ uint8_t iq2_signs[F::kIq2Words ? 128 : 1];
  const int32_t id = ids[blockIdx.y];
  y += (uint64_t)blockIdx.y * m;
  if (id < 0) {  // dropped slot
    if (zero_dropped) {
      for (uint32_t r = threadIdx.x; r < kRows; r += blockDim.x)
        if (blockIdx.x * kRows + r < m) y[blockIdx.x * kRows + r] = 0.0f;
    }
    return;
  }
  if constexpr (F::kIq1Grid) {
    for (uint32_t i = threadIdx.x; i < NGRID_IQ1S; i += blockDim.x) grid[i] = iq1s_grid_gpu[i];
    __syncthreads();
  }
  if constexpr (F::kIq2Words) {
    for (uint32_t i = threadIdx.x; i < F::kIq2Words / 2; i += blockDim.x) {
      const uint64_t g = Iq2GridTable<kType>()[i];
      iq2_words[2 * i] = static_cast<uint32_t>(g);
      iq2_words[2 * i + 1] = static_cast<uint32_t>(g >> 32);
    }
    for (uint32_t i = threadIdx.x; i < 128; i += blockDim.x) iq2_signs[i] = ksigns_iq2xs[i];
    __syncthreads();
  }
  const Iq2Tab tab{iq2_words, iq2_signs};
  const int lane = threadIdx.x % kWave;
  const int wave = threadIdx.x / kWave;
  const int t = lane % 8;
  const uint32_t row = blockIdx.x * kRows + wave * (kPair ? 8 : 4) + lane / 8;
  if (row >= m) return;
  const float* xt = x + (uint64_t)(blockIdx.y / slots_per_x) * x_slot + 32 * t;
  const uint32_t blocks_per_row = k / 256;
  /* The resident layout keeps its original address arithmetic; a runtime
   * choice here rescheduled the IQ2 loops with dependency delays. */
  const uint64_t row_bytes = (uint64_t)row * blocks_per_row * F::kBlockBytes;
  const uint64_t offset = (uint64_t)id * expert_bytes + row_bytes;
  const uint8_t* base = kTable ? w_table[id] + row_bytes : w + offset;
  const uint8_t* base_up = kSwiGLU ? (kTable ? w_up_table[id] + row_bytes : w_up + offset)
                           : kPair  ? base + (uint64_t)4 * blocks_per_row * F::kBlockBytes
                                    : nullptr;
  float acc = 0.0f;
  float acc_up = 0.0f;
  uint32_t blk = 0;
  for (; blk + U <= blocks_per_row; blk += U) {
    typename F::Raw raw[U];
    typename F::Raw raw_up[kTwo ? U : 1];
#pragma unroll
    for (int u = 0; u < U; ++u) {
      raw[u] = F::Load(base + (uint64_t)(blk + u) * F::kBlockBytes, t);
      if constexpr (kTwo) raw_up[u] = F::Load(base_up + (uint64_t)(blk + u) * F::kBlockBytes, t);
    }
#pragma unroll
    for (int u = 0; u < U; ++u) {
      const float* xb = xt + (blk + u) * 256;
      if constexpr (kTwo && F::kXReg) {
        /* x loads once per block, feeding the gate and up accumulations. */
        float xr[32];
#pragma unroll
        for (int i = 0; i < 32; i += 4) {
          const float4 f = *reinterpret_cast<const float4*>(xb + i);
          xr[i] = f.x;
          xr[i + 1] = f.y;
          xr[i + 2] = f.z;
          xr[i + 3] = f.w;
        }
        acc = F::AccR(acc, raw[u], t, xr, grid, tab);
        acc_up = F::AccR(acc_up, raw_up[u], t, xr, grid, tab);
      } else {
        acc = F::Acc(acc, raw[u], t, xb, grid, tab);
        if constexpr (kTwo) acc_up = F::Acc(acc_up, raw_up[u], t, xb, grid, tab);
      }
    }
  }
  for (; blk < blocks_per_row; ++blk) {
    const typename F::Raw raw = F::Load(base + (uint64_t)blk * F::kBlockBytes, t);
    typename F::Raw raw_up{};
    if constexpr (kTwo) raw_up = F::Load(base_up + (uint64_t)blk * F::kBlockBytes, t);
    const float* xb = xt + blk * 256;
    if constexpr (kTwo && F::kXReg) {
      float xr[32];
#pragma unroll
      for (int i = 0; i < 32; i += 4) {
        const float4 f = *reinterpret_cast<const float4*>(xb + i);
        xr[i] = f.x;
        xr[i + 1] = f.y;
        xr[i + 2] = f.z;
        xr[i + 3] = f.w;
      }
      acc = F::AccR(acc, raw, t, xr, grid, tab);
      acc_up = F::AccR(acc_up, raw_up, t, xr, grid, tab);
    } else {
      acc = F::Acc(acc, raw, t, xb, grid, tab);
      if constexpr (kTwo) acc_up = F::Acc(acc_up, raw_up, t, xb, grid, tab);
    }
  }
#pragma unroll
  for (int offset = 4; offset > 0; offset >>= 1) {
    acc += __shfl_xor(acc, offset, kWave);
    if constexpr (kTwo) acc_up += __shfl_xor(acc_up, offset, kWave);
  }
  if (t != 0) return;
  if constexpr (kPair) {
    y[row] = acc;
    y[row + 4] = acc_up;
  } else if constexpr (kSwiGLU) {
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

/* ds41_v41_expert_plan: one block of kPlanSlots threads groups the slots of
 * `ids` by expert. plan[0] is the group count; group g, at plan + 1 + g * kPlanGroup,
 * holds the expert id, the slot count and the slots in ascending order. */
constexpr int kPlanCols = DS41_V41_EXPERT_PLAN_COLS;
constexpr int kPlanGroup = 2 + kPlanCols;
constexpr int kPlanSlots = DS41_V41_EXPERT_PLAN_SLOTS;

__global__ void expert_plan_kernel(const int32_t* __restrict__ ids, uint32_t slots, int32_t* __restrict__ plan) {
  __shared__ int32_t sid[kPlanSlots];
  __shared__ int32_t first_of[kPlanSlots];  // the first slot holding each slot's expert
  __shared__ int32_t group_of[kPlanSlots];  // the group of each leading slot
  const uint32_t s = threadIdx.x;
  if (s < slots) sid[s] = ids[s];
  __syncthreads();
  int32_t id = 0, col = 0, count = 0, first = (int32_t)s;
  if (s < slots) {
    id = sid[s];
    for (uint32_t o = 0; o < slots; ++o) {
      const bool same = sid[o] == id;
      count += same;
      if (o < s && same) {
        ++col;
        if ((int32_t)o < first) first = (int32_t)o;
      }
    }
    first_of[s] = first;
  }
  __syncthreads();
  /* Groups are numbered by their first slot: a leader's group counts the
   * leaders before it. */
  if (s < slots && first == (int32_t)s) {
    int32_t group = 0;
    for (uint32_t o = 0; o < s; ++o) group += first_of[o] == (int32_t)o;
    group_of[s] = group;
  }
  __syncthreads();
  if (s >= slots) return;
  int32_t* g = plan + 1 + group_of[first] * kPlanGroup;
  if (first == (int32_t)s) {
    g[0] = id;
    g[1] = count;
  }
  if (col < kPlanCols) g[2 + col] = (int32_t)s;
  if (s == slots - 1) {
    int32_t groups = 0;
    for (uint32_t o = 0; o < slots; ++o) groups += first_of[o] == (int32_t)o;
    plan[0] = groups;
  }
}

/* expert_gemv_kernel over the groups of a plan: block (row tile, group)
 * decodes its expert's weights once for the group's up to kCols slots, slot
 * s reading x + (s / slots_per_x) * x_slot and writing row s of y. Every
 * slot sees the arithmetic of expert_gemv_kernel. */
template <int kType, int U, bool kSwiGLU, int kCols, bool kTable>
__global__ void __launch_bounds__(kWave* kWaves)
    expert_group_kernel(const uint8_t* __restrict__ w, const uint8_t* __restrict__ w_up,
                        const uint8_t* const* __restrict__ w_table, const uint8_t* const* __restrict__ w_up_table,
                        const float* __restrict__ route_w, float limit, const float* __restrict__ x,
                        float* __restrict__ y, uint32_t m, uint32_t k, const int32_t* __restrict__ plan,
                        uint64_t expert_bytes, uint32_t x_slot, uint32_t slots_per_x, int zero_dropped,
                        int32_t min_count) {
  using F = Format<kType>;
  constexpr uint32_t kRows = 4 * kWaves;
  __shared__ uint32_t grid[F::kIq1Grid ? NGRID_IQ1S : 1];
  __shared__ uint32_t iq2_words[F::kIq2Words ? F::kIq2Words : 1];
  __shared__ uint8_t iq2_signs[F::kIq2Words ? 128 : 1];
  if ((int32_t)blockIdx.y >= plan[0]) return;
  const int32_t* g = plan + 1 + blockIdx.y * kPlanGroup;
  const int32_t id = g[0];
  /* A split launch: this pass takes the groups of min_count to kCols slots. */
  if (min_count > 0 && (g[1] < min_count || g[1] > kCols || (id < 0 && min_count > 1))) return;
  const int cols = g[1] < kCols ? g[1] : kCols;
  int32_t slot[kCols];
#pragma unroll
  for (int c = 0; c < kCols; ++c) slot[c] = c < cols ? g[2 + c] : 0;
  if (id < 0) {  // dropped slots
    if (zero_dropped) {
      for (int c = 0; c < cols; ++c)
        for (uint32_t r = threadIdx.x; r < kRows; r += blockDim.x)
          if (blockIdx.x * kRows + r < m) y[(uint64_t)slot[c] * m + blockIdx.x * kRows + r] = 0.0f;
    }
    return;
  }
  if constexpr (F::kIq1Grid) {
    for (uint32_t i = threadIdx.x; i < NGRID_IQ1S; i += blockDim.x) grid[i] = iq1s_grid_gpu[i];
    __syncthreads();
  }
  if constexpr (F::kIq2Words) {
    for (uint32_t i = threadIdx.x; i < F::kIq2Words / 2; i += blockDim.x) {
      const uint64_t g2 = Iq2GridTable<kType>()[i];
      iq2_words[2 * i] = static_cast<uint32_t>(g2);
      iq2_words[2 * i + 1] = static_cast<uint32_t>(g2 >> 32);
    }
    for (uint32_t i = threadIdx.x; i < 128; i += blockDim.x) iq2_signs[i] = ksigns_iq2xs[i];
    __syncthreads();
  }
  const Iq2Tab tab{iq2_words, iq2_signs};
  const int lane = threadIdx.x % kWave;
  const int wave = threadIdx.x / kWave;
  const int t = lane % 8;
  const uint32_t row = blockIdx.x * kRows + wave * 4 + lane / 8;
  if (row >= m) return;
  const float* xt[kCols];
#pragma unroll
  for (int c = 0; c < kCols; ++c) xt[c] = x + (uint64_t)(slot[c] / slots_per_x) * x_slot + 32 * t;
  const uint32_t blocks_per_row = k / 256;
  const uint64_t row_bytes = (uint64_t)row * blocks_per_row * F::kBlockBytes;
  const uint64_t offset = (uint64_t)id * expert_bytes + row_bytes;
  const uint8_t* base = kTable ? w_table[id] + row_bytes : w + offset;
  const uint8_t* base_up =
      kSwiGLU ? (kTable ? w_up_table[id] + row_bytes : w_up + offset) : nullptr;
  float acc[kCols] = {};
  float acc_up[kCols] = {};
  uint32_t blk = 0;
  for (; blk + U <= blocks_per_row; blk += U) {
    typename F::Raw raw[U];
    typename F::Raw raw_up[kSwiGLU ? U : 1];
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
        const float* xb = xt[c] + (blk + u) * 256;
        if constexpr (kSwiGLU && F::kXReg) {
          float xr[32];
#pragma unroll
          for (int i = 0; i < 32; i += 4) {
            const float4 f = *reinterpret_cast<const float4*>(xb + i);
            xr[i] = f.x;
            xr[i + 1] = f.y;
            xr[i + 2] = f.z;
            xr[i + 3] = f.w;
          }
          acc[c] = F::AccR(acc[c], raw[u], t, xr, grid, tab);
          acc_up[c] = F::AccR(acc_up[c], raw_up[u], t, xr, grid, tab);
        } else {
          acc[c] = F::Acc(acc[c], raw[u], t, xb, grid, tab);
          if constexpr (kSwiGLU) acc_up[c] = F::Acc(acc_up[c], raw_up[u], t, xb, grid, tab);
        }
      }
    }
  }
  for (; blk < blocks_per_row; ++blk) {
    const typename F::Raw raw = F::Load(base + (uint64_t)blk * F::kBlockBytes, t);
    typename F::Raw raw_up{};
    if constexpr (kSwiGLU) raw_up = F::Load(base_up + (uint64_t)blk * F::kBlockBytes, t);
#pragma unroll
    for (int c = 0; c < kCols; ++c) {
      if (c >= cols) break;
      const float* xb = xt[c] + blk * 256;
      if constexpr (kSwiGLU && F::kXReg) {
        float xr[32];
#pragma unroll
        for (int i = 0; i < 32; i += 4) {
          const float4 f = *reinterpret_cast<const float4*>(xb + i);
          xr[i] = f.x;
          xr[i + 1] = f.y;
          xr[i + 2] = f.z;
          xr[i + 3] = f.w;
        }
        acc[c] = F::AccR(acc[c], raw, t, xr, grid, tab);
        acc_up[c] = F::AccR(acc_up[c], raw_up, t, xr, grid, tab);
      } else {
        acc[c] = F::Acc(acc[c], raw, t, xb, grid, tab);
        if constexpr (kSwiGLU) acc_up[c] = F::Acc(acc_up[c], raw_up, t, xb, grid, tab);
      }
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

template <int kType, int U, int kCols, bool kTable>
int launch_group(const void* w, const void* w_up, const void* const* w_table, const void* const* w_up_table,
                 const float* route_w, float limit, uint64_t expert_bytes, const int32_t* plan, uint32_t groups,
                 const float* x, uint32_t x_slot, uint32_t slots_per_x, float* y, uint32_t m, uint32_t k,
                 int zero_dropped, int32_t min_count) {
  constexpr uint32_t kRows = 4 * kWaves;
  const dim3 blocks((m + kRows - 1) / kRows, groups);
  const auto* table = reinterpret_cast<const uint8_t* const*>(w_table);
  const auto* table_up = reinterpret_cast<const uint8_t* const*>(w_up_table);
  if (w_up)
    expert_group_kernel<kType, U, true, kCols, kTable><<<blocks, kWave * kWaves>>>(
        static_cast<const uint8_t*>(w), static_cast<const uint8_t*>(w_up), table, table_up, route_w, limit, x, y, m,
        k, plan, expert_bytes, x_slot, slots_per_x, zero_dropped, min_count);
  else
    expert_group_kernel<kType, U, false, kCols, kTable><<<blocks, kWave * kWaves>>>(
        static_cast<const uint8_t*>(w), nullptr, table, table_up, nullptr, 0.0f, x, y, m, k, plan, expert_bytes,
        x_slot, slots_per_x, zero_dropped, min_count);
  const hipError_t err = hipGetLastError();
  if (err == hipSuccess) return 0;
  fprintf(stderr, "ds41: grouped expert gemv launch failed: %s\n", hipGetErrorString(err));
  return 1;
}

template <int kType, int U, int kCols>
int launch_group(const void* w, const void* w_up, const void* const* w_table, const void* const* w_up_table,
                 const float* route_w, float limit, uint64_t expert_bytes, const int32_t* plan, uint32_t groups,
                 const float* x, uint32_t x_slot, uint32_t slots_per_x, float* y, uint32_t m, uint32_t k,
                 int zero_dropped, int32_t min_count = 0) {
  if (w_table)
    return launch_group<kType, U, kCols, true>(w, w_up, w_table, w_up_table, route_w, limit, expert_bytes, plan,
                                               groups, x, x_slot, slots_per_x, y, m, k, zero_dropped, min_count);
  return launch_group<kType, U, kCols, false>(w, w_up, w_table, w_up_table, route_w, limit, expert_bytes, plan,
                                              groups, x, x_slot, slots_per_x, y, m, k, zero_dropped, min_count);
}

/* UDown: blocks loaded ahead by the down projection. Its 2,304-value rows
 * are 9 blocks; loading a whole row at once measured faster for every group
 * width (the single-slot kernel was slower with it). */
template <int kType, int U, int UDown = U>
int dispatch_group(const void* w, const void* w_up, const void* const* w_table, const void* const* w_up_table,
                   const float* route_w, float limit, uint64_t expert_bytes, const int32_t* plan, uint32_t groups,
                   uint32_t cols, const float* x, uint32_t x_slot, uint32_t slots_per_x, float* y, uint32_t m,
                   uint32_t k, int zero_dropped) {
  if constexpr (UDown != U)
    if (!w_up)
      return dispatch_group<kType, UDown>(w, w_up, w_table, w_up_table, route_w, limit, expert_bytes, plan, groups,
                                          cols, x, x_slot, slots_per_x, y, m, k, zero_dropped);
  if (cols <= 2)
    return launch_group<kType, U, 2>(w, w_up, w_table, w_up_table, route_w, limit, expert_bytes, plan, groups, x,
                                     x_slot, slots_per_x, y, m, k, zero_dropped);
  /* IQ1_S's fused SwiGLU also splits 3- and 4-column launches. */
  const bool split4 = kType == 19 && w_up;
  if (cols <= 4 && !split4)
    return launch_group<kType, U, 4>(w, w_up, w_table, w_up_table, route_w, limit, expert_bytes, plan, groups, x,
                                     x_slot, slots_per_x, y, m, k, zero_dropped);
  /* Wider launches: groups of one or two slots (most of them when sessions
   * route apart) take the 2-column kernel and the rest a wider one. The
   * fused SwiGLU's 6- and 8-column kernels spill registers for every type
   * but IQ1_M, so those wider groups take the 16-column kernel. */
  if (launch_group<kType, U, 2>(w, w_up, w_table, w_up_table, route_w, limit, expert_bytes, plan, groups, x, x_slot,
                                slots_per_x, y, m, k, zero_dropped, 1))
    return 1;
  if (cols <= 4)
    return launch_group<kType, U, 4>(w, w_up, w_table, w_up_table, route_w, limit, expert_bytes, plan, groups, x,
                                     x_slot, slots_per_x, y, m, k, zero_dropped, 3);
  const bool narrow = !w_up || kType == 29;
  if (narrow && cols <= 6)
    return launch_group<kType, U, 6>(w, w_up, w_table, w_up_table, route_w, limit, expert_bytes, plan, groups, x,
                                     x_slot, slots_per_x, y, m, k, zero_dropped, 3);
  if (narrow && cols <= 8)
    return launch_group<kType, U, 8>(w, w_up, w_table, w_up_table, route_w, limit, expert_bytes, plan, groups, x,
                                     x_slot, slots_per_x, y, m, k, zero_dropped, 3);
  return launch_group<kType, U, kPlanCols>(w, w_up, w_table, w_up_table, route_w, limit, expert_bytes, plan, groups,
                                           x, x_slot, slots_per_x, y, m, k, zero_dropped, 3);
}

template <int kType, int U, bool kTable, bool kPair = false>
int launch(const void* w, const void* w_up, const void* const* w_table, const void* const* w_up_table,
           const float* route_w, float limit, uint64_t expert_bytes, const int32_t* ids, uint32_t slots,
           const float* x, uint32_t x_slot, uint32_t slots_per_x, float* y, uint32_t m, uint32_t k,
           int zero_dropped) {
  if (kPair && (w_up || m % (8 * kWaves) != 0)) return 1;
  constexpr uint32_t kRows = (kPair ? 8 : 4) * kWaves;
  const dim3 blocks((m + kRows - 1) / kRows, slots);
  const auto* table = reinterpret_cast<const uint8_t* const*>(w_table);
  const auto* table_up = reinterpret_cast<const uint8_t* const*>(w_up_table);
  if (w_up)
    expert_gemv_kernel<kType, U, true, kTable><<<blocks, kWave * kWaves>>>(
        static_cast<const uint8_t*>(w), static_cast<const uint8_t*>(w_up), table, table_up, route_w, limit, x, y, m,
        k, ids, expert_bytes, x_slot, slots_per_x, zero_dropped);
  else
    expert_gemv_kernel<kType, U, false, kTable, kPair><<<blocks, kWave * kWaves>>>(
            static_cast<const uint8_t*>(w), nullptr, table, table_up, nullptr, 0.0f, x, y, m, k, ids, expert_bytes, x_slot,
            slots_per_x, zero_dropped);
  const hipError_t err = hipGetLastError();
  if (err == hipSuccess) return 0;
  fprintf(stderr, "ds41: expert gemv launch failed: %s\n", hipGetErrorString(err));
  return 1;
}

template <int kType, int U>
int launch(const void* w, const void* w_up, const void* const* w_table, const void* const* w_up_table,
           const float* route_w, float limit, uint64_t expert_bytes, const int32_t* ids, uint32_t slots,
           const float* x, uint32_t x_slot, uint32_t slots_per_x, float* y, uint32_t m, uint32_t k,
           int zero_dropped) {
  /* IQ1 down projections pair rows, loading x once for two (IQ1_S 86 -> 78
   * us, IQ1_M 92 -> 87); the IQ2 formats lose with it. */
  if constexpr (kType == 19 || kType == 29) {
    if (!w_up && m % (8 * kWaves) == 0) {
      if (w_table)
        return launch<kType, U, true, true>(w, w_up, w_table, w_up_table, route_w, limit, expert_bytes, ids, slots, x,
                                            x_slot, slots_per_x, y, m, k, zero_dropped);
      return launch<kType, U, false, true>(w, w_up, w_table, w_up_table, route_w, limit, expert_bytes, ids, slots, x,
                                           x_slot, slots_per_x, y, m, k, zero_dropped);
    }
  }
  if (w_table)
    return launch<kType, U, true>(w, w_up, w_table, w_up_table, route_w, limit, expert_bytes, ids, slots, x, x_slot,
                                  slots_per_x, y, m, k, zero_dropped);
  return launch<kType, U, false>(w, w_up, w_table, w_up_table, route_w, limit, expert_bytes, ids, slots, x, x_slot,
                                 slots_per_x, y, m, k, zero_dropped);
}

}  // namespace

namespace {

int dispatch(int type, const void* w, const void* w_up, const void* const* w_table, const void* const* w_up_table,
             const float* route_w, float limit, uint64_t expert_bytes, const int32_t* ids, uint32_t slots,
             const float* x, uint32_t x_slot, uint32_t slots_per_x, float* y, uint32_t m, uint32_t k,
             int zero_dropped) {
  if (!ids || slots == 0 || slots_per_x == 0 || m == 0 || k % 256 != 0) return 1;
  /* Blocks loaded ahead: 2 measured best for every format but IQ1_M. */
  switch (type) {
  case 10:
    return launch<10, 1>(w, w_up, w_table, w_up_table, route_w, limit, expert_bytes, ids, slots, x, x_slot,
                         slots_per_x, y, m, k, zero_dropped);
  case 12:
    return launch<12, 1>(w, w_up, w_table, w_up_table, route_w, limit, expert_bytes, ids, slots, x, x_slot,
                         slots_per_x, y, m, k, zero_dropped);
  case 19:
    return launch<19, 2>(w, w_up, w_table, w_up_table, route_w, limit, expert_bytes, ids, slots, x, x_slot,
                         slots_per_x, y, m, k, zero_dropped);
  case 29:
    return launch<29, 1>(w, w_up, w_table, w_up_table, route_w, limit, expert_bytes, ids, slots, x, x_slot,
                         slots_per_x, y, m, k, zero_dropped);
  case 16:
    return launch<16, 2>(w, w_up, w_table, w_up_table, route_w, limit, expert_bytes, ids, slots, x, x_slot,
                         slots_per_x, y, m, k, zero_dropped);
  case 17:
    return launch<17, 2>(w, w_up, w_table, w_up_table, route_w, limit, expert_bytes, ids, slots, x, x_slot,
                         slots_per_x, y, m, k, zero_dropped);
  case 22:
    return launch<22, 2>(w, w_up, w_table, w_up_table, route_w, limit, expert_bytes, ids, slots, x, x_slot,
                         slots_per_x, y, m, k, zero_dropped);
  default:
    return 1;
  }
}

}  // namespace

extern "C" int ds41_v41_gemv_experts(int type, const void* w, uint64_t expert_bytes, const int32_t* ids,
                                     uint32_t slots, const float* x, uint32_t x_slot, uint32_t slots_per_x, float* y,
                                     uint32_t m, uint32_t k, const void* const* w_table, int zero_dropped) {
  return dispatch(type, w, nullptr, w_table, nullptr, nullptr, 0.0f, expert_bytes, ids, slots, x, x_slot, slots_per_x,
                  y, m, k, zero_dropped);
}

extern "C" int ds41_v41_gemv_experts_swiglu(int type, const void* gate, const void* up, uint64_t expert_bytes,
                                            const int32_t* ids, uint32_t slots, const float* x, uint32_t x_slot,
                                            uint32_t slots_per_x, const float* route_w, float limit, float* y,
                                            uint32_t m, uint32_t k, const void* const* w_table,
                                            const void* const* w_up_table, int zero_dropped) {
  if (!gate || !up) return 1;
  return dispatch(type, gate, up, w_table, w_up_table, route_w, limit, expert_bytes, ids, slots, x, x_slot,
                  slots_per_x, y, m, k, zero_dropped);
}

extern "C" int ds41_v41_expert_plan(const int32_t* ids, uint32_t slots, int32_t* plan) {
  if (!ids || !plan || slots == 0 || slots > DS41_V41_EXPERT_PLAN_SLOTS) return 1;
  expert_plan_kernel<<<1, kPlanSlots>>>(ids, slots, plan);
  const hipError_t err = hipGetLastError();
  if (err == hipSuccess) return 0;
  fprintf(stderr, "ds41: expert plan launch failed: %s\n", hipGetErrorString(err));
  return 1;
}

extern "C" int ds41_v41_gemv_experts_grouped(int type, const void* w, const void* w_up, const float* route_w,
                                             float limit, uint64_t expert_bytes, const int32_t* plan, uint32_t slots,
                                             uint32_t cols, const float* x, uint32_t x_slot, uint32_t slots_per_x,
                                             float* y, uint32_t m, uint32_t k, const void* const* w_table,
                                             const void* const* w_up_table, int zero_dropped) {
  if (!plan || slots == 0 || slots > DS41_V41_EXPERT_PLAN_SLOTS || cols == 0 || cols > kPlanCols ||
      slots_per_x == 0 || m == 0 || k % 256 != 0)
    return 1;
  switch (type) {
  case 10:
    return dispatch_group<10, 1>(w, w_up, w_table, w_up_table, route_w, limit, expert_bytes, plan, slots, cols, x,
                                 x_slot, slots_per_x, y, m, k, zero_dropped);
  case 12:
    return dispatch_group<12, 1>(w, w_up, w_table, w_up_table, route_w, limit, expert_bytes, plan, slots, cols, x,
                                 x_slot, slots_per_x, y, m, k, zero_dropped);
  case 19:
    return dispatch_group<19, 2, 9>(w, w_up, w_table, w_up_table, route_w, limit, expert_bytes, plan, slots, cols, x,
                                 x_slot, slots_per_x, y, m, k, zero_dropped);
  case 29:
    return dispatch_group<29, 1, 9>(w, w_up, w_table, w_up_table, route_w, limit, expert_bytes, plan, slots, cols, x,
                                 x_slot, slots_per_x, y, m, k, zero_dropped);
  case 16:
    return dispatch_group<16, 2, 9>(w, w_up, w_table, w_up_table, route_w, limit, expert_bytes, plan, slots, cols, x,
                                 x_slot, slots_per_x, y, m, k, zero_dropped);
  case 17:
    return dispatch_group<17, 2>(w, w_up, w_table, w_up_table, route_w, limit, expert_bytes, plan, slots, cols, x,
                                 x_slot, slots_per_x, y, m, k, zero_dropped);
  case 22:
    return dispatch_group<22, 2>(w, w_up, w_table, w_up_table, route_w, limit, expert_bytes, plan, slots, cols, x,
                                 x_slot, slots_per_x, y, m, k, zero_dropped);
  default:
    return 1;
  }
}
