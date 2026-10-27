/* Standalone tuner for the V4.1 Flash routed-expert IQ GEMV (decode path).
 *
 * Includes the production translation unit (same flags) for the reference
 * kernels, quant tables and entry points, then times bit-exact variants
 * attacking the two measured limiters:
 *
 *   vec   one wide load per 32-value task payload instead of 4-7 2-byte
 *         loads (texture-addresser pressure)
 *   lds   IQ2 grid/sign tables copied to LDS, stored as u32 pairs (IQ1
 *         already stages its grid)
 *   fast  IQ2 decode on the u32 halves of each grid entry: byte extract +
 *         cvt (no u64->f32 ldexp dance, no 64-bit shifts) and signs applied
 *         by flipping the float sign bit (no compare/select pair)
 *   xreg  the SwiGLU pass loads x once per block into registers and feeds
 *         the gate and up accumulations (half the activation loads)
 *
 * Every variant keeps each FMA operand and the accumulation order of the
 * production kernel, so outputs must match bit for bit (memcmp, NaN-safe
 * random blocks). Weights rotate over the 384 experts so reads stay
 * DRAM-cold. */
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "src/models/deepseek_v41_flash/kernels/rocm/ds41_expert_gemv.hip.cpp"

namespace tune {
namespace {

using ds41_gemv_tasks::HalfBits;

constexpr int kWave = 32;
constexpr int kWaves = 8;

enum Feat { kVecL = 1, kLdsL = 2, kFastL = 4, kXRegL = 8 };

/* Iq1Dot over a register-resident slice of x (scalar reads, same FMA chain). */
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

/* Eight grid magnitudes with the sign bits of `signs`: on the fast path the
 * entry arrives as its u32 halves (no 64-bit shifts, no u64->f32 ldexp) and
 * -v is the float with its sign bit flipped. */
template <bool kFast>
__device__ __forceinline__ void Iq2GroupT(uint32_t glo, uint32_t ghi, uint32_t signs, float scale, float* w) {
  if constexpr (!kFast) {
    const uint64_t grid = glo | (static_cast<uint64_t>(ghi) << 32);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const float v = static_cast<float>((grid >> (8 * i)) & 0xFF) * scale;
      w[i] = (signs >> i) & 1 ? -v : v;
    }
  } else {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const uint32_t b = (((i < 4) ? glo : ghi) >> (8 * (i & 3))) & 0xFFu;
      const float v = static_cast<float>(b) * scale;
      w[i] = __uint_as_float(__float_as_uint(v) ^ (((signs >> i) & 1u) << 31));
    }
  }
}

struct Tab {
  const uint32_t* iq2_words;  // grid entries as lo/hi u32 pairs (kLds)
  const uint8_t* signs;       // ksigns_iq2xs (kLds)
  const uint32_t* iq1;        // iq1s grid (always LDS, as production)
};

template <int kType, int kFeat>
struct TuneFormat;

/* IQ1_S (50 bytes): d u16; qs 4 bytes at 2+4t; qh u8 at 34+t. */
template <int kFeat>
struct TuneFormat<19, kFeat> {
  static constexpr int kBlockBytes = 50;
  static constexpr bool kIq1Grid = true;
  static constexpr bool kIq2Grid = false;
  static constexpr int kGridEntries = 0;
  struct Raw {
    uint32_t d, qs, qh;
  };
  static __device__ __forceinline__ Raw Load(const uint8_t* b, int t) {
    if constexpr (kFeat & kVecL) {
      const auto* h = reinterpret_cast<const uint16_t*>(b);
      Raw r{};
      __builtin_memcpy(&r.d, b, 2);
      __builtin_memcpy(&r.qs, b + 2 + 4 * t, 4);
      r.qh = h[17 + t];
      return r;
    }
    const auto* h = reinterpret_cast<const uint16_t*>(b);
    return {static_cast<uint32_t>(h[0]),
            static_cast<uint32_t>(h[1 + 2 * t]) | (static_cast<uint32_t>(h[2 + 2 * t]) << 16),
            static_cast<uint32_t>(h[17 + t])};
  }
  static __device__ __forceinline__ float Acc1(float acc, const Raw& a, int, const float* x,
                                               const uint32_t* grid, const Tab&) {
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
  static __device__ __forceinline__ float Acc1R(float acc, const Raw& a, int, const float (&xr)[32],
                                                const uint32_t* grid, const Tab&) {
#pragma clang fp reassociate(off)
    const float d = HalfBits(a.d);
    const uint32_t qh = a.qh;
    const float scale = d * static_cast<float>(2 * ((qh >> 12) & 7) + 1);
    const float c = (qh & 0x8000 ? -IQ1S_DELTA : IQ1S_DELTA) - 1.0f;
    float dot = 0.0f;
#pragma unroll
    for (int l = 0; l < 4; ++l)
      dot += Iq1DotR(grid[((a.qs >> (8 * l)) & 0xFF) | (((qh >> (3 * l)) & 7) << 8)], c, xr, 8 * l);
    return __builtin_fmaf(scale, dot, acc);
  }
};

/* IQ1_M (56 bytes): qs 4 bytes at 4t; qh u16 at 32+2t; scales 8 bytes at 48. */
template <>
struct TuneFormat<29, 0> {  // replica: production Load/Acc verbatim
  static constexpr int kBlockBytes = 56;
  static constexpr bool kIq1Grid = true;
  static constexpr bool kIq2Grid = false;
  static constexpr int kGridEntries = 0;
  struct Raw {
    uint32_t r[7];
  };
  static __device__ __forceinline__ Raw Load(const uint8_t* b, int t) {
    const auto* h = reinterpret_cast<const uint16_t*>(b);
    return {{h[2 * t], h[2 * t + 1], h[16 + t], h[24], h[25], h[26], h[27]}};
  }
  static __device__ __forceinline__ float Acc1(float acc, const Raw& a, int t, const float* x,
                                               const uint32_t* grid, const Tab&) {
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
  static __device__ __forceinline__ float Acc1R(float acc, const Raw& a, int t, const float (&xr)[32],
                                                const uint32_t* grid, const Tab& tab) {
    return Acc1(acc, a, t, xr, grid, tab);
  }
};

template <int kFeat>
struct TuneFormat<29, kFeat> {
  static constexpr int kBlockBytes = 56;
  static constexpr bool kIq1Grid = true;
  static constexpr bool kIq2Grid = false;
  static constexpr int kGridEntries = 0;
  struct Raw {
    uint32_t d, qs, qh, s0, s1;
  };
  static __device__ __forceinline__ Raw Load(const uint8_t* b, int t) {
    const auto* h = reinterpret_cast<const uint16_t*>(b);
    Raw r{};
    __builtin_memcpy(&r.qs, b + 4 * t, 4);
    if constexpr (kFeat & kVecL) {
      r.d = h[0];
      r.qh = h[16 + t];
      uint64_t s;
      __builtin_memcpy(&s, b + 48, 8);
      r.s0 = static_cast<uint32_t>(s);
      r.s1 = static_cast<uint32_t>(s >> 32);
    } else {
      r.d = h[0];
      r.qh = h[16 + t];
      r.s0 = static_cast<uint32_t>(h[24]) | (static_cast<uint32_t>(h[25]) << 16);
      r.s1 = static_cast<uint32_t>(h[26]) | (static_cast<uint32_t>(h[27]) << 16);
    }
    return r;
  }
  static __device__ __forceinline__ void Scales(const Raw& a, float& d, uint32_t (&w)[4]) {
    w[0] = a.s0 & 0xFFFF;
    w[1] = a.s0 >> 16;
    w[2] = a.s1 & 0xFFFF;
    w[3] = a.s1 >> 16;
    d = HalfBits((w[0] >> 12) | ((w[1] >> 8) & 0x00F0) | ((w[2] >> 4) & 0x0F00) | (w[3] & 0xF000));
  }
  static __device__ __forceinline__ float Acc1(float acc, const Raw& a, int t, const float* x,
                                               const uint32_t* grid, const Tab&) {
    float d;
    uint32_t sw[4];
    Scales(a, d, sw);
    const uint32_t word = sw[t >> 1] >> (6 * (t & 1));
    const float s1 = d * static_cast<float>(2 * (word & 7) + 1);
    const float s2 = d * static_cast<float>(2 * ((word >> 3) & 7) + 1);
    const uint32_t qh0 = a.qh & 0xFF, qh1 = a.qh >> 8;
    const uint32_t high[4] = {(qh0 << 8) & 0x700, (qh0 << 4) & 0x700, (qh1 << 8) & 0x700, (qh1 << 4) & 0x700};
    const uint32_t flag[4] = {qh0 & 0x08, qh0 & 0x80, qh1 & 0x08, qh1 & 0x80};
    float dot[4];
#pragma unroll
    for (int l = 0; l < 4; ++l)
      dot[l] = Iq1Dot(grid[((a.qs >> (8 * l)) & 0xFF) | high[l]], (flag[l] ? -IQ1M_DELTA : IQ1M_DELTA) - 1.0f,
                      x + 8 * l);
    return acc + __builtin_fmaf(s1, dot[0] + dot[1], s2 * (dot[2] + dot[3]));
  }
  static __device__ __forceinline__ float Acc1R(float acc, const Raw& a, int t, const float (&xr)[32],
                                                const uint32_t* grid, const Tab& tab) {
    float d;
    uint32_t sw[4];
    Scales(a, d, sw);
    const uint32_t word = sw[t >> 1] >> (6 * (t & 1));
    const float s1 = d * static_cast<float>(2 * (word & 7) + 1);
    const float s2 = d * static_cast<float>(2 * ((word >> 3) & 7) + 1);
    const uint32_t qh0 = a.qh & 0xFF, qh1 = a.qh >> 8;
    const uint32_t high[4] = {(qh0 << 8) & 0x700, (qh0 << 4) & 0x700, (qh1 << 8) & 0x700, (qh1 << 4) & 0x700};
    const uint32_t flag[4] = {qh0 & 0x08, qh0 & 0x80, qh1 & 0x08, qh1 & 0x80};
    float dot[4];
#pragma unroll
    for (int l = 0; l < 4; ++l)
      dot[l] = Iq1DotR(grid[((a.qs >> (8 * l)) & 0xFF) | high[l]], (flag[l] ? -IQ1M_DELTA : IQ1M_DELTA) - 1.0f,
                       xr, 8 * l);
    (void)tab;
    return acc + __builtin_fmaf(s1, dot[0] + dot[1], s2 * (dot[2] + dot[3]));
  }
};

/* IQ2_XXS (66 bytes): d u16; per task 8 bytes at 2+8t (four grid bytes then a
 * u32 of four 7-bit sign indices and a 4-bit scale). */
template <int kFeat>
struct TuneFormat<16, kFeat> {
  static constexpr int kBlockBytes = 66;
  static constexpr bool kIq1Grid = false;
  static constexpr bool kIq2Grid = true;
  static constexpr int kGridEntries = 256;
  struct Raw {
    uint32_t d, qs, aux;
  };
  static __device__ __forceinline__ Raw Load(const uint8_t* b, int t) {
    Raw r{};
    __builtin_memcpy(&r.d, b, 2);
    if constexpr (kFeat & kVecL) {
      uint64_t v;
      __builtin_memcpy(&v, b + 2 + 8 * t, 8);
      r.qs = static_cast<uint32_t>(v);
      r.aux = static_cast<uint32_t>(v >> 32);
    } else {
      const auto* h = reinterpret_cast<const uint16_t*>(b);
      const uint16_t* g = h + 1 + 4 * t;
      r.qs = static_cast<uint32_t>(g[0]) | (static_cast<uint32_t>(g[1]) << 16);
      r.aux = static_cast<uint32_t>(g[2]) | (static_cast<uint32_t>(g[3]) << 16);
    }
    return r;
  }
  template <bool kReg>
  static __device__ __forceinline__ float AccT(float acc, const Raw& a, int, const float* x,
                                               const float (&xr)[32], const Tab& tab) {
    const float d = HalfBits(a.d);
    const uint32_t qs = a.qs, aux = a.aux;
    const float scale = d * (0.5f + static_cast<float>(aux >> 28)) * 0.25f;
    float w[32];
#pragma unroll
    for (int l = 0; l < 4; ++l) {
      const uint32_t idx = (qs >> (8 * l)) & 0xFF;
      uint32_t glo, ghi;
      if constexpr (kFeat & kLdsL) {
        glo = tab.iq2_words[2 * idx];
        ghi = tab.iq2_words[2 * idx + 1];
      } else {
        const uint64_t g = iq2xxs_grid[idx];
        glo = static_cast<uint32_t>(g);
        ghi = static_cast<uint32_t>(g >> 32);
      }
      const uint32_t sidx = (aux >> (7 * l)) & 127;
      const uint32_t signs = (kFeat & kLdsL) ? tab.signs[sidx] : ksigns_iq2xs[sidx];
      Iq2GroupT<(kFeat & kFastL) != 0>(glo, ghi, signs, scale, w + 8 * l);
    }
    if constexpr (kReg)
      return acc + AccumulateR(0.0f, w, xr);
    return acc + ds41_gemv_tasks::Accumulate<32>(0.0f, w, x);
  }
  static __device__ __forceinline__ float Acc1(float acc, const Raw& a, int t, const float* x,
                                               const uint32_t*, const Tab& tab) {
    float xr[32] = {};
    return AccT<false>(acc, a, t, x, xr, tab);
  }
  static __device__ __forceinline__ float Acc1R(float acc, const Raw& a, int t, const float (&xr)[32],
                                                const uint32_t*, const Tab& tab) {
    return AccT<true>(acc, a, t, nullptr, xr, tab);
  }
};

/* IQ2_XS (74 bytes): d u16; four q words at 2+8t; scale pair u16 at 66+2t. */
template <int kFeat>
struct TuneFormat<17, kFeat> {
  static constexpr int kBlockBytes = 74;
  static constexpr bool kIq1Grid = false;
  static constexpr bool kIq2Grid = true;
  static constexpr int kGridEntries = 512;
  struct Raw {
    uint32_t d, s0, s1, sc;
  };
  static __device__ __forceinline__ Raw Load(const uint8_t* b, int t) {
    Raw r{};
    __builtin_memcpy(&r.d, b, 2);
    __builtin_memcpy(&r.sc, b + 66 + 2 * (t >> 1), 2);
    if constexpr (kFeat & kVecL) {
      uint64_t v;
      __builtin_memcpy(&v, b + 2 + 8 * t, 8);
      r.s0 = static_cast<uint32_t>(v);
      r.s1 = static_cast<uint32_t>(v >> 32);
    } else {
      const auto* h = reinterpret_cast<const uint16_t*>(b);
      r.s0 = static_cast<uint32_t>(h[1 + 4 * t]) | (static_cast<uint32_t>(h[2 + 4 * t]) << 16);
      r.s1 = static_cast<uint32_t>(h[3 + 4 * t]) | (static_cast<uint32_t>(h[4 + 4 * t]) << 16);
    }
    return r;
  }
  template <bool kReg>
  static __device__ __forceinline__ float AccT(float acc, const Raw& a, int t, const float* x,
                                               const float (&xr)[32], const Tab& tab) {
    const float d = HalfBits(a.d);
    const uint32_t sc = (a.sc >> (8 * (t & 1))) & 0xFF;
    const float low = d * (0.5f + static_cast<float>(sc & 0xF)) * 0.25f;
    const float high = d * (0.5f + static_cast<float>(sc >> 4)) * 0.25f;
    float w[32];
#pragma unroll
    for (int l = 0; l < 4; ++l) {
      const uint32_t q = ((l < 2 ? a.s0 : a.s1) >> (16 * (l & 1))) & 0xFFFFu;
      const uint32_t idx = q & 0x1FF;
      uint32_t glo, ghi;
      if constexpr (kFeat & kLdsL) {
        glo = tab.iq2_words[2 * idx];
        ghi = tab.iq2_words[2 * idx + 1];
      } else {
        const uint64_t g = iq2xs_grid[idx];
        glo = static_cast<uint32_t>(g);
        ghi = static_cast<uint32_t>(g >> 32);
      }
      const uint32_t signs = (kFeat & kLdsL) ? tab.signs[q >> 9] : ksigns_iq2xs[q >> 9];
      Iq2GroupT<(kFeat & kFastL) != 0>(glo, ghi, signs, l < 2 ? low : high, w + 8 * l);
    }
    if constexpr (kReg)
      return acc + AccumulateR(0.0f, w, xr);
    return acc + ds41_gemv_tasks::Accumulate<32>(0.0f, w, x);
  }
  static __device__ __forceinline__ float Acc1(float acc, const Raw& a, int t, const float* x,
                                               const uint32_t*, const Tab& tab) {
    float xr[32] = {};
    return AccT<false>(acc, a, t, x, xr, tab);
  }
  static __device__ __forceinline__ float Acc1R(float acc, const Raw& a, int t, const float (&xr)[32],
                                                const uint32_t*, const Tab& tab) {
    return AccT<true>(acc, a, t, nullptr, xr, tab);
  }
};

/* IQ2_S (82 bytes): d u16; qs u32 at 2+4t; signs u32 at 34+4t; qh and scale
 * u16s at 66+2t and 74+2t. */
template <int kFeat>
struct TuneFormat<22, kFeat> {
  static constexpr int kBlockBytes = 82;
  static constexpr bool kIq1Grid = false;
  static constexpr bool kIq2Grid = true;
  static constexpr int kGridEntries = 1024;
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
  static __device__ __forceinline__ float AccT(float acc, const Raw& a, int t, const float* x,
                                               const float (&xr)[32], const Tab& tab) {
    const float d = HalfBits(a.d);
    const uint32_t qh = (a.qh >> (8 * (t & 1))) & 0xFF;
    const uint32_t sc = (a.sc >> (8 * (t & 1))) & 0xFF;
    const float low = d * (0.5f + static_cast<float>(sc & 0xF)) * 0.25f;
    const float high = d * (0.5f + static_cast<float>(sc >> 4)) * 0.25f;
    float w[32];
#pragma unroll
    for (int l = 0; l < 4; ++l) {
      const uint32_t idx = ((a.qs >> (8 * l)) & 0xFF) | ((qh << (8 - 2 * l)) & 0x300);
      uint32_t glo, ghi;
      if constexpr (kFeat & kLdsL) {
        glo = tab.iq2_words[2 * idx];
        ghi = tab.iq2_words[2 * idx + 1];
      } else {
        const uint64_t g = iq2s_grid[idx];
        glo = static_cast<uint32_t>(g);
        ghi = static_cast<uint32_t>(g >> 32);
      }
      const uint32_t signs = (a.signs >> (8 * l)) & 0xFF;
      Iq2GroupT<(kFeat & kFastL) != 0>(glo, ghi, signs, l < 2 ? low : high, w + 8 * l);
    }
    if constexpr (kReg)
      return acc + AccumulateR(0.0f, w, xr);
    return acc + ds41_gemv_tasks::Accumulate<32>(0.0f, w, x);
  }
  static __device__ __forceinline__ float Acc1(float acc, const Raw& a, int t, const float* x,
                                               const uint32_t*, const Tab& tab) {
    float xr[32] = {};
    return AccT<false>(acc, a, t, x, xr, tab);
  }
  static __device__ __forceinline__ float Acc1R(float acc, const Raw& a, int t, const float (&xr)[32],
                                                const uint32_t*, const Tab& tab) {
    return AccT<true>(acc, a, t, nullptr, xr, tab);
  }
};

template <int kType, int U, bool kSwiGLU, int kFeat>
__global__ void __launch_bounds__(kWave* kWaves)
    tune_gemv_kernel(const uint8_t* __restrict__ w, const uint8_t* __restrict__ w_up,
                     const float* __restrict__ route_w, float limit, const float* __restrict__ x,
                     float* __restrict__ y, uint32_t m, uint32_t k, const int32_t* __restrict__ ids,
                     uint64_t expert_bytes, uint32_t x_slot, uint32_t slots_per_x) {
  using F = TuneFormat<kType, kFeat>;
  constexpr uint32_t kRows = 4 * kWaves;
  __shared__ uint32_t s_iq1[F::kIq1Grid ? NGRID_IQ1S : 1];
  __shared__ uint32_t s_iq2[(F::kIq2Grid && (kFeat & kLdsL)) ? 2 * F::kGridEntries : 1];
  __shared__ uint8_t s_signs[(F::kIq2Grid && (kFeat & kLdsL)) ? 128 : 1];
  const int32_t id = ids[blockIdx.y];
  y += (uint64_t)blockIdx.y * m;
  if (id < 0) return;
  if constexpr (F::kIq1Grid) {
    for (uint32_t i = threadIdx.x; i < NGRID_IQ1S; i += blockDim.x) s_iq1[i] = iq1s_grid_gpu[i];
    __syncthreads();
  }
  if constexpr (F::kIq2Grid && (kFeat & kLdsL)) {
    for (uint32_t i = threadIdx.x; i < F::kGridEntries; i += blockDim.x) {
      const uint64_t g = kType == 16 ? iq2xxs_grid[i] : kType == 17 ? iq2xs_grid[i] : iq2s_grid[i];
      s_iq2[2 * i] = static_cast<uint32_t>(g);
      s_iq2[2 * i + 1] = static_cast<uint32_t>(g >> 32);
    }
    for (uint32_t i = threadIdx.x; i < 128; i += blockDim.x) s_signs[i] = ksigns_iq2xs[i];
    __syncthreads();
  }
  const Tab tab{s_iq2, s_signs, s_iq1};
  const int lane = threadIdx.x % kWave;
  const int wave = threadIdx.x / kWave;
  const int t = lane % 8;
  const uint32_t row = blockIdx.x * kRows + wave * 4 + lane / 8;
  if (row >= m) return;
  const float* xt = x + (uint64_t)(blockIdx.y / slots_per_x) * x_slot + 32 * t;
  const uint32_t blocks_per_row = k / 256;
  const uint64_t row_bytes = (uint64_t)row * blocks_per_row * F::kBlockBytes;
  const uint64_t offset = (uint64_t)id * expert_bytes + row_bytes;
  const uint8_t* base = w + offset;
  const uint8_t* base_up = kSwiGLU ? w_up + offset : nullptr;
  float acc = 0.0f;
  float acc_up = 0.0f;
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
      const float* xb = xt + (blk + u) * 256;
      if constexpr (kSwiGLU && (kFeat & kXRegL)) {
        float xr[32];
#pragma unroll
        for (int i = 0; i < 32; i += 4) {
          const float4 f = *reinterpret_cast<const float4*>(xb + i);
          xr[i] = f.x;
          xr[i + 1] = f.y;
          xr[i + 2] = f.z;
          xr[i + 3] = f.w;
        }
        acc = F::Acc1R(acc, raw[u], t, xr, s_iq1, tab);
        acc_up = F::Acc1R(acc_up, raw_up[u], t, xr, s_iq1, tab);
      } else {
        acc = F::Acc1(acc, raw[u], t, xb, s_iq1, tab);
        if constexpr (kSwiGLU) acc_up = F::Acc1(acc_up, raw_up[u], t, xb, s_iq1, tab);
      }
    }
  }
  for (; blk < blocks_per_row; ++blk) {
    const typename F::Raw raw = F::Load(base + (uint64_t)blk * F::kBlockBytes, t);
    typename F::Raw raw_up{};
    if constexpr (kSwiGLU) raw_up = F::Load(base_up + (uint64_t)blk * F::kBlockBytes, t);
    const float* xb = xt + blk * 256;
    if constexpr (kSwiGLU && (kFeat & kXRegL)) {
      float xr[32];
#pragma unroll
      for (int i = 0; i < 32; i += 4) {
        const float4 f = *reinterpret_cast<const float4*>(xb + i);
        xr[i] = f.x;
        xr[i + 1] = f.y;
        xr[i + 2] = f.z;
        xr[i + 3] = f.w;
      }
      acc = F::Acc1R(acc, raw, t, xr, s_iq1, tab);
      acc_up = F::Acc1R(acc_up, raw_up, t, xr, s_iq1, tab);
    } else {
      acc = F::Acc1(acc, raw, t, xb, s_iq1, tab);
      if constexpr (kSwiGLU) acc_up = F::Acc1(acc_up, raw_up, t, xb, s_iq1, tab);
    }
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

template <int kType, int U, bool kSwiGLU, int kFeat>
void launch_tune(const void* w, const void* w_up, uint64_t expert_bytes, const int32_t* ids, uint32_t slots,
                 const float* x, uint32_t x_slot, uint32_t slots_per_x, float* y, uint32_t m, uint32_t k) {
  constexpr uint32_t kRows = 4 * kWaves;
  const dim3 blocks((m + kRows - 1) / kRows, slots);
  tune_gemv_kernel<kType, U, kSwiGLU, kFeat><<<blocks, kWave * kWaves>>>(
      static_cast<const uint8_t*>(w), static_cast<const uint8_t*>(w_up), nullptr, 0.0f, x, y, m, k, ids,
      expert_bytes, x_slot, slots_per_x);
}

template <int kType, int U, int kFeat>
void launch_pick(const void* w, const void* w_up, uint64_t expert_bytes, const int32_t* ids, uint32_t slots,
                 const float* x, uint32_t x_slot, uint32_t slots_per_x, float* y, uint32_t m, uint32_t k) {
  if (w_up)
    launch_tune<kType, U, true, kFeat>(w, w_up, expert_bytes, ids, slots, x, x_slot, slots_per_x, y, m, k);
  else
    launch_tune<kType, U, false, kFeat>(w, nullptr, expert_bytes, ids, slots, x, x_slot, slots_per_x, y, m, k);
}

struct Variant {
  const char* name;
  int feat;
  int u;
};

/* One (shape, variant) instantiation chosen by name at run time. */
using Launcher = void (*)(const void*, const void*, uint64_t, const int32_t*, uint32_t, const float*, uint32_t,
                          uint32_t, float*, uint32_t, uint32_t);

template <int kType>
Launcher VariantLauncher(int feat, int u) {
  switch (feat | (u << 8)) {
  case 0x100:
    return launch_pick<kType, 1, 0>;
  case kFastL | 0x100:
    return launch_pick<kType, 1, kFastL>;
  case kVecL | 0x100:
    return launch_pick<kType, 1, kVecL>;
  case (kVecL | kFastL) | 0x100:
    return launch_pick<kType, 1, kVecL | kFastL>;
  case (kVecL | kLdsL | kFastL) | 0x100:
    return launch_pick<kType, 1, kVecL | kLdsL | kFastL>;
  case (kVecL | kLdsL | kFastL | kXRegL) | 0x100:
    return launch_pick<kType, 1, kVecL | kLdsL | kFastL | kXRegL>;
  case (kVecL | kLdsL | kFastL | kXRegL) | 0x200:
    return launch_pick<kType, 2, kVecL | kLdsL | kFastL | kXRegL>;
  case kVecL | 0x200:
    return launch_pick<kType, 2, kVecL>;
  case (kVecL | kXRegL) | 0x100:
    return launch_pick<kType, 1, kVecL | kXRegL>;
  case (kVecL | kXRegL) | 0x200:
    return launch_pick<kType, 2, kVecL | kXRegL>;
  case (kVecL | kXRegL) | 0x400:
    return launch_pick<kType, 4, kVecL | kXRegL>;
  case (kVecL | kLdsL) | 0x100:
    return launch_pick<kType, 1, kVecL | kLdsL>;
  default:
    return nullptr;
  }
}

}  // namespace
}  // namespace tune

int main(int argc, char** argv) {
  setbuf(stdout, nullptr);
  const char* only = argc > 1 ? argv[1] : nullptr;
  struct Shape {
    const char* name;
    int type, block_bytes, m, k;
    bool swiglu;  // gate+up fused (two matrices) or plain down
  };
  const Shape shapes[] = {
      {"iq1s_gu", 19, 50, 2304, 5120, true},    {"iq1s_down", 19, 50, 5120, 2304, false},
      {"iq1m_gu", 29, 56, 2304, 5120, true},    {"iq1m_down", 29, 56, 5120, 2304, false},
      {"iq2xxs_gu", 16, 66, 2304, 5120, true},  {"iq2xxs_down", 16, 66, 5120, 2304, false},
      {"iq2xs_gu", 17, 74, 2304, 5120, true},   {"iq2xs_down", 17, 74, 5120, 2304, false},
      {"iq2s_down", 22, 82, 5120, 2304, false},
  };
  const tune::Variant iq1_variants[] = {
      {"repl", 0, 1}, {"vec", tune::kVecL, 0}, {"vec+x", tune::kVecL | tune::kXRegL, 0}, {"vec+x u2", tune::kVecL | tune::kXRegL, 2}};
  const tune::Variant iq2_variants[] = {
      {"fast", tune::kFastL, 1},
      {"vec", tune::kVecL, 1},
      {"vec+fast", tune::kVecL | tune::kFastL, 1},
      {"vec+lds+fast", tune::kVecL | tune::kLdsL | tune::kFastL, 1},
      {"v+l+f+x", tune::kVecL | tune::kLdsL | tune::kFastL | tune::kXRegL, 1},
      {"v+l+f+x u2", tune::kVecL | tune::kLdsL | tune::kFastL | tune::kXRegL, 2},
  };
  constexpr int kExperts = 384;
  constexpr int kRounds = 64;
  std::mt19937 rng(41);
  std::vector<int32_t> idh(static_cast<std::size_t>(kRounds) * 6);
  for (int r = 0; r < kRounds; ++r)
    for (int j = 0; j < 6; ++j) idh[r * 6 + j] = static_cast<int32_t>((r * 6 + j) * 61 % kExperts);
  for (const Shape& s : shapes) {
    if (only && std::strcmp(only, s.name) != 0) continue;
    std::fprintf(stderr, "shape %s\n", s.name);
    const std::size_t per_expert = static_cast<std::size_t>(s.m) * s.k / 256 * s.block_bytes;
    const int matrices = s.swiglu ? 2 : 1;
    std::vector<uint8_t> host(per_expert * kExperts * matrices);
    for (auto& b : host)
      b = static_cast<uint8_t>(rng());
    const std::size_t x_count = s.swiglu ? 1 : 6;  // down: one x per slot
    std::vector<float> xh(static_cast<std::size_t>(s.k) * x_count);
    std::normal_distribution<float> normal;
    for (auto& v : xh)
      v = normal(rng);
    void *w = nullptr, *w_up = nullptr;
    float *x = nullptr, *y = nullptr, *yv = nullptr;
    int32_t* ids = nullptr;
    (void)hipMalloc(&w, per_expert * kExperts);
    if (s.swiglu) (void)hipMalloc(&w_up, per_expert * kExperts);
    (void)hipMalloc(&x, xh.size() * sizeof(float));
    (void)hipMalloc(&y, static_cast<std::size_t>(s.m) * 6 * sizeof(float));
    (void)hipMalloc(&yv, static_cast<std::size_t>(s.m) * 6 * sizeof(float));
    (void)hipMalloc(&ids, idh.size() * sizeof(int32_t));
    (void)hipMemcpy(w, host.data(), per_expert * kExperts, hipMemcpyHostToDevice);
    if (s.swiglu)
      (void)hipMemcpy(w_up, host.data() + per_expert * kExperts, per_expert * kExperts, hipMemcpyHostToDevice);
    (void)hipMemcpy(x, xh.data(), xh.size() * sizeof(float), hipMemcpyHostToDevice);
    (void)hipMemcpy(ids, idh.data(), idh.size() * sizeof(int32_t), hipMemcpyHostToDevice);
    const uint32_t x_slot = s.swiglu ? 0 : s.k;
    const uint32_t slots_per_x = s.swiglu ? 1 : 1;
    auto production = [&](int r, float* out) {
      if (s.swiglu)
        ds41_v41_gemv_experts_swiglu(s.type, w, w_up, per_expert, ids + r * 6, 6, x, x_slot, slots_per_x, nullptr,
                                     0.0f, out, s.m, s.k, nullptr, nullptr, 1);
      else
        ds41_v41_gemv_experts(s.type, w, per_expert, ids + r * 6, 6, x, x_slot, slots_per_x, out, s.m, s.k, nullptr, 1);
    };
    auto time = [&](auto&& launch) {
      hipEvent_t a, b;
      (void)hipEventCreate(&a);
      (void)hipEventCreate(&b);
      for (int r = 0; r < 8; ++r)
        launch(r, y);
      (void)hipEventRecord(a, 0);
      for (int r = 0; r < kRounds; ++r)
        launch(r, y);
      (void)hipEventRecord(b, 0);
      (void)hipEventSynchronize(b);
      float ms = 0;
      (void)hipEventElapsedTime(&ms, a, b);
      (void)hipEventDestroy(a);
      (void)hipEventDestroy(b);
      return ms * 1000.0 / kRounds;
    };
    const double bytes = matrices * 6.0 * per_expert;
    const double us = time(production);
    production(0, y);
    std::vector<float> ref(static_cast<std::size_t>(s.m) * 6);
    (void)hipMemcpy(ref.data(), y, ref.size() * sizeof(float), hipMemcpyDeviceToHost);
    std::printf("%-11s %5dx%-5d %6.1f MB  prod %7.1f us %5.1f GB/s", s.name, s.m, s.k, bytes / 1e6, us,
                bytes / us / 1e3);
    const bool iq1 = s.type == 19 || s.type == 29;
    const tune::Variant* variants = iq1 ? iq1_variants : iq2_variants;
    const int n_variants = iq1 ? 4 : 6;
    for (int vi = 0; vi < n_variants; ++vi) {
      const tune::Variant& v = variants[vi];
      auto launcher = s.type == 19 ? tune::VariantLauncher<19>(v.feat, v.u ? v.u : (s.type == 19 ? 2 : 1))
                    : s.type == 29 ? tune::VariantLauncher<29>(v.feat, v.u ? v.u : 1)
                    : s.type == 16 ? tune::VariantLauncher<16>(v.feat, v.u ? v.u : 1)
                    : s.type == 17 ? tune::VariantLauncher<17>(v.feat, v.u ? v.u : 1)
                                   : tune::VariantLauncher<22>(v.feat, v.u ? v.u : 1);
      if (!launcher) {
        std::printf("  %s n/a", v.name);
        continue;
      }
      auto run = [&](int r, float* out) {
        launcher(w, s.swiglu ? w_up : nullptr, per_expert, ids + r * 6, 6, x, x_slot, slots_per_x, out, s.m, s.k);
      };
      const double vus = time(run);
      run(0, yv);
      std::vector<float> got(static_cast<std::size_t>(s.m) * 6);
      (void)hipMemcpy(got.data(), yv, got.size() * sizeof(float), hipMemcpyDeviceToHost);
      const bool same = memcmp(got.data(), ref.data(), got.size() * sizeof(float)) == 0;
      std::printf("  %s %7.1f us %5.1f GB/s%s", v.name, vus, bytes / vus / 1e3, same ? "" : " MISMATCH");
      if (!same) {
        for (std::size_t i = 0; i < got.size(); ++i)
          if (memcmp(&got[i], &ref[i], 4) != 0) {
            std::printf("\n   first diff slot %zu row %zu: ref %.9e (%a) got %.9e (%a)", i / s.m, i % s.m,
                        ref[i], ref[i], got[i], got[i]);
            break;
          }
        int diffs = 0;
        for (std::size_t i = 0; i < got.size(); ++i)
          diffs += memcmp(&got[i], &ref[i], 4) != 0;
        std::printf("  (%d of %zu rows differ)", diffs, got.size());
      }
    }
    std::printf("\n");
    (void)hipFree(w);
    (void)hipFree(w_up);
    (void)hipFree(x);
    (void)hipFree(y);
    (void)hipFree(yv);
    (void)hipFree(ids);
  }
  return 0;
}
