/* Batched dense Q5_K and Q6_K projections on the FP16 matrix cores. The activations
 * are rounded to FP16 as they are staged (11-bit mantissa, against the 8-bit
 * block quantization of MMQ's Q8_1 inputs), or arrive as FP16 from a
 * producer that rounded them the same way and each weight decodes to FP16
 * as (d sc) q - dmin m (Q6_K: (d sc) (q - 32)), so the K loop has no per-block scale arithmetic
 * beside the matrix-core products; the products accumulate in FP32.
 *
 * A workgroup takes 256 weight rows and 128 tokens: each of its 16 waves
 * holds 16 rows as the B operand (lane = weight row) against eight 16-token
 * tiles, so each staged activation tile feeds 256 rows. A K16
 * step covers two 8-value groups of a 32-value sub-block; each half-wave
 * loads and decodes one of them (half h: group 2 e + h of step e) and takes
 * the other from the opposite half with permlanex16, so half 1 holds every
 * step with its groups swapped and reads its activation fragment swapped to
 * match (each half of the wave computes its output rows from its own operand
 * copy). */
#include <hip/hip_runtime.h>

#include <cstdint>

#include "ds41_v41.h"

namespace {

typedef _Float16 half2_t __attribute__((ext_vector_type(2)));
typedef _Float16 half8 __attribute__((ext_vector_type(8)));
typedef _Float16 half16 __attribute__((ext_vector_type(16)));
typedef float float8 __attribute__((ext_vector_type(8)));
typedef uint32_t u32x4 __attribute__((ext_vector_type(4)));

constexpr int kRows = 256;              // weight rows per workgroup, 16 per wave
constexpr int kTile = 128;              // tokens per workgroup
constexpr int kThreads = kRows * 2;
constexpr int kStageK = 64;             // activation values per LDS stage
constexpr int kPitch = kStageK + 8;     // halves; token rows land on distinct banks
constexpr uint32_t kHalfMagic = 0x64646464u;

/* The bytes of one Q5_K block a half-wave decodes: the header (d, dmin and
 * the packed 6-bit scales and mins), and for value offsets 8 h .. 8 h + 7
 * and 16 + 8 h .. 16 + 8 h + 7 of each sub-block (groups h and 2 + h) the
 * high-bit bytes and the low-nibble bytes of each sub-block pair. */
struct Q5Lane {
  u32x4 head;
  uint2 qh[2];
  uint2 qs[4][2];
};

__device__ __forceinline__ Q5Lane load_q5(const uint8_t* b, int h) {
  Q5Lane q;
  q.head = *reinterpret_cast<const u32x4*>(b);
#pragma unroll
  for (int part = 0; part < 2; ++part) {
    q.qh[part] = *reinterpret_cast<const uint2*>(b + 16 + 16 * part + 8 * h);
#pragma unroll
    for (int pair = 0; pair < 4; ++pair)
      q.qs[pair][part] = *reinterpret_cast<const uint2*>(b + 48 + 32 * pair + 16 * part + 8 * h);
  }
  return q;
}

__device__ __forceinline__ uint32_t scale_byte(const u32x4& head, int i) {
  const uint32_t word = i < 4 ? head.y : i < 8 ? head.z : head.w;  // scales start at byte 4
  return (word >> (8 * (i % 4))) & 0xffu;
}

/* (scale, bias) of sub-block j as a half2: weight = scale q - bias. */
__device__ __forceinline__ uint32_t q5_affine(const u32x4& head, int j) {
  uint32_t sc, mn;
  if (j < 4) {
    sc = scale_byte(head, j) & 63;
    mn = scale_byte(head, j + 4) & 63;
  } else {
    sc = (scale_byte(head, j + 4) & 0xf) | ((scale_byte(head, j - 4) >> 6) << 4);
    mn = (scale_byte(head, j + 4) >> 4) | ((scale_byte(head, j) >> 6) << 4);
  }
  const float d = static_cast<float>(__builtin_bit_cast(_Float16, static_cast<uint16_t>(head.x & 0xffffu)));
  const float dmin = static_cast<float>(__builtin_bit_cast(_Float16, static_cast<uint16_t>(head.x >> 16)));
  return __builtin_bit_cast(uint32_t, half2_t{static_cast<_Float16>(d * static_cast<float>(sc)),
                                              static_cast<_Float16>(-dmin * static_cast<float>(mn))});
}

/* Eight weights of sub-block j, part `part` of this half-wave, as four half2. */
__device__ __forceinline__ void q5_decode8(const Q5Lane& q, int j, int part, uint32_t affine, uint32_t* out) {
  const uint2 low = q.qs[j / 2][part];
  const uint2 high = q.qh[part];
  const uint32_t b0 = ((low.x >> (4 * (j % 2))) & 0x0f0f0f0fu) | (((high.x >> j) & 0x01010101u) << 4);
  const uint32_t b1 = ((low.y >> (4 * (j % 2))) & 0x0f0f0f0fu) | (((high.y >> j) & 0x01010101u) << 4);
  const half2_t s = __builtin_bit_cast(half2_t, affine);
  const half2_t scale = {s.x, s.x}, bias = {s.y, s.y};
  const half2_t magic = {static_cast<_Float16>(-1024.0f), static_cast<_Float16>(-1024.0f)};
  const uint32_t p[4] = {__builtin_amdgcn_perm(b0, kHalfMagic, 0x01050004u),
                         __builtin_amdgcn_perm(b0, kHalfMagic, 0x03070206u),
                         __builtin_amdgcn_perm(b1, kHalfMagic, 0x01050004u),
                         __builtin_amdgcn_perm(b1, kHalfMagic, 0x03070206u)};
#pragma unroll
  for (int i = 0; i < 4; ++i)
    out[i] = __builtin_bit_cast(uint32_t, (__builtin_bit_cast(half2_t, p[i]) + magic) * scale + bias);
}

/* The weight formats: a lane's bytes of a super-block, the per-sub-block
 * scale word, and eight weights of part `part` of sub-block j as four half2. */
struct Q5K {
  static constexpr int kBlockBytes = 176;  // d, dmin, scales[12], qh[32], qs[128]
  using Lane = Q5Lane;
  static __device__ __forceinline__ Lane load(const uint8_t* b, int h) { return load_q5(b, h); }
  static __device__ __forceinline__ uint32_t affine(const Lane& q, int j) { return q5_affine(q.head, j); }
  static __device__ __forceinline__ void decode8(const Lane& q, int j, int part, uint32_t affine, uint32_t* out) {
    q5_decode8(q, j, part, affine, out);
  }
};

/* The bytes of one Q6_K block a half-wave decodes: for value offsets
 * 8 h .. 8 h + 7 and 16 + 8 h .. 16 + 8 h + 7 of each 32-value sub-block
 * j = 4 n + g, the ql bytes of both ql vectors of half n (low nibbles for
 * g < 2, high for g >= 2) and the qh bytes of half n (bits 2 g), with the
 * 16 signed scales and d. Blocks are 210 bytes, so these loads are only
 * 2-byte aligned. */
struct Q6Lane {
  uint2 ql[2][2][2];  // [n][g & 1][part]
  uint2 qh[2][2];     // [n][part]
  u32x4 sc;
  uint32_t d;
};

struct Q6K {
  static constexpr int kBlockBytes = 210;  // ql[128], qh[64], scales[16], d
  using Lane = Q6Lane;
  static __device__ __forceinline__ Lane load(const uint8_t* b, int h) {
    Lane q;
#pragma unroll
    for (int n = 0; n < 2; ++n)
#pragma unroll
      for (int part = 0; part < 2; ++part) {
#pragma unroll
        for (int a = 0; a < 2; ++a)
          q.ql[n][a][part] = *reinterpret_cast<const uint2*>(b + 64 * n + 32 * a + 16 * part + 8 * h);
        q.qh[n][part] = *reinterpret_cast<const uint2*>(b + 128 + 32 * n + 16 * part + 8 * h);
      }
    q.sc = *reinterpret_cast<const u32x4*>(b + 192);
    q.d = *reinterpret_cast<const uint16_t*>(b + 208);
    return q;
  }
  /* The scales of parts 0 and 1 of sub-block j (scales 8 n + 2 g and the
   * next one) as a half2. */
  static __device__ __forceinline__ uint32_t affine(const Lane& q, int j) {
    const int i = 8 * (j >> 2) + 2 * (j & 3);
    const uint32_t word = q.sc[i / 4] >> (8 * (i % 4));
    const float d = static_cast<float>(__builtin_bit_cast(_Float16, static_cast<uint16_t>(q.d)));
    return __builtin_bit_cast(
        uint32_t, half2_t{static_cast<_Float16>(d * static_cast<float>(static_cast<int8_t>(word & 0xffu))),
                          static_cast<_Float16>(d * static_cast<float>(static_cast<int8_t>((word >> 8) & 0xffu)))});
  }
  static __device__ __forceinline__ void decode8(const Lane& q, int j, int part, uint32_t affine, uint32_t* out) {
    const int n = j >> 2, g = j & 3;
    const uint2 low = q.ql[n][g & 1][part];
    const uint2 high = q.qh[n][part];
    const int shift = 4 * (g >> 1);
    const uint32_t b0 = ((low.x >> shift) & 0x0f0f0f0fu) | (((high.x >> (2 * g)) & 0x03030303u) << 4);
    const uint32_t b1 = ((low.y >> shift) & 0x0f0f0f0fu) | (((high.y >> (2 * g)) & 0x03030303u) << 4);
    const half2_t s = __builtin_bit_cast(half2_t, affine);
    const _Float16 sc = part ? s.y : s.x;
    const half2_t scale = {sc, sc};
    /* 1024 + q - 1056 = q - 32, exact in FP16. */
    const half2_t magic = {static_cast<_Float16>(-1056.0f), static_cast<_Float16>(-1056.0f)};
    const uint32_t p[4] = {__builtin_amdgcn_perm(b0, kHalfMagic, 0x01050004u),
                           __builtin_amdgcn_perm(b0, kHalfMagic, 0x03070206u),
                           __builtin_amdgcn_perm(b1, kHalfMagic, 0x01050004u),
                           __builtin_amdgcn_perm(b1, kHalfMagic, 0x03070206u)};
#pragma unroll
    for (int i = 0; i < 4; ++i)
      out[i] = __builtin_bit_cast(uint32_t, (__builtin_bit_cast(half2_t, p[i]) + magic) * scale);
  }
};

/* Lane L's word from lane L ^ 16. With fetch-inactive and bound-control set
 * the instruction does not keep its destination's old value (every lane is
 * active here anyway), so the compiler no longer copies or zeroes a
 * register for it before each permute: about a fifth of the kernel's
 * vector instructions were those moves. */
__device__ __forceinline__ uint32_t other_half(uint32_t v) {
  return __builtin_amdgcn_permlanex16(v, v, 0x76543210u, 0xfedcba98u, true, true);
}

/* A barrier that waits only for LDS traffic: __syncthreads also waits for
 * every outstanding global load, which would wait for the weight and
 * activation prefetches at once. */
__device__ __forceinline__ void lds_barrier() { asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory"); }

__device__ __forceinline__ float finite_or_zero(float v) { return __builtin_isfinite(v) ? v : 0.0f; }

/* out[t, z m + r] = sum_c W_z[r, c] x[t, z k + c] for group z = blockIdx.z;
 * x rows are groups * k values of X (float or _Float16), out rows groups * m
 * of Y (float, or _Float16 rounded from the float result). */
template <typename F, bool kFull, typename X, typename Y>
__global__ __launch_bounds__(kThreads) void dense_f16_kernel(const uint8_t* __restrict__ w, const X* __restrict__ x,
                                                             Y* __restrict__ out, uint32_t m, uint32_t n, uint32_t k,
                                                             uint32_t groups, uint32_t tile0) {
  __shared__ __attribute__((aligned(16))) _Float16 act[2][kTile * kPitch];
  const int tid = threadIdx.x, wave = tid / 32, lane = tid % 32, h = lane / 16;
  const uint32_t z = blockIdx.z;
  const uint64_t row_bytes = (uint64_t)k / 256 * F::kBlockBytes;
  const uint32_t row = blockIdx.x * kRows + wave * 16 + lane % 16;
  const uint8_t* wr = w + ((uint64_t)z * m + row) * row_bytes;
  const uint32_t t0 = (tile0 + blockIdx.y) * kTile;
  /* Live 16-token tiles; a runtime count even for full tiles, since a
   * constant one makes the compiler hoist every fragment load and spill. */
  const int live = (int)min<uint32_t>(8, (n - t0 + 15) / 16);
  const uint64_t x_stride = (uint64_t)groups * k;

  /* Activation staging: a quarter of a stage row (16 values) of one token
   * per thread, loaded kAhead stages ahead into a ring of registers (FP16
   * rows take half the registers, so they can run further ahead). The ring
   * is of vector types: arrays of uint4 went to scratch memory. */
  constexpr int kPer = kStageK * kTile / kThreads;
  constexpr int kVec = kPer * sizeof(X) / 16;  // 16-byte loads per stage
  constexpr int kAhead = sizeof(X) == 2 ? 4 : 2;
  const uint32_t st = tid % kTile, sh = tid / kTile;
  const X* xs = kFull || t0 + st < n ? x + (t0 + st) * x_stride + (uint64_t)z * k + kPer * sh : nullptr;
  _Float16* as0 = act[0] + st * kPitch + kPer * sh;
  _Float16* as1 = act[1] + st * kPitch + kPer * sh;
  u32x4 ring[kAhead][kVec];
  const auto fetch = [&](u32x4(&slot)[kVec], uint32_t k0) {
    if (kFull || xs) {
#pragma unroll
      for (int c = 0; c < kVec; ++c) slot[c] = reinterpret_cast<const u32x4*>(xs + k0)[c];
    }
  };
  /* Stage g stores into buffer g & 1 while stage g-1 still reads the other
   * one: one barrier per stage instead of two. */
  int g = 0;
  const auto store = [&](const u32x4(&slot)[kVec]) {
    if (kFull || xs) {
      _Float16* as = (g & 1) ? as1 : as0;
#pragma unroll
      for (int c = 0; c < kPer / 8; ++c) {
        if constexpr (sizeof(X) == 2) {
          *reinterpret_cast<u32x4*>(as + 8 * c) = slot[c];
        } else {
          const float4 a = __builtin_bit_cast(float4, slot[2 * c]), b = __builtin_bit_cast(float4, slot[2 * c + 1]);
          *reinterpret_cast<half8*>(as + 8 * c) = half8{(_Float16)a.x, (_Float16)a.y, (_Float16)a.z, (_Float16)a.w,
                                                        (_Float16)b.x, (_Float16)b.y, (_Float16)b.z, (_Float16)b.w};
        }
      }
    }
  };

  float8 acc[8];
#pragma unroll
  for (int t = 0; t < 8; ++t) acc[t] = float8{};

  const uint32_t blocks = k / 256;
  typename F::Lane next = F::load(wr, h);
#pragma unroll
  for (int a = 0; a < kAhead; ++a)
    if (a * kStageK < k) fetch(ring[a], a * kStageK);
  for (uint32_t sb = 0; sb < blocks; ++sb) {
    const typename F::Lane q = next;
    if (sb + 1 < blocks) next = F::load(wr + (uint64_t)(sb + 1) * F::kBlockBytes, h);
#pragma unroll
    for (int stage = 0; stage < 4; ++stage) {
      const uint32_t k0 = sb * 256 + stage * kStageK;
      store(ring[stage % kAhead]);
      lds_barrier();
      if (k0 + kAhead * kStageK < k) fetch(ring[stage % kAhead], k0 + kAhead * kStageK);
#pragma unroll
      for (int jj = 0; jj < 2; ++jj) {
        const int j = stage * 2 + jj;
        const uint32_t affine = F::affine(q, j);
#pragma unroll
        for (int e = 0; e < 2; ++e) {
          uint32_t words[8];
          F::decode8(q, j, e, affine, words);
#pragma unroll
          for (int i = 0; i < 4; ++i)
            words[4 + i] = other_half(words[i]);
          const half16 b = __builtin_bit_cast(half16, words);
          const int kk = jj * 32 + e * 16;
          /* Half h reads values 8 h .. 8 h + 7 of its token row and takes
           * the other eight from the opposite half. */
          const auto frag = [&](int t) {
            const uint4 own =
                *reinterpret_cast<const uint4*>(act[g & 1] + (t * 16 + lane % 16) * kPitch + kk + 8 * h);
            uint32_t v[8] = {own.x, own.y, own.z, own.w};
#pragma unroll
            for (int i = 0; i < 4; ++i)
              v[4 + i] = other_half(v[i]);
            return __builtin_bit_cast(half16, v);
          };
#pragma unroll
          for (int t = 0; t < 8; ++t)
            if (t < live) acc[t] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(frag(t), b, acc[t]);
        }
      }
      ++g;
    }
  }

  /* Accumulator element e of a lane is (token 2 e + lane / 16, row lane % 16). */
  const uint64_t out_stride = (uint64_t)groups * m;
#pragma unroll
  for (int t = 0; t < 8; ++t) {
    if (t >= live) continue;
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      const uint32_t token = t0 + t * 16 + 2 * e + h;
      if (kFull || token < n) out[token * out_stride + (uint64_t)z * m + row] = (Y)finite_or_zero(acc[t][e]);
    }
  }
}

template <typename F, typename X, typename Y>
int dense_launch(const uint8_t* w, const void* x, void* out, uint32_t m, uint32_t n, uint32_t k, uint32_t groups) {
  const auto* xt = static_cast<const X*>(x);
  auto* yt = static_cast<Y*>(out);
  const uint32_t full = n / kTile;
  if (full)
    dense_f16_kernel<F, true, X, Y><<<dim3(m / kRows, full, groups), kThreads>>>(w, xt, yt, m, n, k, groups, 0);
  if (n % kTile)
    dense_f16_kernel<F, false, X, Y><<<dim3(m / kRows, 1, groups), kThreads>>>(w, xt, yt, m, n, k, groups, full);
  return hipGetLastError() == hipSuccess ? 0 : 1;
}

template <typename F>
int dense_io(const uint8_t* w, const void* x, void* out, uint32_t m, uint32_t n, uint32_t k, uint32_t groups,
             uint32_t io) {
  switch (io) {
  case 0:
    return dense_launch<F, float, float>(w, x, out, m, n, k, groups);
  case DS41_V41_DENSE_X_HALF:
    return dense_launch<F, _Float16, float>(w, x, out, m, n, k, groups);
  case DS41_V41_DENSE_OUT_HALF:
    return dense_launch<F, float, _Float16>(w, x, out, m, n, k, groups);
  default:
    return dense_launch<F, _Float16, _Float16>(w, x, out, m, n, k, groups);
  }
}

}  // namespace

extern "C" int ds41_v41_dense_f16(int type, const void* w, const void* x, void* out, uint32_t m, uint32_t n,
                                  uint32_t k, uint32_t groups, uint32_t io) {
  if ((type != 13 && type != 14) || m % kRows != 0 || k % 256 != 0 || n == 0 || groups == 0 ||
      (io & ~(DS41_V41_DENSE_X_HALF | DS41_V41_DENSE_OUT_HALF)))
    return 1;
  const auto* wb = static_cast<const uint8_t*>(w);
  return type == 13 ? dense_io<Q5K>(wb, x, out, m, n, k, groups, io) : dense_io<Q6K>(wb, x, out, m, n, k, groups, io);
}
