/* DeepSeek V4.1 Flash decode and prefill kernels; see ds41_v41.h. */
#include "ds41_v41.h"

#include <hip/hip_fp16.h>

#include <type_traits>
#include <hip/hip_runtime.h>

#include <rocwmma/rocwmma.hpp>

#include <cmath>
#include <cstdio>

namespace {

constexpr uint32_t kThreads = 256;

int launch_ok(const char* what) {
  const hipError_t err = hipGetLastError();
  if (err == hipSuccess) return 0;
  fprintf(stderr, "ds41: %s launch failed: %s\n", what, hipGetErrorString(err));
  return 1;
}

__device__ __forceinline__ float bf16_to_f32(uint16_t v) { return __uint_as_float((uint32_t)v << 16); }

/* Block-wide sum over blockDim.x <= 1024 threads; every thread gets the result. */
__device__ float block_sum(float v) {
  __shared__ float partial[32];
  for (int o = warpSize / 2; o > 0; o >>= 1) v += __shfl_xor(v, o);
  const int lane = threadIdx.x % warpSize;
  const int warp = threadIdx.x / warpSize;
  __syncthreads();
  if (lane == 0) partial[warp] = v;
  __syncthreads();
  const int warps = (blockDim.x + warpSize - 1) / warpSize;
  float total = 0.0f;
  for (int i = 0; i < warps; i++) total += partial[i];
  return total;
}

__device__ float block_max(float v) {
  __shared__ float partial[32];
  for (int o = warpSize / 2; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor(v, o));
  const int lane = threadIdx.x % warpSize;
  const int warp = threadIdx.x / warpSize;
  __syncthreads();
  if (lane == 0) partial[warp] = v;
  __syncthreads();
  const int warps = (blockDim.x + warpSize - 1) / warpSize;
  float m = -INFINITY;
  for (int i = 0; i < warps; i++) m = fmaxf(m, partial[i]);
  return m;
}

/* Exclusive prefix sum of one value per thread. */
__device__ uint32_t block_exclusive_scan(uint32_t v, uint32_t* total) {
  __shared__ uint32_t values[1024];
  values[threadIdx.x] = v;
  __syncthreads();
  for (uint32_t step = 1; step < blockDim.x; step <<= 1) {
    const uint32_t add = threadIdx.x >= step ? values[threadIdx.x - step] : 0u;
    __syncthreads();
    values[threadIdx.x] += add;
    __syncthreads();
  }
  const uint32_t inclusive = values[threadIdx.x];
  *total = values[blockDim.x - 1];
  __syncthreads();
  return inclusive - v;
}

/* 2^ceil(log2(x)) for positive normal x. */
__device__ __forceinline__ float pow2_ceil(float x) {
  int e = 0;
  const float m = frexpf(x, &e);
  return ldexpf(1.0f, m == 0.5f ? e - 1 : e);
}

/* Round to the nearest E4M3FN value (|x| <= 448), ties to even. */
__device__ __forceinline__ float round_e4m3(float x) {
  const float a = fminf(fabsf(x), 448.0f);
  float step;
  if (a < 0.015625f) {  // below the smallest normal 2^-6: subnormal spacing 2^-9
    step = 0.001953125f;
  } else {
    int e = 0;
    frexpf(a, &e);  // a = m * 2^e, m in [0.5, 1)
    step = ldexpf(1.0f, e - 1 - 3);
  }
  return copysignf(fminf(rintf(a / step) * step, 448.0f), x);
}

/* Round to the nearest E2M1 value (|x| <= 6), ties to even. */
__device__ __forceinline__ float round_e2m1(float x) {
  const float a = fminf(fabsf(x), 6.0f);
  const float step = a < 2.0f ? 0.5f : (a < 4.0f ? 1.0f : 2.0f);
  return copysignf(fminf(rintf(a / step) * step, 6.0f), x);
}

/* Compressed rows visible at position p of a ratio-r layer. */
__device__ __forceinline__ uint32_t comp_len(uint32_t p, uint32_t ratio) { return ratio ? (p + 1) / ratio : 0u; }

/* Compressed KV and index-K rows are stored as FP4 (see ds41_fp4_rows).
 * Decoding multiplies the same E2M1 value and block scale as the
 * quantization, so it returns the quantized floats exactly. */
__device__ __forceinline__ float e2m1_value(uint32_t c) {
  const uint32_t m = c & 7u;
  const uint32_t bits = m < 2u ? (m ? 0x3f000000u : 0u) : ((((m >> 1) + 126u) << 23) | ((m & 1u) << 22));
  return __uint_as_float(bits | (c & 8u) << 28);
}
__device__ __forceinline__ uint32_t e2m1_code(float r) {
  const float a = fabsf(r);
  const uint32_t idx = a < 2.0f ? (uint32_t)(a * 2.0f) : (a < 4.0f ? (uint32_t)a + 2u : (uint32_t)a / 2u + 4u);
  return idx | (signbit(r) ? 8u : 0u);
}
/* Positive E4M3FN values (subnormals included) and their bytes. */
__device__ __forceinline__ float e4m3_value(uint32_t b) {
  const uint32_t e = b >> 3, m = b & 7u;
  return e ? __uint_as_float(((e + 120u) << 23) | (m << 20)) : (float)m * 0.001953125f;
}
__device__ __forceinline__ uint32_t e4m3_byte(float s) {
  if (s < 0.015625f) return (uint32_t)(s * 512.0f);
  const uint32_t bits = __float_as_uint(s);
  return (((bits >> 23) - 120u) << 3) | ((bits >> 20) & 7u);
}
__device__ __forceinline__ float4 fp4x4(uint32_t codes, float scale) {
  return make_float4(e2m1_value(codes) * scale, e2m1_value(codes >> 4) * scale, e2m1_value(codes >> 8) * scale,
                     e2m1_value(codes >> 12) * scale);
}
/* Values d .. d + 3 (d a multiple of 4) of a KV row (block 16, E4M3 scales)
 * or an index-K row (block 32, power-of-two scales) of `dim` values. */
__device__ __forceinline__ float4 kv_load4(ds41_fp4_rows r, uint64_t row, uint32_t d, uint32_t dim) {
  const uint32_t codes = *reinterpret_cast<const uint16_t*>(r.codes + row * (dim / 2) + d / 2);
  return fp4x4(codes, e4m3_value(r.scales[row * (dim / 16) + d / 16]));
}
__device__ __forceinline__ float4 ik_load4(ds41_fp4_rows r, uint64_t row, uint32_t d, uint32_t dim) {
  const uint32_t codes = *reinterpret_cast<const uint16_t*>(r.codes + row * (dim / 2) + d / 2);
  return fp4x4(codes, __uint_as_float((uint32_t)r.scales[row * (dim / 32) + d / 32] << 23));
}
/* Eight values d .. d + 7 (d a multiple of 8), as two float4, from one code word. */
__device__ __forceinline__ void kv_load8(ds41_fp4_rows r, uint64_t row, uint32_t d, uint32_t dim, float4* lo,
                                         float4* hi) {
  const uint32_t codes = *reinterpret_cast<const uint32_t*>(r.codes + row * (dim / 2) + d / 2);
  const float scale = e4m3_value(r.scales[row * (dim / 16) + d / 16]);
  *lo = fp4x4(codes, scale);
  *hi = fp4x4(codes >> 16, scale);
}
__device__ __forceinline__ void ik_load8(ds41_fp4_rows r, uint64_t row, uint32_t d, uint32_t dim, float4* lo,
                                         float4* hi) {
  const uint32_t codes = *reinterpret_cast<const uint32_t*>(r.codes + row * (dim / 2) + d / 2);
  const float scale = __uint_as_float((uint32_t)r.scales[row * (dim / 32) + d / 32] << 23);
  *lo = fp4x4(codes, scale);
  *hi = fp4x4(codes >> 16, scale);
}
__device__ __forceinline__ uint2 half4_bits(float4 v) {
  const __half2 lo = __floats2half2_rn(v.x, v.y);
  const __half2 hi = __floats2half2_rn(v.z, v.w);
  return make_uint2(*reinterpret_cast<const uint32_t*>(&lo), *reinterpret_cast<const uint32_t*>(&hi));
}
__device__ __forceinline__ float kv_value(ds41_fp4_rows r, uint64_t row, uint32_t d, uint32_t dim) {
  return e2m1_value(r.codes[row * (dim / 2) + d / 2] >> (4 * (d & 1u))) *
         e4m3_value(r.scales[row * (dim / 16) + d / 16]);
}
__device__ __forceinline__ float ik_value(ds41_fp4_rows r, uint64_t row, uint32_t d, uint32_t dim) {
  return e2m1_value(r.codes[row * (dim / 2) + d / 2] >> (4 * (d & 1u))) *
         __uint_as_float((uint32_t)r.scales[row * (dim / 32) + d / 32] << 23);
}

constexpr uint32_t kTokenTile = 8;

/* Warp per weight row, kTokenTile tokens per block column. */
template <typename W>
__global__ void matmul_kernel(const W* w, const float* x, float* out, uint32_t rows, uint32_t k, uint32_t n) {
  const uint32_t row = blockIdx.x * (blockDim.x / warpSize) + threadIdx.x / warpSize;
  if (row >= rows) return;
  const uint32_t t0 = blockIdx.y * kTokenTile;
  const uint32_t tn = min(kTokenTile, n - t0);
  const W* r = w + (uint64_t)row * k;
  float acc[kTokenTile] = {};
  for (uint32_t i = threadIdx.x % warpSize; i < k; i += warpSize) {
    float wv;
    if constexpr (std::is_same_v<W, uint16_t>)
      wv = bf16_to_f32(r[i]);
    else
      wv = (float)r[i];
#pragma unroll
    for (uint32_t j = 0; j < kTokenTile; j++)
      if (j < tn) acc[j] += wv * x[(uint64_t)(t0 + j) * k + i];
  }
#pragma unroll
  for (uint32_t j = 0; j < kTokenTile; j++) {
    float v = acc[j];
    for (int o = warpSize / 2; o > 0; o >>= 1) v += __shfl_xor(v, o);
    if (j < tn && threadIdx.x % warpSize == 0) out[(uint64_t)(t0 + j) * rows + row] = v;
  }
}

/* BF16 rows with k % 256 == 0: each lane loads eight weights (16 bytes) and
 * the matching eight inputs per step. */
__global__ void matmul_bf16_vec8_kernel(const uint16_t* w, const float* x, float* out, uint32_t rows, uint32_t k,
                                        uint32_t n) {
  const uint32_t row = blockIdx.x * (blockDim.x / warpSize) + threadIdx.x / warpSize;
  if (row >= rows) return;
  const uint32_t lane = threadIdx.x % warpSize;
  const uint32_t t0 = blockIdx.y * kTokenTile;
  const uint32_t tn = min(kTokenTile, n - t0);
  const uint4* r = reinterpret_cast<const uint4*>(w + (uint64_t)row * k);
  float acc[kTokenTile] = {};
  for (uint32_t i = lane; i < k / 8; i += warpSize) {
    const uint4 packed = r[i];
    const uint32_t words[4] = {packed.x, packed.y, packed.z, packed.w};
    float wv[8];
#pragma unroll
    for (uint32_t e = 0; e < 4; e++) {
      wv[2 * e] = __uint_as_float(words[e] << 16);
      wv[2 * e + 1] = __uint_as_float(words[e] & 0xffff0000u);
    }
#pragma unroll
    for (uint32_t j = 0; j < kTokenTile; j++) {
      if (j < tn) {
        const float4* xv = reinterpret_cast<const float4*>(x + (uint64_t)(t0 + j) * k) + 2 * i;
        const float4 a = xv[0], b = xv[1];
        acc[j] += wv[0] * a.x + wv[1] * a.y + wv[2] * a.z + wv[3] * a.w + wv[4] * b.x + wv[5] * b.y +
                  wv[6] * b.z + wv[7] * b.w;
      }
    }
  }
#pragma unroll
  for (uint32_t j = 0; j < kTokenTile; j++) {
    float v = acc[j];
    for (int o = warpSize / 2; o > 0; o >>= 1) v += __shfl_xor(v, o);
    if (j < tn && lane == 0) out[(uint64_t)(t0 + j) * rows + row] = v;
  }
}

/* Batched BF16 matmul on the matrix cores. Each FP32 activation splits into
 * three bfloat16 terms (hi + mid + lo, rounded in turn: together they carry
 * the full FP32 mantissa), so every product with a BF16 weight is exact. The
 * matrix-core accumulator loses precision over long chains of adds (2e-5
 * relative over k = 5120), so each K step of 32 starts from zero, takes the
 * three terms smallest first, and is added to FP32 registers: about 8e-7
 * relative to the largest output, against 2.5e-7 for the FP32 kernel. Block
 * tile 64 weight rows x 64 tokens, 8 waves of 16 x 32; rows a multiple of
 * 64 and k of 32. */
constexpr uint32_t kBf16Tile = 64, kBf16K = 32, kBf16Pitch = kBf16K + 8;
__device__ __forceinline__ uint32_t bf16_rn_bits(float x) {
  const uint32_t u = __float_as_uint(x);
  return (u + 0x7fffu + ((u >> 16) & 1u)) >> 16;
}
__global__ __launch_bounds__(256) void matmul_bf16_wmma_kernel(const uint16_t* w, const float* x, float* out,
                                                               uint32_t rows, uint32_t k, uint32_t n) {
  __shared__ uint16_t w_sh[kBf16Tile * kBf16Pitch];
  __shared__ uint16_t x_sh[3][kBf16Tile * kBf16Pitch];
  const uint32_t tid = threadIdx.x, wave = tid / 32, lane = tid % 32;
  const uint32_t r0 = blockIdx.x * kBf16Tile, t0 = blockIdx.y * kBf16Tile;
  /* Wave tile: weight rows wr .. wr + 15, tokens wt .. wt + 31. */
  const uint32_t wr = (wave % 4) * 16, wt = (wave / 4) * 32;
  using frag_a = rocwmma::fragment<rocwmma::matrix_a, 16, 16, 16, hip_bfloat16, rocwmma::row_major>;
  using frag_b = rocwmma::fragment<rocwmma::matrix_b, 16, 16, 16, hip_bfloat16, rocwmma::col_major>;
  using frag_c = rocwmma::fragment<rocwmma::accumulator, 16, 16, 16, float>;
  float s0[8] = {}, s1[8] = {};
  for (uint32_t k0 = 0; k0 < k; k0 += kBf16K) {
    frag_c c0, c1;
    rocwmma::fill_fragment(c0, 0.0f);
    rocwmma::fill_fragment(c1, 0.0f);
    {
      /* 64 rows x 32 values of W: one uint4 (8 values) per thread. */
      const uint32_t r = tid / 4, c = (tid % 4) * 8;
      *reinterpret_cast<uint4*>(&w_sh[r * kBf16Pitch + c]) =
          *reinterpret_cast<const uint4*>(w + (uint64_t)(r0 + r) * k + k0 + c);
    }
    for (uint32_t i = tid; i < kBf16Tile * kBf16K / 4; i += 256) {
      const uint32_t t = i / (kBf16K / 4), c = (i % (kBf16K / 4)) * 4;
      float4 v = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
      if (t0 + t < n) v = *reinterpret_cast<const float4*>(x + (uint64_t)(t0 + t) * k + k0 + c);
      const float in[4] = {v.x, v.y, v.z, v.w};
      uint32_t hi[4], mid[4], lo[4];
#pragma unroll
      for (int e = 0; e < 4; e++) {
        hi[e] = bf16_rn_bits(in[e]);
        const float r1 = in[e] - __uint_as_float(hi[e] << 16);
        mid[e] = bf16_rn_bits(r1);
        lo[e] = bf16_rn_bits(r1 - __uint_as_float(mid[e] << 16));
      }
      *reinterpret_cast<uint2*>(&x_sh[0][t * kBf16Pitch + c]) = make_uint2(hi[0] | hi[1] << 16, hi[2] | hi[3] << 16);
      *reinterpret_cast<uint2*>(&x_sh[1][t * kBf16Pitch + c]) =
          make_uint2(mid[0] | mid[1] << 16, mid[2] | mid[3] << 16);
      *reinterpret_cast<uint2*>(&x_sh[2][t * kBf16Pitch + c]) = make_uint2(lo[0] | lo[1] << 16, lo[2] | lo[3] << 16);
    }
    __syncthreads();
#pragma unroll
    for (uint32_t kk = 0; kk < kBf16K; kk += 16) {
      frag_a a;
      rocwmma::load_matrix_sync(a, reinterpret_cast<const hip_bfloat16*>(w_sh + wr * kBf16Pitch + kk), kBf16Pitch);
#pragma unroll
      for (int part = 2; part >= 0; part--) {
        frag_b b0, b1;
        rocwmma::load_matrix_sync(b0, reinterpret_cast<const hip_bfloat16*>(x_sh[part] + wt * kBf16Pitch + kk),
                                  kBf16Pitch);
        rocwmma::load_matrix_sync(b1, reinterpret_cast<const hip_bfloat16*>(x_sh[part] + (wt + 16) * kBf16Pitch + kk),
                                  kBf16Pitch);
        rocwmma::mma_sync(c0, a, b0, c0);
        rocwmma::mma_sync(c1, a, b1, c1);
      }
    }
#pragma unroll
    for (uint32_t e = 0; e < 8; e++) {
      s0[e] += c0.x[e];
      s1[e] += c1.x[e];
    }
    __syncthreads();
  }
  /* Accumulator element e of a lane is (row 2 * e + lane / 16, token lane % 16). */
#pragma unroll
  for (uint32_t e = 0; e < 8; e++) {
    const uint32_t row = r0 + wr + 2 * e + lane / 16;
    const uint32_t ta = t0 + wt + lane % 16, tb = ta + 16;
    if (ta < n) out[(uint64_t)ta * rows + row] = s0[e];
    if (tb < n) out[(uint64_t)tb * rows + row] = s1[e];
  }
}

__global__ void copy_rows_kernel(float* dst, uint64_t dst_stride, const float* src, uint64_t src_stride,
                                 uint32_t cols) {
  const float* s = src + (uint64_t)blockIdx.x * src_stride;
  float* d = dst + (uint64_t)blockIdx.x * dst_stride;
  for (uint32_t i = threadIdx.x; i < cols; i += blockDim.x) d[i] = s[i];
}

__global__ void rms_norm_kernel(const float* x, const float* weight, float* out, uint32_t n, float eps) {
  const float* r = x + (uint64_t)blockIdx.x * n;
  float* o = out + (uint64_t)blockIdx.x * n;
  float ss = 0.0f;
  for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) ss += r[i] * r[i];
  const float scale = rsqrtf(block_sum(ss) / (float)n + eps);
  for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) o[i] = (weight ? weight[i] : 1.0f) * (r[i] * scale);
}

__global__ void rope_tail_kernel(float* x, uint32_t rows, uint32_t row_dim, uint32_t rope_dim,
                                 const float* inv_freq, uint32_t pos0, uint32_t rows_per_pos, uint32_t pos_step,
                                 int inverse) {
  const uint32_t half = rope_dim / 2;
  const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (uint64_t)rows * half) return;
  const uint32_t row = (uint32_t)(i / half);
  const uint32_t pair = (uint32_t)(i % half);
  const uint32_t pos = pos0 + (row / rows_per_pos) * pos_step;
  float* p = x + (uint64_t)row * row_dim + (row_dim - rope_dim) + 2 * pair;
  const float angle = (float)pos * inv_freq[pair];
  const float c = cosf(angle);
  const float s = inverse ? -sinf(angle) : sinf(angle);
  const float re = p[0];
  const float im = p[1];
  p[0] = re * c - im * s;
  p[1] = re * s + im * c;
}

/* The 32 values at column col of a row at position pos: the rope_tail
 * rotation of their pairs when they lie in the rope tail, then the FP8 fake
 * quantization, stored to out. */
__device__ __forceinline__ void rope_fp8_block(float (&v)[32], float* out, uint32_t col, uint32_t row_dim,
                                               uint32_t rope_dim, const float* inv_freq, uint32_t pos, int inverse) {
  if (col >= row_dim - rope_dim) {
    const uint32_t pair0 = (col - (row_dim - rope_dim)) / 2;
#pragma unroll
    for (int k = 0; k < 16; k++) {
      const float angle = (float)pos * inv_freq[pair0 + k];
      const float c = cosf(angle);
      const float s = inverse ? -sinf(angle) : sinf(angle);
      const float re = v[2 * k];
      const float im = v[2 * k + 1];
      v[2 * k] = re * c - im * s;
      v[2 * k + 1] = re * s + im * c;
    }
  }
  float amax = 0.0f;
#pragma unroll
  for (int i = 0; i < 32; i++) amax = fmaxf(amax, fabsf(v[i]));
  const float scale = pow2_ceil(fmaxf(amax, 1.0e-4f) / 448.0f);
#pragma unroll
  for (int i = 0; i < 32; i++) out[i] = round_e4m3(fminf(fmaxf(v[i] / scale, -448.0f), 448.0f)) * scale;
}

/* Thread per 32-value block. */
__global__ void rope_fp8_kernel(float* x, uint32_t blocks, uint32_t row_dim, uint32_t rope_dim,
                                const float* inv_freq, uint32_t pos0, uint32_t rows_per_pos, uint32_t pos_step,
                                int inverse) {
  const uint32_t b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= blocks) return;
  const uint32_t per_row = row_dim / 32;
  const uint32_t row = b / per_row;
  float* p = x + (uint64_t)b * 32;
  float v[32];
#pragma unroll
  for (int i = 0; i < 32; i++) v[i] = p[i];
  rope_fp8_block(v, p, (b % per_row) * 32, row_dim, rope_dim, inv_freq, pos0 + (row / rows_per_pos) * pos_step,
                 inverse);
}

/* Block (part, token): part 0 writes qr = RMSNorm(qa) * q_weight; part 1
 * writes kvn = RMSNorm(kv) * kv_weight followed by rope_fp8 at position
 * pos0 + token. The norm loops are rms_norm_kernel's, so the values match
 * the separate launches; the KV row passes through shared memory to its
 * 32-value blocks. */
constexpr uint32_t kQkvThreads = 256;
constexpr uint32_t kQkvMaxKv = 1024;
__global__ void __launch_bounds__(kQkvThreads)
    q_kv_norm_kernel(const float* qa, const float* q_weight, float* qr, uint32_t q_dim, const float* kv,
                     const float* kv_weight, float* kvn, uint32_t kv_dim, uint32_t rope_dim, const float* inv_freq,
                     uint32_t pos0, float eps) {
  const uint32_t token = blockIdx.y;
  const bool is_kv = blockIdx.x == 1;
  const uint32_t n = is_kv ? kv_dim : q_dim;
  const float* r = (is_kv ? kv : qa) + (uint64_t)token * n;
  const float* weight = is_kv ? kv_weight : q_weight;
  float ss = 0.0f;
  for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) ss += r[i] * r[i];
  const float scale = rsqrtf(block_sum(ss) / (float)n + eps);
  if (!is_kv) {
    float* o = qr + (uint64_t)token * n;
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) o[i] = weight[i] * (r[i] * scale);
    return;
  }
  __shared__ float row[kQkvMaxKv];
  for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) row[i] = weight[i] * (r[i] * scale);
  __syncthreads();
  const uint32_t col = threadIdx.x * 32;
  if (col >= n) return;
  float b[32];
#pragma unroll
  for (int i = 0; i < 32; i++) b[i] = row[col + i];
  rope_fp8_block(b, kvn + (uint64_t)token * n + col, col, n, rope_dim, inv_freq, pos0 + token, 0);
}

/* With codes set, the quantized values are packed there (ds41_fp4_rows) and
 * x is left unchanged. */
__global__ void fp4_fake_quant_kernel(float* x, uint8_t* codes, uint8_t* scales, uint32_t blocks, uint32_t block,
                                      int e4m3_scale) {
  const uint32_t b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= blocks) return;
  float* v = x + (uint64_t)b * block;
  float amax = 0.0f;
  for (uint32_t i = 0; i < block; i++) amax = fmaxf(amax, fabsf(v[i]));
  const float scale = e4m3_scale ? round_e4m3(fmaxf(amax, 6.0f * 0.001953125f) / 6.0f)
                                 : pow2_ceil(fmaxf(amax, 6.0f * 1.1754943508e-38f) / 6.0f);
  if (codes) scales[b] = (uint8_t)(e4m3_scale ? e4m3_byte(scale) : __float_as_uint(scale) >> 23);
  for (uint32_t i = 0; i < block; i += 2) {
    const float r0 = round_e2m1(fminf(fmaxf(v[i] / scale, -6.0f), 6.0f));
    const float r1 = round_e2m1(fminf(fmaxf(v[i + 1] / scale, -6.0f), 6.0f));
    if (codes) {
      codes[((uint64_t)b * block + i) / 2] = (uint8_t)(e2m1_code(r0) | e2m1_code(r1) << 4);
    } else {
      v[i] = r0 * scale;
      v[i + 1] = r1 * scale;
    }
  }
}

/* Block (mix, token). */
__global__ void hc_mixes_kernel(const float* fn, const float* h, float* mixes, uint32_t n, uint32_t mixes_n,
                                float eps) {
  const float* r = fn + (uint64_t)blockIdx.x * n;
  const float* x = h + (uint64_t)blockIdx.y * n;
  float dot = 0.0f;
  float ss = 0.0f;
  for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
    dot += r[i] * x[i];
    ss += x[i] * x[i];
  }
  dot = block_sum(dot);
  ss = block_sum(ss);
  if (threadIdx.x == 0) mixes[(uint64_t)blockIdx.y * mixes_n + blockIdx.x] = dot * rsqrtf(ss / (float)n + eps);
}

/* Batched form: a block takes kMixTokens tokens and all MIX rows, so each
 * token's state is read once (the per-row form reads it MIX times). 256
 * threads; n % 4 == 0. */
constexpr uint32_t kMixTokens = 2;
template <uint32_t MIX>
__global__ __launch_bounds__(256) void hc_mixes_rows_kernel(const float* fn, const float* h, float* mixes,
                                                            uint32_t n, uint32_t tokens, float eps) {
  constexpr uint32_t T = kMixTokens, V = MIX * T + T;
  const uint32_t t0 = blockIdx.x * T;
  const uint32_t n4 = n / 4;
  float acc[V] = {};
  for (uint32_t i = threadIdx.x; i < n4; i += blockDim.x) {
    float4 x[T];
#pragma unroll
    for (uint32_t t = 0; t < T; t++) {
      x[t] = t0 + t < tokens ? reinterpret_cast<const float4*>(h + (uint64_t)(t0 + t) * n)[i]
                             : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
      acc[MIX * T + t] += x[t].x * x[t].x + x[t].y * x[t].y + x[t].z * x[t].z + x[t].w * x[t].w;
    }
#pragma unroll
    for (uint32_t j = 0; j < MIX; j++) {
      const float4 w = reinterpret_cast<const float4*>(fn + (uint64_t)j * n)[i];
#pragma unroll
      for (uint32_t t = 0; t < T; t++) acc[j * T + t] += w.x * x[t].x + w.y * x[t].y + w.z * x[t].z + w.w * x[t].w;
    }
  }
  __shared__ float partial[V][8];
  const uint32_t lane = threadIdx.x % 32, warp = threadIdx.x / 32;
#pragma unroll
  for (uint32_t v = 0; v < V; v++) {
    float a = acc[v];
    for (int o = 16; o > 0; o >>= 1) a += __shfl_xor(a, o, 32);
    if (lane == 0) partial[v][warp] = a;
  }
  __syncthreads();
  for (uint32_t v = threadIdx.x; v < MIX * T; v += blockDim.x) {
    const uint32_t t = v % T;
    if (t0 + t >= tokens) continue;
    float dot = 0.0f, ss = 0.0f;
    for (uint32_t w = 0; w < 8; w++) {
      dot += partial[v][w];
      ss += partial[MIX * T + t][w];
    }
    mixes[(uint64_t)(t0 + t) * MIX + v / T] = dot * rsqrtf(ss / (float)n + eps);
  }
}

/* Batched pass over the mHC state (hc = 4) of kMixTokens tokens per block,
 * each thread taking two 4-channel groups of every copy: with kExpand the
 * state is first expanded, s = post y + comb^T h, and stored to h_out;
 * then the mixes of s (as hc_mixes_rows_kernel) and the collapse
 * weight * RMSNorm(sum_c pre_c s_c) come from the same registers, so the
 * state is read once. */
template <bool kExpand>
__global__ __launch_bounds__(1024) void hc_state_pass_kernel(const float* y, const float* post, const float* comb,
                                                             const float* h, float* h_out, const float* pre,
                                                             const float* norm_w, float* xn, const float* fn,
                                                             float* mixes, uint32_t dim, uint32_t tokens, float eps) {
  constexpr uint32_t T = kMixTokens, HC = 4, MIX = 24, G = 2;
  constexpr uint32_t V = T * (MIX + 2);  // per token: the mix dots, the state and the collapse sums of squares
  const uint32_t t0 = blockIdx.x * T, groups = dim / 4, n = HC * dim;
  float4 st[T][G][HC], col[T][G];
  float acc[2 * T] = {};  // the state and the collapse sums of squares
#pragma unroll
  for (uint32_t t = 0; t < T; t++) {
    const bool live = t0 + t < tokens;
    const uint64_t tok = live ? t0 + t : 0;
    const float4* ht = reinterpret_cast<const float4*>(h + tok * n);
#pragma unroll
    for (uint32_t g = 0; g < G; g++) {
      const uint32_t q = threadIdx.x + g * blockDim.x;
      if constexpr (kExpand) {
        const float4 yv = reinterpret_cast<const float4*>(y + tok * dim)[q];
        float4 hv[HC];
#pragma unroll
        for (uint32_t a = 0; a < HC; a++) hv[a] = ht[a * groups + q];
#pragma unroll
        for (uint32_t c = 0; c < HC; c++) {
          const float pc = post[tok * HC + c];
          float4 v = make_float4(pc * yv.x, pc * yv.y, pc * yv.z, pc * yv.w);
#pragma unroll
          for (uint32_t a = 0; a < HC; a++) {
            const float w = comb[tok * HC * HC + a * HC + c];
            v.x += w * hv[a].x, v.y += w * hv[a].y, v.z += w * hv[a].z, v.w += w * hv[a].w;
          }
          st[t][g][c] = v;
          if (live) reinterpret_cast<float4*>(h_out + tok * n)[c * groups + q] = v;
        }
      } else {
#pragma unroll
        for (uint32_t c = 0; c < HC; c++) st[t][g][c] = ht[c * groups + q];
      }
      if (!live) {
#pragma unroll
        for (uint32_t c = 0; c < HC; c++) st[t][g][c] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
      }
      float4 v = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
#pragma unroll
      for (uint32_t c = 0; c < HC; c++) {
        const float pc = live ? pre[tok * HC + c] : 0.0f;
        const float4 x = st[t][g][c];
        v.x += pc * x.x, v.y += pc * x.y, v.z += pc * x.z, v.w += pc * x.w;
        acc[t] += x.x * x.x + x.y * x.y + x.z * x.z + x.w * x.w;
      }
      col[t][g] = v;
      acc[T + t] += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }
  }
  __shared__ float partial[V][32];
  __shared__ float total[V];
  const uint32_t lane = threadIdx.x % 32, warp = threadIdx.x / 32, warps = blockDim.x / 32;
  /* One copy of the mix body (the kernel's code stays within the
   * instruction cache), each mix reduced across the wave in turn. */
#pragma unroll 1
  for (uint32_t j = 0; j < MIX; j++) {
    float dot[T] = {};
#pragma unroll
    for (uint32_t g = 0; g < G; g++)
#pragma unroll
      for (uint32_t c = 0; c < HC; c++) {
        const float4 w =
            reinterpret_cast<const float4*>(fn + (uint64_t)j * n)[c * groups + threadIdx.x + g * blockDim.x];
#pragma unroll
        for (uint32_t t = 0; t < T; t++) {
          const float4 x = st[t][g][c];
          dot[t] += w.x * x.x + w.y * x.y + w.z * x.z + w.w * x.w;
        }
      }
#pragma unroll
    for (uint32_t t = 0; t < T; t++) {
      float a = dot[t];
      for (int o = 16; o > 0; o >>= 1) a += __shfl_xor(a, o, 32);
      if (lane == 0) partial[j * T + t][warp] = a;
    }
  }
#pragma unroll
  for (uint32_t v = MIX * T; v < V; v++) {
    float a = acc[v - MIX * T];
    for (int o = 16; o > 0; o >>= 1) a += __shfl_xor(a, o, 32);
    if (lane == 0) partial[v][warp] = a;
  }
  __syncthreads();
  for (uint32_t v = threadIdx.x; v < V; v += blockDim.x) {
    float sum = 0.0f;
    for (uint32_t w = 0; w < warps; w++) sum += partial[v][w];
    total[v] = sum;
  }
  __syncthreads();
  for (uint32_t v = threadIdx.x; v < MIX * T; v += blockDim.x) {
    const uint32_t t = v % T;
    if (t0 + t < tokens)
      mixes[(uint64_t)(t0 + t) * MIX + v / T] = total[v] * rsqrtf(total[MIX * T + t] / (float)n + eps);
  }
#pragma unroll
  for (uint32_t t = 0; t < T; t++) {
    if (t0 + t >= tokens) continue;
    const float scale = rsqrtf(total[MIX * T + T + t] / (float)dim + eps);
#pragma unroll
    for (uint32_t g = 0; g < G; g++) {
      const uint32_t q = threadIdx.x + g * blockDim.x;
      const float4 w = reinterpret_cast<const float4*>(norm_w)[q], v = col[t][g];
      reinterpret_cast<float4*>(xn + (uint64_t)(t0 + t) * dim)[q] =
          make_float4(w.x * (v.x * scale), w.y * (v.y * scale), w.z * (v.z * scale), w.w * (v.w * scale));
    }
  }
}

__device__ __forceinline__ float sigmoidf(float x) { return 1.0f / (1.0f + expf(-x)); }

/* Thread per token. */
/* hc is a template constant so the matrix stays in registers; a runtime
 * bound puts it in scratch memory (90 us per call at hc = 4). */
template <uint32_t hc>
__device__ __forceinline__ void sinkhorn_token(const float* mixes, const float* scale, const float* base, float* pre,
                                               float* post, float* comb, uint32_t iters, float eps) {
  float c[hc][hc];
#pragma unroll
  for (uint32_t i = 0; i < hc; i++) {
    pre[i] = sigmoidf(mixes[i] * scale[0] + base[i]) + eps;
    post[i] = 2.0f * sigmoidf(mixes[hc + i] * scale[1] + base[hc + i]);
  }
#pragma unroll
  for (uint32_t a = 0; a < hc; a++) {
    float m = -INFINITY;
#pragma unroll
    for (uint32_t b = 0; b < hc; b++) {
      c[a][b] = mixes[2 * hc + a * hc + b] * scale[2] + base[2 * hc + a * hc + b];
      m = fmaxf(m, c[a][b]);
    }
    float sum = 0.0f;
#pragma unroll
    for (uint32_t b = 0; b < hc; b++) {
      c[a][b] = expf(c[a][b] - m);
      sum += c[a][b];
    }
#pragma unroll
    for (uint32_t b = 0; b < hc; b++) c[a][b] = c[a][b] / sum + eps;
  }
  auto normalize_columns = [&] {
#pragma unroll
    for (uint32_t b = 0; b < hc; b++) {
      float sum = 0.0f;
#pragma unroll
      for (uint32_t a = 0; a < hc; a++) sum += c[a][b];
#pragma unroll
      for (uint32_t a = 0; a < hc; a++) c[a][b] /= sum + eps;
    }
  };
  normalize_columns();
  for (uint32_t it = 1; it < iters; it++) {
#pragma unroll
    for (uint32_t a = 0; a < hc; a++) {
      float sum = 0.0f;
#pragma unroll
      for (uint32_t b = 0; b < hc; b++) sum += c[a][b];
#pragma unroll
      for (uint32_t b = 0; b < hc; b++) c[a][b] /= sum + eps;
    }
    normalize_columns();
  }
#pragma unroll
  for (uint32_t a = 0; a < hc; a++)
#pragma unroll
    for (uint32_t b = 0; b < hc; b++) comb[a * hc + b] = c[a][b];
}

/* Batched form. It keeps its own copy of sinkhorn_token's arithmetic: under
 * fast-math, calling the shared function rounds differently and moved the
 * batched 3K scores. */
template <uint32_t hc>
__global__ void hc_sinkhorn_kernel(const float* all_mixes, const float* scale, const float* base, float* all_pre,
                                   float* all_post, float* all_comb, uint32_t iters, float eps, uint32_t n) {
  const uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= n) return;
  const float* mixes = all_mixes + (uint64_t)t * (2 * hc + hc * hc);
  float* pre = all_pre + (uint64_t)t * hc;
  float* post = all_post + (uint64_t)t * hc;
  float* comb = all_comb + (uint64_t)t * hc * hc;
  float c[hc][hc];
#pragma unroll
  for (uint32_t i = 0; i < hc; i++) {
    pre[i] = sigmoidf(mixes[i] * scale[0] + base[i]) + eps;
    post[i] = 2.0f * sigmoidf(mixes[hc + i] * scale[1] + base[hc + i]);
  }
#pragma unroll
  for (uint32_t a = 0; a < hc; a++) {
    float m = -INFINITY;
#pragma unroll
    for (uint32_t b = 0; b < hc; b++) {
      c[a][b] = mixes[2 * hc + a * hc + b] * scale[2] + base[2 * hc + a * hc + b];
      m = fmaxf(m, c[a][b]);
    }
    float sum = 0.0f;
#pragma unroll
    for (uint32_t b = 0; b < hc; b++) {
      c[a][b] = expf(c[a][b] - m);
      sum += c[a][b];
    }
#pragma unroll
    for (uint32_t b = 0; b < hc; b++) c[a][b] = c[a][b] / sum + eps;
  }
  auto normalize_columns = [&] {
#pragma unroll
    for (uint32_t b = 0; b < hc; b++) {
      float sum = 0.0f;
#pragma unroll
      for (uint32_t a = 0; a < hc; a++) sum += c[a][b];
#pragma unroll
      for (uint32_t a = 0; a < hc; a++) c[a][b] /= sum + eps;
    }
  };
  normalize_columns();
  for (uint32_t it = 1; it < iters; it++) {
#pragma unroll
    for (uint32_t a = 0; a < hc; a++) {
      float sum = 0.0f;
#pragma unroll
      for (uint32_t b = 0; b < hc; b++) sum += c[a][b];
#pragma unroll
      for (uint32_t b = 0; b < hc; b++) c[a][b] /= sum + eps;
    }
    normalize_columns();
  }
#pragma unroll
  for (uint32_t a = 0; a < hc; a++)
#pragma unroll
    for (uint32_t b = 0; b < hc; b++) comb[a * hc + b] = c[a][b];
}

/* Small batches: each mix row of each token is split over kMixSlices blocks
 * with 16-byte loads, unrolled so a thread's loads are in flight together.
 * Blocks leave partial sums in g_mix_dot and g_mix_ss; the last block to finish a
 * token (counted in g_mix_done, which it resets) reduces them and runs that
 * token's Sinkhorn. */
constexpr uint32_t kMixSmallTokens = 16;
constexpr uint32_t kMixSmallThreads = 256;
constexpr uint32_t kMixSlices = 8;
constexpr uint32_t kMixMax = 24;
__device__ uint32_t g_mix_done[kMixSmallTokens];
__device__ float g_mix_dot[kMixSmallTokens][kMixMax * kMixSlices];
__device__ float g_mix_ss[kMixSmallTokens][kMixMax * kMixSlices];

template <uint32_t hc>
__global__ void __launch_bounds__(kMixSmallThreads)
    hc_mix_small_kernel(const float* fn, const float* h, const float* scale, const float* base, float* all_mixes,
                        float* all_pre, float* all_post, float* all_comb, uint32_t n, float eps, uint32_t iters,
                        float hc_eps) {
  constexpr uint32_t kMix = 2 * hc + hc * hc;
  static_assert(kMix <= kMixMax, "mix rows");
  const uint32_t t = blockIdx.y;
  const uint32_t row = blockIdx.x / kMixSlices;
  const uint32_t slice = blockIdx.x % kMixSlices;
  const uint32_t n4 = n / 4;
  const uint32_t per = (n4 + kMixSlices - 1) / kMixSlices;
  const uint32_t end = min(n4, (slice + 1) * per);
  const float4* r = reinterpret_cast<const float4*>(fn + (uint64_t)row * n);
  const float4* x = reinterpret_cast<const float4*>(h + (uint64_t)t * n);
  float dot = 0.0f;
  float ss = 0.0f;
#pragma unroll 4
  for (uint32_t i = slice * per + threadIdx.x; i < end; i += kMixSmallThreads) {
    const float4 w = r[i];
    const float4 v = x[i];
    dot += w.x * v.x + w.y * v.y + w.z * v.z + w.w * v.w;
    ss += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
  }
  dot = block_sum(dot);
  ss = block_sum(ss);
  __shared__ bool last;
  if (threadIdx.x == 0) {
    g_mix_dot[t][blockIdx.x] = dot;
    g_mix_ss[t][blockIdx.x] = ss;
    __threadfence();
    last = atomicAdd(&g_mix_done[t], 1u) == kMix * kMixSlices - 1;
  }
  __syncthreads();
  if (!last) return;
  __threadfence();
  __shared__ float mixes[kMix];
  if (threadIdx.x < kMix) {
    float d = 0.0f, q = 0.0f;
#pragma unroll
    for (uint32_t k = 0; k < kMixSlices; k++) {
      d += __hip_atomic_load(&g_mix_dot[t][threadIdx.x * kMixSlices + k], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
      q += __hip_atomic_load(&g_mix_ss[t][k], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    }
    mixes[threadIdx.x] = d * rsqrtf(q / (float)n + eps);
    all_mixes[(uint64_t)t * kMix + threadIdx.x] = mixes[threadIdx.x];
  }
  __syncthreads();
  if (threadIdx.x != 0) return;
  g_mix_done[t] = 0;
  sinkhorn_token<hc>(mixes, scale, base, all_pre + (uint64_t)t * hc, all_post + (uint64_t)t * hc,
                     all_comb + (uint64_t)t * hc * hc, iters, hc_eps);
}

/* Block per token; each thread keeps its kPer collapsed values in registers
 * between the sum of squares and the scaled store, so every load of a row is
 * issued up front instead of one per loop iteration. */
constexpr uint32_t kCollapseThreads = 1024;
constexpr uint32_t kCollapsePer = 8;

template <uint32_t kHcN>
__global__ void __launch_bounds__(kCollapseThreads)
    hc_collapse_norm_kernel(const float* h, const float* pre, const float* weight, float* out, uint32_t dim,
                            float eps) {
  const uint64_t t = blockIdx.x;
  const float* ht = h + t * kHcN * dim;
  float p[kHcN];
#pragma unroll
  for (uint32_t c = 0; c < kHcN; c++) p[c] = pre[t * kHcN + c];
  float v[kCollapsePer];
  float ss = 0.0f;
#pragma unroll
  for (uint32_t j = 0; j < kCollapsePer; j++) {
    const uint32_t i = threadIdx.x + j * kCollapseThreads;
    float acc = 0.0f;
    if (i < dim) {
#pragma unroll
      for (uint32_t c = 0; c < kHcN; c++) acc += p[c] * ht[(uint64_t)c * dim + i];
    }
    v[j] = acc;
    ss += acc * acc;
  }
  const float scale = rsqrtf(block_sum(ss) / (float)dim + eps);
#pragma unroll
  for (uint32_t j = 0; j < kCollapsePer; j++) {
    const uint32_t i = threadIdx.x + j * kCollapseThreads;
    if (i < dim) out[t * dim + i] = weight[i] * (v[j] * scale);
  }
}

/* DSpark acceptance head for one row: sigmoid(w . [collapse(h, pre); m] + b),
 * the collapse before its RMSNorm and m the row's Markov W1 input. */
__global__ void __launch_bounds__(kCollapseThreads)
    dspark_confidence_kernel(const float* h, const float* pre, const float* markov_x, const float* w,
                             const float* bias, uint32_t dim, uint32_t rank, uint32_t row, float* out) {
  const float* ht = h + (uint64_t)row * 4 * dim;
  float p[4];
#pragma unroll
  for (uint32_t c = 0; c < 4; c++) p[c] = pre[row * 4 + c];
  float acc = 0.0f;
  for (uint32_t i = threadIdx.x; i < dim; i += blockDim.x) {
    float v = 0.0f;
#pragma unroll
    for (uint32_t c = 0; c < 4; c++) v += p[c] * ht[(uint64_t)c * dim + i];
    acc += v * w[i];
  }
  for (uint32_t i = threadIdx.x; i < rank; i += blockDim.x) acc += markov_x[i] * w[dim + i];
  acc = block_sum(acc);
  if (threadIdx.x == 0) out[row] = 1.0f / (1.0f + expf(-(acc + (bias ? *bias : 0.0f))));
}

/* hc_expand of y = extra + sum_r x[r]. */
__global__ void hc_expand_sum_kernel(const float* x, const float* extra, uint32_t rows, const float* residual,
                                     const float* post, const float* comb, float* out, uint32_t hc, uint32_t dim) {
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= dim) return;
  const uint64_t t = blockIdx.y;
  float y = extra ? extra[t * dim + i] : 0.0f;
  for (uint32_t r = 0; r < rows; r++) y += x[(t * rows + r) * dim + i];
  const float* rt = residual + t * hc * dim;
  const float* ct = comb + t * hc * hc;
  float* ot = out + t * hc * dim;
  for (uint32_t b = 0; b < hc; b++) {
    float acc = post[t * hc + b] * y;
    for (uint32_t a = 0; a < hc; a++) acc += ct[a * hc + b] * rt[(uint64_t)a * dim + i];
    ot[(uint64_t)b * dim + i] = acc;
  }
}

__global__ void hc_expand_kernel(const float* y, const float* residual, const float* post, const float* comb,
                                 float* out, uint32_t hc, uint32_t dim) {
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= dim) return;
  const uint64_t t = blockIdx.y;
  const float* rt = residual + t * hc * dim;
  const float* ct = comb + t * hc * hc;
  float* ot = out + t * hc * dim;
  for (uint32_t b = 0; b < hc; b++) {
    float acc = post[t * hc + b] * y[t * dim + i];
    for (uint32_t a = 0; a < hc; a++) acc += ct[a * hc + b] * rt[(uint64_t)a * dim + i];
    ot[(uint64_t)b * dim + i] = acc;
  }
}

/* Thread per channel; tokens in order, since groups span the frontier. */
__global__ void compress_pool_kernel(const float* kv, const float* score, float* state_kv, float* state_score,
                                     uint32_t pos0, uint32_t n, uint32_t ratio, float* out, uint32_t dim) {
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= dim) return;
  uint32_t emitted = 0;
  for (uint32_t t = 0; t < n; t++) {
    const uint32_t pos = pos0 + t;
    const uint32_t slot = pos % ratio;
    state_kv[(uint64_t)slot * dim + i] = kv[(uint64_t)t * dim + i];
    state_score[(uint64_t)slot * dim + i] = score[(uint64_t)t * dim + i];
    if ((pos + 1) % ratio != 0) continue;
    float m = -INFINITY;
    for (uint32_t s = 0; s < ratio; s++) m = fmaxf(m, state_score[(uint64_t)s * dim + i]);
    float sum = 0.0f;
    float acc = 0.0f;
    for (uint32_t s = 0; s < ratio; s++) {
      const float e = expf(state_score[(uint64_t)s * dim + i] - m);
      sum += e;
      acc += e * state_kv[(uint64_t)s * dim + i];
    }
    out[(uint64_t)emitted++ * dim + i] = acc / sum;
  }
}

/* Block (key tile, token); one thread per key; the query heads live in shared memory. */
__global__ void indexer_score_kernel(const float* q, ds41_fp4_rows keys, const float* w, float w_scale, float* score,
                                     uint64_t score_stride, uint32_t pos0, uint32_t ratio, uint32_t heads,
                                     uint32_t head_dim) {
  const uint32_t token = blockIdx.y;
  const uint32_t len = comp_len(pos0 + token, ratio);
  if (blockIdx.x * blockDim.x >= len) return;
  extern __shared__ float sq[];
  const float* qt = q + (uint64_t)token * heads * head_dim;
  for (uint32_t i = threadIdx.x; i < heads * head_dim; i += blockDim.x) sq[i] = qt[i];
  __shared__ float sw[64];
  for (uint32_t i = threadIdx.x; i < heads; i += blockDim.x) sw[i] = w[(uint64_t)token * heads + i] * w_scale;
  __syncthreads();
  const uint32_t j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= len) return;
  float acc = 0.0f;
  for (uint32_t h = 0; h < heads; h++) {
    float dot = 0.0f;
    for (uint32_t d = 0; d < head_dim; d++) dot += sq[h * head_dim + d] * ik_value(keys, j, d, head_dim);
    acc += fmaxf(dot, 0.0f) * sw[h];
  }
  score[token * score_stride + j] = acc;
}

/* Small batches with head_dim 128 and 32 heads: a block takes 64 keys,
 * staged in padded shared memory by coalesced loads; thread (key, group)
 * computes the fp32 dots of heads 8 * group .. 8 * group + 7 against the
 * queries (float4 broadcasts), and the four groups' relu-weighted sums are
 * added in shared memory. Grid (ceil(len / 64), n). */
constexpr uint32_t kScoreHeads = 32;
constexpr uint32_t kScoreSmallTokens = 4;
constexpr uint32_t kScoreKeys = 64;
template <bool Compact>
__global__ __launch_bounds__(256) void indexer_score_small_kernel(const float* q, ds41_fp4_rows keys, const float* w,
                                                                  float w_scale, float* score, uint64_t score_stride,
                                                                  uint32_t pos0, uint32_t ratio, const uint32_t* candidates,
                                                                  uint32_t candidate_stride) {
  constexpr uint32_t DIM = 128, D4 = DIM / 4, PITCH4 = D4 + 1, GROUPS = 256 / kScoreKeys,
                     HG = kScoreHeads / GROUPS;
  const uint32_t token = blockIdx.y;
  const uint32_t len = comp_len(pos0 + token, ratio);
  const uint32_t key0 = blockIdx.x * kScoreKeys;
  const uint32_t slots = Compact ? min((len + 7) / 8, candidate_stride) * 8 : len;
  if (key0 >= slots) return;
  const auto source_key = [&](uint32_t key) {
    if constexpr (Compact) return candidates[(uint64_t)token * candidate_stride + key / 8] * 8 + key % 8;
    return key;
  };
  __shared__ float4 sk[kScoreKeys * PITCH4];
  __shared__ float4 sq[kScoreHeads * D4];
  __shared__ float sw[kScoreHeads];
  __shared__ float part[GROUPS][kScoreKeys];
  const float4* qt = reinterpret_cast<const float4*>(q + (uint64_t)token * kScoreHeads * DIM);
  for (uint32_t i = threadIdx.x; i < kScoreHeads * D4; i += 256) sq[i] = qt[i];
  const uint32_t rows = min(kScoreKeys, slots - key0);
  for (uint32_t i = threadIdx.x; i < kScoreKeys * D4; i += 256) {
    const uint32_t r = i / D4;
    sk[r * PITCH4 + i % D4] =
        r < rows && source_key(key0 + r) < len ? ik_load4(keys, source_key(key0 + r), (i % D4) * 4, DIM) : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  }
  if (threadIdx.x < kScoreHeads) sw[threadIdx.x] = w[(uint64_t)token * kScoreHeads + threadIdx.x] * w_scale;
  __syncthreads();
  const uint32_t key = threadIdx.x % kScoreKeys;
  const uint32_t group = threadIdx.x / kScoreKeys;
  float dot[HG] = {};
#pragma unroll 4
  for (uint32_t d4 = 0; d4 < D4; d4++) {
    const float4 kv = sk[key * PITCH4 + d4];
#pragma unroll
    for (uint32_t h = 0; h < HG; h++) {
      const float4 qv = sq[(group * HG + h) * D4 + d4];
      dot[h] += qv.x * kv.x + qv.y * kv.y + qv.z * kv.z + qv.w * kv.w;
    }
  }
  float acc = 0.0f;
#pragma unroll
  for (uint32_t h = 0; h < HG; h++) acc += fmaxf(dot[h], 0.0f) * sw[group * HG + h];
  part[group][key] = acc;
  __syncthreads();
  if (group != 0 || key >= rows) return;
#pragma unroll
  for (uint32_t g = 1; g < GROUPS; g++) acc += part[g][key];
  const uint32_t dest = source_key(key0 + key);
  if (dest < len) score[token * score_stride + dest] = acc;
}

/* The same scores on the matrix cores, after the V4 engine's
 * indexer_scores_wmma128_kernel: a block covers 16 tokens and 128 compressed
 * rows, stages the rows once as half and each head's queries (double
 * buffered), and keeps the relu-weighted head sum in fp32. head_dim 128,
 * 256 threads; grid (ceil(len / 128), ceil(n / 16)). */
__global__ __launch_bounds__(256) void indexer_score_wmma_kernel(const float* q, ds41_fp4_rows keys, const float* w,
                                                                 float w_scale, float* score, uint64_t score_stride,
                                                                 uint32_t pos0, uint32_t n, uint32_t ratio,
                                                                 uint32_t heads) {
  constexpr uint32_t DIM = 128, QT_PITCH = 18;
  const uint32_t tile_c = blockIdx.x * 128;
  const uint32_t tile_t = blockIdx.y * 16;
  const uint32_t tid = threadIdx.x;
  const uint32_t warp = tid / 32;
  const uint32_t lane = tid % 32;
  const uint32_t last = min(tile_t + 16, n);
  const uint32_t visible_max = comp_len(pos0 + last - 1, ratio);
  if (tile_c >= visible_max) return;

  __shared__ half q_sh[2][DIM * QT_PITCH];
  __shared__ half k_sh[128 * DIM];
  for (uint32_t i8 = tid; i8 < 128 * (DIM / 8); i8 += 256) {
    const uint32_t c = i8 / (DIM / 8);
    const uint32_t d = (i8 % (DIM / 8)) * 8;
    uint4 packed = make_uint4(0u, 0u, 0u, 0u);
    if (tile_c + c < visible_max) {
      float4 lo, hi;
      ik_load8(keys, tile_c + c, d, DIM, &lo, &hi);
      const uint2 a = half4_bits(lo), b = half4_bits(hi);
      packed = make_uint4(a.x, a.y, b.x, b.y);
    }
    *reinterpret_cast<uint4*>(&k_sh[c * DIM + d]) = packed;
  }
  /* Q^T of one head as [d][token]. */
  const auto stage_q = [&](uint32_t h, half* dst) {
    for (uint32_t i4 = tid; i4 < 16 * (DIM / 4); i4 += 256) {
      const uint32_t r = i4 / (DIM / 4);
      const uint32_t d = (i4 % (DIM / 4)) * 4;
      float4 v4 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
      if (tile_t + r < n) v4 = *reinterpret_cast<const float4*>(q + ((uint64_t)(tile_t + r) * heads + h) * DIM + d);
      dst[(d + 0) * QT_PITCH + r] = __float2half_rn(v4.x);
      dst[(d + 1) * QT_PITCH + r] = __float2half_rn(v4.y);
      dst[(d + 2) * QT_PITCH + r] = __float2half_rn(v4.z);
      dst[(d + 3) * QT_PITCH + r] = __float2half_rn(v4.w);
    }
  };
  stage_q(0, q_sh[0]);
  __syncthreads();

  /* Accumulator element e is compressed row 2 * e + lane / 16 of this wave's
   * 16 and token lane % 16 (gfx1151 wave32 layout). */
  const uint32_t token = tile_t + lane % 16;
  const uint32_t comp_base = tile_c + warp * 16 + lane / 16;
  float acc[8] = {};
  for (uint32_t h = 0; h < heads; h++) {
    const uint32_t cur = h & 1u;
    if (h + 1 < heads) stage_q(h + 1, q_sh[cur ^ 1u]);
    rocwmma::fragment<rocwmma::matrix_a, 16, 16, 16, half, rocwmma::row_major> a;
    rocwmma::fragment<rocwmma::matrix_b, 16, 16, 16, half, rocwmma::row_major> b;
    rocwmma::fragment<rocwmma::accumulator, 16, 16, 16, float> c;
    rocwmma::fill_fragment(c, 0.0f);
#pragma unroll
    for (uint32_t k0 = 0; k0 < DIM; k0 += 16) {
      rocwmma::load_matrix_sync(a, k_sh + warp * 16 * DIM + k0, DIM);
      rocwmma::load_matrix_sync(b, q_sh[cur] + k0 * QT_PITCH, QT_PITCH);
      rocwmma::mma_sync(c, a, b, c);
    }
    if (token < n) {
      const float wh = w[(uint64_t)token * heads + h] * w_scale;
#pragma unroll
      for (uint32_t e = 0; e < 8; e++) acc[e] += fmaxf(c.x[e], 0.0f) * wh;
    }
    __syncthreads();
  }
  if (token >= n) return;
  const uint32_t visible = comp_len(pos0 + token, ratio);
#pragma unroll
  for (uint32_t e = 0; e < 8; e++) {
    const uint32_t j = comp_base + 2 * e;
    if (j < visible) score[token * score_stride + j] = acc[e];
  }
}

__device__ __forceinline__ uint32_t ordered_key(float v) {
  uint32_t u = __float_as_uint(v);
  if ((u & 0x7fffffffu) == 0u) u = 0u;
  return (u & 0x80000000u) ? ~u : (u ^ 0x80000000u);
}

/* Block per token: binary search for the k-th largest key, then emit keys
 * above it and the lowest-index ties, both in index order. */
__global__ void topk_kernel(const float* all_score, uint64_t score_stride, uint32_t* all_out, uint32_t k_max,
                            uint32_t pos0, uint32_t ratio, uint32_t div) {
  __shared__ uint32_t hist[256];
  __shared__ uint32_t sel_prefix, sel_rank;
  const uint32_t token = blockIdx.x;
  const uint32_t n = (comp_len(pos0 + token, ratio) + div - 1) / div;
  const uint32_t k = min(k_max, n);
  if (k == 0) return;
  const float* score = all_score + token * score_stride;
  uint32_t* out = all_out + (uint64_t)token * k_max;
  const uint32_t per = (n + blockDim.x - 1) / blockDim.x;
  const uint32_t begin = min(n, threadIdx.x * per);
  const uint32_t end = min(n, begin + per);
  /* The k-th largest key, a byte per pass from the top: histogram the keys
   * that match the bytes chosen so far, then one wave finds the byte holding
   * the rank (lane L scans digits 255 - 8L down to 248 - 8L). */
  uint32_t prefix = 0u;
  uint32_t mask = 0u;
  uint32_t rank = k;
  for (int shift = 24; shift >= 0; shift -= 8) {
    for (uint32_t i = threadIdx.x; i < 256; i += blockDim.x) hist[i] = 0u;
    __syncthreads();
    for (uint32_t i = begin; i < end; i++) {
      const uint32_t key = ordered_key(score[i]);
      if ((key & mask) == prefix) atomicAdd(&hist[(key >> shift) & 255u], 1u);
    }
    __syncthreads();
    if (threadIdx.x < 32) {
      const uint32_t lane = threadIdx.x;
      uint32_t c[8];
      uint32_t sum = 0u;
#pragma unroll
      for (uint32_t j = 0; j < 8; j++) {
        c[j] = hist[255u - 8u * lane - j];
        sum += c[j];
      }
      uint32_t incl = sum;
#pragma unroll
      for (int o = 1; o < 32; o <<= 1) {
        const uint32_t up = __shfl_up(incl, o, 32);
        if ((int)lane >= o) incl += up;
      }
      uint32_t acc = incl - sum;
      if (acc < rank && rank <= incl) {
#pragma unroll
        for (uint32_t j = 0; j < 8; j++) {
          if (acc + c[j] >= rank) {
            sel_prefix = prefix | ((255u - 8u * lane - j) << shift);
            sel_rank = rank - acc;
            break;
          }
          acc += c[j];
        }
      }
    }
    __syncthreads();
    prefix = sel_prefix;
    rank = sel_rank;
    mask |= 255u << shift;
  }
  const uint32_t t = prefix;
  uint32_t above = 0;
  uint32_t equal = 0;
  for (uint32_t i = begin; i < end; i++) {
    const uint32_t key = ordered_key(score[i]);
    above += key > t;
    equal += key == t;
  }
  uint32_t total_above = 0;
  uint32_t total_equal = 0;
  uint32_t at = block_exclusive_scan(above, &total_above);
  uint32_t eq_at = block_exclusive_scan(equal, &total_equal);
  const uint32_t need = k - total_above;
  for (uint32_t i = begin; i < end; i++) {
    const uint32_t key = ordered_key(score[i]);
    if (key > t) {
      out[at++] = i;
    } else if (key == t) {
      if (eq_at < need) out[total_above + eq_at] = i;
      eq_at++;
    }
  }
}

/* The candidate mask holds one bit per block (bit b % 32 of word b / 32). */
__global__ void clear_mask_kernel(uint32_t* mask, uint64_t stride, uint32_t pos0, uint32_t ratio, uint32_t block) {
  const uint32_t token = blockIdx.y;
  const uint32_t w = blockIdx.x * blockDim.x + threadIdx.x;
  if (w < ((comp_len(pos0 + token, ratio) + block - 1) / block + 31) / 32) mask[token * stride + w] = 0u;
}

__global__ void block_max_kernel(const float* score, uint64_t score_stride, float* blocks, uint64_t block_stride,
                                 uint32_t pos0, uint32_t ratio, uint32_t block) {
  const uint32_t token = blockIdx.y;
  const uint32_t n = comp_len(pos0 + token, ratio);
  const uint32_t nb = (n + block - 1) / block;
  const uint32_t b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= nb) return;
  const float* s = score + token * score_stride;
  float m = -INFINITY;
  for (uint32_t i = b * block; i < min(n, (b + 1) * block); i++) m = fmaxf(m, s[i]);
  blocks[token * block_stride + b] = b == nb - 1 ? INFINITY : m;
}

/* Thread per selected block, grid.y per token. */
__global__ void fill_mask_kernel(uint32_t* mask, uint64_t stride, const uint32_t* selected, uint32_t k_max,
                                 uint32_t pos0, uint32_t ratio, uint32_t block) {
  const uint32_t token = blockIdx.y;
  const uint32_t n = comp_len(pos0 + token, ratio);
  const uint32_t j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= min(k_max, (n + block - 1) / block)) return;
  const uint32_t b = selected[(uint64_t)token * k_max + j];
  atomicOr(&mask[token * stride + b / 32], 1u << (b % 32));
}

__global__ void apply_mask_kernel(float* score, uint64_t stride, const uint32_t* mask, uint64_t mask_stride,
                                  uint32_t pos0, uint32_t ratio, uint32_t block) {
  const uint32_t token = blockIdx.y;
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t b = i / block;
  if (i < comp_len(pos0 + token, ratio) && !((mask[token * mask_stride + b / 32] >> (b % 32)) & 1u))
    score[token * stride + i] = -INFINITY;
}

/* Stable block compaction keeps physical key order, including the newest
 * partial block. Scratch is the source layer's now-dead block-select buffer. */
__global__ void compact_candidate_blocks_kernel(const uint32_t* mask, uint64_t mask_stride, uint32_t* out,
                                                uint32_t out_stride, uint32_t pos0, uint32_t ratio) {
  const uint32_t token = blockIdx.x;
  const uint32_t blocks = (comp_len(pos0 + token, ratio) + 7) / 8;
  const uint32_t words = (blocks + 31) / 32;
  const uint32_t per = (words + blockDim.x - 1) / blockDim.x;
  const uint32_t begin = min(words, threadIdx.x * per), end = min(words, begin + per);
  uint32_t count = 0;
  for (uint32_t i = begin; i < end; i++) count += __popc(mask[(uint64_t)token * mask_stride + i]);
  uint32_t total = 0;
  uint32_t at = block_exclusive_scan(count, &total);
  for (uint32_t i = begin; i < end; i++) {
    uint32_t bits = mask[(uint64_t)token * mask_stride + i];
    while (bits) {
      const uint32_t bit = __ffs(bits) - 1;
      if (at < out_stride) out[(uint64_t)token * out_stride + at] = i * 32 + bit;
      ++at;
      bits &= bits - 1;
    }
  }
}

__global__ void clear_index_scores_kernel(float* score, uint64_t stride, uint32_t pos0, uint32_t ratio) {
  const uint32_t token = blockIdx.y;
  const uint32_t key = blockIdx.x * blockDim.x + threadIdx.x;
  if (key < comp_len(pos0 + token, ratio)) score[(uint64_t)token * stride + key] = -INFINITY;
}

/* Block (head, token). Scores for every key are kept in shared memory. */
__global__ void attention_kernel(const float* q, const float* ring, const float* chunk, uint32_t window,
                                 uint32_t floor, ds41_fp4_rows comp, const uint32_t* all_idx, uint32_t k,
                                 uint32_t ratio, const float* sinks, float* out, uint32_t pos0, uint32_t heads,
                                 uint32_t head_dim, float scale) {
  extern __shared__ float scores[];
  const uint32_t head = blockIdx.x;
  const uint32_t token = blockIdx.y;
  const uint32_t p = pos0 + token;
  const uint32_t lo = max(p + 1 > window ? p + 1 - window : 0u, min(floor, p));
  const uint32_t n_window = p - lo + 1;
  const uint32_t n_comp = min(k, comp_len(p, ratio));
  const uint32_t* idx = all_idx + (uint64_t)token * k;
  const float* qh = q + ((uint64_t)token * heads + head) * head_dim;
  auto key_at = [&](uint32_t j, uint32_t d) -> float {
    if (j >= n_window) return kv_value(comp, idx[j - n_window], d, head_dim);
    const uint32_t pos = lo + j;
    return pos < pos0 ? ring[(uint64_t)(pos % window) * head_dim + d] : chunk[(uint64_t)(pos - pos0) * head_dim + d];
  };
  const uint32_t n = n_window + n_comp;
  const uint32_t warps = blockDim.x / warpSize;
  const uint32_t warp = threadIdx.x / warpSize;
  const uint32_t lane = threadIdx.x % warpSize;
  for (uint32_t j = warp; j < n; j += warps) {
    float dot = 0.0f;
    for (uint32_t d = lane; d < head_dim; d += warpSize) dot += qh[d] * key_at(j, d);
    for (int o = warpSize / 2; o > 0; o >>= 1) dot += __shfl_xor(dot, o);
    if (lane == 0) scores[j] = dot * scale;
  }
  __syncthreads();
  float m = -INFINITY;
  for (uint32_t j = threadIdx.x; j < n; j += blockDim.x) m = fmaxf(m, scores[j]);
  m = fmaxf(block_max(m), -1.0e30f);
  float sum = 0.0f;
  for (uint32_t j = threadIdx.x; j < n; j += blockDim.x) {
    scores[j] = expf(scores[j] - m);
    sum += scores[j];
  }
  sum = block_sum(sum) + expf(sinks[head] - m);
  __syncthreads();
  for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
    float acc = 0.0f;
    for (uint32_t j = 0; j < n; j++) acc += scores[j] * key_at(j, d);
    out[((uint64_t)token * heads + head) * head_dim + d] = acc / sum;
  }
}

/* (cos, sin) of each rope pair of n positions from pos0: table[t * half + i]. */
__global__ void rope_table_kernel(const float* inv_freq, float2* table, uint32_t half, uint32_t pos0, uint32_t n) {
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n * half) return;
  const float angle = (float)(pos0 + i / half) * inv_freq[i % half];
  table[i] = make_float2(cosf(angle), sinf(angle));
}

/* pow2_ceil(x) = 2^pow2_ceil_exp(x). */
__device__ __forceinline__ int pow2_ceil_exp(float x) {
  int e = 0;
  const float m = frexpf(x, &e);
  return m == 0.5f ? e - 1 : e;
}

/* round_e4m3(clamp(v / 2^eb, -448, 448)) * 2^eb, with the power-of-two
 * divisions as exact exponent shifts. */
__device__ __forceinline__ float fp8_round_scaled(float v, int eb) {
  const float x = fminf(fmaxf(ldexpf(v, -eb), -448.0f), 448.0f);
  const float a = fabsf(x);
  int e = 0;
  frexpf(a, &e);
  const int se = a < 0.015625f ? -9 : e - 4;
  return copysignf(ldexpf(fminf(ldexpf(rintf(ldexpf(a, -se)), se), 448.0f), eb), x);
}

/* Max and sum over the 16 lanes of each half-wave (DPP quad permutes, then
 * the half-row and row mirrors). */
template <int kCtrl>
__device__ __forceinline__ float dpp_f32(float v) {
  return __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(v), kCtrl, 0xf, 0xf, false));
}
__device__ __forceinline__ float max16(float v) {
  v = fmaxf(v, dpp_f32<0xb1>(v));
  v = fmaxf(v, dpp_f32<0x4e>(v));
  v = fmaxf(v, dpp_f32<0x141>(v));
  return fmaxf(v, dpp_f32<0x140>(v));
}
__device__ __forceinline__ float sum16(float v) {
  v += dpp_f32<0xb1>(v);
  v += dpp_f32<0x4e>(v);
  v += dpp_f32<0x141>(v);
  return v + dpp_f32<0x140>(v);
}

constexpr uint32_t kAttnRope = 64;  // the fused rotations' rope_dim
constexpr uint32_t kAttnRows = 16;
constexpr uint32_t kAttnRowPitch = 512 + 8;  // halves per staged KV row
constexpr uint32_t kAttnProbPitch = 16 + 8;  // halves per probability row
typedef _Float16 attn_half16 __attribute__((ext_vector_type(16)));
typedef _Float16 attn_half2 __attribute__((ext_vector_type(2)));
typedef float attn_float8 __attribute__((ext_vector_type(8)));

__device__ __forceinline__ attn_half16 lds_half16(const _Float16* p) {
  const uint4 a = reinterpret_cast<const uint4*>(p)[0], b = reinterpret_cast<const uint4*>(p)[1];
  const uint32_t w[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
  return __builtin_bit_cast(attn_half16, w);
}

/* The opposite half-wave's copy of a fragment. */
__device__ __forceinline__ attn_half16 swap_halves(attn_half16 v) {
  uint32_t w[8];
  __builtin_memcpy(w, &v, sizeof(w));
#pragma unroll
  for (int i = 0; i < 8; i++) w[i] = __builtin_amdgcn_permlanex16(w[i], w[i], 0x76543210u, 0xfedcba98u, false, false);
  return __builtin_bit_cast(attn_half16, w);
}

/* A workgroup barrier that waits for the LDS accesses only (__syncthreads
 * also waits for the outstanding global loads, the next tile's prefetch). */
__device__ __forceinline__ void lds_barrier() { asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory"); }

/* The same attention for 32 heads per block on the matrix cores: block
 * (token, 32 heads), 256 threads; head_dim 512, window a power of two
 * <= 256, k <= 512. The window rows and the compressed rows run as separate
 * 16-row tiles. Thread t stages dimensions 2 t, 2 t + 1 of each tile,
 * loaded into registers one tile ahead: as FP16 rows in LDS for the scores,
 * and as two FP16 columns of 16 rows in its own registers for the output
 * products. Wave w scores heads 16 (w % 2) .. + 15 over dimensions
 * 128 (w / 2) .. + 127 with its Q fragments in registers; the four partial
 * scores meet in LDS for the online softmax (scores and sums in fp32, half a
 * wave per head). Wave w accumulates the output of all 32 heads over the
 * dimensions its lanes stage, 64 w .. 64 w + 63, in fp32 registers: each
 * half of a wave multiplies by its own B operand copy, so a product with
 * the lanes' own columns and one with the opposite half's (permlanex16)
 * cover them. With `rope` (rope_table_kernel at pos0 for the batch), q's
 * tail kAttnRope dimensions are rotated as ds41_v41_rope_tail does, and the
 * output gets ds41_v41_rope_fp8's inverse rotation and FP8 rounding. The
 * kernel's code is kept within the instruction cache (a rolled epilogue,
 * DPP reductions, table decoding); larger code ran several times slower. */
__global__ __launch_bounds__(256) void attention_f16_kernel(const float* q, const float* ring, const float* chunk,
                                                            uint32_t window, uint32_t floor, ds41_fp4_rows comp,
                                                            const uint32_t* all_idx, uint32_t k, uint32_t ratio,
                                                            const float* sinks, float* out, uint32_t pos0,
                                                            uint32_t heads, float scale, const float2* rope) {
  constexpr uint32_t DIM = 512, HEADS = 32, ROWS = kAttnRows, ROPE = kAttnRope, TAIL = DIM - ROPE;
  const uint32_t token = blockIdx.x;
  const uint32_t head0 = blockIdx.y * HEADS;
  const uint32_t tid = threadIdx.x, wave = tid / 32, lane = tid % 32, h = lane / 16, l16 = lane % 16;
  const uint32_t p = pos0 + token;
  const uint32_t lo = max(p + 1 > window ? p + 1 - window : 0u, min(floor, p));
  const uint32_t n_window = p - lo + 1;
  const uint32_t n_comp = min(k, comp_len(p, ratio));
  const uint32_t n_score = n_window + n_comp;

  __shared__ uint32_t fp4_pairs[256];  // code byte -> its two E2M1 values as FP16
  __shared__ uint16_t e4m3_half[256];  // scale byte -> FP16
  __shared__ __attribute__((aligned(16))) _Float16 kv_row[ROWS * kAttnRowPitch];
  __shared__ __attribute__((aligned(16))) _Float16 probs[HEADS * kAttnProbPitch];
  __shared__ float part[4][ROWS][HEADS + 1];
  __shared__ float softmax_rescale[HEADS];
  __shared__ float softmax_den[HEADS];

  /* The window's positions lo .. p fill the first window tiles (chunk row
   * pos - pos0 when the position is in this batch, else ring slot
   * pos % window), the compressed rows idx[0 .. n_comp) the tiles after
   * them. A tile's rows past the end repeat its last row; the softmax masks
   * them. The products of E2M1 values and E4M3 scales are exact in FP16. */
  const uint32_t* idx = all_idx + (uint64_t)token * k;
  const uint32_t window_tiles = (n_window + ROWS - 1) / ROWS;
  const uint32_t tiles = window_tiles + (n_comp + ROWS - 1) / ROWS;
  {
    const _Float16 lo_value = (_Float16)e2m1_value(tid), hi_value = (_Float16)e2m1_value(tid >> 4);
    fp4_pairs[tid] = (uint32_t)__builtin_bit_cast(uint16_t, lo_value) |
                     (uint32_t)__builtin_bit_cast(uint16_t, hi_value) << 16;
    e4m3_half[tid] = __builtin_bit_cast(uint16_t, (_Float16)e4m3_value(tid));
  }

  /* Q fragments (B operand: lane = head, 16 dimensions per step). */
  const uint32_t qk_heads = 16 * (wave % 2), qk_dim = 128 * (wave / 2);
  attn_half16 qf[8];
  {
    const float* qrow = q + ((uint64_t)token * heads + head0 + qk_heads + l16) * DIM;
    float2 cs[8][8];  // rope dimensions only
    /* Only the last quarter holds rope dimensions. */
    const bool rotate = rope && qk_dim == DIM - 128;
    if (rotate) {
#pragma unroll
      for (uint32_t i = 8 - ROPE / 16; i < 8; i++)
#pragma unroll
        for (uint32_t j = 0; j < 8; j++) cs[i][j] = rope[token * (ROPE / 2) + (16 * i - (128 - ROPE)) / 2 + j];
    }
#pragma unroll
    for (uint32_t i = 0; i < 8; i++) {
      const uint32_t d0 = qk_dim + 16 * i;
      float v[16];
#pragma unroll
      for (uint32_t c = 0; c < 4; c++) {
        const float4 f = reinterpret_cast<const float4*>(qrow + d0)[c];
        v[4 * c] = f.x;
        v[4 * c + 1] = f.y;
        v[4 * c + 2] = f.z;
        v[4 * c + 3] = f.w;
      }
      if (i >= 8 - ROPE / 16 && rotate) {
#pragma unroll
        for (uint32_t j = 0; j < 8; j++) {
          const float c = cs[i][j].x;
          const float s = cs[i][j].y;
          const float re = v[2 * j];
          const float im = v[2 * j + 1];
          v[2 * j] = re * c - im * s;
          v[2 * j + 1] = re * s + im * c;
        }
      }
#pragma unroll
      for (uint32_t j = 0; j < 16; j++) qf[i][j] = (_Float16)v[j];
    }
  }
  /* Softmax state of the heads this half-wave owns (4 wave + 2 jj + h). */
  float soft_max[2], soft_den[2];
#pragma unroll
  for (uint32_t jj = 0; jj < 2; jj++) {
    soft_max[jj] = sinks[head0 + 4 * wave + 2 * jj + h];
    soft_den[jj] = 1.0f;
  }
  /* acc[4 hb + s]: heads 16 hb .. + 15 by the lanes' own even (s 0) and odd
   * (s 1) dimensions, then the opposite half's (s 2, 3). */
  attn_float8 acc[8];
#pragma unroll
  for (uint32_t j = 0; j < 8; j++) acc[j] = attn_float8{};
  __syncthreads();

  /* Staging: two floats of window rows, (code byte, scale byte) of
   * compressed rows. */
  uint2 raw[ROWS];
  const auto fetch = [&](uint32_t tile) {
    if (tile < window_tiles) {
#pragma unroll
      for (uint32_t r = 0; r < ROWS; r++) {
        const uint32_t pos = lo + min(tile * ROWS + r, n_window - 1);
        const float* src =
            pos < pos0 ? ring + (uint64_t)(pos & (window - 1)) * DIM : chunk + (uint64_t)(pos - pos0) * DIM;
        raw[r] = *reinterpret_cast<const uint2*>(src + 2 * tid);
      }
    } else {
      const uint32_t c0 = (tile - window_tiles) * ROWS;
#pragma unroll
      for (uint32_t r = 0; r < ROWS; r++) {
        const uint64_t row = idx[min(c0 + r, n_comp - 1)];
        raw[r] = make_uint2(comp.codes[row * (DIM / 2) + tid], comp.scales[row * (DIM / 16) + tid / 8]);
      }
    }
  };
  /* The tile's columns 2 t (even) and 2 t + 1 (odd) as B operands. */
  attn_half16 col_even, col_odd;
  const auto store = [&](uint32_t tile) {
    const bool window_tile = tile < window_tiles;
#pragma unroll
    for (uint32_t r = 0; r < ROWS; r++) {
      uint32_t pair;
      if (window_tile) {
        const _Float16 a = (_Float16)__uint_as_float(raw[r].x), b = (_Float16)__uint_as_float(raw[r].y);
        pair = (uint32_t)__builtin_bit_cast(uint16_t, a) | (uint32_t)__builtin_bit_cast(uint16_t, b) << 16;
      } else {
        const _Float16 sc = __builtin_bit_cast(_Float16, e4m3_half[raw[r].y]);
        const attn_half2 v = __builtin_bit_cast(attn_half2, fp4_pairs[raw[r].x]) * attn_half2{sc, sc};
        pair = __builtin_bit_cast(uint32_t, v);
      }
      const attn_half2 v = __builtin_bit_cast(attn_half2, pair);
      col_even[r] = v.x;
      col_odd[r] = v.y;
      *reinterpret_cast<uint32_t*>(&kv_row[r * kAttnRowPitch + 2 * tid]) = pair;
    }
  };

  /* Tile -1 only prefetches tile 0 (one copy of the staging code). */
  for (int32_t tile = -1; tile < (int32_t)tiles; tile++) {
    if (tile >= 0) {
      store(tile);
      lds_barrier();
    }
    if (tile + 1 < (int32_t)tiles) fetch(tile + 1);
    if (tile < 0) continue;
    const uint32_t nr = tile < (int32_t)window_tiles ? min(ROWS, n_window - tile * ROWS)
                                                     : min(ROWS, n_comp - (tile - window_tiles) * ROWS);

    /* Partial scores: rows (A operand: lane = row) by this wave's heads. */
    {
      attn_float8 s = attn_float8{};
#pragma unroll
      for (uint32_t i = 0; i < 8; i++)
        s = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(lds_half16(&kv_row[l16 * kAttnRowPitch + qk_dim + 16 * i]),
                                                       qf[i], s);
      /* Accumulator element e of a lane is (row 2 e + lane / 16, head lane % 16). */
#pragma unroll
      for (uint32_t e = 0; e < 8; e++) part[wave / 2][2 * e + h][qk_heads + l16] = s[e];
    }
    lds_barrier();

    /* Online softmax: half-wave per head, lane = row. */
#pragma unroll
    for (uint32_t jj = 0; jj < 2; jj++) {
      const uint32_t head = 4 * wave + 2 * jj + h;
      const float s =
          l16 < nr ? (part[0][l16][head] + part[1][l16][head] + part[2][l16][head] + part[3][l16][head]) * scale
                   : -INFINITY;
      const float m_new = fmaxf(soft_max[jj], max16(s));
      const float e = l16 < nr ? expf(s - m_new) : 0.0f;
      const float sum = sum16(e);
      const float factor = expf(soft_max[jj] - m_new);
      soft_den[jj] = soft_den[jj] * factor + sum;
      soft_max[jj] = m_new;
      if (l16 == 0) softmax_rescale[head] = factor;
      probs[head * kAttnProbPitch + l16] = (_Float16)e;
    }
    lds_barrier();

    /* Output: probabilities (A operand: lane = head) by the staged columns. */
    {
      const attn_half16 cols[4] = {col_even, col_odd, swap_halves(col_even), swap_halves(col_odd)};
#pragma unroll
      for (uint32_t hb = 0; hb < 2; hb++) {
        const attn_half16 pa = lds_half16(&probs[(16 * hb + l16) * kAttnProbPitch]);
        float factor[8];
#pragma unroll
        for (uint32_t e = 0; e < 8; e++) factor[e] = softmax_rescale[16 * hb + 2 * e + h];
#pragma unroll
        for (uint32_t c = 0; c < 4; c++) {
#pragma unroll
          for (uint32_t e = 0; e < 8; e++) acc[4 * hb + c][e] *= factor[e];
          acc[4 * hb + c] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(pa, cols[c], acc[4 * hb + c]);
        }
      }
    }
    /* The next store overwrites kv_row and the next softmax probs. */
    lds_barrier();
  }

  if (l16 == 0) {
    softmax_den[4 * wave + h] = soft_den[0];
    softmax_den[4 * wave + 2 + h] = soft_den[1];
  }
  __syncthreads();

  /* Accumulator element e of a lane is head 16 hb + 2 e + lane / 16 at
   * dimensions (d, d + 1): d = 64 wave + 32 h + 2 (lane % 16) for the own
   * columns and 64 wave + 32 (1 - h) + 2 (lane % 16) for the opposite
   * half's. */
  const uint32_t d_own = 64 * wave + 32 * h + 2 * l16, d_other = 64 * wave + 32 * (1 - h) + 2 * l16;
  /* The rope dimensions are the last wave's. */
  const bool rotate = rope && wave == 7;
  float2 cs_own = make_float2(1.0f, 0.0f), cs_other = cs_own;
  if (rotate && d_own >= TAIL) cs_own = rope[token * (ROPE / 2) + (d_own - TAIL) / 2];
  if (rotate && d_other >= TAIL) cs_other = rope[token * (ROPE / 2) + (d_other - TAIL) / 2];
#pragma unroll
  for (uint32_t hb = 0; hb < 2; hb++) {
    /* Element 0 of the accumulators, rotated after each head (a rolled
     * loop keeps the kernel's code within the instruction cache). */
#pragma unroll 1
    for (uint32_t e = 0; e < 8; e++) {
      const uint32_t head = 16 * hb + 2 * e + h;
      const float den = softmax_den[head];
      float v[4];
#pragma unroll
      for (uint32_t c = 0; c < 4; c++) {
        v[c] = acc[4 * hb + c][0] / den;
        acc[4 * hb + c] = __builtin_shufflevector(acc[4 * hb + c], acc[4 * hb + c], 1, 2, 3, 4, 5, 6, 7, 0);
      }
      if (rotate) {
#pragma unroll
        for (uint32_t pr = 0; pr < 2; pr++) {
          const float2 cs = pr ? cs_other : cs_own;
          if ((pr ? d_other : d_own) >= TAIL) {
            const float c = cs.x;
            const float s = -cs.y;
            const float re = v[2 * pr];
            const float im = v[2 * pr + 1];
            v[2 * pr] = re * c - im * s;
            v[2 * pr + 1] = re * s + im * c;
          }
        }
      }
      if (rope) {
        /* FP8 blocks of 32 dimensions: the own columns of a half-wave form
         * one, the opposite half's the other. */
#pragma unroll
        for (uint32_t pr = 0; pr < 2; pr++) {
          const float amax = max16(fmaxf(fabsf(v[2 * pr]), fabsf(v[2 * pr + 1])));
          const int eb = pow2_ceil_exp(fmaxf(amax, 1.0e-4f) / 448.0f);
#pragma unroll
          for (uint32_t c = 0; c < 2; c++) v[2 * pr + c] = fp8_round_scaled(v[2 * pr + c], eb);
        }
      }
      float* o = out + ((uint64_t)token * heads + head0 + head) * DIM;
      *reinterpret_cast<float2*>(o + d_own) = make_float2(v[0], v[1]);
      *reinterpret_cast<float2*>(o + d_other) = make_float2(v[2], v[3]);
    }
  }
}

/* Decode attention split over the rows (flash decoding): block (group, split,
 * token) stages its share of the rows in LDS once for 16 heads, one wave per
 * head with an online softmax, and leaves (max, sum, out[512]) per head and
 * split in `partial`; attention_combine_kernel merges the splits with the
 * sink. head_dim 512, 512 threads. */
constexpr uint32_t kAttnSplits = 20;
constexpr uint32_t kAttnSplitHeads = 16;
/* (max, sum, pad, pad, out[512]) per head and split. */
constexpr uint32_t kAttnPartial = 512 + 4;

__global__ __launch_bounds__(512) void attention_split_kernel(const float* q, const float* ring, const float* chunk,
                                                              uint32_t window, uint32_t floor, ds41_fp4_rows comp,
                                                              const uint32_t* all_idx, uint32_t k, uint32_t ratio,
                                                              float* partial, uint32_t pos0, uint32_t heads,
                                                              float scale) {
  constexpr uint32_t DIM = 512, STAGE = 16;
  const uint32_t token = blockIdx.z;
  const uint32_t split = blockIdx.y;
  const uint32_t warp = threadIdx.x / 32;
  const uint32_t lane = threadIdx.x % 32;
  const uint32_t head = blockIdx.x * kAttnSplitHeads + warp;
  const uint32_t p = pos0 + token;
  const uint32_t lo = max(p + 1 > window ? p + 1 - window : 0u, min(floor, p));
  const uint32_t n_window = p - lo + 1;
  const uint32_t n_rows = n_window + min(k, comp_len(p, ratio));
  const uint32_t per = (n_rows + kAttnSplits - 1) / kAttnSplits;
  const uint32_t r0 = min(n_rows, split * per);
  const uint32_t r1 = min(n_rows, r0 + per);
  const uint32_t* idx = all_idx + (uint64_t)token * k;
  __shared__ float4 kv_sh[STAGE * DIM / 4];

  const float4* q4 = reinterpret_cast<const float4*>(q + ((uint64_t)token * heads + head) * DIM);
  const float4 q0 = q4[lane], q1 = q4[lane + 32], q2 = q4[lane + 64], q3 = q4[lane + 96];
  float m = -INFINITY, l = 0.0f;
  float4 o0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f), o1 = o0, o2 = o0, o3 = o0;
  auto fma4 = [](float4& acc, float a, const float4& v, float b) {
    acc.x = acc.x * a + v.x * b;
    acc.y = acc.y * a + v.y * b;
    acc.z = acc.z * a + v.z * b;
    acc.w = acc.w * a + v.w * b;
  };
  auto dot4 = [](const float4& a, const float4& b) { return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w; };
  for (uint32_t row0 = r0; row0 < r1; row0 += STAGE) {
    const uint32_t nr = min(STAGE, r1 - row0);
    for (uint32_t i = threadIdx.x; i < nr * (DIM / 4); i += blockDim.x) {
      const uint32_t j = row0 + i / (DIM / 4);
      if (j >= n_window) {
        kv_sh[i] = kv_load4(comp, idx[j - n_window], (i % (DIM / 4)) * 4, DIM);
      } else {
        const uint32_t pos = lo + j;
        const float* src = pos < pos0 ? ring + (uint64_t)(pos % window) * DIM : chunk + (uint64_t)(pos - pos0) * DIM;
        kv_sh[i] = reinterpret_cast<const float4*>(src)[i % (DIM / 4)];
      }
    }
    __syncthreads();
    for (uint32_t r = 0; r < nr; r++) {
      const float4* kv = kv_sh + r * (DIM / 4);
      const float4 k0 = kv[lane], k1 = kv[lane + 32], k2 = kv[lane + 64], k3 = kv[lane + 96];
      float dot = dot4(q0, k0) + dot4(q1, k1) + dot4(q2, k2) + dot4(q3, k3);
      for (int off = 16; off > 0; off >>= 1) dot += __shfl_xor(dot, off, 32);
      const float score = dot * scale;
      const float m_new = fmaxf(m, score);
      const float old_scale = expf(m - m_new);
      const float row_scale = expf(score - m_new);
      l = l * old_scale + row_scale;
      fma4(o0, old_scale, k0, row_scale);
      fma4(o1, old_scale, k1, row_scale);
      fma4(o2, old_scale, k2, row_scale);
      fma4(o3, old_scale, k3, row_scale);
      m = m_new;
    }
    __syncthreads();
  }
  float* out = partial + (((uint64_t)token * heads + head) * kAttnSplits + split) * kAttnPartial;
  if (lane == 0) {
    out[0] = m;
    out[1] = l;
  }
  float4* out4 = reinterpret_cast<float4*>(out + 4);
  out4[lane] = o0;
  out4[lane + 32] = o1;
  out4[lane + 64] = o2;
  out4[lane + 96] = o3;
}

/* Block (head, token): out = sum_s exp(m_s - M) o_s / (exp(sink - M) + sum_s exp(m_s - M) l_s). */
__global__ void attention_combine_kernel(const float* partial, const float* sinks, float* out, uint32_t heads) {
  const uint32_t head = blockIdx.x;
  const uint32_t token = blockIdx.y;
  const float* part = partial + ((uint64_t)token * heads + head) * kAttnSplits * kAttnPartial;
  float m = sinks[head];
  for (uint32_t s = 0; s < kAttnSplits; s++) m = fmaxf(m, part[s * kAttnPartial]);
  float den = expf(sinks[head] - m);
  float factor[kAttnSplits];
  for (uint32_t s = 0; s < kAttnSplits; s++) {
    const float ms = part[s * kAttnPartial];
    factor[s] = ms == -INFINITY ? 0.0f : expf(ms - m);
    den += factor[s] * part[s * kAttnPartial + 1];
  }
  for (uint32_t d = threadIdx.x; d < 512; d += blockDim.x) {
    float acc = 0.0f;
    for (uint32_t s = 0; s < kAttnSplits; s++) acc += factor[s] * part[s * kAttnPartial + 4 + d];
    out[((uint64_t)token * heads + head) * 512 + d] = acc / den;
  }
}

/* Block (head, token): the DSpark block attention. Keys are the ring rows of
 * positions first .. first + n_ring - 1 (slot pos % window) and every block row. */
__global__ void dspark_attention_kernel(const float* q, const float* ring, uint32_t first, uint32_t n_ring,
                                        uint32_t window, const float* block_kv, uint32_t n_block,
                                        const float* sinks, float* out, uint32_t heads, uint32_t head_dim,
                                        float scale) {
  extern __shared__ float scores[];
  const uint32_t head = blockIdx.x;
  const uint32_t token = blockIdx.y;
  const float* qh = q + ((uint64_t)token * heads + head) * head_dim;
  auto key = [&](uint32_t j) {
    return j < n_ring ? ring + (uint64_t)((first + j) % window) * head_dim
                      : block_kv + (uint64_t)(j - n_ring) * head_dim;
  };
  const uint32_t n = n_ring + n_block;
  const uint32_t warps = blockDim.x / warpSize;
  const uint32_t warp = threadIdx.x / warpSize;
  const uint32_t lane = threadIdx.x % warpSize;
  for (uint32_t j = warp; j < n; j += warps) {
    const float* k = key(j);
    float dot = 0.0f;
    for (uint32_t d = lane; d < head_dim; d += warpSize) dot += qh[d] * k[d];
    for (int o = warpSize / 2; o > 0; o >>= 1) dot += __shfl_xor(dot, o);
    if (lane == 0) scores[j] = dot * scale;
  }
  __syncthreads();
  float m = -INFINITY;
  for (uint32_t j = threadIdx.x; j < n; j += blockDim.x) m = fmaxf(m, scores[j]);
  m = fmaxf(block_max(m), sinks[head]);
  float sum = 0.0f;
  for (uint32_t j = threadIdx.x; j < n; j += blockDim.x) {
    scores[j] = expf(scores[j] - m);
    sum += scores[j];
  }
  sum = block_sum(sum) + expf(sinks[head] - m);
  __syncthreads();
  for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
    float acc = 0.0f;
    for (uint32_t j = 0; j < n; j++) acc += scores[j] * key(j)[d];
    out[((uint64_t)token * heads + head) * head_dim + d] = acc / sum;
  }
}

__global__ void hc_mean_kernel(const float* h, float* out, uint64_t out_stride, uint32_t hc, uint32_t dim) {
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t t = blockIdx.y;
  if (i >= dim) return;
  const float* row = h + (uint64_t)t * hc * dim + i;
  float sum = 0.0f;
  for (uint32_t c = 0; c < hc; c++) sum += row[(uint64_t)c * dim];
  out[t * out_stride + i] = sum / (float)hc;
}

/* out[0:rank] = table row tokens[index] (F32, or BF16 when bf16). */
__global__ void gather_row_kernel(const void* table, int format, const int32_t* tokens, uint32_t index, float* out,
                                  uint32_t rank) {
  const uint64_t row = (uint64_t)tokens[index] * rank;
  for (uint32_t d = threadIdx.x; d < rank; d += blockDim.x)
    out[d] = format == DS41_V41_ROW_BF16  ? bf16_to_f32(((const uint16_t*)table)[row + d])
             : format == DS41_V41_ROW_F16 ? (float)((const _Float16*)table)[row + d]
                                          : ((const float*)table)[row + d];
}

/* tokens[index] = argmax over finite x[v] + bias[v], the first on ties; -1 when none. One block. */
__global__ void __launch_bounds__(1024) argmax_bias_kernel(const float* x, const float* bias, uint32_t n,
                                                           int32_t* tokens, uint32_t index,
                                                           const int32_t* id_map = nullptr) {
  __shared__ float best_v[1024];
  __shared__ int32_t best_i[1024];
  float v = -INFINITY;
  int32_t at = -1;
  for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
    const float s = x[i] + (bias ? bias[i] : 0.0f);
    if (isfinite(s) && (at < 0 || s > v)) {
      v = s;
      at = (int32_t)i;
    }
  }
  best_v[threadIdx.x] = v;
  best_i[threadIdx.x] = at;
  __syncthreads();
  for (uint32_t step = blockDim.x / 2; step > 0; step >>= 1) {
    if (threadIdx.x < step) {
      const float ov = best_v[threadIdx.x + step];
      const int32_t oi = best_i[threadIdx.x + step];
      if (oi >= 0 && (best_i[threadIdx.x] < 0 || ov > best_v[threadIdx.x] ||
                      (ov == best_v[threadIdx.x] && oi < best_i[threadIdx.x]))) {
        best_v[threadIdx.x] = ov;
        best_i[threadIdx.x] = oi;
      }
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) tokens[index] = id_map && best_i[0] >= 0 ? id_map[best_i[0]] : best_i[0];
}

/* Markov rows fused with their selection: block b owns tokens
 * [b * span, (b + 1) * span) and emits its kSel best finite values of
 * logits[t] + W2[t] . x (W2 FP16 rows of 256), descending and then lowest
 * token. Contiguous ownership makes the candidate index order the token
 * order, so a merge over candidates keeps the lowest-token tie rule. */
constexpr uint32_t kMarkovBlocks = 256;

template <uint32_t kSel>
__global__ void __launch_bounds__(256) markov_select_kernel(const _Float16* w2, const float* x, const float* logits,
                                                           uint32_t n, float* cand_v, int32_t* cand_i) {
  const uint32_t lane = threadIdx.x % 32, wave = threadIdx.x / 32;
  const uint32_t span = (n + gridDim.x - 1) / gridDim.x;
  const uint32_t begin = blockIdx.x * span, end = min(n, begin + span);
  float xv[8];
  {
    const float4* xp = reinterpret_cast<const float4*>(x) + 2 * lane;
    const float4 a = xp[0], b = xp[1];
    xv[0] = a.x, xv[1] = a.y, xv[2] = a.z, xv[3] = a.w, xv[4] = b.x, xv[5] = b.y, xv[6] = b.z, xv[7] = b.w;
  }
  float best_v[kSel];
  int32_t best_i[kSel];
#pragma unroll
  for (uint32_t j = 0; j < kSel; ++j) {
    best_v[j] = -INFINITY;
    best_i[j] = -1;
  }
  constexpr uint32_t kUnroll = 4;
  for (uint32_t r0 = begin + wave; r0 < end; r0 += 8 * kUnroll) {
    uint4 packed[kUnroll];
#pragma unroll
    for (uint32_t u = 0; u < kUnroll; ++u) {
      const uint32_t r = r0 + 8 * u;
      packed[u] = r < end ? reinterpret_cast<const uint4*>(w2 + (uint64_t)r * 256)[lane] : uint4{};
    }
#pragma unroll
    for (uint32_t u = 0; u < kUnroll; ++u) {
      const uint32_t r = r0 + 8 * u;
      const _Float16* h = reinterpret_cast<const _Float16*>(&packed[u]);
      float dot = 0.0f;
#pragma unroll
      for (uint32_t e = 0; e < 8; ++e) dot += (float)h[e] * xv[e];
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) dot += __shfl_xor(dot, o);
      if (r >= end) continue;
      const float v = logits[r] + dot;
      if (!isfinite(v)) continue;
      // Rows arrive in increasing token order, so a strict test keeps ties stable.
      uint32_t at = kSel;
#pragma unroll
      for (uint32_t j = 0; j < kSel; ++j)
        if (at == kSel && (best_i[j] < 0 || v > best_v[j])) at = j;
#pragma unroll
      for (uint32_t j = kSel - 1; j > 0; --j)
        if (j > at) {
          best_v[j] = best_v[j - 1];
          best_i[j] = best_i[j - 1];
        }
      if (at < kSel) {
        best_v[at] = v;
        best_i[at] = (int32_t)r;
      }
    }
  }
  __shared__ float sv[8][kSel];
  __shared__ int32_t si[8][kSel];
  if (lane == 0)
#pragma unroll
    for (uint32_t j = 0; j < kSel; ++j) {
      sv[wave][j] = best_v[j];
      si[wave][j] = best_i[j];
    }
  __syncthreads();
  if (threadIdx.x != 0) return;
  uint32_t head[8] = {};
  for (uint32_t j = 0; j < kSel; ++j) {
    int w = -1;
    for (int c = 0; c < 8; ++c) {
      if (head[c] >= kSel || si[c][head[c]] < 0) continue;
      const float cv = sv[c][head[c]];
      const int32_t ci = si[c][head[c]];
      if (w < 0 || cv > sv[w][head[w]] || (cv == sv[w][head[w]] && ci < si[w][head[w]])) w = c;
    }
    cand_v[blockIdx.x * kSel + j] = w < 0 ? -INFINITY : sv[w][head[w]];
    cand_i[blockIdx.x * kSel + j] = w < 0 ? -1 : si[w][head[w]];
    if (w >= 0) ++head[w];
  }
}

/* Eight best finite biased logits, descending and then lowest token id.
 * Each lane retains its local eight; global rounds merge those sorted lists. */
template <bool kPropose>
__global__ void __launch_bounds__(256) top8_bias_kernel(const float* x, const float* bias, uint32_t n,
                                                       int32_t* ids, float* logits, const uint32_t* draw,
                                                       float temperature, ds41_v41_proposal* proposal,
                                                       int32_t* token, const int32_t* id_map = nullptr) {
  constexpr uint32_t K = 8;
  float local_v[K];
  int32_t local_i[K];
#pragma unroll
  for (uint32_t j = 0; j < K; ++j) {
    local_v[j] = -INFINITY;
    local_i[j] = -1;
  }
  for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
    const float v = x[i] + (bias ? bias[i] : 0.0f);
    if (!isfinite(v)) continue;
    uint32_t at = K;
#pragma unroll
    for (uint32_t j = 0; j < K; ++j)
      if (at == K && (local_i[j] < 0 || v > local_v[j])) at = j;
#pragma unroll
    for (uint32_t j = K - 1; j > 0; --j)
      if (j > at) {
        local_v[j] = local_v[j - 1];
        local_i[j] = local_i[j - 1];
      }
    if (at < K) {
      local_v[at] = v;
      local_i[at] = (int32_t)i;
    }
  }
  __shared__ float best_v[256];
  __shared__ int32_t best_i[256];
  uint32_t cursor = 0;
  for (uint32_t j = 0; j < K; ++j) {
    best_v[threadIdx.x] = cursor < K ? local_v[cursor] : -INFINITY;
    best_i[threadIdx.x] = cursor < K ? local_i[cursor] : -1;
    __syncthreads();
    for (uint32_t step = blockDim.x / 2; step; step >>= 1) {
      if (threadIdx.x < step) {
        const float ov = best_v[threadIdx.x + step];
        const int32_t oi = best_i[threadIdx.x + step];
        if (oi >= 0 && (best_i[threadIdx.x] < 0 || ov > best_v[threadIdx.x] ||
                        (ov == best_v[threadIdx.x] && oi < best_i[threadIdx.x]))) {
          best_v[threadIdx.x] = ov;
          best_i[threadIdx.x] = oi;
        }
      }
      __syncthreads();
    }
    const int32_t winner = best_i[0];
    if (threadIdx.x == 0) {
      ids[j] = id_map && winner >= 0 ? id_map[winner] : winner;
      logits[j] = best_v[0];
    }
    if (cursor < K && local_i[cursor] == winner) ++cursor;
    __syncthreads();  // every lane consumes the winner before the next round
  }
  if constexpr (kPropose) {
    if (threadIdx.x != 0) return;
    /* q: the temperature softmax of the eight candidates in 24-bit units
     * (the remainder joins the best), or a point mass on a concentrated best;
     * the draw uses this exact q, which the host retains for verification. */
    ds41_v41_proposal q{};
    if (ids[K - 1] < 0) {
      q.token = -1;
      *proposal = q;
      return;
    }
    float e[K], total = 0.0f;
    for (uint32_t j = 0; j < K; ++j) total += e[j] = expf((logits[j] - logits[0]) / temperature);
    constexpr uint32_t kUnits = 1u << 24;
    if (e[0] / total >= 0.7f) {
      q.size = 1;
      q.ids[0] = q.token = ids[0];
      q.masses[0] = kUnits;
    } else {
      uint32_t sum = 0;
      for (uint32_t j = 0; j < K; ++j) {
        q.ids[j] = ids[j];
        sum += q.masses[j] = (uint32_t)floorf(e[j] / total * (float)kUnits);
      }
      q.masses[0] += kUnits - sum;
      q.size = K;
            uint32_t cumulative = 0;
      q.token = ids[K - 1];
      for (uint32_t j = 0; j < K; ++j) {
        cumulative += q.masses[j];
        if (*draw < cumulative) {
          q.token = ids[j];
          break;
        }
      }
    }
    *proposal = q;
    *token = q.token;
  }
}

/* Block per token. */
__global__ void route_kernel(const float* all_logits, const float* bias, int32_t* all_ids, float* all_weights,
                             uint32_t n_experts, uint32_t k, float route_scale) {
  __shared__ float score[512];
  __shared__ float choice[512];
  __shared__ float best_value[32];
  __shared__ uint32_t best_index[32];
  __shared__ float picked[16];
  const float* logits = all_logits + (uint64_t)blockIdx.x * n_experts;
  int32_t* ids = all_ids + (uint64_t)blockIdx.x * k;
  float* weights = all_weights + (uint64_t)blockIdx.x * k;
  for (uint32_t e = threadIdx.x; e < n_experts; e += blockDim.x) {
    const float x = logits[e];
    const float softplus = x > 20.0f ? x : log1pf(expf(x));
    score[e] = sqrtf(softplus);
    choice[e] = score[e] + bias[e];
  }
  __syncthreads();
  for (uint32_t s = 0; s < k; s++) {
    float v = -INFINITY;
    uint32_t at = 0xffffffffu;
    for (uint32_t e = threadIdx.x; e < n_experts; e += blockDim.x) {
      if (choice[e] > v || (choice[e] == v && e < at)) {
        v = choice[e];
        at = e;
      }
    }
    for (int o = warpSize / 2; o > 0; o >>= 1) {
      const float ov = __shfl_xor(v, o);
      const uint32_t oa = __shfl_xor(at, o);
      if (ov > v || (ov == v && oa < at)) {
        v = ov;
        at = oa;
      }
    }
    if (threadIdx.x % warpSize == 0) {
      best_value[threadIdx.x / warpSize] = v;
      best_index[threadIdx.x / warpSize] = at;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      float bv = -INFINITY;
      uint32_t ba = 0xffffffffu;
      for (uint32_t w = 0; w < blockDim.x / warpSize; w++) {
        if (best_value[w] > bv || (best_value[w] == bv && best_index[w] < ba)) {
          bv = best_value[w];
          ba = best_index[w];
        }
      }
      ids[s] = (int32_t)ba;
      picked[s] = score[ba];
      choice[ba] = -INFINITY;
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    float sum = 0.0f;
    for (uint32_t s = 0; s < k; s++) sum += picked[s];
    for (uint32_t s = 0; s < k; s++) weights[s] = picked[s] / (sum + 1.0e-20f) * route_scale;
  }
}

__global__ void swiglu_kernel(const float* gate, const float* up, const float* weight, float* out, uint32_t rows,
                              uint32_t n, float limit) {
  const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (uint64_t)rows * n) return;
  float g = gate[i];
  float u = up[i];
  if (limit > 0.0f) {
    u = fminf(fmaxf(u, -limit), limit);
    g = fminf(g, limit);
  }
  const float h = g / (1.0f + expf(-g)) * u;
  out[i] = weight ? weight[i / n] * h : h;
}

/* Block (copy, token). */
__global__ void engram_gate_kernel(float* h, const float* kv, const uint16_t* q, const uint16_t* k, uint32_t dim,
                                   uint32_t hc, float eps) {
  const uint32_t c = blockIdx.x;
  const uint64_t t = blockIdx.y;
  float* hc_row = h + (t * hc + c) * dim;
  const float* kvt = kv + t * (hc + 1) * dim;
  const float* key = kvt + (uint64_t)c * dim;
  const float* value = kvt + (uint64_t)hc * dim;
  float hh = 0.0f;
  float kk = 0.0f;
  float dot = 0.0f;
  for (uint32_t i = threadIdx.x; i < dim; i += blockDim.x) {
    const float w = bf16_to_f32(q[(uint64_t)c * dim + i]) * bf16_to_f32(k[(uint64_t)c * dim + i]);
    hh += hc_row[i] * hc_row[i];
    kk += key[i] * key[i];
    dot += hc_row[i] * w * key[i];
  }
  hh = block_sum(hh);
  kk = block_sum(kk);
  dot = block_sum(dot);
  const float rstd = rsqrtf(hh / (float)dim + eps) * rsqrtf(kk / (float)dim + eps);
  const float d = dot * rstd * rsqrtf((float)dim);
  const float gate = 1.0f / (1.0f + expf(-copysignf(sqrtf(fmaxf(fabsf(d), 1.0e-6f)), d)));
  __syncthreads();
  for (uint32_t i = threadIdx.x; i < dim; i += blockDim.x) hc_row[i] += gate * value[i];
}

uint32_t blocks_for(uint64_t n, uint32_t threads = kThreads) { return (uint32_t)((n + threads - 1) / threads); }

uint32_t max_comp_len(uint32_t pos0, uint32_t n, uint32_t ratio) { return ratio ? (pos0 + n) / ratio : 0u; }

}  // namespace

extern "C" int ds41_v41_matmul_bf16(const uint16_t* w, const float* x, float* out, uint32_t rows, uint32_t k,
                                    uint32_t n) {
  const uint32_t rows_per_block = kThreads / 32;
  const dim3 grid((rows + rows_per_block - 1) / rows_per_block, (n + kTokenTile - 1) / kTokenTile);
  if (n >= kBf16Tile / 4 && rows % kBf16Tile == 0 && k % kBf16K == 0 && (uintptr_t)w % 16 == 0 &&
      (uintptr_t)x % 16 == 0) {
    matmul_bf16_wmma_kernel<<<dim3(rows / kBf16Tile, (n + kBf16Tile - 1) / kBf16Tile), 256>>>(w, x, out, rows, k, n);
    return launch_ok("bf16 matmul");
  }
  if (k % 256 == 0 && (uintptr_t)w % 16 == 0 && (uintptr_t)x % 16 == 0)
    matmul_bf16_vec8_kernel<<<grid, kThreads>>>(w, x, out, rows, k, n);
  else
    matmul_kernel<uint16_t><<<grid, kThreads>>>(w, x, out, rows, k, n);
  return launch_ok("bf16 matmul");
}

extern "C" int ds41_v41_matmul_f16(const uint16_t* w, const float* x, float* out, uint32_t rows, uint32_t k,
                                   uint32_t n) {
  const uint32_t rows_per_block = kThreads / 32;
  const dim3 grid((rows + rows_per_block - 1) / rows_per_block, (n + kTokenTile - 1) / kTokenTile);
  matmul_kernel<_Float16><<<grid, kThreads>>>((const _Float16*)w, x, out, rows, k, n);
  return launch_ok("f16 matmul");
}

extern "C" int ds41_v41_matmul_f32(const float* w, const float* x, float* out, uint32_t rows, uint32_t k,
                                   uint32_t n) {
  const uint32_t rows_per_block = kThreads / 32;
  const dim3 grid((rows + rows_per_block - 1) / rows_per_block, (n + kTokenTile - 1) / kTokenTile);
  matmul_kernel<float><<<grid, kThreads>>>(w, x, out, rows, k, n);
  return launch_ok("f32 matmul");
}

extern "C" int ds41_v41_copy_rows(float* dst, uint64_t dst_stride, const float* src, uint64_t src_stride,
                                  uint32_t rows, uint32_t cols) {
  if (rows == 0) return 0;
  copy_rows_kernel<<<rows, kThreads>>>(dst, dst_stride, src, src_stride, cols);
  return launch_ok("copy rows");
}

extern "C" int ds41_v41_rms_norm(const float* x, const float* weight, float* out, uint32_t rows, uint32_t n,
                                 float eps) {
  if (rows == 0) return 0;
  rms_norm_kernel<<<rows, kThreads>>>(x, weight, out, n, eps);
  return launch_ok("rms norm");
}

extern "C" int ds41_v41_rope_tail(float* x, uint32_t rows, uint32_t row_dim, uint32_t rope_dim,
                                  const float* inv_freq, uint32_t pos0, uint32_t rows_per_pos, uint32_t pos_step,
                                  int inverse) {
  if (rows == 0) return 0;
  rope_tail_kernel<<<blocks_for((uint64_t)rows * rope_dim / 2), kThreads>>>(x, rows, row_dim, rope_dim, inv_freq,
                                                                           pos0, rows_per_pos, pos_step, inverse);
  return launch_ok("rope");
}

extern "C" int ds41_v41_rope_fp8(float* x, uint32_t rows, uint32_t row_dim, uint32_t rope_dim,
                                 const float* inv_freq, uint32_t pos0, uint32_t rows_per_pos, uint32_t pos_step,
                                 int inverse) {
  if (row_dim % 32 != 0 || rope_dim % 32 != 0 || rope_dim > row_dim) return 1;
  if (rows == 0) return 0;
  const uint32_t blocks = rows * (row_dim / 32);
  rope_fp8_kernel<<<blocks_for(blocks), kThreads>>>(x, blocks, row_dim, rope_dim, inv_freq, pos0, rows_per_pos,
                                                    pos_step, inverse);
  return launch_ok("rope fp8 quant");
}

extern "C" int ds41_v41_q_kv_norm(const float* qa, const float* q_weight, float* qr, uint32_t q_dim, const float* kv,
                                  const float* kv_weight, float* kvn, uint32_t kv_dim, uint32_t rope_dim,
                                  const float* inv_freq, uint32_t pos0, uint32_t n, float eps) {
  if (kv_dim > kQkvMaxKv || kv_dim % 32 != 0 || rope_dim % 32 != 0 || rope_dim > kv_dim)
    return 1;
  if (n == 0) return 0;
  q_kv_norm_kernel<<<dim3(2, n), kQkvThreads>>>(qa, q_weight, qr, q_dim, kv, kv_weight, kvn, kv_dim, rope_dim,
                                                inv_freq, pos0, eps);
  return launch_ok("q kv norm");
}

extern "C" int ds41_v41_fp4_fake_quant(float* x, uint32_t n, uint32_t block, int e4m3_scale) {
  if (n == 0) return 0;
  fp4_fake_quant_kernel<<<blocks_for(n / block), kThreads>>>(x, nullptr, nullptr, n / block, block, e4m3_scale);
  return launch_ok("fp4 quant");
}

extern "C" int ds41_v41_fp4_store(const float* x, uint8_t* codes, uint8_t* scales, uint32_t n, uint32_t block,
                                  int e4m3_scale) {
  if (block == 0 || n % block != 0) return 1;
  fp4_fake_quant_kernel<<<blocks_for(n / block), kThreads>>>(const_cast<float*>(x), codes, scales, n / block, block,
                                                              e4m3_scale);
  return launch_ok("fp4 store");
}

extern "C" int ds41_v41_hc_mixes(const float* fn, const float* h, float* mixes, uint32_t hc, uint32_t dim,
                                 float eps, uint32_t n) {
  const uint32_t mixes_n = 2 * hc + hc * hc;
  if (n >= 16 && mixes_n == 24 && (hc * dim) % 4 == 0) {
    hc_mixes_rows_kernel<24><<<(n + kMixTokens - 1) / kMixTokens, 256>>>(fn, h, mixes, hc * dim, n, eps);
    return launch_ok("hc mixes");
  }
  hc_mixes_kernel<<<dim3(mixes_n, n), 1024>>>(fn, h, mixes, hc * dim, mixes_n, eps);
  return launch_ok("hc mixes");
}

extern "C" int ds41_v41_hc_mix(const float* fn, const float* h, const float* scale, const float* base, float* mixes,
                               float* pre, float* post, float* comb, uint32_t hc, uint32_t dim, float eps,
                               uint32_t iters, float hc_eps, uint32_t n) {
  if (n < kMixSmallTokens && hc == 4 && (hc * dim) % 4 == 0) {
    hc_mix_small_kernel<4><<<dim3((2 * hc + hc * hc) * kMixSlices, n), kMixSmallThreads>>>(fn, h, scale, base, mixes, pre, post,
                                                                           comb, hc * dim, eps, iters, hc_eps);
    return launch_ok("hc mix");
  }
  return ds41_v41_hc_mixes(fn, h, mixes, hc, dim, eps, n) ||
         ds41_v41_hc_sinkhorn(mixes, scale, base, pre, post, comb, hc, iters, hc_eps, n);
}

extern "C" int ds41_v41_hc_sinkhorn(const float* mixes, const float* scale, const float* base, float* pre,
                                    float* post, float* comb, uint32_t hc, uint32_t iters, float eps, uint32_t n) {
  if (hc != 4) return 1;
  hc_sinkhorn_kernel<4><<<blocks_for(n, 64), 64>>>(mixes, scale, base, pre, post, comb, iters, eps, n);
  return launch_ok("hc sinkhorn");
}

extern "C" int ds41_v41_hc_state_pass(const float* y, const float* post, const float* comb, const float* h,
                                     float* h_out, const float* pre, const float* norm_w, float* xn, const float* fn,
                                     float* mixes, uint32_t hc, uint32_t dim, float eps, uint32_t n) {
  if (hc != 4 || dim % 256 != 0 || dim / 8 > 1024) return 1;
  if (n == 0) return 0;
  const dim3 grid((n + kMixTokens - 1) / kMixTokens);
  if (y)
    hc_state_pass_kernel<true><<<grid, dim / 8>>>(y, post, comb, h, h_out, pre, norm_w, xn, fn, mixes, dim, n, eps);
  else
    hc_state_pass_kernel<false><<<grid, dim / 8>>>(y, post, comb, h, h_out, pre, norm_w, xn, fn, mixes, dim, n, eps);
  return launch_ok("hc state pass");
}

extern "C" int ds41_v41_hc_collapse_norm(const float* h, const float* pre, const float* weight, float* out,
                                         uint32_t hc, uint32_t dim, float eps, uint32_t n) {
  if (hc != 4 || dim > kCollapseThreads * kCollapsePer) return 1;
  if (n == 0) return 0;
  hc_collapse_norm_kernel<4><<<n, kCollapseThreads>>>(h, pre, weight, out, dim, eps);
  return launch_ok("hc collapse norm");
}

extern "C" int ds41_v41_dspark_confidence(const float* h, const float* pre, const float* markov_x, const float* w,
                                          const float* bias, uint32_t dim, uint32_t rank, uint32_t row,
                                          float* out) {
  dspark_confidence_kernel<<<1, kCollapseThreads>>>(h, pre, markov_x, w, bias, dim, rank, row, out);
  return launch_ok("dspark confidence");
}

extern "C" int ds41_v41_hc_expand(const float* y, const float* residual, const float* post, const float* comb,
                                  float* out, uint32_t hc, uint32_t dim, uint32_t n) {
  hc_expand_kernel<<<dim3(blocks_for(dim), n), kThreads>>>(y, residual, post, comb, out, hc, dim);
  return launch_ok("hc expand");
}

extern "C" int ds41_v41_hc_expand_sum(const float* x, const float* extra, uint32_t rows, const float* residual,
                                      const float* post, const float* comb, float* out, uint32_t hc, uint32_t dim,
                                      uint32_t n) {
  hc_expand_sum_kernel<<<dim3(blocks_for(dim), n), kThreads>>>(x, extra, rows, residual, post, comb, out, hc, dim);
  return launch_ok("hc expand sum");
}

extern "C" int ds41_v41_compress_pool(const float* kv, const float* score, float* state_kv, float* state_score,
                                      uint32_t pos0, uint32_t n, uint32_t ratio, float* out, uint32_t dim) {
  compress_pool_kernel<<<blocks_for(dim), kThreads>>>(kv, score, state_kv, state_score, pos0, n, ratio, out, dim);
  return launch_ok("compressor pool");
}

extern "C" int ds41_v41_indexer_score(const float* q, ds41_fp4_rows keys, const float* w, float w_scale,
                                      float* score, uint64_t score_stride, uint32_t pos0, uint32_t n,
                                      uint32_t ratio, uint32_t heads, uint32_t head_dim) {
  if (heads > 64) return 1;
  const uint32_t len = max_comp_len(pos0, n, ratio);
  if (len == 0) return 0;
  if (head_dim == 128 && heads == kScoreHeads && n <= kScoreSmallTokens) {
    indexer_score_small_kernel<false><<<dim3((len + kScoreKeys - 1) / kScoreKeys, n), 256>>>(q, keys, w, w_scale, score, score_stride, pos0,
                                                                    ratio, nullptr, 0);
    return launch_ok("indexer score");
  }
  if (head_dim == 128) {
    indexer_score_wmma_kernel<<<dim3((len + 127) / 128, (n + 15) / 16), 256>>>(q, keys, w, w_scale, score,
                                                                               score_stride, pos0, n, ratio, heads);
    return launch_ok("indexer score");
  }
  indexer_score_kernel<<<dim3(blocks_for(len), n), kThreads, heads * head_dim * sizeof(float)>>>(
      q, keys, w, w_scale, score, score_stride, pos0, ratio, heads, head_dim);
  return launch_ok("indexer score");
}

extern "C" int ds41_v41_indexer_score_candidates(const float* q, ds41_fp4_rows keys, const float* w,
                                                 float w_scale, float* score, uint64_t score_stride,
                                                 const uint32_t* mask, uint64_t mask_stride, uint32_t* candidates,
                                                 uint32_t candidate_stride, uint32_t pos0, uint32_t n,
                                                 uint32_t ratio) {
  if (n == 0 || n > kScoreSmallTokens || candidate_stride == 0) return 1;
  const uint32_t len = max_comp_len(pos0, n, ratio);
  compact_candidate_blocks_kernel<<<n, 256>>>(mask, mask_stride, candidates, candidate_stride, pos0, ratio);
  clear_index_scores_kernel<<<dim3(blocks_for(len), n), kThreads>>>(score, score_stride, pos0, ratio);
  indexer_score_small_kernel<true><<<dim3((min((len + 7) / 8, candidate_stride) * 8 + kScoreKeys - 1) / kScoreKeys, n), 256>>>(
      q, keys, w, w_scale, score, score_stride, pos0, ratio, candidates, candidate_stride);
  return launch_ok("candidate indexer score");
}

extern "C" int ds41_v41_topk(const float* score, uint64_t score_stride, uint32_t* out, uint32_t k, uint32_t pos0,
                             uint32_t n, uint32_t ratio, uint32_t div) {
  if (k == 0 || div == 0) return 1;
  topk_kernel<<<n, 1024>>>(score, score_stride, out, k, pos0, ratio, div);
  return launch_ok("topk");
}

extern "C" int ds41_v41_candidate_mask(const float* score, uint64_t score_stride, uint32_t* mask,
                                       uint64_t mask_stride, float* block_scratch, uint64_t block_stride, uint32_t* select_scratch,
                                       uint32_t pos0, uint32_t n, uint32_t ratio, uint32_t block,
                                       uint32_t topk_blocks) {
  const uint32_t len = max_comp_len(pos0, n, ratio);
  if (len == 0) return 0;
  const uint32_t nb = (len + block - 1) / block;
  clear_mask_kernel<<<dim3(blocks_for((nb + 31) / 32), n), kThreads>>>(mask, mask_stride, pos0, ratio, block);
  block_max_kernel<<<dim3(blocks_for(nb), n), kThreads>>>(score, score_stride, block_scratch, block_stride, pos0,
                                                          ratio, block);
  if (launch_ok("candidate block max")) return 1;
  /* Every block is kept while there are at most topk_blocks of them. */
  if (ds41_v41_topk(block_scratch, block_stride, select_scratch, topk_blocks, pos0, n, ratio, block)) return 1;
  fill_mask_kernel<<<dim3(blocks_for(topk_blocks), n), kThreads>>>(mask, mask_stride, select_scratch, topk_blocks,
                                                                    pos0, ratio, block);
  return launch_ok("candidate mask");
}

extern "C" int ds41_v41_apply_mask(float* score, uint64_t score_stride, const uint32_t* mask, uint64_t mask_stride,
                                   uint32_t pos0, uint32_t n, uint32_t ratio, uint32_t block) {
  const uint32_t len = max_comp_len(pos0, n, ratio);
  if (len == 0) return 0;
  apply_mask_kernel<<<dim3(blocks_for(len), n), kThreads>>>(score, score_stride, mask, mask_stride, pos0, ratio,
                                                            block);
  return launch_ok("apply mask");
}

extern "C" int ds41_v41_attention(float* q, const float* ring, const float* chunk, uint32_t window,
                                  uint32_t floor, ds41_fp4_rows comp, const uint32_t* idx, uint32_t k, uint32_t ratio,
                                  const float* sinks, float* out, float* partial, uint32_t pos0, uint32_t n,
                                  uint32_t heads, uint32_t head_dim, float scale, const float* inv_freq,
                                  uint32_t rope_dim, float* rope_scratch) {
  static_assert(DS41_V41_ATTN_PARTIAL_PER_HEAD == kAttnSplits * kAttnPartial, "partial layout");
  if (inv_freq && (rope_dim % 32 != 0 || rope_dim > head_dim)) return 1;
  /* The matrix-core form once there are enough blocks to fill the device;
   * it applies the rotations itself. */
  if (n >= DS41_V41_ATTN_SPLIT_TOKENS && head_dim == 512 && heads % 32 == 0 && window <= 256 &&
      (window & (window - 1)) == 0 &&
      (ratio == 0 || k <= 512) && (!inv_freq || rope_dim == kAttnRope)) {
    float2* rope = nullptr;
    if (inv_freq) {
      if (!rope_scratch) return 1;
      rope = reinterpret_cast<float2*>(rope_scratch);
      rope_table_kernel<<<blocks_for((uint64_t)n * rope_dim / 2), kThreads>>>(inv_freq, rope, rope_dim / 2, pos0, n);
    }
    attention_f16_kernel<<<dim3(n, heads / 32), 256>>>(q, ring, chunk, window, floor, comp, idx, k, ratio, sinks, out,
                                                       pos0, heads, scale, rope);
    return launch_ok("attention");
  }
  if (inv_freq && ds41_v41_rope_tail(q, n * heads, head_dim, rope_dim, inv_freq, pos0, heads, 1, 0)) return 1;
  if (n < DS41_V41_ATTN_SPLIT_TOKENS && partial && head_dim == 512 && heads % kAttnSplitHeads == 0) {
    attention_split_kernel<<<dim3(heads / kAttnSplitHeads, kAttnSplits, n), 32 * kAttnSplitHeads>>>(
        q, ring, chunk, window, floor, comp, idx, k, ratio, partial, pos0, heads, scale);
    if (launch_ok("attention split")) return 1;
    attention_combine_kernel<<<dim3(heads, n), 128>>>(partial, sinks, out, heads);
    if (launch_ok("attention combine")) return 1;
  } else {
    const size_t shared = (size_t)(window + (ratio ? k : 0)) * sizeof(float);
    if (shared > 60 * 1024) return 1;
    attention_kernel<<<dim3(heads, n), kThreads, shared>>>(q, ring, chunk, window, floor, comp, idx, k, ratio, sinks,
                                                           out, pos0, heads, head_dim, scale);
    if (launch_ok("attention")) return 1;
  }
  return inv_freq ? ds41_v41_rope_fp8(out, n * heads, head_dim, rope_dim, inv_freq, pos0, heads, 1, 1) : 0;
}

extern "C" int ds41_v41_route(const float* logits, const float* bias, int32_t* ids, float* weights,
                              uint32_t n_experts, uint32_t k, float route_scale, uint32_t n) {
  if (n_experts > 512 || k > 16) return 1;
  route_kernel<<<n, kThreads>>>(logits, bias, ids, weights, n_experts, k, route_scale);
  return launch_ok("route");
}

extern "C" int ds41_v41_swiglu(const float* gate, const float* up, const float* weight, float* out, uint32_t rows,
                               uint32_t n, float limit) {
  swiglu_kernel<<<blocks_for((uint64_t)rows * n), kThreads>>>(gate, up, weight, out, rows, n, limit);
  return launch_ok("swiglu");
}

extern "C" int ds41_v41_engram_gate(float* h, const float* kv, const uint16_t* q, const uint16_t* k, uint32_t hc,
                                    uint32_t dim, float eps, uint32_t n) {
  engram_gate_kernel<<<dim3(hc, n), 1024>>>(h, kv, q, k, dim, hc, eps);
  return launch_ok("engram gate");
}

extern "C" int ds41_v41_dspark_attention(const float* q, const float* ring, uint32_t first, uint32_t n_ring,
                                         uint32_t window, const float* block_kv, uint32_t n_block,
                                         const float* sinks, float* out, uint32_t heads, uint32_t head_dim,
                                         float scale) {
  const size_t shared = (size_t)(n_ring + n_block) * sizeof(float);
  if (n_ring > window || shared > 60 * 1024) return 1;
  dspark_attention_kernel<<<dim3(heads, n_block), kThreads, shared>>>(q, ring, first, n_ring, window, block_kv,
                                                                      n_block, sinks, out, heads, head_dim, scale);
  return launch_ok("dspark attention");
}

extern "C" int ds41_v41_hc_mean(const float* h, float* out, uint64_t out_stride, uint32_t hc, uint32_t dim,
                                uint32_t n) {
  if (n == 0) return 0;
  hc_mean_kernel<<<dim3(blocks_for(dim), n), kThreads>>>(h, out, out_stride, hc, dim);
  return launch_ok("hc mean");
}

extern "C" int ds41_v41_gather_row(const void* table, int format, const int32_t* tokens, uint32_t index, float* out,
                                   uint32_t rank) {
  gather_row_kernel<<<1, kThreads>>>(table, format, tokens, index, out, rank);
  return launch_ok("gather row");
}

extern "C" int ds41_v41_top8_bias(const float* x, const float* bias, uint32_t n, int32_t* ids, float* logits) {
  top8_bias_kernel<false><<<1, 256>>>(x, bias, n, ids, logits, nullptr, 1.0f, nullptr, nullptr);
  return launch_ok("biased top8");
}

extern "C" int ds41_v41_markov_argmax(const uint16_t* w2, const float* x, const float* logits, uint32_t n,
                                      float* cand_v, int32_t* cand_i, int32_t* tokens, uint32_t index) {
  markov_select_kernel<1><<<kMarkovBlocks, 256>>>((const _Float16*)w2, x, logits, n, cand_v, cand_i);
  argmax_bias_kernel<<<1, 1024>>>(cand_v, nullptr, kMarkovBlocks, tokens, index, cand_i);
  return launch_ok("markov argmax");
}

extern "C" int ds41_v41_markov_propose(const uint16_t* w2, const float* x, const float* logits, uint32_t n,
                                       float* cand_v, int32_t* cand_i, int32_t* ids, float* top_logits,
                                       const uint32_t* draw, float temperature, ds41_v41_proposal* proposal,
                                       int32_t* token) {
  markov_select_kernel<8><<<kMarkovBlocks, 256>>>((const _Float16*)w2, x, logits, n, cand_v, cand_i);
  top8_bias_kernel<true><<<1, 256>>>(cand_v, nullptr, kMarkovBlocks * 8, ids, top_logits, draw, temperature,
                                     proposal, token, cand_i);
  return launch_ok("markov proposal");
}

extern "C" int ds41_v41_propose_top8(const float* x, const float* bias, uint32_t n, int32_t* ids, float* logits,
                                     const uint32_t* draw, float temperature, ds41_v41_proposal* proposal,
                                     int32_t* token) {
  top8_bias_kernel<true><<<1, 256>>>(x, bias, n, ids, logits, draw, temperature, proposal, token);
  return launch_ok("top8 proposal");
}

extern "C" int ds41_v41_argmax_bias(const float* x, const float* bias, uint32_t n, int32_t* tokens, uint32_t index) {
  argmax_bias_kernel<<<1, 1024>>>(x, bias, n, tokens, index);
  return launch_ok("argmax");
}
