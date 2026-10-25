/* DeepSeek V4.1 Flash decode and prefill kernels; see ds41_v41.h. */
#include "ds41_v41.h"

#include <hip/hip_fp16.h>
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
    if constexpr (sizeof(W) == 2)
      wv = bf16_to_f32(r[i]);
    else
      wv = r[i];
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

/* Thread per 32-value block: the rope_tail rotation of the block's pairs
 * when it lies in the rope tail, then the FP8 fake quantization. */
__global__ void rope_fp8_kernel(float* x, uint32_t blocks, uint32_t row_dim, uint32_t rope_dim,
                                const float* inv_freq, uint32_t pos0, uint32_t rows_per_pos, uint32_t pos_step,
                                int inverse) {
  const uint32_t b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= blocks) return;
  const uint32_t per_row = row_dim / 32;
  const uint32_t row = b / per_row;
  const uint32_t col = (b % per_row) * 32;
  float* p = x + (uint64_t)b * 32;
  float v[32];
#pragma unroll
  for (int i = 0; i < 32; i++) v[i] = p[i];
  if (col >= row_dim - rope_dim) {
    const uint32_t pos = pos0 + (row / rows_per_pos) * pos_step;
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
  for (int i = 0; i < 32; i++) p[i] = round_e4m3(fminf(fmaxf(v[i] / scale, -448.0f), 448.0f)) * scale;
}

__global__ void fp4_fake_quant_kernel(float* x, uint32_t blocks, uint32_t block, int e4m3_scale) {
  const uint32_t b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= blocks) return;
  float* v = x + (uint64_t)b * block;
  float amax = 0.0f;
  for (uint32_t i = 0; i < block; i++) amax = fmaxf(amax, fabsf(v[i]));
  const float scale = e4m3_scale ? round_e4m3(fmaxf(amax, 6.0f * 0.001953125f) / 6.0f)
                                 : pow2_ceil(fmaxf(amax, 6.0f * 1.1754943508e-38f) / 6.0f);
  for (uint32_t i = 0; i < block; i++) v[i] = round_e2m1(fminf(fmaxf(v[i] / scale, -6.0f), 6.0f)) * scale;
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

template <uint32_t hc>
__global__ void hc_sinkhorn_kernel(const float* all_mixes, const float* scale, const float* base, float* all_pre,
                                   float* all_post, float* all_comb, uint32_t iters, float eps, uint32_t n) {
  const uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= n) return;
  sinkhorn_token<hc>(all_mixes + (uint64_t)t * (2 * hc + hc * hc), scale, base, all_pre + (uint64_t)t * hc,
                     all_post + (uint64_t)t * hc, all_comb + (uint64_t)t * hc * hc, iters, eps);
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
__global__ void indexer_score_kernel(const float* q, const float* keys, const float* w, float w_scale, float* score,
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
  const float* k = keys + (uint64_t)j * head_dim;
  float acc = 0.0f;
  for (uint32_t h = 0; h < heads; h++) {
    float dot = 0.0f;
    for (uint32_t d = 0; d < head_dim; d++) dot += sq[h * head_dim + d] * k[d];
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
__global__ __launch_bounds__(256) void indexer_score_small_kernel(const float* q, const float* keys, const float* w,
                                                                  float w_scale, float* score, uint64_t score_stride,
                                                                  uint32_t pos0, uint32_t ratio) {
  constexpr uint32_t DIM = 128, D4 = DIM / 4, PITCH4 = D4 + 1, GROUPS = 256 / kScoreKeys,
                     HG = kScoreHeads / GROUPS;
  const uint32_t token = blockIdx.y;
  const uint32_t len = comp_len(pos0 + token, ratio);
  const uint32_t key0 = blockIdx.x * kScoreKeys;
  if (key0 >= len) return;
  __shared__ float4 sk[kScoreKeys * PITCH4];
  __shared__ float4 sq[kScoreHeads * D4];
  __shared__ float sw[kScoreHeads];
  __shared__ float part[GROUPS][kScoreKeys];
  const float4* qt = reinterpret_cast<const float4*>(q + (uint64_t)token * kScoreHeads * DIM);
  for (uint32_t i = threadIdx.x; i < kScoreHeads * D4; i += 256) sq[i] = qt[i];
  const float4* kt = reinterpret_cast<const float4*>(keys + (uint64_t)key0 * DIM);
  const uint32_t rows = min(kScoreKeys, len - key0);
  for (uint32_t i = threadIdx.x; i < kScoreKeys * D4; i += 256) {
    const uint32_t r = i / D4;
    sk[r * PITCH4 + i % D4] = r < rows ? kt[i] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
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
  score[token * score_stride + key0 + key] = acc;
}

/* The same scores on the matrix cores, after the V4 engine's
 * indexer_scores_wmma128_kernel: a block covers 16 tokens and 128 compressed
 * rows, stages the rows once as half and each head's queries (double
 * buffered), and keeps the relu-weighted head sum in fp32. head_dim 128,
 * 256 threads; grid (ceil(len / 128), ceil(n / 16)). */
__global__ __launch_bounds__(256) void indexer_score_wmma_kernel(const float* q, const float* keys, const float* w,
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
  for (uint32_t i4 = tid; i4 < 128 * (DIM / 4); i4 += 256) {
    const uint32_t c = i4 / (DIM / 4);
    const uint32_t d = (i4 % (DIM / 4)) * 4;
    uint2 packed = make_uint2(0u, 0u);
    if (tile_c + c < visible_max) {
      const float4 v4 = *reinterpret_cast<const float4*>(keys + (uint64_t)(tile_c + c) * DIM + d);
      const __half2 h_lo = __floats2half2_rn(v4.x, v4.y);
      const __half2 h_hi = __floats2half2_rn(v4.z, v4.w);
      packed.x = *reinterpret_cast<const uint32_t*>(&h_lo);
      packed.y = *reinterpret_cast<const uint32_t*>(&h_hi);
    }
    *reinterpret_cast<uint2*>(&k_sh[c * DIM + d]) = packed;
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

__global__ void clear_mask_kernel(uint8_t* mask, uint64_t stride, uint32_t pos0, uint32_t ratio) {
  const uint32_t token = blockIdx.y;
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < comp_len(pos0 + token, ratio)) mask[token * stride + i] = 0;
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

/* Block (selected block, token). */
__global__ void fill_mask_kernel(uint8_t* mask, uint64_t stride, const uint32_t* selected, uint32_t k_max,
                                 uint32_t pos0, uint32_t ratio, uint32_t block) {
  const uint32_t token = blockIdx.y;
  const uint32_t n = comp_len(pos0 + token, ratio);
  const uint32_t j = blockIdx.x;
  if (j >= min(k_max, (n + block - 1) / block)) return;
  const uint32_t b = selected[(uint64_t)token * k_max + j];
  for (uint32_t i = b * block + threadIdx.x; i < min(n, (b + 1) * block); i += blockDim.x)
    mask[token * stride + i] = 1;
}

__global__ void apply_mask_kernel(float* score, uint64_t stride, const uint8_t* mask, uint32_t pos0, uint32_t ratio) {
  const uint32_t token = blockIdx.y;
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < comp_len(pos0 + token, ratio) && !mask[token * stride + i]) score[token * stride + i] = -INFINITY;
}

/* Block (head, token). Scores for every key are kept in shared memory. */
__global__ void attention_kernel(const float* q, const float* ring, const float* chunk, uint32_t window,
                                 const float* comp, const uint32_t* all_idx, uint32_t k, uint32_t ratio,
                                 const float* sinks, float* out, uint32_t pos0, uint32_t heads, uint32_t head_dim,
                                 float scale) {
  extern __shared__ float scores[];
  const uint32_t head = blockIdx.x;
  const uint32_t token = blockIdx.y;
  const uint32_t p = pos0 + token;
  const uint32_t lo = p + 1 > window ? p + 1 - window : 0u;
  const uint32_t n_window = p - lo + 1;
  const uint32_t n_comp = min(k, comp_len(p, ratio));
  const uint32_t* idx = all_idx + (uint64_t)token * k;
  const float* qh = q + ((uint64_t)token * heads + head) * head_dim;
  auto key_row = [&](uint32_t j) -> const float* {
    if (j >= n_window) return comp + (uint64_t)idx[j - n_window] * head_dim;
    const uint32_t pos = lo + j;
    return pos < pos0 ? ring + (uint64_t)(pos % window) * head_dim : chunk + (uint64_t)(pos - pos0) * head_dim;
  };
  const uint32_t n = n_window + n_comp;
  const uint32_t warps = blockDim.x / warpSize;
  const uint32_t warp = threadIdx.x / warpSize;
  const uint32_t lane = threadIdx.x % warpSize;
  for (uint32_t j = warp; j < n; j += warps) {
    const float* key = key_row(j);
    float dot = 0.0f;
    for (uint32_t d = lane; d < head_dim; d += warpSize) dot += qh[d] * key[d];
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
    for (uint32_t j = 0; j < n; j++) acc += scores[j] * key_row(j)[d];
    out[((uint64_t)token * heads + head) * head_dim + d] = acc / sum;
  }
}

/* The same attention for 32 heads per block on the matrix cores, from the V4
 * engine's attention_mixed_heads32_wmma_kernel: Q and each 16-row KV tile are
 * staged as half, scores and the online softmax stay in fp32, and the output
 * accumulates in fp32 across one pass over the rows. Grid (n, heads / 32),
 * 1024 threads; head_dim 512, window <= 256, k <= 512. */
__global__ __launch_bounds__(1024, 1) void attention_heads32_wmma_kernel(
    const float* q, const float* ring, const float* chunk, uint32_t window, const float* comp,
    const uint32_t* all_idx, uint32_t k, uint32_t ratio, const float* sinks, float* out, uint32_t pos0,
    uint32_t heads, float scale) {
  constexpr uint32_t BM = 16, BN = 16, BK = 16;
  constexpr uint32_t HEADS = 32, ROWS = 16, DIM = 512, LDS_DIM = DIM + 4;
  constexpr uint32_t QT_PITCH = HEADS + 2;
  const uint32_t token = blockIdx.x;
  const uint32_t head0 = blockIdx.y * HEADS;
  const uint32_t tid = threadIdx.x;
  const uint32_t wave = tid / 32;
  const uint32_t lane = tid % 32;
  const uint32_t p = pos0 + token;
  const uint32_t lo = p + 1 > window ? p + 1 - window : 0u;
  const uint32_t n_window = p - lo + 1;
  const uint32_t n_comp = min(k, comp_len(p, ratio));
  const uint32_t n_score = n_window + n_comp;

  __shared__ uint32_t rows[256 + 512];
  /* Q transposed as [d][head], so both score operands are row-major. */
  __shared__ half qt_half[DIM * QT_PITCH];
  __shared__ half kv_half[ROWS * LDS_DIM];
  __shared__ float scores[HEADS * ROWS];
  __shared__ float qk_part[2 * BM * ROWS];
  __shared__ half probs[HEADS * ROWS];
  __shared__ float softmax_max[HEADS];
  __shared__ float softmax_den[HEADS];
  __shared__ float softmax_rescale[HEADS];

  /* Row r < n_window is window position lo + r: chunk row (pos - pos0) when
   * it is in this batch, else ring slot pos % window, tagged with bit 31.
   * Later rows are compressed rows. */
  for (uint32_t r = tid; r < n_window; r += blockDim.x) {
    const uint32_t pos = lo + r;
    rows[r] = pos < pos0 ? (pos % window) | 0x80000000u : pos - pos0;
  }
  for (uint32_t r = tid; r < n_comp; r += blockDim.x) rows[n_window + r] = all_idx[(uint64_t)token * k + r];
  for (uint32_t j = tid; j < HEADS * DIM; j += blockDim.x) {
    const uint32_t h = j / DIM;
    const uint32_t d = j - h * DIM;
    qt_half[d * QT_PITCH + h] = __float2half(q[((uint64_t)token * heads + head0 + h) * DIM + d]);
  }
  if (tid < HEADS) {
    softmax_max[tid] = sinks[head0 + tid];
    softmax_den[tid] = 1.0f;
  }
  __syncthreads();

  using frag_a = rocwmma::fragment<rocwmma::matrix_a, BM, BN, BK, half, rocwmma::row_major>;
  using frag_b = rocwmma::fragment<rocwmma::matrix_b, BM, BN, BK, half, rocwmma::row_major>;
  using frag_c = rocwmma::fragment<rocwmma::accumulator, BM, BN, BK, float>;
  frag_c out0, out1;
  rocwmma::fill_fragment(out0, 0.0f);
  rocwmma::fill_fragment(out1, 0.0f);
  /* Accumulator element (row, col) of a wave32 16x16 F32 fragment on gfx1151
   * is (2 * e + lane / 16, lane % 16); out0 rows are heads 0..15, out1 16..31. */
  const uint32_t acc_row_base = lane / 16;

  for (uint32_t row0 = 0; row0 < n_score; row0 += ROWS) {
    const uint32_t nr = min(ROWS, n_score - row0);
    constexpr uint32_t DIM4 = DIM / 4;
    for (uint32_t j = tid; j < ROWS * DIM4; j += blockDim.x) {
      const uint32_t rr = j / DIM4;
      const uint32_t d = (j - rr * DIM4) * 4;
      uint2 packed = make_uint2(0u, 0u);
      if (rr < nr) {
        const uint32_t sr = row0 + rr;
        const uint32_t row = rows[sr];
        const float* src = sr >= n_window          ? comp + (uint64_t)row * DIM
                           : (row & 0x80000000u) ? ring + (uint64_t)(row & 0x7fffffffu) * DIM
                                                 : chunk + (uint64_t)row * DIM;
        const float4 v4 = *reinterpret_cast<const float4*>(src + d);
        const __half2 h_lo = __floats2half2_rn(v4.x, v4.y);
        const __half2 h_hi = __floats2half2_rn(v4.z, v4.w);
        packed.x = *reinterpret_cast<const uint32_t*>(&h_lo);
        packed.y = *reinterpret_cast<const uint32_t*>(&h_hi);
      }
      *reinterpret_cast<uint2*>(&kv_half[rr * LDS_DIM + d]) = packed;
    }
    __syncthreads();

    /* scoresT = KV . Q^T over two K halves per head block (waves 0..3). */
    const uint32_t qk_head_block = wave / 2;
    const uint32_t qk_group = wave % 2;
    if (wave < 4) {
      frag_c score_acc;
      rocwmma::fill_fragment(score_acc, 0.0f);
      for (uint32_t k0 = qk_group * (DIM / 2); k0 < (qk_group + 1) * (DIM / 2); k0 += BK) {
        frag_a ka;
        frag_b qb;
        rocwmma::load_matrix_sync(ka, kv_half + k0, LDS_DIM);
        rocwmma::load_matrix_sync(qb, qt_half + k0 * QT_PITCH + qk_head_block * BM, QT_PITCH);
        rocwmma::mma_sync(score_acc, ka, qb, score_acc);
      }
      if (qk_group == 0)
        rocwmma::store_matrix_sync(scores + qk_head_block * BM, score_acc, HEADS, rocwmma::mem_row_major);
      else
        rocwmma::store_matrix_sync(qk_part + qk_head_block * BM * ROWS, score_acc, ROWS, rocwmma::mem_row_major);
    }
    __syncthreads();
    for (uint32_t j = tid; j < 2 * BM * ROWS; j += blockDim.x) {
      const uint32_t b = j / (BM * ROWS);
      const uint32_t o = j - b * (BM * ROWS);
      const uint32_t r = o / BM;
      scores[r * HEADS + b * BM + (o - r * BM)] += qk_part[j];
    }
    __syncthreads();

    /* Online softmax: one wave per head, one row per lane. */
    {
      const float s = lane < nr ? scores[lane * HEADS + wave] * scale : -INFINITY;
      float m_new = s;
      for (int off = 16; off > 0; off >>= 1) m_new = fmaxf(m_new, __shfl_xor(m_new, off, 32));
      const float m_old = softmax_max[wave];
      m_new = fmaxf(m_old, m_new);
      const float e = lane < nr ? expf(s - m_new) : 0.0f;
      float block_sum = e;
      for (int off = 16; off > 0; off >>= 1) block_sum += __shfl_xor(block_sum, off, 32);
      if (lane == 0) {
        const float factor = expf(m_old - m_new);
        softmax_den[wave] = softmax_den[wave] * factor + block_sum;
        softmax_max[wave] = m_new;
        softmax_rescale[wave] = factor;
      }
      if (lane < ROWS) probs[wave * ROWS + lane] = __float2half(e);
    }
    __syncthreads();

    if (row0 != 0) {
#pragma unroll
      for (uint32_t e = 0; e < out0.num_elements; e++) {
        const uint32_t r = 2 * e + acc_row_base;
        out0.x[e] *= softmax_rescale[r];
        out1.x[e] *= softmax_rescale[BM + r];
      }
    }
    frag_a p0, p1;
    frag_b vb;
    rocwmma::load_matrix_sync(p0, probs, ROWS);
    rocwmma::load_matrix_sync(p1, probs + BM * ROWS, ROWS);
    rocwmma::load_matrix_sync(vb, kv_half + wave * BN, LDS_DIM);
    rocwmma::mma_sync(out0, p0, vb, out0);
    rocwmma::mma_sync(out1, p1, vb, out1);
    __syncthreads();
  }

#pragma unroll
  for (uint32_t e = 0; e < out0.num_elements; e++) {
    const uint32_t r = 2 * e + acc_row_base;
    out0.x[e] /= softmax_den[r];
    out1.x[e] /= softmax_den[BM + r];
  }
  float* o = out + ((uint64_t)token * heads + head0) * DIM;
  rocwmma::store_matrix_sync(o + wave * BN, out0, DIM, rocwmma::mem_row_major);
  rocwmma::store_matrix_sync(o + BM * DIM + wave * BN, out1, DIM, rocwmma::mem_row_major);
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
                                                              uint32_t window, const float* comp,
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
  const uint32_t lo = p + 1 > window ? p + 1 - window : 0u;
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
      const float* src;
      if (j >= n_window) {
        src = comp + (uint64_t)idx[j - n_window] * DIM;
      } else {
        const uint32_t pos = lo + j;
        src = pos < pos0 ? ring + (uint64_t)(pos % window) * DIM : chunk + (uint64_t)(pos - pos0) * DIM;
      }
      kv_sh[i] = reinterpret_cast<const float4*>(src)[i % (DIM / 4)];
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

__global__ void sum_rows_kernel(const float* x, const float* extra, float* out, uint32_t rows, uint32_t dim) {
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= dim) return;
  const uint64_t t = blockIdx.y;
  float acc = extra ? extra[t * dim + i] : 0.0f;
  for (uint32_t r = 0; r < rows; r++) acc += x[(t * rows + r) * dim + i];
  out[t * dim + i] = acc;
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
  if (k % 256 == 0 && (uintptr_t)w % 16 == 0 && (uintptr_t)x % 16 == 0)
    matmul_bf16_vec8_kernel<<<grid, kThreads>>>(w, x, out, rows, k, n);
  else
    matmul_kernel<uint16_t><<<grid, kThreads>>>(w, x, out, rows, k, n);
  return launch_ok("bf16 matmul");
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

extern "C" int ds41_v41_fp4_fake_quant(float* x, uint32_t n, uint32_t block, int e4m3_scale) {
  if (n == 0) return 0;
  fp4_fake_quant_kernel<<<blocks_for(n / block), kThreads>>>(x, n / block, block, e4m3_scale);
  return launch_ok("fp4 quant");
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

extern "C" int ds41_v41_hc_collapse_norm(const float* h, const float* pre, const float* weight, float* out,
                                         uint32_t hc, uint32_t dim, float eps, uint32_t n) {
  if (hc != 4 || dim > kCollapseThreads * kCollapsePer) return 1;
  if (n == 0) return 0;
  hc_collapse_norm_kernel<4><<<n, kCollapseThreads>>>(h, pre, weight, out, dim, eps);
  return launch_ok("hc collapse norm");
}

extern "C" int ds41_v41_hc_expand(const float* y, const float* residual, const float* post, const float* comb,
                                  float* out, uint32_t hc, uint32_t dim, uint32_t n) {
  hc_expand_kernel<<<dim3(blocks_for(dim), n), kThreads>>>(y, residual, post, comb, out, hc, dim);
  return launch_ok("hc expand");
}

extern "C" int ds41_v41_compress_pool(const float* kv, const float* score, float* state_kv, float* state_score,
                                      uint32_t pos0, uint32_t n, uint32_t ratio, float* out, uint32_t dim) {
  compress_pool_kernel<<<blocks_for(dim), kThreads>>>(kv, score, state_kv, state_score, pos0, n, ratio, out, dim);
  return launch_ok("compressor pool");
}

extern "C" int ds41_v41_indexer_score(const float* q, const float* keys, const float* w, float w_scale,
                                      float* score, uint64_t score_stride, uint32_t pos0, uint32_t n,
                                      uint32_t ratio, uint32_t heads, uint32_t head_dim) {
  if (heads > 64) return 1;
  const uint32_t len = max_comp_len(pos0, n, ratio);
  if (len == 0) return 0;
  if (head_dim == 128 && heads == kScoreHeads && n <= kScoreSmallTokens) {
    indexer_score_small_kernel<<<dim3((len + kScoreKeys - 1) / kScoreKeys, n), 256>>>(q, keys, w, w_scale, score, score_stride, pos0,
                                                                    ratio);
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

extern "C" int ds41_v41_topk(const float* score, uint64_t score_stride, uint32_t* out, uint32_t k, uint32_t pos0,
                             uint32_t n, uint32_t ratio, uint32_t div) {
  if (k == 0 || div == 0) return 1;
  topk_kernel<<<n, 1024>>>(score, score_stride, out, k, pos0, ratio, div);
  return launch_ok("topk");
}

extern "C" int ds41_v41_candidate_mask(const float* score, uint64_t score_stride, uint8_t* mask,
                                       float* block_scratch, uint64_t block_stride, uint32_t* select_scratch,
                                       uint32_t pos0, uint32_t n, uint32_t ratio, uint32_t block,
                                       uint32_t topk_blocks) {
  const uint32_t len = max_comp_len(pos0, n, ratio);
  if (len == 0) return 0;
  const uint32_t nb = (len + block - 1) / block;
  clear_mask_kernel<<<dim3(blocks_for(len), n), kThreads>>>(mask, score_stride, pos0, ratio);
  block_max_kernel<<<dim3(blocks_for(nb), n), kThreads>>>(score, score_stride, block_scratch, block_stride, pos0,
                                                          ratio, block);
  if (launch_ok("candidate block max")) return 1;
  /* Every block is kept while there are at most topk_blocks of them. */
  if (ds41_v41_topk(block_scratch, block_stride, select_scratch, topk_blocks, pos0, n, ratio, block)) return 1;
  fill_mask_kernel<<<dim3(topk_blocks, n), 32>>>(mask, score_stride, select_scratch, topk_blocks, pos0, ratio,
                                                 block);
  return launch_ok("candidate mask");
}

extern "C" int ds41_v41_apply_mask(float* score, uint64_t score_stride, const uint8_t* mask, uint32_t pos0,
                                   uint32_t n, uint32_t ratio) {
  const uint32_t len = max_comp_len(pos0, n, ratio);
  if (len == 0) return 0;
  apply_mask_kernel<<<dim3(blocks_for(len), n), kThreads>>>(score, score_stride, mask, pos0, ratio);
  return launch_ok("apply mask");
}

extern "C" int ds41_v41_attention(const float* q, const float* ring, const float* chunk, uint32_t window,
                                  const float* comp, const uint32_t* idx, uint32_t k, uint32_t ratio,
                                  const float* sinks, float* out, float* partial, uint32_t pos0, uint32_t n,
                                  uint32_t heads, uint32_t head_dim, float scale) {
  static_assert(DS41_V41_ATTN_PARTIAL_PER_HEAD == kAttnSplits * kAttnPartial, "partial layout");
  if (n < DS41_V41_ATTN_SPLIT_TOKENS && partial && head_dim == 512 && heads % kAttnSplitHeads == 0) {
    attention_split_kernel<<<dim3(heads / kAttnSplitHeads, kAttnSplits, n), 32 * kAttnSplitHeads>>>(
        q, ring, chunk, window, comp, idx, k, ratio, partial, pos0, heads, scale);
    if (launch_ok("attention split")) return 1;
    attention_combine_kernel<<<dim3(heads, n), 128>>>(partial, sinks, out, heads);
    return launch_ok("attention combine");
  }
  /* The matrix-core form once there are enough blocks to fill the device. */
  if (n >= DS41_V41_ATTN_SPLIT_TOKENS && head_dim == 512 && heads % 32 == 0 && window <= 256 && (ratio == 0 || k <= 512)) {
    attention_heads32_wmma_kernel<<<dim3(n, heads / 32), 1024>>>(q, ring, chunk, window, comp, idx, k, ratio, sinks,
                                                                 out, pos0, heads, scale);
    return launch_ok("attention");
  }
  const size_t shared = (size_t)(window + (ratio ? k : 0)) * sizeof(float);
  if (shared > 60 * 1024) return 1;
  attention_kernel<<<dim3(heads, n), kThreads, shared>>>(q, ring, chunk, window, comp, idx, k, ratio, sinks, out,
                                                         pos0, heads, head_dim, scale);
  return launch_ok("attention");
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

extern "C" int ds41_v41_sum_rows(const float* x, const float* extra, float* out, uint32_t rows, uint32_t dim,
                                 uint32_t n) {
  sum_rows_kernel<<<dim3(blocks_for(dim), n), kThreads>>>(x, extra, out, rows, dim);
  return launch_ok("sum rows");
}

extern "C" int ds41_v41_engram_gate(float* h, const float* kv, const uint16_t* q, const uint16_t* k, uint32_t hc,
                                    uint32_t dim, float eps, uint32_t n) {
  engram_gate_kernel<<<dim3(hc, n), 1024>>>(h, kv, q, k, dim, hc, eps);
  return launch_ok("engram gate");
}
