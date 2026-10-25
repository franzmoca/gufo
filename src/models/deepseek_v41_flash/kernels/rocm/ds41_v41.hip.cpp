/* DeepSeek V4.1 Flash single-token decode kernels; see ds41_v41.h. */
#include "ds41_v41.h"

#include <hip/hip_runtime.h>

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

__global__ void matvec_bf16_kernel(const uint16_t* w, const float* x, float* out, uint32_t rows, uint32_t k) {
  const uint32_t row = blockIdx.x * (blockDim.x / warpSize) + threadIdx.x / warpSize;
  if (row >= rows) return;
  const uint16_t* r = w + (uint64_t)row * k;
  float acc = 0.0f;
  for (uint32_t i = threadIdx.x % warpSize; i < k; i += warpSize) acc += bf16_to_f32(r[i]) * x[i];
  for (int o = warpSize / 2; o > 0; o >>= 1) acc += __shfl_xor(acc, o);
  if (threadIdx.x % warpSize == 0) out[row] = acc;
}

__global__ void matvec_f32_kernel(const float* w, const float* x, float* out, uint32_t rows, uint32_t k) {
  const uint32_t row = blockIdx.x;
  const float* r = w + (uint64_t)row * k;
  float acc = 0.0f;
  for (uint32_t i = threadIdx.x; i < k; i += blockDim.x) acc += r[i] * x[i];
  acc = block_sum(acc);
  if (threadIdx.x == 0) out[row] = acc;
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
                                 const float* inv_freq, uint32_t pos, int inverse) {
  const uint32_t half = rope_dim / 2;
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= rows * half) return;
  const uint32_t row = i / half;
  const uint32_t pair = i % half;
  float* p = x + (uint64_t)row * row_dim + (row_dim - rope_dim) + 2 * pair;
  const float angle = (float)pos * inv_freq[pair];
  const float c = cosf(angle);
  const float s = inverse ? -sinf(angle) : sinf(angle);
  const float re = p[0];
  const float im = p[1];
  p[0] = re * c - im * s;
  p[1] = re * s + im * c;
}

__global__ void fp8_fake_quant_kernel(float* x, uint32_t blocks) {
  const uint32_t b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= blocks) return;
  float* v = x + (uint64_t)b * 32;
  float amax = 0.0f;
  for (int i = 0; i < 32; i++) amax = fmaxf(amax, fabsf(v[i]));
  const float scale = pow2_ceil(fmaxf(amax, 1.0e-4f) / 448.0f);
  for (int i = 0; i < 32; i++) v[i] = round_e4m3(fminf(fmaxf(v[i] / scale, -448.0f), 448.0f)) * scale;
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

__global__ void hc_mixes_kernel(const float* fn, const float* h, float* mixes, uint32_t n, float eps) {
  const float* r = fn + (uint64_t)blockIdx.x * n;
  float dot = 0.0f;
  float ss = 0.0f;
  for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
    dot += r[i] * h[i];
    ss += h[i] * h[i];
  }
  dot = block_sum(dot);
  ss = block_sum(ss);
  if (threadIdx.x == 0) mixes[blockIdx.x] = dot * rsqrtf(ss / (float)n + eps);
}

__device__ __forceinline__ float sigmoidf(float x) { return 1.0f / (1.0f + expf(-x)); }

__global__ void hc_sinkhorn_kernel(const float* mixes, const float* scale, const float* base, float* pre,
                                   float* post, float* comb, uint32_t hc, uint32_t iters, float eps) {
  if (threadIdx.x != 0) return;
  float c[8][8];
  for (uint32_t i = 0; i < hc; i++) {
    pre[i] = sigmoidf(mixes[i] * scale[0] + base[i]) + eps;
    post[i] = 2.0f * sigmoidf(mixes[hc + i] * scale[1] + base[hc + i]);
  }
  for (uint32_t a = 0; a < hc; a++) {
    float m = -INFINITY;
    for (uint32_t b = 0; b < hc; b++) {
      c[a][b] = mixes[2 * hc + a * hc + b] * scale[2] + base[2 * hc + a * hc + b];
      m = fmaxf(m, c[a][b]);
    }
    float sum = 0.0f;
    for (uint32_t b = 0; b < hc; b++) {
      c[a][b] = expf(c[a][b] - m);
      sum += c[a][b];
    }
    for (uint32_t b = 0; b < hc; b++) c[a][b] = c[a][b] / sum + eps;
  }
  auto normalize_columns = [&] {
    for (uint32_t b = 0; b < hc; b++) {
      float sum = 0.0f;
      for (uint32_t a = 0; a < hc; a++) sum += c[a][b];
      for (uint32_t a = 0; a < hc; a++) c[a][b] /= sum + eps;
    }
  };
  normalize_columns();
  for (uint32_t it = 1; it < iters; it++) {
    for (uint32_t a = 0; a < hc; a++) {
      float sum = 0.0f;
      for (uint32_t b = 0; b < hc; b++) sum += c[a][b];
      for (uint32_t b = 0; b < hc; b++) c[a][b] /= sum + eps;
    }
    normalize_columns();
  }
  for (uint32_t a = 0; a < hc; a++)
    for (uint32_t b = 0; b < hc; b++) comb[a * hc + b] = c[a][b];
}

__global__ void hc_collapse_kernel(const float* h, const float* pre, float* out, uint32_t hc, uint32_t dim) {
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= dim) return;
  float acc = 0.0f;
  for (uint32_t c = 0; c < hc; c++) acc += pre[c] * h[(uint64_t)c * dim + i];
  out[i] = acc;
}

__global__ void hc_expand_kernel(const float* y, const float* residual, const float* post, const float* comb,
                                 float* out, uint32_t hc, uint32_t dim) {
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= dim) return;
  for (uint32_t b = 0; b < hc; b++) {
    float acc = post[b] * y[i];
    for (uint32_t a = 0; a < hc; a++) acc += comb[a * hc + b] * residual[(uint64_t)a * dim + i];
    out[(uint64_t)b * dim + i] = acc;
  }
}

__global__ void compress_pool_kernel(const float* kv, const float* score, float* state_kv, float* state_score,
                                     uint32_t slot, uint32_t ratio, int emit, float* out, uint32_t dim) {
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= dim) return;
  state_kv[(uint64_t)slot * dim + i] = kv[i];
  state_score[(uint64_t)slot * dim + i] = score[i];
  if (!emit) return;
  float m = -INFINITY;
  for (uint32_t s = 0; s < ratio; s++) m = fmaxf(m, state_score[(uint64_t)s * dim + i]);
  float sum = 0.0f;
  float acc = 0.0f;
  for (uint32_t s = 0; s < ratio; s++) {
    const float e = expf(state_score[(uint64_t)s * dim + i] - m);
    sum += e;
    acc += e * state_kv[(uint64_t)s * dim + i];
  }
  out[i] = acc / sum;
}

/* One thread per key; the query heads live in shared memory. */
__global__ void indexer_score_kernel(const float* q, const float* keys, const float* w, float w_scale, float* score,
                                     uint32_t n, uint32_t heads, uint32_t head_dim) {
  extern __shared__ float sq[];
  for (uint32_t i = threadIdx.x; i < heads * head_dim; i += blockDim.x) sq[i] = q[i];
  __shared__ float sw[64];
  for (uint32_t i = threadIdx.x; i < heads; i += blockDim.x) sw[i] = w[i] * w_scale;
  __syncthreads();
  const uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= n) return;
  const float* k = keys + (uint64_t)t * head_dim;
  float acc = 0.0f;
  for (uint32_t h = 0; h < heads; h++) {
    float dot = 0.0f;
    for (uint32_t d = 0; d < head_dim; d++) dot += sq[h * head_dim + d] * k[d];
    acc += fmaxf(dot, 0.0f) * sw[h];
  }
  score[t] = acc;
}

__device__ __forceinline__ uint32_t ordered_key(float v) {
  uint32_t u = __float_as_uint(v);
  if ((u & 0x7fffffffu) == 0u) u = 0u;
  return (u & 0x80000000u) ? ~u : (u ^ 0x80000000u);
}

/* Single-block top-k: binary search for the k-th largest key, then emit keys
 * above it and the lowest-index ties, both in index order. */
__global__ void topk_kernel(const float* score, uint32_t* out, uint32_t n, uint32_t k) {
  __shared__ uint32_t threshold;
  const uint32_t per = (n + blockDim.x - 1) / blockDim.x;
  const uint32_t begin = min(n, threadIdx.x * per);
  const uint32_t end = min(n, begin + per);
  uint32_t lo = 0u;
  uint32_t hi = 0xffffffffu;
  /* Largest t with count(key >= t) >= k. */
  while (lo < hi) {
    const uint32_t mid = lo + (uint32_t)(((uint64_t)hi - lo + 1u) / 2u);
    uint32_t count = 0;
    for (uint32_t i = begin; i < end; i++) count += ordered_key(score[i]) >= mid;
    uint32_t total = 0;
    (void)block_exclusive_scan(count, &total);
    if (total >= k)
      lo = mid;
    else
      hi = mid - 1u;
  }
  if (threadIdx.x == 0) threshold = lo;
  __syncthreads();
  const uint32_t t = threshold;
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

__global__ void block_max_kernel(const float* score, float* blocks, uint32_t n, uint32_t block, uint32_t nb) {
  const uint32_t b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= nb) return;
  float m = -INFINITY;
  for (uint32_t i = b * block; i < min(n, (b + 1) * block); i++) m = fmaxf(m, score[i]);
  blocks[b] = b == nb - 1 ? INFINITY : m;
}

__global__ void fill_mask_kernel(uint8_t* mask, const uint32_t* selected, uint32_t k, uint32_t n, uint32_t block) {
  const uint32_t j = blockIdx.x;
  if (j >= k) return;
  const uint32_t b = selected[j];
  for (uint32_t i = b * block + threadIdx.x; i < min(n, (b + 1) * block); i += blockDim.x) mask[i] = 1;
}

__global__ void apply_mask_kernel(float* score, const uint8_t* mask, uint32_t n) {
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n && !mask[i]) score[i] = -INFINITY;
}

/* One block per head. Scores for every key are kept in shared memory. */
__global__ void attention_kernel(const float* q, const float* window, uint32_t n_window, const float* comp,
                                 const uint32_t* idx, uint32_t n_comp, const float* sinks, float* out,
                                 uint32_t head_dim, float scale) {
  extern __shared__ float scores[];
  const uint32_t head = blockIdx.x;
  const float* qh = q + (uint64_t)head * head_dim;
  const uint32_t n = n_window + n_comp;
  const uint32_t warps = blockDim.x / warpSize;
  const uint32_t warp = threadIdx.x / warpSize;
  const uint32_t lane = threadIdx.x % warpSize;
  for (uint32_t j = warp; j < n; j += warps) {
    const float* key = j < n_window ? window + (uint64_t)j * head_dim : comp + (uint64_t)idx[j - n_window] * head_dim;
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
    for (uint32_t j = 0; j < n; j++) {
      const float* key =
          j < n_window ? window + (uint64_t)j * head_dim : comp + (uint64_t)idx[j - n_window] * head_dim;
      acc += scores[j] * key[d];
    }
    out[(uint64_t)head * head_dim + d] = acc / sum;
  }
}

__global__ void route_kernel(const float* logits, const float* bias, int32_t* ids, float* weights,
                             uint32_t n_experts, uint32_t k, float route_scale) {
  __shared__ float score[512];
  __shared__ float choice[512];
  __shared__ float best_value[32];
  __shared__ uint32_t best_index[32];
  __shared__ float picked[16];
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

__global__ void sum_rows_kernel(const float* x, const float* extra, float* out, uint32_t rows, uint32_t n) {
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float acc = extra ? extra[i] : 0.0f;
  for (uint32_t r = 0; r < rows; r++) acc += x[(uint64_t)r * n + i];
  out[i] = acc;
}

__global__ void engram_gate_kernel(float* h, const float* kv, const uint16_t* q, const uint16_t* k, uint32_t dim,
                                   uint32_t hc, float eps) {
  const uint32_t c = blockIdx.x;
  float* hc_row = h + (uint64_t)c * dim;
  const float* key = kv + (uint64_t)c * dim;
  const float* value = kv + (uint64_t)hc * dim;
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

}  // namespace

extern "C" int ds41_v41_matvec_bf16(const uint16_t* w, const float* x, float* out, uint32_t rows, uint32_t k) {
  const uint32_t rows_per_block = kThreads / 32;
  matvec_bf16_kernel<<<(rows + rows_per_block - 1) / rows_per_block, kThreads>>>(w, x, out, rows, k);
  return launch_ok("bf16 matvec");
}

extern "C" int ds41_v41_matvec_f32(const float* w, const float* x, float* out, uint32_t rows, uint32_t k) {
  matvec_f32_kernel<<<rows, kThreads>>>(w, x, out, rows, k);
  return launch_ok("f32 matvec");
}

extern "C" int ds41_v41_rms_norm(const float* x, const float* weight, float* out, uint32_t rows, uint32_t n,
                                 float eps) {
  rms_norm_kernel<<<rows, kThreads>>>(x, weight, out, n, eps);
  return launch_ok("rms norm");
}

extern "C" int ds41_v41_rope_tail(float* x, uint32_t rows, uint32_t row_dim, uint32_t rope_dim,
                                  const float* inv_freq, uint32_t pos, int inverse) {
  rope_tail_kernel<<<blocks_for((uint64_t)rows * rope_dim / 2), kThreads>>>(x, rows, row_dim, rope_dim, inv_freq,
                                                                           pos, inverse);
  return launch_ok("rope");
}

extern "C" int ds41_v41_fp8_fake_quant(float* x, uint32_t n) {
  fp8_fake_quant_kernel<<<blocks_for(n / 32), kThreads>>>(x, n / 32);
  return launch_ok("fp8 quant");
}

extern "C" int ds41_v41_fp4_fake_quant(float* x, uint32_t n, uint32_t block, int e4m3_scale) {
  fp4_fake_quant_kernel<<<blocks_for(n / block), kThreads>>>(x, n / block, block, e4m3_scale);
  return launch_ok("fp4 quant");
}

extern "C" int ds41_v41_hc_mixes(const float* fn, const float* h, float* mixes, uint32_t hc, uint32_t dim,
                                 float eps) {
  const uint32_t mixes_n = 2 * hc + hc * hc;
  hc_mixes_kernel<<<mixes_n, 1024>>>(fn, h, mixes, hc * dim, eps);
  return launch_ok("hc mixes");
}

extern "C" int ds41_v41_hc_sinkhorn(const float* mixes, const float* scale, const float* base, float* pre,
                                    float* post, float* comb, uint32_t hc, uint32_t iters, float eps) {
  if (hc > 8) return 1;
  hc_sinkhorn_kernel<<<1, 32>>>(mixes, scale, base, pre, post, comb, hc, iters, eps);
  return launch_ok("hc sinkhorn");
}

extern "C" int ds41_v41_hc_collapse(const float* h, const float* pre, float* out, uint32_t hc, uint32_t dim) {
  hc_collapse_kernel<<<blocks_for(dim), kThreads>>>(h, pre, out, hc, dim);
  return launch_ok("hc collapse");
}

extern "C" int ds41_v41_hc_expand(const float* y, const float* residual, const float* post, const float* comb,
                                  float* out, uint32_t hc, uint32_t dim) {
  hc_expand_kernel<<<blocks_for(dim), kThreads>>>(y, residual, post, comb, out, hc, dim);
  return launch_ok("hc expand");
}

extern "C" int ds41_v41_compress_pool(const float* kv, const float* score, float* state_kv, float* state_score,
                                      uint32_t slot, uint32_t ratio, int emit, float* out, uint32_t dim) {
  compress_pool_kernel<<<blocks_for(dim), kThreads>>>(kv, score, state_kv, state_score, slot, ratio, emit, out,
                                                     dim);
  return launch_ok("compressor pool");
}

extern "C" int ds41_v41_indexer_score(const float* q, const float* keys, const float* w, float w_scale,
                                      float* score, uint32_t n, uint32_t heads, uint32_t head_dim) {
  if (heads > 64) return 1;
  indexer_score_kernel<<<blocks_for(n), kThreads, heads * head_dim * sizeof(float)>>>(q, keys, w, w_scale, score, n,
                                                                                     heads, head_dim);
  return launch_ok("indexer score");
}

extern "C" int ds41_v41_topk(const float* score, uint32_t* out, uint32_t n, uint32_t k, uint32_t* scratch,
                             uint64_t scratch_bytes) {
  (void)scratch;
  (void)scratch_bytes;
  if (k == 0 || k > n) return 1;
  topk_kernel<<<1, 1024>>>(score, out, n, k);
  return launch_ok("topk");
}

extern "C" int ds41_v41_candidate_mask(const float* score, uint8_t* mask, float* block_scratch,
                                       uint32_t* select_scratch, uint32_t n, uint32_t block,
                                       uint32_t topk_blocks) {
  const uint32_t nb = (n + block - 1) / block;
  if (hipMemsetAsync(mask, nb <= topk_blocks ? 1 : 0, n, 0) != hipSuccess) return 1;
  if (nb <= topk_blocks) return 0;
  block_max_kernel<<<blocks_for(nb), kThreads>>>(score, block_scratch, n, block, nb);
  if (launch_ok("candidate block max")) return 1;
  if (ds41_v41_topk(block_scratch, select_scratch, nb, topk_blocks, nullptr, 0)) return 1;
  fill_mask_kernel<<<topk_blocks, 32>>>(mask, select_scratch, topk_blocks, n, block);
  return launch_ok("candidate mask");
}

extern "C" int ds41_v41_apply_mask(float* score, const uint8_t* mask, uint32_t n) {
  apply_mask_kernel<<<blocks_for(n), kThreads>>>(score, mask, n);
  return launch_ok("apply mask");
}

extern "C" int ds41_v41_attention(const float* q, const float* window, uint32_t n_window, const float* comp,
                                  const uint32_t* idx, uint32_t n_comp, const float* sinks, float* out,
                                  uint32_t heads, uint32_t head_dim, float scale) {
  const size_t shared = (size_t)(n_window + n_comp) * sizeof(float);
  if (shared > 60 * 1024) return 1;
  attention_kernel<<<heads, kThreads, shared>>>(q, window, n_window, comp, idx, n_comp, sinks, out, head_dim, scale);
  return launch_ok("attention");
}

extern "C" int ds41_v41_route(const float* logits, const float* bias, int32_t* ids, float* weights,
                              uint32_t n_experts, uint32_t k, float route_scale) {
  if (n_experts > 512 || k > 16) return 1;
  route_kernel<<<1, kThreads>>>(logits, bias, ids, weights, n_experts, k, route_scale);
  return launch_ok("route");
}

extern "C" int ds41_v41_swiglu(const float* gate, const float* up, const float* weight, float* out, uint32_t rows,
                               uint32_t n, float limit) {
  swiglu_kernel<<<blocks_for((uint64_t)rows * n), kThreads>>>(gate, up, weight, out, rows, n, limit);
  return launch_ok("swiglu");
}

extern "C" int ds41_v41_sum_rows(const float* x, const float* extra, float* out, uint32_t rows, uint32_t n) {
  sum_rows_kernel<<<blocks_for(n), kThreads>>>(x, extra, out, rows, n);
  return launch_ok("sum rows");
}

extern "C" int ds41_v41_engram_gate(float* h, const float* kv, const uint16_t* q, const uint16_t* k, uint32_t hc,
                                    uint32_t dim, float eps) {
  engram_gate_kernel<<<hc, 1024>>>(h, kv, q, k, dim, hc, eps);
  return launch_ok("engram gate");
}
