#include "src/models/gemma4/kernels/rocm/kernels.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace gufo::models::gemma4::rocm {
namespace {

constexpr int kThreads = 256;
constexpr int kWave = 32;
constexpr int kWaves = kThreads / kWave;

/// Keys per split in the batch-invariant attention mode.
constexpr std::uint32_t SplitChunk(std::uint32_t head_dim) {
  return head_dim <= 256 ? 128U : 512U;
}

__device__ inline float WaveSum(float v) {
#pragma unroll
  for (int offset = kWave / 2; offset > 0; offset >>= 1) {
    v += __shfl_xor(v, offset, kWave);
  }
  return v;
}

/// Sum over the whole 256-thread block; every thread receives the result.
/// `scratch` holds kWaves floats and is reused only after a barrier.
__device__ inline float BlockSum(float v, float* scratch) {
  v = WaveSum(v);
  const int lane = threadIdx.x % kWave;
  const int wave = threadIdx.x / kWave;
  __syncthreads();
  if (lane == 0) {
    scratch[wave] = v;
  }
  __syncthreads();
  float total = 0.0F;
#pragma unroll
  for (int w = 0; w < kWaves; ++w) {
    total += scratch[w];
  }
  return total;
}

__device__ inline float RmsScale(float sum_squares, std::uint32_t dim,
                                 float eps) {
  return rsqrtf(sum_squares / static_cast<float>(dim) + eps);
}

// ---------------------------------------------------------------------------
// Norms
// ---------------------------------------------------------------------------

__global__ void __launch_bounds__(kThreads)
    ScaleRmsNormKernel(float* x, float scale, const float* weight, float* h,
                       std::uint32_t dim, float eps) {
  __shared__ float scratch[kWaves];
  float* xr = x + static_cast<std::size_t>(blockIdx.x) * dim;
  float* hr = h + static_cast<std::size_t>(blockIdx.x) * dim;
  float ss = 0.0F;
  for (std::uint32_t i = threadIdx.x; i < dim; i += kThreads) {
    const float v = xr[i] * scale;
    xr[i] = v;
    ss += v * v;
  }
  const float r = RmsScale(BlockSum(ss, scratch), dim, eps);
  for (std::uint32_t i = threadIdx.x; i < dim; i += kThreads) {
    hr[i] = xr[i] * r * weight[i];
  }
}

__global__ void __launch_bounds__(kThreads)
    RmsNormKernel(const float* x, const float* weight, float* y,
                  std::uint32_t dim, float eps) {
  __shared__ float scratch[kWaves];
  const float* xr = x + static_cast<std::size_t>(blockIdx.x) * dim;
  float* yr = y + static_cast<std::size_t>(blockIdx.x) * dim;
  float ss = 0.0F;
  for (std::uint32_t i = threadIdx.x; i < dim; i += kThreads) {
    ss += xr[i] * xr[i];
  }
  const float r = RmsScale(BlockSum(ss, scratch), dim, eps);
  for (std::uint32_t i = threadIdx.x; i < dim; i += kThreads) {
    const float v = xr[i] * r;
    yr[i] = weight != nullptr ? v * weight[i] : v;
  }
}

__global__ void __launch_bounds__(kThreads)
    PostAttentionNormKernel(const float* o, const float* post_norm, float* x,
                            const float* next_norm, float* h, std::uint32_t dim,
                            float eps) {
  __shared__ float scratch[kWaves];
  const std::size_t base = static_cast<std::size_t>(blockIdx.x) * dim;
  float ss = 0.0F;
  for (std::uint32_t i = threadIdx.x; i < dim; i += kThreads) {
    ss += o[base + i] * o[base + i];
  }
  const float r = RmsScale(BlockSum(ss, scratch), dim, eps);
  float ss2 = 0.0F;
  for (std::uint32_t i = threadIdx.x; i < dim; i += kThreads) {
    const float v = x[base + i] + o[base + i] * r * post_norm[i];
    x[base + i] = v;
    ss2 += v * v;
  }
  const float r2 = RmsScale(BlockSum(ss2, scratch), dim, eps);
  for (std::uint32_t i = threadIdx.x; i < dim; i += kThreads) {
    h[base + i] = x[base + i] * r2 * next_norm[i];
  }
}

__global__ void __launch_bounds__(kThreads)
    PostFeedForwardNormKernel(const float* f, const float* post_norm,
                              float scale, float* x, const float* next_norm,
                              float* h, std::uint32_t dim, float eps) {
  __shared__ float scratch[kWaves];
  const std::size_t base = static_cast<std::size_t>(blockIdx.x) * dim;
  float ss = 0.0F;
  for (std::uint32_t i = threadIdx.x; i < dim; i += kThreads) {
    ss += f[base + i] * f[base + i];
  }
  const float r = RmsScale(BlockSum(ss, scratch), dim, eps);
  float ss2 = 0.0F;
  for (std::uint32_t i = threadIdx.x; i < dim; i += kThreads) {
    const float v = (x[base + i] + f[base + i] * r * post_norm[i]) * scale;
    x[base + i] = v;
    ss2 += v * v;
  }
  if (next_norm == nullptr) {
    return;
  }
  const float r2 = RmsScale(BlockSum(ss2, scratch), dim, eps);
  for (std::uint32_t i = threadIdx.x; i < dim; i += kThreads) {
    h[base + i] = x[base + i] * r2 * next_norm[i];
  }
}

// ---------------------------------------------------------------------------
// Q/K/V post-processing and rope
// ---------------------------------------------------------------------------

/// Normalizes the head in `buf` (weight may be null) and applies NEOX rope
/// when `rope` is set; the result stays in `buf`.
__device__ void NormRopeHead(float* buf, const float* weight, std::uint32_t dim,
                             float eps, bool rope, std::uint32_t position,
                             float theta_scale, const float* freq_factors,
                             float* scratch) {
  float ss = 0.0F;
  for (std::uint32_t i = threadIdx.x; i < dim; i += kThreads) {
    ss += buf[i] * buf[i];
  }
  const float r = RmsScale(BlockSum(ss, scratch), dim, eps);
  for (std::uint32_t i = threadIdx.x; i < dim; i += kThreads) {
    const float v = buf[i] * r;
    buf[i] = weight != nullptr ? v * weight[i] : v;
  }
  __syncthreads();
  if (!rope) {
    return;
  }
  // dim <= 512, so every thread owns at most one rotation pair.
  const std::uint32_t half = dim / 2;
  const std::uint32_t i = threadIdx.x;
  float lo = 0.0F;
  float hi = 0.0F;
  if (i < half) {
    float theta =
        static_cast<float>(position) * powf(theta_scale, static_cast<float>(i));
    if (freq_factors != nullptr) {
      theta /= freq_factors[i];
    }
    // Fast-math sinf/cosf lose accuracy for large angles (positions reach
    // 262144 rad); double evaluation of the float angle rounds correctly.
    const float c = static_cast<float>(cos(static_cast<double>(theta)));
    const float s = static_cast<float>(sin(static_cast<double>(theta)));
    const float a = buf[i];
    const float b = buf[i + half];
    lo = a * c - b * s;
    hi = a * s + b * c;
  }
  __syncthreads();
  if (i < half) {
    buf[i] = lo;
    buf[i + half] = hi;
  }
  __syncthreads();
}

__global__ void __launch_bounds__(kThreads) QkvPostKernel(QkvPostArgs a) {
  __shared__ float buf[512];
  __shared__ float scratch[kWaves];
  const std::uint32_t row = blockIdx.x;
  const std::uint32_t head = blockIdx.y;
  const std::uint32_t dim = a.head_dim;
  const std::uint32_t position = a.first_position + row;
  if (head < a.heads) {
    float* q = a.q + (static_cast<std::size_t>(row) * a.heads + head) * dim;
    for (std::uint32_t i = threadIdx.x; i < dim; i += kThreads) {
      buf[i] = q[i];
    }
    __syncthreads();
    NormRopeHead(buf, a.q_norm, dim, a.eps, true, position, a.theta_scale,
                 a.freq_factors, scratch);
    for (std::uint32_t i = threadIdx.x; i < dim; i += kThreads) {
      q[i] = buf[i];
    }
    return;
  }
  const std::uint32_t kvh = head - a.heads;
  const std::size_t in =
      (static_cast<std::size_t>(row) * a.kv_heads + kvh) * dim;
  const std::uint32_t slot = a.ring != 0 ? position % a.ring : position;
  const std::size_t cache =
      (static_cast<std::size_t>(slot) * a.kv_heads + kvh) * dim;
  for (std::uint32_t i = threadIdx.x; i < dim; i += kThreads) {
    buf[i] = a.k[in + i];
  }
  __syncthreads();
  NormRopeHead(buf, a.k_norm, dim, a.eps, true, position, a.theta_scale,
               a.freq_factors, scratch);
  auto* k_cache = reinterpret_cast<__half*>(a.k_cache);
  auto* v_cache = reinterpret_cast<__half*>(a.v_cache);
  for (std::uint32_t i = threadIdx.x; i < dim; i += kThreads) {
    k_cache[cache + i] = __float2half(buf[i]);
  }
  __syncthreads();
  for (std::uint32_t i = threadIdx.x; i < dim; i += kThreads) {
    buf[i] = a.v[in + i];
  }
  __syncthreads();
  NormRopeHead(buf, nullptr, dim, a.eps, false, 0, 0.0F, nullptr, scratch);
  for (std::uint32_t i = threadIdx.x; i < dim; i += kThreads) {
    v_cache[cache + i] = __float2half(buf[i]);
  }
}

__global__ void __launch_bounds__(kThreads)
    QueryPostKernel(float* q, const float* q_norm, float theta_scale,
                    const float* freq_factors, std::uint32_t heads,
                    std::uint32_t dim, std::uint32_t position,
                    bool shared_position, float eps) {
  __shared__ float buf[512];
  __shared__ float scratch[kWaves];
  const std::uint32_t row = blockIdx.x;
  float* qh = q + (static_cast<std::size_t>(row) * heads + blockIdx.y) * dim;
  for (std::uint32_t i = threadIdx.x; i < dim; i += kThreads) {
    buf[i] = qh[i];
  }
  __syncthreads();
  NormRopeHead(buf, q_norm, dim, eps, true,
               shared_position ? position : position + row, theta_scale,
               freq_factors, scratch);
  for (std::uint32_t i = threadIdx.x; i < dim; i += kThreads) {
    qh[i] = buf[i];
  }
}

// ---------------------------------------------------------------------------
// Attention
// ---------------------------------------------------------------------------

struct KeyRange {
  std::uint32_t lo;
  std::uint32_t hi;
};

__device__ inline KeyRange RowKeys(const AttentionArgs& a, std::uint32_t row) {
  const std::uint32_t position =
      a.shared_position ? a.first_position : a.first_position + row;
  const std::uint32_t hi = min(position + 1, a.key_limit);
  std::uint32_t lo = 0;
  if (a.window != 0 && position + 1 > a.window) {
    lo = position + 1 - a.window;
  }
  return {min(lo, hi), hi};
}

/// One wave's online-softmax state over its keys, per lane D/32 values.
template<int D>
struct WaveState {
  static constexpr int kPerLane = D / kWave;
  float m;
  float l;
  float acc[kPerLane];
};

template<int D>
__device__ inline void AttendKeys(const AttentionArgs& a, const float* q_lane,
                                  std::uint32_t kvh, std::uint32_t begin,
                                  std::uint32_t end, WaveState<D>& st) {
  constexpr int P = D / kWave;
  const int lane = threadIdx.x % kWave;
  const int wave = threadIdx.x / kWave;
  const std::size_t stride = static_cast<std::size_t>(a.kv_heads) * D;
  // Keys are assigned to waves by absolute index, so the partition of a
  // chunk never depends on the query batch.
  std::uint32_t first =
      begin + ((wave - static_cast<int>(begin % kWaves)) + kWaves) % kWaves;
  for (std::uint32_t key = first; key < end; key += kWaves) {
    const std::uint32_t slot = a.ring != 0 ? key % a.ring : key;
    const auto* k = reinterpret_cast<const __half*>(a.k_cache) + slot * stride +
                    kvh * D + lane * P;
    const auto* v = reinterpret_cast<const __half*>(a.v_cache) + slot * stride +
                    kvh * D + lane * P;
    float dot = 0.0F;
#pragma unroll
    for (int i = 0; i < P; ++i) {
      dot += q_lane[i] * __half2float(k[i]);
    }
    const float s = WaveSum(dot);
    const float m_new = fmaxf(st.m, s);
    const float correction = expf(st.m - m_new);
    const float p = expf(s - m_new);
    st.l = st.l * correction + p;
#pragma unroll
    for (int i = 0; i < P; ++i) {
      st.acc[i] = st.acc[i] * correction + p * __half2float(v[i]);
    }
    st.m = m_new;
  }
}

/// Merges the waves' states in wave order; thread t owns dims t, t+256.
/// Writes (m, l, acc[D]) to `dst` or, when `normalize`, acc / l.
template<int D>
__device__ inline void MergeWaves(const WaveState<D>& st, float* dst,
                                  bool normalize) {
  constexpr int P = D / kWave;
  __shared__ float s_m[kWaves];
  __shared__ float s_l[kWaves];
  __shared__ float s_acc[kWaves * D];
  const int lane = threadIdx.x % kWave;
  const int wave = threadIdx.x / kWave;
  if (lane == 0) {
    s_m[wave] = st.m;
    s_l[wave] = st.l;
  }
#pragma unroll
  for (int i = 0; i < P; ++i) {
    s_acc[wave * D + lane * P + i] = st.acc[i];
  }
  __syncthreads();
  float m = -INFINITY;
#pragma unroll
  for (int w = 0; w < kWaves; ++w) {
    m = fmaxf(m, s_m[w]);
  }
  float l = 0.0F;
  float scale[kWaves];
#pragma unroll
  for (int w = 0; w < kWaves; ++w) {
    scale[w] = s_m[w] == -INFINITY ? 0.0F : expf(s_m[w] - m);
    l += s_l[w] * scale[w];
  }
  for (int d = threadIdx.x; d < D; d += kThreads) {
    float acc = 0.0F;
#pragma unroll
    for (int w = 0; w < kWaves; ++w) {
      acc += s_acc[w * D + d] * scale[w];
    }
    if (normalize) {
      dst[d] = l > 0.0F ? acc / l : 0.0F;
    } else {
      dst[2 + d] = acc;
    }
  }
  if (!normalize && threadIdx.x == 0) {
    dst[0] = m;
    dst[1] = l;
  }
}

template<int D>
__global__ void __launch_bounds__(kThreads)
    AttentionKernel(AttentionArgs a, std::uint32_t chunk, bool split) {
  constexpr int P = D / kWave;
  const std::uint32_t split_index = blockIdx.x;
  const std::uint32_t head = blockIdx.y;
  const std::uint32_t row = blockIdx.z;
  const std::uint32_t kvh = head / (a.heads / a.kv_heads);
  const KeyRange keys = RowKeys(a, row);
  std::uint32_t begin = keys.lo;
  std::uint32_t end = keys.hi;
  if (split) {
    begin = max(begin, split_index * chunk);
    end = min(end, (split_index + 1) * chunk);
  }
  const int lane = threadIdx.x % kWave;
  const float* qh =
      a.q + (static_cast<std::size_t>(row) * a.heads + head) * D + lane * P;
  float q_lane[P];
#pragma unroll
  for (int i = 0; i < P; ++i) {
    q_lane[i] = qh[i];
  }
  WaveState<D> st;
  st.m = -INFINITY;
  st.l = 0.0F;
#pragma unroll
  for (int i = 0; i < P; ++i) {
    st.acc[i] = 0.0F;
  }
  if (begin < end) {
    AttendKeys<D>(a, q_lane, kvh, begin, end, st);
  }
  if (split) {
    const std::size_t splits = gridDim.x;
    float* dst = a.partials +
                 ((static_cast<std::size_t>(row) * a.heads + head) * splits +
                  split_index) *
                     (D + 2);
    MergeWaves<D>(st, dst, false);
  } else {
    float* dst = a.out + (static_cast<std::size_t>(row) * a.heads + head) * D;
    MergeWaves<D>(st, dst, true);
  }
}

template<int D>
__global__ void __launch_bounds__(kThreads)
    AttentionMergeKernel(const float* partials, float* out, std::uint32_t heads,
                         std::uint32_t splits) {
  const std::uint32_t head = blockIdx.x;
  const std::uint32_t row = blockIdx.y;
  const float* base =
      partials +
      (static_cast<std::size_t>(row) * heads + head) * splits * (D + 2);
  float m = -INFINITY;
  for (std::uint32_t s = 0; s < splits; ++s) {
    m = fmaxf(m, base[s * (D + 2)]);
  }
  float l = 0.0F;
  for (std::uint32_t s = 0; s < splits; ++s) {
    const float ms = base[s * (D + 2)];
    if (ms != -INFINITY) {
      l += base[s * (D + 2) + 1] * expf(ms - m);
    }
  }
  float* dst = out + (static_cast<std::size_t>(row) * heads + head) * D;
  for (int d = threadIdx.x; d < D; d += kThreads) {
    float acc = 0.0F;
    for (std::uint32_t s = 0; s < splits; ++s) {
      const float ms = base[s * (D + 2)];
      if (ms != -INFINITY) {
        acc += base[s * (D + 2) + 2 + d] * expf(ms - m);
      }
    }
    dst[d] = l > 0.0F ? acc / l : 0.0F;
  }
}

template<int D>
void LaunchAttention(const AttentionArgs& a, hipStream_t stream) {
  const bool split = a.rows <= kSplitRows;
  if (!split) {
    AttentionKernel<D>
        <<<dim3(1, a.heads, a.rows), kThreads, 0, stream>>>(a, 0, false);
    return;
  }
  // Splits cover [0, max key) of the batch in absolute-position chunks.
  const std::uint32_t chunk = SplitChunk(D);
  const std::uint32_t last_position =
      a.shared_position ? a.first_position : a.first_position + a.rows - 1;
  const std::uint32_t max_hi = std::min(last_position + 1, a.key_limit);
  const std::uint32_t splits =
      std::max<std::uint32_t>(1, (max_hi + chunk - 1) / chunk);
  AttentionKernel<D>
      <<<dim3(splits, a.heads, a.rows), kThreads, 0, stream>>>(a, chunk, true);
  AttentionMergeKernel<D><<<dim3(a.heads, a.rows), kThreads, 0, stream>>>(
      a.partials, a.out, a.heads, splits);
}

// ---------------------------------------------------------------------------
// Elementwise
// ---------------------------------------------------------------------------

__global__ void GeGluKernel(const float* gate, const float* up, float* out,
                            std::size_t count) {
  constexpr float kSqrt2OverPi = 0.79788456080286535587989211986876F;
  constexpr float kCoefA = 0.044715F;
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count) {
    const float x = gate[i];
    const float g =
        0.5F * x * (1.0F + tanhf(kSqrt2OverPi * x * (1.0F + kCoefA * x * x)));
    out[i] = g * up[i];
  }
}

__global__ void SoftcapKernel(float* logits, std::size_t count, float cap) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count) {
    logits[i] = tanhf(logits[i] / cap) * cap;
  }
}

__global__ void GatherRowsKernel(const float* src, const std::uint32_t* index,
                                 float* dst, std::uint32_t dim) {
  const float* s = src + static_cast<std::size_t>(index[blockIdx.x]) * dim;
  float* d = dst + static_cast<std::size_t>(blockIdx.x) * dim;
  for (std::uint32_t i = threadIdx.x; i < dim; i += kThreads) {
    d[i] = s[i];
  }
}

unsigned Blocks(std::size_t count) {
  return static_cast<unsigned>((count + kThreads - 1) / kThreads);
}

}  // namespace

void ScaleRmsNorm(float* x, float scale, const float* weight, float* h,
                  std::uint32_t rows, std::uint32_t dim, float eps,
                  hipStream_t stream) {
  ScaleRmsNormKernel<<<rows, kThreads, 0, stream>>>(x, scale, weight, h, dim,
                                                    eps);
}

void RmsNorm(const float* x, const float* weight, float* y, std::uint32_t rows,
             std::uint32_t dim, float eps, hipStream_t stream) {
  RmsNormKernel<<<rows, kThreads, 0, stream>>>(x, weight, y, dim, eps);
}

void QkvPost(const QkvPostArgs& args, hipStream_t stream) {
  if (args.head_dim > 512) {
    throw std::invalid_argument("QkvPost supports head_dim <= 512");
  }
  QkvPostKernel<<<dim3(args.rows, args.heads + args.kv_heads), kThreads, 0,
                  stream>>>(args);
}

void QueryPost(float* q, const float* q_norm, float theta_scale,
               const float* freq_factors, std::uint32_t rows,
               std::uint32_t heads, std::uint32_t head_dim,
               std::uint32_t position, bool shared_position, float eps,
               hipStream_t stream) {
  QueryPostKernel<<<dim3(rows, heads), kThreads, 0, stream>>>(
      q, q_norm, theta_scale, freq_factors, heads, head_dim, position,
      shared_position, eps);
}

void Attention(const AttentionArgs& args, hipStream_t stream) {
  switch (args.head_dim) {
    case 256:
      LaunchAttention<256>(args, stream);
      return;
    case 512:
      LaunchAttention<512>(args, stream);
      return;
    default:
      throw std::invalid_argument("Attention supports head_dim 256 and 512");
  }
}

std::size_t AttentionPartialFloats(std::uint32_t rows, std::uint32_t heads,
                                   std::uint32_t head_dim,
                                   std::uint32_t max_keys) {
  const std::uint32_t split_rows = std::min(rows, kSplitRows);
  const std::uint32_t chunk = SplitChunk(head_dim);
  const std::size_t splits =
      std::max<std::uint32_t>(1, (max_keys + chunk - 1) / chunk);
  return static_cast<std::size_t>(split_rows) * heads * splits * (head_dim + 2);
}

void PostAttentionNorm(const float* o, const float* post_norm, float* x,
                       const float* next_norm, float* h, std::uint32_t rows,
                       std::uint32_t dim, float eps, hipStream_t stream) {
  PostAttentionNormKernel<<<rows, kThreads, 0, stream>>>(
      o, post_norm, x, next_norm, h, dim, eps);
}

void PostFeedForwardNorm(const float* f, const float* post_norm, float scale,
                         float* x, const float* next_norm, float* h,
                         std::uint32_t rows, std::uint32_t dim, float eps,
                         hipStream_t stream) {
  PostFeedForwardNormKernel<<<rows, kThreads, 0, stream>>>(
      f, post_norm, scale, x, next_norm, h, dim, eps);
}

void GeGlu(const float* gate, const float* up, float* out, std::size_t count,
           hipStream_t stream) {
  GeGluKernel<<<Blocks(count), kThreads, 0, stream>>>(gate, up, out, count);
}

void Softcap(float* logits, std::size_t count, float cap, hipStream_t stream) {
  SoftcapKernel<<<Blocks(count), kThreads, 0, stream>>>(logits, count, cap);
}

void GatherRows(const float* src, const std::uint32_t* index, float* dst,
                std::uint32_t rows, std::uint32_t dim, hipStream_t stream) {
  GatherRowsKernel<<<rows, kThreads, 0, stream>>>(src, index, dst, dim);
}

}  // namespace gufo::models::gemma4::rocm
