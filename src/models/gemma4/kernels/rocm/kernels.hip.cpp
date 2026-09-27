#include "src/models/gemma4/kernels/rocm/kernels.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "src/models/gemma4/kernels/rocm/attention_wmma.hpp"

namespace gufo::models::gemma4::rocm {
namespace {

constexpr int kThreads = 256;
constexpr int kWave = 32;
constexpr int kWaves = kThreads / kWave;

/// Keys per split in the batch-invariant attention mode.
constexpr std::uint32_t SplitChunk(std::uint32_t head_dim) {
  return head_dim <= 256 ? 128U : 512U;
}

/// v from lane (lane ^ M) through DPP or a cross-row permute instead of an
/// LDS permute.
template<int M>
__device__ __forceinline__ float XorLane(float v) {
  static_assert(M > 0 && M < kWave && (M & (M - 1)) == 0);
  const int x = __float_as_int(v);
  if constexpr (M == 16) {
    return __int_as_float(__builtin_amdgcn_permlanex16(
        x, x, 0x76543210, 0xfedcba98, false, false));
  } else {
    // row_xmask:M
    return __int_as_float(
        __builtin_amdgcn_update_dpp(x, x, 0x160 | M, 0xf, 0xf, false));
  }
}

__device__ __forceinline__ float WaveMax(float v) {
  v = fmaxf(v, XorLane<16>(v));
  v = fmaxf(v, XorLane<8>(v));
  v = fmaxf(v, XorLane<4>(v));
  v = fmaxf(v, XorLane<2>(v));
  return fmaxf(v, XorLane<1>(v));
}

__device__ __forceinline__ float WaveSum(float v) {
  v += XorLane<16>(v);
  v += XorLane<8>(v);
  v += XorLane<4>(v);
  v += XorLane<2>(v);
  return v + XorLane<1>(v);
}

/// XorLane for a mask known once loops unroll.
__device__ __forceinline__ float XorLaneStep(float v, int step) {
  switch (step) {
    case 1:
      return XorLane<1>(v);
    case 2:
      return XorLane<2>(v);
    case 4:
      return XorLane<4>(v);
    case 8:
      return XorLane<8>(v);
    default:
      return XorLane<16>(v);
  }
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

/// Elements of a row each thread keeps in registers across the passes of the
/// fused residual norms: element i of thread t is t + i * kThreads, the order
/// of every strided loop here.
constexpr std::uint32_t kRowRegisters = 24;

__global__ void __launch_bounds__(kThreads)
    PostAttentionNormKernel(const float* o, const float* post_norm, float* x,
                            const float* next_norm, float* h, std::uint32_t dim,
                            float eps) {
  __shared__ float scratch[kWaves];
  const std::size_t base = static_cast<std::size_t>(blockIdx.x) * dim;
  float ov[kRowRegisters];
  float xv[kRowRegisters];
  float ss = 0.0F;
#pragma unroll
  for (std::uint32_t j = 0; j < kRowRegisters; ++j) {
    const std::uint32_t i = threadIdx.x + j * kThreads;
    ov[j] = i < dim ? o[base + i] : 0.0F;
    xv[j] = i < dim ? x[base + i] : 0.0F;
    ss += ov[j] * ov[j];
  }
  const float r = RmsScale(BlockSum(ss, scratch), dim, eps);
  float ss2 = 0.0F;
#pragma unroll
  for (std::uint32_t j = 0; j < kRowRegisters; ++j) {
    const std::uint32_t i = threadIdx.x + j * kThreads;
    if (i < dim) {
      xv[j] = xv[j] + ov[j] * r * post_norm[i];
      x[base + i] = xv[j];
      ss2 += xv[j] * xv[j];
    }
  }
  const float r2 = RmsScale(BlockSum(ss2, scratch), dim, eps);
#pragma unroll
  for (std::uint32_t j = 0; j < kRowRegisters; ++j) {
    const std::uint32_t i = threadIdx.x + j * kThreads;
    if (i < dim) {
      h[base + i] = xv[j] * r2 * next_norm[i];
    }
  }
}

__global__ void __launch_bounds__(kThreads)
    PostFeedForwardNormKernel(const float* f, const float* post_norm,
                              float scale, float* x, const float* next_norm,
                              float* h, std::uint32_t dim, float eps) {
  __shared__ float scratch[kWaves];
  const std::size_t base = static_cast<std::size_t>(blockIdx.x) * dim;
  float fv[kRowRegisters];
  float xv[kRowRegisters];
  float ss = 0.0F;
#pragma unroll
  for (std::uint32_t j = 0; j < kRowRegisters; ++j) {
    const std::uint32_t i = threadIdx.x + j * kThreads;
    fv[j] = i < dim ? f[base + i] : 0.0F;
    xv[j] = i < dim ? x[base + i] : 0.0F;
    ss += fv[j] * fv[j];
  }
  const float r = RmsScale(BlockSum(ss, scratch), dim, eps);
  float ss2 = 0.0F;
#pragma unroll
  for (std::uint32_t j = 0; j < kRowRegisters; ++j) {
    const std::uint32_t i = threadIdx.x + j * kThreads;
    if (i < dim) {
      xv[j] = (xv[j] + fv[j] * r * post_norm[i]) * scale;
      x[base + i] = xv[j];
      ss2 += xv[j] * xv[j];
    }
  }
  if (next_norm == nullptr) {
    return;
  }
  const float r2 = RmsScale(BlockSum(ss2, scratch), dim, eps);
#pragma unroll
  for (std::uint32_t j = 0; j < kRowRegisters; ++j) {
    const std::uint32_t i = threadIdx.x + j * kThreads;
    if (i < dim) {
      h[base + i] = xv[j] * r2 * next_norm[i];
    }
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
    // Fast-math sinf/cosf become native approximations that lose accuracy
    // for large angles (positions reach 262144 rad); the OCML routine keeps
    // full float accuracy, as ggml's sinf/cosf do.
    const float c = __ocml_cos_f32(theta);
    const float s = __ocml_sin_f32(theta);
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

/// Derived-key layers keep only the rotated key dims; every other key dim
/// is its value times k_norm, so the query carries that weight instead.
__device__ void FoldKeyWeight(float* buf, std::uint32_t dim,
                              const float* key_weight, std::uint32_t pairs) {
  for (std::uint32_t i = threadIdx.x; i < dim; i += kThreads) {
    if (i % (dim / 2) >= pairs) {
      buf[i] *= key_weight[i];
    }
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
    if (a.rotated_pairs != 0) {
      FoldKeyWeight(buf, dim, a.k_norm, a.rotated_pairs);
    }
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
  auto* v_cache = reinterpret_cast<__half*>(a.v_cache);
  for (std::uint32_t i = threadIdx.x; i < dim; i += kThreads) {
    buf[i] = a.k[in + i];
  }
  __syncthreads();
  NormRopeHead(buf, a.k_norm, dim, a.eps, true, position, a.theta_scale,
               a.freq_factors, scratch);
  auto* k_cache = reinterpret_cast<__half*>(a.k_cache);
  if (a.rotated_pairs == 0) {
    for (std::uint32_t i = threadIdx.x; i < dim; i += kThreads) {
      k_cache[cache + i] = __float2half(buf[i]);
    }
  } else {
    // Rotated dims only: [0, pairs) then [dim / 2, dim / 2 + pairs).
    const std::uint32_t width = 2 * a.rotated_pairs;
    const std::size_t at =
        (static_cast<std::size_t>(slot) * a.kv_heads + kvh) * width;
    for (std::uint32_t i = threadIdx.x; i < width; i += kThreads) {
      const std::uint32_t d =
          i < a.rotated_pairs ? i : dim / 2 + i - a.rotated_pairs;
      k_cache[at + i] = __float2half(buf[d]);
    }
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
                    bool shared_position, float eps, const float* key_weight,
                    std::uint32_t rotated_pairs) {
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
  if (rotated_pairs != 0) {
    FoldKeyWeight(buf, dim, key_weight, rotated_pairs);
  }
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

/// Single-pass attention, one block per (head, row): the fallback when the
/// WMMA prefill kernel does not take a shape.
template<int D>
__global__ void __launch_bounds__(kThreads) AttentionKernel(AttentionArgs a) {
  constexpr int P = D / kWave;
  const std::uint32_t head = blockIdx.y;
  const std::uint32_t row = blockIdx.z;
  const std::uint32_t kvh = head / (a.heads / a.kv_heads);
  const KeyRange keys = RowKeys(a, row);
  const std::uint32_t begin = keys.lo;
  const std::uint32_t end = keys.hi;
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
  float* dst = a.out + (static_cast<std::size_t>(row) * a.heads + head) * D;
  MergeWaves<D>(st, dst, true);
}

/// A wave-uniform value moved to a scalar register.
__device__ __forceinline__ float Uniform(float v) {
  return __int_as_float(__builtin_amdgcn_readfirstlane(__float_as_int(v)));
}

/// Rotation pairs a derived-key tile can hold (Gemma 4 global layers: 64).
constexpr int kMaxRopePairs = static_cast<int>(kMaxDerivedKeyPairs);

/// Rows per block of the row-shared split kernel.
constexpr int kRowBlock = 5;

/// Leaves in every lane L the wave-wide sum of v[L % N].
template<int N>
__device__ __forceinline__ float ReduceScatterN(float (&v)[N]) {
  const int lane = threadIdx.x % kWave;
#pragma unroll
  for (int step = N / 2; step > 0; step >>= 1) {
    const bool upper = (lane & step) != 0;
#pragma unroll
    for (int i = 0; i < step; ++i) {
      const float send = upper ? v[i] : v[i + step];
      const float keep = upper ? v[i + step] : v[i];
      v[i] = keep + XorLaneStep(send, step);
    }
  }
  float s = v[0];
#pragma unroll
  for (int offset = N; offset < kWave; offset <<= 1) {
    s += XorLaneStep(s, offset);
  }
  return s;
}

/// One wave per (query head, absolute-position chunk) over up to R rows: the
/// rows' queries and accumulators stay in registers, so the waves of a block
/// (the heads of one KV head) stream each key and value once for every row.
/// Online softmax per 32-key tile. A row's arithmetic does not depend on R or
/// on the rows sharing its block: scores reduce in groups of KG keys and only
/// the load batching (LG keys) varies with R. Writes (m, l, unnormalized acc).
template<int D, int G, int R, int WPH>
__global__ void __launch_bounds__(kThreads* WPH)
    RowSplitAttentionKernel(AttentionArgs a, std::uint32_t first_split,
                            std::uint32_t splits) {
  constexpr int P = D / kWave;
  constexpr int C = static_cast<int>(SplitChunk(D));
  constexpr int CW = kWaves / G;  // chunks per block
  constexpr int KG = P >= 16 ? 2 : 4;
  constexpr int LG = R == 1 && WPH == 1 ? 8 : KG;
  constexpr int kBlockThreads = kThreads * WPH;
  const int lane = threadIdx.x % kWave;
  const int wave = (threadIdx.x / kWave) % kWaves;
  // Row group: WPH waves per head split the block's rows, R each.
  const int rg = static_cast<int>(threadIdx.x) / kThreads;
  const std::uint32_t local = blockIdx.z * CW + wave / G;
  if (local >= splits) {
    return;
  }
  const std::uint32_t head = blockIdx.y * G + wave % G;
  const std::uint32_t block_row0 = blockIdx.x * R * WPH;
  const std::uint32_t block_rows =
      min(static_cast<std::uint32_t>(R * WPH), a.rows - block_row0);
  const std::uint32_t row0 = block_row0 + rg * R;
  const int rows =
      max(0, min(R, static_cast<int>(a.rows) - static_cast<int>(row0)));
  const std::uint32_t base = (first_split + local) * C;
  const std::size_t stride = static_cast<std::size_t>(a.kv_heads) * D;
  const auto* k_head =
      reinterpret_cast<const __half*>(a.k_cache) + blockIdx.y * D;
  const auto* v_head =
      reinterpret_cast<const __half*>(a.v_cache) + blockIdx.y * D;
  // At hd512 the eight head waves (per row group) read every key and value:
  // the block stages each 32-key tile once in shared memory instead of
  // issuing a copy of every load per wave.
  constexpr bool kStaged = G == kWaves;
  static_assert(kStaged || WPH == 1);
  constexpr int kStageLoads = kStaged ? kWave * D / 8 / kBlockThreads : 1;
  __shared__ __align__(16) __half tile[kStaged ? kWave * D : 8];

  std::uint32_t lo[R];
  std::uint32_t hi[R];
  float q[R][P];
  float m[R];
  float l[R];
  float acc[R][P];
#pragma unroll
  for (int r = 0; r < R; ++r) {
    const std::uint32_t row =
        block_row0 + min(rg * R + r, static_cast<int>(block_rows) - 1);
    const KeyRange keys = RowKeys(a, row);
    lo[r] = r < rows ? keys.lo : 0;
    hi[r] = r < rows ? keys.hi : 0;
    const float* qr =
        a.q + (static_cast<std::size_t>(row) * a.heads + head) * D + lane * P;
#pragma unroll
    for (int i = 0; i < P; i += 4) {
      const float4 f = *reinterpret_cast<const float4*>(qr + i);
      q[r][i] = f.x;
      q[r][i + 1] = f.y;
      q[r][i + 2] = f.z;
      q[r][i + 3] = f.w;
    }
    m[r] = -INFINITY;
    l[r] = 0.0F;
#pragma unroll
    for (int i = 0; i < P; ++i) {
      acc[r][i] = 0.0F;
    }
  }
  // Keys any row of the block attends in this chunk; rows' ranges grow with
  // the row index. Keys outside [begin, end) load as zeros.
  const std::uint32_t begin = max(base, RowKeys(a, block_row0).lo);
  const std::uint32_t end =
      min(base + C, RowKeys(a, block_row0 + block_rows - 1).hi);
  // Derived keys (rope_pairs > 0): the cache keeps only the rotated key
  // dims, staged beside the value tile; every other key dim is its value
  // (the query carries the key weight).
  const bool derived = a.rope_pairs != 0;
  const int pairs = static_cast<int>(a.rope_pairs);
  __shared__ __align__(16)
      __half rotated[kStaged ? kWave * 2 * kMaxRopePairs : 8];
  // This lane's key dims [lane P, lane P + P) in the rotated tile, or -1.
  const int dim0 = lane * P;
  const int rot_offset = dim0 < pairs ? dim0
                         : dim0 >= D / 2 && dim0 < D / 2 + pairs
                             ? pairs + dim0 - D / 2
                             : -1;
  // Row kk of the tile at key0 (keys or values): shared memory when staged,
  // else the cache.
  auto load = [&](const __half* head_cache, bool keys, std::uint32_t key0,
                  int kk, uint4(&raw)[P / 8]) {
    if constexpr (kStaged) {
      const __half* row = keys && rot_offset >= 0
                              ? rotated + kk * 2 * pairs + rot_offset
                              : tile + kk * D + dim0;
      const auto* src = reinterpret_cast<const uint4*>(row);
#pragma unroll
      for (int i = 0; i < P / 8; ++i) {
        raw[i] = src[i];
      }
    } else {
      const std::uint32_t key = key0 + kk;
      if (key >= begin && key < end) {
        const std::uint32_t slot = a.ring != 0 ? key % a.ring : key;
        const auto* src =
            reinterpret_cast<const uint4*>(head_cache + slot * stride + dim0);
#pragma unroll
        for (int i = 0; i < P / 8; ++i) {
          raw[i] = src[i];
        }
      } else {
#pragma unroll
        for (int i = 0; i < P / 8; ++i) {
          raw[i] = uint4{0, 0, 0, 0};
        }
      }
    }
  };
  // Staging: thread t moves, for keys t / 32 + 8 m of a tile, the two
  // 8-dim chunks of rotation pair group t % 32 (dims 8 (t % 32) + j and
  // D / 2 + 8 (t % 32) + j), so derived keys rotate within one thread.
  constexpr int kGroups = D / 16;
  constexpr int kKeysPerPass = kBlockThreads / kGroups;
  static_assert(!kStaged || kStageLoads == 2 * kWave / kKeysPerPass);
  const int group = static_cast<int>(threadIdx.x) % kGroups;
  uint4 staged[kStageLoads];
  // Rotated key dims of a tile: 32 keys x 2 pairs values in 16-byte pieces.
  constexpr int kRotLoads =
      kStaged ? kWave * 2 * kMaxRopePairs / 8 / kBlockThreads : 1;
  uint4 staged_rot[kRotLoads];
  const int rot_chunks = 2 * pairs / 8;  // per key
  const auto* k_rotated = reinterpret_cast<const __half*>(a.k_cache);
  auto stage_load = [&](const __half* head_cache, std::uint32_t key0) {
#pragma unroll
    for (int n = 0; n < kStageLoads / 2; ++n) {
      const std::uint32_t key =
          key0 + static_cast<int>(threadIdx.x) / kGroups + n * kKeysPerPass;
      if (key >= begin && key < end) {
        const std::uint32_t slot = a.ring != 0 ? key % a.ring : key;
        const __half* row = head_cache + slot * stride + group * 8;
        staged[2 * n] = *reinterpret_cast<const uint4*>(row);
        staged[2 * n + 1] = *reinterpret_cast<const uint4*>(row + D / 2);
      } else {
        staged[2 * n] = uint4{0, 0, 0, 0};
        staged[2 * n + 1] = uint4{0, 0, 0, 0};
      }
    }
  };
  auto rot_store = [&] {
#pragma unroll
    for (int n = 0; n < kRotLoads; ++n) {
      const int u = static_cast<int>(threadIdx.x) + n * kBlockThreads;
      if (u < kWave * rot_chunks) {
        reinterpret_cast<uint4*>(rotated)[u] = staged_rot[n];
      }
    }
  };
  auto stage_store = [&] {
    __syncthreads();
#pragma unroll
    for (int n = 0; n < kStageLoads / 2; ++n) {
      const int kk = static_cast<int>(threadIdx.x) / kGroups + n * kKeysPerPass;
      __half* dst = tile + kk * D + group * 8;
      *reinterpret_cast<uint4*>(dst) = staged[2 * n];
      *reinterpret_cast<uint4*>(dst + D / 2) = staged[2 * n + 1];
    }
    if (derived) {
      rot_store();
    }
    __syncthreads();
  };
  auto rot_load = [&](std::uint32_t key0) {
#pragma unroll
    for (int n = 0; n < kRotLoads; ++n) {
      const int u = static_cast<int>(threadIdx.x) + n * kBlockThreads;
      const std::uint32_t key = key0 + u / rot_chunks;
      staged_rot[n] = uint4{0, 0, 0, 0};
      if (u < kWave * rot_chunks && key >= begin && key < end) {
        const std::uint32_t slot = a.ring != 0 ? key % a.ring : key;
        staged_rot[n] = *reinterpret_cast<const uint4*>(
            k_rotated +
            (static_cast<std::size_t>(slot) * a.kv_heads + blockIdx.y) * 2 *
                pairs +
            (u % rot_chunks) * 8);
      }
    }
  };
  if constexpr (kStaged) {
    const std::uint32_t first = begin & ~std::uint32_t{kWave - 1};
    if (derived) {
      rot_load(first);
      stage_load(v_head, first);
    } else {
      stage_load(k_head, first);
    }
  }

  for (std::uint32_t key0 = begin & ~std::uint32_t{kWave - 1}; key0 < end;
       key0 += kWave) {
    const bool more = key0 + kWave < end;
    if constexpr (kStaged) {
      stage_store();
      if (derived) {
        // One value tile serves keys and values: the next one (values and
        // rotated key dims) loads under both passes.
        if (more) {
          rot_load(key0 + kWave);
          stage_load(v_head, key0 + kWave);
        }
      } else {
        // Keys to shared memory; the values load under the scores.
        stage_load(v_head, key0);
      }
    }
    // Scores: lane L holds key key0 + L of every row.
    float s[R];
#pragma unroll
    for (int r = 0; r < R; ++r) {
      s[r] = -INFINITY;
    }
#pragma unroll 1
    for (int g0 = 0; g0 < kWave; g0 += LG) {
      uint4 kraw[LG][P / 8];
#pragma unroll
      for (int kk = 0; kk < LG; ++kk) {
        load(k_head, true, key0, g0 + kk, kraw[kk]);
      }
#pragma unroll
      for (int g = 0; g < LG; g += KG) {
#pragma unroll
        for (int r = 0; r < R; ++r) {
          if (r < rows) {
            float part[KG];
#pragma unroll
            for (int kk = 0; kk < KG; ++kk) {
              float dot = 0.0F;
#pragma unroll
              for (int i = 0; i < P / 8; ++i) {
                const auto* k2 =
                    reinterpret_cast<const __half2*>(&kraw[g + kk][i]);
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                  const float2 f = __half22float2(k2[j]);
                  dot = __builtin_fmaf(q[r][8 * i + 2 * j], f.x, dot);
                  dot = __builtin_fmaf(q[r][8 * i + 2 * j + 1], f.y, dot);
                }
              }
              part[kk] = dot;
            }
            const float score = ReduceScatterN<KG>(part);
            if (lane / KG == (g0 + g) / KG) {
              s[r] = score;
            }
          }
        }
      }
    }
    // The first values load under the softmax.
    uint4 vraw[LG][P / 8];
    if constexpr (!kStaged) {
#pragma unroll
      for (int kk = 0; kk < LG; ++kk) {
        load(v_head, false, key0, kk, vraw[kk]);
      }
    }
    // Online softmax per row; a tile without keys for a row leaves it as is.
    float p[R];
#pragma unroll
    for (int r = 0; r < R; ++r) {
      p[r] = 0.0F;
      if (r < rows) {
        const std::uint32_t key = key0 + lane;
        const float sr = key >= lo[r] && key < hi[r] ? s[r] : -INFINITY;
        // Wave-uniform statistics live in scalar registers.
        const float mt = Uniform(WaveMax(sr));
        if (mt != -INFINITY) {
          const float mn = fmaxf(m[r], mt);
          const float scale = Uniform(expf(m[r] - mn));
          p[r] = expf(sr - mn);
          l[r] = __builtin_fmaf(l[r], scale, Uniform(WaveSum(p[r])));
          m[r] = mn;
#pragma unroll
          for (int i = 0; i < P; ++i) {
            acc[r][i] *= scale;
          }
        }
      }
    }
    if constexpr (kStaged) {
      if (!derived) {
        // Values to shared memory; the next keys load under the products.
        stage_store();
        if (more) {
          stage_load(k_head, key0 + kWave);
        }
      }
    }
    // Values, the next batch loading under the current one.
#pragma unroll
    for (int g0 = 0; g0 < kWave; g0 += LG) {
      uint4 cur[LG][P / 8];
      if constexpr (kStaged) {
#pragma unroll
        for (int kk = 0; kk < LG; ++kk) {
          load(v_head, false, key0, g0 + kk, cur[kk]);
        }
      } else {
#pragma unroll
        for (int kk = 0; kk < LG; ++kk) {
#pragma unroll
          for (int i = 0; i < P / 8; ++i) {
            cur[kk][i] = vraw[kk][i];
          }
        }
        if (g0 + LG < kWave) {
#pragma unroll
          for (int kk = 0; kk < LG; ++kk) {
            load(v_head, false, key0, g0 + LG + kk, vraw[kk]);
          }
        }
      }
#pragma unroll
      for (int kk = 0; kk < LG; ++kk) {
        float v[P];
#pragma unroll
        for (int i = 0; i < P / 8; ++i) {
          const auto* v2 = reinterpret_cast<const __half2*>(&cur[kk][i]);
#pragma unroll
          for (int j = 0; j < 4; ++j) {
            const float2 f = __half22float2(v2[j]);
            v[8 * i + 2 * j] = f.x;
            v[8 * i + 2 * j + 1] = f.y;
          }
        }
#pragma unroll
        for (int r = 0; r < R; ++r) {
          if (r < rows) {
            const float pk = __int_as_float(
                __builtin_amdgcn_readlane(__float_as_int(p[r]), g0 + kk));
#pragma unroll
            for (int i = 0; i < P; ++i) {
              acc[r][i] = __builtin_fmaf(pk, v[i], acc[r][i]);
            }
          }
        }
      }
    }
  }
#pragma unroll
  for (int r = 0; r < R; ++r) {
    if (r < rows) {
      float* dst =
          a.partials +
          ((static_cast<std::size_t>(row0 + r) * a.heads + head) * splits +
           local) *
              (D + 2);
      if (lane == 0) {
        dst[0] = m[r];
        dst[1] = l[r];
      }
#pragma unroll
      for (int i = 0; i < P; i += 4) {
        *reinterpret_cast<float4*>(dst + 2 + lane * P + i) =
            float4{acc[r][i], acc[r][i + 1], acc[r][i + 2], acc[r][i + 3]};
      }
    }
  }
}

/// Threads per merge block; each block merges kMergeThreads dimensions of one
/// (row, head).
constexpr int kMergeThreads = 128;
/// Splits one merge can weigh: 262144 keys at the smallest chunk.
constexpr std::uint32_t kMaxSplits = 262144 / SplitChunk(256) + 2;

/// Merges a (row, head)'s splits in split order. Splits without keys carry
/// m = -inf and zero sums, so the leading splits a wider batch adds leave a
/// row's result unchanged.
template<int D>
__global__ void __launch_bounds__(kMergeThreads)
    AttentionMergeKernel(const float* partials, float* out, std::uint32_t heads,
                         std::uint32_t splits) {
  __shared__ float scale[kMaxSplits];
  __shared__ float wave_max[kMergeThreads / kWave];
  const std::uint32_t head = blockIdx.x;
  const std::uint32_t row = blockIdx.y;
  const float* base =
      partials +
      (static_cast<std::size_t>(row) * heads + head) * splits * (D + 2);
  float m = -INFINITY;
  for (std::uint32_t s = threadIdx.x; s < splits; s += kMergeThreads) {
    m = fmaxf(m, base[s * (D + 2)]);
  }
  m = WaveMax(m);
  if (threadIdx.x % kWave == 0) {
    wave_max[threadIdx.x / kWave] = m;
  }
  __syncthreads();
#pragma unroll
  for (int w = 0; w < kMergeThreads / kWave; ++w) {
    m = fmaxf(m, wave_max[w]);
  }
  for (std::uint32_t s = threadIdx.x; s < splits; s += kMergeThreads) {
    const float ms = base[s * (D + 2)];
    scale[s] = ms != -INFINITY ? expf(ms - m) : 0.0F;
  }
  __syncthreads();
  float l = 0.0F;
  for (std::uint32_t s = 0; s < splits; ++s) {
    l = __builtin_fmaf(base[s * (D + 2) + 1], scale[s], l);
  }
  const std::uint32_t d = blockIdx.z * kMergeThreads + threadIdx.x;
  const float* src = base + 2 + d;
  float acc = 0.0F;
#pragma unroll 8
  for (std::uint32_t s = 0; s < splits; ++s) {
    acc = __builtin_fmaf(src[s * (D + 2)], scale[s], acc);
  }
  out[(static_cast<std::size_t>(row) * heads + head) * D + d] =
      l > 0.0F ? acc / l : 0.0F;
}

template<int D, int G>
void LaunchSplitAttention(const AttentionArgs& a, hipStream_t stream) {
  // Splits cover the batch's keys [lowest window start, highest key) in
  // absolute-position chunks. A block holds the G query heads of one KV head
  // for kWaves / G consecutive chunks and up to kRowBlock rows.
  const std::uint32_t chunk = SplitChunk(D);
  const std::uint32_t last_position =
      a.shared_position ? a.first_position : a.first_position + a.rows - 1;
  const std::uint32_t max_hi = std::min(last_position + 1, a.key_limit);
  const std::uint32_t min_lo = a.window != 0 && a.first_position + 1 > a.window
                                   ? a.first_position + 1 - a.window
                                   : 0;
  const std::uint32_t first_split = std::min(min_lo, max_hi) / chunk;
  const std::uint32_t splits =
      std::max<std::uint32_t>(1, (max_hi + chunk - 1) / chunk - first_split);
  if (splits > kMaxSplits) {
    throw std::invalid_argument("split attention context too long");
  }
  constexpr std::uint32_t kChunksPerBlock = kWaves / G;
  const std::uint32_t chunk_blocks =
      (splits + kChunksPerBlock - 1) / kChunksPerBlock;
  if (a.rows == 1) {
    // Single-token decode batches loads deeper; the arithmetic is the same.
    RowSplitAttentionKernel<D, G, 1, 1>
        <<<dim3(1, a.kv_heads, chunk_blocks), kThreads, 0, stream>>>(
            a, first_split, splits);
  } else if constexpr (G == kWaves) {
    // hd512: two waves per head, three rows each, keep a block's rows within
    // the register file.
    constexpr std::uint32_t kRows = 3 * 2;
    RowSplitAttentionKernel<D, G, 3, 2>
        <<<dim3((a.rows + kRows - 1) / kRows, a.kv_heads, chunk_blocks),
           2 * kThreads, 0, stream>>>(a, first_split, splits);
  } else {
    RowSplitAttentionKernel<D, G, kRowBlock, 1>
        <<<dim3((a.rows + kRowBlock - 1) / kRowBlock, a.kv_heads, chunk_blocks),
           kThreads, 0, stream>>>(a, first_split, splits);
  }
  AttentionMergeKernel<D>
      <<<dim3(a.heads, a.rows, D / kMergeThreads), kMergeThreads, 0, stream>>>(
          a.partials, a.out, a.heads, splits);
}

template<int D>
void LaunchAttention(const AttentionArgs& a, hipStream_t stream) {
  if (a.rows > kSplitRows) {
    if (!LaunchWmmaPrefillAttention(a, stream)) {
      if (a.rope_pairs != 0) {
        throw std::invalid_argument("derived keys need the WMMA prefill path");
      }
      AttentionKernel<D><<<dim3(1, a.heads, a.rows), kThreads, 0, stream>>>(a);
    }
    return;
  }
  // Gemma 4 groups 2 query heads per KV head at hd256, 8 at hd512.
  constexpr std::uint32_t kGroup = D == 256 ? 2 : 8;
  if (a.heads != a.kv_heads * kGroup) {
    throw std::invalid_argument("split attention head grouping unsupported");
  }
  LaunchSplitAttention<D, kGroup>(a, stream);
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

__global__ void ScaleKernel(const float* x, float scale, float* y,
                            std::size_t count) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count) {
    y[i] = x[i] * scale;
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
               const float* key_weight, std::uint32_t rotated_pairs,
               hipStream_t stream) {
  QueryPostKernel<<<dim3(rows, heads), kThreads, 0, stream>>>(
      q, q_norm, theta_scale, freq_factors, heads, head_dim, position,
      shared_position, eps, key_weight, rotated_pairs);
}

void Attention(const AttentionArgs& args, hipStream_t stream) {
  if (args.rope_pairs != 0 &&
      (args.head_dim != 512 || args.rope_pairs % 16 != 0 ||
       args.rope_pairs > static_cast<std::uint32_t>(kMaxRopePairs))) {
    throw std::invalid_argument(
        "derived keys need head_dim 512 and 16-aligned pairs up to 64");
  }
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
  // One more split than the keys need: a window's first chunk may start
  // before it.
  const std::size_t splits = (max_keys + chunk - 1) / chunk + 1;
  return static_cast<std::size_t>(split_rows) * heads * splits * (head_dim + 2);
}

void PostAttentionNorm(const float* o, const float* post_norm, float* x,
                       const float* next_norm, float* h, std::uint32_t rows,
                       std::uint32_t dim, float eps, hipStream_t stream) {
  if (dim > kRowRegisters * kThreads) {
    throw std::invalid_argument("PostAttentionNorm row is too wide");
  }
  PostAttentionNormKernel<<<rows, kThreads, 0, stream>>>(
      o, post_norm, x, next_norm, h, dim, eps);
}

void PostFeedForwardNorm(const float* f, const float* post_norm, float scale,
                         float* x, const float* next_norm, float* h,
                         std::uint32_t rows, std::uint32_t dim, float eps,
                         hipStream_t stream) {
  if (dim > kRowRegisters * kThreads) {
    throw std::invalid_argument("PostFeedForwardNorm row is too wide");
  }
  PostFeedForwardNormKernel<<<rows, kThreads, 0, stream>>>(
      f, post_norm, scale, x, next_norm, h, dim, eps);
}

void GeGlu(const float* gate, const float* up, float* out, std::size_t count,
           hipStream_t stream) {
  GeGluKernel<<<Blocks(count), kThreads, 0, stream>>>(gate, up, out, count);
}

void Scale(const float* x, float scale, float* y, std::size_t count,
           hipStream_t stream) {
  ScaleKernel<<<Blocks(count), kThreads, 0, stream>>>(x, scale, y, count);
}

void Softcap(float* logits, std::size_t count, float cap, hipStream_t stream) {
  SoftcapKernel<<<Blocks(count), kThreads, 0, stream>>>(logits, count, cap);
}

void GatherRows(const float* src, const std::uint32_t* index, float* dst,
                std::uint32_t rows, std::uint32_t dim, hipStream_t stream) {
  GatherRowsKernel<<<rows, kThreads, 0, stream>>>(src, index, dst, dim);
}

}  // namespace gufo::models::gemma4::rocm
