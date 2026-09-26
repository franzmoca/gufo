#ifndef GUFO_MODELS_GEMMA4_KERNELS_ROCM_KERNELS_HPP_
#define GUFO_MODELS_GEMMA4_KERNELS_ROCM_KERNELS_HPP_

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

namespace gufo::models::gemma4::rocm {

/// x[r] *= scale; h[r] = rms(x[r]) * weight. Used for the scaled token
/// embedding feeding the first attention norm.
void ScaleRmsNorm(float* x, float scale, const float* weight, float* h,
                  std::uint32_t rows, std::uint32_t dim, float eps,
                  hipStream_t stream);

/// Plain row RMSNorm: y[r] = rms(x[r]) * weight (weight may be null).
void RmsNorm(const float* x, const float* weight, float* y, std::uint32_t rows,
             std::uint32_t dim, float eps, hipStream_t stream);

/// Per-head attention input post-processing for `rows` consecutive positions
/// starting at `first_position`:
///   Q = rope(rms(q) * q_norm) in place (FP32);
///   K = rope(rms(k) * k_norm), V = rms(v_source) written as binary16 into the
///   caches at slot (position % ring) — ring = 0 means a linear cache.
/// `v` may alias `k` (global layers use the raw K projection as V).
/// Pair i turns by position * theta_scale^i / freq_factors[i] with
/// theta_scale = theta^(-2/dim) (`freq_factors` null means 1), as ggml's
/// NEOX rope computes it.
struct QkvPostArgs {
  float* q;
  const float* k;
  const float* v;
  const float* q_norm;
  const float* k_norm;
  float theta_scale;
  const float* freq_factors;
  std::uint16_t* k_cache;  ///< binary16
  std::uint16_t* v_cache;  ///< binary16
  std::uint32_t rows;
  std::uint32_t heads;
  std::uint32_t kv_heads;
  std::uint32_t head_dim;
  std::uint32_t first_position;
  std::uint32_t ring;
  float eps;
};
void QkvPost(const QkvPostArgs& args, hipStream_t stream);

/// Query-only post-processing (MTP draft layers): Q = rope(rms(q) * q_norm)
/// at one shared position for every row.
void QueryPost(float* q, const float* q_norm, float theta_scale,
               const float* freq_factors, std::uint32_t rows,
               std::uint32_t heads, std::uint32_t head_dim,
               std::uint32_t position, bool shared_position, float eps,
               hipStream_t stream);

/// Scale-1 attention of FP32 queries over binary16 K/V caches.
/// Row r queries position `first_position + r` (or `first_position` for all
/// rows when `shared_position`), attends keys [lo, hi) with
/// hi = min(position + 1, key_limit) and lo = hi-window bound
/// (position + 1 - window, clamped at 0) when `window` > 0. Keys live at slot
/// (key % ring) when ring > 0. Head h reads KV head h / (heads / kv_heads).
/// Up to kSplitRows rows split keys into fixed chunks aligned to absolute
/// key positions and merge them in a fixed order, so a row's result does not
/// depend on the batch it runs in; `partials` must hold
/// AttentionPartialFloats(...) floats. Larger batches (prefill) attend in a
/// single pass per (row, head).
struct AttentionArgs {
  const float* q;
  const std::uint16_t* k_cache;  ///< binary16
  const std::uint16_t* v_cache;  ///< binary16
  float* out;
  float* partials;
  std::uint32_t rows;
  std::uint32_t heads;
  std::uint32_t kv_heads;
  std::uint32_t head_dim;
  std::uint32_t first_position;
  bool shared_position;
  std::uint32_t key_limit;
  std::uint32_t window;
  std::uint32_t ring;
};
inline constexpr std::uint32_t kSplitRows = 16;
void Attention(const AttentionArgs& args, hipStream_t stream);
[[nodiscard]] std::size_t AttentionPartialFloats(std::uint32_t rows,
                                                 std::uint32_t heads,
                                                 std::uint32_t head_dim,
                                                 std::uint32_t max_keys);

/// x[r] += rms(o[r]) * post_norm; h[r] = rms(x[r]) * next_norm.
void PostAttentionNorm(const float* o, const float* post_norm, float* x,
                       const float* next_norm, float* h, std::uint32_t rows,
                       std::uint32_t dim, float eps, hipStream_t stream);

/// x[r] = (x[r] + rms(f[r]) * post_norm) * scale;
/// h[r] = rms(x[r]) * next_norm (next_norm null skips h).
void PostFeedForwardNorm(const float* f, const float* post_norm, float scale,
                         float* x, const float* next_norm, float* h,
                         std::uint32_t rows, std::uint32_t dim, float eps,
                         hipStream_t stream);

/// out = gelu_tanh(gate) * up, elementwise over `count` values.
void GeGlu(const float* gate, const float* up, float* out, std::size_t count,
           hipStream_t stream);

/// logits = cap * tanh(logits / cap), elementwise.
void Softcap(float* logits, std::size_t count, float cap, hipStream_t stream);

/// Copies rows `src[index[i]]` into dst[i] (row width `dim` floats).
void GatherRows(const float* src, const std::uint32_t* index, float* dst,
                std::uint32_t rows, std::uint32_t dim, hipStream_t stream);

}  // namespace gufo::models::gemma4::rocm

#endif  // GUFO_MODELS_GEMMA4_KERNELS_ROCM_KERNELS_HPP_
