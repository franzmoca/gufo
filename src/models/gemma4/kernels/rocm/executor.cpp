#include "src/models/gemma4/kernels/rocm/executor.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <stdexcept>

#include "src/core/hip/hip_utils.hpp"
#include "src/models/gemma4/kernels/rocm/gemv.hpp"
#include "src/models/gemma4/kernels/rocm/kernels.hpp"
#include "src/models/qwen/hip/ops/gemm.hpp"
#include "src/models/qwen/hip/ops/token.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"
#include "src/models/qwen38_flash_next/mtp_sampling.hpp"

namespace gufo::models::gemma4::rocm {
namespace {

constexpr std::size_t kAlign = 256;

std::size_t AlignUp(std::size_t n) {
  return (n + kAlign - 1) / kAlign * kAlign;
}

struct Layout {
  std::size_t tokens, logit_index, x, h, q, k, v, attn, o, gate, up, hsel,
      logits, partials, q8;
  std::size_t draft_tokens, draft_concat, draft_x, draft_h, draft_q, draft_attn,
      draft_o, draft_gate, draft_up, draft_logits, draft_next, draft_candidates,
      draft_candidate_scratch;
  std::size_t total;
};

std::uint32_t MaxQDim(const Config& c) {
  std::uint32_t m = 0;
  for (std::uint32_t l = 0; l < c.num_layers; ++l) {
    m = std::max(m, c.QDim(l));
  }
  return m;
}

std::uint32_t MaxKvDim(const Config& c) {
  std::uint32_t m = 0;
  for (std::uint32_t l = 0; l < c.num_layers; ++l) {
    m = std::max(m, c.KvDim(l));
  }
  return m;
}

Layout Plan(const Config& c, const Config* draft, std::uint32_t vocab,
            std::size_t max_cols, std::uint32_t rows, std::uint32_t logit_rows,
            std::uint32_t max_context) {
  const std::size_t f = sizeof(float);
  const std::uint32_t ring = RingSlots(c, rows);
  const std::size_t partial_floats = std::max(
      AttentionPartialFloats(rows, c.num_heads, c.head_dim_global, max_context),
      AttentionPartialFloats(rows, c.num_heads, c.head_dim_sliding,
                             std::min(max_context, ring)));
  Layout l{};
  std::size_t at = 0;
  const auto take = [&](std::size_t bytes) {
    const std::size_t here = at;
    at += AlignUp(bytes);
    return here;
  };
  l.tokens = take(rows * sizeof(std::int32_t));
  l.logit_index = take(logit_rows * sizeof(std::uint32_t));
  l.x = take(std::size_t{rows} * c.hidden_size * f);
  l.h = take(std::size_t{rows} * c.hidden_size * f);
  l.q = take(std::size_t{rows} * MaxQDim(c) * f);
  l.k = take(std::size_t{rows} * MaxKvDim(c) * f);
  l.v = take(std::size_t{rows} * MaxKvDim(c) * f);
  l.attn = take(std::size_t{rows} * MaxQDim(c) * f);
  l.o = take(std::size_t{rows} * c.hidden_size * f);
  l.gate = take(std::size_t{rows} * c.ffn_size * f);
  l.up = take(std::size_t{rows} * c.ffn_size * f);
  l.hsel = take(std::size_t{logit_rows} * c.hidden_size * f);
  l.logits = take(std::size_t{logit_rows} * vocab * f);
  l.partials = take(partial_floats * f);
  const std::uint32_t q8_rows = std::max(rows, logit_rows);
  l.q8 = take(q8_rows > kSplitRows
                  ? hip::QuantizedActivationBytes(q8_rows, max_cols)
                  : 0);
  if (draft != nullptr) {
    l.draft_tokens = take((kMaxDraftTokens + 1) * sizeof(std::uint32_t));
    l.draft_concat = take(2 * std::size_t{draft->target_hidden_size} * f);
    l.draft_x = take(std::size_t{draft->hidden_size} * f);
    l.draft_h = take(std::size_t{draft->hidden_size} * f);
    l.draft_q = take(std::size_t{MaxQDim(*draft)} * f);
    l.draft_attn = take(std::size_t{MaxQDim(*draft)} * f);
    l.draft_o = take(std::size_t{draft->hidden_size} * f);
    l.draft_gate = take(std::size_t{draft->ffn_size} * f);
    l.draft_up = take(std::size_t{draft->ffn_size} * f);
    l.draft_logits = take(std::size_t{vocab} * f);
    l.draft_next = take(std::size_t{draft->target_hidden_size} * f);
    // Top-64 proposal candidates for sampled drafting: ids, then scores.
    const std::size_t candidate_ids =
        qwen38_flash_next::rocm::MtpCandidateWorkspaceSize(vocab);
    l.draft_candidates = take(candidate_ids * sizeof(std::uint32_t));
    l.draft_candidate_scratch = take(candidate_ids * sizeof(std::uint32_t));
  }
  l.total = at;
  return l;
}

}  // namespace

KvCache::~KvCache() {
  if (allocation != nullptr) {
    hip::LogCleanupError(hipFree(allocation));
  }
}

std::uint32_t RingSlots(const Config& config, std::uint32_t max_rows) noexcept {
  const std::uint32_t slots = config.sliding_window + max_rows;
  return (slots + 255U) / 256U * 256U;
}

std::size_t Executor::ScratchBytes(const Config& config, const Config* draft,
                                   std::uint32_t vocab, std::size_t max_cols,
                                   std::uint32_t max_rows,
                                   std::uint32_t max_logit_rows,
                                   std::uint32_t max_context) {
  return Plan(config, draft, vocab, max_cols, max_rows, max_logit_rows,
              max_context)
      .total;
}

std::size_t Executor::CacheBytes(const Config& c, std::uint32_t max_context,
                                 std::uint32_t ring) {
  std::size_t bytes = 0;
  for (std::uint32_t l = 0; l < c.num_layers; ++l) {
    const std::size_t slots =
        c.IsSliding(l) ? std::min(ring, max_context) : max_context;
    bytes += 2 * AlignUp(slots * c.KvDim(l) * sizeof(std::uint16_t));
  }
  return bytes + AlignUp(std::size_t{c.hidden_size} * sizeof(float));
}

Executor::Executor(const DeviceModel& model, std::uint32_t max_rows,
                   std::uint32_t max_logit_rows, std::uint32_t max_context)
    : model_(model),
      max_rows_(max_rows),
      max_logit_rows_(max_logit_rows),
      max_context_(max_context),
      ring_(RingSlots(model.config(), max_rows)) {
  const Layout l =
      Plan(model.config(), model.has_draft() ? &model.draft().config : nullptr,
           model.vocab_size(), model.max_cols(), max_rows, max_logit_rows,
           max_context);
  HIP_CHECK(hipStreamCreateWithFlags(&stream_, hipStreamNonBlocking));
  HIP_CHECK(hipMalloc(&scratch_, l.total));
  auto* base = static_cast<std::uint8_t*>(scratch_);
  tokens_ = reinterpret_cast<std::int32_t*>(base + l.tokens);
  logit_index_ = reinterpret_cast<std::uint32_t*>(base + l.logit_index);
  x_ = reinterpret_cast<float*>(base + l.x);
  h_ = reinterpret_cast<float*>(base + l.h);
  q_ = reinterpret_cast<float*>(base + l.q);
  k_ = reinterpret_cast<float*>(base + l.k);
  v_ = reinterpret_cast<float*>(base + l.v);
  attn_ = reinterpret_cast<float*>(base + l.attn);
  o_ = reinterpret_cast<float*>(base + l.o);
  gate_ = reinterpret_cast<float*>(base + l.gate);
  up_ = reinterpret_cast<float*>(base + l.up);
  hsel_ = reinterpret_cast<float*>(base + l.hsel);
  logits_ = reinterpret_cast<float*>(base + l.logits);
  partials_ = reinterpret_cast<float*>(base + l.partials);
  q8_ = base + l.q8;
  if (model.has_draft()) {
    draft_tokens_ = reinterpret_cast<std::uint32_t*>(base + l.draft_tokens);
    draft_concat_ = reinterpret_cast<float*>(base + l.draft_concat);
    draft_x_ = reinterpret_cast<float*>(base + l.draft_x);
    draft_h_ = reinterpret_cast<float*>(base + l.draft_h);
    draft_q_ = reinterpret_cast<float*>(base + l.draft_q);
    draft_attn_ = reinterpret_cast<float*>(base + l.draft_attn);
    draft_o_ = reinterpret_cast<float*>(base + l.draft_o);
    draft_gate_ = reinterpret_cast<float*>(base + l.draft_gate);
    draft_up_ = reinterpret_cast<float*>(base + l.draft_up);
    draft_logits_ = reinterpret_cast<float*>(base + l.draft_logits);
    draft_next_ = reinterpret_cast<float*>(base + l.draft_next);
    draft_candidates_ =
        reinterpret_cast<std::uint32_t*>(base + l.draft_candidates);
    draft_candidate_scratch_ =
        reinterpret_cast<std::uint32_t*>(base + l.draft_candidate_scratch);
    HIP_CHECK(hipHostMalloc(&draft_candidates_host_,
                            sizeof(qwen38_flash_next::MtpCandidateLogits)));
  }
}

Executor::~Executor() {
  if (draft_candidates_host_ != nullptr) {
    hip::LogCleanupError(hipHostFree(draft_candidates_host_));
  }
  if (scratch_ != nullptr) {
    hip::LogCleanupError(hipFree(scratch_));
  }
  if (stream_ != nullptr) {
    hip::LogCleanupError(hipStreamDestroy(stream_));
  }
}

std::unique_ptr<KvCache> Executor::CreateCache(std::uint32_t max_context,
                                               std::string* error_msg) const {
  const Config& c = model_.config();
  if (max_context == 0 || max_context > max_context_) {
    if (error_msg != nullptr) {
      *error_msg = "session context exceeds the executor capacity";
    }
    return nullptr;
  }
  auto cache = std::make_unique<KvCache>();
  cache->max_context = max_context;
  cache->ring = ring_;
  cache->bytes = CacheBytes(c, max_context, ring_);
  if (hipMalloc(&cache->allocation, cache->bytes) != hipSuccess) {
    if (error_msg != nullptr) {
      *error_msg = "KV cache allocation failed (" +
                   std::to_string(cache->bytes) + " bytes)";
    }
    cache->allocation = nullptr;
    return nullptr;
  }
  auto* at = static_cast<std::uint8_t*>(cache->allocation);
  for (std::uint32_t l = 0; l < c.num_layers; ++l) {
    const std::size_t slots =
        c.IsSliding(l) ? std::min(ring_, max_context) : max_context;
    const std::size_t bytes =
        AlignUp(slots * c.KvDim(l) * sizeof(std::uint16_t));
    cache->k.push_back(reinterpret_cast<std::uint16_t*>(at));
    cache->v.push_back(reinterpret_cast<std::uint16_t*>(at + bytes));
    at += 2 * bytes;
  }
  cache->hidden = reinterpret_cast<float*>(at);
  return cache;
}

const void* Executor::Quantize(const float* x, std::uint32_t rows,
                               std::uint32_t cols) {
  if (rows <= kSplitRows) {
    return nullptr;
  }
  hip::LaunchQuantizeActivationQ8_1FromFp32(x, q8_, rows, cols, stream_);
  return q8_;
}

namespace {

std::optional<GemvFormat> GemvFormatOf(core::GgmlType type) {
  switch (type) {
    case core::GgmlType::kQ4_K:
      return GemvFormat::kQ4_K;
    case core::GgmlType::kQ5_K:
      return GemvFormat::kQ5_K;
    case core::GgmlType::kQ6_K:
      return GemvFormat::kQ6_K;
    default:
      return std::nullopt;
  }
}

}  // namespace

void Executor::Project(const DeviceTensor& w, const float* x, const void* xq,
                       std::uint32_t rows, float* y) {
  if (rows > kSplitRows) {
    hip::LaunchBatchedQuantGEMMPreQuantized(w.type, w.data, xq, y, rows, w.rows,
                                            w.cols, stream_);
    return;
  }
  if (rows > 1) {
    hip::LaunchBatchedQuantGEMMFp32(w.type, w.data, x, y, rows, w.rows, w.cols,
                                    stream_);
    return;
  }
  if (model_.has_draft()) {
    // Speculation verifies with the small-batch kernel, and single tokens
    // must round identically: its bit-identical one-row twins, whichever is
    // faster for the shape (measured with cold weights).
    if (w.rows <= 4096) {
      hip::LaunchGEMV(w.data, w.type, x, y, w.rows, w.cols, stream_);
    } else {
      hip::LaunchBatchedQuantGEMMFp32(w.type, w.data, x, y, 1, w.rows, w.cols,
                                      stream_);
    }
    return;
  }
  // Autoregressive decode: the Gemma GEMV serves every K-quant projection.
  if (const auto format = GemvFormatOf(w.type);
      format &&
      LaunchKQuantGemv(*format, w.data, x, y, w.rows, w.cols, stream_)) {
    return;
  }
  hip::LaunchGEMV(w.data, w.type, x, y, w.rows, w.cols, stream_);
}

void Executor::Forward(KvCache& cache, std::span<const std::int32_t> tokens,
                       std::uint32_t first_position,
                       std::span<const std::uint32_t> logit_rows) {
  const Config& c = model_.config();
  const auto n = static_cast<std::uint32_t>(tokens.size());
  if (n == 0 || n > max_rows_ ||
      first_position + static_cast<std::uint64_t>(n) > cache.max_context ||
      logit_rows.size() > max_logit_rows_ || cache.ring != ring_) {
    throw std::invalid_argument("gemma4 forward exceeds its capacity");
  }
  const std::uint32_t d = c.hidden_size;
  const float eps = c.rms_eps;
  const auto& layers = model_.layers();
  HIP_CHECK(hipMemcpyAsync(tokens_, tokens.data(), n * sizeof(std::int32_t),
                           hipMemcpyHostToDevice, stream_));
  hip::LaunchBatchedEmbeddingLookup(
      model_.token_embd().data, model_.token_embd().type,
      reinterpret_cast<const std::uint32_t*>(tokens_), x_, n, d, stream_);
  ScaleRmsNorm(x_, std::sqrt(static_cast<float>(d)), layers[0].attn_norm.f32(),
               h_, n, d, eps, stream_);

  for (std::uint32_t l = 0; l < c.num_layers; ++l) {
    const DeviceLayer& L = layers[l];
    const bool sliding = c.IsSliding(l);
    const std::uint32_t dim = c.HeadDim(l);
    const void* hq = Quantize(h_, n, d);
    Project(L.attn_q, h_, hq, n, q_);
    Project(L.attn_k, h_, hq, n, k_);
    const float* v_source = k_;
    if (!L.attn_v.empty()) {
      Project(L.attn_v, h_, hq, n, v_);
      v_source = v_;
    }
    QkvPostArgs post{};
    post.q = q_;
    post.k = k_;
    post.v = v_source;
    post.q_norm = L.attn_q_norm.f32();
    post.k_norm = L.attn_k_norm.f32();
    post.theta_scale =
        std::pow(c.RopeTheta(l), -2.0F / static_cast<float>(dim));
    post.freq_factors = sliding ? nullptr : model_.rope_factors();
    post.k_cache = cache.k[l];
    post.v_cache = cache.v[l];
    post.rows = n;
    post.heads = c.num_heads;
    post.kv_heads = c.kv_heads[l];
    post.head_dim = dim;
    post.first_position = first_position;
    post.ring = sliding ? cache.ring : 0;
    post.eps = eps;
    QkvPost(post, stream_);

    AttentionArgs att{};
    att.q = q_;
    att.k_cache = cache.k[l];
    att.v_cache = cache.v[l];
    att.out = attn_;
    att.partials = partials_;
    att.rows = n;
    att.heads = c.num_heads;
    att.kv_heads = c.kv_heads[l];
    att.head_dim = dim;
    att.first_position = first_position;
    att.shared_position = false;
    att.key_limit = std::numeric_limits<std::uint32_t>::max();
    att.window = sliding ? c.sliding_window : 0;
    att.ring = sliding ? cache.ring : 0;
    Attention(att, stream_);

    Project(L.attn_output, attn_, Quantize(attn_, n, c.QDim(l)), n, o_);
    PostAttentionNorm(o_, L.post_attn_norm.f32(), x_, L.ffn_norm.f32(), h_, n,
                      d, eps, stream_);
    const void* fq = Quantize(h_, n, d);
    Project(L.ffn_gate, h_, fq, n, gate_);
    Project(L.ffn_up, h_, fq, n, up_);
    GeGlu(gate_, up_, gate_, std::size_t{n} * c.ffn_size, stream_);
    Project(L.ffn_down, gate_, Quantize(gate_, n, c.ffn_size), n, o_);
    const float* next = l + 1 < c.num_layers ? layers[l + 1].attn_norm.f32()
                                             : model_.output_norm().f32();
    PostFeedForwardNorm(o_, L.post_ffn_norm.f32(), L.output_scale, x_, next, h_,
                        n, d, eps, stream_);
  }

  const auto m = static_cast<std::uint32_t>(logit_rows.size());
  if (m == 0) {
    return;
  }
  HIP_CHECK(hipMemcpyAsync(logit_index_, logit_rows.data(),
                           m * sizeof(std::uint32_t), hipMemcpyHostToDevice,
                           stream_));
  GatherRows(h_, logit_index_, hsel_, m, d, stream_);
  Project(model_.output(), hsel_, Quantize(hsel_, m, d), m, logits_);
  if (c.final_logit_softcap > 0.0F) {
    Softcap(logits_, std::size_t{m} * model_.vocab_size(),
            c.final_logit_softcap, stream_);
  }
}

void Executor::CommitHidden(KvCache& cache, std::uint32_t row) {
  const std::size_t d = model_.config().hidden_size;
  HIP_CHECK(hipMemcpyAsync(cache.hidden, h_ + row * d, d * sizeof(float),
                           hipMemcpyDeviceToDevice, stream_));
}

void Executor::DraftChain(KvCache& cache, std::int32_t token,
                          std::uint32_t position, std::uint32_t steps,
                          std::vector<std::int32_t>* drafts,
                          const DraftProposer& propose) {
  if (!model_.has_draft() || steps == 0 || steps > kMaxDraftTokens) {
    throw std::invalid_argument("gemma4 draft chain exceeds its capacity");
  }
  const Config& tc = model_.config();
  const DeviceDraft& dm = model_.draft();
  const Config& c = dm.config;
  const std::size_t target_hidden = tc.hidden_size;
  const float eps = c.rms_eps;
  const float embed_scale = std::sqrt(static_cast<float>(target_hidden));
  HIP_CHECK(hipMemcpyAsync(draft_tokens_, &token, sizeof(token),
                           hipMemcpyHostToDevice, stream_));
  for (std::uint32_t step = 0; step < steps; ++step) {
    // [scaled target embedding of the token ; target-width hidden state]
    hip::LaunchBatchedEmbeddingLookup(
        model_.token_embd().data, model_.token_embd().type,
        draft_tokens_ + step, draft_concat_, 1, target_hidden, stream_);
    Scale(draft_concat_, embed_scale, draft_concat_, target_hidden, stream_);
    HIP_CHECK(hipMemcpyAsync(
        draft_concat_ + target_hidden, step == 0 ? cache.hidden : draft_next_,
        target_hidden * sizeof(float), hipMemcpyDeviceToDevice, stream_));
    Project(dm.pre_projection, draft_concat_, nullptr, 1, draft_x_);
    RmsNorm(draft_x_, dm.layers[0].attn_norm.f32(), draft_h_, 1, c.hidden_size,
            eps, stream_);
    for (std::uint32_t l = 0; l < c.num_layers; ++l) {
      const DeviceLayer& L = dm.layers[l];
      const bool sliding = c.IsSliding(l);
      const std::uint32_t dim = c.HeadDim(l);
      const std::uint32_t source = c.SharedKvSource(l, tc);
      Project(L.attn_q, draft_h_, nullptr, 1, draft_q_);
      QueryPost(draft_q_, L.attn_q_norm.f32(),
                std::pow(c.RopeTheta(l), -2.0F / static_cast<float>(dim)),
                sliding ? nullptr : dm.rope_factors, 1, c.num_heads, dim,
                position, true, eps, stream_);
      AttentionArgs att{};
      att.q = draft_q_;
      att.k_cache = cache.k[source];
      att.v_cache = cache.v[source];
      att.out = draft_attn_;
      att.partials = partials_;
      att.rows = 1;
      att.heads = c.num_heads;
      att.kv_heads = c.kv_heads[l];
      att.head_dim = dim;
      att.first_position = position;
      att.shared_position = true;
      att.key_limit = position;  // committed keys only
      att.window = sliding ? c.sliding_window : 0;
      att.ring = sliding ? cache.ring : 0;
      Attention(att, stream_);
      Project(L.attn_output, draft_attn_, nullptr, 1, draft_o_);
      PostAttentionNorm(draft_o_, L.post_attn_norm.f32(), draft_x_,
                        L.ffn_norm.f32(), draft_h_, 1, c.hidden_size, eps,
                        stream_);
      Project(L.ffn_gate, draft_h_, nullptr, 1, draft_gate_);
      Project(L.ffn_up, draft_h_, nullptr, 1, draft_up_);
      GeGlu(draft_gate_, draft_up_, draft_gate_, c.ffn_size, stream_);
      Project(L.ffn_down, draft_gate_, nullptr, 1, draft_o_);
      const float* next = l + 1 < c.num_layers
                              ? dm.layers[l + 1].attn_norm.f32()
                              : dm.output_norm.f32();
      PostFeedForwardNorm(draft_o_, L.post_ffn_norm.f32(), L.output_scale,
                          draft_x_, next, draft_h_, 1, c.hidden_size, eps,
                          stream_);
    }
    // draft_h_ is the output-normalized state: vocabulary head and the
    // projection back to the target width for the next step.
    Project(dm.token_embd, draft_h_, nullptr, 1, draft_logits_);
    if (propose) {
      // Sampled drafting: the host draws from the top-64 candidates and
      // returns the proposal the next step embeds.
      using qwen38_flash_next::kMtpCandidates;
      using qwen38_flash_next::MtpCandidateLogits;
      const std::uint32_t vocab = model_.vocab_size();
      qwen38_flash_next::rocm::MtpTopCandidates(
          draft_logits_, draft_candidates_, draft_candidate_scratch_,
          reinterpret_cast<float*>(draft_candidates_ + kMtpCandidates), vocab,
          stream_);
      auto& host = *draft_candidates_host_;
      host.size = std::min<std::size_t>(vocab, kMtpCandidates);
      static_assert(offsetof(MtpCandidateLogits, logits) ==
                    kMtpCandidates * sizeof(std::uint32_t));
      HIP_CHECK(hipMemcpyAsync(
          &host, draft_candidates_,
          offsetof(MtpCandidateLogits, logits) + host.size * sizeof(float),
          hipMemcpyDeviceToHost, stream_));
      HIP_CHECK(hipStreamSynchronize(stream_));
      const std::int32_t proposal = propose(host);
      HIP_CHECK(hipMemcpyAsync(draft_tokens_ + step + 1, &proposal,
                               sizeof(proposal), hipMemcpyHostToDevice,
                               stream_));
    } else {
      // The FFN gate buffer is idle after the vocabulary projection. Reuse
      // it for parallel vocabulary tiles instead of reducing 262K logits in
      // one workgroup.
      hip::LaunchBatchedGPUArgmax(
          draft_logits_, draft_tokens_ + step + 1, 1, model_.vocab_size(),
          std::span<float>(draft_gate_, c.ffn_size), stream_);
    }
    if (step + 1 < steps) {
      Project(dm.post_projection, draft_h_, nullptr, 1, draft_next_);
    }
  }
  drafts->resize(steps);
  HIP_CHECK(hipMemcpyAsync(drafts->data(), draft_tokens_ + 1,
                           steps * sizeof(std::int32_t), hipMemcpyDeviceToHost,
                           stream_));
  HIP_CHECK(hipStreamSynchronize(stream_));
}

void Executor::CopyLogits(std::size_t rows, std::vector<float>* out) const {
  out->resize(rows * model_.vocab_size());
  HIP_CHECK(hipMemcpyAsync(out->data(), logits_, out->size() * sizeof(float),
                           hipMemcpyDeviceToHost, stream_));
  HIP_CHECK(hipStreamSynchronize(stream_));
}

}  // namespace gufo::models::gemma4::rocm
