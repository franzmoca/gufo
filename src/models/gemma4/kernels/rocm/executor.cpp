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
                                 std::uint32_t ring,
                                 const std::vector<std::uint32_t>& key_widths) {
  std::size_t bytes = 0;
  for (std::uint32_t l = 0; l < c.num_layers; ++l) {
    const std::size_t slots =
        c.IsSliding(l) ? std::min(ring, max_context) : max_context;
    bytes += AlignUp(slots * key_widths[l] * sizeof(std::uint16_t)) +
             AlignUp(slots * c.KvDim(l) * sizeof(std::uint16_t));
  }
  return bytes + AlignUp(std::size_t{c.hidden_size} * sizeof(float));
}

std::vector<std::uint32_t> Executor::KeyWidths(const DeviceModel& model) {
  const Config& c = model.config();
  std::vector<std::uint32_t> widths(c.num_layers);
  for (std::uint32_t l = 0; l < c.num_layers; ++l) {
    // One projection feeds K and V: the cache keeps only the rotated key
    // dims; the others are the values scaled by k_norm.
    const std::uint32_t pairs = model.global_rope_pairs();
    const bool derived = !c.IsSliding(l) && c.HeadDim(l) == 512 &&
                         model.layers()[l].attn_v.empty() && pairs % 16 == 0 &&
                         pairs <= kMaxDerivedKeyPairs;
    widths[l] = derived ? c.kv_heads[l] * 2 * pairs : c.KvDim(l);
  }
  return widths;
}

Executor::Executor(const DeviceModel& model, std::uint32_t max_rows,
                   std::uint32_t max_logit_rows, std::uint32_t max_context)
    : model_(model),
      max_rows_(max_rows),
      max_logit_rows_(max_logit_rows),
      max_context_(max_context),
      ring_(RingSlots(model.config(), max_rows)),
      key_widths_(KeyWidths(model)) {
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
  cache->bytes = CacheBytes(c, max_context, ring_, key_widths_);
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
    at += AlignUp(slots * key_widths_[l] * sizeof(std::uint16_t));
    cache->v.push_back(reinterpret_cast<std::uint16_t*>(at));
    at += bytes;
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
    // Gemma's K-quant shapes lie outside the Qwen-tuned dispatch; the
    // double-stage configuration is faster on them (the vocabulary head keeps
    // the default). Both round like the one-row twins below.
    if (w.rows < 65536 &&
        hip::LaunchKQuantSmallBatchDoubleStage(w.type, w.data, x, y, rows,
                                               w.rows, w.cols, stream_)) {
      return;
    }
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

bool Executor::DerivedKeys(std::uint32_t layer) const {
  return key_widths_[layer] != model_.config().KvDim(layer);
}

void Executor::DeriveKeys(AttentionArgs& att, std::uint32_t layer) const {
  if (DerivedKeys(layer)) {
    att.rope_pairs = model_.global_rope_pairs();
  }
}

void Executor::Forward(KvCache& cache, std::span<const std::int32_t> tokens,
                       std::uint32_t first_position,
                       std::span<const std::uint32_t> logit_rows) {
  const Segment segment{&cache, first_position,
                        static_cast<std::uint32_t>(tokens.size())};
  Forward(std::span<const Segment>(&segment, 1), tokens, logit_rows);
}

void Executor::Forward(std::span<const Segment> segments,
                       std::span<const std::int32_t> tokens,
                       std::span<const std::uint32_t> logit_rows) {
  const Config& c = model_.config();
  const auto n = static_cast<std::uint32_t>(tokens.size());
  std::uint64_t total = 0;
  for (const Segment& s : segments) {
    if (s.rows == 0 || s.cache->ring != ring_ ||
        s.first_position + std::uint64_t{s.rows} > s.cache->max_context) {
      throw std::invalid_argument("gemma4 forward segment exceeds its cache");
    }
    total += s.rows;
  }
  // Several sessions share a forward only at decode widths, where every
  // projection keeps its single-session arithmetic.
  if (n == 0 || n > max_rows_ || total != n ||
      (segments.size() > 1 && n > kSplitRows) ||
      logit_rows.size() > max_logit_rows_) {
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

  // Prefill (rows past the small-batch width) quantizes activations for the
  // W8A8 GEMMs; the norms and GeGLU write that encoding directly.
  const bool prefill = n > kSplitRows;
  for (std::uint32_t l = 0; l < c.num_layers; ++l) {
    const DeviceLayer& L = layers[l];
    const bool sliding = c.IsSliding(l);
    const std::uint32_t dim = c.HeadDim(l);
    // Layer l > 0 of a prefill reads the Q8_1 rows the previous layer's
    // post-FFN norm wrote.
    const void* hq = prefill && l > 0 ? q8_ : Quantize(h_, n, d);
    Project(L.attn_q, h_, hq, n, q_);
    Project(L.attn_k, h_, hq, n, k_);
    const float* v_source = k_;
    if (!L.attn_v.empty()) {
      Project(L.attn_v, h_, hq, n, v_);
      v_source = v_;
    }
    std::uint32_t row0 = 0;
    for (const Segment& seg : segments) {
      KvCache& cache = *seg.cache;
      QkvPostArgs post{};
      post.q = q_ + std::size_t{row0} * c.QDim(l);
      post.k = k_ + std::size_t{row0} * c.KvDim(l);
      post.v = v_source + std::size_t{row0} * c.KvDim(l);
      post.q_norm = L.attn_q_norm.f32();
      post.k_norm = L.attn_k_norm.f32();
      post.theta_scale =
          std::pow(c.RopeTheta(l), -2.0F / static_cast<float>(dim));
      post.freq_factors = sliding ? nullptr : model_.rope_factors();
      post.k_cache = cache.k[l];
      post.v_cache = cache.v[l];
      post.rows = seg.rows;
      post.heads = c.num_heads;
      post.kv_heads = c.kv_heads[l];
      post.head_dim = dim;
      post.first_position = seg.first_position;
      post.ring = sliding ? cache.ring : 0;
      post.eps = eps;
      post.rotated_pairs = DerivedKeys(l) ? model_.global_rope_pairs() : 0;
      QkvPost(post, stream_);

      AttentionArgs att{};
      att.q = q_ + std::size_t{row0} * c.QDim(l);
      att.k_cache = cache.k[l];
      att.v_cache = cache.v[l];
      att.out = attn_ + std::size_t{row0} * c.QDim(l);
      att.partials = partials_;
      att.rows = seg.rows;
      att.heads = c.num_heads;
      att.kv_heads = c.kv_heads[l];
      att.head_dim = dim;
      att.first_position = seg.first_position;
      att.shared_position = false;
      att.key_limit = std::numeric_limits<std::uint32_t>::max();
      att.window = sliding ? c.sliding_window : 0;
      att.ring = sliding ? cache.ring : 0;
      DeriveKeys(att, l);
      Attention(att, stream_);
      row0 += seg.rows;
    }

    Project(L.attn_output, attn_, Quantize(attn_, n, c.QDim(l)), n, o_);
    PostAttentionNorm(o_, L.post_attn_norm.f32(), x_, L.ffn_norm.f32(), h_, n,
                      d, eps, stream_, prefill ? q8_ : nullptr);
    const void* fq = prefill ? q8_ : nullptr;
    Project(L.ffn_gate, h_, fq, n, gate_);
    Project(L.ffn_up, h_, fq, n, up_);
    const void* gq = nullptr;
    if (prefill) {
      GeGluQuantize(gate_, up_, q8_, n, c.ffn_size, stream_);
      gq = q8_;
    } else {
      GeGlu(gate_, up_, gate_, std::size_t{n} * c.ffn_size, stream_);
    }
    Project(L.ffn_down, gate_, gq, n, o_);
    const bool last = l + 1 == c.num_layers;
    const float* next =
        !last ? layers[l + 1].attn_norm.f32() : model_.output_norm().f32();
    // The output-normed rows of the last layer feed the vocabulary head and
    // the drafter in FP32.
    PostFeedForwardNorm(o_, L.post_ffn_norm.f32(), L.output_scale, x_, next, h_,
                        n, d, eps, stream_, prefill && !last ? q8_ : nullptr);
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
  if (!model_.has_draft() || steps == 0 || steps > kMaxDraftTokens ||
      !propose) {
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
  drafts->clear();
  drafts->reserve(steps);
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
                position, true, eps, model_.layers()[source].attn_k_norm.f32(),
                DerivedKeys(source) ? model_.global_rope_pairs() : 0, stream_);
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
      DeriveKeys(att, source);
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
    // projection back to the target width for the next step. Proposals have
    // no decode twin to match, so the fastest one-row kernel reads the head
    // (repacked as Q4_K at load).
    if (const auto format = GemvFormatOf(dm.token_embd.type);
        !format ||
        !LaunchKQuantGemv(*format, dm.token_embd.data, draft_h_, draft_logits_,
                          dm.token_embd.rows, dm.token_embd.cols, stream_)) {
      Project(dm.token_embd, draft_h_, nullptr, 1, draft_logits_);
    }
    // The host picks each proposal (and where the chain ends) from the
    // top-64 candidates; the next step embeds it.
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
    const DraftProposal proposal = propose(host);
    if (!proposal.token) {
      break;
    }
    drafts->push_back(*proposal.token);
    if (proposal.last) {
      break;
    }
    HIP_CHECK(hipMemcpyAsync(draft_tokens_ + step + 1, &drafts->back(),
                             sizeof(std::int32_t), hipMemcpyHostToDevice,
                             stream_));
    if (step + 1 < steps) {
      Project(dm.post_projection, draft_h_, nullptr, 1, draft_next_);
    }
  }
}

void Executor::CopyLogits(std::size_t rows, std::vector<float>* out) const {
  out->resize(rows * model_.vocab_size());
  HIP_CHECK(hipMemcpyAsync(out->data(), logits_, out->size() * sizeof(float),
                           hipMemcpyDeviceToHost, stream_));
  HIP_CHECK(hipStreamSynchronize(stream_));
}

}  // namespace gufo::models::gemma4::rocm
