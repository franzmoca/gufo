#include "src/models/gemma4/engine.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <exception>
#include <optional>
#include <utility>

#include "src/core/gguf_reader.hpp"
#include "src/models/gemma4/chat_template.hpp"
#include "src/models/gemma4/kernels/rocm/device_model.hpp"
#include "src/models/gemma4/kernels/rocm/executor.hpp"
#include "src/models/gemma4/weights.hpp"
#include "src/models/qwen38_flash_next/mtp_sampling.hpp"

namespace gufo::models::gemma4 {
namespace {

constexpr std::array<char, 8> kSnapshotMagic = {'G', '4', 'S', 'N',
                                                'A', 'P', '0', '1'};

struct SnapshotHeader {
  std::array<char, 8> magic;
  std::uint32_t version;
  std::uint32_t position;
  std::uint32_t hidden;
  std::uint32_t vocab;
  std::uint32_t layers;
  std::uint32_t window;
};
static_assert(sizeof(SnapshotHeader) == 32);

/// First position whose KV a later token may attend in `layer`.
std::uint32_t FirstLiveRow(const Config& c, std::uint32_t layer,
                           std::uint32_t position) {
  if (!c.IsSliding(layer) || position < c.sliding_window) {
    return 0;
  }
  return position - (c.sliding_window - 1);
}

bool Fail(std::string* error_msg, std::string message) {
  if (error_msg != nullptr) {
    *error_msg = std::move(message);
  }
  return false;
}

}  // namespace

Model::Model() = default;
Model::~Model() = default;

std::shared_ptr<Model> Model::Load(const std::string& model_path,
                                   const ModelOptions& options,
                                   std::string* error_msg) {
  std::shared_ptr<Model> m(new Model());
  m->options_ = options;
  if (options.max_context == 0 || options.prefill_chunk == 0 ||
      options.max_logit_rows == 0) {
    Fail(error_msg, "gemma4 model options must be positive");
    return nullptr;
  }
  std::string error;
  std::unique_ptr<core::GgufReader> reader =
      core::GgufReader::OpenFile(model_path, &error);
  if (!reader) {
    Fail(error_msg, error);
    return nullptr;
  }
  m->reader_ = std::move(reader);
  if (!ChatTemplate::ValidateGgufTemplate(*m->reader_, &error)) {
    Fail(error_msg, error);
    return nullptr;
  }
  auto weights = ModelWeights::Bind(*m->reader_, &error);
  if (!weights) {
    Fail(error_msg, error);
    return nullptr;
  }
  m->weights_ = std::make_unique<ModelWeights>(std::move(*weights));
  if (options.max_context > m->weights_->config.context_length) {
    Fail(error_msg, "requested context exceeds the model's native " +
                        std::to_string(m->weights_->config.context_length));
    return nullptr;
  }
  m->tokenizer_ = Tokenizer::CreateFromGguf(*m->reader_, &error);
  if (!m->tokenizer_) {
    Fail(error_msg, error);
    return nullptr;
  }
  if (!options.mtp_model_path.empty()) {
    if (options.draft_tokens == 0 ||
        options.draft_tokens > rocm::kMaxDraftTokens) {
      Fail(error_msg, "gemma4 draft tokens must be 1.." +
                          std::to_string(rocm::kMaxDraftTokens));
      return nullptr;
    }
    std::unique_ptr<core::GgufReader> draft_reader =
        core::GgufReader::OpenFile(options.mtp_model_path, &error);
    if (!draft_reader) {
      Fail(error_msg, error);
      return nullptr;
    }
    m->draft_reader_ = std::move(draft_reader);
    auto draft = DraftWeights::Bind(*m->draft_reader_, *m->weights_, &error);
    if (!draft) {
      Fail(error_msg, "MTP drafter: " + error);
      return nullptr;
    }
    m->draft_weights_ = std::make_unique<DraftWeights>(std::move(*draft));
  }
  m->device_ = rocm::DeviceModel::Upload(*m->weights_, *m->reader_,
                                         m->draft_weights_.get(),
                                         m->draft_reader_.get(), &error);
  if (!m->device_) {
    Fail(error_msg, error);
    return nullptr;
  }
  try {
    m->executor_ = std::make_unique<rocm::Executor>(
        *m->device_, options.prefill_chunk, options.max_logit_rows,
        options.max_context);
  } catch (const std::exception& e) {
    Fail(error_msg, e.what());
    return nullptr;
  }
  return m;
}

std::unique_ptr<Session> Model::CreateSession(std::uint32_t max_context,
                                              std::string* error_msg) {
  if (max_context == 0) {
    max_context = options_.max_context;
  }
  auto cache = executor_->CreateCache(max_context, error_msg);
  if (!cache) {
    return nullptr;
  }
  return std::unique_ptr<Session>(
      new Session(shared_from_this(), std::move(cache)));
}

std::vector<TokenId> Model::Tokenize(std::string_view text) const {
  return tokenizer_->Encode(text, false, true);
}

std::string Model::Decode(std::span<const TokenId> tokens) const {
  return tokenizer_->Decode(tokens, false);
}

std::string Model::TokenText(TokenId token) const {
  return tokenizer_->TokenText(token, false);
}

bool Model::IsStopToken(TokenId token) const noexcept {
  return tokenizer_->IsEndOfGeneration(token);
}

std::uint32_t Model::VocabSize() const noexcept {
  return weights_->vocab_size;
}

const Config& Model::config() const noexcept {
  return weights_->config;
}

std::size_t Model::ResidentBytes() const noexcept {
  return device_->resident_bytes() +
         rocm::Executor::ScratchBytes(
             config(), HasMtp() ? &draft_weights_->config : nullptr,
             VocabSize(), device_->max_cols(), options_.prefill_chunk,
             options_.max_logit_rows, options_.max_context);
}

std::size_t Model::SessionBytes(std::uint32_t context) const noexcept {
  return rocm::Executor::CacheBytes(config(), context, executor_->ring(),
                                    executor_->key_widths());
}

std::size_t Model::SessionRingSlots() const noexcept {
  return executor_->ring();
}

std::string Model::ModelName() const {
  const auto name = reader_->GetMetadataString("general.name");
  return name ? std::string(*name) : std::string("gemma4");
}

Session::Session(std::shared_ptr<Model> model,
                 std::unique_ptr<rocm::KvCache> cache)
    : model_(std::move(model)), cache_(std::move(cache)) {}

Session::~Session() = default;

std::uint32_t Session::ContextSize() const noexcept {
  return cache_->max_context;
}

std::size_t Session::AllocatedBytes() const noexcept {
  return cache_->bytes;
}

void Session::Reset() {
  pending_.reset();
  tokens_.clear();
  logits_.clear();
  valid_ = false;
}

bool Session::Extend(std::size_t begin, std::string* error_msg) {
  auto& executor = *model_->executor_;
  const std::size_t chunk = executor.max_rows();
  valid_ = false;
  try {
    std::lock_guard lock(model_->mutex_);
    for (std::size_t start = begin; start < tokens_.size(); start += chunk) {
      const std::size_t count = std::min(chunk, tokens_.size() - start);
      const bool last = start + count == tokens_.size();
      const std::uint32_t last_row = static_cast<std::uint32_t>(count - 1);
      executor.Forward(*cache_, std::span(tokens_).subspan(start, count),
                       static_cast<std::uint32_t>(start),
                       last ? std::span<const std::uint32_t>(&last_row, 1)
                            : std::span<const std::uint32_t>{});
    }
    const std::size_t tail = (tokens_.size() - begin - 1) % chunk;
    executor.CommitHidden(*cache_, static_cast<std::uint32_t>(tail));
    executor.CopyLogits(1, &logits_);
  } catch (const std::exception& e) {
    tokens_.resize(begin);
    return Fail(error_msg, e.what());
  }
  valid_ = true;
  return true;
}

bool Session::Sync(std::span<const TokenId> prompt, std::string* error_msg) {
  if (prompt.empty()) {
    Reset();
    return Fail(error_msg, "prompt is empty");
  }
  if (prompt.size() > cache_->max_context) {
    return Fail(error_msg, "prompt exceeds the session context");
  }
  pending_.reset();
  std::size_t common = 0;
  const std::size_t limit = std::min(prompt.size(), tokens_.size());
  while (common < limit && prompt[common] == tokens_[common]) {
    ++common;
  }
  if (common == prompt.size() && common == tokens_.size() && valid_) {
    return true;
  }
  // The last prompt token is re-evaluated to produce its logits.
  common = std::min(common, prompt.size() - 1);
  // Rewinding needs the sliding window before `common` still in the ring.
  const Config& c = model_->config();
  if (tokens_.size() - common + c.sliding_window > cache_->ring) {
    common = 0;
  }
  tokens_.assign(prompt.begin(), prompt.end());
  return Extend(common, error_msg);
}

bool Session::Evaluate(TokenId token, std::string* error_msg) {
  pending_.reset();
  if (tokens_.size() >= cache_->max_context) {
    return Fail(error_msg, "session context is full");
  }
  tokens_.push_back(token);
  return Extend(tokens_.size() - 1, error_msg);
}

bool Session::EvaluateAll(std::span<const TokenId> tokens,
                          std::vector<float>* logits, std::string* error_msg) {
  auto& executor = *model_->executor_;
  if (tokens_.size() + tokens.size() > cache_->max_context) {
    return Fail(error_msg, "tokens exceed the session context");
  }
  const std::size_t vocab = model_->VocabSize();
  const std::size_t chunk = std::min<std::size_t>(
      executor.max_rows(), model_->options_.max_logit_rows);
  logits->clear();
  valid_ = false;
  pending_.reset();
  try {
    std::lock_guard lock(model_->mutex_);
    std::vector<std::uint32_t> rows;
    std::vector<float> part;
    for (std::size_t start = 0; start < tokens.size(); start += chunk) {
      const std::size_t count = std::min(chunk, tokens.size() - start);
      rows.resize(count);
      for (std::size_t i = 0; i < count; ++i) {
        rows[i] = static_cast<std::uint32_t>(i);
      }
      const auto first = static_cast<std::uint32_t>(tokens_.size());
      tokens_.insert(tokens_.end(), tokens.begin() + start,
                     tokens.begin() + start + count);
      executor.Forward(*cache_, tokens.subspan(start, count), first, rows);
      executor.CommitHidden(*cache_, static_cast<std::uint32_t>(count - 1));
      executor.CopyLogits(count, &part);
      logits->insert(logits->end(), part.begin(), part.end());
    }
  } catch (const std::exception& e) {
    Reset();
    return Fail(error_msg, e.what());
  }
  logits_.assign(logits->end() - static_cast<std::ptrdiff_t>(vocab),
                 logits->end());
  valid_ = true;
  return true;
}

namespace {

// Draft-length control: a chain continues while the drafter's estimate that
// every draft so far is accepted (the product of its confidences) stays at
// or above a floor; the draft that falls below it is not verified. Greedy
// chains use the drafter's top-1 share of its top-64 candidates, sampled
// chains the probability of the sampled proposal, capped at four drafts
// because temperature-1 acceptance does not pay for longer ones. Fitted on
// instrumented prose, repetitive and sampled runs from 0 to 64K (see
// docs/models/gemma-4-31b/EXPERIMENTS.md).
constexpr float kGreedyChainFloor = 0.5F;
constexpr float kSampledChainFloor = 0.3F;
constexpr std::uint32_t kSampledDraftCap = 4;

/// The top candidate's softmax share among the top-64 logits.
float TopShare(const qwen38_flash_next::MtpCandidateLogits& candidates) {
  float top = candidates.logits[0];
  for (std::size_t i = 1; i < candidates.size; ++i) {
    top = std::max(top, candidates.logits[i]);
  }
  double total = 0.0;
  for (std::size_t i = 0; i < candidates.size; ++i) {
    total += std::exp(static_cast<double>(candidates.logits[i] - top));
  }
  return static_cast<float>(1.0 / total);
}

/// The argmax candidate, lowest token id on ties.
std::int32_t TopToken(const qwen38_flash_next::MtpCandidateLogits& candidates) {
  std::size_t best = 0;
  for (std::size_t i = 1; i < candidates.size; ++i) {
    if (candidates.logits[i] > candidates.logits[best] ||
        (candidates.logits[i] == candidates.logits[best] &&
         candidates.ids[i] < candidates.ids[best])) {
      best = i;
    }
  }
  return static_cast<std::int32_t>(candidates.ids[best]);
}

}  // namespace

bool Session::DecodeStep(std::size_t max_tokens,
                         sampling::SamplerState& sampler, DecodeResult* result,
                         std::string* error_msg, bool stop_at_eos) {
  result->tokens.clear();
  result->stop = false;
  if (max_tokens == 0) {
    return true;
  }
  const auto emit = [&](TokenId token) {
    sampler.Accept(static_cast<sampling::TokenId>(token));
    result->tokens.push_back(token);
    if (stop_at_eos && model_->IsStopToken(token)) {
      result->stop = true;
    }
  };
  if (!pending_) {
    if (!valid_) {
      return Fail(error_msg, "session has no logits to decode from");
    }
    emit(static_cast<TokenId>(sampler.Sample(logits_)));
    pending_ = result->tokens.back();
    if (result->stop || result->tokens.size() >= max_tokens) {
      return true;
    }
  }
  const TokenId pending = *pending_;
  const auto position = static_cast<std::uint32_t>(tokens_.size());
  if (position + 1 >= cache_->max_context) {
    return Fail(error_msg, "session context is full");
  }
  const std::size_t budget = max_tokens - result->tokens.size();
  std::uint32_t steps = 0;
  if (model_->HasMtp() && budget > 1) {
    steps = static_cast<std::uint32_t>(
        std::min<std::size_t>({model_->DraftTokens(), budget - 1,
                               cache_->max_context - position - 1}));
  }
  auto& executor = *model_->executor_;
  std::vector<float> logits;
  std::vector<TokenId> rows{pending};
  // With random sampling the drafter samples its proposals too, and each is
  // verified by p/q rejection with residual correction; greedy decoding
  // keeps argmax drafts accepted when the target's own choice agrees.
  const bool sampled = steps > 0 && sampler.config().uses_random_sampling();
  if (sampled) {
    steps = std::min(steps, kSampledDraftCap);
  }
  std::vector<qwen38_flash_next::MtpProposal> proposals;
  try {
    std::lock_guard lock(model_->mutex_);
    if (steps > 0) {
      std::vector<std::int32_t> drafts;
      float chain = 1.0F;
      std::uint32_t proposed = 0;
      if (sampled) {
        // A cycle-local proposal stream; target draws keep the sampler's.
        std::uint64_t draft_rng =
            sampling::NextRandom(sampler.mutable_rng_state());
        sampling::SamplerState draft_sampler = sampler;
        executor.DraftChain(
            *cache_, pending, position, steps, &drafts,
            [&](const qwen38_flash_next::MtpCandidateLogits& candidates)
                -> std::optional<std::int32_t> {
              auto proposal = qwen38_flash_next::SampleMtpProposal(
                  candidates, draft_sampler, &draft_rng);
              chain *= proposal.probability;
              if (!proposals.empty() && chain < kSampledChainFloor) {
                return std::nullopt;
              }
              draft_sampler.Accept(proposal.token);
              proposals.push_back(proposal);
              return static_cast<std::int32_t>(proposal.token);
            });
      } else {
        executor.DraftChain(
            *cache_, pending, position, steps, &drafts,
            [&](const qwen38_flash_next::MtpCandidateLogits& candidates)
                -> std::optional<std::int32_t> {
              chain *= TopShare(candidates);
              if (proposed++ > 0 && chain < kGreedyChainFloor) {
                return std::nullopt;
              }
              return TopToken(candidates);
            });
      }
      rows.insert(rows.end(), drafts.begin(), drafts.end());
    }
    std::vector<std::uint32_t> logit_rows(rows.size());
    for (std::uint32_t i = 0; i < logit_rows.size(); ++i) {
      logit_rows[i] = i;
    }
    executor.Forward(*cache_, rows, position, logit_rows);
    executor.CopyLogits(rows.size(), &logits);
  } catch (const std::exception& e) {
    Reset();
    return Fail(error_msg, e.what());
  }
  // Row i predicts the token after rows[i]; drafts stay while the target's
  // own sample agrees with them.
  const std::size_t vocab = model_->VocabSize();
  std::size_t row = 0;
  std::uint64_t accepted = 0;
  for (;; ++row) {
    const std::span<const float> row_logits(logits.data() + row * vocab, vocab);
    bool agrees = false;
    if (sampled && row + 1 < rows.size()) {
      const auto verified = qwen38_flash_next::VerifyMtpProposal(
          row_logits, proposals[row], sampler);
      emit(static_cast<TokenId>(verified.token));
      agrees = verified.accepted;
    } else {
      emit(static_cast<TokenId>(sampler.Sample(row_logits)));
      agrees = row + 1 < rows.size() && result->tokens.back() == rows[row + 1];
    }
    if (!agrees || result->stop || result->tokens.size() >= max_tokens) {
      break;
    }
    ++accepted;
  }
  // Rows [0, row] are committed; the last emitted token becomes pending.
  tokens_.insert(tokens_.end(), rows.begin(), rows.begin() + row + 1);
  executor.CommitHidden(*cache_, static_cast<std::uint32_t>(row));
  logits_.assign(logits.begin() + row * vocab,
                 logits.begin() + (row + 1) * vocab);
  valid_ = true;
  pending_ = result->tokens.back();
  stats_.cycles += 1;
  stats_.drafted += rows.size() - 1;
  stats_.accepted += accepted;
  return true;
}

bool SessionSnapshot::CopyTo(std::span<std::uint8_t> destination) const {
  if (destination.size() != data_.size()) {
    return false;
  }
  std::memcpy(destination.data(), data_.data(), data_.size());
  return true;
}

std::uint64_t Session::SnapshotBytes() const {
  const Config& c = model_->config();
  const auto n = static_cast<std::uint32_t>(tokens_.size());
  std::uint64_t bytes = sizeof(SnapshotHeader) + std::uint64_t{n} * 4;
  const auto& key_widths = model_->executor_->key_widths();
  for (std::uint32_t l = 0; l < c.num_layers; ++l) {
    bytes += std::uint64_t{n - FirstLiveRow(c, l, n)} *
             (key_widths[l] + c.KvDim(l)) * 2;
  }
  return bytes + std::uint64_t{c.hidden_size} * 4 +
         std::uint64_t{model_->VocabSize()} * 4;
}

std::unique_ptr<SessionSnapshot> Session::SaveSnapshot(
    std::string* error_msg) const {
  if (!valid_ || tokens_.empty()) {
    Fail(error_msg, "only an evaluated session can be saved");
    return nullptr;
  }
  const Config& c = model_->config();
  const auto n = static_cast<std::uint32_t>(tokens_.size());
  auto snapshot = std::unique_ptr<SessionSnapshot>(new SessionSnapshot());
  auto& data = snapshot->data_;
  data.resize(SnapshotBytes());
  const SnapshotHeader header{
      kSnapshotMagic,  kSnapshotPayloadVersion, n,
      c.hidden_size,   model_->VocabSize(),     c.num_layers,
      c.sliding_window};
  std::uint8_t* at = data.data();
  std::memcpy(at, &header, sizeof(header));
  at += sizeof(header);
  std::memcpy(at, tokens_.data(), tokens_.size() * 4);
  at += tokens_.size() * 4;
  const hipStream_t stream = model_->executor_->stream();
  std::lock_guard lock(model_->mutex_);
  const auto copy_rows = [&](const std::uint16_t* cache, std::uint32_t layer,
                             std::uint32_t first, std::uint32_t width) {
    const std::size_t row = std::size_t{width} * 2;
    const bool ring = c.IsSliding(layer);
    for (std::uint32_t p = first; p < n;) {
      const std::uint32_t slot = ring ? p % cache_->ring : p;
      const std::uint32_t run =
          ring ? std::min(n - p, cache_->ring - slot) : n - p;
      (void)hipMemcpyAsync(at,
                           reinterpret_cast<const std::uint8_t*>(cache) +
                               std::size_t{slot} * row,
                           std::size_t{run} * row, hipMemcpyDeviceToHost,
                           stream);
      at += std::size_t{run} * row;
      p += run;
    }
  };
  for (std::uint32_t l = 0; l < c.num_layers; ++l) {
    const std::uint32_t first = FirstLiveRow(c, l, n);
    copy_rows(cache_->k[l], l, first, model_->executor_->key_widths()[l]);
    copy_rows(cache_->v[l], l, first, c.KvDim(l));
  }
  (void)hipMemcpyAsync(at, cache_->hidden, std::size_t{c.hidden_size} * 4,
                       hipMemcpyDeviceToHost, stream);
  at += std::size_t{c.hidden_size} * 4;
  if (hipStreamSynchronize(stream) != hipSuccess) {
    Fail(error_msg, "snapshot copy failed");
    return nullptr;
  }
  std::memcpy(at, logits_.data(), logits_.size() * 4);
  return snapshot;
}

bool Session::RestoreSnapshot(const SessionSnapshot& snapshot,
                              std::string* error_msg) {
  return RestoreSnapshot(snapshot.bytes(), error_msg);
}

bool Session::RestoreSnapshot(std::span<const std::uint8_t> payload,
                              std::string* error_msg) {
  Reset();
  const Config& c = model_->config();
  SnapshotHeader header{};
  if (payload.size() < sizeof(header)) {
    return Fail(error_msg, "snapshot is truncated");
  }
  std::memcpy(&header, payload.data(), sizeof(header));
  if (header.magic != kSnapshotMagic ||
      header.version != kSnapshotPayloadVersion ||
      header.hidden != c.hidden_size || header.vocab != model_->VocabSize() ||
      header.layers != c.num_layers || header.window != c.sliding_window) {
    return Fail(error_msg, "snapshot belongs to another model or version");
  }
  const std::uint32_t n = header.position;
  if (n == 0 || n > cache_->max_context) {
    return Fail(error_msg, "snapshot position exceeds the session context");
  }
  tokens_.resize(n);
  if (payload.size() != SnapshotBytes()) {
    tokens_.clear();
    return Fail(error_msg, "snapshot size does not match its header");
  }
  const std::uint8_t* at = payload.data() + sizeof(header);
  std::memcpy(tokens_.data(), at, std::size_t{n} * 4);
  at += std::size_t{n} * 4;
  const hipStream_t stream = model_->executor_->stream();
  std::lock_guard lock(model_->mutex_);
  const auto copy_rows = [&](std::uint16_t* cache, std::uint32_t layer,
                             std::uint32_t first, std::uint32_t width) {
    const std::size_t row = std::size_t{width} * 2;
    const bool ring = c.IsSliding(layer);
    for (std::uint32_t p = first; p < n;) {
      const std::uint32_t slot = ring ? p % cache_->ring : p;
      const std::uint32_t run =
          ring ? std::min(n - p, cache_->ring - slot) : n - p;
      (void)hipMemcpyAsync(
          reinterpret_cast<std::uint8_t*>(cache) + std::size_t{slot} * row, at,
          std::size_t{run} * row, hipMemcpyHostToDevice, stream);
      at += std::size_t{run} * row;
      p += run;
    }
  };
  for (std::uint32_t l = 0; l < c.num_layers; ++l) {
    const std::uint32_t first = FirstLiveRow(c, l, n);
    copy_rows(cache_->k[l], l, first, model_->executor_->key_widths()[l]);
    copy_rows(cache_->v[l], l, first, c.KvDim(l));
  }
  (void)hipMemcpyAsync(cache_->hidden, at, std::size_t{c.hidden_size} * 4,
                       hipMemcpyHostToDevice, stream);
  at += std::size_t{c.hidden_size} * 4;
  if (hipStreamSynchronize(stream) != hipSuccess) {
    Reset();
    return Fail(error_msg, "snapshot restore failed");
  }
  logits_.resize(model_->VocabSize());
  std::memcpy(logits_.data(), at, logits_.size() * 4);
  valid_ = true;
  return true;
}

}  // namespace gufo::models::gemma4
