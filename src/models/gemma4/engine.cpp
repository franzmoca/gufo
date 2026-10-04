#include "src/models/gemma4/engine.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <exception>
#include <optional>
#include <stdexcept>
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
/// Snapshot image record: u32 offset, u32 rows, 32-byte identity.
constexpr std::size_t kImageRecordBytes = 8 + 32;

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

void Model::SetTapSink(LayerTapSink sink) {
  std::lock_guard lock(mutex_);
  if (!sink) {
    executor_->SetTapSink({});
    return;
  }
  executor_->SetTapSink(
      [sink = std::move(sink)](std::uint32_t layer, const float* rows,
                               std::uint32_t count, hipStream_t stream) {
        if (hipStreamSynchronize(stream) != hipSuccess) {
          throw std::runtime_error("gemma4 layer tap synchronization failed");
        }
        sink(layer, rows, count);
      });
}
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
    if (options.min_draft_tokens == 0 ||
        options.min_draft_tokens > options.draft_tokens) {
      Fail(error_msg, "gemma4 minimum draft tokens must be 1..draft tokens");
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
             VocabSize(), device_->max_cols(), device_->max_half_cols(),
             options_.prefill_chunk, options_.max_logit_rows,
             options_.max_context);
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
  images_.clear();
  lookup_.Clear();
  logits_.clear();
  valid_ = false;
  // A restored snapshot starts a request without Sync.
  if (model_->options_.draft_calibration == DraftCalibrationScope::kRequest) {
    for (auto& calibration : calibration_) {
      calibration.Reset();
    }
  }
}

bool Session::Extend(std::size_t begin, std::string* error_msg,
                     const ImageEmbeddings& embed) {
  auto& executor = *model_->executor_;
  const std::size_t chunk = executor.max_rows();
  valid_ = false;
  try {
    std::lock_guard lock(model_->mutex_);
    std::vector<rocm::ImageRows> rows;
    std::size_t count = 0;
    for (std::size_t start = begin; start < tokens_.size(); start += count) {
      std::size_t end = std::min(start + chunk, tokens_.size());
      rows.clear();
      for (std::size_t i = 0; i < images_.size(); ++i) {
        const ImageSpan& image = images_[i];
        const std::size_t image_end = std::size_t{image.offset} + image.rows;
        if (image_end <= start || image.offset >= end) {
          continue;
        }
        if (image.offset < start) {
          throw std::logic_error("gemma4 prefill resumed inside an image");
        }
        if (image_end > end) {
          // Stop before an image that does not fit; one that starts the
          // chunk fits because images are at most a chunk long.
          end = image.offset > start ? image.offset : image_end;
          if (image.offset > start) {
            break;
          }
        }
        const float* embedding = embed ? embed(i) : nullptr;
        if (embedding == nullptr) {
          throw std::invalid_argument("gemma4 image has no embeddings");
        }
        rows.push_back({static_cast<std::uint32_t>(image.offset - start),
                        image.rows, embedding});
      }
      count = end - start;
      const bool last = end == tokens_.size();
      const auto last_row = static_cast<std::uint32_t>(count - 1);
      executor.Forward(*cache_, std::span(tokens_).subspan(start, count),
                       static_cast<std::uint32_t>(start),
                       last ? std::span<const std::uint32_t>(&last_row, 1)
                            : std::span<const std::uint32_t>{},
                       rows);
    }
    executor.CommitHidden(*cache_, static_cast<std::uint32_t>(count - 1));
    executor.CopyLogits(1, &logits_);
  } catch (const std::exception& e) {
    tokens_.resize(begin);
    std::erase_if(images_, [&](const ImageSpan& image) {
      return std::size_t{image.offset} + image.rows > begin;
    });
    lookup_.Clear();
    return Fail(error_msg, e.what());
  }
  valid_ = true;
  return true;
}

bool Session::Sync(std::span<const TokenId> prompt, std::string* error_msg) {
  return Sync(prompt, {}, {}, error_msg);
}

bool Session::Sync(std::span<const TokenId> prompt,
                   std::span<const ImageSpan> images,
                   const ImageEmbeddings& embed, std::string* error_msg) {
  if (prompt.empty()) {
    Reset();
    return Fail(error_msg, "prompt is empty");
  }
  if (model_->options_.draft_calibration == DraftCalibrationScope::kRequest) {
    for (auto& calibration : calibration_) {
      calibration.Reset();
    }
  }
  if (prompt.size() > cache_->max_context) {
    return Fail(error_msg, "prompt exceeds the session context");
  }
  std::size_t previous_end = 0;
  for (const ImageSpan& image : images) {
    // A text token follows every image, so the frontier is never inside one.
    if (image.rows == 0 || image.offset < previous_end ||
        std::size_t{image.offset} + image.rows >= prompt.size() ||
        image.rows > model_->executor_->max_rows()) {
      return Fail(error_msg, "invalid image placement");
    }
    previous_end = std::size_t{image.offset} + image.rows;
  }
  pending_.reset();
  std::size_t common = 0;
  const std::size_t limit = std::min(prompt.size(), tokens_.size());
  while (common < limit && prompt[common] == tokens_[common]) {
    ++common;
  }
  // Equal tokens do not mean equal images: stop at the first image that
  // differs, and never resume inside an image of the new prompt.
  for (std::size_t i = 0; i < std::max(images.size(), images_.size()); ++i) {
    if (i < images.size() && i < images_.size() && images[i] == images_[i]) {
      continue;
    }
    std::size_t differs = common;
    if (i < images.size()) {
      differs = std::min<std::size_t>(differs, images[i].offset);
    }
    if (i < images_.size()) {
      differs = std::min<std::size_t>(differs, images_[i].offset);
    }
    common = differs;
    break;
  }
  if (common == prompt.size() && common == tokens_.size() && valid_) {
    return true;
  }
  // The last prompt token is re-evaluated to produce its logits.
  common = std::min(common, prompt.size() - 1);
  for (const ImageSpan& image : images) {
    if (common > image.offset && common < image.offset + image.rows) {
      common = image.offset;
    }
  }
  // Rewinding needs the sliding window before `common` still in the ring.
  const Config& c = model_->config();
  if (tokens_.size() - common + c.sliding_window > cache_->ring) {
    common = 0;
  }
  tokens_.assign(prompt.begin(), prompt.end());
  images_.assign(images.begin(), images.end());
  lookup_.Clear();
  return Extend(common, error_msg, embed);
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

// The confidence policy: a chain continues while the drafter's estimate that
// every draft so far is accepted (the product of its confidences) stays at
// or above a floor; the draft that falls below it is not verified. Greedy
// chains use the drafter's top-1 share of its top-64 candidates, sampled
// chains the probability of the sampled proposal, capped at four drafts.
// Fitted on the UD-Q4_K_XL drafter (docs/models/gemma-4-31b/EXPERIMENTS.md);
// the calibrated policy (draft_policy.hpp) replaces these constants.
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

/// One cycle's drafting: the length policy, sampled proposals and
/// prompt-lookup copies, asked for one proposal per drafter step.
struct Session::CycleDraft {
  CycleDraft(Session& owner, Cycle& owner_cycle, std::uint32_t chain_steps,
             std::uint32_t slots, const DraftBatch& others,
             DraftShare* batch_share)
      : session(owner),
        cycle(owner_cycle),
        policy(owner.model_->options_.draft_policy),
        min_drafts(owner.model_->options_.min_draft_tokens),
        steps(chain_steps),
        max_drafts(slots),
        alone(Alone(owner, owner_cycle)),
        costs(SharedCosts(owner, owner_cycle, alone ? nullptr : batch_share)),
        calibrated(owner.Calibration(owner_cycle.sampled), costs, min_drafts,
                   chain_steps, alone ? DraftBatch{} : others),
        share(batch_share),
        context{*owner.pending_} {
    session.lookup_.Extend(session.tokens_);
    if (cycle.sampled) {
      // A cycle-local proposal stream; target draws keep the sampler's.
      draft_rng = sampling::NextRandom(cycle.sampler->mutable_rng_state());
      // Drafts are proposals: only the target's verification is
      // constrained (a grammar may exclude every drafter candidate).
      draft_sampler = cycle.sampler->WithoutConstraint();
    }
  }

  /// Whether the cycle prices its drafts as if alone. The tokens a seed
  /// draws depend on the drafts proposed, so with per-request calibration
  /// (exact seeded replay) a sampled cycle ignores the sessions batched
  /// beside it. Greedy output is exact either way.
  static bool Alone(const Session& owner, const Cycle& c) {
    return c.sampled && owner.model_->options_.draft_calibration ==
                            DraftCalibrationScope::kRequest;
  }

  /// Drafter steps shared by the sessions of a batch cost each a share.
  static DraftCosts SharedCosts(const Session& owner, const Cycle& c,
                                const DraftShare* batch_share) {
    DraftCosts costs =
        DraftCostsAt(c.position, owner.model_->config().HasExperts());
    if (batch_share != nullptr && batch_share->sessions > 1) {
      for (float& ms : costs.draft) {
        ms /= static_cast<float>(batch_share->sessions);
      }
    }
    return costs;
  }

  [[nodiscard]] bool Worthwhile() const {
    return policy != DraftPolicy::kCalibrated || calibrated.FirstDraftCanPay();
  }

  /// Fills the remaining slots with the tokens that followed an earlier
  /// occurrence of the context (at least 12 tokens); they end the chain.
  bool Copy() {
    const auto match = session.lookup_.Find(session.tokens_, context);
    const std::size_t room = max_drafts - (context.size() - 1);
    if (match.length == 0 || room == 0) {
      return false;
    }
    const std::size_t count =
        std::min(room, session.tokens_.size() - match.start);
    copies.assign(
        session.tokens_.begin() + static_cast<std::ptrdiff_t>(match.start),
        session.tokens_.begin() +
            static_cast<std::ptrdiff_t>(match.start + count));
    return count != 0;
  }

  /// Whether the draft just proposed ends the chain unverified. `kept`
  /// drafts precede it; `confidence` feeds the confidence policy's chain.
  bool Stop(std::size_t kept, float confidence,
            const qwen38_flash_next::MtpCandidateLogits& c) {
    switch (policy) {
      case DraftPolicy::kCalibrated: {
        const float signal = DraftSignal(c);
        const float before = calibrated.Expected();
        if (share != nullptr && !alone) {
          // The other sessions as they stand now.
          const auto own_rows = static_cast<std::uint32_t>(kept + 1);
          calibrated.SetOthers({.rows = share->rows - own_rows,
                                .expected = share->expected - before});
        }
        if (!calibrated.Include(signal)) {
          return true;
        }
        if (share != nullptr) {
          share->rows += 1;
          share->expected += calibrated.Expected() - before;
        }
        cycle.signals.push_back(signal);
        return false;
      }
      case DraftPolicy::kConfidence:
        chain *= confidence;
        return kept >= min_drafts &&
               chain < (cycle.sampled ? kSampledChainFloor : kGreedyChainFloor);
      case DraftPolicy::kFixed:
        break;
    }
    return false;
  }

  rocm::DraftProposal Propose(const qwen38_flash_next::MtpCandidateLogits& c) {
    if (cycle.sampled) {
      auto proposal =
          qwen38_flash_next::SampleMtpProposal(c, draft_sampler, &draft_rng);
      if (Stop(cycle.proposals.size(), proposal.probability, c)) {
        (void)Copy();
        return rocm::DraftProposal{};
      }
      draft_sampler.Accept(proposal.token);
      cycle.proposals.push_back(proposal);
      context.push_back(static_cast<std::int32_t>(proposal.token));
      return rocm::DraftProposal{
          .token = static_cast<std::int32_t>(proposal.token), .last = Copy()};
    }
    if (Stop(context.size() - 1, TopShare(c), c)) {
      (void)Copy();
      return rocm::DraftProposal{};
    }
    const std::int32_t token = TopToken(c);
    context.push_back(token);
    return rocm::DraftProposal{.token = token, .last = Copy()};
  }

  Session& session;
  Cycle& cycle;
  DraftPolicy policy;
  std::uint32_t min_drafts;
  std::uint32_t steps;  ///< drafter steps the chain may run
  std::uint32_t max_drafts;
  bool alone;
  DraftCosts costs;
  CalibratedChain calibrated;
  DraftShare* share;
  float chain{1.0F};
  /// The context the next token continues: pending plus kept drafts.
  std::vector<std::int32_t> context;
  std::vector<std::int32_t> copies;
  std::vector<std::int32_t> drafts;
  sampling::SamplerState draft_sampler{sampling::SamplingConfig{}, {}};
  std::uint64_t draft_rng{0};
};

bool Session::DecodeStep(std::size_t max_tokens,
                         sampling::SamplerState& sampler, DecodeResult* result,
                         std::string* error_msg, bool stop_at_eos) {
  Cycle cycle{.max_tokens = max_tokens,
              .sampler = &sampler,
              .result = result,
              .stop_at_eos = stop_at_eos};
  std::lock_guard lock(model_->mutex_);
  if (!BeginCycle(cycle, model_->DraftTokens(), error_msg)) {
    return false;
  }
  if (cycle.done) {
    return true;
  }
  auto& executor = *model_->executor_;
  std::vector<float> logits;
  try {
    std::vector<std::uint32_t> logit_rows(cycle.rows.size());
    for (std::uint32_t i = 0; i < logit_rows.size(); ++i) {
      logit_rows[i] = i;
    }
    executor.Forward(*cache_, cycle.rows, cycle.position, logit_rows);
    executor.CopyLogits(cycle.rows.size(), &logits);
  } catch (const std::exception& e) {
    Reset();
    return Fail(error_msg, e.what());
  }
  FinishCycle(cycle, logits, 0);
  return true;
}

DraftCalibration& Session::Calibration(bool sampled) {
  auto& tables =
      model_->options_.draft_calibration == DraftCalibrationScope::kRequest
          ? calibration_
          : model_->calibration_;
  return tables[sampled ? 1 : 0];
}

bool Session::BeginCycle(Cycle& cycle, std::uint32_t draft_limit,
                         std::string* error_msg, const DraftBatch& others,
                         DraftShare* share) {
  auto& sampler = *cycle.sampler;
  auto* result = cycle.result;
  result->tokens.clear();
  result->stop = false;
  cycle.done = true;
  if (cycle.max_tokens == 0) {
    return true;
  }
  if (!pending_) {
    if (!valid_) {
      return Fail(error_msg, "session has no logits to decode from");
    }
    const auto token = static_cast<TokenId>(sampler.Sample(logits_));
    sampler.Accept(static_cast<sampling::TokenId>(token));
    result->tokens.push_back(token);
    result->stop = cycle.stop_at_eos && model_->IsStopToken(token);
    pending_ = token;
    if (result->stop || result->tokens.size() >= cycle.max_tokens) {
      return true;
    }
  }
  const TokenId pending = *pending_;
  const auto position = static_cast<std::uint32_t>(tokens_.size());
  if (position + 1 >= cache_->max_context) {
    return Fail(error_msg, "session context is full");
  }
  cycle.done = false;
  cycle.position = position;
  cycle.rows.assign(1, pending);
  const std::size_t budget = cycle.max_tokens - result->tokens.size();
  std::uint32_t steps = 0;
  if (model_->HasMtp() && budget > 1) {
    steps = static_cast<std::uint32_t>(std::min<std::size_t>(
        {draft_limit, budget - 1, cache_->max_context - position - 1}));
  }
  // With random sampling the drafter samples its proposals too, and each is
  // verified by p/q rejection with residual correction; greedy decoding
  // keeps argmax drafts accepted when the target's own choice agrees.
  cycle.sampled = steps > 0 && sampler.config().uses_random_sampling();
  // Prompt lookup may fill every slot; the policy may stop MTP drafts
  // earlier.
  const std::uint32_t max_drafts = steps;
  const ModelOptions& options = model_->options_;
  const DraftPolicy policy = options.draft_policy;
  if (cycle.sampled && policy == DraftPolicy::kConfidence) {
    steps = std::min(steps, kSampledDraftCap);
  }
  cycle.signals.clear();
  if (steps == 0) {
    return true;
  }
  try {
    cycle.draft = std::make_unique<CycleDraft>(*this, cycle, steps, max_drafts,
                                               others, share);
    if (!cycle.draft->Worthwhile()) {
      // Beside other sessions not even a certain draft would pay: verify
      // the pending token alone without running the drafter.
      cycle.draft.reset();
      return true;
    }
    if (share == nullptr) {
      CycleDraft& draft = *cycle.draft;
      model_->executor_->DraftChain(
          *cache_, pending, position, steps, &draft.drafts,
          [&draft](const qwen38_flash_next::MtpCandidateLogits& c) {
            return draft.Propose(c);
          });
      FinishDraft(cycle);
    }
  } catch (const std::exception& e) {
    cycle.draft.reset();
    Reset();
    return Fail(error_msg, e.what());
  }
  return true;
}

void Session::FinishDraft(Cycle& cycle) {
  CycleDraft& draft = *cycle.draft;
  if (cycle.sampled) {
    // A copied token is a point-mass proposal: accepted with the target's
    // probability, a rejection resampling without it.
    for (const std::int32_t token : draft.copies) {
      qwen38_flash_next::MtpProposal proposal;
      proposal.ids[0] = static_cast<sampling::TokenId>(token);
      proposal.probabilities[0] = 1.0F;
      proposal.size = 1;
      proposal.token = proposal.ids[0];
      proposal.probability = 1.0F;
      cycle.proposals.push_back(proposal);
    }
  }
  cycle.rows.insert(cycle.rows.end(), draft.drafts.begin(), draft.drafts.end());
  cycle.rows.insert(cycle.rows.end(), draft.copies.begin(), draft.copies.end());
  cycle.copied = draft.copies.size();
  if (draft.policy == DraftPolicy::kCalibrated) {
    cycle.expected = draft.calibrated.Expected();
    cycle.draft_ms = draft.calibrated.DraftMs();
  }
  cycle.draft.reset();
}

void Session::FinishCycle(Cycle& cycle, std::span<const float> logits,
                          std::uint32_t first_hidden_row) {
  auto& sampler = *cycle.sampler;
  auto* result = cycle.result;
  const auto& rows = cycle.rows;
  const auto emit = [&](TokenId token) {
    sampler.Accept(static_cast<sampling::TokenId>(token));
    result->tokens.push_back(token);
    if (cycle.stop_at_eos && model_->IsStopToken(token)) {
      result->stop = true;
    }
  };
  // Row i predicts the token after rows[i]; drafts stay while the target's
  // own sample agrees with them.
  const std::size_t vocab = model_->VocabSize();
  std::size_t row = 0;
  std::uint64_t accepted = 0;
  bool agrees = false;
  for (;; ++row) {
    const auto row_logits = logits.subspan(row * vocab, vocab);
    agrees = false;
    if (cycle.sampled && row + 1 < rows.size()) {
      const auto verified = qwen38_flash_next::VerifyMtpProposal(
          row_logits, cycle.proposals[row], sampler);
      emit(static_cast<TokenId>(verified.token));
      agrees = verified.accepted;
    } else {
      emit(static_cast<TokenId>(sampler.Sample(row_logits)));
      agrees = row + 1 < rows.size() && result->tokens.back() == rows[row + 1];
    }
    if (!agrees || result->stop || result->tokens.size() >= cycle.max_tokens) {
      break;
    }
    ++accepted;
  }
  // Rows [0, row] are committed; the last emitted token becomes pending.
  tokens_.insert(tokens_.end(), rows.begin(), rows.begin() + row + 1);
  model_->executor_->CommitHidden(
      *cache_, first_hidden_row + static_cast<std::uint32_t>(row));
  const auto kept = logits.subspan(row * vocab, vocab);
  logits_.assign(kept.begin(), kept.end());
  valid_ = true;
  pending_ = result->tokens.back();
  stats_.cycles += 1;
  stats_.verified += rows.size() > 1 ? 1 : 0;
  stats_.drafted += rows.size() - 1;
  stats_.accepted += accepted;
  // Copies trail the MTP drafts.
  const std::size_t mtp = rows.size() - 1 - cycle.copied;
  stats_.copied += cycle.copied;
  // The calibrated policy learns from every draft verification judged: the
  // accepted ones and the one that ended the chain (copies trail the MTP
  // drafts and carry no signal).
  if (!cycle.signals.empty()) {
    DraftCalibration& calibration = Calibration(cycle.sampled);
    const std::size_t judged =
        std::min(row + (row + 1 < rows.size() ? 1 : 0), cycle.signals.size());
    for (std::size_t j = 0; j < judged; ++j) {
      calibration.Observe(cycle.signals[j], j < row || agrees);
    }
  }
  stats_.copied_accepted += accepted > mtp ? accepted - mtp : 0;
}

bool Session::DecodeBatch(std::span<BatchDecode> decodes) {
  if (decodes.empty()) {
    return true;
  }
  Model& model = *decodes.front().session->model_;
  std::lock_guard lock(model.mutex_);
  // One verification forward keeps its single-session arithmetic up to
  // kSplitRows rows, shared evenly.
  const auto draft_limit = static_cast<std::uint32_t>(std::min<std::size_t>(
      model.DraftTokens(),
      std::max<std::size_t>(1, rocm::kSplitRows / decodes.size()) - 1));
  std::vector<Cycle> cycles(decodes.size());
  std::vector<rocm::Executor::Segment> segments;
  std::vector<std::int32_t> tokens;
  std::vector<std::size_t> active;
  bool ok = true;
  // The sessions draft together, one drafter forward per step; the
  // calibrated policy prices each session's drafts against the whole
  // forward as it stands (every session starts with its pending row) and
  // charges each a share of the drafter steps.
  const auto sessions = static_cast<std::uint32_t>(decodes.size());
  DraftShare share{.sessions = sessions,
                   .rows = sessions,
                   .expected = static_cast<float>(sessions)};
  const DraftBatch others{.rows = sessions - 1,
                          .expected = static_cast<float>(sessions - 1)};
  std::vector<rocm::Executor::DraftJob> jobs;
  std::vector<std::size_t> drafting;
  for (std::size_t i = 0; i < decodes.size(); ++i) {
    auto& d = decodes[i];
    if (d.session->model_.get() != &model) {
      throw std::invalid_argument("gemma4 batch mixes models");
    }
    cycles[i] = Cycle{.max_tokens = d.max_tokens,
                      .sampler = d.sampler,
                      .result = d.result,
                      .stop_at_eos = d.stop_at_eos};
    if (!d.session->BeginCycle(cycles[i], draft_limit, &d.error, others,
                               sessions > 1 ? &share : nullptr)) {
      ok = false;
      continue;
    }
    if (cycles[i].draft) {
      CycleDraft& draft = *cycles[i].draft;
      jobs.push_back({d.session->cache_.get(), cycles[i].rows.front(),
                      cycles[i].position, draft.steps,
                      [&draft](const qwen38_flash_next::MtpCandidateLogits& c) {
                        return draft.Propose(c);
                      },
                      &draft.drafts});
      drafting.push_back(i);
    }
  }
  if (!jobs.empty()) {
    try {
      model.executor_->DraftChains(jobs);
    } catch (const std::exception& e) {
      for (const std::size_t i : drafting) {
        cycles[i].draft.reset();
        decodes[i].session->Reset();
        decodes[i].error = e.what();
        cycles[i].done = true;
      }
      ok = false;
    }
    for (const std::size_t i : drafting) {
      if (cycles[i].draft) {
        decodes[i].session->FinishDraft(cycles[i]);
      }
    }
  }
  for (std::size_t i = 0; i < decodes.size(); ++i) {
    if (cycles[i].done || !decodes[i].error.empty()) {
      continue;
    }
    segments.push_back({decodes[i].session->cache_.get(), cycles[i].position,
                        static_cast<std::uint32_t>(cycles[i].rows.size())});
    tokens.insert(tokens.end(), cycles[i].rows.begin(), cycles[i].rows.end());
    active.push_back(i);
  }
  if (active.empty()) {
    return ok;
  }
  auto& executor = *model.executor_;
  std::vector<float> logits;
  try {
    std::vector<std::uint32_t> logit_rows(tokens.size());
    for (std::uint32_t i = 0; i < logit_rows.size(); ++i) {
      logit_rows[i] = i;
    }
    executor.Forward(segments, tokens, logit_rows);
    executor.CopyLogits(tokens.size(), &logits);
  } catch (const std::exception& e) {
    for (const std::size_t i : active) {
      decodes[i].session->Reset();
      decodes[i].error = e.what();
    }
    return false;
  }
  const std::size_t vocab = model.VocabSize();
  std::uint32_t row0 = 0;
  for (const std::size_t i : active) {
    const std::size_t count = cycles[i].rows.size();
    decodes[i].session->FinishCycle(
        cycles[i],
        std::span<const float>(logits).subspan(row0 * vocab, count * vocab),
        row0);
    row0 += static_cast<std::uint32_t>(count);
  }
  return ok;
}

bool Session::EvaluateBatch(std::span<Session* const> sessions,
                            std::span<const TokenId> tokens,
                            std::string* error_msg) {
  if (sessions.size() != tokens.size()) {
    return Fail(error_msg, "gemma4 batch needs one token per session");
  }
  if (sessions.empty()) {
    return true;
  }
  if (sessions.size() > rocm::kSplitRows) {
    return Fail(error_msg, "gemma4 batch is wider than one decode forward");
  }
  Model& model = *sessions.front()->model_;
  std::lock_guard lock(model.mutex_);
  std::vector<rocm::Executor::Segment> segments;
  for (std::size_t i = 0; i < sessions.size(); ++i) {
    Session& s = *sessions[i];
    if (s.model_.get() != &model) {
      throw std::invalid_argument("gemma4 batch mixes models");
    }
    if (s.tokens_.size() >= s.cache_->max_context) {
      return Fail(error_msg, "session context is full");
    }
    segments.push_back(
        {s.cache_.get(), static_cast<std::uint32_t>(s.tokens_.size()), 1});
  }
  std::vector<std::int32_t> rows(tokens.begin(), tokens.end());
  std::vector<std::uint32_t> logit_rows(rows.size());
  for (std::uint32_t i = 0; i < logit_rows.size(); ++i) {
    logit_rows[i] = i;
  }
  auto& executor = *model.executor_;
  std::vector<float> logits;
  try {
    executor.Forward(segments, rows, logit_rows);
    for (std::uint32_t i = 0; i < sessions.size(); ++i) {
      executor.CommitHidden(*sessions[i]->cache_, i);
    }
    executor.CopyLogits(rows.size(), &logits);
  } catch (const std::exception& e) {
    for (Session* s : sessions) {
      s->Reset();
    }
    return Fail(error_msg, e.what());
  }
  const std::size_t vocab = model.VocabSize();
  for (std::size_t i = 0; i < sessions.size(); ++i) {
    Session& s = *sessions[i];
    s.pending_.reset();
    s.tokens_.push_back(tokens[i]);
    s.logits_.assign(
        logits.begin() + static_cast<std::ptrdiff_t>(i * vocab),
        logits.begin() + static_cast<std::ptrdiff_t>((i + 1) * vocab));
    s.valid_ = true;
  }
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
  std::uint64_t bytes = sizeof(SnapshotHeader) + std::uint64_t{n} * 4 + 4 +
                        images_.size() * kImageRecordBytes;
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
  const auto image_count = static_cast<std::uint32_t>(images_.size());
  std::memcpy(at, &image_count, 4);
  at += 4;
  for (const ImageSpan& image : images_) {
    std::memcpy(at, &image.offset, 4);
    std::memcpy(at + 4, &image.rows, 4);
    std::memcpy(at + 8, image.identity.data(), image.identity.size());
    at += kImageRecordBytes;
  }
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
  lookup_.Clear();
  const std::size_t images_at = sizeof(header) + std::size_t{n} * 4;
  std::uint32_t image_count = 0;
  if (payload.size() >= images_at + 4) {
    std::memcpy(&image_count, payload.data() + images_at, 4);
  }
  if (payload.size() < images_at + 4 ||
      image_count > (payload.size() - images_at - 4) / kImageRecordBytes) {
    Reset();
    return Fail(error_msg, "snapshot size does not match its header");
  }
  images_.resize(image_count);
  if (payload.size() != SnapshotBytes()) {
    Reset();
    return Fail(error_msg, "snapshot size does not match its header");
  }
  const std::uint8_t* at = payload.data() + sizeof(header);
  std::memcpy(tokens_.data(), at, std::size_t{n} * 4);
  at += std::size_t{n} * 4 + 4;
  std::size_t previous_end = 0;
  for (ImageSpan& image : images_) {
    std::memcpy(&image.offset, at, 4);
    std::memcpy(&image.rows, at + 4, 4);
    std::memcpy(image.identity.data(), at + 8, image.identity.size());
    at += kImageRecordBytes;
    if (image.rows == 0 || image.offset < previous_end ||
        std::size_t{image.offset} + image.rows >= n) {
      Reset();
      return Fail(error_msg, "snapshot image placement is invalid");
    }
    previous_end = std::size_t{image.offset} + image.rows;
  }
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
