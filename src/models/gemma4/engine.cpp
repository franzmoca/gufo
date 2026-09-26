#include "src/models/gemma4/engine.hpp"

#include <algorithm>
#include <exception>
#include <utility>

#include "src/core/gguf_reader.hpp"
#include "src/models/gemma4/chat_template.hpp"
#include "src/models/gemma4/kernels/rocm/device_model.hpp"
#include "src/models/gemma4/kernels/rocm/executor.hpp"
#include "src/models/gemma4/weights.hpp"

namespace gufo::models::gemma4 {
namespace {

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
  return rocm::Executor::CacheBytes(config(), context, executor_->ring());
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
  try {
    std::lock_guard lock(model_->mutex_);
    if (steps > 0) {
      std::vector<std::int32_t> drafts;
      executor.DraftChain(*cache_, pending, position, steps, &drafts);
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
    emit(static_cast<TokenId>(sampler.Sample(row_logits)));
    const bool agrees =
        row + 1 < rows.size() && result->tokens.back() == rows[row + 1];
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
  stats_.drafted += steps;
  stats_.accepted += accepted;
  return true;
}

}  // namespace gufo::models::gemma4
