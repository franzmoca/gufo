#ifndef GUFO_MODELS_GEMMA4_ENGINE_HPP_
#define GUFO_MODELS_GEMMA4_ENGINE_HPP_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/sampling.hpp"
#include "src/models/gemma4/config.hpp"
#include "src/models/gemma4/tokenizer.hpp"

namespace gufo::core {
class GgufReader;
}

namespace gufo::models::gemma4 {

struct ModelWeights;
struct DraftWeights;
namespace rocm {
class DeviceModel;
class Executor;
struct KvCache;
}  // namespace rocm

struct ModelOptions {
  /// Largest context any session may use.
  std::uint32_t max_context = 4096;
  /// Tokens per prefill forward; sets the sliding-window ring size.
  std::uint32_t prefill_chunk = 2048;
  /// Rows of one forward that may request logits (qualification only needs
  /// more than one).
  std::uint32_t max_logit_rows = 64;
  /// Optional `gemma4-assistant` MTP drafter; empty leaves speculation off.
  std::string mtp_model_path;
  /// Draft tokens per cycle (at most rocm::kMaxDraftTokens).
  std::uint32_t draft_tokens = 4;
};

class Session;

/// Host copy of one session's context: tokens, the KV rows later tokens can
/// still attend (every global row, the last window-1 sliding rows in logical
/// order), the frontier hidden state and the last logits.
class SessionSnapshot final {
public:
  [[nodiscard]] std::uint64_t SizeBytes() const noexcept {
    return data_.size();
  }
  [[nodiscard]] std::span<const std::uint8_t> bytes() const noexcept {
    return data_;
  }
  [[nodiscard]] bool CopyTo(std::span<std::uint8_t> destination) const;

private:
  std::vector<std::uint8_t> data_;
  friend class Session;
};

/// Gufo-owned API over the Gemma 4 ROCm runtime: one resident model and any
/// number of sessions with independent context state. Forwards of different
/// sessions are serialized on the model's executor.
class Model final : public std::enable_shared_from_this<Model> {
public:
  ~Model();
  Model(const Model&) = delete;
  Model& operator=(const Model&) = delete;

  [[nodiscard]] static std::shared_ptr<Model> Load(
      const std::string& model_path, const ModelOptions& options,
      std::string* error_msg = nullptr);

  [[nodiscard]] std::unique_ptr<Session> CreateSession(
      std::uint32_t max_context, std::string* error_msg = nullptr);

  /// Tokenizes text with special tokens parsed and no BOS; rendered chat
  /// prompts carry their own `<bos>`.
  [[nodiscard]] std::vector<TokenId> Tokenize(std::string_view text) const;
  [[nodiscard]] std::string Decode(std::span<const TokenId> tokens) const;
  [[nodiscard]] std::string TokenText(TokenId token) const;
  [[nodiscard]] bool IsStopToken(TokenId token) const noexcept;
  [[nodiscard]] std::uint32_t VocabSize() const noexcept;
  [[nodiscard]] std::uint32_t MaxContext() const noexcept {
    return options_.max_context;
  }
  [[nodiscard]] const Config& config() const noexcept;
  [[nodiscard]] const Tokenizer& tokenizer() const noexcept {
    return *tokenizer_;
  }
  [[nodiscard]] const core::GgufReader& reader() const noexcept {
    return *reader_;
  }
  [[nodiscard]] std::size_t ResidentBytes() const noexcept;
  [[nodiscard]] std::size_t SessionBytes(std::uint32_t context) const noexcept;
  /// Sliding-window ring slots per session (window + prefill chunk).
  [[nodiscard]] std::size_t SessionRingSlots() const noexcept;
  [[nodiscard]] std::string ModelName() const;
  [[nodiscard]] bool HasMtp() const noexcept {
    return draft_weights_ != nullptr;
  }
  [[nodiscard]] std::uint32_t DraftTokens() const noexcept {
    return options_.draft_tokens;
  }

private:
  Model();

  ModelOptions options_;
  std::shared_ptr<core::GgufReader> reader_;
  std::unique_ptr<ModelWeights> weights_;
  std::shared_ptr<core::GgufReader> draft_reader_;
  std::unique_ptr<DraftWeights> draft_weights_;
  std::unique_ptr<Tokenizer> tokenizer_;
  std::unique_ptr<rocm::DeviceModel> device_;
  std::unique_ptr<rocm::Executor> executor_;
  std::mutex mutex_;

  friend class Session;
};

/// One conversation's context. Sync feeds a prompt reusing the longest
/// common prefix the sliding-window ring can still serve; Evaluate appends
/// one token. The logits of the last evaluated token are kept on the host.
class Session final {
public:
  ~Session();
  Session(const Session&) = delete;
  Session& operator=(const Session&) = delete;

  [[nodiscard]] bool Sync(std::span<const TokenId> prompt,
                          std::string* error_msg = nullptr);
  [[nodiscard]] bool Evaluate(TokenId token, std::string* error_msg = nullptr);
  /// Appends `tokens` and returns the logits of every row ([n][vocab]);
  /// used for teacher-forced qualification.
  [[nodiscard]] bool EvaluateAll(std::span<const TokenId> tokens,
                                 std::vector<float>* logits,
                                 std::string* error_msg = nullptr);

  struct DecodeResult {
    std::vector<TokenId> tokens;
    bool stop{false};
  };
  /// Emits the next tokens. With the MTP drafter one call drafts, verifies
  /// and emits the accepted prefix plus the target's own next sample, all
  /// drawn with `sampler`, so greedy decoding reproduces autoregressive
  /// output exactly and sampled decoding keeps the target distribution.
  /// Without it (or with a one-token budget) one token is emitted. The last
  /// emitted token stays pending until the next call or a Sync.
  [[nodiscard]] bool DecodeStep(std::size_t max_tokens,
                                sampling::SamplerState& sampler,
                                DecodeResult* result,
                                std::string* error_msg = nullptr,
                                bool stop_at_eos = true);
  struct SpeculativeStats {
    std::uint64_t cycles{0};
    std::uint64_t drafted{0};
    std::uint64_t accepted{0};
  };
  [[nodiscard]] const SpeculativeStats& Statistics() const noexcept {
    return stats_;
  }

  [[nodiscard]] std::span<const float> Logits() const noexcept {
    return valid_ ? std::span<const float>(logits_) : std::span<const float>{};
  }
  [[nodiscard]] std::span<const TokenId> Tokens() const noexcept {
    return tokens_;
  }
  [[nodiscard]] std::uint32_t Position() const noexcept {
    return static_cast<std::uint32_t>(tokens_.size());
  }
  [[nodiscard]] std::uint32_t ContextSize() const noexcept;
  [[nodiscard]] bool IsValid() const noexcept { return valid_; }
  [[nodiscard]] std::size_t AllocatedBytes() const noexcept;
  void Reset();

  /// Compatibility version; bump on payload or inference arithmetic changes.
  static constexpr std::uint32_t kSnapshotPayloadVersion = 1;
  [[nodiscard]] std::uint64_t SnapshotBytes() const;
  [[nodiscard]] std::unique_ptr<SessionSnapshot> SaveSnapshot(
      std::string* error_msg = nullptr) const;
  [[nodiscard]] bool RestoreSnapshot(const SessionSnapshot& snapshot,
                                     std::string* error_msg = nullptr);
  [[nodiscard]] bool RestoreSnapshot(std::span<const std::uint8_t> payload,
                                     std::string* error_msg = nullptr);

private:
  Session(std::shared_ptr<Model> model, std::unique_ptr<rocm::KvCache> cache);
  /// Evaluates tokens_[begin, end) in prefill chunks; the last row's logits
  /// land in logits_.
  bool Extend(std::size_t begin, std::string* error_msg);

  std::shared_ptr<Model> model_;
  std::unique_ptr<rocm::KvCache> cache_;
  std::vector<TokenId> tokens_;
  std::vector<float> logits_;
  bool valid_{false};
  /// Emitted but not yet evaluated; its predecessor's hidden state is the
  /// cache's frontier hidden.
  std::optional<TokenId> pending_;
  SpeculativeStats stats_;

  friend class Model;
};

}  // namespace gufo::models::gemma4

#endif  // GUFO_MODELS_GEMMA4_ENGINE_HPP_
