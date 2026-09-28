#ifndef GUFO_MODELS_GEMMA4_DRAFT_POLICY_HPP_
#define GUFO_MODELS_GEMMA4_DRAFT_POLICY_HPP_

#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>

#include "src/models/qwen38_flash_next/mtp_sampling.hpp"

namespace gufo::models::gemma4 {

/// How many MTP drafts a cycle verifies. Every policy decides from drafter
/// outputs, verification outcomes and fixed costs, never request timings, so
/// greedy output is unchanged by the choice and sampled output keeps the
/// target distribution.
enum class DraftPolicy : std::uint8_t {
  /// Verify another draft while its calibrated survival probability is
  /// worth its marginal draft-plus-verification cost (the default).
  kCalibrated,
  /// Continue while the product of raw drafter confidences stays above a
  /// fixed floor (0.5 greedy, 0.3 sampled with at most four drafts).
  kConfidence,
  /// Always draft up to the configured limit.
  kFixed,
};

/// "calibrated" (also the empty string), "confidence" or "fixed"; throws
/// std::invalid_argument otherwise.
[[nodiscard]] DraftPolicy ParseDraftPolicy(std::string_view name);
[[nodiscard]] std::string_view DraftPolicyName(DraftPolicy policy) noexcept;

/// Where the calibrated policy keeps what it learned.
enum class DraftCalibrationScope : std::uint8_t {
  kShared,   ///< one table per model, learned across requests (default)
  kRequest,  ///< reset for every request: seeded sampled replay is exact
};

/// "shared" (also the empty string) or "request".
[[nodiscard]] DraftCalibrationScope ParseDraftCalibrationScope(
    std::string_view name);
[[nodiscard]] std::string_view DraftCalibrationScopeName(
    DraftCalibrationScope scope) noexcept;

/// The drafter's confidence in its own step: AdaEDL's entropy bound
/// 1 - sqrt(0.2 H) over the top-64 softmax, clamped to [0, 1].
[[nodiscard]] float DraftSignal(
    const qwen38_flash_next::MtpCandidateLogits& candidates) noexcept;

/// gfx1151 costs of one cycle in milliseconds at a context depth:
/// `draft[n]` for n drafter steps and `verify[r]` for r verified rows, for
/// the dense 31B family or the mixture-of-experts 26B-A4B (`experts`).
struct DraftCosts {
  std::array<float, 9> draft{};
  std::array<float, 10> verify{};
};
[[nodiscard]] DraftCosts DraftCostsAt(std::uint32_t context,
                                      bool experts = false) noexcept;

/// Signal to acceptance probability, learned from verification: accepted
/// drafts and the first rejected one. Each bin starts at its midpoint with
/// the weight of kPriorWeight observations.
class DraftCalibration {
public:
  static constexpr std::size_t kBins = 8;
  static constexpr float kPriorWeight = 8.0F;

  [[nodiscard]] float Estimate(float signal) const noexcept;
  void Observe(float signal, bool accepted) noexcept;
  void Reset() noexcept;

private:
  [[nodiscard]] static std::size_t Bin(float signal) noexcept;

  std::array<float, kBins> accepted_{};
  std::array<float, kBins> seen_{};
};

/// One cycle's calibrated decisions. A draft is verified while the extra
/// tokens it is expected to add per millisecond of extra work (its survival
/// over one more verification row and drafter step) at least match the
/// cycle's current rate; the first `min_drafts` are always verified.
class CalibratedChain {
public:
  CalibratedChain(const DraftCalibration& calibration, const DraftCosts& costs,
                  std::uint32_t min_drafts, std::uint32_t cap) noexcept;

  /// Decides the next draft from its signal.
  [[nodiscard]] bool Include(float signal) noexcept;

private:
  const DraftCalibration& calibration_;
  const DraftCosts& costs_;
  std::uint32_t min_drafts_;
  std::uint32_t cap_;
  std::uint32_t kept_{0};
  float survival_{1.0F};
  float expected_{1.0F};  ///< the target's own token is always emitted
};

}  // namespace gufo::models::gemma4

#endif  // GUFO_MODELS_GEMMA4_DRAFT_POLICY_HPP_
