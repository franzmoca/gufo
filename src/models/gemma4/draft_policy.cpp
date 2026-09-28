#include "src/models/gemma4/draft_policy.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace gufo::models::gemma4 {
namespace {

// Median complete-cycle costs on gfx1151 for the dense 31B family (QAT Q4_0
// target and drafter; the UD-Q4_K_XL pair has the same draft/verify ratios),
// measured 2026-09-28 by timing drafter chains and verification forwards at
// each depth. Only cost ratios steer the decision.
constexpr std::array<std::uint32_t, 5> kDepths = {0, 4096, 16384, 32768, 65536};
constexpr std::array<float, 5> kDraftStepMs = {1.95F, 2.05F, 2.40F, 2.82F,
                                               3.66F};
constexpr std::array<std::array<float, 8>, 5> kVerifyMs = {{
    {90.6F, 90.6F, 92.7F, 94.3F, 97.4F, 101.4F, 106.6F, 115.1F},
    {92.7F, 93.2F, 95.3F, 96.9F, 100.7F, 105.2F, 109.5F, 119.2F},
    {96.0F, 97.5F, 99.8F, 102.1F, 107.2F, 113.4F, 121.1F, 131.5F},
    {100.2F, 103.8F, 106.4F, 109.5F, 116.2F, 123.5F, 133.1F, 144.8F},
    {108.7F, 117.0F, 120.8F, 124.9F, 134.9F, 144.7F, 159.1F, 173.6F},
}};

// The 26B-A4B (UD-Q4_K_XL target, Unsloth Q8_0 drafter), measured
// 2026-09-29 on varied English prose (the rows' routing, and so the experts
// they read, follows the text) up to the 16 rows a batched forward verifies:
// every verified row adds the experts it routes to, so verification grows
// several times faster per row than on the dense family.
constexpr std::array<float, 5> kExpertDraftStepMs = {1.81F, 2.03F, 2.25F, 2.51F,
                                                     3.04F};
constexpr std::array<std::array<float, 16>, 5> kExpertVerifyMs = {{
    {18.3F, 21.1F, 23.1F, 25.4F, 27.3F, 29.1F, 30.8F, 31.8F, 33.2F, 34.4F,
     37.1F, 40.6F, 42.3F, 44.8F, 44.6F, 46.0F},
    {19.8F, 22.9F, 25.8F, 28.1F, 30.0F, 32.8F, 34.9F, 36.4F, 37.8F, 39.5F,
     42.5F, 46.0F, 48.9F, 51.1F, 51.0F, 52.6F},
    {20.8F, 24.6F, 27.3F, 29.5F, 32.1F, 35.1F, 37.9F, 39.6F, 41.9F, 43.5F,
     46.8F, 50.5F, 52.6F, 55.3F, 55.5F, 57.6F},
    {21.7F, 26.7F, 29.7F, 32.0F, 34.7F, 37.5F, 40.2F, 42.3F, 45.8F, 47.6F,
     50.9F, 55.4F, 58.0F, 61.2F, 61.5F, 63.6F},
    {23.6F, 28.2F, 31.9F, 34.2F, 37.6F, 41.6F, 45.6F, 48.0F, 53.5F, 55.6F,
     60.0F, 64.7F, 69.0F, 72.4F, 73.0F, 76.0F},
}};

}  // namespace

DraftPolicy ParseDraftPolicy(std::string_view name) {
  if (name.empty() || name == "calibrated") {
    return DraftPolicy::kCalibrated;
  }
  if (name == "confidence") {
    return DraftPolicy::kConfidence;
  }
  if (name == "fixed") {
    return DraftPolicy::kFixed;
  }
  throw std::invalid_argument(
      "Gemma 4 draft policy must be calibrated, confidence or fixed");
}

std::string_view DraftPolicyName(DraftPolicy policy) noexcept {
  switch (policy) {
    case DraftPolicy::kConfidence:
      return "confidence";
    case DraftPolicy::kFixed:
      return "fixed";
    case DraftPolicy::kCalibrated:
      break;
  }
  return "calibrated";
}

DraftCalibrationScope ParseDraftCalibrationScope(std::string_view name) {
  if (name.empty() || name == "shared") {
    return DraftCalibrationScope::kShared;
  }
  if (name == "request") {
    return DraftCalibrationScope::kRequest;
  }
  throw std::invalid_argument("draft calibration must be shared or request");
}

std::string_view DraftCalibrationScopeName(
    DraftCalibrationScope scope) noexcept {
  return scope == DraftCalibrationScope::kRequest ? "request" : "shared";
}

float DraftSignal(
    const qwen38_flash_next::MtpCandidateLogits& candidates) noexcept {
  if (candidates.size == 0) {
    return 0.0F;
  }
  float top = candidates.logits[0];
  for (std::size_t i = 1; i < candidates.size; ++i) {
    top = std::max(top, candidates.logits[i]);
  }
  double total = 0.0;
  double weighted = 0.0;
  for (std::size_t i = 0; i < candidates.size; ++i) {
    const double shifted = static_cast<double>(candidates.logits[i] - top);
    const double e = std::exp(shifted);
    total += e;
    weighted += e * shifted;
  }
  const double entropy = std::max(0.0, std::log(total) - weighted / total);
  const double signal = 1.0 - std::sqrt(0.2 * entropy);
  return static_cast<float>(std::clamp(signal, 0.0, 1.0));
}

namespace {

/// Costs from one table: linear between measured depths, past the deepest
/// one the last interval's slope continues; rows past the table keep its
/// last row step.
template<std::size_t kRows>
DraftCosts Interpolate(
    std::uint32_t context, const std::array<float, 5>& draft_step,
    const std::array<std::array<float, kRows>, 5>& verify) noexcept {
  std::size_t hi = 1;
  while (hi + 1 < kDepths.size() && context > kDepths[hi]) {
    ++hi;
  }
  const std::size_t lo = hi - 1;
  const float t = (static_cast<float>(context) - kDepths[lo]) /
                  static_cast<float>(kDepths[hi] - kDepths[lo]);
  const auto lerp = [t](float a, float b) {
    return std::max(a, a + t * (b - a));
  };
  DraftCosts costs;
  const float step = lerp(draft_step[lo], draft_step[hi]);
  for (std::size_t n = 0; n < costs.draft.size(); ++n) {
    costs.draft[n] = step * static_cast<float>(n);
  }
  for (std::size_t r = 1; r <= kRows && r < costs.verify.size(); ++r) {
    costs.verify[r] = lerp(verify[lo][r - 1], verify[hi][r - 1]);
  }
  for (std::size_t r = kRows + 1; r < costs.verify.size(); ++r) {
    costs.verify[r] =
        costs.verify[r - 1] + (costs.verify[kRows] - costs.verify[kRows - 1]);
  }
  costs.verify[0] = costs.verify[1];
  return costs;
}

}  // namespace

DraftCosts DraftCostsAt(std::uint32_t context, bool experts) noexcept {
  return experts ? Interpolate(context, kExpertDraftStepMs, kExpertVerifyMs)
                 : Interpolate(context, kDraftStepMs, kVerifyMs);
}

std::size_t DraftCalibration::Bin(float signal) noexcept {
  const float clamped = std::clamp(signal, 0.0F, 1.0F);
  return std::min(
      kBins - 1, static_cast<std::size_t>(clamped * static_cast<float>(kBins)));
}

float DraftCalibration::Estimate(float signal) const noexcept {
  const std::size_t bin = Bin(signal);
  const float prior =
      (static_cast<float>(bin) + 0.5F) / static_cast<float>(kBins);
  return (accepted_[bin] + kPriorWeight * prior) / (seen_[bin] + kPriorWeight);
}

void DraftCalibration::Observe(float signal, bool accepted) noexcept {
  const std::size_t bin = Bin(signal);
  accepted_[bin] += accepted ? 1.0F : 0.0F;
  seen_[bin] += 1.0F;
}

void DraftCalibration::Reset() noexcept {
  accepted_.fill(0.0F);
  seen_.fill(0.0F);
}

CalibratedChain::CalibratedChain(const DraftCalibration& calibration,
                                 const DraftCosts& costs,
                                 std::uint32_t min_drafts, std::uint32_t cap,
                                 DraftBatch others) noexcept
    : calibration_(calibration),
      costs_(costs),
      // Beside other sessions even the first draft must pay for its row.
      min_drafts_(others.rows > 0 ? 0U : std::max(1U, min_drafts)),
      cap_(std::min<std::uint32_t>(cap, costs.draft.size() - 1)),
      others_(others) {}

float CalibratedChain::Verify(std::uint32_t rows) const noexcept {
  return costs_.verify[std::min<std::size_t>(others_.rows + rows,
                                             costs_.verify.size() - 1)];
}

bool CalibratedChain::FirstDraftCanPay() const noexcept {
  if (min_drafts_ > 0 || cap_ == 0) {
    return cap_ > 0;
  }
  // Include's rule for the first draft with survival one, against a cycle
  // that runs no drafter step.
  const float without = others_.draft_ms + Verify(1);
  const float with = others_.draft_ms + costs_.draft[1] + Verify(2);
  return without >= (others_.expected + expected_) * (with - without);
}

bool CalibratedChain::Include(float signal) noexcept {
  const std::uint32_t index = kept_;  // drafter steps so far: index + 1
  if (index >= cap_) {
    return false;
  }
  steps_ = index + 1;
  const float survival = survival_ * calibration_.Estimate(signal);
  if (index >= min_drafts_) {
    // Stopping here verifies `index` drafts after one more drafter step than
    // needed; verifying this one adds a row, and going on adds a step.
    const float current =
        others_.draft_ms + costs_.draft[index + 1] + Verify(index + 1);
    const float next = others_.draft_ms +
                       costs_.draft[std::min(index + 2, cap_)] +
                       Verify(index + 2);
    if (survival * current <
        (others_.expected + expected_) * (next - current)) {
      return false;
    }
  }
  survival_ = survival;
  expected_ += survival;
  ++kept_;
  return true;
}

}  // namespace gufo::models::gemma4
