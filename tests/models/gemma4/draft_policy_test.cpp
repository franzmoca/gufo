// Host contract for the Gemma 4 draft-length policy: option names, the
// drafter signal, calibration learning, the cost table and the calibrated
// chain decision.
#include "src/models/gemma4/draft_policy.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

#include "tests/models/gemma4/check.hpp"

namespace g4 = gufo::models::gemma4;
using gemma4_test::Require;

namespace {

gufo::models::qwen38_flash_next::MtpCandidateLogits Candidates(float spread) {
  gufo::models::qwen38_flash_next::MtpCandidateLogits c;
  c.size = 64;
  for (std::size_t i = 0; i < c.size; ++i) {
    c.ids[i] = static_cast<gufo::sampling::TokenId>(i);
    c.logits[i] = i == 0 ? spread : 0.0F;
  }
  return c;
}

void CheckNames() {
  Require(g4::ParseDraftPolicy("") == g4::DraftPolicy::kCalibrated,
          "empty policy is not calibrated");
  for (const auto policy :
       {g4::DraftPolicy::kCalibrated, g4::DraftPolicy::kConfidence,
        g4::DraftPolicy::kFixed}) {
    Require(g4::ParseDraftPolicy(g4::DraftPolicyName(policy)) == policy,
            "policy name does not round-trip");
  }
  bool threw = false;
  try {
    (void)g4::ParseDraftPolicy("adaptive");
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  Require(threw, "unknown policy accepted");
  Require(
      g4::ParseDraftCalibrationScope("") == g4::DraftCalibrationScope::kShared,
      "empty scope is not shared");
  Require(g4::ParseDraftCalibrationScope("request") ==
              g4::DraftCalibrationScope::kRequest,
          "request scope not parsed");
}

void CheckSignal() {
  // A flat top-64 has entropy ln 64: 1 - sqrt(0.2 ln 64) = 0.088.
  const float flat = g4::DraftSignal(Candidates(0.0F));
  Require(std::fabs(flat - 0.0880F) < 1e-3F,
          "flat signal " + std::to_string(flat));
  const float peaked = g4::DraftSignal(Candidates(30.0F));
  Require(peaked > 0.99F && peaked <= 1.0F,
          "peaked signal " + std::to_string(peaked));
  Require(g4::DraftSignal(Candidates(4.0F)) > g4::DraftSignal(Candidates(2.0F)),
          "signal does not grow with confidence");
}

void CheckCalibration() {
  g4::DraftCalibration calibration;
  Require(std::fabs(calibration.Estimate(0.95F) - 0.9375F) < 1e-6F,
          "prior is not the bin midpoint");
  for (int i = 0; i < 1000; ++i) {
    calibration.Observe(0.95F, i % 4 != 0);  // 75% accepted
  }
  const float learned = calibration.Estimate(0.9F);
  Require(std::fabs(learned - 0.75F) < 0.01F,
          "calibration did not learn: " + std::to_string(learned));
  Require(std::fabs(calibration.Estimate(0.3F) - 0.3125F) < 1e-6F,
          "another bin moved");
  calibration.Reset();
  Require(std::fabs(calibration.Estimate(0.95F) - 0.9375F) < 1e-6F,
          "reset kept observations");
}

void CheckCosts() {
  const g4::DraftCosts shallow = g4::DraftCostsAt(0);
  Require(shallow.verify[1] == 90.6F && shallow.verify[8] == 115.1F,
          "d0 verification costs");
  Require(std::fabs(shallow.draft[7] - 7 * 1.95F) < 1e-4F, "d0 draft steps");
  const g4::DraftCosts mid = g4::DraftCostsAt(2048);
  Require(std::fabs(mid.verify[3] - 0.5F * (92.7F + 95.3F)) < 1e-3F,
          "interpolation between depths");
  const g4::DraftCosts deep = g4::DraftCostsAt(131072);
  const g4::DraftCosts measured = g4::DraftCostsAt(65536);
  for (std::size_t r = 1; r < deep.verify.size(); ++r) {
    Require(deep.verify[r] > measured.verify[r], "extrapolation past 64K");
    Require(deep.verify[r] >= deep.verify[r - 1], "rows are not monotone");
  }
}

void CheckChain() {
  const g4::DraftCosts costs = g4::DraftCostsAt(0);
  g4::DraftCalibration sure;
  g4::DraftCalibration hopeless;
  for (int i = 0; i < 10000; ++i) {
    for (float s = 0.05F; s < 1.0F; s += 0.125F) {
      sure.Observe(s, true);
      hopeless.Observe(s, false);
    }
  }
  {
    g4::CalibratedChain chain(sure, costs, 1, 7);
    for (int i = 0; i < 7; ++i) {
      Require(chain.Include(0.5F), "a sure chain stopped");
    }
    Require(!chain.Include(0.5F), "the cap was exceeded");
  }
  {
    g4::CalibratedChain chain(hopeless, costs, 1, 7);
    Require(chain.Include(0.5F), "the first draft was not verified");
    Require(!chain.Include(0.5F), "a hopeless chain continued");
  }
  {
    g4::CalibratedChain chain(hopeless, costs, 3, 7);
    for (int i = 0; i < 3; ++i) {
      Require(chain.Include(0.5F), "the minimum was not verified");
    }
    Require(!chain.Include(0.5F), "a hopeless chain passed its minimum");
  }
  {
    // Prior only: signal s is taken as the acceptance probability of its bin
    // midpoint. Draft two is worth it while survival * T >= E * dT.
    g4::DraftCalibration prior;
    g4::CalibratedChain chain(prior, costs, 1, 7);
    Require(chain.Include(0.95F), "first draft");  // survival 0.9375
    const float current = costs.draft[2] + costs.verify[2];
    const float next = costs.draft[3] + costs.verify[3];
    const float needed = (1.0F + 0.9375F) * (next - current) / current;
    // Bin midpoints 0.0625 (0.0-0.125) and 0.1875: survival 0.059 / 0.176.
    Require(0.9375F * 0.0625F < needed && 0.9375F * 0.1875F > needed,
            "test premise: threshold between two bins");
    g4::CalibratedChain again(prior, costs, 1, 7);
    Require(again.Include(0.95F), "first draft");
    Require(!again.Include(0.05F), "draft below the cost threshold kept");
    Require(chain.Include(0.15F), "draft above the cost threshold dropped");
  }
}

}  // namespace

int main() {
  return gemma4_test::Run([] {
    CheckNames();
    CheckSignal();
    CheckCalibration();
    CheckCosts();
    CheckChain();
  });
}
