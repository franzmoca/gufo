#ifndef GUFO_MODELS_DEEPSEEK_V41_FLASH_DSPARK_SAMPLER_HPP_
#define GUFO_MODELS_DEEPSEEK_V41_FLASH_DSPARK_SAMPLER_HPP_

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <span>
#include <stdexcept>

#include "src/core/sampling.hpp"
#include "src/models/deepseek_v41_flash/runtime/model.h"

namespace gufo::models::deepseek_v41_flash {

/// Lends a request sampler to DSpark cycles.
///
/// Sampled proposals use bounded q with exact rejection/residual correction.
/// Target filters and penalties always see the complete vocabulary and only
/// accepted history. Greedy requests use point-mass proposals; sampled
/// requests can draw from bounded q, including unfiltered target sampling.
/// A rejected draw becomes the next cycle's anchor. The owner records emitted
/// tokens and copies RNG state back after each cycle with `rng_state()`.
class DsparkSamplerBridge {
public:
  explicit DsparkSamplerBridge(const sampling::SamplerState& sampler)
      : working_(sampler),
        hook_{.ctx = this,
              .sample = &Sample,
              .accept = &Accept,
              .propose = UseStochastic(sampler.config()) ? &Propose : nullptr,
              .verify = UseStochastic(sampler.config()) ? &Verify : nullptr,
              .draws = UseStochastic(sampler.config()) ? &Draws : nullptr,
              .retain = UseStochastic(sampler.config()) ? &Retain : nullptr,
              .try_accept_argmax = sampler.config().temperature == 0.F &&
                                           !sampler.config().penalties_enabled()
                                       ? &TryAcceptArgmax
                                       : nullptr} {}

  DsparkSamplerBridge(const DsparkSamplerBridge&) = delete;
  DsparkSamplerBridge& operator=(const DsparkSamplerBridge&) = delete;

  [[nodiscard]] const ds41_dspark_sampler* hook() const { return &hook_; }
  [[nodiscard]] std::uint64_t rng_state() const { return working_.rng_state(); }

private:
  static bool UseStochastic(const sampling::SamplingConfig& config) {
    return config.uses_random_sampling();
  }

  struct Proposal {
    std::array<sampling::TokenId, DS41_DSPARK_CANDIDATES> ids{};
    std::array<float, DS41_DSPARK_CANDIDATES> probabilities{};
    sampling::TokenId token{};
    float probability{};
    float confidence{};
    std::size_t size{};
  };

  static int Propose(void* ctx, uint32_t row,
                     const ds41_dspark_candidates* candidates) {
    auto& self = *static_cast<DsparkSamplerBridge*>(ctx);
    if (row >= self.proposals_.size() || candidates == nullptr)
      throw std::invalid_argument("invalid DSpark proposal position");
    for (std::size_t i = 0; i < DS41_DSPARK_CANDIDATES; ++i) {
      if (std::isnan(candidates->logits[i]) ||
          candidates->logits[i] == INFINITY)
        throw std::invalid_argument("non-finite DSpark proposal logit");
      if (candidates->ids[i] < 0)
        throw std::invalid_argument("invalid DSpark proposal token");
      for (std::size_t j = 0; j < i; ++j)
        if (candidates->ids[i] == candidates->ids[j])
          throw std::invalid_argument("duplicate DSpark proposal token");
    }
    // Upstream q applies temperature; request filters/penalties define p.
    // Restricting q to this shortlist changes acceptance, never the target.
    const auto distribution = sampling::BuildDistribution(
        candidates->logits,
        {.temperature = self.working_.config().temperature});
    auto& proposal = self.proposals_[row];
    proposal = {};
    proposal.confidence = candidates->confidence;
    // Captured-row controls show that diffuse proposals gain acceptance from
    // p/q, while concentrated proposals do better as a point mass. This
    // choice depends only on q before drawing, so both routes remain exact.
    if (distribution.entries().front().value >= 0.7) {
      const auto id = candidates->ids[distribution.best_token()];
      proposal.size = 1;
      proposal.ids[0] = proposal.token = static_cast<sampling::TokenId>(id);
      proposal.probabilities[0] = proposal.probability = 1.F;
      return id;
    }
    constexpr std::uint32_t units = 1U << 24;
    std::array<std::uint32_t, DS41_DSPARK_CANDIDATES> masses{};
    std::uint32_t total = 0;
    for (const auto& entry : distribution.entries()) {
      const auto id = candidates->ids[entry.token];
      proposal.ids[proposal.size] = static_cast<sampling::TokenId>(id);
      masses[proposal.size] =
          static_cast<std::uint32_t>(std::floor(entry.value * units));
      total += masses[proposal.size++];
    }
    // F32 q sums to exactly one and matches the 24-bit draw exactly.
    masses[0] += units - total;
    const auto draw =
        static_cast<std::uint32_t>(self.working_.Uniform() * units);
    std::uint32_t cumulative = 0;
    for (std::size_t i = 0; i < proposal.size; ++i) {
      proposal.probabilities[i] = static_cast<float>(masses[i]) / units;
      cumulative += masses[i];
      if (proposal.probability == 0 && draw < cumulative) {
        proposal.token = proposal.ids[i];
        proposal.probability = proposal.probabilities[i];
      }
    }
    return static_cast<int>(proposal.token);
  }

  static float Draws(void* ctx, uint32_t count, uint32_t* out) {
    auto& self = *static_cast<DsparkSamplerBridge*>(ctx);
    for (uint32_t i = 0; i < count; ++i)
      out[i] = static_cast<std::uint32_t>(self.working_.Uniform() * (1U << 24));
    return self.working_.config().temperature;
  }

  static void Retain(void* ctx, uint32_t row, const int32_t* ids,
                     const uint32_t* masses, uint32_t size, int token) {
    auto& self = *static_cast<DsparkSamplerBridge*>(ctx);
    if (row >= self.proposals_.size() || size == 0 ||
        size > DS41_DSPARK_CANDIDATES)
      throw std::invalid_argument("invalid DSpark device proposal");
    auto& proposal = self.proposals_[row];
    proposal = {};
    proposal.size = size;
    proposal.token = static_cast<sampling::TokenId>(token);
    for (uint32_t i = 0; i < size; ++i) {
      if (ids[i] < 0)
        throw std::invalid_argument("invalid DSpark proposal token");
      proposal.ids[i] = static_cast<sampling::TokenId>(ids[i]);
      proposal.probabilities[i] = static_cast<float>(masses[i]) / (1U << 24);
      if (proposal.ids[i] == proposal.token && proposal.probability == 0)
        proposal.probability = proposal.probabilities[i];
    }
  }

  static int Verify(void* ctx, uint32_t row, const float* logits,
                    uint32_t vocabulary_size, int draft) {
    auto& self = *static_cast<DsparkSamplerBridge*>(ctx);
    if (row >= self.proposals_.size())
      throw std::invalid_argument("invalid DSpark verification position");
    const auto& proposal = self.proposals_[row];
    if (draft < 0 || proposal.token != static_cast<sampling::TokenId>(draft) ||
        !(proposal.probability > 0) || !std::isfinite(proposal.probability))
      throw std::invalid_argument("missing DSpark proposal distribution");
    for (std::size_t i = 0; i < proposal.size; ++i)
      if (proposal.ids[i] >= vocabulary_size)
        throw std::invalid_argument(
            "DSpark proposal exceeds target vocabulary");
    // Sampling p once implements exact delta acceptance/residual correction
    // and retains the fast AR sampler, without materializing target p.
    if (proposal.size == 1)
      return Sample(ctx, logits, vocabulary_size);
    const auto target = self.working_.Distribution({logits, vocabulary_size});
    const auto token =
        self.working_.Uniform() * proposal.probability <
                target.probability(proposal.token)
            ? proposal.token
            : target.SampleResidual(
                  std::span(proposal.ids).first(proposal.size),
                  std::span(proposal.probabilities).first(proposal.size),
                  self.working_.mutable_rng_state());
    if (std::getenv("GUFO_DEEPSEEK41_DSPARK_TRACE"))
      std::fprintf(stderr,
                   "ds41: DSpark sample row=%u confidence=%.6f accepted=%d "
                   "p=%.6f q=%.6f\n",
                   row, proposal.confidence, token == proposal.token,
                   target.probability(proposal.token), proposal.probability);
    self.working_.Accept(token);
    return static_cast<int>(token);
  }

  static int Sample(void* ctx, const float* logits,
                    std::uint32_t vocabulary_size) {
    auto& working = static_cast<DsparkSamplerBridge*>(ctx)->working_;
    const auto token =
        working.Sample(std::span<const float>(logits, vocabulary_size));
    working.Accept(token);
    return static_cast<int>(token);
  }

  static void Accept(void* ctx, int token) {
    static_cast<DsparkSamplerBridge*>(ctx)->working_.Accept(
        static_cast<sampling::TokenId>(token));
  }

  static bool TryAcceptArgmax(void* ctx, int token) {
    auto& working = static_cast<DsparkSamplerBridge*>(ctx)->working_;
    if (token < 0 ||
        !working.CanSelectArgmax(static_cast<sampling::TokenId>(token)))
      return false;
    working.Accept(static_cast<sampling::TokenId>(token));
    return true;
  }

  sampling::SamplerState working_;
  std::array<Proposal, 16> proposals_{};
  ds41_dspark_sampler hook_;
};

}  // namespace gufo::models::deepseek_v41_flash

#endif  // GUFO_MODELS_DEEPSEEK_V41_FLASH_DSPARK_SAMPLER_HPP_
