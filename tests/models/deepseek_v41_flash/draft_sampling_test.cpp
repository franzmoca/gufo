#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <vector>

#include "ds41_v41.h"
#include "src/models/deepseek_v41_flash/dspark_sampler.hpp"

namespace {
void Check(bool condition, const char* message) {
  if (!condition) {
    std::fprintf(stderr, "FAIL: %s\n", message);
    std::abort();
  }
}

void Top8() {
  constexpr uint32_t size = 129283;
  std::vector<float> x(size), bias(size, 0.F);
  for (uint32_t i = 0; i < size; ++i)
    x[i] = static_cast<float>(i % 17);
  x[0] = NAN;
  x[1] = -INFINITY;
  x.back() = INFINITY;
  bias[7] = 22.F;
  std::vector<int32_t> expected;
  for (uint32_t i = 0; i < size; ++i)
    if (std::isfinite(x[i] + bias[i]))
      expected.push_back(static_cast<int32_t>(i));
  std::partial_sort(expected.begin(), expected.begin() + 8, expected.end(),
                    [&](int32_t a, int32_t b) {
                      const float va = x[a] + bias[a], vb = x[b] + bias[b];
                      return va != vb ? va > vb : a < b;
                    });
  float *dx = nullptr, *db = nullptr, *dv = nullptr;
  int32_t* di = nullptr;
  Check(hipMalloc(&dx, size * sizeof(float)) == hipSuccess &&
            hipMalloc(&db, size * sizeof(float)) == hipSuccess &&
            hipMalloc(&dv, 8 * sizeof(float)) == hipSuccess &&
            hipMalloc(&di, 8 * sizeof(int32_t)) == hipSuccess,
        "top8 allocation");
  Check(hipMemcpy(dx, x.data(), size * sizeof(float), hipMemcpyHostToDevice) ==
                hipSuccess &&
            hipMemcpy(db, bias.data(), size * sizeof(float),
                      hipMemcpyHostToDevice) == hipSuccess &&
            ds41_v41_top8_bias(dx, db, size, di, dv) == 0,
        "top8 launch");
  std::array<int32_t, 8> ids;
  std::array<float, 8> values;
  Check(hipMemcpy(ids.data(), di, sizeof(ids), hipMemcpyDeviceToHost) ==
                hipSuccess &&
            hipMemcpy(values.data(), dv, sizeof(values),
                      hipMemcpyDeviceToHost) == hipSuccess,
        "top8 copy");
  for (size_t i = 0; i < ids.size(); ++i)
    Check(ids[i] == expected[i] && values[i] == x[ids[i]] + bias[ids[i]],
          "biased top8 and stable ties");
  for (void* p : {static_cast<void*>(dx), static_cast<void*>(db),
                  static_cast<void*>(dv), static_cast<void*>(di)})
    (void)hipFree(p);
}

void Sampler() {
  using gufo::models::deepseek_v41_flash::DsparkSamplerBridge;
  using gufo::sampling::SamplerState;
  const std::array<float, 10> logits{-0.5F, 0.1F,  -1.F, 0.2F, -0.7F,
                                     0.3F,  -0.2F, 0.F,  1.F,  0.8F};
  ds41_dspark_candidates candidates{};
  for (uint32_t i = 0; i < DS41_DSPARK_CANDIDATES; ++i) {
    candidates.ids[i] = static_cast<int32_t>(i);
    candidates.logits[i] = -0.1F * i;
  }
  std::array<double, 10> expected{};
  double total = 0;
  for (size_t i = 0; i < expected.size(); ++i)
    total += expected[i] = std::exp(static_cast<double>(logits[i]));
  for (auto& value : expected)
    value /= total;
  constexpr uint32_t trials = 40000;
  std::array<uint32_t, 10> counts{};
  for (uint32_t seed = 0; seed < trials; ++seed) {
    DsparkSamplerBridge bridge(
        SamplerState({.temperature = 1.F, .top_p = 1.F, .seed = seed}));
    const auto* hook = bridge.hook();
    Check(hook->propose && hook->verify,
          "unfiltered requests use stochastic proposals");
    const int draft = hook->propose(hook->ctx, 0, &candidates);
    const int selected =
        hook->verify(hook->ctx, 0, logits.data(), logits.size(), draft);
    Check(selected >= 0 && selected < static_cast<int>(counts.size()),
          "verified token range");
    ++counts[selected];
  }
  for (size_t i = 0; i < counts.size(); ++i) {
    const double mean = trials * expected[i];
    Check(std::abs(counts[i] - mean) <
              6 * std::sqrt(mean * (1 - expected[i])) + 10,
          "p/q plus residual reproduces independent target probabilities");
  }
  Check(counts[8] && counts[9], "residual correction covers tokens outside q");
  DsparkSamplerBridge a(SamplerState({.temperature = 1.F, .seed = 73}));
  DsparkSamplerBridge b(SamplerState({.temperature = 1.F, .seed = 73}));
  for (uint32_t row = 0; row < 3; ++row) {
    const auto *ha = a.hook(), *hb = b.hook();
    const int da = ha->propose(ha->ctx, row, &candidates),
              db = hb->propose(hb->ctx, row, &candidates);
    Check(da == db &&
              ha->verify(ha->ctx, row, logits.data(), logits.size(), da) ==
                  hb->verify(hb->ctx, row, logits.data(), logits.size(), db) &&
              a.rng_state() == b.rng_state(),
          "seeded proposal and residual replay");
  }
}
// Device q: temperature softmax of the biased top eight in 2^-24 units, the
// draw following those masses, and exact target recovery via Retain/Verify.
void DeviceProposal() {
  using gufo::models::deepseek_v41_flash::DsparkSamplerBridge;
  using gufo::sampling::SamplerState;
  constexpr uint32_t size = 4096, units = 1U << 24;
  std::vector<float> x(size, -30.F), bias(size, 0.F);
  const std::array<float, 10> top{1.F, .8F,  .5F,  .3F,  .2F,
                                  0.F, -.1F, -.4F, -.9F, -2.F};
  for (uint32_t i = 0; i < top.size(); ++i)
    x[100 + 37 * i] = top[i];
  bias[100 + 37 * 9] = 1.5F;  // the bias reorders candidates
  float *dx = nullptr, *db = nullptr, *dv = nullptr;
  int32_t *di = nullptr, *dt = nullptr;
  uint32_t* dd = nullptr;
  ds41_v41_proposal* dq = nullptr;
  Check(hipMalloc(&dx, size * 4) == hipSuccess &&
            hipMalloc(&db, size * 4) == hipSuccess &&
            hipMalloc(&dv, 32) == hipSuccess &&
            hipMalloc(&di, 32) == hipSuccess &&
            hipMalloc(&dt, 4) == hipSuccess &&
            hipMalloc(&dd, 4) == hipSuccess &&
            hipMalloc(&dq, sizeof(ds41_v41_proposal)) == hipSuccess,
        "proposal allocation");
  Check(
      hipMemcpy(dx, x.data(), size * 4, hipMemcpyHostToDevice) == hipSuccess &&
          hipMemcpy(db, bias.data(), size * 4, hipMemcpyHostToDevice) ==
              hipSuccess,
      "proposal upload");
  const auto run = [&](uint32_t draw, float temperature) {
    ds41_v41_proposal q{};
    int32_t token = -2;
    Check(
        hipMemcpy(dd, &draw, 4, hipMemcpyHostToDevice) == hipSuccess &&
            ds41_v41_propose_top8(dx, db, size, di, dv, dd, temperature, dq,
                                  dt) == 0 &&
            hipMemcpy(&q, dq, sizeof(q), hipMemcpyDeviceToHost) == hipSuccess &&
            hipMemcpy(&token, dt, 4, hipMemcpyDeviceToHost) == hipSuccess,
        "proposal launch");
    Check(token == q.token, "device token matches q's draw");
    return q;
  };
  const auto q = run(0, 1.F);
  Check(q.size == 8, "diffuse q keeps eight candidates");
  std::vector<int32_t> order;
  for (uint32_t i = 0; i < size; ++i)
    order.push_back(static_cast<int32_t>(i));
  std::stable_sort(order.begin(), order.end(), [&](int32_t a, int32_t b) {
    return x[a] + bias[a] > x[b] + bias[b];
  });
  double total = 0;
  for (uint32_t i = 0; i < 8; ++i)
    total +=
        std::exp(x[order[i]] + bias[order[i]] - x[order[0]] - bias[order[0]]);
  uint64_t sum = 0;
  for (uint32_t i = 0; i < 8; ++i) {
    Check(q.ids[i] == order[i], "q ids are the biased top eight");
    const double p =
        std::exp(x[order[i]] + bias[order[i]] - x[order[0]] - bias[order[0]]) /
        total;
    Check(std::abs(q.masses[i] / double(units) - p) < 1e-5,
          "q is the temperature softmax");
    sum += q.masses[i];
  }
  Check(sum == units, "q masses sum to one");
  uint32_t cumulative = 0;
  for (uint32_t i = 0; i < 8; ++i) {
    Check(run(cumulative, 1.F).token == q.ids[i] &&
              run(cumulative + q.masses[i] - 1, 1.F).token == q.ids[i],
          "draw boundaries follow q masses");
    cumulative += q.masses[i];
  }
  const auto point = run(units - 1, .05F);
  Check(point.size == 1 && point.token == order[0] && point.masses[0] == units,
        "concentrated q becomes a point mass");

  // Exactness: target logits over the first ids; proposals from device q.
  std::vector<float> target(size, -INFINITY);
  for (uint32_t i = 0; i < 10; ++i)
    target[order[i]] = 0.3F * static_cast<float>(i % 4) - 0.1F * i;
  std::vector<double> expected(size, 0.);
  double z = 0;
  for (uint32_t i = 0; i < 10; ++i)
    z += expected[order[i]] = std::exp(static_cast<double>(target[order[i]]));
  constexpr uint32_t trials = 30000;
  std::vector<uint32_t> counts(size, 0);
  for (uint32_t seed = 0; seed < trials; ++seed) {
    DsparkSamplerBridge bridge(
        SamplerState({.temperature = 1.F, .top_p = 1.F, .seed = seed}));
    const auto* hook = bridge.hook();
    Check(hook->draws && hook->retain, "sampled requests use device proposals");
    uint32_t draw = 0;
    const float temperature = hook->draws(hook->ctx, 1, &draw);
    const auto d = run(draw, temperature);
    hook->retain(hook->ctx, 0, d.ids, d.masses, d.size, d.token);
    const int selected =
        hook->verify(hook->ctx, 0, target.data(), size, d.token);
    Check(selected >= 0 && selected < static_cast<int>(size),
          "verified token range");
    ++counts[selected];
  }
  for (uint32_t i = 0; i < 10; ++i) {
    const double pr = expected[order[i]] / z, mean = trials * pr;
    Check(
        std::abs(counts[order[i]] - mean) < 6 * std::sqrt(mean * (1 - pr)) + 10,
        "device q plus residual reproduces target probabilities");
  }
  Check(counts[order[8]] && counts[order[9]],
        "residual covers tokens outside q");
  for (void* ptr :
       {static_cast<void*>(dx), static_cast<void*>(db), static_cast<void*>(dv),
        static_cast<void*>(di), static_cast<void*>(dt), static_cast<void*>(dd),
        static_cast<void*>(dq)})
    (void)hipFree(ptr);
}
}  // namespace

int main() {
  int devices = 0;
  if (hipGetDeviceCount(&devices) != hipSuccess || !devices)
    return 77;
  Top8();
  Sampler();
  DeviceProposal();
  std::puts(
      "PASS: post-Markov top8, device q, exact sampled correction and seeded "
      "replay");
}
