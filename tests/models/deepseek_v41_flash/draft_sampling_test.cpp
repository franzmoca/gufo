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
// Fused Markov selection: argmax/top eight of logits + W2 x against an FP64
// reference, with exact ties resolved to the lowest token across blocks.
void MarkovFused() {
  constexpr uint32_t n = 129280, rank = 256;
  std::vector<_Float16> w(static_cast<std::size_t>(n) * rank);
  std::vector<float> x(rank), logits(n);
  uint64_t state = 12345;
  const auto next = [&] {
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    return static_cast<float>((state >> 40) & 0xffffff) / 16777216.F - .5F;
  };
  for (auto& v : w)
    v = static_cast<_Float16>(next());
  for (auto& v : x)
    v = next();
  for (auto& v : logits)
    v = 4.F * next();
  logits[3] = NAN;
  logits[n - 1] = -INFINITY;
  // Two exact ties far apart (different blocks), above every other value.
  for (const uint32_t t : {70001U, 9U}) {
    std::copy_n(w.begin() + 5 * rank, rank,
                w.begin() + static_cast<std::size_t>(t) * rank);
    logits[t] = 40.F;
  }
  std::vector<double> ref(n);
  for (uint32_t t = 0; t < n; ++t) {
    double dot = 0;
    for (uint32_t i = 0; i < rank; ++i)
      dot +=
          static_cast<double>(w[static_cast<std::size_t>(t) * rank + i]) * x[i];
    ref[t] = logits[t] + dot;
  }
  std::vector<int32_t> order;
  for (uint32_t t = 0; t < n; ++t)
    if (std::isfinite(ref[t]))
      order.push_back(static_cast<int32_t>(t));
  std::stable_sort(order.begin(), order.end(),
                   [&](int32_t a, int32_t b) { return ref[a] > ref[b]; });
  Check(order[0] == 9 && order[1] == 70001, "reference ties");
  _Float16* dw = nullptr;
  float *dx = nullptr, *dl = nullptr, *cv = nullptr, *tv = nullptr;
  int32_t *ci = nullptr, *ids = nullptr, *tok = nullptr;
  uint32_t* dd = nullptr;
  ds41_v41_proposal* dq = nullptr;
  Check(hipMalloc(&dw, w.size() * 2) == hipSuccess &&
            hipMalloc(&dx, rank * 4) == hipSuccess &&
            hipMalloc(&dl, n * 4) == hipSuccess &&
            hipMalloc(&cv, DS41_V41_MARKOV_CANDIDATES * 4) == hipSuccess &&
            hipMalloc(&ci, DS41_V41_MARKOV_CANDIDATES * 4) == hipSuccess &&
            hipMalloc(&tv, 32) == hipSuccess &&
            hipMalloc(&ids, 32) == hipSuccess &&
            hipMalloc(&tok, 16) == hipSuccess &&
            hipMalloc(&dd, 4) == hipSuccess &&
            hipMalloc(&dq, sizeof(ds41_v41_proposal)) == hipSuccess,
        "markov allocation");
  const uint32_t zero = 0;
  Check(hipMemcpy(dw, w.data(), w.size() * 2, hipMemcpyHostToDevice) ==
                hipSuccess &&
            hipMemcpy(dx, x.data(), rank * 4, hipMemcpyHostToDevice) ==
                hipSuccess &&
            hipMemcpy(dl, logits.data(), n * 4, hipMemcpyHostToDevice) ==
                hipSuccess &&
            hipMemcpy(dd, &zero, 4, hipMemcpyHostToDevice) == hipSuccess,
        "markov upload");
  const auto* w16 = reinterpret_cast<const uint16_t*>(dw);
  int32_t tokens[4] = {-2, -2, -2, -2};
  Check(ds41_v41_markov_argmax(w16, dx, dl, n, cv, ci, tok, 2) == 0 &&
            hipMemcpy(tokens, tok, sizeof(tokens), hipMemcpyDeviceToHost) ==
                hipSuccess &&
            tokens[2] == 9,
        "fused argmax keeps the lowest tied token");
  // Without the tie: the fused argmax/top eight follow the reference order.
  logits[9] = logits[70001] = -40.F;
  Check(
      hipMemcpy(dl, logits.data(), n * 4, hipMemcpyHostToDevice) == hipSuccess,
      "markov upload");
  for (const int32_t t : {9, 70001})
    ref[t] = logits[t] - 1e9;
  std::stable_sort(order.begin(), order.end(),
                   [&](int32_t a, int32_t b) { return ref[a] > ref[b]; });
  Check(ds41_v41_markov_argmax(w16, dx, dl, n, cv, ci, tok, 1) == 0 &&
            hipMemcpy(tokens, tok, sizeof(tokens), hipMemcpyDeviceToHost) ==
                hipSuccess &&
            tokens[1] == order[0],
        "fused argmax");
  ds41_v41_proposal q{};
  std::array<int32_t, 8> top{};
  std::array<float, 8> top_v{};
  Check(
      ds41_v41_markov_propose(w16, dx, dl, n, cv, ci, ids, tv, dd, 1.F, dq,
                              tok) == 0 &&
          hipMemcpy(top.data(), ids, 32, hipMemcpyDeviceToHost) == hipSuccess &&
          hipMemcpy(top_v.data(), tv, 32, hipMemcpyDeviceToHost) ==
              hipSuccess &&
          hipMemcpy(&q, dq, sizeof(q), hipMemcpyDeviceToHost) == hipSuccess,
      "fused proposal");
  for (uint32_t j = 0; j < 8; ++j)
    Check(top[j] == order[j] && std::abs(top_v[j] - ref[order[j]]) < 1e-4,
          "fused top eight");
  Check(q.token == top[0] && (q.size == 1 || q.size == 8),
        "fused proposal draw");
  for (void* ptr :
       {static_cast<void*>(dw), static_cast<void*>(dx), static_cast<void*>(dl),
        static_cast<void*>(cv), static_cast<void*>(ci), static_cast<void*>(tv),
        static_cast<void*>(ids), static_cast<void*>(tok),
        static_cast<void*>(dd), static_cast<void*>(dq)})
    (void)hipFree(ptr);
}
// Learned acceptance head: sigmoid of the pre-norm head collapse and Markov
// input against an FP64 reference, for a row other than the first.
void Confidence() {
  constexpr uint32_t dim = 4096, rank = 256, rows = 3, row = 2;
  std::vector<float> h(static_cast<std::size_t>(rows) * 4 * dim), pre(rows * 4),
      x(rank), w(dim + rank);
  uint64_t state = 777;
  const auto next = [&] {
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    return static_cast<float>((state >> 40) & 0xffffff) / 16777216.F - .5F;
  };
  for (auto* v : {&h, &pre, &x, &w})
    for (auto& e : *v)
      e = next();
  const float bias = .25F;
  double dot = bias;
  for (uint32_t i = 0; i < dim; ++i) {
    double v = 0;
    for (uint32_t c = 0; c < 4; ++c)
      v += static_cast<double>(pre[row * 4 + c]) *
           h[(static_cast<std::size_t>(row) * 4 + c) * dim + i];
    dot += v * w[i];
  }
  for (uint32_t i = 0; i < rank; ++i)
    dot += static_cast<double>(x[i]) * w[dim + i];
  const double ref = 1 / (1 + std::exp(-dot));
  float *dh = nullptr, *dp = nullptr, *dx = nullptr, *dw = nullptr,
        *db = nullptr, *out = nullptr;
  Check(hipMalloc(&dh, h.size() * 4) == hipSuccess &&
            hipMalloc(&dp, pre.size() * 4) == hipSuccess &&
            hipMalloc(&dx, rank * 4) == hipSuccess &&
            hipMalloc(&dw, w.size() * 4) == hipSuccess &&
            hipMalloc(&db, 4) == hipSuccess &&
            hipMalloc(&out, rows * 4) == hipSuccess,
        "confidence allocation");
  Check(hipMemcpy(dh, h.data(), h.size() * 4, hipMemcpyHostToDevice) ==
                hipSuccess &&
            hipMemcpy(dp, pre.data(), pre.size() * 4, hipMemcpyHostToDevice) ==
                hipSuccess &&
            hipMemcpy(dx, x.data(), rank * 4, hipMemcpyHostToDevice) ==
                hipSuccess &&
            hipMemcpy(dw, w.data(), w.size() * 4, hipMemcpyHostToDevice) ==
                hipSuccess &&
            hipMemcpy(db, &bias, 4, hipMemcpyHostToDevice) == hipSuccess,
        "confidence upload");
  float got[rows] = {};
  Check(
      ds41_v41_dspark_confidence(dh, dp, dx, dw, db, dim, rank, row, out) ==
              0 &&
          hipMemcpy(got, out, sizeof(got), hipMemcpyDeviceToHost) == hipSuccess,
      "confidence launch");
  Check(std::abs(got[row] - ref) < 1e-5,
        "confidence matches the FP64 reference");
  Check(
      ds41_v41_dspark_confidence(dh, dp, dx, dw, nullptr, dim, rank, row,
                                 out) == 0 &&
          hipMemcpy(got, out, sizeof(got), hipMemcpyDeviceToHost) == hipSuccess,
      "confidence launch without bias");
  Check(std::abs(got[row] - 1 / (1 + std::exp(-(dot - bias)))) < 1e-5,
        "confidence without bias");
  for (float* ptr : {dh, dp, dx, dw, db, out})
    (void)hipFree(ptr);
}
// Device target draws against the shared host sampler: identical tokens,
// acceptance and uniform consumption for plain draws, accepted and rejected
// p/q proposals and residual draws, over rows with NaN/-inf logits. Rows the
// host draws without a uniform are reported degenerate.
void DeviceTargetDraw() {
  using gufo::sampling::SamplerState;
  constexpr uint32_t n = 129280;
  std::vector<float> logits(n);
  uint64_t state = 4242;
  const auto next = [&] {
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    return static_cast<float>((state >> 40) & 0xffffff) / 16777216.F;
  };
  float* dl = nullptr;
  void* scratch = nullptr;
  ds41_v41_proposal* dq = nullptr;
  Check(hipMalloc(&dl, n * 4) == hipSuccess &&
            hipMalloc(&scratch, DS41_V41_SAMPLE_SCRATCH_BYTES) == hipSuccess &&
            hipMalloc(&dq, sizeof(ds41_v41_proposal)) == hipSuccess,
        "target draw allocation");
  uint32_t accepted = 0, rejected = 0, plain = 0;
  for (uint32_t trial = 0; trial < 240; ++trial) {
    const float scale = trial % 3 == 0 ? 1.F : (trial % 3 == 1 ? 4.F : 9.F);
    const float temperature = trial % 4 == 0 ? .6F : 1.F;
    for (auto& v : logits)
      v = scale * (next() + next() + next() - 1.5F);
    logits[next() * n] = NAN;
    logits[next() * n] = -INFINITY;
    // A peaked head so proposals overlap with most of the target mass.
    std::vector<int32_t> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::partial_sort(order.begin(), order.begin() + 16, order.end(),
                      [&](int32_t a, int32_t b) {
                        return std::isnan(logits[b]) || logits[a] > logits[b];
                      });
    for (uint32_t k = 0; k < 16; ++k)
      logits[order[k]] += 10.F;
    Check(hipMemcpy(dl, logits.data(), n * 4, hipMemcpyHostToDevice) ==
              hipSuccess,
          "target draw upload");
    const bool propose = trial % 2 == 1;
    ds41_v41_proposal q{};
    std::vector<gufo::sampling::TokenId> ids;
    std::vector<float> probabilities;
    if (propose) {
      // Shuffled head tokens with random masses summing to 2^24.
      q.size = 8;
      uint32_t total = 0;
      for (uint32_t k = 0; k < 8; ++k) {
        q.ids[k] = order[(k * 5 + trial) % 16];
        q.masses[k] = 1 + static_cast<uint32_t>(next() * 1500000.F);
        total += q.masses[k];
      }
      q.masses[0] += (1U << 24) - total;
      q.token = q.ids[(trial / 2) % 8];
      for (uint32_t k = 0; k < 8; ++k) {
        ids.push_back(static_cast<gufo::sampling::TokenId>(q.ids[k]));
        probabilities.push_back(static_cast<float>(q.masses[k]) / (1U << 24));
      }
      Check(hipMemcpy(dq, &q, sizeof(q), hipMemcpyHostToDevice) == hipSuccess,
            "proposal upload");
    }
    SamplerState host(
        {.temperature = temperature, .seed = static_cast<int64_t>(trial) + 1});
    SamplerState device = host;
    int expected = -1;
    bool expected_accept = false;
    if (propose) {
      const auto target = host.Distribution(logits);
      const double probability = probabilities[(trial / 2) % 8];
      expected_accept =
          host.Uniform() * probability < target.probability(q.token);
      expected = expected_accept
                     ? q.token
                     : static_cast<int>(target.SampleResidual(
                           ids, probabilities, host.mutable_rng_state()));
    } else {
      expected = static_cast<int>(host.Sample(logits));
    }
    const double u = device.Uniform();
    const uint64_t after_one = device.rng_state();
    const double u2 = device.Uniform();
    ds41_v41_sample_result r{};
    Check(ds41_v41_sample_row(dl, n, propose ? dq : nullptr, temperature, u, u2,
                              scratch, &r) == 0,
          "target draw launch");
    Check(!r.degenerate, "target draw is not degenerate");
    Check(r.token == expected && (r.accepted != 0) == expected_accept,
          "device draw matches the host sampler");
    if (r.uniforms == 1)
      device.SetRngState(after_one);
    Check(device.rng_state() == host.rng_state(),
          "device draw consumes the host's uniforms");
    int best = -1;
    for (uint32_t i = 0; i < n; ++i)
      if (std::isfinite(logits[i]) && (best < 0 || logits[i] > logits[best]))
        best = static_cast<int>(i);
    Check(r.argmax == best, "device argmax");
    accepted += propose && expected_accept;
    rejected += propose && !expected_accept;
    plain += !propose;
  }
  Check(accepted > 10 && rejected > 10 && plain == 120, "target draw coverage");
  // One finite logit far above the rest: the host draws without a uniform.
  std::fill(logits.begin(), logits.end(), 0.F);
  logits[77] = 2000.F;
  Check(
      hipMemcpy(dl, logits.data(), n * 4, hipMemcpyHostToDevice) == hipSuccess,
      "target draw upload");
  ds41_v41_sample_result r{};
  Check(ds41_v41_sample_row(dl, n, nullptr, 1.F, .5, .5, scratch, &r) == 0 &&
            r.degenerate && r.argmax == 77,
        "single-token rows are degenerate");
  for (void* ptr : {static_cast<void*>(dl), scratch, static_cast<void*>(dq)})
    (void)hipFree(ptr);
  std::printf("device target draws: %u accepted, %u rejected, %u plain\n",
              accepted, rejected, plain);
}
}  // namespace

int main() {
  int devices = 0;
  if (hipGetDeviceCount(&devices) != hipSuccess || !devices)
    return 77;
  Top8();
  Sampler();
  DeviceProposal();
  MarkovFused();
  Confidence();
  DeviceTargetDraw();
  std::puts(
      "PASS: post-Markov top8, fused Markov selection, learned confidence, "
      "device target draws, device q, exact sampled "
      "correction and seeded replay");
}
