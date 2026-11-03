// Direction adapter edits against a double-precision host reference: the
// fused expand (attention form: one row, whole-output vectors; FFN form: the
// shared expert's vectors on extra, then the whole-sum vectors over extra
// plus six routed rows) and the in-place form followed by the plain expand
// sum, in project and add modes. Project vectors are not orthogonal, so the
// order of the per-vector edits is checked too.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <string_view>
#include <vector>

#include "ds41_v41.h"

namespace {

constexpr uint32_t kHc = 4, kDim = 5120;

float* Device(const std::vector<float>& host) {
  void* p = nullptr;
  if (hipMalloc(&p, host.size() * sizeof(float)) != hipSuccess)
    return nullptr;
  (void)hipMemcpy(p, host.data(), host.size() * sizeof(float),
                  hipMemcpyHostToDevice);
  return static_cast<float*>(p);
}

std::vector<float> Host(const float* p, std::size_t count) {
  std::vector<float> out(count);
  (void)hipMemcpy(out.data(), p, count * sizeof(float), hipMemcpyDeviceToHost);
  return out;
}

std::vector<float> Random(std::size_t count, float lo, float hi,
                          std::mt19937& rng) {
  std::uniform_real_distribution<float> dist(lo, hi);
  std::vector<float> v(count);
  for (auto& x : v)
    x = dist(rng);
  return v;
}

double RelErr(const std::vector<float>& got, const std::vector<double>& want) {
  double err = 0.0, ref = 0.0;
  for (std::size_t i = 0; i < want.size(); ++i) {
    err = std::max(err, std::fabs(got[i] - want[i]));
    ref = std::max(ref, std::fabs(want[i]));
  }
  return ref > 0.0 ? err / ref : err;
}

void Apply(std::vector<double>& y, const std::vector<float>& v, uint32_t k,
           double scale, bool add) {
  for (uint32_t j = 0; j < k; ++j) {
    const float* vj = v.data() + (std::size_t)j * kDim;
    double d = scale;
    if (!add) {
      double dot = 0.0;
      for (uint32_t i = 0; i < kDim; ++i)
        dot += y[i] * vj[i];
      d = -scale * dot;
    }
    for (uint32_t i = 0; i < kDim; ++i)
      y[i] += d * vj[i];
  }
}

/* extra_k vectors on extra, total_k on the sum with `rows` rows. */
bool Case(uint32_t n, uint32_t rows, uint32_t extra_k, uint32_t total_k,
          bool add, std::mt19937& rng) {
  const float scale = add ? 0.75f : 1.3f;
  const auto h_res = Random((std::size_t)n * kHc * kDim, -2.0f, 2.0f, rng);
  const auto h_extra = Random((std::size_t)n * kDim, -2.0f, 2.0f, rng);
  const auto h_x = Random((std::size_t)n * (rows + 1) * kDim, -2.0f, 2.0f, rng);
  const auto h_post = Random((std::size_t)n * kHc, 0.0f, 2.0f, rng);
  const auto h_comb = Random((std::size_t)n * kHc * kHc, 0.0f, 1.0f, rng);
  // Norms near 1 (as the published files) from entries of about 1/sqrt(dim).
  const float a = 1.7f / std::sqrt((float)kDim);
  const auto h_ev = Random((std::size_t)(extra_k + 1) * kDim, -a, a, rng);
  const auto h_tv = Random((std::size_t)(total_k + 1) * kDim, -a, a, rng);

  std::vector<double> want((std::size_t)n * kHc * kDim);
  for (uint32_t t = 0; t < n; ++t) {
    std::vector<double> y(kDim);
    for (uint32_t i = 0; i < kDim; ++i)
      y[i] = h_extra[(std::size_t)t * kDim + i];
    Apply(y, h_ev, extra_k, scale, add);
    for (uint32_t r = 0; r < rows; ++r)
      for (uint32_t i = 0; i < kDim; ++i)
        y[i] += h_x[((std::size_t)t * rows + r) * kDim + i];
    Apply(y, h_tv, total_k, scale, add);
    for (uint32_t b = 0; b < kHc; ++b)
      for (uint32_t i = 0; i < kDim; ++i) {
        double acc = h_post[t * kHc + b] * y[i];
        for (uint32_t c = 0; c < kHc; ++c)
          acc += (double)h_comb[(std::size_t)t * kHc * kHc + c * kHc + b] *
                 h_res[((std::size_t)t * kHc + c) * kDim + i];
        want[((std::size_t)t * kHc + b) * kDim + i] = acc;
      }
  }

  auto* res = Device(h_res);
  auto* extra = Device(h_extra);
  auto* x = Device(h_x);
  auto* post = Device(h_post);
  auto* comb = Device(h_comb);
  auto* ev = Device(h_ev);
  auto* tv = Device(h_tv);
  auto* fused = Device(std::vector<float>(want.size()));
  auto* split = Device(std::vector<float>(want.size()));
  std::vector<float> h_gram(DS41_V41_STEER_GRAM);
  ds41_v41_steer_gram(h_ev.data(), extra_k, h_tv.data(), total_k, kDim,
                      h_gram.data());
  auto* gram = Device(h_gram);
  const ds41_v41_steer s{ev, extra_k, tv, total_k, scale, add ? 1 : 0, gram};
  const bool ok =
      ds41_v41_hc_expand_steer(rows ? x : nullptr, extra, rows, res, post, comb,
                               fused, kHc, kDim, n, &s) == 0 &&
      ds41_v41_steer_rows(extra, rows ? x : nullptr, rows, kDim, n, &s) == 0 &&
      ds41_v41_hc_expand_sum(rows ? x : nullptr, extra, rows, res, post, comb,
                             split, kHc, kDim, n) == 0 &&
      hipDeviceSynchronize() == hipSuccess;
  bool pass = ok;
  if (ok) {
    const double e_fused = RelErr(Host(fused, want.size()), want);
    const double e_split = RelErr(Host(split, want.size()), want);
    pass = e_fused < 2e-6 && e_split < 2e-6;
    std::printf(
        "steer n=%u rows=%u extra_k=%u total_k=%u %s: fused %.2e "
        "in place %.2e %s\n",
        n, rows, extra_k, total_k, add ? "add" : "project", e_fused, e_split,
        pass ? "ok" : "FAIL");
  } else {
    std::printf("steer n=%u rows=%u: launch failed\n", n, rows);
  }
  for (float* p : {res, extra, x, post, comb, ev, tv, gram, fused, split})
    (void)hipFree(p);
  return pass;
}

double RelErrF(const std::vector<float>& got, const std::vector<float>& want) {
  double err = 0.0, ref = 0.0;
  for (std::size_t i = 0; i < want.size(); ++i) {
    err = std::max(err, (double)std::fabs(got[i] - want[i]));
    ref = std::max(ref, (double)std::fabs(want[i]));
  }
  return ref > 0.0 ? err / ref : err;
}

/* The batched state pass with the edits against the in-place edits followed
 * by the plain state pass: the expanded state, mixes and collapse. */
bool StatePassCase(uint32_t n, uint32_t rows, uint32_t extra_k,
                   uint32_t total_k, bool add, std::mt19937& rng) {
  constexpr uint32_t kMix = 24;
  const float eps = 1e-6f, scale = add ? 0.75f : 1.3f;
  const std::size_t state = (std::size_t)n * kHc * kDim;
  const float a = 1.7f / std::sqrt((float)kDim);
  const auto h_ev = Random((std::size_t)(extra_k + 1) * kDim, -a, a, rng);
  const auto h_tv = Random((std::size_t)(total_k + 1) * kDim, -a, a, rng);
  std::vector<float> h_gram(DS41_V41_STEER_GRAM);
  ds41_v41_steer_gram(h_ev.data(), extra_k, h_tv.data(), total_k, kDim,
                      h_gram.data());
  const auto h_y = Random((std::size_t)n * kDim, -2.0f, 2.0f, rng);
  auto* h = Device(Random(state, -2.0f, 2.0f, rng));
  auto* y = Device(h_y);
  auto* y_edit = Device(h_y);
  auto* x =
      Device(Random((std::size_t)n * (rows + 1) * kDim, -2.0f, 2.0f, rng));
  auto* post = Device(Random((std::size_t)n * kHc, 0.0f, 2.0f, rng));
  auto* comb = Device(Random((std::size_t)n * kHc * kHc, 0.0f, 1.0f, rng));
  auto* pre = Device(Random((std::size_t)n * kHc, 0.0f, 1.0f, rng));
  auto* norm_w = Device(Random(kDim, 0.5f, 1.5f, rng));
  auto* fn = Device(Random((std::size_t)kMix * kHc * kDim, -0.05f, 0.05f, rng));
  auto* ev = Device(h_ev);
  auto* tv = Device(h_tv);
  auto* gram = Device(h_gram);
  float* bufs[6];
  const std::size_t sizes[6] = {state,
                                state,
                                (std::size_t)n * kMix,
                                (std::size_t)n * kMix,
                                (std::size_t)n * kDim,
                                (std::size_t)n * kDim};
  for (int i = 0; i < 6; ++i)
    bufs[i] = Device(std::vector<float>(sizes[i]));
  const ds41_v41_steer s{ev, extra_k, tv, total_k, scale, add ? 1 : 0, gram};
  float* xr = rows ? x : nullptr;
  const bool ok = ds41_v41_steer_rows(y_edit, xr, rows, kDim, n, &s) == 0 &&
                  ds41_v41_hc_state_pass(y_edit, xr, rows, post, comb, h,
                                         bufs[0], pre, norm_w, bufs[4], fn,
                                         bufs[2], kHc, kDim, eps, n) == 0 &&
                  ds41_v41_hc_state_pass_steer(
                      y, xr, rows, post, comb, h, bufs[1], pre, norm_w, bufs[5],
                      fn, bufs[3], kHc, kDim, eps, n, &s) == 0 &&
                  hipDeviceSynchronize() == hipSuccess;
  bool pass = ok;
  if (ok) {
    const double e_h = RelErrF(Host(bufs[1], state), Host(bufs[0], state));
    const double e_mix =
        RelErrF(Host(bufs[3], sizes[2]), Host(bufs[2], sizes[2]));
    const double e_xn =
        RelErrF(Host(bufs[5], sizes[4]), Host(bufs[4], sizes[4]));
    pass = e_h < 2e-6 && e_mix < 1e-5 && e_xn < 1e-5;
    std::printf(
        "state pass steer n=%u rows=%u extra_k=%u total_k=%u %s: "
        "state %.2e mixes %.2e xn %.2e %s\n",
        n, rows, extra_k, total_k, add ? "add" : "project", e_h, e_mix, e_xn,
        pass ? "ok" : "FAIL");
  } else {
    std::printf("state pass steer n=%u: launch failed\n", n);
  }
  for (float* p : {h, y, y_edit, x, post, comb, pre, norm_w, fn, ev, tv, gram})
    (void)hipFree(p);
  for (float* p : bufs)
    (void)hipFree(p);
  return pass;
}

/* Microseconds per launch for one decode token: the plain expands against
 * the fused forms at the attention (k = 2) and FFN (shared k = 2, whole
 * k = 1) sites. */
void Bench(std::mt19937& rng) {
  const uint32_t n = 1, rows = 6;
  auto* res = Device(Random(kHc * kDim, -2.0f, 2.0f, rng));
  auto* y = Device(Random(kDim, -2.0f, 2.0f, rng));
  auto* x = Device(Random(rows * kDim, -2.0f, 2.0f, rng));
  auto* post = Device(Random(kHc, 0.0f, 2.0f, rng));
  auto* comb = Device(Random(kHc * kHc, 0.0f, 1.0f, rng));
  auto* v = Device(Random(3 * kDim, -0.02f, 0.02f, rng));
  auto* out = Device(std::vector<float>(kHc * kDim));
  auto* gram = Device(std::vector<float>(DS41_V41_STEER_GRAM));
  const ds41_v41_steer attn{nullptr, 0, v, 2, 1.0f, 0, gram};
  const ds41_v41_steer ffn{v, 2, v + 2 * kDim, 1, 1.0f, 0, gram};
  auto time = [&](const char* name, auto&& launch) {
    for (int i = 0; i < 100; ++i)
      launch();
    (void)hipDeviceSynchronize();
    hipEvent_t a, b;
    (void)hipEventCreate(&a);
    (void)hipEventCreate(&b);
    constexpr int kIters = 2000;
    (void)hipEventRecord(a, nullptr);
    for (int i = 0; i < kIters; ++i)
      launch();
    (void)hipEventRecord(b, nullptr);
    (void)hipEventSynchronize(b);
    float ms = 0.0f;
    (void)hipEventElapsedTime(&ms, a, b);
    std::printf("%-28s %6.2f us\n", name, ms * 1000.0f / kIters);
  };
  time("hc_expand",
       [&] { ds41_v41_hc_expand(y, res, post, comb, out, kHc, kDim, n); });
  time("hc_expand_steer attn k=2", [&] {
    ds41_v41_hc_expand_steer(nullptr, y, 0, res, post, comb, out, kHc, kDim, n,
                             &attn);
  });
  time("hc_expand_sum rows=6", [&] {
    ds41_v41_hc_expand_sum(x, y, rows, res, post, comb, out, kHc, kDim, n);
  });
  time("hc_expand_steer ffn 2+1", [&] {
    ds41_v41_hc_expand_steer(x, y, rows, res, post, comb, out, kHc, kDim, n,
                             &ffn);
  });
  const ds41_v41_steer shexp{v, 2, nullptr, 0, 1.0f, 0, gram};
  const ds41_v41_steer whole{nullptr, 0, v, 1, 1.0f, 0, gram};
  time("hc_expand_steer ffn 2+0", [&] {
    ds41_v41_hc_expand_steer(x, y, rows, res, post, comb, out, kHc, kDim, n,
                             &shexp);
  });
  time("hc_expand_steer ffn 0+1", [&] {
    ds41_v41_hc_expand_steer(x, y, rows, res, post, comb, out, kHc, kDim, n,
                             &whole);
  });
  for (float* p : {res, y, x, post, comb, v, gram, out})
    (void)hipFree(p);
}

}  // namespace

int main(int argc, char** argv) {
  int devices = 0;
  if (hipGetDeviceCount(&devices) != hipSuccess || devices == 0) {
    std::printf("no HIP device\n");
    return 77;
  }
  std::mt19937 rng(25);
  if (argc > 1 && std::string_view(argv[1]) == "--bench") {
    Bench(rng);
    return 0;
  }
  bool ok = true;
  for (uint32_t n : {1u, 5u, 37u}) {
    for (bool add : {false, true}) {
      ok = Case(n, 0, 0, 2, add, rng) && ok;  // attention output
      ok = Case(n, 6, 2, 0, add, rng) && ok;  // shared expert
      ok = Case(n, 6, 0, 1, add, rng) && ok;  // whole FFN
      ok = Case(n, 6, 2, 3, add, rng) && ok;  // both
      ok = Case(n, 6, 4, 4, add, rng) && ok;  // the most vectors per site
    }
  }
  for (uint32_t n : {16u, 37u})
    for (bool add : {false, true}) {
      ok = StatePassCase(n, 0, 0, 2, add, rng) && ok;
      ok = StatePassCase(n, 6, 2, 0, add, rng) && ok;
      ok = StatePassCase(n, 6, 0, 1, add, rng) && ok;
      ok = StatePassCase(n, 6, 4, 4, add, rng) && ok;
    }
  return ok ? 0 : 1;
}
