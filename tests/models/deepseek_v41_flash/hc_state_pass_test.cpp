// The batched mHC state pass against the separate kernels it fuses: the
// expand (or the expand of a sum of rows, or the plain state), the mixes over
// it and the collapse with its RMS norm, on random data with an odd token
// count. The small-batch mix with Sinkhorn (several tokens in one launch)
// must match the one-token launch bit for bit.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "ds41_v41.h"

namespace {

constexpr uint32_t kHc = 4, kDim = 5120, kMix = 24;

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

/* Largest |got - want| relative to the largest |want|. */
double RelErr(const std::vector<float>& got, const std::vector<float>& want) {
  double err = 0.0, ref = 0.0;
  for (std::size_t i = 0; i < want.size(); ++i) {
    err = std::max(err, (double)std::fabs(got[i] - want[i]));
    ref = std::max(ref, (double)std::fabs(want[i]));
  }
  return ref > 0.0 ? err / ref : err;
}

/* rows < 0: the plain state; otherwise expanded from y plus `rows` rows. */
bool Case(uint32_t n, int rows, std::mt19937& rng) {
  const bool expand = rows >= 0;
  const uint32_t r = expand ? (uint32_t)rows : 0;
  const float eps = 1e-6f;
  const std::size_t state = (std::size_t)n * kHc * kDim;
  auto* h = Device(Random(state, -2.0f, 2.0f, rng));
  auto* y = Device(Random((std::size_t)n * kDim, -2.0f, 2.0f, rng));
  auto* x = Device(Random((std::size_t)n * (r + 1) * kDim, -2.0f, 2.0f, rng));
  auto* post = Device(Random((std::size_t)n * kHc, 0.0f, 2.0f, rng));
  auto* comb = Device(Random((std::size_t)n * kHc * kHc, 0.0f, 1.0f, rng));
  auto* pre = Device(Random((std::size_t)n * kHc, 0.0f, 1.0f, rng));
  auto* norm_w = Device(Random(kDim, 0.5f, 1.5f, rng));
  auto* fn = Device(Random((std::size_t)kMix * kHc * kDim, -0.05f, 0.05f, rng));
  std::vector<float> zeros(state, 0.0f);
  float* want_h = Device(zeros);
  float* got_h = Device(zeros);
  float* want_mix = Device(std::vector<float>((std::size_t)n * kMix));
  float* got_mix = Device(std::vector<float>((std::size_t)n * kMix));
  float* want_xn = Device(std::vector<float>((std::size_t)n * kDim));
  float* got_xn = Device(std::vector<float>((std::size_t)n * kDim));
  const float* src = h;
  bool ok = true;
  if (expand) {
    ok = ds41_v41_hc_expand_sum(x, y, r, h, post, comb, want_h, kHc, kDim, n) ==
         0;
    src = want_h;
  }
  ok = ok && ds41_v41_hc_mixes(fn, src, want_mix, kHc, kDim, eps, n) == 0 &&
       ds41_v41_hc_collapse_norm(src, pre, norm_w, want_xn, kHc, kDim, eps,
                                 n) == 0 &&
       ds41_v41_hc_state_pass(expand ? y : nullptr, x, r, post, comb, h, got_h,
                              pre, norm_w, got_xn, fn, got_mix, kHc, kDim, eps,
                              n) == 0 &&
       hipDeviceSynchronize() == hipSuccess;
  if (!ok) {
    std::printf("hc state pass n=%u rows=%d: launch failed\n", n, rows);
    return false;
  }
  const double e_mix = RelErr(Host(got_mix, (std::size_t)n * kMix),
                              Host(want_mix, (std::size_t)n * kMix));
  const double e_xn = RelErr(Host(got_xn, (std::size_t)n * kDim),
                             Host(want_xn, (std::size_t)n * kDim));
  const double e_h =
      expand ? RelErr(Host(got_h, state), Host(want_h, state)) : 0.0;
  const bool pass = e_mix < 1e-5 && e_xn < 1e-5 && e_h < 1e-6;
  std::printf("hc state pass n=%u rows=%d mixes %.2e xn %.2e state %.2e %s\n",
              n, rows, e_mix, e_xn, e_h, pass ? "ok" : "FAIL");
  for (float* p : {h, y, x, post, comb, pre, norm_w, fn, want_h, got_h,
                   want_mix, got_mix, want_xn, got_xn})
    (void)hipFree(p);
  return pass;
}

/* Tokens in one launch against one launch per token: identical mixes, pre,
 * post and comb. */
bool MixCase(uint32_t n, std::mt19937& rng) {
  const float eps = 1e-6f, hc_eps = 1e-6f;
  const uint32_t iters = 20;
  auto* h = Device(Random((std::size_t)n * kHc * kDim, -2.0f, 2.0f, rng));
  auto* fn = Device(Random((std::size_t)kMix * kHc * kDim, -0.05f, 0.05f, rng));
  auto* scale = Device(Random(3, 0.5f, 1.5f, rng));
  auto* base = Device(Random(kMix, -0.5f, 0.5f, rng));
  const std::size_t sizes[4] = {(std::size_t)n * kMix, (std::size_t)n * kHc,
                                (std::size_t)n * kHc,
                                (std::size_t)n * kHc * kHc};
  float *want[4], *got[4];
  for (int i = 0; i < 4; i++) {
    want[i] = Device(std::vector<float>(sizes[i]));
    got[i] = Device(std::vector<float>(sizes[i]));
  }
  bool ok = true;
  for (uint32_t t = 0; t < n && ok; t++)
    ok = ds41_v41_hc_mix(fn, h + (std::size_t)t * kHc * kDim, scale, base,
                         want[0] + t * kMix, want[1] + t * kHc,
                         want[2] + t * kHc, want[3] + t * kHc * kHc, kHc, kDim,
                         eps, iters, hc_eps, 1, nullptr) == 0;
  ok = ok &&
       ds41_v41_hc_mix(fn, h, scale, base, got[0], got[1], got[2], got[3], kHc,
                       kDim, eps, iters, hc_eps, n, nullptr) == 0 &&
       hipDeviceSynchronize() == hipSuccess;
  bool same = ok;
  for (int i = 0; i < 4 && ok; i++) {
    const auto a = Host(want[i], sizes[i]), b = Host(got[i], sizes[i]);
    same =
        same && std::memcmp(a.data(), b.data(), sizes[i] * sizeof(float)) == 0;
  }
  std::printf("hc mix n=%u against one token per launch: %s\n", n,
              !ok    ? "launch failed"
              : same ? "identical"
                     : "FAIL");
  for (float* p : {h, fn, scale, base})
    (void)hipFree(p);
  for (int i = 0; i < 4; i++) {
    (void)hipFree(want[i]);
    (void)hipFree(got[i]);
  }
  return ok && same;
}

}  // namespace

int main() {
  int devices = 0;
  if (hipGetDeviceCount(&devices) != hipSuccess || devices == 0) {
    std::printf("no HIP device\n");
    return 77;
  }
  std::mt19937 rng(41);
  bool ok = true;
  for (uint32_t n : {16u, 37u})
    for (int rows : {-1, 0, 6})
      ok = Case(n, rows, rng) && ok;
  for (uint32_t n : {2u, 5u, 6u, 9u, 16u})
    ok = MixCase(n, rng) && ok;
  return ok ? 0 : 1;
}
