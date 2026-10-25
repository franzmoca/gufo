// Batched (matrix-core) and split decode attention against a double
// precision reference of the plain softmax-with-sink over window and
// compressed rows.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

#include "ds41_v41.h"

namespace {

constexpr uint32_t kHeads = 64;
constexpr uint32_t kDim = 512;
constexpr uint32_t kWindow = 128;
constexpr uint32_t kTopK = 512;

template<typename T>
T* Device(const std::vector<T>& host) {
  void* p = nullptr;
  if (hipMalloc(&p, std::max<std::size_t>(host.size(), 1) * sizeof(T)) !=
      hipSuccess)
    return nullptr;
  (void)hipMemcpy(p, host.data(), host.size() * sizeof(T),
                  hipMemcpyHostToDevice);
  return static_cast<T*>(p);
}

/* n tokens at pos0 with `comp_rows` compressed rows at the given ratio;
 * returns the worst relative RMS error over tokens. */
double Case(uint32_t pos0, uint32_t n, uint32_t ratio, uint32_t comp_rows,
            std::mt19937& rng) {
  std::normal_distribution<float> normal;
  auto fill = [&](std::size_t count, float scale) {
    std::vector<float> v(count);
    for (auto& x : v)
      x = normal(rng) * scale;
    return v;
  };
  const auto q = fill((std::size_t)n * kHeads * kDim, 0.08f);
  const auto ring = fill((std::size_t)kWindow * kDim, 1.0f);
  const auto chunk = fill((std::size_t)n * kDim, 1.0f);
  const auto comp = fill((std::size_t)std::max(comp_rows, 1u) * kDim, 1.0f);
  const auto sinks = fill(kHeads, 1.0f);
  std::vector<uint32_t> idx((std::size_t)n * kTopK, 0);
  for (uint32_t t = 0; t < n; t++) {
    const uint32_t len =
        ratio ? std::min(comp_rows, (pos0 + t + 1) / ratio) : 0;
    std::vector<uint32_t> rows(len);
    for (uint32_t j = 0; j < len; j++)
      rows[j] = j;
    std::shuffle(rows.begin(), rows.end(), rng);
    for (uint32_t j = 0; j < std::min(len, kTopK); j++)
      idx[(std::size_t)t * kTopK + j] = rows[j];
  }
  auto* dq = Device(q);
  auto* dring = Device(ring);
  auto* dchunk = Device(chunk);
  auto* dcomp = Device(comp);
  auto* dsinks = Device(sinks);
  auto* didx = Device(idx);
  std::vector<float> partial_host((std::size_t)n * kHeads *
                                  DS41_V41_ATTN_PARTIAL_PER_HEAD);
  auto* dpartial = Device(partial_host);
  float* dout = nullptr;
  (void)hipMalloc(&dout, sizeof(float) * n * kHeads * kDim);
  const float scale = 1.0f / std::sqrt((float)kDim);
  const int rc =
      ds41_v41_attention(dq, dring, dchunk, kWindow, dcomp, didx, kTopK, ratio,
                         dsinks, dout, dpartial, pos0, n, kHeads, kDim, scale);
  std::vector<float> out((std::size_t)n * kHeads * kDim);
  (void)hipDeviceSynchronize();
  (void)hipMemcpy(out.data(), dout, out.size() * sizeof(float),
                  hipMemcpyDeviceToHost);
  for (void* p : {(void*)dq, (void*)dring, (void*)dchunk, (void*)dcomp,
                  (void*)dsinks, (void*)didx, (void*)dpartial, (void*)dout})
    (void)hipFree(p);
  if (rc != 0)
    return INFINITY;

  double worst = 0.0;
  for (uint32_t t = 0; t < n; t++) {
    const uint32_t p = pos0 + t;
    const uint32_t lo = p + 1 > kWindow ? p + 1 - kWindow : 0;
    std::vector<const float*> keys;
    for (uint32_t pos = lo; pos <= p; pos++)
      keys.push_back(pos < pos0 ? &ring[(std::size_t)(pos % kWindow) * kDim]
                                : &chunk[(std::size_t)(pos - pos0) * kDim]);
    const uint32_t len =
        ratio ? std::min({comp_rows, (p + 1) / ratio, kTopK}) : 0;
    for (uint32_t j = 0; j < len; j++)
      keys.push_back(
          &comp[(std::size_t)idx[(std::size_t)t * kTopK + j] * kDim]);
    double err = 0.0, ref = 0.0;
    for (uint32_t h = 0; h < kHeads; h++) {
      const float* qh = &q[((std::size_t)t * kHeads + h) * kDim];
      std::vector<double> s(keys.size());
      double m = sinks[h];
      for (std::size_t j = 0; j < keys.size(); j++) {
        double dot = 0.0;
        for (uint32_t d = 0; d < kDim; d++)
          dot += (double)qh[d] * keys[j][d];
        s[j] = dot * scale;
        m = std::max(m, s[j]);
      }
      double den = std::exp(sinks[h] - m);
      for (auto& v : s)
        den += (v = std::exp(v - m));
      for (uint32_t d = 0; d < kDim; d++) {
        double want = 0.0;
        for (std::size_t j = 0; j < keys.size(); j++)
          want += s[j] * keys[j][d];
        want /= den;
        const double got = out[((std::size_t)t * kHeads + h) * kDim + d];
        if (!std::isfinite(got))
          return INFINITY;
        err += (got - want) * (got - want);
        ref += want * want;
      }
    }
    worst = std::max(worst, std::sqrt(err / std::max(ref, 1e-30)));
  }
  return worst;
}

}  // namespace

int main() {
  int devices = 0;
  if (hipGetDeviceCount(&devices) != hipSuccess || devices == 0) {
    std::printf("no HIP device\n");
    return 77;
  }
  struct Shape {
    uint32_t pos0, n, ratio, comp_rows;
    double tolerance;  // split decode is fp32; the batched form stages half
                       // operands
  };
  const Shape shapes[] = {
      {12, 1, 0, 0, 1e-5},      {12, 1, 2, 6, 1e-5},
      {700, 1, 2, 350, 1e-5},   {2047, 1, 1, 2048, 1e-5},
      {3000, 3, 1, 3003, 1e-5}, {0, 37, 2, 18, 2e-3},
      {200, 64, 1, 264, 2e-3},  {2048, 40, 1, 2088, 2e-3},
  };
  std::mt19937 rng(41);
  bool ok = true;
  for (const Shape& s : shapes) {
    const double e = Case(s.pos0, s.n, s.ratio, s.comp_rows, rng);
    std::printf("pos0=%u n=%u ratio=%u comp=%u rel_err=%.2e\n", s.pos0, s.n,
                s.ratio, s.comp_rows, e);
    ok = e < s.tolerance && ok;
  }
  std::printf("%s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
