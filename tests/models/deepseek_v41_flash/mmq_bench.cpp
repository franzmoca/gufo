/* Routed-expert MMQ timing at the model's shapes: 384 experts of one weight
 * type (gate/up 2304 x 5120, down 5120 x 2304), 2048 tokens with six slots
 * each, per-expert tile widths as in the decoder. The expert row counts come
 * from a "COUNTS <tokens> <count per expert>..." line (as recorded from a
 * prefill), or are drawn at random. Weights are random blocks, larger than
 * the caches, so the reads stay DRAM-cold.
 *
 * With --f16 the same passes run through the FP16 matrix-core kernel
 * (IQ1_S, IQ1_M, IQ2_XXS). Either path is checked against the
 * FP32-activation expert GEMV on the first slots. --tokens sets the batch
 * (2048 by default).
 *
 *   ds41_mmq_bench [--type 19|29|16|22] [--counts FILE] [--line N] [--reps N]
 *                  [--f16 1] [--tokens N]
 *   ds41_mmq_bench --type 13|8 --dense M,K[,GROUPS] [--tokens N] [--reps N]
 */
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "src/models/deepseek_v41_flash/kernels/rocm/ds41_v41.h"
#include "src/models/deepseek_v41_flash/kernels/rocm/mmq/ds41_mmq.h"

namespace {

constexpr int kExperts = 384, kUsed = 6, kFf = 2304, kDim = 5120;

int BlockBytes(int type) {
  switch (type) {
    case 19:
      return 50;
    case 29:
      return 56;
    case 16:
      return 66;
    case 22:
      return 82;
    default:
      return 0;
  }
}

/* Random weight blocks with small finite f16 scales: a 64 MiB random pattern
 * repeated over the allocation. */
void* RandomWeights(int type, std::size_t bytes, std::mt19937& rng) {
  const int block = BlockBytes(type);
  std::vector<uint8_t> pattern((64u << 20) / block * block);
  for (auto& b : pattern)
    b = static_cast<uint8_t>(rng());
  for (std::size_t at = 0; at < pattern.size(); at += block) {
    const uint16_t d = 0x2000 | (rng() & 0x03ff);  // ~2^-7
    if (type == 29) {
      for (int i = 0; i < 4; ++i) {
        uint16_t word;
        std::memcpy(&word, &pattern[at + 48 + 2 * i], 2);
        word = static_cast<uint16_t>((word & 0x0fff) | ((d >> (4 * i)) & 0xf)
                                                           << 12);
        std::memcpy(&pattern[at + 48 + 2 * i], &word, 2);
      }
    } else {
      std::memcpy(&pattern[at], &d, 2);
    }
  }
  void* p = nullptr;
  if (hipMalloc(&p, bytes) != hipSuccess)
    return nullptr;
  for (std::size_t at = 0; at < bytes; at += pattern.size())
    (void)hipMemcpy(static_cast<char*>(p) + at, pattern.data(),
                    std::min(pattern.size(), bytes - at),
                    hipMemcpyHostToDevice);
  return p;
}

std::vector<unsigned int> Counts(const char* path, int line, int tokens,
                                 std::mt19937& rng) {
  std::vector<unsigned int> counts(kExperts, 0u);
  if (path) {
    std::ifstream in(path);
    std::string text;
    for (int i = 0; i <= line && std::getline(in, text); ++i) {
    }
    std::istringstream fields(text);
    std::string tag;
    int n = 0;
    fields >> tag >> n;
    for (auto& c : counts)
      fields >> c;
    if (tag == "COUNTS" && n == tokens)
      return counts;
    std::fprintf(stderr, "bad counts line %d in %s\n", line, path);
    std::exit(1);
  }
  std::uniform_int_distribution<int> pick(0, kExperts - 1);
  for (int i = 0; i < tokens * kUsed; ++i)
    counts[pick(rng)]++;
  return counts;
}

/* Dense projection of `kTokens` tokens (2048 by default), MMQ against the FP16
 * kernel, over `groups` [m, k] matrices with their own input columns (Q5_K or
 * Q8_0). */
int DenseBench(int type, int m, int k, int groups, int reps, int kTokens) {
  const int block = type == 13 ? 176 : 34, values = type == 13 ? 256 : 32;
  std::mt19937 rng(7);
  std::vector<uint8_t> w(static_cast<std::size_t>(groups) * m * k / values *
                         block);
  for (std::size_t at = 0; at < w.size(); ++at)
    w[at] = static_cast<uint8_t>(rng());
  for (std::size_t at = 0; at < w.size(); at += block) {
    const uint16_t d = 0x2000 | (rng() & 0x03ff);
    std::memcpy(&w[at], &d, 2);
    if (type == 13)
      std::memcpy(&w[at + 2], &d, 2);
  }
  std::vector<float> x(static_cast<std::size_t>(kTokens) * groups * k);
  std::normal_distribution<float> normal;
  for (auto& v : x)
    v = normal(rng);
  void *dw = nullptr, *dx = nullptr, *out = nullptr;
  (void)hipMalloc(&dw, w.size());
  (void)hipMalloc(&dx, x.size() * 4);
  (void)hipMalloc(&out, static_cast<std::size_t>(kTokens) * groups * m * 4);
  (void)hipMemcpy(dw, w.data(), w.size(), hipMemcpyHostToDevice);
  (void)hipMemcpy(dx, x.data(), x.size() * 4, hipMemcpyHostToDevice);
  hipEvent_t e0, e1;
  (void)hipEventCreate(&e0);
  (void)hipEventCreate(&e1);
  for (int path = 0; path < 2; ++path) {
    std::vector<float> ms;
    for (int r = 0; r < reps + 1; ++r) {
      (void)hipEventRecord(e0, 0);
      int rc;
      if (path == 1)
        rc =
            ds41_v41_dense_f16(type, dw, static_cast<float*>(dx),
                               static_cast<float*>(out), m, kTokens, k, groups);
      else if (groups > 1)
        rc = ds41_mmq_dense_grouped_typed(type, dw, static_cast<float*>(dx),
                                          static_cast<float*>(out), m, kTokens,
                                          k, groups, 0);
      else
        rc = ds41_mmq_dense_typed(type, dw, static_cast<float*>(dx),
                                  static_cast<float*>(out), m, kTokens, k, 0);
      (void)hipEventRecord(e1, 0);
      (void)hipEventSynchronize(e1);
      if (rc) {
        std::printf("%s: launch failed\n", path ? "f16" : "mmq");
        break;
      }
      float t = 0.0f;
      (void)hipEventElapsedTime(&t, e0, e1);
      if (r > 0)
        ms.push_back(t);
    }
    if (ms.empty())
      continue;
    std::sort(ms.begin(), ms.end());
    const float t = ms[ms.size() / 2];
    std::printf("dense type %d m %d k %d groups %d %s: %.3f ms (%.1f TOPS)\n",
                type, m, k, groups, path ? "f16" : "mmq", t,
                2.0 * kTokens * m * k * groups / t / 1e9);
  }
  ds41_mmq_cleanup();
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  int type = 19, line = 0, reps = 5, f16 = 0;
  int dense_m = 0, dense_k = 0, dense_groups = 1, dense_tokens = 2048;
  const char* counts_path = nullptr;
  for (int i = 1; i + 1 < argc; i += 2) {
    if (!std::strcmp(argv[i], "--type"))
      type = std::atoi(argv[i + 1]);
    else if (!std::strcmp(argv[i], "--counts"))
      counts_path = argv[i + 1];
    else if (!std::strcmp(argv[i], "--line"))
      line = std::atoi(argv[i + 1]);
    else if (!std::strcmp(argv[i], "--reps"))
      reps = std::atoi(argv[i + 1]);
    else if (!std::strcmp(argv[i], "--f16"))
      f16 = std::atoi(argv[i + 1]);
    else if (!std::strcmp(argv[i], "--tokens"))
      dense_tokens = std::atoi(argv[i + 1]);
    else if (!std::strcmp(argv[i], "--dense"))
      std::sscanf(argv[i + 1], "%d,%d,%d", &dense_m, &dense_k, &dense_groups);
  }
  if (dense_m > 0)
    return DenseBench(type, dense_m, dense_k, dense_groups, reps, dense_tokens);
  if (!BlockBytes(type)) {
    std::fprintf(stderr, "unsupported type %d\n", type);
    return 1;
  }
  /* With --counts, --tokens N scales the recorded 2048-token line by
   * N / 2048. */
  const int tokens = dense_tokens;
  std::mt19937 rng(41);
  auto counts = Counts(counts_path, line, counts_path ? 2048 : tokens, rng);
  if (counts_path)
    for (auto& c : counts)
      c = c * (unsigned)tokens / 2048u;
  /* Slot ids with exactly these counts, spread over the tokens. */
  std::vector<int32_t> ids;
  for (int e = 0; e < kExperts; ++e)
    ids.insert(ids.end(), counts[e], e);
  std::shuffle(ids.begin(), ids.end(), rng);
  /* A token's experts are distinct, as top-k routing makes them: swap
   * repeats with slots of other tokens until none is left. */
  const auto repeats = [&](std::size_t slot, int32_t e) {
    const std::size_t t = slot / kUsed;
    for (std::size_t s = t * kUsed; s < t * kUsed + kUsed; ++s)
      if (s != slot && ids[s] == e)
        return true;
    return false;
  };
  std::uniform_int_distribution<std::size_t> any(0, ids.size() - 1);
  for (std::size_t s = 0; s < ids.size(); ++s)
    while (repeats(s, ids[s])) {
      const std::size_t o = any(rng);
      if (o / kUsed == s / kUsed || repeats(o, ids[s]) || repeats(s, ids[o]))
        continue;
      std::swap(ids[s], ids[o]);
    }
  unsigned int most = *std::max_element(counts.begin(), counts.end());

  uint16_t *x16 = nullptr, *mid16 = nullptr;
  (void)hipMalloc(&x16, static_cast<std::size_t>(tokens) * kDim * 2);
  (void)hipMalloc(&mid16, static_cast<std::size_t>(tokens) * kUsed * kFf * 2);
  const std::size_t expert_bytes =
      static_cast<std::size_t>(kFf) * kDim / 256 * BlockBytes(type);
  void* gate = RandomWeights(type, expert_bytes * kExperts, rng);
  void* up = RandomWeights(type, expert_bytes * kExperts, rng);
  void* down = RandomWeights(type, expert_bytes * kExperts, rng);
  std::normal_distribution<float> normal;
  std::vector<float> x(static_cast<std::size_t>(tokens) * kDim),
      w(static_cast<std::size_t>(tokens) * kUsed, 0.25f);
  for (auto& v : x)
    v = normal(rng);
  float *dx = nullptr, *dw = nullptr, *g = nullptr, *mid = nullptr,
        *out = nullptr;
  int32_t* dids = nullptr;
  uint8_t* dclass = nullptr;
  const std::size_t rows = ids.size();
  (void)hipMalloc(&dx, x.size() * sizeof(float));
  (void)hipMalloc(&dw, w.size() * sizeof(float));
  (void)hipMalloc(&dids, rows * sizeof(int32_t));
  (void)hipMalloc(&dclass, kExperts);
  (void)hipMalloc(&g, rows * kFf * sizeof(float));
  (void)hipMalloc(&mid, rows * kFf * sizeof(float));
  (void)hipMalloc(&out, rows * kDim * sizeof(float));
  if (!gate || !up || !down || !dx || !dw || !dids || !dclass || !g || !mid ||
      !out) {
    std::fprintf(stderr, "allocation failed\n");
    return 1;
  }
  (void)hipMemcpy(dx, x.data(), x.size() * sizeof(float),
                  hipMemcpyHostToDevice);
  (void)hipMemcpy(dw, w.data(), w.size() * sizeof(float),
                  hipMemcpyHostToDevice);
  (void)hipMemcpy(dids, ids.data(), rows * sizeof(int32_t),
                  hipMemcpyHostToDevice);
  uint8_t classes[kExperts];
  int widths[3], class_rows[3];
  const int n_classes = ds41_mmq_routed_classes_for_counts(
      counts.data(), kExperts, classes, widths, class_rows);
  (void)hipMemcpy(dclass, classes, kExperts, hipMemcpyHostToDevice);

  std::vector<int32_t> plan_rows(rows), bounds(kExperts + 1),
      tiles(rows / 64 + kExperts);
  const uint32_t n_tiles =
      ds41_v41_routed_plan(ids.data(), static_cast<uint32_t>(rows), kExperts,
                           plan_rows.data(), bounds.data(), tiles.data());
  int32_t *drows = nullptr, *dbounds = nullptr, *dtiles = nullptr;
  (void)hipMalloc(&drows, rows * sizeof(int32_t));
  (void)hipMalloc(&dbounds, bounds.size() * sizeof(int32_t));
  (void)hipMalloc(&dtiles, tiles.size() * sizeof(int32_t));
  (void)hipMemcpy(drows, plan_rows.data(), rows * sizeof(int32_t),
                  hipMemcpyHostToDevice);
  (void)hipMemcpy(dbounds, bounds.data(), bounds.size() * sizeof(int32_t),
                  hipMemcpyHostToDevice);
  (void)hipMemcpy(dtiles, tiles.data(), tiles.size() * sizeof(int32_t),
                  hipMemcpyHostToDevice);

  hipEvent_t e0, e1, e2;
  (void)hipEventCreate(&e0);
  (void)hipEventCreate(&e1);
  (void)hipEventCreate(&e2);
  std::vector<float> gu_ms, down_ms;
  for (int r = 0; r < reps + 1; ++r) {
    ds41_mmq_set_routed_max_expert_rows(static_cast<int>(most));
    ds41_mmq_set_routed_tile_cols(
        ds41_mmq_routed_tile_cols_for_counts(counts.data(), kExperts));
    ds41_mmq_set_routed_classes(dclass, n_classes, widths, class_rows);
    (void)hipEventRecord(e0, 0);
    int rc = 0;
    if (f16)
      rc = ds41_v41_to_half(dx, x16, static_cast<uint64_t>(tokens) * kDim) |
           ds41_v41_routed_f16(type, gate, up, expert_bytes, x16, kUsed, drows,
                               dbounds, dtiles, n_tiles, dw, 10.0f, mid16, 1,
                               kFf, kDim);
    else
      rc = ds41_mmq_moe_swiglu_typed(type, gate, up, dx, dids, dw, g, mid, kFf,
                                     kDim, tokens, kExperts, kUsed, 10.0f, 0);
    (void)hipEventRecord(e1, 0);
    if (f16)
      rc |= ds41_v41_routed_f16(type, down, nullptr, expert_bytes, mid16, 1,
                                drows, dbounds, dtiles, n_tiles, nullptr, 0.0f,
                                out, 0, kDim, kFf);
    else
      rc |= ds41_mmq_moe_typed(type, down, mid, dids, out, kDim, kFf,
                               static_cast<int>(rows), kExperts, 1, 0);
    (void)hipEventRecord(e2, 0);
    (void)hipEventSynchronize(e2);
    if (rc) {
      std::fprintf(stderr, "launch failed\n");
      return 1;
    }
    float a = 0.0f, b = 0.0f;
    (void)hipEventElapsedTime(&a, e0, e1);
    (void)hipEventElapsedTime(&b, e1, e2);
    if (r > 0) {
      gu_ms.push_back(a);
      down_ms.push_back(b);
    }
  }
  ds41_mmq_set_routed_classes(nullptr, 0, nullptr, nullptr);
  ds41_mmq_set_routed_tile_cols(0);
  ds41_mmq_set_routed_max_expert_rows(0);
  std::sort(gu_ms.begin(), gu_ms.end());
  std::sort(down_ms.begin(), down_ms.end());
  const double flops = 2.0 * rows * kFf * kDim;
  const float gu = gu_ms[gu_ms.size() / 2], dn = down_ms[down_ms.size() / 2];
  std::printf(
      "type %d tokens %d rows %zu max %u classes %d/%d/%d: gate+up %.2f ms "
      "(%.1f TOPS) down %.2f ms (%.1f TOPS)\n",
      type, tokens, rows, most, class_rows[0], class_rows[1], class_rows[2], gu,
      2 * flops / gu / 1e9, dn, flops / dn / 1e9);
  {
    /* Reference on the first slots: FP32-activation GEMV of each slot. */
    constexpr uint32_t kCheck = 384;
    std::vector<float> xr(static_cast<std::size_t>(kCheck) * kDim);
    for (uint32_t s = 0; s < kCheck; ++s)
      std::memcpy(&xr[static_cast<std::size_t>(s) * kDim],
                  &x[static_cast<std::size_t>(s / kUsed) * kDim],
                  kDim * sizeof(float));
    if (f16) {
      std::vector<_Float16> h(rows * kFf);
      std::vector<float> f(rows * kFf);
      (void)hipMemcpy(h.data(), mid16, h.size() * 2, hipMemcpyDeviceToHost);
      for (std::size_t i = 0; i < h.size(); ++i)
        f[i] = static_cast<float>(h[i]);
      (void)hipMemcpy(mid, f.data(), f.size() * 4, hipMemcpyHostToDevice);
    }
    float *dxr = nullptr, *rg = nullptr, *ru = nullptr, *rd = nullptr;
    (void)hipMalloc(&dxr, xr.size() * sizeof(float));
    (void)hipMalloc(&rg, kCheck * kFf * sizeof(float));
    (void)hipMalloc(&ru, kCheck * kFf * sizeof(float));
    (void)hipMalloc(&rd, kCheck * kDim * sizeof(float));
    (void)hipMemcpy(dxr, xr.data(), xr.size() * sizeof(float),
                    hipMemcpyHostToDevice);
    int rc = ds41_v41_gemv_experts(type, gate, expert_bytes, dids, kCheck, dxr,
                                   kDim, rg, kFf, kDim);
    rc |= ds41_v41_gemv_experts(type, up, expert_bytes, dids, kCheck, dxr, kDim,
                                ru, kFf, kDim);
    rc |= ds41_v41_gemv_experts(type, down, expert_bytes, dids, kCheck, mid,
                                kFf, rd, kDim, kFf);
    std::vector<float> hg(kCheck * kFf), hu(kCheck * kFf), hd(kCheck * kDim),
        hm(kCheck * kFf), ho(kCheck * kDim);
    (void)hipMemcpy(hg.data(), rg, hg.size() * 4, hipMemcpyDeviceToHost);
    (void)hipMemcpy(hu.data(), ru, hu.size() * 4, hipMemcpyDeviceToHost);
    (void)hipMemcpy(hd.data(), rd, hd.size() * 4, hipMemcpyDeviceToHost);
    (void)hipMemcpy(hm.data(), mid, hm.size() * 4, hipMemcpyDeviceToHost);
    (void)hipMemcpy(ho.data(), out, ho.size() * 4, hipMemcpyDeviceToHost);
    double em = 0, mm = 0, ed = 0, md = 0;
    for (std::size_t i = 0; i < hm.size(); ++i) {
      float ga = std::min(hg[i], 10.0f);
      const float ua = std::min(std::max(hu[i], -10.0f), 10.0f);
      const float ref = ga / (1.0f + std::exp(-ga)) * ua * 0.25f;
      em = std::max(em, static_cast<double>(std::fabs(ref - hm[i])));
      mm = std::max(mm, static_cast<double>(std::fabs(ref)));
    }
    for (std::size_t i = 0; i < hd.size(); ++i) {
      ed = std::max(ed, static_cast<double>(std::fabs(hd[i] - ho[i])));
      md = std::max(md, static_cast<double>(std::fabs(hd[i])));
    }
    std::printf(
        "check %u slots: mid max err %.3g of %.3g (%.2g), down %.3g of "
        "%.3g (%.2g)%s\n",
        kCheck, em, mm, em / mm, ed, md, ed / md,
        rc ? " (reference failed)" : "");
  }
  ds41_mmq_cleanup();
  return 0;
}
