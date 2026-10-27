/* Prefill-at-depth timing of the V4.1 Flash indexer: WMMA scores of a
 * 2048-token batch against every visible compressed row, then the top-512
 * selection, in the score-row groups the decoder uses (256 MiB of scores).
 *
 * Includes the production translation unit (same flags). Usage:
 *   ds41_indexer_bench [depth=131072] [reps=3] [dump file] [context]
 * The context (default depth + 2048) sets the score row stride, and with it
 * the group size, as a session of that capacity does.
 * The dump holds every group's scores and selections for byte comparison
 * between builds. */
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "src/models/deepseek_v41_flash/kernels/rocm/ds41_v41.hip.cpp"

int main(int argc, char** argv) {
  const uint32_t depth = argc > 1 ? (uint32_t)std::atoi(argv[1]) : 131072u;
  const int reps = argc > 2 ? std::atoi(argv[2]) : 3;
  const char* dump = argc > 3 && argv[3][0] != '-' ? argv[3] : nullptr;
  constexpr uint32_t n = 2048, ratio = 2, heads = 32, dim = 128, top_k = 512;
  const uint32_t context = argc > 4 ? (uint32_t)std::atoi(argv[4]) : depth + n;
  const uint64_t stride = (uint64_t)context + 1;
  const uint32_t group =
      (uint32_t)std::clamp<uint64_t>((256ull << 20) / (stride * 4) / 16 * 16, 16, 4096);
  const uint32_t rows = context / ratio + 1;

  std::mt19937 rng(41);
  std::normal_distribution<float> nd;
  std::vector<float> kh((size_t)rows * dim), qh((size_t)n * heads * dim), wh((size_t)n * heads);
  for (auto& v : kh) v = nd(rng);
  for (auto& v : qh) v = nd(rng);
  for (auto& v : wh) v = nd(rng);
  float *kf, *q, *w, *score;
  uint8_t *codes, *scales;
  uint32_t* topk;
  (void)hipMalloc(&kf, kh.size() * 4);
  (void)hipMalloc(&q, qh.size() * 4);
  (void)hipMalloc(&w, wh.size() * 4);
  (void)hipMalloc(&codes, (size_t)rows * dim / 2);
  (void)hipMalloc(&scales, (size_t)rows * dim / 32);
  (void)hipMalloc(&score, (size_t)group * stride * 4);
  (void)hipMalloc(&topk, (size_t)n * top_k * 4);
  (void)hipMemcpy(kf, kh.data(), kh.size() * 4, hipMemcpyHostToDevice);
  (void)hipMemcpy(q, qh.data(), qh.size() * 4, hipMemcpyHostToDevice);
  (void)hipMemcpy(w, wh.data(), wh.size() * 4, hipMemcpyHostToDevice);
  if (ds41_v41_fp4_store(kf, codes, scales, rows * dim, 32, 0) || ds41_v41_fp4_fake_quant(q, n * heads * dim, 32, 0))
    return 1;
  const ds41_fp4_rows keys{codes, scales};
  const float w_scale = 1.0f / sqrtf((float)dim) / sqrtf((float)heads);

  FILE* out = dump ? std::fopen(dump, "wb") : nullptr;
  std::vector<float> sh;
  std::vector<uint32_t> th;
  hipEvent_t ev[4];
  for (auto& e : ev) (void)hipEventCreate(&e);
  double best_score = 1e30, best_topk = 1e30;
  for (int rep = 0; rep <= reps; rep++) {
    float score_ms = 0, topk_ms = 0;
    for (uint32_t t0 = 0; t0 < n; t0 += group) {
      const uint32_t m = std::min(group, n - t0);
      const uint32_t p0 = depth + t0;
      (void)hipEventRecord(ev[0]);
      if (ds41_v41_indexer_score(q + (uint64_t)t0 * heads * dim, keys, w + (uint64_t)t0 * heads, w_scale, score,
                                 stride, p0, m, ratio, heads, dim))
        return 1;
      (void)hipEventRecord(ev[1]);
      if (ds41_v41_topk(score, stride, topk + (uint64_t)t0 * top_k, top_k, p0, m, ratio, 1)) return 1;
      (void)hipEventRecord(ev[2]);
      (void)hipEventSynchronize(ev[2]);
      float a = 0, b = 0;
      (void)hipEventElapsedTime(&a, ev[0], ev[1]);
      (void)hipEventElapsedTime(&b, ev[1], ev[2]);
      score_ms += a;
      topk_ms += b;
      if (out && rep == 0) {
        const uint32_t len = (p0 + m) / ratio;
        sh.resize((size_t)m * stride);
        (void)hipMemcpy(sh.data(), score, sh.size() * 4, hipMemcpyDeviceToHost);
        for (uint32_t r = 0; r < m; r++) std::fwrite(&sh[(size_t)r * stride], 4, std::min<uint64_t>(len, stride), out);
      }
    }
    if (rep > 0) {
      best_score = std::min<double>(best_score, score_ms);
      best_topk = std::min<double>(best_topk, topk_ms);
    }
    if (out && rep == 0) {
      th.resize((size_t)n * top_k);
      (void)hipMemcpy(th.data(), topk, th.size() * 4, hipMemcpyDeviceToHost);
      std::fwrite(th.data(), 4, th.size(), out);
      std::fclose(out);
      out = nullptr;
    }
  }
  const double flops = 2.0 * heads * dim * n * (depth / ratio + n / ratio / 2.0);
  std::printf("depth %u group %u: score %.2f ms (%.1f TFLOP/s), topk %.2f ms\n", depth, group, best_score,
              flops / best_score / 1e9, best_topk);
  return 0;
}
