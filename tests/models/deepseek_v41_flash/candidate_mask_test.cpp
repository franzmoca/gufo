// Candidate-block mask and its application against a host port of the
// reference: per token, max over `block`-row blocks of the source layer's
// indexer scores, newest block forced in, top `topk` blocks kept; masked rows
// score -inf in the later layers. Also the indexer top-k with tied scores.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <numeric>
#include <random>
#include <vector>

#include "ds41_v41.h"

namespace {

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

/* n tokens at pos0; returns the number of mismatched mask or score entries. */
std::size_t Case(uint32_t pos0, uint32_t n, uint32_t ratio, uint32_t block,
                 uint32_t topk, std::mt19937& rng) {
  const uint32_t max_len = (pos0 + n) / ratio;
  const uint64_t stride = max_len + 16;
  const uint64_t block_stride = max_len / block + 2;
  std::normal_distribution<float> normal;
  std::vector<float> score((std::size_t)n * stride);
  for (auto& v : score)
    v = normal(rng);

  float* d_score = Device(score);
  uint32_t* d_mask = nullptr;
  float* d_blocks = nullptr;
  uint32_t* d_select = nullptr;
  const uint64_t mask_words = (block_stride + 31) / 32;
  const std::size_t mask_bytes = (std::size_t)n * mask_words * sizeof(uint32_t);
  (void)hipMalloc(&d_mask, mask_bytes);
  (void)hipMalloc(&d_blocks, (std::size_t)n * block_stride * sizeof(float));
  (void)hipMalloc(&d_select, (std::size_t)n * topk * sizeof(uint32_t));
  (void)hipMemset(d_mask, 0xff, mask_bytes);
  if (!d_score || !d_mask || !d_blocks || !d_select ||
      ds41_v41_candidate_mask(d_score, stride, d_mask, mask_words, d_blocks,
                              block_stride, d_select, pos0, n, ratio, block,
                              topk) ||
      ds41_v41_apply_mask(d_score, stride, d_mask, mask_words, pos0, n, ratio,
                          block) ||
      hipDeviceSynchronize() != hipSuccess) {
    std::fprintf(stderr, "launch failed\n");
    return SIZE_MAX;
  }
  std::vector<uint32_t> mask((std::size_t)n * mask_words);
  std::vector<float> masked(score.size());
  (void)hipMemcpy(mask.data(), d_mask, mask_bytes, hipMemcpyDeviceToHost);
  (void)hipMemcpy(masked.data(), d_score, masked.size() * sizeof(float),
                  hipMemcpyDeviceToHost);
  (void)hipFree(d_score);
  (void)hipFree(d_mask);
  (void)hipFree(d_blocks);
  (void)hipFree(d_select);

  std::size_t bad = 0;
  uint32_t dropped = 0;
  for (uint32_t t = 0; t < n; t++) {
    const uint32_t len = (pos0 + t + 1) / ratio;
    const uint32_t nb = (len + block - 1) / block;
    const float* s = score.data() + t * stride;
    std::vector<float> best(nb, -INFINITY);
    for (uint32_t j = 0; j < len; j++)
      best[j / block] = std::max(best[j / block], s[j]);
    if (nb)
      best[nb - 1] = INFINITY;
    std::vector<uint32_t> order(nb);
    std::iota(order.begin(), order.end(), 0u);
    std::sort(order.begin(), order.end(),
              [&](uint32_t a, uint32_t b) { return best[a] > best[b]; });
    std::vector<uint8_t> keep(nb, 0);
    for (uint32_t i = 0; i < std::min(topk, nb); i++)
      keep[order[i]] = 1;
    for (uint32_t b = 0; b < nb; b++)
      bad += ((mask[(std::size_t)t * mask_words + b / 32] >> (b % 32)) & 1u) !=
             keep[b];
    for (uint32_t j = 0; j < len; j++) {
      const uint8_t want = keep[j / block];
      dropped += !want;
      const float want_score = want ? s[j] : -INFINITY;
      bad += masked[(std::size_t)t * stride + j] != want_score;
    }
  }
  std::printf(
      "pos0 %6u n %4u ratio %u block %u topk %4u: %u rows dropped, "
      "%zu mismatches\n",
      pos0, n, ratio, block, topk, dropped, bad);
  return bad;
}

/* ds41_v41_topk alone, with keys drawn from `levels` values so the threshold
 * is tied: keys above it in index order, then the first tied keys. */
std::size_t Topk(uint32_t pos0, uint32_t n, uint32_t k, uint32_t levels,
                 std::mt19937& rng) {
  const uint32_t max_len = pos0 + n;
  const uint64_t stride = max_len + 16;
  std::uniform_int_distribution<uint32_t> level(0, levels - 1);
  std::vector<float> score((std::size_t)n * stride);
  for (auto& v : score)
    v = (float)level(rng) * 0.25f - 1.0f;
  float* d_score = Device(score);
  uint32_t* d_out = nullptr;
  (void)hipMalloc(&d_out, (std::size_t)n * k * sizeof(uint32_t));
  if (!d_score || !d_out ||
      ds41_v41_topk(d_score, stride, d_out, k, pos0, n, 1, 1) ||
      hipDeviceSynchronize() != hipSuccess) {
    std::fprintf(stderr, "topk launch failed\n");
    return SIZE_MAX;
  }
  std::vector<uint32_t> out((std::size_t)n * k);
  (void)hipMemcpy(out.data(), d_out, out.size() * sizeof(uint32_t),
                  hipMemcpyDeviceToHost);
  (void)hipFree(d_score);
  (void)hipFree(d_out);
  std::size_t bad = 0;
  for (uint32_t t = 0; t < n; t++) {
    const uint32_t len = pos0 + t + 1;
    const uint32_t kk = std::min(k, len);
    const float* s = score.data() + t * stride;
    std::vector<float> sorted(s, s + len);
    std::nth_element(sorted.begin(), sorted.begin() + (kk - 1), sorted.end(),
                     std::greater<float>());
    const float thr = sorted[kk - 1];
    std::vector<uint32_t> want;
    for (uint32_t j = 0; j < len; j++)
      if (s[j] > thr)
        want.push_back(j);
    for (uint32_t j = 0; j < len && want.size() < kk; j++)
      if (s[j] == thr)
        want.push_back(j);
    std::vector<uint32_t> got(out.begin() + (std::size_t)t * k,
                              out.begin() + (std::size_t)t * k + kk);
    std::vector<uint32_t> want_sorted = want;
    std::sort(want_sorted.begin(), want_sorted.end());
    std::sort(got.begin(), got.end());
    bad += got != want_sorted;
  }
  std::printf("topk pos0 %6u n %3u k %4u levels %u: %zu mismatched tokens\n",
              pos0, n, k, levels, bad);
  return bad;
}

}  // namespace

int main() {
  int devices = 0;
  if (hipGetDeviceCount(&devices) != hipSuccess || devices == 0) {
    std::fprintf(stderr, "no HIP device\n");
    return 77;
  }
  std::mt19937 rng(41);
  std::size_t bad = 0;
  /* Small shapes: prefill tokens below, at and past the block budget, with
   * partial newest blocks. */
  bad += Case(0, 64, 1, 8, 4, rng);
  bad += Case(100, 37, 2, 8, 3, rng);
  /* Production shape (8-row blocks, 2048 kept) crossing 16,384 rows in one
   * prefill batch, and single decode tokens well past it. */
  bad += Case(16370, 40, 1, 8, 2048, rng);
  bad += Case(18431, 1, 1, 8, 2048, rng);
  bad += Case(40000, 3, 1, 8, 2048, rng);
  /* The indexer's top-512 over tied and untied scores, below and above k. */
  bad += Topk(0, 600, 512, 7, rng);
  bad += Topk(2047, 2, 512, 5, rng);
  bad += Topk(70000, 2, 512, 1000000, rng);
  std::printf("%s\n", bad ? "FAIL" : "PASS");
  return bad ? 1 : 0;
}
