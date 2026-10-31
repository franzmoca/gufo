/* Byte comparison of the two-pass top-k (fcf839d7) against the radix kernel
 * it replaced, on adversarial rows (ties, all-equal rows, overflowing bins,
 * signed zeros, k >= n, k 2048 with div 8). old.inc is the replaced
 * topk_kernel renamed old_topk_kernel:
 *   git show 05bcbe05:src/models/deepseek_v41_flash/kernels/rocm/ds41_v41.hip.cpp |
 *     awk '/^__global__ void topk_kernel/,/^}$/' | sed 's/topk_kernel/old_topk_kernel/' > /tmp/topk-test/old.inc
 * Copied to tools/bench and built with tools/bench/build.sh: 400 cases (seed
 * 7) and 3000 (seed 1234), 0 failures. */
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>
#include "src/models/deepseek_v41_flash/kernels/rocm/ds41_v41.hip.cpp"
#include "/tmp/topk-test/old.inc"
int main() {
  std::mt19937 rng(7);
  int fails = 0, cases = 0;
  for (int c = 0; c < 400; c++) {
    const uint32_t mode = c % 8;
    const uint32_t ratio = 1 + (c / 8) % 2;
    const uint32_t div = (c % 3 == 0) ? 8 : 1;
    const uint32_t kmax = div == 8 ? 2048 : 512;
    const uint32_t nt = 1 + rng() % 24;
    const uint32_t pos0 = (mode == 7) ? rng() % 3000 : rng() % 140000;
    const uint64_t stride = (pos0 + nt) / ratio + 7 + (rng() % 5);
    std::vector<float> h(nt * stride);
    std::normal_distribution<float> nd;
    for (auto& v : h) {
      float x = nd(rng);
      switch (mode) {
        case 0: break;
        case 1: x = std::round(x * 4) / 4; break;            // many ties
        case 2: x = 1.0f; break;                             // all equal
        case 3: x = std::round(x * 64) / 64; break;          // bin overflow, ties
        case 4: x = (rng() % 2) ? 0.0f : -0.0f; if (rng() % 50 == 0) x = nd(rng); break;
        case 5: x = -std::fabs(x); break;
        case 6: x = 1.0f + (rng() % 3000) * 1e-7f; break;    // dense in one bin
        case 7: break;
      }
      v = x;
    }
    float* s; uint32_t *a, *b;
    hipMalloc(&s, h.size() * 4);
    hipMalloc(&a, (size_t)nt * kmax * 4);
    hipMalloc(&b, (size_t)nt * kmax * 4);
    hipMemcpy(s, h.data(), h.size() * 4, hipMemcpyHostToDevice);
    hipMemset(a, 0xff, (size_t)nt * kmax * 4);
    hipMemset(b, 0xff, (size_t)nt * kmax * 4);
    old_topk_kernel<<<nt, 1024>>>(s, stride, a, kmax, pos0, ratio, div);
    topk_kernel<<<nt, kTopkThreads>>>(s, stride, b, kmax, pos0, ratio, div);
    std::vector<uint32_t> ha((size_t)nt * kmax), hb((size_t)nt * kmax);
    hipMemcpy(ha.data(), a, ha.size() * 4, hipMemcpyDeviceToHost);
    hipMemcpy(hb.data(), b, hb.size() * 4, hipMemcpyDeviceToHost);
    cases++;
    if (ha != hb) { fails++; if (fails < 6) printf("FAIL case %d mode %u ratio %u div %u pos0 %u nt %u\n", c, mode, ratio, div, pos0, nt); }
    hipFree(s); hipFree(a); hipFree(b);
  }
  printf("%d cases, %d failures\n", cases, fails);
  return fails != 0;
}
