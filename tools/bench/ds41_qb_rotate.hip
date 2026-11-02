/* Dense attn_q_b (32768 x 1280, Q8_0) timing at 2048 tokens with rotating
 * weight buffers: the shared ds41_mmq_bench --dense reuses one 44.6 MiB
 * weight matrix across repetitions, ~72% of which can sit in gfx1151's
 * 32 MiB MALL.  This variant repeats the projection over N independent
 * weight copies so every call reads DRAM-cold weights, as the served model
 * does (each layer owns its own q_b), and reports both the warm and cold
 * rates side by side.
 *
 *   /tmp/ds41_qb_rotate [copies]
 */
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "src/models/deepseek_v41_flash/kernels/rocm/ds41_dense_f16.hip.cpp"

int main(int argc, char** argv) {
  const int copies = argc > 1 ? std::atoi(argv[1]) : 4;
  const int m = 32768, k = 1280, tokens = 2048, groups = 1, type = 8;
  const int block = 34, values = 32;
  std::mt19937 rng(7);
  const std::size_t bytes = (std::size_t)groups * m * k / values * block;
  std::vector<std::vector<uint8_t>> w;
  std::vector<void*> dw(copies, nullptr);
  for (int c = 0; c < copies; ++c) {
    w.emplace_back(bytes);
    for (std::size_t at = 0; at < bytes; ++at) w[c][at] = (uint8_t)rng();
    for (std::size_t at = 0; at < bytes; at += block) {
      const uint16_t d = 0x2000 | (rng() & 0x03ff);
      std::memcpy(&w[c][at], &d, 2);
    }
    if (hipMalloc(&dw[c], bytes) != hipSuccess) { std::printf("oom\n"); return 1; }
    hipMemcpy(dw[c], w[c].data(), bytes, hipMemcpyHostToDevice);
  }
  std::vector<float> x((std::size_t)tokens * groups * k);
  std::normal_distribution<float> normal;
  for (auto& v : x) v = normal(rng);
  void *dx = nullptr, *out = nullptr;
  hipMalloc(&dx, x.size() * 4);
  hipMalloc(&out, (std::size_t)tokens * groups * m * 4);
  hipMemcpy(dx, x.data(), x.size() * 4, hipMemcpyHostToDevice);
  void* dx16 = nullptr;
  hipMalloc(&dx16, x.size() * 2);
  {
    std::vector<_Float16> h16(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) h16[i] = (_Float16)x[i];
    hipMemcpy(dx16, h16.data(), x.size() * 2, hipMemcpyHostToDevice);
  }
  hipEvent_t e0, e1;
  hipEventCreate(&e0);
  hipEventCreate(&e1);
  const int reps = 30;
  for (int mode = 0; mode < 2; ++mode) {
    std::vector<float> ms;
    for (int r = 0; r <= reps; ++r) {
      hipEventRecord(e0, 0);
      ds41_v41_dense_f16(type, mode ? dw[r % copies] : dw[0], dx16, (float*)out, m,
                         tokens, k, groups, DS41_V41_DENSE_X_HALF);
      hipEventRecord(e1, 0);
      hipEventSynchronize(e1);
      float t = 0;
      hipEventElapsedTime(&t, e0, e1);
      if (r > 0) ms.push_back(t);
    }
    std::sort(ms.begin(), ms.end());
    std::printf("%s: %.3f ms (%.1f TOPS)\n", mode ? "rotating" : "single-buffer",
                ms[ms.size() / 2], 2.0 * tokens * m * k / ms[ms.size() / 2] / 1e9);
  }
  return 0;
}
