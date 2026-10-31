/* DRAM-cold timing of the V4.1 Flash dense Q5_K decode GEMV (ds41_v41_gemv)
 * on the decode shapes at 1-8 tokens, the DSpark verification widths.
 *
 * Includes the production translation unit (same flags). Weights rotate
 * through copies larger than the caches, so every launch reads DRAM; each
 * multi-token result is checked bit for bit against one single-token launch
 * per token, the contract the verification rows rely on. */
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "src/models/deepseek_v41_flash/kernels/rocm/ds41_gemv.hip.cpp"

namespace {

struct Shape {
  const char* name;
  uint32_t m, k, rows_per_x;
  int slices;  // as gemv_slices chooses them
};

std::vector<uint8_t> RandomQ5K(size_t blocks, std::mt19937& rng) {
  std::vector<uint8_t> data(blocks * 176);
  for (auto& b : data) b = static_cast<uint8_t>(rng());
  for (size_t i = 0; i < blocks; ++i) {
    const uint16_t d = 0x1c00 | (rng() & 0x3ff), dmin = 0x1800 | (rng() & 0x3ff);
    std::memcpy(&data[i * 176], &d, 2);
    std::memcpy(&data[i * 176 + 2], &dmin, 2);
  }
  return data;
}

}  // namespace

int main(int argc, char** argv) {
  const int reps = argc > 1 ? std::atoi(argv[1]) : 50;
  const Shape shapes[] = {
      {"wo_b", 5120, 8192, 5120, 8},     {"wo_a", 8192, 4096, 1024, 8}, {"q_b", 32768, 1280, 32768, 2},
      {"sh_down", 5120, 2304, 5120, 2}, {"wq_a", 1280, 5120, 1280, 8},
  };
  std::mt19937 rng(41);
  hipEvent_t a, b;
  (void)hipEventCreate(&a);
  (void)hipEventCreate(&b);
  bool all_exact = true;
  for (const Shape& s : shapes) {
    const size_t bytes = (size_t)s.m * s.k / 256 * 176;
    const int copies = (int)std::max<size_t>(2, (768u << 20) / bytes);
    const auto host = RandomQ5K((size_t)s.m * s.k / 256, rng);
    std::vector<void*> w(copies);
    for (auto& p : w) {
      (void)hipMalloc(&p, bytes);
      (void)hipMemcpy(p, host.data(), bytes, hipMemcpyHostToDevice);
    }
    const uint32_t groups = s.m / s.rows_per_x;
    const uint64_t xs = (uint64_t)groups * s.k;  // floats per token
    std::vector<float> xh(8 * xs);
    std::normal_distribution<float> nd;
    for (auto& v : xh) v = nd(rng);
    float *x, *y, *y1;
    (void)hipMalloc(&x, xh.size() * 4);
    (void)hipMemcpy(x, xh.data(), xh.size() * 4, hipMemcpyHostToDevice);
    (void)hipMalloc(&y, (size_t)8 * s.m * 4);
    (void)hipMalloc(&y1, (size_t)8 * s.m * 4);
    for (uint32_t n : {1u, 2u, 4u, 8u}) {
      const auto run = [&](int i) {
        return ds41_v41_gemv(13, w[i % copies], x, y, s.m, s.k, s.rows_per_x, s.slices, n, xs, s.m);
      };
      for (int i = 0; i < copies; ++i) (void)run(i);
      (void)hipDeviceSynchronize();
      (void)hipEventRecord(a);
      for (int i = 0; i < reps; ++i) (void)run(i);
      (void)hipEventRecord(b);
      (void)hipEventSynchronize(b);
      float ms = 0;
      (void)hipEventElapsedTime(&ms, a, b);
      const double us = 1000.0 * ms / reps;
      (void)run(0);
      for (uint32_t c = 0; c < n; ++c)
        (void)ds41_v41_gemv(13, w[0], x + c * xs, y1 + (size_t)c * s.m, s.m, s.k, s.rows_per_x, s.slices, 1, xs,
                            s.m);
      std::vector<float> r((size_t)n * s.m), r1((size_t)n * s.m);
      (void)hipMemcpy(r.data(), y, r.size() * 4, hipMemcpyDeviceToHost);
      (void)hipMemcpy(r1.data(), y1, r1.size() * 4, hipMemcpyDeviceToHost);
      const bool exact = std::memcmp(r.data(), r1.data(), r.size() * 4) == 0;
      all_exact = all_exact && exact;
      std::printf("%-8s n=%u  %7.1f us  %5.1f GB/s  %s\n", s.name, n, us, bytes / us / 1e3,
                  exact ? "exact" : "DIFF");
    }
    for (auto p : w) (void)hipFree(p);
    (void)hipFree(x);
    (void)hipFree(y);
    (void)hipFree(y1);
  }
  return all_exact ? 0 : 1;
}
