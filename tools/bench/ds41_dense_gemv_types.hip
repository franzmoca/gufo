/* DRAM-cold slice sweep of the V4.1 Flash dense decode GEMV
 * (ds41_v41_gemv) on the decode shapes for Q8_0 (8), Q6_K (14) and Q5_K
 * (13), the formats the streamed build mixes per layer, at 1-8 tokens (the
 * DSpark verification widths), over slices 1-8. Weights rotate through
 * copies larger than the caches so every launch reads DRAM; each
 * multi-token result is checked bit for bit against single-token launches
 * on the same weights, the contract the verification rows rely on. The
 * model adds upload and side-stream contention on top of these numbers. */
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
};

std::vector<uint8_t> RandomWeights(int type, size_t blocks, std::mt19937& rng) {
  const size_t bytes = blocks * (type == 8 ? 34 : type == 14 ? 210 : 176);
  std::vector<uint8_t> data(bytes);
  for (auto& b : data) b = static_cast<uint8_t>(rng());
  for (size_t i = 0; i < blocks; ++i) {
    const uint16_t d = 0x1c00 | (rng() & 0x3ff);
    std::memcpy(&data[i * (type == 8 ? 34 : type == 14 ? 210 : 176) + (type == 14 ? 208 : 0)], &d, 2);
    if (type == 13) {
      const uint16_t dmin = 0x1800 | (rng() & 0x3ff);
      std::memcpy(&data[i * 176 + 2], &dmin, 2);
    }
  }
  return data;
}

}  // namespace

int main(int argc, char** argv) {
  const int reps = argc > 1 ? std::atoi(argv[1]) : 50;
  const Shape shapes[] = {
      {"wo_b", 5120, 8192, 5120},    {"wo_a", 8192, 4096, 1024}, {"q_b", 32768, 1280, 32768},
      {"sh_down", 5120, 2304, 5120}, {"wq_akv", 1792, 5120, 1792},
  };
  std::mt19937 rng(41);
  hipEvent_t a, b;
  (void)hipEventCreate(&a);
  (void)hipEventCreate(&b);
  for (int type : {8, 14, 13}) {
    const int block = type == 8 ? 34 : type == 14 ? 210 : 176;
    const int values = type == 8 ? 32 : 256;
    for (const Shape& s : shapes) {
      const size_t bytes = (size_t)s.m * s.k / values * block;
      const int copies = (int)std::max<size_t>(2, (768u << 20) / bytes);
      const auto host = RandomWeights(type, (size_t)s.m * s.k / values, rng);
      std::vector<void*> w(copies);
      for (auto& p : w) {
        (void)hipMalloc(&p, bytes);
        (void)hipMemcpy(p, host.data(), bytes, hipMemcpyHostToDevice);
      }
      const uint32_t groups = s.m / s.rows_per_x;
      const uint64_t xs = (uint64_t)groups * s.k;
      std::vector<float> xh(8 * xs);
      std::normal_distribution<float> nd;
      for (auto& v : xh) v = nd(rng);
      float *x, *y, *y1;
      (void)hipMalloc(&x, xh.size() * 4);
      (void)hipMemcpy(x, xh.data(), xh.size() * 4, hipMemcpyHostToDevice);
      (void)hipMalloc(&y, (size_t)8 * s.m * 4);
      (void)hipMalloc(&y1, (size_t)8 * s.m * 4);
      for (uint32_t n : {1u, 2u, 4u, 6u, 8u}) {
        for (int sl : {1, 2, 4, 8}) {
          const auto run = [&](int i) {
            return ds41_v41_gemv(type, w[i % copies], x, y, s.m, s.k, s.rows_per_x, sl, n, xs, s.m);
          };
          for (int i = 0; i < copies; ++i) (void)run(i);
          (void)hipDeviceSynchronize();
          (void)hipEventRecord(a);
          for (int i = 0; i < reps; ++i) (void)run(i);
          (void)hipEventRecord(b);
          (void)hipEventSynchronize(b);
          float ms = 0.0f;
          (void)hipEventElapsedTime(&ms, a, b);
          ms /= reps;
          /* Bit-exactness against single-token launches on the same weights. */
          bool exact = true;
          if (n > 1) {
            (void)ds41_v41_gemv(type, w[0], x, y, s.m, s.k, s.rows_per_x, sl, n, xs, s.m);
            for (uint32_t t = 0; t < n && exact; ++t)
              if (ds41_v41_gemv(type, w[0], x + (uint64_t)t * xs, y1 + (size_t)t * s.m, s.m, s.k, s.rows_per_x, sl,
                                1, xs, s.m))
                exact = false;
            (void)hipDeviceSynchronize();
            std::vector<float> host_y((size_t)n * s.m), host_y1((size_t)n * s.m);
            (void)hipMemcpy(host_y.data(), y, host_y.size() * 4, hipMemcpyDeviceToHost);
            (void)hipMemcpy(host_y1.data(), y1, host_y1.size() * 4, hipMemcpyDeviceToHost);
            for (size_t i = 0; i < host_y.size() && exact; ++i) exact = host_y[i] == host_y1[i];
          }
          std::printf("type %2d %-7s m %5u k %5u S %d n %u: %8.1f us  %6.1f GB/s  exact=%d\n", type, s.name, s.m,
                      s.k, sl, n, ms * 1e3, (double)bytes / ms / 1e6, (int)exact);
          if (!exact) std::printf("  NOT bit-exact\n");
        }
      }
      for (auto p : w) (void)hipFree(p);
      (void)hipFree(x);
      (void)hipFree(y);
      (void)hipFree(y1);
    }
  }
  return 0;
}
