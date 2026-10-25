// Decode matvec bandwidth on the V4.1 Flash dense Q5_K shapes and routed
// expert shapes: the vendored mmvq path (Q8_1 activations) against
// ds41_v41_gemv at every slice count and ds41_v41_gemv_experts.
// Weights rotate through copies larger than the caches, so every launch reads
// DRAM. Rows differ by the Q8_1 activation rounding of mmvq.
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "ds41_mmq.h"
#include "ds41_v41.h"

namespace {

struct Shape {
  const char* name;
  int m, k;
  int rows_per_x = 0;  // grouped output LoRA: rows per activation group
};

std::vector<uint8_t> RandomQ5K(std::size_t blocks, std::mt19937& rng) {
  std::vector<uint8_t> data(blocks * 176);
  for (auto& b : data)
    b = static_cast<uint8_t>(rng());
  for (std::size_t i = 0; i < blocks; ++i) {
    const uint16_t d = 0x1c00 | (rng() & 0x3ff),
                   dmin = 0x1800 | (rng() & 0x3ff);
    std::memcpy(&data[i * 176], &d, 2);
    std::memcpy(&data[i * 176 + 2], &dmin, 2);
  }
  return data;
}

}  // namespace

int main() {
  if (ds41_mmq_init(0) != 0)
    return 1;
  const Shape shapes[] = {
      {"q_b", 32768, 1280},     {"wq_a", 1280, 5120},
      {"wkv", 512, 5120},       {"wo_b", 5120, 8192},
      {"sh_gate", 2304, 5120},  {"sh_down", 5120, 2304},
      {"output", 129280, 5120}, {"wo_a", 8192, 4096, 1024},
  };
  std::mt19937 rng(41);
  for (const Shape& s : shapes) {
    const std::size_t bytes = static_cast<std::size_t>(s.m) * s.k / 256 * 176;
    const int copies =
        static_cast<int>(std::max<std::size_t>(2, (512u << 20) / bytes));
    const auto host = RandomQ5K(static_cast<std::size_t>(s.m) * s.k / 256, rng);
    std::vector<void*> w(copies);
    for (auto& p : w) {
      (void)hipMalloc(&p, bytes);
      (void)hipMemcpy(p, host.data(), bytes, hipMemcpyHostToDevice);
    }
    const int x_groups = s.rows_per_x ? s.m / s.rows_per_x : 1;
    std::vector<float> xh(static_cast<std::size_t>(s.k) * x_groups);
    std::normal_distribution<float> normal;
    for (auto& v : xh)
      v = normal(rng);
    float *x = nullptr, *y0 = nullptr, *y1 = nullptr;
    (void)hipMalloc(&x, xh.size() * 4);
    (void)hipMalloc(&y0, s.m * 4);
    (void)hipMalloc(&y1, s.m * 4);
    (void)hipMemcpy(x, xh.data(), xh.size() * 4, hipMemcpyHostToDevice);
    hipEvent_t a, b;
    (void)hipEventCreate(&a);
    (void)hipEventCreate(&b);
    auto time = [&](auto&& launch) {
      for (int i = 0; i < copies; ++i)
        launch(w[i]);
      const int iters = std::max(copies, 40);
      (void)hipEventRecord(a, 0);
      for (int i = 0; i < iters; ++i)
        launch(w[i % copies]);
      (void)hipEventRecord(b, 0);
      (void)hipEventSynchronize(b);
      float ms = 0;
      (void)hipEventElapsedTime(&ms, a, b);
      return ms * 1000.0 / iters;
    };
    const double us_mmvq =
        s.rows_per_x ? time([&](void* p) {
          ds41_mmq_grouped_vec_typed(13, p, x, y0, s.rows_per_x, 1, s.k,
                                     x_groups, 0);
        })
                     : time([&](void* p) {
                         ds41_mmq_dense_vec_typed(13, p, x, y0, s.m, 1, s.k, 0);
                       });
    std::printf("%-8s %6dx%-5d %6.1f MB  mmvq %6.1f GB/s  ds41", s.name, s.m,
                s.k, bytes / 1e6, bytes / us_mmvq / 1e3);
    for (const int slices : {1, 2, 4, 8}) {
      const double us = time([&](void* p) {
        ds41_v41_gemv(13, p, x, y1, s.m, s.k, s.rows_per_x, slices);
      });
      std::vector<float> r0(s.m), r1(s.m);
      (void)hipMemcpy(r0.data(), y0, s.m * 4, hipMemcpyDeviceToHost);
      (void)hipMemcpy(r1.data(), y1, s.m * 4, hipMemcpyDeviceToHost);
      double e = 0, ref = 0;
      for (int i = 0; i < s.m; ++i) {
        e += (r0[i] - r1[i]) * (r0[i] - r1[i]);
        ref += r1[i] * r1[i];
      }
      std::printf("  S%d %6.1f (%.0e)", slices, bytes / us / 1e3,
                  std::sqrt(e / ref));
    }
    std::printf("\n");
    for (auto* p : w)
      (void)hipFree(p);
    (void)hipFree(x);
    (void)hipFree(y0);
    (void)hipFree(y1);
  }
  // Routed experts: six of 384, a different six per launch so the reads stay
  // DRAM-cold; gate/up share one input, down has one input per slot. Random
  // blocks: the timings do not depend on the values.
  struct Expert {
    const char* name;
    int type, block_bytes, m, k;
    bool per_slot;
  };
  const Expert experts[] = {
      {"iq1s_up", 19, 50, 2304, 5120, false},
      {"iq1s_down", 19, 50, 5120, 2304, true},
      {"iq1m_up", 29, 56, 2304, 5120, false},
      {"iq1m_down", 29, 56, 5120, 2304, true},
      {"iq2xxs_up", 16, 66, 2304, 5120, false},
      {"iq2xxs_down", 16, 66, 5120, 2304, true},
      {"iq2s_down", 22, 82, 5120, 2304, true},
  };
  constexpr int kExperts = 384;
  constexpr int kRounds = 64;
  std::vector<int32_t> idh(static_cast<std::size_t>(kRounds) * 6);
  for (int r = 0; r < kRounds; ++r)
    for (int j = 0; j < 6; ++j)
      idh[r * 6 + j] = static_cast<int32_t>((r * 6 + j) * 61 % kExperts);
  for (const Expert& e : experts) {
    const std::size_t per_expert =
        static_cast<std::size_t>(e.m) * e.k / 256 * e.block_bytes;
    std::vector<uint8_t> host(per_expert * kExperts);
    for (auto& b : host)
      b = static_cast<uint8_t>(rng());
    std::vector<float> xh(static_cast<std::size_t>(e.per_slot ? 6 : 1) * e.k);
    std::normal_distribution<float> normal;
    for (auto& v : xh)
      v = normal(rng);
    void* w = nullptr;
    float *x = nullptr, *y = nullptr;
    int32_t* ids = nullptr;
    (void)hipMalloc(&w, host.size());
    (void)hipMalloc(&x, xh.size() * sizeof(float));
    (void)hipMalloc(&y, static_cast<std::size_t>(e.m) * 6 * sizeof(float));
    (void)hipMalloc(&ids, idh.size() * sizeof(int32_t));
    (void)hipMemcpy(w, host.data(), host.size(), hipMemcpyHostToDevice);
    (void)hipMemcpy(x, xh.data(), xh.size() * sizeof(float),
                    hipMemcpyHostToDevice);
    (void)hipMemcpy(ids, idh.data(), idh.size() * sizeof(int32_t),
                    hipMemcpyHostToDevice);
    auto launch = [&](int r, bool fp32) {
      if (fp32)
        ds41_v41_gemv_experts(e.type, w, per_expert, ids + r * 6, 6, x,
                              e.per_slot ? e.k : 0, y, e.m, e.k);
      else if (e.per_slot)
        ds41_mmq_moe_vec_typed(e.type, w, x, ids + r * 6, y, e.m, e.k, 6,
                               kExperts, 1, 0);
      else
        ds41_mmq_moe_vec_typed(e.type, w, x, ids + r * 6, y, e.m, e.k, 1,
                               kExperts, 6, 0);
    };
    std::printf("%-12s %5dx%-5d x6 %5.1f MB", e.name, e.m, e.k,
                6.0 * per_expert / 1e6);
    for (const bool fp32 : {false, true}) {
      hipEvent_t a, b;
      (void)hipEventCreate(&a);
      (void)hipEventCreate(&b);
      for (int r = 0; r < 8; ++r)
        launch(r, fp32);
      (void)hipEventRecord(a, 0);
      for (int r = 0; r < kRounds; ++r)
        launch(r, fp32);
      (void)hipEventRecord(b, 0);
      (void)hipEventSynchronize(b);
      float ms = 0;
      (void)hipEventElapsedTime(&ms, a, b);
      const double us = ms * 1000.0 / kRounds;
      std::printf("  %s %6.1f us %5.1f GB/s", fp32 ? "fp32" : "mmvq", us,
                  6.0 * per_expert / us / 1e3);
      (void)hipEventDestroy(a);
      (void)hipEventDestroy(b);
    }
    std::printf("\n");
    (void)hipFree(w);
    (void)hipFree(x);
    (void)hipFree(y);
    (void)hipFree(ids);
  }
  ds41_mmq_cleanup();
  return 0;
}
