/* Standalone bench of the dense Q5_K FP16 prefill kernel over the decoder's
 * four shapes with rotating weights and an output hash, so two builds can be
 * compared for both time and bits. Used to check the iterative-ILP scheduler
 * flag (-Xarch_device -mllvm=-amdgpu-sched-strategy=iterative-ilp, as the
 * Qwen quant16 kernel builds): 2026-10-27 it measured 32.1-33.4 against the
 * default's 32.2-33.5 TFLOP/s with identical hashes -- rejected. The kernel
 * is WMMA-issue bound (128 WMMAs against ~2.1k VALU per body), like the
 * Gemma 4 prefill kernel whose audit exhausted the tile/scheduling knobs.
 *
 * The build may need the host's older glibc on the link line explicitly
 * (-L/nix/store/...-glibc-2.42.../lib) when the default picks a libm whose
 * sqrtf is newer than the loader. */
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "src/models/deepseek_v41_flash/kernels/rocm/ds41_dense_f16.hip.cpp"

namespace {

std::vector<uint8_t> RandomQ5K(std::size_t blocks, std::mt19937& rng) {
  std::vector<uint8_t> data(blocks * 176);
  for (auto& b : data)
    b = static_cast<uint8_t>(rng());
  for (std::size_t i = 0; i < blocks; ++i) {
    const uint16_t d = 0x1c00 | (rng() & 0x3ff), dmin = 0x1800 | (rng() & 0x3ff);
    std::memcpy(&data[i * 176], &d, 2);
    std::memcpy(&data[i * 176 + 2], &dmin, 2);
  }
  return data;
}

}  // namespace

int main() {
  setbuf(stdout, nullptr);
  const struct {
    const char* name;
    uint32_t m, k, groups;
  } shapes[] = {
      {"q_b", 32768, 1280, 1},    {"o_b", 5120, 8192, 1},     {"sh_gate", 2304, 5120, 1},
      {"sh_down", 5120, 2304, 1}, {"wo_a", 1024, 4096, 8},    {"engram", 25600, 1280, 1},
  };
  const uint32_t n = 2048;
  std::mt19937 rng(41);
  for (const auto& s : shapes) {
    const std::size_t bytes = static_cast<std::size_t>(s.groups) * s.m * s.k / 256 * 176;
    const int copies = static_cast<int>(std::max<std::size_t>(2, (512u << 20) / bytes));
    const auto host = RandomQ5K(static_cast<std::size_t>(s.groups) * s.m * s.k / 256, rng);
    std::vector<void*> w(copies);
    for (auto& p : w) {
      (void)hipMalloc(&p, bytes);
      (void)hipMemcpy(p, host.data(), bytes, hipMemcpyHostToDevice);
    }
    const std::size_t x_n = static_cast<std::size_t>(s.groups) * s.k * n;
    std::vector<float> xh(x_n);
    std::normal_distribution<float> normal;
    for (auto& v : xh)
      v = normal(rng);
    float *x = nullptr, *y0 = nullptr, *y1 = nullptr;
    (void)hipMalloc(&x, xh.size() * 4);
    (void)hipMalloc(&y0, static_cast<std::size_t>(s.groups) * s.m * n * 4);
    (void)hipMalloc(&y1, static_cast<std::size_t>(s.groups) * s.m * n * 4);
    (void)hipMemcpy(x, xh.data(), xh.size() * 4, hipMemcpyHostToDevice);
    hipEvent_t a, b;
    (void)hipEventCreate(&a);
    (void)hipEventCreate(&b);
    auto time = [&](float* out) {
      for (int i = 0; i < copies; ++i)
        (void)ds41_v41_dense_f16(13, w[i], x, out, s.m, n, s.k, s.groups, 0);
      const int iters = std::max(copies, 6);
      (void)hipEventRecord(a, 0);
      for (int i = 0; i < iters; ++i)
        (void)ds41_v41_dense_f16(13, w[i % copies], x, out, s.m, n, s.k, s.groups, 0);
      (void)hipEventRecord(b, 0);
      (void)hipEventSynchronize(b);
      float ms = 0;
      (void)hipEventElapsedTime(&ms, a, b);
      return ms / iters;
    };
    const double ms0 = time(y0);
    (void)ds41_v41_dense_f16(13, w[0], x, y0, s.m, n, s.k, s.groups, 0);
    std::vector<float> ref(static_cast<std::size_t>(s.groups) * s.m * n);
    (void)hipMemcpy(ref.data(), y0, ref.size() * 4, hipMemcpyDeviceToHost);
    uint64_t hash = 1469598103934665603ull;
    for (float v : ref) {
      uint32_t bits;
      std::memcpy(&bits, &v, 4);
      hash = (hash ^ bits) * 1099511628211ull;
    }
    const double flops = 2.0 * s.m * s.k * n * s.groups;
    std::printf("%-8s %5ux%-5d g%-2d %7.2f ms %5.1f TF hash %016llx\n", s.name, s.m, s.k, s.groups, ms0,
                flops / ms0 / 1e9, static_cast<unsigned long long>(hash));
    for (auto* p : w)
      (void)hipFree(p);
    (void)hipFree(x);
    (void)hipFree(y0);
    (void)hipFree(y1);
    (void)hipEventDestroy(a);
    (void)hipEventDestroy(b);
  }
  return 0;
}
