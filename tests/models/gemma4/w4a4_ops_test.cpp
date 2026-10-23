// The int4 prefill kernels (--prefill-sidecar, w4a4.hpp) against a CPU
// reference: the activation rotation and quantization reproduce the
// simulator's rule (tools/gemma4/quant_sim/quant.py act_quant after the
// signs and block Walsh-Hadamard transform), padding rows stay zero, and the
// GEMM matches an FP64 sum over the same codes and scales on ragged row
// counts.
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "src/core/hip/hip_utils.hpp"
#include "src/models/gemma4/kernels/rocm/w4a4.hpp"
#include "tests/models/gemma4/check.hpp"

using gemma4_test::Require;
namespace rocm = gufo::models::gemma4::rocm;

namespace {

template<class T>
T* Device(const std::vector<T>& host) {
  T* p = nullptr;
  HIP_CHECK(hipMalloc(&p, host.size() * sizeof(T)));
  HIP_CHECK(hipMemcpy(p, host.data(), host.size() * sizeof(T),
                      hipMemcpyHostToDevice));
  return p;
}

int Nibble(std::uint8_t byte, int i) {
  const int v = (byte >> (4 * i)) & 0xF;
  return v >= 8 ? v - 16 : v;
}

/// The kernel's rotation in the same FP32 operations (butterflies of span
/// 1, 2, ..., 64, then 1/sqrt(128)), and the simulator's clip search.
void QuantizeBlock(const float* x, const float* sign, int* codes,
                   float* scale) {
  float v[128];
  for (int i = 0; i < 128; ++i) {
    v[i] = x[i] * sign[i];
  }
  for (int h = 1; h < 128; h *= 2) {
    for (int i = 0; i < 128; i += 2 * h) {
      for (int j = i; j < i + h; ++j) {
        const float a = v[j];
        const float b = v[j + h];
        v[j] = a + b;
        v[j + h] = a - b;
      }
    }
  }
  float amax = 0.0F;
  for (float& f : v) {
    f *= 0.08838834764831845F;
    amax = std::max(amax, std::fabs(f));
  }
  const float clip[4] = {1.0F, 0.9F, 0.8F, 0.7F};
  float best = 0.0F;
  double best_err = 0.0;
  for (int c = 0; c < 4; ++c) {
    const float s = amax * clip[c] / 7.0F;
    double err = 0.0;
    for (float f : v) {
      const float q =
          s > 0.0F ? std::fmin(std::fmax(std::rint(f / s), -8.0F), 7.0F) : 0.0F;
      err += static_cast<double>(q * s - f) * (q * s - f);
    }
    if (c == 0 || err < best_err) {
      best = s;
      best_err = err;
    }
  }
  for (int i = 0; i < 128; ++i) {
    codes[i] = best > 0.0F
                   ? static_cast<int>(std::fmin(
                         std::fmax(std::rint(v[i] / best), -8.0F), 7.0F))
                   : 0;
  }
  *scale = best;
}

void Check(std::uint32_t rows, std::uint32_t m, std::uint32_t k,
           std::mt19937& rng) {
  const std::uint32_t blocks = k / 128;
  const std::uint32_t padded = rocm::W4A4PaddedRows(rows);
  // Activations with outlier channels, as projection inputs have.
  std::normal_distribution<float> normal(0.0F, 1.0F);
  std::vector<__half> x(std::size_t{rows} * k);
  std::vector<float> xf(x.size());
  for (std::size_t i = 0; i < x.size(); ++i) {
    const float scale = i % 97 == 0 ? 40.0F : 1.0F;
    x[i] = __float2half(normal(rng) * scale);
    xf[i] = __half2float(x[i]);
  }
  for (std::size_t i = 0; i < 128; ++i) {
    x[i] = __float2half(0.0F);  // an all-zero block: scale 0
    xf[i] = 0.0F;
  }
  std::vector<float> sign(k);
  for (float& s : sign) {
    s = (rng() & 1U) != 0 ? 1.0F : -1.0F;
  }
  std::vector<std::uint8_t> wq(std::size_t{blocks} * m * 64);
  for (auto& b : wq) {
    b = static_cast<std::uint8_t>(rng());
  }
  std::uniform_real_distribution<float> uniform(0.001F, 0.01F);
  std::vector<__half> ws(std::size_t{blocks} * m);
  for (auto& s : ws) {
    s = __float2half(uniform(rng));
  }

  auto* dx = Device(x);
  auto* dsign = Device(sign);
  auto* dwq = Device(wq);
  auto* dws = Device(ws);
  const std::size_t act_bytes = rocm::W4A4ActivationBytes(rows, k);
  void* dxq = nullptr;
  float* dy = nullptr;
  HIP_CHECK(hipMalloc(&dxq, act_bytes));
  HIP_CHECK(hipMemset(dxq, 0x5A, act_bytes));  // padding must be overwritten
  HIP_CHECK(hipMalloc(&dy, std::size_t{rows + 1} * m * sizeof(float)));
  HIP_CHECK(hipMemset(dy, 0xFF, std::size_t{rows + 1} * m * sizeof(float)));
  rocm::LaunchW4A4Quantize(dx, dsign, dxq, rows, k, nullptr);
  Require(rocm::LaunchW4A4Gemm(dwq, dws, dxq, dy, rows, m, k, nullptr),
          "GEMM shape rejected");
  HIP_CHECK(hipDeviceSynchronize());
  std::vector<std::uint8_t> xq(act_bytes);
  std::vector<float> y(std::size_t{rows + 1} * m);
  HIP_CHECK(hipMemcpy(xq.data(), dxq, act_bytes, hipMemcpyDeviceToHost));
  HIP_CHECK(
      hipMemcpy(y.data(), dy, y.size() * sizeof(float), hipMemcpyDeviceToHost));
  const std::uint8_t* codes = xq.data();
  const auto* scales = reinterpret_cast<const float*>(
      xq.data() + std::size_t{blocks} * padded * 64);

  // Quantization: codes and scales per (row, block); the clip choice may
  // differ only where two candidates' errors tie to rounding.
  std::size_t differing = 0;
  for (std::uint32_t t = 0; t < padded; ++t) {
    for (std::uint32_t b = 0; b < blocks; ++b) {
      const std::size_t g = std::size_t{b} * padded + t;
      const std::uint8_t* got = codes + g * 64;
      if (t >= rows) {
        Require(scales[g] == 0.0F &&
                    std::all_of(got, got + 64, [](auto v) { return v == 0; }),
                "padding row not zeroed");
        continue;
      }
      int want[128];
      float want_scale = 0.0F;
      QuantizeBlock(&xf[std::size_t{t} * k + b * 128], &sign[b * 128], want,
                    &want_scale);
      bool same = want_scale == scales[g];
      for (int i = 0; i < 128 && same; ++i) {
        same = Nibble(got[i / 2], i & 1) == want[i];
      }
      differing += same ? 0 : 1;
    }
  }
  Require(differing * 1000 <= std::size_t{rows} * blocks,
          "activation quantization differs from the reference in " +
              std::to_string(differing) + " blocks");

  // GEMM over the kernel's own activation codes, FP64.
  double worst = 0.0;
  for (std::uint32_t t = 0; t < rows; ++t) {
    for (std::uint32_t r = 0; r < m; ++r) {
      double ref = 0.0;
      double mag = 0.0;
      for (std::uint32_t b = 0; b < blocks; ++b) {
        const std::uint8_t* wr = &wq[(std::size_t{b} * m + r) * 64];
        const std::uint8_t* xr = codes + (std::size_t{b} * padded + t) * 64;
        long dot = 0;
        long abs_dot = 0;
        for (int i = 0; i < 128; ++i) {
          const int p = Nibble(wr[i / 2], i & 1) * Nibble(xr[i / 2], i & 1);
          dot += p;
          abs_dot += std::abs(p);
        }
        const double s =
            static_cast<double>(__half2float(ws[std::size_t{b} * m + r])) *
            scales[std::size_t{b} * padded + t];
        ref += s * static_cast<double>(dot);
        mag += s * static_cast<double>(abs_dot);
      }
      const double got = y[std::size_t{t} * m + r];
      worst = std::max(worst, std::fabs(got - ref) / (mag + 1e-30));
    }
  }
  // Untouched past the last row.
  for (std::uint32_t r = 0; r < m; ++r) {
    std::uint32_t bits;
    std::memcpy(&bits, &y[std::size_t{rows} * m + r], 4);
    Require(bits == 0xFFFFFFFFU, "GEMM wrote past the last row");
  }
  std::cout << "rows " << rows << " m " << m << " k " << k
            << ": quantization blocks differing " << differing << "/"
            << rows * blocks << ", GEMM max error " << worst
            << " (of the absolute sum)\n";
  Require(worst < 1e-6, "GEMM differs from the FP64 reference");
  HIP_CHECK(hipFree(dx));
  HIP_CHECK(hipFree(dsign));
  HIP_CHECK(hipFree(dwq));
  HIP_CHECK(hipFree(dws));
  HIP_CHECK(hipFree(dxq));
  HIP_CHECK(hipFree(dy));
}

}  // namespace

int main() {
  return gemma4_test::Run([] {
    std::mt19937 rng(11);
    Require(!rocm::LaunchW4A4Gemm(nullptr, nullptr, nullptr, nullptr, 17, 200,
                                  640, nullptr),
            "a row count off the 128-row tile was accepted");
    Check(17, 256, 640, rng);
    Check(128, 384, 1024, rng);
    Check(301, 128, 5376, rng);
  });
}
