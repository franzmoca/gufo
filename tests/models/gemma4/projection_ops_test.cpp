// Tripwire over the shared projection kernels at Gemma 4 shapes. Decode
// (one-row GEMV) and verification (small-batch FP32 GEMM) must agree bit for
// bit at every width, so greedy speculation reproduces autoregressive
// decoding; both must match the CPU dequantized dot product, and the W8A8
// prefill GEMM must stay within its activation-rounding envelope. A change in
// the Qwen-owned kernels that breaks any of this fails here.
#include <hip/hip_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <optional>
#include <random>
#include <span>
#include <string>
#include <vector>

#include "src/core/hip/hip_utils.hpp"
#include "src/core/quant/ggml_gemm.hpp"
#include "src/models/gemma4/kernels/rocm/gemv.hpp"
#include "src/models/qwen/hip/ops/gemm.hpp"
#include "tests/models/gemma4/check.hpp"

using gemma4_test::Require;
using gufo::core::GgmlType;

namespace {

struct Format {
  GgmlType type;
  std::optional<gufo::models::gemma4::rocm::GemvFormat> gemv;
  std::size_t block;
  std::size_t bytes;
  std::vector<std::size_t> half_offsets;  ///< fp16 scale fields per block
  const char* name;
};

std::uint16_t Half(float x) {
  const _Float16 h = static_cast<_Float16>(x);
  std::uint16_t bits;
  std::memcpy(&bits, &h, 2);
  return bits;
}

/// Random blocks with bounded, finite scales.
std::vector<std::uint8_t> RandomMatrix(const Format& f, std::size_t rows,
                                       std::size_t cols, std::mt19937& rng) {
  const std::size_t blocks = rows * cols / f.block;
  std::vector<std::uint8_t> data(blocks * f.bytes + 4096, 0);
  std::uniform_int_distribution<int> byte(0, 255);
  std::uniform_real_distribution<float> scale(0.0005F, 0.004F);
  for (std::size_t i = 0; i < blocks * f.bytes; ++i) {
    data[i] = static_cast<std::uint8_t>(byte(rng));
  }
  for (std::size_t b = 0; b < blocks; ++b) {
    for (std::size_t off : f.half_offsets) {
      const std::uint16_t h = Half(scale(rng));
      std::memcpy(&data[b * f.bytes + off], &h, 2);
    }
  }
  return data;
}

template<class T>
T* Device(const T* host, std::size_t n) {
  T* ptr = nullptr;
  HIP_CHECK(hipMalloc(&ptr, n * sizeof(T)));
  HIP_CHECK(hipMemcpy(ptr, host, n * sizeof(T), hipMemcpyHostToDevice));
  return ptr;
}

void CheckShape(const Format& f, std::size_t m, std::size_t k,
                std::mt19937& rng) {
  const std::string name =
      std::string(f.name) + " " + std::to_string(m) + "x" + std::to_string(k);
  const auto w = RandomMatrix(f, m, k, rng);
  constexpr std::size_t kRows = 16;
  constexpr std::size_t kPrefillRows = 40;
  std::normal_distribution<float> normal(0.0F, 1.0F);
  std::vector<float> x(kPrefillRows * k);
  for (float& v : x)
    v = normal(rng);
  auto* dw = Device(w.data(), w.size());
  float* dx = Device(x.data(), x.size());
  float* dy = nullptr;
  HIP_CHECK(hipMalloc(&dy, kPrefillRows * m * sizeof(float)));

  // Decode reference rows.
  std::vector<float> gemv(kRows * m);
  for (std::size_t r = 0; r < kRows; ++r) {
    gufo::hip::LaunchGEMV(dw, f.type, dx + r * k, dy + r * m, m, k, nullptr);
  }
  HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(hipMemcpy(gemv.data(), dy, gemv.size() * 4, hipMemcpyDeviceToHost));

  // Every verification width reproduces the decode rows bit for bit.
  std::vector<float> batch(kRows * m);
  for (std::size_t width : {2, 3, 5, 8, 9, 16}) {
    HIP_CHECK(hipMemset(dy, 0xFF, width * m * sizeof(float)));
    gufo::hip::LaunchBatchedQuantGEMMFp32(f.type, dw, dx, dy, width, m, k,
                                          nullptr);
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(
        hipMemcpy(batch.data(), dy, width * m * 4, hipMemcpyDeviceToHost));
    Require(std::memcmp(batch.data(), gemv.data(), width * m * 4) == 0,
            name + ": width " + std::to_string(width) +
                " differs from decode GEMV");
  }

  // Decode against an FP64 dot of the CPU-dequantized row, on a row sample;
  // the error is bounded relative to sum |w x| (FP32 accumulation).
  const std::size_t encoded_row = k / f.block * f.bytes;
  std::vector<float> row(k);
  double worst = 0.0;
  for (std::size_t o = 0; o < m; o += std::max<std::size_t>(1, m / 97)) {
    gufo::quant::Dequantize(f.type, w.data() + o * encoded_row, row.data(), k);
    for (std::size_t r = 0; r < 2; ++r) {
      double want = 0.0, magnitude = 0.0;
      for (std::size_t c = 0; c < k; ++c) {
        const double p = double{row[c]} * x[r * k + c];
        want += p;
        magnitude += std::fabs(p);
      }
      worst = std::max(worst, std::fabs(gemv[r * m + o] - want) / magnitude);
    }
  }
  Require(worst < 2e-6, name + ": decode error " + std::to_string(worst));

  // The Gemma autoregressive GEMV matches the FP64 dot.
  if (f.gemv) {
    namespace g4k = gufo::models::gemma4::rocm;
    std::vector<float> one(2 * m);
    for (std::size_t r = 0; r < 2; ++r) {
      Require(g4k::LaunchKQuantGemv(*f.gemv, dw, dx + r * k, dy + r * m,
                                    static_cast<std::uint32_t>(m),
                                    static_cast<std::uint32_t>(k), nullptr),
              name + ": Gemma GEMV rejected the shape");
    }
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(one.data(), dy, one.size() * 4, hipMemcpyDeviceToHost));
    double gworst = 0.0;
    for (std::size_t o = 0; o < m; o += std::max<std::size_t>(1, m / 97)) {
      gufo::quant::Dequantize(f.type, w.data() + o * encoded_row, row.data(),
                              k);
      for (std::size_t r = 0; r < 2; ++r) {
        double want = 0.0, magnitude = 0.0;
        for (std::size_t c = 0; c < k; ++c) {
          const double p = double{row[c]} * x[r * k + c];
          want += p;
          magnitude += std::fabs(p);
        }
        gworst = std::max(gworst, std::fabs(one[r * m + o] - want) / magnitude);
      }
    }
    Require(gworst < 2e-6,
            name + ": Gemma GEMV error " + std::to_string(gworst));

    // Cold-weight bandwidth, rotating copies beyond the 32 MiB MALL.
    const std::size_t bytes = m * encoded_row;
    const std::size_t copies = std::max<std::size_t>(2, (256u << 20) / bytes);
    std::vector<std::uint8_t*> rot;
    for (std::size_t c = 0; c < copies; ++c) {
      rot.push_back(Device(w.data(), w.size()));
    }
    const auto time = [&](auto&& launch) {
      for (int warm = 0; warm < 2; ++warm)
        launch(rot[warm % copies]);
      HIP_CHECK(hipDeviceSynchronize());
      const auto t0 = std::chrono::steady_clock::now();
      const int iters = 40;
      for (int i = 0; i < iters; ++i)
        launch(rot[i % copies]);
      HIP_CHECK(hipDeviceSynchronize());
      return std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                           t0)
                 .count() /
             iters;
    };
    const double gemma = time([&](std::uint8_t* wp) {
      (void)g4k::LaunchKQuantGemv(*f.gemv, wp, dx, dy,
                                  static_cast<std::uint32_t>(m),
                                  static_cast<std::uint32_t>(k), nullptr);
    });
    const double qwen = time([&](std::uint8_t* wp) {
      gufo::hip::LaunchGEMV(wp, f.type, dx, dy, m, k, nullptr);
    });
    const double qwen1 = time([&](std::uint8_t* wp) {
      gufo::hip::LaunchBatchedQuantGEMMFp32(f.type, wp, dx, dy, 1, m, k,
                                            nullptr);
    });
    const double qwen5 = time([&](std::uint8_t* wp) {
      gufo::hip::LaunchBatchedQuantGEMMFp32(f.type, wp, dx, dy, 5, m, k,
                                            nullptr);
    });
    std::cout << name << ": one row: Gemma GEMV " << gemma * 1e6
              << " us, shared GEMV " << qwen * 1e6 << " us, shared small-batch "
              << qwen1 * 1e6 << " us; five rows: shared small-batch "
              << qwen5 * 1e6 << " us (" << bytes / 1e6 << " MB)\n";
    for (auto* ptr : rot)
      HIP_CHECK(hipFree(ptr));
  }

  // Prefill W8A8 stays within Q8_1 activation rounding.
  void* dq = nullptr;
  HIP_CHECK(
      hipMalloc(&dq, gufo::hip::QuantizedActivationBytes(kPrefillRows, k)));
  gufo::hip::LaunchQuantizeActivationQ8_1FromFp32(dx, dq, kPrefillRows, k,
                                                  nullptr);
  gufo::hip::LaunchBatchedQuantGEMMPreQuantized(f.type, dw, dq, dy,
                                                kPrefillRows, m, k, nullptr);
  std::vector<float> prefill(kPrefillRows * m);
  HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(
      hipMemcpy(prefill.data(), dy, prefill.size() * 4, hipMemcpyDeviceToHost));
  double err2 = 0.0, ref2 = 0.0;
  for (std::size_t r = 0; r < kRows; ++r) {
    for (std::size_t o = 0; o < m; ++o) {
      const double d = prefill[r * m + o] - gemv[r * m + o];
      err2 += d * d;
      ref2 += double{gemv[r * m + o]} * gemv[r * m + o];
      Require(std::isfinite(prefill[r * m + o]), name + ": non-finite prefill");
    }
  }
  const double rel = std::sqrt(err2 / ref2);
  Require(rel < 2e-2, name + ": prefill relative error " + std::to_string(rel));
  HIP_CHECK(hipFree(dq));
  HIP_CHECK(hipFree(dw));
  HIP_CHECK(hipFree(dx));
  HIP_CHECK(hipFree(dy));
}

}  // namespace

int main() {
  int devices = 0;
  if (hipGetDeviceCount(&devices) != hipSuccess || devices == 0) {
    std::cout << "SKIP: no HIP device\n";
    return 77;
  }
  return gemma4_test::Run([] {
    std::mt19937 rng(11);
    const Format formats[] = {
        {GgmlType::kQ4_K,
         gufo::models::gemma4::rocm::GemvFormat::kQ4_K,
         256,
         144,
         {0, 2},
         "Q4_K"},
        {GgmlType::kQ5_K,
         gufo::models::gemma4::rocm::GemvFormat::kQ5_K,
         256,
         176,
         {0, 2},
         "Q5_K"},
        {GgmlType::kQ6_K,
         gufo::models::gemma4::rocm::GemvFormat::kQ6_K,
         256,
         210,
         {208},
         "Q6_K"},
        {GgmlType::kQ8_0, std::nullopt, 32, 34, {0}, "Q8_0"},
    };
    // (M, K) of every target projection, plus the drafter's widths.
    const std::pair<std::size_t, std::size_t> shapes[] = {
        {8192, 5376},  {4096, 5376},  {16384, 5376}, {2048, 5376},
        {5376, 8192},  {5376, 16384}, {21504, 5376}, {5376, 21504},
        {1024, 10752}, {8192, 1024},  {5376, 1024}};
    for (const auto& f : formats) {
      for (const auto& [m, k] : shapes) {
        CheckShape(f, m, k, rng);
      }
    }
  });
}
