// Batched mmq GEMMs against the mmvq matvec path, which the decoder parity
// runs validate against the oracle, for every V4.1 Flash weight type.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "ds41_mmq.h"
#include "ds41_v41.h"
#include "src/core/quant/ggml_dequant.hpp"

namespace {

struct TypeCase {
  const char* name;
  int type;
  int block_bytes;
  int block_values;
};

constexpr TypeCase kTypes[] = {
    {"Q8_0", 8, 34, 32},    {"Q5_K", 13, 176, 256},   {"IQ1_S", 19, 50, 256},
    {"IQ1_M", 29, 56, 256}, {"IQ2_XXS", 16, 66, 256}, {"IQ2_S", 22, 82, 256},
};

uint16_t RandomHalf(std::mt19937& rng) {
  // Normal f16 scales in [2^-8, 2^-5) with a random sign.
  const uint16_t exponent = static_cast<uint16_t>(7 + rng() % 3);
  return static_cast<uint16_t>((rng() & 1) << 15 | exponent << 10 |
                               (rng() & 0x3ff));
}

std::vector<uint8_t> RandomBlocks(const TypeCase& t, std::size_t blocks,
                                  std::mt19937& rng) {
  std::vector<uint8_t> data(blocks * t.block_bytes);
  for (auto& byte : data)
    byte = static_cast<uint8_t>(rng());
  for (std::size_t b = 0; b < blocks; ++b) {
    uint8_t* block = data.data() + b * t.block_bytes;
    auto put = [&](int offset, uint16_t value) {
      std::memcpy(block + offset, &value, 2);
    };
    if (t.type == 29) {
      // IQ1_M keeps its f16 scale in the top nibbles of the four scale words.
      const uint16_t d = RandomHalf(rng) & 0x7fff;
      for (int i = 0; i < 4; ++i) {
        uint16_t word;
        std::memcpy(&word, block + 48 + 2 * i, 2);
        word = static_cast<uint16_t>((word & 0x0fff) | ((d >> (4 * i)) & 0xf)
                                                           << 12);
        put(48 + 2 * i, word);
      }
    } else {
      put(0, RandomHalf(rng));
      if (t.type == 13)
        put(2, RandomHalf(rng));  // Q5_K dmin
    }
  }
  return data;
}

template<typename T>
T* Device(const std::vector<T>& host) {
  void* p = nullptr;
  if (hipMalloc(&p, host.size() * sizeof(T)) != hipSuccess)
    return nullptr;
  (void)hipMemcpy(p, host.data(), host.size() * sizeof(T),
                  hipMemcpyHostToDevice);
  return static_cast<T*>(p);
}

std::vector<float> Host(const float* device, std::size_t n) {
  std::vector<float> out(n);
  (void)hipDeviceSynchronize();
  (void)hipMemcpy(out.data(), device, n * sizeof(float), hipMemcpyDeviceToHost);
  return out;
}

/* Relative RMS error of `got` against `want`. */
double RelativeError(const std::vector<float>& got,
                     const std::vector<float>& want) {
  double err = 0.0, ref = 0.0;
  for (std::size_t i = 0; i < want.size(); ++i) {
    if (!std::isfinite(got[i]))
      return INFINITY;
    err += (got[i] - want[i]) * (got[i] - want[i]);
    ref += want[i] * want[i];
  }
  return std::sqrt(err / std::max(ref, 1e-30));
}

bool DenseCase(const TypeCase& t, int M, int N, int K, std::mt19937& rng) {
  const auto w =
      RandomBlocks(t, static_cast<std::size_t>(M) * K / t.block_values, rng);
  std::normal_distribution<float> normal;
  std::vector<float> x(static_cast<std::size_t>(N) * K);
  for (auto& v : x)
    v = normal(rng);
  auto* dw = Device(w);
  auto* dx = Device(x);
  float *batched = nullptr, *vec = nullptr;
  (void)hipMalloc(&batched, sizeof(float) * M * N);
  (void)hipMalloc(&vec, sizeof(float) * M * N);
  bool ok = ds41_mmq_dense_typed(t.type, dw, dx, batched, M, N, K, 0) == 0;
  for (int col = 0; col < N && ok; ++col)
    ok = ds41_mmq_dense_vec_typed(
             t.type, dw, dx + static_cast<std::size_t>(col) * K,
             vec + static_cast<std::size_t>(col) * M, M, 1, K, 0) == 0;
  const double e =
      ok ? RelativeError(Host(batched, static_cast<std::size_t>(M) * N),
                         Host(vec, static_cast<std::size_t>(M) * N))
         : INFINITY;
  std::printf("dense %-8s M=%d N=%d K=%d rel_err=%.2e\n", t.name, M, N, K, e);
  (void)hipFree(dw);
  (void)hipFree(dx);
  (void)hipFree(batched);
  (void)hipFree(vec);
  return e < 2e-3;
}

/* The single-token FP32-activation GEMV against a double-precision dot of
 * the dequantized rows, for every slice count and a grouped layout. */
bool GemvCase(const TypeCase& t, int M, int K, int rows_per_x,
              std::mt19937& rng) {
  const auto w =
      RandomBlocks(t, static_cast<std::size_t>(M) * K / t.block_values, rng);
  const int x_groups = rows_per_x ? M / rows_per_x : 1;
  std::normal_distribution<float> normal;
  std::vector<float> x(static_cast<std::size_t>(x_groups) * K);
  for (auto& v : x)
    v = normal(rng);
  std::vector<float> row(K);
  std::vector<double> want(M);
  const std::size_t row_bytes =
      static_cast<std::size_t>(K) / t.block_values * t.block_bytes;
  for (int r = 0; r < M; ++r) {
    if (t.type == 13)
      gufo::quant::DequantizeQ5_K(w.data() + r * row_bytes, row.data(), K);
    else
      gufo::quant::DequantizeQ8_0(w.data() + r * row_bytes, row.data(), K);
    const float* xr =
        x.data() +
        static_cast<std::size_t>(rows_per_x ? r / rows_per_x : 0) * K;
    double dot = 0.0;
    for (int i = 0; i < K; ++i)
      dot += static_cast<double>(row[i]) * xr[i];
    want[r] = dot;
  }
  auto* dw = Device(w);
  auto* dx = Device(x);
  float* dy = nullptr;
  (void)hipMalloc(&dy, sizeof(float) * M);
  bool ok = true;
  for (const int slices : {1, 2, 4, 8}) {
    double e = INFINITY;
    if (ds41_v41_gemv(t.type, dw, dx, dy, M, K, rows_per_x, slices) == 0) {
      const auto got = Host(dy, M);
      double err = 0.0, ref = 0.0;
      for (int r = 0; r < M; ++r) {
        err += (got[r] - want[r]) * (got[r] - want[r]);
        ref += want[r] * want[r];
      }
      e = std::sqrt(err / std::max(ref, 1e-30));
    }
    std::printf("gemv  %-8s M=%d K=%d rows_per_x=%d slices=%d rel_err=%.2e\n",
                t.name, M, K, rows_per_x, slices, e);
    ok = e < 1e-5 && ok;
  }
  (void)hipFree(dw);
  (void)hipFree(dx);
  (void)hipFree(dy);
  return ok;
}

/* One grouped launch against a matvec per group. */
bool GroupedCase(const TypeCase& t, int M, int N, int K, int groups,
                 std::mt19937& rng) {
  const auto w = RandomBlocks(
      t, static_cast<std::size_t>(groups) * M * K / t.block_values, rng);
  std::normal_distribution<float> normal;
  std::vector<float> x(static_cast<std::size_t>(N) * groups * K);
  for (auto& v : x)
    v = normal(rng);
  auto* dw = Device(w);
  auto* dx = Device(x);
  const std::size_t n_out = static_cast<std::size_t>(N) * groups * M;
  float *grouped = nullptr, *single = nullptr;
  (void)hipMalloc(&grouped, sizeof(float) * n_out);
  (void)hipMalloc(&single, sizeof(float) * n_out);
  bool ok = ds41_mmq_grouped_vec_typed(t.type, dw, dx, grouped, M, N, K, groups,
                                       0) == 0;
  const std::size_t group_bytes =
      static_cast<std::size_t>(M) * K / t.block_values * t.block_bytes;
  for (int col = 0; col < N && ok; ++col)
    for (int g = 0; g < groups && ok; ++g)
      ok = ds41_mmq_dense_vec_typed(
               t.type, reinterpret_cast<const uint8_t*>(dw) + g * group_bytes,
               dx + (static_cast<std::size_t>(col) * groups + g) * K,
               single + (static_cast<std::size_t>(col) * groups + g) * M, M, 1,
               K, 0) == 0;
  const auto a = Host(grouped, n_out);
  const auto b = Host(single, n_out);
  // Multi-column launches may pick a different row tiling than N = 1, so
  // only rounding may differ.
  const double e = ok ? RelativeError(a, b) : INFINITY;
  std::printf("group %-8s M=%d N=%d K=%d groups=%d rel_err=%.2e\n", t.name, M,
              N, K, groups, e);
  (void)hipFree(dw);
  (void)hipFree(dx);
  (void)hipFree(grouped);
  (void)hipFree(single);
  return e < 1e-5;
}

bool MoeCase(const TypeCase& t, int M, int K, int n_tokens, int n_experts,
             int used, bool bounded, std::mt19937& rng) {
  const auto w = RandomBlocks(
      t, static_cast<std::size_t>(n_experts) * M * K / t.block_values, rng);
  std::normal_distribution<float> normal;
  std::vector<float> x(static_cast<std::size_t>(n_tokens) * K);
  for (auto& v : x)
    v = normal(rng);
  std::vector<int32_t> ids(static_cast<std::size_t>(n_tokens) * used);
  for (int token = 0; token < n_tokens; ++token)
    for (int s = 0; s < used;
         ++s)  // distinct experts per token, skewed to share some
      ids[token * used + s] = static_cast<int32_t>(
          (token % 3 + s * 7 + (rng() % 2) * 31) % n_experts);
  auto* dw = Device(w);
  auto* dx = Device(x);
  auto* dids = Device(ids);
  const std::size_t n_out = static_cast<std::size_t>(M) * n_tokens * used;
  float *batched = nullptr, *vec = nullptr;
  (void)hipMalloc(&batched, sizeof(float) * n_out);
  (void)hipMalloc(&vec, sizeof(float) * n_out);
  bool ok = true;
  if (bounded) {
    // The decoder's launch bounds: the largest bucket and the tile width
    // modeled from the bucket distribution.
    std::vector<unsigned int> counts(n_experts, 0u);
    unsigned int most = 0;
    for (const int32_t e : ids)
      most = std::max(most, ++counts[e]);
    ds41_mmq_set_routed_max_expert_rows(static_cast<int>(most));
    ds41_mmq_set_routed_tile_cols(
        ds41_mmq_routed_tile_cols_for_counts(counts.data(), n_experts));
  }
  ok = ds41_mmq_moe_typed(t.type, dw, dx, dids, batched, M, K, n_tokens,
                          n_experts, used, 0) == 0;
  ds41_mmq_set_routed_max_expert_rows(0);
  ds41_mmq_set_routed_tile_cols(0);
  for (int token = 0; token < n_tokens && ok; ++token)
    ok = ds41_mmq_moe_vec_typed(
             t.type, dw, dx + static_cast<std::size_t>(token) * K,
             dids + token * used,
             vec + static_cast<std::size_t>(token) * used * M, M, K, 1,
             n_experts, used, 0) == 0;
  const double e =
      ok ? RelativeError(Host(batched, n_out), Host(vec, n_out)) : INFINITY;
  std::printf(
      "moe   %-8s M=%d K=%d tokens=%d used=%d bounded=%d rel_err=%.2e\n",
      t.name, M, K, n_tokens, used, bounded, e);
  (void)hipFree(dw);
  (void)hipFree(dx);
  (void)hipFree(dids);
  (void)hipFree(batched);
  (void)hipFree(vec);
  return e < 2e-3;
}

}  // namespace

int main() {
  int devices = 0;
  if (hipGetDeviceCount(&devices) != hipSuccess || devices == 0) {
    std::printf("no HIP device\n");
    return 77;
  }
  if (ds41_mmq_init(0) != 0)
    return 1;
  std::mt19937 rng(41);
  bool ok = true;
  for (const auto& t : kTypes) {
    ok = DenseCase(t, 256, 37, 1024, rng) && ok;
    ok = DenseCase(t, 384, 130, 2304, rng) && ok;
    if (t.type == 8 || t.type == 13) {
      ok = GemvCase(t, 300, 1280, 0, rng) && ok;
      ok = GemvCase(t, 256, 4096, 64, rng) && ok;
    }
    ok = GroupedCase(t, 128, 1, 1024, 8, rng) && ok;
    ok = GroupedCase(t, 256, 3, 512, 8, rng) && ok;
    for (const bool bounded : {false, true}) {
      ok = MoeCase(t, 256, 1024, 19, 16, 6, bounded, rng) && ok;
      ok = MoeCase(t, 128, 2304, 23, 16, 1, bounded, rng) && ok;
      ok = MoeCase(t, 256, 1024, 200, 384, 6, bounded, rng) && ok;
    }
  }
  ds41_mmq_cleanup();
  std::printf("%s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
