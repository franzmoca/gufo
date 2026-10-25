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
/* Two weights over one input (quantized once) against two separate batched
 * launches: must match exactly. */
bool PairCase(const TypeCase& t, int M1, int M2, int N, int K,
              std::mt19937& rng) {
  const auto w1 =
      RandomBlocks(t, static_cast<std::size_t>(M1) * K / t.block_values, rng);
  const auto w2 =
      RandomBlocks(t, static_cast<std::size_t>(M2) * K / t.block_values, rng);
  std::normal_distribution<float> normal;
  std::vector<float> x(static_cast<std::size_t>(N) * K);
  for (auto& v : x)
    v = normal(rng);
  auto* dw1 = Device(w1);
  auto* dw2 = Device(w2);
  auto* dx = Device(x);
  float *a1 = nullptr, *a2 = nullptr, *b1 = nullptr, *b2 = nullptr;
  (void)hipMalloc(&a1, sizeof(float) * N * M1);
  (void)hipMalloc(&b1, sizeof(float) * N * M1);
  (void)hipMalloc(&a2, sizeof(float) * N * M2);
  (void)hipMalloc(&b2, sizeof(float) * N * M2);
  const bool ok = ds41_mmq_dense_pair_typed(t.type, dw1, a1, M1, dw2, a2, M2,
                                            dx, N, K, 0) == 0 &&
                  ds41_mmq_dense_typed(t.type, dw1, dx, b1, M1, N, K, 0) == 0 &&
                  ds41_mmq_dense_typed(t.type, dw2, dx, b2, M2, N, K, 0) == 0;
  const double e =
      ok ? std::max(RelativeError(Host(a1, static_cast<std::size_t>(N) * M1),
                                  Host(b1, static_cast<std::size_t>(N) * M1)),
                    RelativeError(Host(a2, static_cast<std::size_t>(N) * M2),
                                  Host(b2, static_cast<std::size_t>(N) * M2)))
         : INFINITY;
  std::printf("pair  %-8s M=%d+%d N=%d K=%d rel_err=%.2e\n", t.name, M1, M2, N,
              K, e);
  for (void* p :
       {static_cast<void*>(dw1), static_cast<void*>(dw2),
        static_cast<void*>(dx), static_cast<void*>(a1), static_cast<void*>(a2),
        static_cast<void*>(b1), static_cast<void*>(b2)})
    (void)hipFree(p);
  return e == 0.0;
}

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
  // Batches run every group as an MMQ channel and must match one batched
  // launch per group exactly; small N runs the grouped matvec.
  const bool batched = N > 8;
  bool ok = (batched ? ds41_mmq_dense_grouped_typed(t.type, dw, dx, grouped, M,
                                                    N, K, groups, 0)
                     : ds41_mmq_grouped_vec_typed(t.type, dw, dx, grouped, M, N,
                                                  K, groups, 0)) == 0;
  const std::size_t group_bytes =
      static_cast<std::size_t>(M) * K / t.block_values * t.block_bytes;
  for (int g = 0; g < groups && ok && batched; ++g) {
    std::vector<float> xg(static_cast<std::size_t>(N) * K);
    for (int col = 0; col < N; ++col)
      std::copy_n(x.begin() + (static_cast<std::size_t>(col) * groups + g) * K,
                  K, xg.begin() + static_cast<std::size_t>(col) * K);
    auto* dxg = Device(xg);
    float* yg = nullptr;
    (void)hipMalloc(&yg, sizeof(float) * N * M);
    ok = ds41_mmq_dense_typed(
             t.type, reinterpret_cast<const uint8_t*>(dw) + g * group_bytes,
             dxg, yg, M, N, K, 0) == 0;
    const auto hy = Host(yg, static_cast<std::size_t>(N) * M);
    for (int col = 0; col < N && ok; ++col)
      (void)hipMemcpy(single + (static_cast<std::size_t>(col) * groups + g) * M,
                      hy.data() + static_cast<std::size_t>(col) * M,
                      sizeof(float) * M, hipMemcpyHostToDevice);
    (void)hipFree(dxg);
    (void)hipFree(yg);
  }
  for (int col = 0; col < N && ok && !batched; ++col)
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
  return batched ? e == 0.0 : e < 1e-5;
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

/* The routed gate/up pair with the SwiGLU store against two routed launches
 * and the separate SwiGLU pass, under the decoder's launch bounds. */
bool MoeSwigluCase(const TypeCase& t, int M, int K, int n_tokens, int n_experts,
                   int used, std::mt19937& rng) {
  constexpr float kLimit = 7.0f;
  const std::size_t blocks =
      static_cast<std::size_t>(n_experts) * M * K / t.block_values;
  const auto gate = RandomBlocks(t, blocks, rng);
  const auto up = RandomBlocks(t, blocks, rng);
  std::normal_distribution<float> normal;
  std::vector<float> x(static_cast<std::size_t>(n_tokens) * K);
  for (auto& v : x)
    v = normal(rng);
  const int rows = n_tokens * used;
  std::vector<float> route(rows);
  for (auto& v : route)
    v = 1.0f + 0.1f * normal(rng);
  std::vector<int32_t> ids(rows);
  std::vector<unsigned int> counts(n_experts, 0u);
  unsigned int most = 0;
  for (int token = 0; token < n_tokens; ++token)
    for (int s = 0; s < used; ++s) {
      const int e = (token % 3 + s * 7 + (rng() % 2) * 31) % n_experts;
      ids[token * used + s] = e;
      most = std::max(most, ++counts[e]);
    }
  auto* dg = Device(gate);
  auto* du = Device(up);
  auto* dx = Device(x);
  auto* dids = Device(ids);
  auto* droute = Device(route);
  const std::size_t n_out = static_cast<std::size_t>(M) * rows;
  float *g = nullptr, *u = nullptr, *want = nullptr, *g2 = nullptr,
        *got = nullptr;
  for (float** p : {&g, &u, &want, &g2, &got})
    (void)hipMalloc(p, sizeof(float) * n_out);
  ds41_mmq_set_routed_max_expert_rows(static_cast<int>(most));
  ds41_mmq_set_routed_tile_cols(
      ds41_mmq_routed_tile_cols_for_counts(counts.data(), n_experts));
  bool ok =
      ds41_mmq_moe_typed(t.type, dg, dx, dids, g, M, K, n_tokens, n_experts,
                         used, 0) == 0 &&
      ds41_mmq_moe_typed(t.type, du, dx, dids, u, M, K, n_tokens, n_experts,
                         used, 0) == 0 &&
      ds41_v41_swiglu(g, u, droute, want, rows, M, kLimit) == 0 &&
      ds41_mmq_moe_swiglu_typed(t.type, dg, du, dx, dids, droute, g2, got, M, K,
                                n_tokens, n_experts, used, kLimit, 0) == 0;
  ds41_mmq_set_routed_max_expert_rows(0);
  ds41_mmq_set_routed_tile_cols(0);
  double eg = INFINITY, e = INFINITY;
  if (ok) {
    eg = RelativeError(Host(g2, n_out), Host(g, n_out));
    e = RelativeError(Host(got, n_out), Host(want, n_out));
  }
  std::printf(
      "moe swiglu %-8s M=%d K=%d tokens=%d used=%d gate_err=%.2e "
      "rel_err=%.2e\n",
      t.name, M, K, n_tokens, used, eg, e);
  for (void* p :
       {static_cast<void*>(dg), static_cast<void*>(du), static_cast<void*>(dx),
        static_cast<void*>(dids), static_cast<void*>(droute),
        static_cast<void*>(g), static_cast<void*>(u), static_cast<void*>(want),
        static_cast<void*>(g2), static_cast<void*>(got)})
    (void)hipFree(p);
  return eg == 0.0 && e < 1e-6;
}

/* Routed launches split into per-expert tile-width classes against one
 * launch: same rows, rounding-level differences at most. Buckets from 1 to
 * ~150 rows so every class runs. */
bool MoeClassCase(const TypeCase& t, int M, int K, int n_tokens, int n_experts,
                  int used, std::mt19937& rng) {
  const auto w = RandomBlocks(
      t, static_cast<std::size_t>(n_experts) * M * K / t.block_values, rng);
  std::normal_distribution<float> normal;
  std::vector<float> x(static_cast<std::size_t>(n_tokens) * K);
  for (auto& v : x)
    v = normal(rng);
  std::vector<int32_t> ids(static_cast<std::size_t>(n_tokens) * used);
  std::vector<unsigned int> counts(n_experts, 0u);
  unsigned int most = 0;
  for (int token = 0; token < n_tokens; ++token)
    for (int s = 0; s < used; ++s) {
      // Six experts take ~150 rows (64-column tiles), thirty ~30 rows
      // (32 columns) and the rest a few rows each (16 columns).
      const int e = token % 4 == 0   ? s * 5
                    : token % 4 == 1 ? 40 + (token / 4) % 5 * 6 + s
                                     : 80 + (token * 7 + s * 13) % 300;
      ids[token * used + s] = e;
      most = std::max(most, ++counts[e]);
    }
  auto* dw = Device(w);
  auto* dx = Device(x);
  auto* dids = Device(ids);
  const std::size_t n_out = static_cast<std::size_t>(M) * n_tokens * used;
  float *one = nullptr, *split = nullptr;
  (void)hipMalloc(&one, sizeof(float) * n_out);
  (void)hipMalloc(&split, sizeof(float) * n_out);
  (void)hipMemset(split, 0xff, sizeof(float) * n_out);
  std::vector<uint8_t> classes(n_experts);
  int widths[3], rows[3];
  const int n_classes = ds41_mmq_routed_classes_for_counts(
      counts.data(), n_experts, classes.data(), widths, rows);
  auto* dclasses = Device(classes);
  ds41_mmq_set_routed_max_expert_rows(static_cast<int>(most));
  bool ok = ds41_mmq_moe_typed(t.type, dw, dx, dids, one, M, K, n_tokens,
                               n_experts, used, 0) == 0;
  ds41_mmq_set_routed_classes(dclasses, n_classes, widths, rows);
  ok = ok && ds41_mmq_moe_typed(t.type, dw, dx, dids, split, M, K, n_tokens,
                                n_experts, used, 0) == 0;
  ds41_mmq_set_routed_classes(nullptr, 0, nullptr, nullptr);
  ds41_mmq_set_routed_max_expert_rows(0);
  const double e =
      ok ? RelativeError(Host(split, n_out), Host(one, n_out)) : INFINITY;
  std::printf(
      "moe classes %-8s M=%d K=%d tokens=%d rows/class %d/%d/%d "
      "rel_err=%.2e\n",
      t.name, M, K, n_tokens, rows[0], rows[1], rows[2], e);
  for (void* p : {static_cast<void*>(dw), static_cast<void*>(dx),
                  static_cast<void*>(dids), static_cast<void*>(dclasses),
                  static_cast<void*>(one), static_cast<void*>(split)})
    (void)hipFree(p);
  return n_classes == 3 && rows[0] > 0 && rows[1] > 0 && rows[2] > 0 &&
         e < 1e-6;
}

/* Routed-expert GEMV against the ggml matvec (Q8_1 activations, so the
 * agreement is at that rounding, ~5e-3): one shared input (gate/up) or one
 * input per slot (down), with a dropped slot (id -1) that must write 0. */
bool ExpertGemvCase(const TypeCase& t, int M, int K, bool per_slot,
                    std::mt19937& rng) {
  constexpr int kExperts = 24;
  constexpr int kSlots = 6;
  const std::size_t expert_bytes =
      static_cast<std::size_t>(M) * K / t.block_values * t.block_bytes;
  const auto w = RandomBlocks(
      t, static_cast<std::size_t>(kExperts) * M * K / t.block_values, rng);
  std::normal_distribution<float> normal;
  std::vector<float> x(static_cast<std::size_t>(per_slot ? kSlots : 1) * K);
  for (auto& v : x)
    v = normal(rng);
  const std::vector<int32_t> ids = {3, 17, 0, 23, -1, 9};
  std::vector<int32_t> live = ids;
  live[4] = 0;
  auto* dw = Device(w);
  auto* dx = Device(x);
  auto* dids = Device(ids);
  auto* dlive = Device(live);
  const std::size_t n_out = static_cast<std::size_t>(M) * kSlots;
  float *got = nullptr, *want = nullptr;
  (void)hipMalloc(&got, sizeof(float) * n_out);
  (void)hipMalloc(&want, sizeof(float) * n_out);
  bool ok = ds41_v41_gemv_experts(t.type, dw, expert_bytes, dids, kSlots, dx,
                                  per_slot ? K : 0, got, M, K) == 0;
  ok =
      ok && (per_slot ? ds41_mmq_moe_vec_typed(t.type, dw, dx, dlive, want, M,
                                               K, kSlots, kExperts, 1, 0)
                      : ds41_mmq_moe_vec_typed(t.type, dw, dx, dlive, want, M,
                                               K, 1, kExperts, kSlots, 0)) == 0;
  std::vector<float> g = ok ? Host(got, n_out) : std::vector<float>();
  std::vector<float> r = ok ? Host(want, n_out) : std::vector<float>();
  bool dropped_zero = ok;
  for (int i = 0; ok && i < M; ++i) {
    dropped_zero = dropped_zero && g[4 * M + i] == 0.0f;
    g[4 * M + i] = r[4 * M + i] = 0.0f;
  }
  const double e = ok ? RelativeError(g, r) : INFINITY;
  std::printf("experts %-8s M=%d K=%d per_slot=%d rel_err=%.2e dropped=%s\n",
              t.name, M, K, per_slot, e, dropped_zero ? "0" : "NONZERO");
  (void)hipFree(dw);
  (void)hipFree(dx);
  (void)hipFree(dids);
  (void)hipFree(dlive);
  (void)hipFree(got);
  (void)hipFree(want);
  return e < 2e-2 && dropped_zero;
}

/* Fused gate/up SwiGLU (dense GEMV for Q5_K/Q8_0, routed experts for the IQ
 * types) against the two unfused matvecs and a host SwiGLU. */
bool SwigluCase(const TypeCase& t, int M, int K, std::mt19937& rng) {
  constexpr int kExperts = 8;
  constexpr int kSlots = 6;
  constexpr float kLimit = 7.0f;
  const bool experts = t.type != 8 && t.type != 13;
  const int matrices = experts ? kExperts : 1;
  const int slots = experts ? kSlots : 1;
  const std::size_t expert_bytes =
      static_cast<std::size_t>(M) * K / t.block_values * t.block_bytes;
  const std::size_t blocks =
      static_cast<std::size_t>(matrices) * M * K / t.block_values;
  const auto gate = RandomBlocks(t, blocks, rng);
  const auto up = RandomBlocks(t, blocks, rng);
  std::normal_distribution<float> normal;
  std::vector<float> x(K), route(slots);
  for (auto& v : x)
    v = normal(rng);
  for (auto& v : route)
    v = 1.0f + 0.1f * normal(rng);
  const std::vector<int32_t> ids = {5, 1, -1, 7, 0, 2};
  auto* dg = Device(gate);
  auto* du = Device(up);
  auto* dx = Device(x);
  auto* dids = Device(ids);
  auto* droute = Device(route);
  const std::size_t n_out = static_cast<std::size_t>(M) * slots;
  float *g = nullptr, *u = nullptr, *y = nullptr;
  (void)hipMalloc(&g, sizeof(float) * n_out);
  (void)hipMalloc(&u, sizeof(float) * n_out);
  (void)hipMalloc(&y, sizeof(float) * n_out);
  bool ok;
  if (experts) {
    ok = ds41_v41_gemv_experts(t.type, dg, expert_bytes, dids, slots, dx, 0, g,
                               M, K) == 0 &&
         ds41_v41_gemv_experts(t.type, du, expert_bytes, dids, slots, dx, 0, u,
                               M, K) == 0 &&
         ds41_v41_gemv_experts_swiglu(t.type, dg, du, expert_bytes, dids, slots,
                                      dx, droute, kLimit, y, M, K) == 0;
  } else {
    ok = ds41_v41_gemv(t.type, dg, dx, g, M, K, 0, 4) == 0 &&
         ds41_v41_gemv(t.type, du, dx, u, M, K, 0, 4) == 0 &&
         ds41_v41_gemv_swiglu(t.type, dg, du, dx, y, M, K, kLimit, 4) == 0;
  }
  double e = INFINITY;
  if (ok) {
    const auto hg = Host(g, n_out);
    const auto hu = Host(u, n_out);
    const auto got = Host(y, n_out);
    std::vector<float> want(n_out);
    for (std::size_t i = 0; i < n_out; ++i) {
      const float gv = std::min(hg[i], kLimit);
      const float uv = std::min(std::max(hu[i], -kLimit), kLimit);
      const float h = gv / (1.0f + std::exp(-gv)) * uv;
      want[i] = experts ? route[i / M] * h : h;
    }
    e = RelativeError(got, want);
  }
  std::printf("swiglu %-8s M=%d K=%d rel_err=%.2e\n", t.name, M, K, e);
  for (void* p :
       {static_cast<void*>(dg), static_cast<void*>(du), static_cast<void*>(dx),
        static_cast<void*>(dids), static_cast<void*>(droute),
        static_cast<void*>(g), static_cast<void*>(u), static_cast<void*>(y)})
    (void)hipFree(p);
  return e < 1e-6;
}

/* The FP16 matrix-core dense kernel against the FP32-activation GEMV per
 * token, over `groups` matrices with their own input columns. */
bool DenseF16Case(const TypeCase& t, int M, int N, int K, int groups,
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
  float *got = nullptr, *want = nullptr;
  (void)hipMalloc(&got, sizeof(float) * n_out);
  (void)hipMalloc(&want, sizeof(float) * n_out);
  const std::size_t matrix_bytes =
      static_cast<std::size_t>(M) * K / t.block_values * t.block_bytes;
  bool ok = ds41_v41_dense_f16(t.type, dw, dx, got, M, N, K, groups) == 0;
  for (int token = 0; ok && token < N; ++token)
    for (int z = 0; ok && z < groups; ++z)
      ok = ds41_v41_gemv(
               t.type,
               static_cast<const uint8_t*>(static_cast<void*>(dw)) +
                   z * matrix_bytes,
               dx + (static_cast<std::size_t>(token) * groups + z) * K,
               want + (static_cast<std::size_t>(token) * groups + z) * M, M, K,
               0, 4) == 0;
  const double e =
      ok ? RelativeError(Host(got, n_out), Host(want, n_out)) : INFINITY;
  std::printf("dense f16 %-8s M=%d N=%d K=%d groups=%d rel_err=%.2e\n", t.name,
              M, N, K, groups, e);
  (void)hipFree(dw);
  (void)hipFree(dx);
  (void)hipFree(got);
  (void)hipFree(want);
  return e < 2e-3;
}

/* The FP16 matrix-core routed kernel against the FP32-activation expert
 * GEMV: the gate/up pair with its SwiGLU store (FP16 or FP32) over
 * per-token inputs, and the single projection over per-slot inputs. */
bool RoutedF16Case(const TypeCase& t, int M, int K, int n_tokens, int n_experts,
                   int used, bool pair, std::mt19937& rng) {
  constexpr float kLimit = 7.0f;
  const std::size_t expert_bytes =
      static_cast<std::size_t>(M) * K / t.block_values * t.block_bytes;
  const std::size_t blocks =
      static_cast<std::size_t>(n_experts) * M * K / t.block_values;
  const auto gate = RandomBlocks(t, blocks, rng);
  const auto up = RandomBlocks(t, blocks, rng);
  const int rows = n_tokens * used;
  const int x_rows = pair ? n_tokens : rows;
  std::normal_distribution<float> normal;
  std::vector<float> x(static_cast<std::size_t>(x_rows) * K), route(rows);
  for (auto& v : x)
    v = normal(rng);
  for (auto& v : route)
    v = 1.0f + 0.1f * normal(rng);
  /* Distinct experts per token, as top-k routing gives. */
  std::vector<int32_t> ids(rows);
  for (int token = 0; token < n_tokens; ++token)
    for (int s = 0; s < used; ++s)
      ids[token * used + s] =
          (token % 3 + s * 7 + static_cast<int>(rng() % 2) * 31) % n_experts;
  std::vector<int32_t> plan(rows + (n_experts + 1) + rows / 64 + n_experts);
  int32_t* bounds = plan.data() + rows;
  const uint32_t n_tiles = ds41_v41_routed_plan(
      ids.data(), rows, n_experts, plan.data(), bounds, bounds + n_experts + 1);
  /* The reference reads one FP32 input row per slot. */
  std::vector<float> xs(static_cast<std::size_t>(rows) * K);
  for (int slot = 0; slot < rows; ++slot)
    std::memcpy(&xs[static_cast<std::size_t>(slot) * K],
                &x[static_cast<std::size_t>(pair ? slot / used : slot) * K],
                K * sizeof(float));
  auto* dg = Device(gate);
  auto* du = Device(up);
  auto* dx = Device(x);
  auto* dxs = Device(xs);
  auto* dids = Device(ids);
  auto* dplan = Device(plan);
  auto* droute = Device(route);
  const std::size_t n_out = static_cast<std::size_t>(M) * rows;
  float *g = nullptr, *u = nullptr, *y = nullptr;
  uint16_t *x16 = nullptr, *y16 = nullptr;
  (void)hipMalloc(&g, sizeof(float) * n_out);
  (void)hipMalloc(&u, sizeof(float) * n_out);
  (void)hipMalloc(&y, sizeof(float) * n_out);
  (void)hipMalloc(&y16, sizeof(uint16_t) * n_out);
  (void)hipMalloc(&x16, sizeof(uint16_t) * x.size());
  const int32_t* dbounds = dplan + rows;
  bool ok = n_tiles > 0 &&
            ds41_v41_gemv_experts(t.type, dg, expert_bytes, dids, rows, dxs, K,
                                  g, M, K) == 0 &&
            ds41_v41_gemv_experts(t.type, du, expert_bytes, dids, rows, dxs, K,
                                  u, M, K) == 0 &&
            ds41_v41_to_half(dx, x16, x.size()) == 0;
  if (pair)
    ok = ok &&
         ds41_v41_routed_f16(t.type, dg, du, expert_bytes, x16, used, dplan,
                             dbounds, dbounds + n_experts + 1, n_tiles, droute,
                             kLimit, y, 0, M, K) == 0 &&
         ds41_v41_routed_f16(t.type, dg, du, expert_bytes, x16, used, dplan,
                             dbounds, dbounds + n_experts + 1, n_tiles, droute,
                             kLimit, y16, 1, M, K) == 0;
  else
    ok = ok && ds41_v41_routed_f16(t.type, dg, nullptr, expert_bytes, x16, 1,
                                   dplan, dbounds, dbounds + n_experts + 1,
                                   n_tiles, nullptr, 0.0f, y, 0, M, K) == 0;
  double e = INFINITY, e16 = 0.0;
  if (ok) {
    const auto hg = Host(g, n_out);
    const auto hu = Host(u, n_out);
    std::vector<float> want(n_out);
    for (std::size_t i = 0; i < n_out; ++i) {
      const float gv = std::min(hg[i], kLimit);
      const float uv = std::min(std::max(hu[i], -kLimit), kLimit);
      want[i] =
          pair ? route[i / M] * (gv / (1.0f + std::exp(-gv)) * uv) : hg[i];
    }
    e = RelativeError(Host(y, n_out), want);
    if (pair) {
      std::vector<_Float16> h(n_out);
      std::vector<float> got16(n_out);
      (void)hipMemcpy(h.data(), y16, n_out * sizeof(uint16_t),
                      hipMemcpyDeviceToHost);
      for (std::size_t i = 0; i < n_out; ++i)
        got16[i] = static_cast<float>(h[i]);
      e16 = RelativeError(got16, want);
    }
  }
  std::printf(
      "routed f16 %-8s M=%d K=%d tokens=%d experts=%d pair=%d tiles=%u "
      "rel_err=%.2e half_out_err=%.2e\n",
      t.name, M, K, n_tokens, n_experts, pair, n_tiles, e, e16);
  for (void* p :
       {static_cast<void*>(dg), static_cast<void*>(du), static_cast<void*>(dx),
        static_cast<void*>(dxs), static_cast<void*>(dids),
        static_cast<void*>(dplan), static_cast<void*>(droute),
        static_cast<void*>(g), static_cast<void*>(u), static_cast<void*>(y),
        static_cast<void*>(y16), static_cast<void*>(x16)})
    (void)hipFree(p);
  return e < 2e-3 && e16 < 2e-3;
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
    if (t.type == 13) {
      ok = DenseF16Case(t, 256, 37, 1280, 1, rng) && ok;
      ok = DenseF16Case(t, 512, 300, 2304, 1, rng) && ok;
      ok = DenseF16Case(t, 256, 130, 512, 4, rng) && ok;
    }
    if (t.type == 8 || t.type == 13) {
      ok = GemvCase(t, 300, 1280, 0, rng) && ok;
      ok = GemvCase(t, 256, 4096, 64, rng) && ok;
      ok = SwigluCase(t, 2304, 5120, rng) && ok;
    }
    if (t.type != 8 && t.type != 13) {
      ok = ExpertGemvCase(t, 2304, 5120, false, rng) && ok;
      ok = ExpertGemvCase(t, 512, 2304, true, rng) && ok;
      if (t.type == 19 || t.type == 29 || t.type == 16)
        ok = SwigluCase(t, 2304, 5120, rng) && ok;
    }
    ok = GroupedCase(t, 128, 1, 1024, 8, rng) && ok;
    ok = GroupedCase(t, 256, 3, 512, 8, rng) && ok;
    ok = GroupedCase(t, 256, 37, 512, 8, rng) && ok;
    ok = PairCase(t, 320, 128, 37, 1024, rng) && ok;
    for (const bool bounded : {false, true}) {
      ok = MoeCase(t, 256, 1024, 19, 16, 6, bounded, rng) && ok;
      ok = MoeCase(t, 128, 2304, 23, 16, 1, bounded, rng) && ok;
      ok = MoeCase(t, 256, 1024, 200, 384, 6, bounded, rng) && ok;
    }
    ok = MoeSwigluCase(t, 256, 1024, 200, 384, 6, rng) && ok;
    if (t.type == 19 || t.type == 29 || t.type == 16) {
      for (const int n_experts : {16, 384}) {
        ok = RoutedF16Case(t, 256, 1024, 200, n_experts, 6, true, rng) && ok;
        ok = RoutedF16Case(t, 512, 2304, 200, n_experts, 6, false, rng) && ok;
      }
    }
    ok = MoeClassCase(t, 256, 1024, 600, 384, 6, rng) && ok;
  }
  ds41_mmq_cleanup();
  std::printf("%s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
