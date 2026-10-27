/* Dense Q5_K FP16 prefill kernel variants: double-buffered activation LDS
 * (one barrier per K stage instead of two; the second buffer lets stage g's
 * stores overlap stage g-1's fragment reads) and a 64-token tile for
 * comparison. Includes the production translation unit for the reference
 * timing; every variant must reproduce the production outputs bit for bit
 * (same WMMA fragments, same order). */
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "src/models/deepseek_v41_flash/kernels/rocm/ds41_dense_f16.hip.cpp"

namespace tune {
namespace {

/* The production TU's helpers (load_q5, q5_affine, q5_decode8, Q5Lane,
 * lds_barrier, finite_or_zero and the half/float vector typedefs) are
 * anonymous-namespace members of this translation unit after the include:
 * visible here without qualification. */
constexpr int kVStageK = 64;
constexpr int kVPitch = kVStageK + 8;

template <bool kFull, int kTileT, bool kDBuf>
__global__ void __launch_bounds__(kTileT* 4)
    dense_kernel(const uint8_t* __restrict__ w, const float* __restrict__ x, float* __restrict__ out, uint32_t m,
                 uint32_t n, uint32_t k, uint32_t groups, uint32_t tile0) {
  constexpr int kRows = 256;
  constexpr int kThreads = kTileT * 4;
  constexpr int kPitch = kVPitch;
  constexpr int kTiles = kTileT / 16;
  __shared__ __attribute__((aligned(16))) _Float16 act[kDBuf ? 2 : 1][kTileT * kPitch];
  const int tid = threadIdx.x, wave = tid / 32, lane = tid % 32, h = lane / 16;
  const uint32_t z = blockIdx.z;
  const uint64_t row_bytes = (uint64_t)k / 256 * kQ5Bytes;
  const uint32_t row = blockIdx.x * kRows + wave * 16 + lane % 16;
  const uint8_t* wr = w + ((uint64_t)z * m + row) * row_bytes;
  const uint32_t t0 = (tile0 + blockIdx.y) * kTileT;
  const int live = (int)min<uint32_t>(kTiles, (n - t0 + 15) / 16);
  const uint64_t x_stride = (uint64_t)groups * k;

  constexpr int kPer = kVStageK * kTileT / kThreads;
  const uint32_t st = tid % kTileT, sh = tid / kTileT;
  const float* xs = kFull || t0 + st < n ? x + (t0 + st) * x_stride + (uint64_t)z * k + kPer * sh : nullptr;
  _Float16* as0 = act[0] + st * kPitch + kPer * sh;
  _Float16* as1 = act[kDBuf ? 1 : 0] + st * kPitch + kPer * sh;
  float4 pre[kPer / 4], pre2[kPer / 4];
  const auto fetch = [&](uint32_t k0) {
    if (kFull || xs) {
#pragma unroll
      for (int c = 0; c < kPer / 4; ++c) pre2[c] = reinterpret_cast<const float4*>(xs + k0)[c];
    }
  };
  const auto store = [&](int buf) {
    if (kFull || xs) {
#pragma unroll
      for (int c = 0; c < kPer / 8; ++c) {
        const float4 a = pre[2 * c], b = pre[2 * c + 1];
        *reinterpret_cast<half8*>((buf ? as1 : as0) + 8 * c) = half8{
            (_Float16)a.x, (_Float16)a.y, (_Float16)a.z, (_Float16)a.w,
            (_Float16)b.x, (_Float16)b.y, (_Float16)b.z, (_Float16)b.w};
      }
    }
  };

  float8 acc[kTiles];
#pragma unroll
  for (int t = 0; t < kTiles; ++t) acc[t] = float8{};

  const uint32_t blocks = k / 256;
  Q5Lane next = load_q5(wr, h);
  fetch(0);
#pragma unroll
  for (int c = 0; c < kPer / 4; ++c) pre[c] = pre2[c];
  if (kVStageK < k) fetch(kVStageK);
  int g = 0;
  for (uint32_t sb = 0; sb < blocks; ++sb) {
    const Q5Lane q = next;
    if (sb + 1 < blocks) next = load_q5(wr + (uint64_t)(sb + 1) * kQ5Bytes, h);
#pragma unroll
    for (int stage = 0; stage < 4; ++stage) {
      const uint32_t k0 = sb * 256 + stage * kVStageK;
      if constexpr (kDBuf) {
        /* Store into this stage's buffer while the previous stage still
         * reads the other one; one barrier makes the stores visible. */
        store(g & 1);
        lds_barrier();
      } else {
        lds_barrier();
        store(0);
        lds_barrier();
      }
#pragma unroll
      for (int c = 0; c < kPer / 4; ++c) pre[c] = pre2[c];
      if (k0 + 2 * kVStageK < k) fetch(k0 + 2 * kVStageK);
#pragma unroll
      for (int jj = 0; jj < 2; ++jj) {
        const int j = stage * 2 + jj;
        const uint32_t affine = q5_affine(q.head, j);
#pragma unroll
        for (int e = 0; e < 2; ++e) {
          uint32_t words[8];
          q5_decode8(q, j, e, affine, words);
#pragma unroll
          for (int i = 0; i < 4; ++i)
            words[4 + i] = __builtin_amdgcn_permlanex16(words[i], words[i], 0x76543210u, 0xfedcba98u, false, false);
          const half16 b = __builtin_bit_cast(half16, words);
          const int kk = jj * 32 + e * 16;
          const auto frag = [&](int t) {
            const uint4 own =
                *reinterpret_cast<const uint4*>(act[g & 1] + (t * 16 + lane % 16) * kPitch + kk + 8 * h);
            uint32_t v[8] = {own.x, own.y, own.z, own.w};
#pragma unroll
            for (int i = 0; i < 4; ++i)
              v[4 + i] = __builtin_amdgcn_permlanex16(v[i], v[i], 0x76543210u, 0xfedcba98u, false, false);
            return __builtin_bit_cast(half16, v);
          };
#pragma unroll
          for (int t = 0; t < kTiles; ++t)
            if (t < live) acc[t] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(frag(t), b, acc[t]);
        }
      }
      ++g;
    }
  }

  const uint64_t out_stride = (uint64_t)groups * m;
#pragma unroll
  for (int t = 0; t < kTiles; ++t) {
    if (t >= live) continue;
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      const uint32_t token = t0 + t * 16 + 2 * e + h;
      if (kFull || token < n) out[token * out_stride + (uint64_t)z * m + row] = finite_or_zero(acc[t][e]);
    }
  }
}

template <int kTileT, bool kDBuf>
int launch(const void* w, const float* x, float* out, uint32_t m, uint32_t n, uint32_t k, uint32_t groups) {
  if (m % 256 != 0 || k % 256 != 0 || n == 0 || groups == 0) return 1;
  const auto* wb = static_cast<const uint8_t*>(w);
  const uint32_t full = n / kTileT;
  if (full) dense_kernel<true, kTileT, kDBuf><<<dim3(m / 256, full, groups), kTileT * 4>>>(wb, x, out, m, n, k, groups, 0);
  if (n % kTileT)
    dense_kernel<false, kTileT, kDBuf><<<dim3(m / 256, 1, groups), kTileT * 4>>>(wb, x, out, m, n, k, groups, full);
  return hipGetLastError() == hipSuccess ? 0 : 1;
}

using Launcher = int (*)(const void*, const float*, float*, uint32_t, uint32_t, uint32_t, uint32_t);

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
}  // namespace tune

int main() {
  setbuf(stdout, nullptr);
  struct Shape {
    const char* name;
    uint32_t m, k, groups;
  };
  const Shape shapes[] = {
      {"q_b", 32768, 1280, 1}, {"o_b", 5120, 8192, 1}, {"sh_gate", 2304, 5120, 1}, {"wo_a", 8192, 4096, 8},
  };
  for (uint32_t n : {2048u, 1904u}) {
  const struct {
    const char* name;
    tune::Launcher launcher;
  } variants[] = {
      {"prod", [](const void* w, const float* x, float* o, uint32_t m, uint32_t n, uint32_t k, uint32_t g) { return ds41_v41_dense_f16(13, w, x, o, m, n, k, g); }},
      {"t64", tune::launch<64, false>},
      {"t64+db", tune::launch<64, true>},
      {"t128+db", tune::launch<128, true>},
  };
  std::mt19937 rng(41);
  for (const auto& s : shapes) {
    const std::size_t bytes = static_cast<std::size_t>(s.groups) * s.m * s.k / 256 * 176;
    const int copies = static_cast<int>(std::max<std::size_t>(2, (512u << 20) / bytes));
    const auto host = tune::RandomQ5K(static_cast<std::size_t>(s.groups) * s.m * s.k / 256, rng);
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
    float *x = nullptr, *y = nullptr;
    (void)hipMalloc(&x, xh.size() * 4);
    (void)hipMalloc(&y, static_cast<std::size_t>(s.groups) * s.m * n * 4);
    (void)hipMemcpy(x, xh.data(), xh.size() * 4, hipMemcpyHostToDevice);
    hipEvent_t a, b;
    (void)hipEventCreate(&a);
    (void)hipEventCreate(&b);
    std::printf("%-8s %5ux%-5d g%-2d", s.name, s.m, s.k, s.groups);
    std::vector<float> ref;
    for (const auto& v : variants) {
      auto run = [&](int i) { (void)v.launcher(w[i % copies], x, y, s.m, n, s.k, s.groups); };
      for (int i = 0; i < copies; ++i)
        run(i);
      const int iters = std::max(copies, 6);
      (void)hipEventRecord(a, 0);
      for (int i = 0; i < iters; ++i)
        run(i);
      (void)hipEventRecord(b, 0);
      (void)hipEventSynchronize(b);
      float ms = 0;
      (void)hipEventElapsedTime(&ms, a, b);
      const double t = ms / iters;
      (void)v.launcher(w[0], x, y, s.m, n, s.k, s.groups);
      std::vector<float> got(static_cast<std::size_t>(s.groups) * s.m * n);
      (void)hipMemcpy(got.data(), y, got.size() * 4, hipMemcpyDeviceToHost);
      uint64_t hash = 1469598103934665603ull;
      for (float val : got) {
        uint32_t bits;
        std::memcpy(&bits, &val, 4);
        hash = (hash ^ bits) * 1099511628211ull;
      }
      const bool same = ref.empty() || memcmp(ref.data(), got.data(), got.size() * 4) == 0;
      if (ref.empty()) ref = got;
      const double flops = 2.0 * s.m * s.k * n * s.groups;
      std::printf("  %s %6.2f ms %5.1f TF%s(hash %016llx)", v.name, t, flops / t / 1e9, same ? " " : " MISMATCH",
                  static_cast<unsigned long long>(hash));
    }
    std::printf("\n");
    for (auto* p : w)
      (void)hipFree(p);
    (void)hipFree(x);
    (void)hipFree(y);
    (void)hipEventDestroy(a);
    (void)hipEventDestroy(b);
  }
  }
  return 0;
}
