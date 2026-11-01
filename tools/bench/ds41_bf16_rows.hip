// BF16 x FP32 matvec for a few tokens (DeepSeek V4.1 Flash router, indexer
// q_b, compressor kv): the decoder's wave-per-(row, token) kernel against
// variants that keep a row's weights for every token. Outputs are compared
// bit for bit with the reference; weights rotate through copies larger than
// the caches. build.sh compiles it with the library's -ffast-math, under
// which the variants round differently from the reference (EXPERIMENTS E66).
//
//   nix develop -c tools/bench/build.sh ds41_bf16_rows && /tmp/ds41_bf16_rows
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace {

constexpr uint32_t kThreads = 256;

// The decoder's kernel (ds41_v41.hip.cpp, matmul_bf16_vec8_kernel).
__global__ void ref_kernel(const uint16_t* w, const float* x, float* out, uint32_t rows, uint32_t k) {
  const uint32_t row = blockIdx.x * (blockDim.x / warpSize) + threadIdx.x / warpSize;
  if (row >= rows) return;
  const uint32_t lane = threadIdx.x % warpSize;
  const uint32_t t = blockIdx.y;
  const uint4* r = reinterpret_cast<const uint4*>(w + (uint64_t)row * k);
  const float4* xt = reinterpret_cast<const float4*>(x + (uint64_t)t * k);
  float acc = 0.0f;
#pragma unroll 4
  for (uint32_t i = lane; i < k / 8; i += warpSize) {
    const uint4 packed = r[i];
    const uint32_t words[4] = {packed.x, packed.y, packed.z, packed.w};
    float wv[8];
#pragma unroll
    for (uint32_t e = 0; e < 4; e++) {
      wv[2 * e] = __uint_as_float(words[e] << 16);
      wv[2 * e + 1] = __uint_as_float(words[e] & 0xffff0000u);
    }
    const float4 a = xt[2 * i], b = xt[2 * i + 1];
    acc += wv[0] * a.x + wv[1] * a.y + wv[2] * a.z + wv[3] * a.w + wv[4] * b.x + wv[5] * b.y + wv[6] * b.z +
           wv[7] * b.w;
  }
  for (int o = warpSize / 2; o > 0; o >>= 1) acc += __shfl_xor(acc, o);
  if (lane == 0) out[(uint64_t)t * rows + row] = acc;
}

// R rows per wave, every token: each weight vector loads once, each x vector
// once per R rows. Each (row, token) sum keeps the reference's order.
template <int R, int kMaxT>
__global__ void rows_kernel(const uint16_t* w, const float* x, float* out, uint32_t rows, uint32_t k, uint32_t n) {
  const uint32_t row0 = (blockIdx.x * (blockDim.x / warpSize) + threadIdx.x / warpSize) * R;
  if (row0 >= rows) return;
  const uint32_t lane = threadIdx.x % warpSize;
  const uint32_t t0 = blockIdx.y * kMaxT;
  x += (uint64_t)t0 * k;
  out += (uint64_t)t0 * rows;
  n = n - t0 < (uint32_t)kMaxT ? n - t0 : (uint32_t)kMaxT;
  float acc[R][kMaxT] = {};
#pragma unroll 2
  for (uint32_t i = lane; i < k / 8; i += warpSize) {
    float wv[R][8];
#pragma unroll
    for (int q = 0; q < R; q++) {
      const uint4 packed = reinterpret_cast<const uint4*>(w + (uint64_t)(row0 + q) * k)[i];
      const uint32_t words[4] = {packed.x, packed.y, packed.z, packed.w};
#pragma unroll
      for (uint32_t e = 0; e < 4; e++) {
        wv[q][2 * e] = __uint_as_float(words[e] << 16);
        wv[q][2 * e + 1] = __uint_as_float(words[e] & 0xffff0000u);
      }
    }
#pragma unroll
    for (int t = 0; t < kMaxT; t++) {
      if (t >= (int)n) break;
      const float4* xt = reinterpret_cast<const float4*>(x + (uint64_t)t * k);
      const float4 a = xt[2 * i], b = xt[2 * i + 1];
#pragma unroll
      for (int q = 0; q < R; q++)
        acc[q][t] += wv[q][0] * a.x + wv[q][1] * a.y + wv[q][2] * a.z + wv[q][3] * a.w + wv[q][4] * b.x +
                     wv[q][5] * b.y + wv[q][6] * b.z + wv[q][7] * b.w;
    }
  }
#pragma unroll
  for (int q = 0; q < R; q++)
#pragma unroll
    for (int t = 0; t < kMaxT; t++) {
      if (t >= (int)n) break;
      float v = acc[q][t];
      for (int o = warpSize / 2; o > 0; o >>= 1) v += __shfl_xor(v, o);
      if (lane == 0) out[(uint64_t)t * rows + row0 + q] = v;
    }
}

struct Shape {
  const char* name;
  uint32_t rows, k;
};

}  // namespace

int main() {
  const Shape shapes[] = {{"router", 384, 5120}, {"indexer_q_b", 4096, 1280}, {"compressor", 512, 5120}};
  std::mt19937 rng(7);
  std::normal_distribution<float> nd;
  hipEvent_t a, b;
  (void)hipEventCreate(&a);
  (void)hipEventCreate(&b);
  for (const Shape& s : shapes) {
    const size_t bytes = (size_t)s.rows * s.k * 2;
    const int copies = (int)std::max<size_t>(4, (512u << 20) / bytes);
    std::vector<uint16_t> hw((size_t)s.rows * s.k);
    for (auto& v : hw) {
      const float f = nd(rng) * 0.05f;
      uint32_t u;
      std::memcpy(&u, &f, 4);
      v = (uint16_t)(u >> 16);
    }
    std::vector<uint16_t*> w(copies);
    for (auto& p : w) {
      (void)hipMalloc(&p, bytes);
      (void)hipMemcpy(p, hw.data(), bytes, hipMemcpyHostToDevice);
    }
    std::vector<float> hx((size_t)16 * s.k);
    for (auto& v : hx) v = nd(rng);
    float *x, *y0, *y1;
    (void)hipMalloc(&x, hx.size() * 4);
    (void)hipMalloc(&y0, (size_t)16 * s.rows * 4);
    (void)hipMalloc(&y1, (size_t)16 * s.rows * 4);
    (void)hipMemcpy(x, hx.data(), hx.size() * 4, hipMemcpyHostToDevice);
    auto time = [&](auto&& launch) {
      for (int i = 0; i < copies; i++) launch(w[i]);
      const int iters = std::max(copies, 200);
      (void)hipEventRecord(a, 0);
      for (int i = 0; i < iters; i++) launch(w[i % copies]);
      (void)hipEventRecord(b, 0);
      (void)hipEventSynchronize(b);
      float ms = 0;
      (void)hipEventElapsedTime(&ms, a, b);
      return ms * 1000.0 / iters;
    };
    auto same = [&](uint32_t n) {
      std::vector<float> r0((size_t)n * s.rows), r1((size_t)n * s.rows);
      (void)hipMemcpy(r0.data(), y0, r0.size() * 4, hipMemcpyDeviceToHost);
      (void)hipMemcpy(r1.data(), y1, r1.size() * 4, hipMemcpyDeviceToHost);
      return std::memcmp(r0.data(), r1.data(), r0.size() * 4) == 0;
    };
    std::printf("%s %ux%u (%.2f MB)\n", s.name, s.rows, s.k, bytes / 1e6);
    for (uint32_t n : {1u, 2u, 3u, 4u, 5u, 6u, 8u}) {
      const dim3 g0((s.rows + 7) / 8, n);
      const double t0 = time([&](uint16_t* p) { ref_kernel<<<g0, kThreads>>>(p, x, y0, s.rows, s.k); });
      std::printf("  n=%u ref %6.1f us (%3.0f GB/s)", n, t0, bytes / t0 / 1e3);
      auto variant = [&](const char* name, auto kern, int r, uint32_t tg = 8) {
        const uint32_t waves = (s.rows + r - 1) / r;
        const dim3 g((waves + 7) / 8, (n + tg - 1) / tg);
        const double t = time([&](uint16_t* p) { kern<<<g, kThreads>>>(p, x, y1, s.rows, s.k, n); });
        std::printf("  %s %6.1f (%3.0f GB/s, %s)", name, t, bytes / t / 1e3, same(n) ? "same" : "DIFF");
      };
      variant("R1", rows_kernel<1, 8>, 1);
      variant("R2", rows_kernel<2, 8>, 2);
      variant("R4", rows_kernel<4, 8>, 4);
      variant("R1T2", rows_kernel<1, 2>, 1, 2);
      variant("R1T3", rows_kernel<1, 3>, 1, 3);
      variant("R2T2", rows_kernel<2, 2>, 2, 2);
      variant("R2T3", rows_kernel<2, 3>, 2, 3);
      std::printf("\n");
    }
    for (auto* p : w) (void)hipFree(p);
    (void)hipFree(x);
    (void)hipFree(y0);
    (void)hipFree(y1);
  }
  return 0;
}
