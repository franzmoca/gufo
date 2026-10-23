// Standalone prefill GEMM with trellis-coded weights (QTIP bitshift trellis,
// L = 16, EXL3/HT "mul1" codebook) decoded on the fly into binary16 for
// v_wmma_f32_16x16x16_f16, against the same kernel reading binary16 weights.
// Experiment for the weight-format study; not a production kernel.
//
//   nix develop -c tools/bench/build.sh gemma4_trellis_gemm_bench
//   /tmp/gemma4_trellis_gemm_bench [tokens] [iterations]
//
// Weight tile: 16x16 values in one 256*K-bit stream (K bits per value), value
// i = 16 * row + col; its 16-bit state is stream bits [K*(i+1) - 16, K*(i+1))
// (little-endian bit order; the tail-biting wrap of value 0 is ignored, which
// does not change the cost). Codebook: x = state * 0x83DCD12D, a = byte sum
// of x, w = (1024 + a) * kScale + kBias in binary16. Tiles are stored
// [K/16][M/16][tile]: one 16-column strip of all rows after another.
//
// Each CTA computes 128 rows x 128 tokens over K blocks of 64: the 256
// threads first decode the block's 128 x 64 weights into LDS (32 values per
// thread), so every weight is decoded once per 128 tokens; then 8 waves
// (4 x 2, each 32 rows x 64 tokens) run the WMMAs from LDS. Mode f16 loads
// the same block from a binary16 copy instead of decoding it.
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

typedef _Float16 h16 __attribute__((ext_vector_type(16)));
typedef _Float16 h8 __attribute__((ext_vector_type(8)));
typedef float f8 __attribute__((ext_vector_type(8)));

#define CHECK(x)                                                         \
  do {                                                                   \
    hipError_t e_ = (x);                                                 \
    if (e_ != hipSuccess) {                                              \
      std::printf("%s:%d %s\n", __FILE__, __LINE__, hipGetErrorString(e_)); \
      std::exit(1);                                                      \
    }                                                                    \
  } while (0)

constexpr float kScale = 1.0F / 147.7F;
constexpr float kBias = -(1024.0F + 510.0F) / 147.7F;
constexpr int TM = 128, TN = 128, KB = 64, kThreads = 256;
constexpr int kWBytes = TM * KB * 2;  // decoded block, binary16
constexpr int kXBytes = TN * KB * 2;

__host__ __device__ inline float Codebook(std::uint32_t state) {
  const std::uint32_t x = state * 0x83DCD12DU;
  const std::uint32_t a = (x & 0xFF) + ((x >> 8) & 0xFF) + ((x >> 16) & 0xFF) + (x >> 24);
  return (1024.0F + static_cast<float>(a)) * kScale + kBias;
}

/// Fragment-major LDS offset (bytes) of row/token r, K16 step s within a
/// KB block: each 16x16 operand fragment is one 512-byte run, a lane's 16
/// values contiguous.
__device__ inline int FragOff(int r, int s) {
  return (((r >> 4) * (KB / 16) + s) * 16 + (r & 15)) * 32;
}

/// Decodes the 16 values of row `row` of one tile into binary16.
template <int K>
__device__ inline h16 DecodeRow(const std::uint32_t* tile, int row) {
  // Bits [K*16*row - 16 + K, K*16*(row + 1)) cover the row's 16 states.
  const int first = K * 16 * row - 16 + K;
  const int w0 = first < 0 ? 0 : (first >> 5);
  std::uint32_t w[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    w[i] = (w0 + i) < 8 * K ? tile[w0 + i] : 0U;
  }
  const int base = first < 0 ? first : first - 32 * w0;  // bit offset in w
  h16 out;
#pragma unroll
  for (int j = 0; j < 16; ++j) {
    const int lo = base + K * j;  // state bits [lo, lo + 16)
    std::uint32_t state;
    if (lo < 0) {
      state = (w[0] << (-lo)) & 0xFFFFU;
    } else {
      const int wi = lo >> 5, bi = lo & 31;
      const std::uint64_t pair = (static_cast<std::uint64_t>(w[wi + 1]) << 32) | w[wi];
      state = static_cast<std::uint32_t>(pair >> bi) & 0xFFFFU;
    }
    const std::uint32_t x = state * 0x83DCD12DU;
    const std::uint32_t a = __builtin_amdgcn_sad_u8(x, 0U, 0U);
    const _Float16 h = __builtin_bit_cast(_Float16, static_cast<std::uint16_t>(0x6400U + a));
    out[j] = h * static_cast<_Float16>(kScale) + static_cast<_Float16>(kBias);
  }
  return out;
}

/// K = 0: binary16 weights [M][Kdim]; K = 3 or 4: trellis tiles.
template <int K>
__global__ __launch_bounds__(kThreads) void TrellisGemm(
    const std::uint32_t* __restrict__ tiles, const _Float16* __restrict__ wf16,
    const _Float16* __restrict__ x, float* __restrict__ y, int m, int kdim, int n) {
  __shared__ __attribute__((aligned(16))) std::uint8_t lds[kWBytes + kXBytes];
  const int tid = threadIdx.x;
  const int lane = tid & 31, wave = tid >> 5;
  const int lr = lane & 15, hi = lane >> 4;
  const int wm = wave & 3, wn = wave >> 2;
  const int col0 = blockIdx.x * TN, row0 = blockIdx.y * TM;
  constexpr int kTileWords = 8 * (K == 0 ? 1 : K);
  const int strips = kdim / 16;

  f8 acc[2][4];
#pragma unroll
  for (int a = 0; a < 2; ++a)
#pragma unroll
    for (int c = 0; c < 4; ++c) acc[a][c] = (f8){};

  for (int k0 = 0; k0 < kdim; k0 += KB) {
    // Weights: thread t -> row t / 2, K16 steps 2(t&1), 2(t&1) + 1.
    {
      const int r = tid >> 1;
#pragma unroll
      for (int q = 0; q < 2; ++q) {
        const int s = (tid & 1) * 2 + q;
        h16 v;
        if constexpr (K == 0) {
          const _Float16* src = wf16 + static_cast<std::size_t>(row0 + r) * kdim + k0 + s * 16;
          v = *reinterpret_cast<const h16*>(src);
        } else {
          const int strip = k0 / 16 + s;
          const int tile = strip * (m / 16) + (row0 + r) / 16;
          v = DecodeRow<K>(tiles + static_cast<std::size_t>(tile) * kTileWords, r & 15);
        }
        *reinterpret_cast<h16*>(lds + FragOff(r, s)) = v;
      }
    }
    // Activations: thread t -> token t / 2, K16 steps 2(t&1), 2(t&1) + 1.
    {
      const int t = tid >> 1;
#pragma unroll
      for (int q = 0; q < 2; ++q) {
        const int s = (tid & 1) * 2 + q;
        const _Float16* src = x + static_cast<std::size_t>(col0 + t) * kdim + k0 + s * 16;
        *reinterpret_cast<h16*>(lds + kWBytes + FragOff(t, s)) =
            *reinterpret_cast<const h16*>(src);
      }
    }
    __syncthreads();
#pragma unroll
    for (int s = 0; s < KB / 16; ++s) {
      h16 av[2], xv[4];
#pragma unroll
      for (int a = 0; a < 2; ++a)
        av[a] = *reinterpret_cast<const h16*>(lds + FragOff(wm * 32 + a * 16 + lr, s));
#pragma unroll
      for (int c = 0; c < 4; ++c)
        xv[c] = *reinterpret_cast<const h16*>(lds + kWBytes + FragOff(wn * 64 + c * 16 + lr, s));
#pragma unroll
      for (int a = 0; a < 2; ++a)
#pragma unroll
        for (int c = 0; c < 4; ++c)
          acc[a][c] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(av[a], xv[c], acc[a][c]);
    }
    __syncthreads();
  }
  (void)strips;
#pragma unroll
  for (int c = 0; c < 4; ++c) {
    float* yr = y + static_cast<std::size_t>(col0 + wn * 64 + c * 16 + lr) * m + row0 +
                wm * 32 + hi;
#pragma unroll
    for (int a = 0; a < 2; ++a)
#pragma unroll
      for (int j = 0; j < 8; ++j) yr[a * 16 + 2 * j] = acc[a][c][j];
  }
}

__global__ void FillRandom(std::uint8_t* p, std::size_t n, std::uint32_t seed) {
  for (std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x; i < n;
       i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    std::uint32_t v = static_cast<std::uint32_t>(i) * 2654435761U + seed;
    v ^= v >> 13;
    v *= 0x5bd1e995U;
    p[i] = static_cast<std::uint8_t>(v >> 24);
  }
}

__global__ void FillHalf(_Float16* p, std::size_t n, std::uint32_t seed) {
  for (std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x; i < n;
       i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    std::uint32_t v = static_cast<std::uint32_t>(i) * 2654435761U + seed;
    v ^= v >> 15;
    v *= 0x2c1b3c6dU;
    p[i] = static_cast<_Float16>((static_cast<int>(v >> 20) - 2048) / 2048.0F);
  }
}

template <int K>
void Launch(const std::uint32_t* tiles, const _Float16* wf16, const _Float16* x, float* y,
            int m, int k, int n, hipStream_t s) {
  TrellisGemm<K><<<dim3(n / TN, m / TM), kThreads, 0, s>>>(tiles, wf16, x, y, m, k, n);
}

/// CPU check of the K = 4 decode and GEMM on a small shape.
bool Check() {
  const int m = 256, k = 128, n = 128, K = 4;
  const int tiles = (m / 16) * (k / 16);
  std::vector<std::uint32_t> ht(static_cast<std::size_t>(tiles) * 8 * K);
  std::vector<_Float16> hx(static_cast<std::size_t>(n) * k);
  std::uint32_t s = 12345;
  for (auto& v : ht) v = (s = s * 1664525U + 1013904223U);
  for (auto& v : hx) v = static_cast<_Float16>(((s = s * 1664525U + 1013904223U) >> 20) / 4096.0F - 0.5F);
  std::uint32_t* dt;
  _Float16* dx;
  float* dy;
  CHECK(hipMalloc(&dt, ht.size() * 4));
  CHECK(hipMalloc(&dx, hx.size() * 2));
  CHECK(hipMalloc(&dy, static_cast<std::size_t>(n) * m * 4));
  CHECK(hipMemcpy(dt, ht.data(), ht.size() * 4, hipMemcpyHostToDevice));
  CHECK(hipMemcpy(dx, hx.data(), hx.size() * 2, hipMemcpyHostToDevice));
  Launch<4>(dt, nullptr, dx, dy, m, k, n, nullptr);
  CHECK(hipDeviceSynchronize());
  std::vector<float> hy(static_cast<std::size_t>(n) * m);
  CHECK(hipMemcpy(hy.data(), dy, hy.size() * 4, hipMemcpyDeviceToHost));
  // Reference weights.
  std::vector<float> w(static_cast<std::size_t>(m) * k);
  for (int r = 0; r < m; ++r) {
    for (int c = 0; c < k; ++c) {
      const int tile = (c / 16) * (m / 16) + r / 16;
      const std::uint32_t* t = &ht[static_cast<std::size_t>(tile) * 8 * K];
      const int i = (r % 16) * 16 + c % 16;
      const int hi_bit = K * (i + 1);
      std::uint32_t state = 0;
      for (int b = 0; b < 16; ++b) {
        const int bit = hi_bit - 16 + b;
        if (bit >= 0) state |= ((t[bit >> 5] >> (bit & 31)) & 1U) << b;
      }
      // The kernel's arithmetic: binary16 (1024 + a), one binary16 FMA.
      const std::uint32_t xs = state * 0x83DCD12DU;
      const std::uint32_t a = (xs & 0xFF) + ((xs >> 8) & 0xFF) + ((xs >> 16) & 0xFF) + (xs >> 24);
      const double fma = static_cast<double>(1024 + a) *
                             static_cast<double>(static_cast<_Float16>(kScale)) +
                         static_cast<double>(static_cast<_Float16>(kBias));
      w[static_cast<std::size_t>(r) * k + c] = static_cast<float>(static_cast<_Float16>(fma));
    }
  }
  double worst = 0;
  for (int t = 0; t < n; ++t)
    for (int r = 0; r < m; ++r) {
      double ref = 0, mag = 0;
      for (int c = 0; c < k; ++c) {
        const double p = w[static_cast<std::size_t>(r) * k + c] *
                         static_cast<double>(static_cast<float>(hx[static_cast<std::size_t>(t) * k + c]));
        ref += p;
        mag += std::fabs(p);
      }
      worst = std::max(worst, std::fabs(hy[static_cast<std::size_t>(t) * m + r] - ref) / (mag + 1e-6));
    }
  CHECK(hipFree(dt));
  CHECK(hipFree(dx));
  CHECK(hipFree(dy));
  std::printf("check trellis K=4  max err / sum|wx| %.2e %s\n", worst, worst < 1e-4 ? "ok" : "FAIL");
  return worst < 1e-4;
}

struct Shape {
  const char* name;
  int m, k;
};
const Shape kShapes[] = {
    {"gate|up", 43008, 5376}, {"down", 5376, 21504}, {"q|k sliding", 12288, 5376},
    {"o sliding", 5376, 8192}, {"o global", 5376, 16384},
};

}  // namespace

int main(int argc, char** argv) {
  const int n = argc > 1 ? std::atoi(argv[1]) : 2048;
  const int iters = argc > 2 ? std::atoi(argv[2]) : 10;
  if (!Check()) return 1;
  hipStream_t stream;
  CHECK(hipStreamCreate(&stream));
  hipEvent_t e0, e1;
  CHECK(hipEventCreate(&e0));
  CHECK(hipEventCreate(&e1));
  std::printf("\n%d tokens, median of %d (ms, TFLOP/s)\n%-12s %18s %18s %18s\n", n, iters,
              "shape", "binary16 weights", "trellis K=4", "trellis K=3");
  for (const Shape& sh : kShapes) {
    const std::size_t count = static_cast<std::size_t>(sh.m) * sh.k;
    _Float16 *wf16, *x;
    std::uint32_t *t4, *t3;
    float* y;
    CHECK(hipMalloc(&wf16, count * 2));
    CHECK(hipMalloc(&t4, count / 2));
    CHECK(hipMalloc(&t3, count * 3 / 8));
    CHECK(hipMalloc(&x, static_cast<std::size_t>(n) * sh.k * 2));
    CHECK(hipMalloc(&y, static_cast<std::size_t>(n) * sh.m * 4));
    FillHalf<<<1024, 256>>>(wf16, count, 3);
    FillRandom<<<1024, 256>>>(reinterpret_cast<std::uint8_t*>(t4), count / 2, 5);
    FillRandom<<<1024, 256>>>(reinterpret_cast<std::uint8_t*>(t3), count * 3 / 8, 7);
    FillHalf<<<1024, 256>>>(x, static_cast<std::size_t>(n) * sh.k, 9);
    CHECK(hipDeviceSynchronize());
    auto time = [&](auto&& body) {
      body();
      CHECK(hipStreamSynchronize(stream));
      std::vector<float> t;
      for (int i = 0; i < iters; ++i) {
        CHECK(hipEventRecord(e0, stream));
        body();
        CHECK(hipEventRecord(e1, stream));
        CHECK(hipEventSynchronize(e1));
        float ms;
        CHECK(hipEventElapsedTime(&ms, e0, e1));
        t.push_back(ms);
      }
      std::sort(t.begin(), t.end());
      return t[t.size() / 2];
    };
    const double flop = 2.0 * sh.m * sh.k * n;
    const float a = time([&] { Launch<0>(nullptr, wf16, x, y, sh.m, sh.k, n, stream); });
    const float b = time([&] { Launch<4>(t4, nullptr, x, y, sh.m, sh.k, n, stream); });
    const float c = time([&] { Launch<3>(t3, nullptr, x, y, sh.m, sh.k, n, stream); });
    std::printf("%-12s %9.3f %6.1f   %9.3f %6.1f   %9.3f %6.1f\n", sh.name, a, flop / a / 1e9,
                b, flop / b / 1e9, c, flop / c / 1e9);
    CHECK(hipFree(wf16));
    CHECK(hipFree(t4));
    CHECK(hipFree(t3));
    CHECK(hipFree(x));
    CHECK(hipFree(y));
  }
  return 0;
}
