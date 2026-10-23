// Standalone W4A4 prefill GEMM on v_wmma_i32_16x16x16_iu4 for Gemma 4 31B
// projection shapes, plus the activation producer (128-point FWHT and
// per-128 int4 quantization) it needs. Experiment for the weight-format study
// (backlog L7); not a production kernel.
//
//   nix develop -c tools/bench/build.sh gemma4_w4a4_gemm_bench
//   /tmp/gemma4_w4a4_gemm_bench [tokens] [iterations]
//
// Layouts (K-block-major so each block's tile is one contiguous run):
//   Wq [K/128][M][64 B] signed int4, even k in the low nibble; Ws [K/128][M] f16
//   Xq [K/128][N][64 B] the same for activations;              Xs [K/128][N] f32
//   Y  [N][M] f32 = sum over blocks b of Ws[b][m] * Xs[b][n] * int(Wq . Xq)
//
// The weights rotate over enough copies to exceed the 32 MiB MALL, as in
// gemma4_prefill_bench, so repeats read them from memory.
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace {

typedef int v2i __attribute__((ext_vector_type(2)));
typedef int v4i __attribute__((ext_vector_type(4)));
typedef int v8i __attribute__((ext_vector_type(8)));

#define CHECK(x)                                                         \
  do {                                                                   \
    hipError_t e_ = (x);                                                 \
    if (e_ != hipSuccess) {                                              \
      std::printf("%s:%d %s\n", __FILE__, __LINE__, hipGetErrorString(e_)); \
      std::exit(1);                                                      \
    }                                                                    \
  } while (0)

/// Lane k of the lane's own row of 16 (DPP row_share, gfx10+).
__device__ __forceinline__ float RowShare(float v, int k) {
  const int x = __float_as_int(v);
  int r;
  switch (k) {
#define SHARE(i) \
  case i:        \
    r = __builtin_amdgcn_update_dpp(0, x, 0x150 + i, 0xf, 0xf, false); break;
    SHARE(0) SHARE(2) SHARE(4) SHARE(6) SHARE(8) SHARE(10) SHARE(12) SHARE(14)
#undef SHARE
    default:
      __builtin_unreachable();
  }
  return __int_as_float(r);
}

/// One CTA computes TM rows x TN tokens; waves WM x WN, each NA x NC
/// 16x16 fragments. K walks in 128-value blocks, double-buffered in LDS in
/// fragment-major order (each 16x16 operand fragment one 128-byte run).
template <int TM, int TN, int WM, int WN>
__global__ __launch_bounds__(32 * WM* WN) void W4A4Gemm(
    const std::uint8_t* __restrict__ wq, const __half* __restrict__ ws,
    const std::uint8_t* __restrict__ xq, const float* __restrict__ xs,
    float* __restrict__ y, int m, int k, int n) {
  constexpr int kThreads = 32 * WM * WN;
  constexpr int NA = TM / WM / 16;
  constexpr int NC = TN / WN / 16;
  constexpr int kABytes = TM * 64;
  constexpr int kSlot = (TM + TN) * 64;
  constexpr int kAVec = kABytes / 32 / kThreads;  // 32-byte loads per thread
  constexpr int kXVec = TN * 64 / 32 / kThreads;
  static_assert(kAVec >= 1 && kXVec >= 1, "staging split");
  __shared__ __attribute__((aligned(16))) std::uint8_t lds[2 * kSlot];

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int wave = tid >> 5;
  const int lr = lane & 15;
  const int hi = lane >> 4;
  const int wm = wave % WM;
  const int wn = wave / WM;
  // Token tiles fastest: the CTAs in flight share a few weight row tiles
  // and the whole activation matrix stays in the MALL.
  const int col0 = blockIdx.x * TN;
  const int row0 = blockIdx.y * TM;
  const int blocks = k / 128;

  // Staging: thread t loads 32 bytes (K16 slices 4h..4h+3 of one row).
  auto stage_load = [&](int b, v4i (&ra)[kAVec][2], v4i (&rx)[kXVec][2]) {
    const std::uint8_t* ga = wq + (static_cast<std::size_t>(b) * m + row0) * 64;
    const std::uint8_t* gx = xq + (static_cast<std::size_t>(b) * n + col0) * 64;
#pragma unroll
    for (int i = 0; i < kAVec; ++i) {
      const int off = (tid + i * kThreads) * 32;
      ra[i][0] = *reinterpret_cast<const v4i*>(ga + off);
      ra[i][1] = *reinterpret_cast<const v4i*>(ga + off + 16);
    }
#pragma unroll
    for (int i = 0; i < kXVec; ++i) {
      const int off = (tid + i * kThreads) * 32;
      rx[i][0] = *reinterpret_cast<const v4i*>(gx + off);
      rx[i][1] = *reinterpret_cast<const v4i*>(gx + off + 16);
    }
  };
  auto frag_off = [](int r, int s) { return ((r >> 4) * 8 + s) * 128 + (r & 15) * 8; };
  auto stage_store = [&](std::uint8_t* slot, const v4i (&ra)[kAVec][2],
                         const v4i (&rx)[kXVec][2]) {
#pragma unroll
    for (int i = 0; i < kAVec; ++i) {
      const int idx = tid + i * kThreads;
      const int r = idx >> 1;
      const int s0 = (idx & 1) * 4;
      const v4i a = ra[i][0], b = ra[i][1];
      *reinterpret_cast<v2i*>(slot + frag_off(r, s0)) = (v2i){a.x, a.y};
      *reinterpret_cast<v2i*>(slot + frag_off(r, s0 + 1)) = (v2i){a.z, a.w};
      *reinterpret_cast<v2i*>(slot + frag_off(r, s0 + 2)) = (v2i){b.x, b.y};
      *reinterpret_cast<v2i*>(slot + frag_off(r, s0 + 3)) = (v2i){b.z, b.w};
    }
#pragma unroll
    for (int i = 0; i < kXVec; ++i) {
      const int idx = tid + i * kThreads;
      const int r = idx >> 1;
      const int s0 = (idx & 1) * 4;
      const v4i a = rx[i][0], b = rx[i][1];
      std::uint8_t* sx = slot + kABytes;
      *reinterpret_cast<v2i*>(sx + frag_off(r, s0)) = (v2i){a.x, a.y};
      *reinterpret_cast<v2i*>(sx + frag_off(r, s0 + 1)) = (v2i){a.z, a.w};
      *reinterpret_cast<v2i*>(sx + frag_off(r, s0 + 2)) = (v2i){b.x, b.y};
      *reinterpret_cast<v2i*>(sx + frag_off(r, s0 + 3)) = (v2i){b.z, b.w};
    }
  };
  // Scales: lane (hi, lr) holds row lr + hi of each A fragment, so
  // RowShare(v, 2j) hands both halves rows 2j + hi; and token lr of each X
  // fragment.
  const int arow = row0 + wm * NA * 16 + lr + hi;
  const int xcol = col0 + wn * NC * 16 + lr;
  auto scale_load = [&](int b, float (&sa)[NA], float (&sx)[NC]) {
#pragma unroll
    for (int a = 0; a < NA; ++a) {
      const int r = min(arow + a * 16, m - 1);
      sa[a] = __half2float(ws[static_cast<std::size_t>(b) * m + r]);
    }
#pragma unroll
    for (int c = 0; c < NC; ++c) {
      sx[c] = xs[static_cast<std::size_t>(b) * n + xcol + c * 16];
    }
  };

  float sum[NA][NC][8];
#pragma unroll
  for (int a = 0; a < NA; ++a)
#pragma unroll
    for (int c = 0; c < NC; ++c)
#pragma unroll
      for (int j = 0; j < 8; ++j) sum[a][c][j] = 0.0F;

  v4i ra[kAVec][2];
  v4i rx[kXVec][2];
  float sa[NA], sx[NC], sa_n[NA], sx_n[NC];
  stage_load(0, ra, rx);
  scale_load(0, sa, sx);
  stage_store(lds, ra, rx);
  __syncthreads();

  for (int b = 0; b < blocks; ++b) {
    const std::uint8_t* slot = lds + (b & 1) * kSlot;
    const bool more = b + 1 < blocks;
    if (more) {
      stage_load(b + 1, ra, rx);
      scale_load(b + 1, sa_n, sx_n);
    }
    const std::uint8_t* la = slot + (wm * NA * 16 / 16) * 8 * 128 + lr * 8;
    const std::uint8_t* lx = slot + kABytes + (wn * NC * 16 / 16) * 8 * 128 + lr * 8;
    // Flattened (a, s) steps, operands loaded one step ahead. The fences pin
    // the order: without them the WMMAs sink below every step's loads and
    // all operands stay live at once.
    auto load_a = [&](int i) {
      int o = static_cast<int>(la - lds) + i * 128;
      asm volatile("" : "+v"(o));
      return *reinterpret_cast<const v2i*>(lds + o);
    };
    auto load_x = [&](int s, v2i (&xv)[NC]) {
      int o = static_cast<int>(lx - lds) + s * 128;
      asm volatile("" : "+v"(o));
#pragma unroll
      for (int c = 0; c < NC; ++c) {
        xv[c] = *reinterpret_cast<const v2i*>(lds + o + c * 8 * 128);
      }
    };
    v2i av = load_a(0);
    v2i xv[NC];
    load_x(0, xv);
#pragma unroll
    for (int a = 0; a < NA; ++a) {
      v8i acc[NC];
#pragma unroll
      for (int s = 0; s < 8; ++s) {
        const int i = a * 8 + s + 1;
        v2i av_n;
        v2i xv_n[NC];
        if (i < NA * 8) {
          av_n = load_a(i);
          load_x(i & 7, xv_n);
        }
#pragma unroll
        for (int c = 0; c < NC; ++c) {
          acc[c] = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(
              true, av, true, xv[c], s == 0 ? (v8i){} : acc[c], false);
          asm volatile("" : "+v"(acc[c]));
        }
        if (i < NA * 8) {
          av = av_n;
#pragma unroll
          for (int c = 0; c < NC; ++c) xv[c] = xv_n[c];
        }
      }
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const float wsc = RowShare(sa[a], 2 * j);
#pragma unroll
        for (int c = 0; c < NC; ++c) {
          sum[a][c][j] = __builtin_fmaf(wsc * sx[c],
                                        static_cast<float>(acc[c][j]),
                                        sum[a][c][j]);
        }
      }
      // This fragment's sums exist here, so its int32 chains end here.
#pragma unroll
      for (int c = 0; c < NC; ++c) {
        asm volatile("" : "+v"(sum[a][c][0]), "+v"(sum[a][c][1]),
                     "+v"(sum[a][c][2]), "+v"(sum[a][c][3]),
                     "+v"(sum[a][c][4]), "+v"(sum[a][c][5]),
                     "+v"(sum[a][c][6]), "+v"(sum[a][c][7]));
      }
      __builtin_amdgcn_sched_barrier(0);
    }
    if (more) {
      stage_store(lds + ((b + 1) & 1) * kSlot, ra, rx);
#pragma unroll
      for (int a = 0; a < NA; ++a) sa[a] = sa_n[a];
#pragma unroll
      for (int c = 0; c < NC; ++c) sx[c] = sx_n[c];
    }
    __syncthreads();
  }
  // Lane (hi, lr): token xcol + 16c, rows 2j + hi of each fragment.
#pragma unroll
  for (int c = 0; c < NC; ++c) {
    float* yr = y + static_cast<std::size_t>(xcol + c * 16) * m + row0 +
                wm * NA * 16 + hi;
#pragma unroll
    for (int a = 0; a < NA; ++a)
#pragma unroll
      for (int j = 0; j < 8; ++j) yr[a * 16 + 2 * j] = sum[a][c][j];
  }
}

/// Activation producer: one wave per (token, 128-block) rotates the block
/// (signs, then an orthonormal 128-point Walsh-Hadamard transform) and
/// quantizes it to int4 with the better of d = amax/7 and d = amax/5.
__global__ __launch_bounds__(256) void RotateQuantize(
    const float* __restrict__ x, const float* __restrict__ sign,
    std::uint8_t* __restrict__ xq, float* __restrict__ xs, int n, int k) {
  const int lane = threadIdx.x & 31;
  const int group = blockIdx.x * 8 + (threadIdx.x >> 5);
  const int blocks = k / 128;
  if (group >= n * blocks) return;
  const int t = group / blocks;
  const int b = group % blocks;
  const float4 in = *reinterpret_cast<const float4*>(
      x + static_cast<std::size_t>(t) * k + b * 128 + lane * 4);
  const float4 sg =
      *reinterpret_cast<const float4*>(sign + b * 128 + lane * 4);
  float v[4] = {in.x * sg.x, in.y * sg.y, in.z * sg.z, in.w * sg.w};
  // Stages 1 and 2 inside the lane, 4..64 across lanes.
  float a0 = v[0] + v[1], a1 = v[0] - v[1], a2 = v[2] + v[3], a3 = v[2] - v[3];
  v[0] = a0 + a2;
  v[1] = a1 + a3;
  v[2] = a0 - a2;
  v[3] = a1 - a3;
#pragma unroll
  for (int mask = 1; mask < 32; mask <<= 1) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const float p = __shfl_xor(v[i], mask, 32);
      v[i] = (lane & mask) ? p - v[i] : v[i] + p;
    }
  }
  float amax = 0.0F;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    v[i] *= 0.08838834764831845F;  // 1/sqrt(128)
    amax = fmaxf(amax, fabsf(v[i]));
  }
#pragma unroll
  for (int mask = 16; mask >= 1; mask >>= 1) {
    amax = fmaxf(amax, __shfl_xor(amax, mask, 32));
  }
  float err[2] = {0.0F, 0.0F};
  const float d[2] = {amax / 7.0F, amax / 5.0F};
#pragma unroll
  for (int c = 0; c < 2; ++c) {
    const float id = d[c] > 0.0F ? 1.0F / d[c] : 0.0F;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const float q = fminf(fmaxf(rintf(v[i] * id), -8.0F), 7.0F);
      const float e = v[i] - q * d[c];
      err[c] += e * e;
    }
#pragma unroll
    for (int mask = 16; mask >= 1; mask >>= 1) {
      err[c] += __shfl_xor(err[c], mask, 32);
    }
  }
  const float dc = err[1] < err[0] ? d[1] : d[0];
  const float id = dc > 0.0F ? 1.0F / dc : 0.0F;
  std::uint32_t packed = 0;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int q = static_cast<int>(fminf(fmaxf(rintf(v[i] * id), -8.0F), 7.0F));
    packed |= (static_cast<std::uint32_t>(q) & 0xFU) << (4 * i);
  }
  const std::size_t g = static_cast<std::size_t>(b) * n + t;
  *reinterpret_cast<std::uint16_t*>(xq + g * 64 + lane * 2) =
      static_cast<std::uint16_t>(packed);
  if (lane == 0) xs[g] = dc;
}

__global__ void FillRandom(std::uint8_t* p, std::size_t n, std::uint32_t seed) {
  for (std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
       i < n; i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    std::uint32_t v = static_cast<std::uint32_t>(i) * 2654435761U + seed;
    v ^= v >> 13;
    v *= 0x5bd1e995U;
    p[i] = static_cast<std::uint8_t>(v >> 24);
  }
}

/// Random int4 codes, so the WMMAs see realistic toggling.
void Randomize(void* p, std::size_t n, std::uint32_t seed) {
  FillRandom<<<1024, 256>>>(static_cast<std::uint8_t*>(p), n, seed);
}

int Nibble(const std::uint8_t* row, int k) {
  const int v = (row[k / 2] >> ((k & 1) * 4)) & 0xF;
  return v >= 8 ? v - 16 : v;
}

using Launch = void (*)(const std::uint8_t*, const __half*, const std::uint8_t*,
                        const float*, float*, int, int, int, hipStream_t);

template <int TM, int TN, int WM, int WN>
void Run(const std::uint8_t* wq, const __half* ws, const std::uint8_t* xq,
         const float* xs, float* y, int m, int k, int n, hipStream_t s) {
  W4A4Gemm<TM, TN, WM, WN><<<dim3(n / TN, m / TM), 32 * WM * WN, 0, s>>>(
      wq, ws, xq, xs, y, m, k, n);
}

struct Config {
  const char* name;
  int tm, tn;
  Launch launch;
};

const Config kConfigs[] = {
    {"128x128 w4x2", 128, 128, Run<128, 128, 4, 2>},
    {"256x128 w4x2", 256, 128, Run<256, 128, 4, 2>},
    {"128x256 w2x4", 128, 256, Run<128, 256, 2, 4>},
    {"256x256 w4x4", 256, 256, Run<256, 256, 4, 4>},
};

bool Check(const Config& cfg) {
  const int m = 512, k = 640, n = 256;
  const int blocks = k / 128;
  std::mt19937 rng(7);
  std::vector<std::uint8_t> hwq(static_cast<std::size_t>(blocks) * m * 64), hxq(static_cast<std::size_t>(blocks) * n * 64);
  std::vector<__half> hws(static_cast<std::size_t>(blocks) * m);
  std::vector<float> hxs(static_cast<std::size_t>(blocks) * n);
  for (auto& v : hwq) v = rng() & 0xFF;
  for (auto& v : hxq) v = rng() & 0xFF;
  for (auto& v : hws) v = __float2half(0.5F + (rng() % 1000) / 1000.0F);
  for (auto& v : hxs) v = 0.25F + (rng() % 1000) / 1000.0F;
  std::uint8_t *wq, *xq;
  __half* ws;
  float *xs, *y;
  CHECK(hipMalloc(&wq, hwq.size()));
  CHECK(hipMalloc(&xq, hxq.size()));
  CHECK(hipMalloc(&ws, hws.size() * 2));
  CHECK(hipMalloc(&xs, hxs.size() * 4));
  CHECK(hipMalloc(&y, static_cast<std::size_t>(n) * m * 4));
  CHECK(hipMemcpy(wq, hwq.data(), hwq.size(), hipMemcpyHostToDevice));
  CHECK(hipMemcpy(xq, hxq.data(), hxq.size(), hipMemcpyHostToDevice));
  CHECK(hipMemcpy(ws, hws.data(), hws.size() * 2, hipMemcpyHostToDevice));
  CHECK(hipMemcpy(xs, hxs.data(), hxs.size() * 4, hipMemcpyHostToDevice));
  cfg.launch(wq, ws, xq, xs, y, m, k, n, nullptr);
  CHECK(hipDeviceSynchronize());
  std::vector<float> hy(static_cast<std::size_t>(n) * m);
  CHECK(hipMemcpy(hy.data(), y, hy.size() * 4, hipMemcpyDeviceToHost));
  double worst = 0.0;
  for (int t = 0; t < n; ++t) {
    for (int r = 0; r < m; ++r) {
      double ref = 0.0;
      for (int b = 0; b < blocks; ++b) {
        const std::uint8_t* wr = &hwq[(static_cast<std::size_t>(b) * m + r) * 64];
        const std::uint8_t* xr = &hxq[(static_cast<std::size_t>(b) * n + t) * 64];
        long dot = 0;
        for (int i = 0; i < 128; ++i) dot += Nibble(wr, i) * Nibble(xr, i);
        ref += static_cast<double>(__half2float(hws[static_cast<std::size_t>(b) * m + r])) *
               hxs[static_cast<std::size_t>(b) * n + t] * dot;
      }
      const double got = hy[static_cast<std::size_t>(t) * m + r];
      worst = std::max(worst, std::fabs(got - ref) / (std::fabs(ref) + 1.0));
    }
  }
  CHECK(hipFree(wq));
  CHECK(hipFree(xq));
  CHECK(hipFree(ws));
  CHECK(hipFree(xs));
  CHECK(hipFree(y));
  std::printf("check %-14s max rel err %.2e %s\n", cfg.name, worst,
              worst < 1e-4 ? "ok" : "FAIL");
  return worst < 1e-4;
}

struct Shape {
  const char* name;
  int m, k;
  int per_layer;  // layers with this shape (of 60)
};

// 31B: 50 sliding layers (q 8192, k|v 4096 each, o from 8192) and 10 global
// layers (q 16384, k 2048, no v, o from 16384); every layer gate|up 2 x 21504
// and down from 21504.
const Shape kShapes[] = {
    {"gate|up", 43008, 5376, 60}, {"down", 5376, 21504, 60},
    {"q sliding", 8192, 5376, 50}, {"k|v sliding", 8192, 5376, 50},
    {"o sliding", 5376, 8192, 50}, {"q global", 16384, 5376, 10},
    {"k global", 2048, 5376, 10},  {"o global", 5376, 16384, 10},
};

}  // namespace

int main(int argc, char** argv) {
  const int n = argc > 1 ? std::atoi(argv[1]) : 2048;
  const int iters = argc > 2 ? std::atoi(argv[2]) : 10;
  bool ok = true;
  for (const Config& c : kConfigs) ok &= Check(c);
  if (!ok) return 1;

  hipStream_t stream;
  CHECK(hipStreamCreate(&stream));
  hipEvent_t e0, e1;
  CHECK(hipEventCreate(&e0));
  CHECK(hipEventCreate(&e1));
  auto time_ms = [&](auto&& body) {
    body(0);
    CHECK(hipStreamSynchronize(stream));
    std::vector<float> t;
    for (int i = 0; i < iters; ++i) {
      CHECK(hipEventRecord(e0, stream));
      body(i);
      CHECK(hipEventRecord(e1, stream));
      CHECK(hipEventSynchronize(e1));
      float ms;
      CHECK(hipEventElapsedTime(&ms, e0, e1));
      t.push_back(ms);
    }
    std::sort(t.begin(), t.end());
    return t[t.size() / 2];
  };

  std::printf("\nGEMM, %d tokens, median of %d (ms, TOPS)\n", n, iters);
  std::printf("%-12s", "shape");
  for (const Config& c : kConfigs) std::printf(" %20s", c.name);
  std::printf("\n");
  std::vector<double> chunk(std::size(kConfigs), 0.0);
  for (const Shape& s : kShapes) {
    const int blocks = s.k / 128;
    const std::size_t wbytes = static_cast<std::size_t>(blocks) * s.m * 64;
    const int copies = static_cast<int>(std::max<std::size_t>(1, (96u << 20) / wbytes + 1));
    std::vector<std::uint8_t*> wq(copies);
    std::vector<__half*> ws(copies);
    for (int i = 0; i < copies; ++i) {
      CHECK(hipMalloc(&wq[i], wbytes));
      CHECK(hipMalloc(&ws[i], static_cast<std::size_t>(blocks) * s.m * 2));
      Randomize(wq[i], wbytes, 17 + i);
      CHECK(hipMemset(ws[i], 0x11, static_cast<std::size_t>(blocks) * s.m * 2));
    }
    std::uint8_t* xq;
    float *xs, *y;
    CHECK(hipMalloc(&xq, static_cast<std::size_t>(blocks) * n * 64));
    CHECK(hipMalloc(&xs, static_cast<std::size_t>(blocks) * n * 4));
    CHECK(hipMalloc(&y, static_cast<std::size_t>(n) * s.m * 4));
    Randomize(xq, static_cast<std::size_t>(blocks) * n * 64, 99);
    CHECK(hipMemset(xs, 0x3C, static_cast<std::size_t>(blocks) * n * 4));
    std::printf("%-12s", s.name);
    for (std::size_t ci = 0; ci < std::size(kConfigs); ++ci) {
      const Config& c = kConfigs[ci];
      if (s.m % c.tm != 0 || n % c.tn != 0) {
        std::printf(" %20s", "-");
        continue;
      }
      const float ms = time_ms([&](int i) {
        c.launch(wq[i % copies], ws[i % copies], xq, xs, y, s.m, s.k, n, stream);
      });
      const double tops = 2.0 * s.m * s.k * n / (ms * 1e-3) / 1e12;
      std::printf(" %10.3f %6.1f   ", ms, tops);
      chunk[ci] += ms * s.per_layer;
    }
    std::printf("\n");
    for (int i = 0; i < copies; ++i) {
      CHECK(hipFree(wq[i]));
      CHECK(hipFree(ws[i]));
    }
    CHECK(hipFree(xq));
    CHECK(hipFree(xs));
    CHECK(hipFree(y));
  }
  std::printf("%-12s", "chunk total");
  for (double t : chunk) std::printf(" %10.1f ms        ", t);
  std::printf("\n");

  std::printf("\nProducer (rotate + quantize), %d tokens\n", n);
  for (int k : {5376, 8192, 16384, 21504}) {
    float *x, *sign, *xs;
    std::uint8_t* xq;
    CHECK(hipMalloc(&x, static_cast<std::size_t>(n) * k * 4));
    CHECK(hipMalloc(&sign, static_cast<std::size_t>(k) * 4));
    CHECK(hipMalloc(&xq, static_cast<std::size_t>(n) * k / 2));
    CHECK(hipMalloc(&xs, static_cast<std::size_t>(n) * k / 128 * 4));
    CHECK(hipMemset(x, 0x3E, static_cast<std::size_t>(n) * k * 4));
    CHECK(hipMemset(sign, 0x3F, static_cast<std::size_t>(k) * 4));
    const int groups = n * (k / 128);
    const float ms = time_ms([&](int) {
      RotateQuantize<<<(groups + 7) / 8, 256, 0, stream>>>(x, sign, xq, xs, n, k);
    });
    const double gb = (static_cast<std::size_t>(n) * k * 4.5) / (ms * 1e-3) / 1e9;
    std::printf("k %5d  %7.3f ms  %6.1f GB/s (FP32 in, int4 out)\n", k, ms, gb);
    CHECK(hipFree(x));
    CHECK(hipFree(sign));
    CHECK(hipFree(xq));
    CHECK(hipFree(xs));
  }
  return 0;
}
