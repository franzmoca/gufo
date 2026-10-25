/* Single-token decode projections for the V4.1 Flash dense Q5_K and Q8_0
 * weights, after the Gemma 4 K-quant GEMV (src/models/gemma4/kernels/rocm/
 * gemv.hip.cpp): weights stay in their GGUF blocks, activations stay FP32. A
 * lane's unit of work is one 16-byte vector of quantized values; lanes
 * holding the same task of different rows read the same activations, which
 * the hardware broadcasts. A workgroup's eight waves form 8 / S row groups
 * of S waves; the S waves of a group take interleaved super-blocks of its
 * rows and are summed in wave order. */
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>

#include "ds41_gemv_tasks.hpp"
#include "ds41_v41.h"

namespace {

using namespace ds41_gemv_tasks;

constexpr int kWave = 32;
constexpr int kWaves = 8;

template <int kType>
struct Format;
template <>
struct Format<13> {  // Q5_K
  static constexpr int kBlockBytes = 176;
  static constexpr int kTasks = 8;
  using Raw = RawQ45K;
  static __device__ __forceinline__ Raw Load(const uint8_t* b, int t) { return LoadQ45K<true>(b, t); }
  static __device__ __forceinline__ void Decode(const Raw& r, int t, float (&w)[32]) { DecodeQ45K<true>(r, t, w); }
  static __device__ __forceinline__ void Decode(const uint8_t* b, int t, float (&w)[32]) { DecodeQ45K<true>(b, t, w); }
  static __device__ __forceinline__ float Dot(float acc, const float (&w)[32], int t, const float* x) {
    return DotQ45K(acc, w, t, x);
  }
};
template <>
struct Format<8> {  // Q8_0, eight 34-byte blocks per 256 values
  static constexpr int kBlockBytes = 272;
  static constexpr int kTasks = 8;
  static __device__ __forceinline__ void Decode(const uint8_t* b, int t, float (&w)[32]) {
    DecodeQ8_0(b + 34 * t, (t & 1) != 0, w);
  }
  static __device__ __forceinline__ float Dot(float acc, const float (&w)[32], int t, const float* x) {
    return Accumulate<32>(acc, w, x + 32 * t);
  }
};

/* Row r of token c reads x + c * x_stride + (r / rows_per_x) * k, for the
 * first n <= kCols tokens. Rows from m1 on come from w2 and go to y2 (row
 * r - m1, token stride m - m1); the others to y (token stride y_stride). With
 * kSwiGLU, w_up holds a second matrix of the same shape whose rows are taken
 * in the same pass, and row r stores silu(min(gate, limit)) * clamp(up,
 * +-limit). Each lane decodes its weights once for all tokens; every token
 * sees the FMA order of a single-token pass. */
template <int kType, int S, bool kSwiGLU, int kCols>
__global__ void __launch_bounds__(kWave* kWaves)
    gemv_kernel(const uint8_t* __restrict__ w, const uint8_t* __restrict__ w_up, float limit,
                const float* __restrict__ x, uint64_t x_stride, float* __restrict__ y, uint32_t y_stride, uint32_t m,
                uint32_t k, uint32_t rows_per_x, const uint8_t* __restrict__ w2, float* __restrict__ y2, uint32_t m1,
                uint32_t n) {
  using F = Format<kType>;
  constexpr int kRowsPerWave = kWave / F::kTasks;
  constexpr int kGroups = kWaves / S;
  __shared__ float partial[S > 1 ? kCols : 1][kWaves][kWave];
  __shared__ float partial_up[S > 1 && kSwiGLU ? kCols : 1][kWaves][kWave];
  const int lane = threadIdx.x % kWave;
  const int wave = threadIdx.x / kWave;
  const int group = wave / S;
  const int slice = wave % S;
  const int t = lane % F::kTasks;
  const uint32_t row = (blockIdx.x * kGroups + group) * kRowsPerWave + lane / F::kTasks;
  const uint32_t blocks_per_row = k / 256;
  float acc[kCols] = {};
  float acc_up[kCols] = {};
  const bool second = row >= m1;
  if (row < m) {
    const uint64_t offset = (uint64_t)(second ? row - m1 : row) * blocks_per_row * F::kBlockBytes;
    const uint8_t* base = (second ? w2 : w) + offset;
    const float* xr = x + (uint64_t)(row / rows_per_x) * k;
    if constexpr (kCols > 1 && requires { typename F::Raw; }) {
      /* Several tokens make each block's FMAs long; the next block's
       * weights load before this block is decoded. */
      const uint8_t* base_up = kSwiGLU ? w_up + offset : nullptr;
      typename F::Raw next{}, next_up{};
      if (slice < (int)blocks_per_row) {
        next = F::Load(base + (uint64_t)slice * F::kBlockBytes, t);
        if constexpr (kSwiGLU) next_up = F::Load(base_up + (uint64_t)slice * F::kBlockBytes, t);
      }
      for (uint32_t blk = slice; blk < blocks_per_row; blk += S) {
        const typename F::Raw raw = next, raw_up = next_up;
        if (blk + S < blocks_per_row) {
          next = F::Load(base + (uint64_t)(blk + S) * F::kBlockBytes, t);
          if constexpr (kSwiGLU) next_up = F::Load(base_up + (uint64_t)(blk + S) * F::kBlockBytes, t);
        }
        float wv[32];
        F::Decode(raw, t, wv);
#pragma unroll
        for (int c = 0; c < kCols; ++c)
          if (c < (int)n) acc[c] = F::Dot(acc[c], wv, t, xr + c * x_stride + blk * 256);
        if constexpr (kSwiGLU) {
          F::Decode(raw_up, t, wv);
#pragma unroll
          for (int c = 0; c < kCols; ++c)
            if (c < (int)n) acc_up[c] = F::Dot(acc_up[c], wv, t, xr + c * x_stride + blk * 256);
        }
      }
    } else
#pragma unroll 2
    for (uint32_t blk = slice; blk < blocks_per_row; blk += S) {
      float wv[32];
      F::Decode(base + (uint64_t)blk * F::kBlockBytes, t, wv);
#pragma unroll
      for (int c = 0; c < kCols; ++c)
        if (kCols == 1 || c < (int)n) acc[c] = F::Dot(acc[c], wv, t, xr + c * x_stride + blk * 256);
      if constexpr (kSwiGLU) {
        F::Decode(w_up + offset + (uint64_t)blk * F::kBlockBytes, t, wv);
#pragma unroll
        for (int c = 0; c < kCols; ++c)
          if (kCols == 1 || c < (int)n) acc_up[c] = F::Dot(acc_up[c], wv, t, xr + c * x_stride + blk * 256);
      }
    }
  }
#pragma unroll
  for (int c = 0; c < kCols; ++c) {
#pragma unroll
    for (int offset = F::kTasks / 2; offset > 0; offset >>= 1) {
      acc[c] += __shfl_xor(acc[c], offset, kWave);
      if constexpr (kSwiGLU) acc_up[c] += __shfl_xor(acc_up[c], offset, kWave);
    }
  }
  if constexpr (S > 1) {
#pragma unroll
    for (int c = 0; c < kCols; ++c) {
      partial[c][wave][lane] = acc[c];
      if constexpr (kSwiGLU) partial_up[c][wave][lane] = acc_up[c];
    }
    __syncthreads();
    if (slice != 0) return;
#pragma unroll
    for (int c = 0; c < kCols; ++c) {
#pragma unroll
      for (int s = 1; s < S; ++s) {
        acc[c] += partial[c][wave + s][lane];
        if constexpr (kSwiGLU) acc_up[c] += partial_up[c][wave + s][lane];
      }
    }
  }
  if (t != 0 || row >= m) return;
#pragma unroll
  for (int c = 0; c < kCols; ++c) {
    if (kCols > 1 && c >= (int)n) break;
    if constexpr (kSwiGLU) {
      float g = acc[c];
      float u = acc_up[c];
      if (limit > 0.0f) {
        u = fminf(fmaxf(u, -limit), limit);
        g = fminf(g, limit);
      }
      y[(uint64_t)c * y_stride + row] = g / (1.0f + expf(-g)) * u;
    } else if (second) {
      y2[(uint64_t)c * (m - m1) + row - m1] = acc[c];
    } else {
      y[(uint64_t)c * y_stride + row] = acc[c];
    }
  }
}

/* The second matrix of a pair (w2, y2 from row m1), or none (m1 = m). */
struct Second {
  const void* w;
  float* y;
  uint32_t m1;
};

/* Tokens of one launch: x rows x_stride floats apart, y rows y_stride apart. */
struct Tokens {
  uint32_t n;
  uint64_t x_stride;
  uint32_t y_stride;
};

template <int kType, int S, int kCols>
void launch(const void* w, const void* w_up, float limit, const float* x, float* y, uint32_t m, uint32_t k,
            uint32_t rows_per_x, Second second, Tokens tokens) {
  constexpr uint32_t kRows = (kWave / Format<kType>::kTasks) * (kWaves / S);
  const uint32_t blocks = (m + kRows - 1) / kRows;
  const auto* w2 = static_cast<const uint8_t*>(second.w);
  if (w_up)
    gemv_kernel<kType, S, true, kCols><<<blocks, kWave * kWaves>>>(
        static_cast<const uint8_t*>(w), static_cast<const uint8_t*>(w_up), limit, x, tokens.x_stride, y,
        tokens.y_stride, m, k, rows_per_x, w2, second.y, second.m1, tokens.n);
  else
    gemv_kernel<kType, S, false, kCols><<<blocks, kWave * kWaves>>>(
        static_cast<const uint8_t*>(w), nullptr, 0.0f, x, tokens.x_stride, y, tokens.y_stride, m, k, rows_per_x, w2,
        second.y, second.m1, tokens.n);
}

template <int kType, int kCols>
bool dispatch_slices(const void* w, const void* w_up, float limit, const float* x, float* y, uint32_t m, uint32_t k,
                     uint32_t rows_per_x, int slices, Second second, Tokens tokens) {
  switch (slices) {
  case 1:
    launch<kType, 1, kCols>(w, w_up, limit, x, y, m, k, rows_per_x, second, tokens);
    return true;
  case 2:
    launch<kType, 2, kCols>(w, w_up, limit, x, y, m, k, rows_per_x, second, tokens);
    return true;
  case 4:
    launch<kType, 4, kCols>(w, w_up, limit, x, y, m, k, rows_per_x, second, tokens);
    return true;
  case 8:
    launch<kType, 8, kCols>(w, w_up, limit, x, y, m, k, rows_per_x, second, tokens);
    return true;
  default:
    return false;
  }
}

template <int kType>
int dispatch(const void* w, const void* w_up, float limit, const float* x, float* y, uint32_t m, uint32_t k,
             uint32_t rows_per_x, int slices, Second second, Tokens tokens) {
  bool ok = false;
  if (tokens.n == 1)
    ok = dispatch_slices<kType, 1>(w, w_up, limit, x, y, m, k, rows_per_x, slices, second, tokens);
  else if (tokens.n == 2)
    ok = dispatch_slices<kType, 2>(w, w_up, limit, x, y, m, k, rows_per_x, slices, second, tokens);
  else if (tokens.n <= 4)
    ok = dispatch_slices<kType, 4>(w, w_up, limit, x, y, m, k, rows_per_x, slices, second, tokens);
  else if (tokens.n <= 8)
    ok = dispatch_slices<kType, 8>(w, w_up, limit, x, y, m, k, rows_per_x, slices, second, tokens);
  if (!ok) return 1;
  const hipError_t err = hipGetLastError();
  if (err == hipSuccess) return 0;
  fprintf(stderr, "ds41: gemv launch failed: %s\n", hipGetErrorString(err));
  return 1;
}

}  // namespace

namespace {

int run(int type, const void* w, const void* w_up, float limit, const float* x, float* y, uint32_t m, uint32_t k,
        uint32_t rows_per_x, int slices, Second second, Tokens tokens) {
  if (k % 256 != 0 || m == 0 || tokens.n == 0 || tokens.n > DS41_V41_GEMV_MAX_TOKENS) return 1;
  if (rows_per_x == 0) rows_per_x = m;
  if (slices <= 0) slices = 8;
  if (!second.w) second.m1 = m;
  switch (type) {
  case 13:
    return dispatch<13>(w, w_up, limit, x, y, m, k, rows_per_x, slices, second, tokens);
  case 8:
    return dispatch<8>(w, w_up, limit, x, y, m, k, rows_per_x, slices, second, tokens);
  default:
    return 1;
  }
}

}  // namespace

extern "C" int ds41_v41_gemv(int type, const void* w, const float* x, float* y, uint32_t m, uint32_t k,
                             uint32_t rows_per_x, int slices, uint32_t n, uint64_t x_stride, uint32_t y_stride) {
  return run(type, w, nullptr, 0.0f, x, y, m, k, rows_per_x, slices, Second{nullptr, nullptr, 0},
             Tokens{n, x_stride, y_stride});
}

extern "C" int ds41_v41_gemv_pair(int type, const void* w1, float* y1, uint32_t m1, const void* w2, float* y2,
                                  uint32_t m2, const float* x, uint32_t k, int slices, uint32_t n) {
  if (!w2 || m2 == 0) return 1;
  return run(type, w1, nullptr, 0.0f, x, y1, m1 + m2, k, 0, slices, Second{w2, y2, m1}, Tokens{n, k, m1});
}

extern "C" int ds41_v41_gemv_swiglu(int type, const void* gate, const void* up, const float* x, float* y, uint32_t m,
                                    uint32_t k, float limit, int slices, uint32_t n) {
  if (!up) return 1;
  return run(type, gate, up, limit, x, y, m, k, 0, slices, Second{nullptr, nullptr, 0}, Tokens{n, k, m});
}
