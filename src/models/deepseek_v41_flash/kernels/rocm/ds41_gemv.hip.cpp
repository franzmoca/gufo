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
  static __device__ __forceinline__ float Task(const uint8_t* b, int t, const float* x, float acc) {
    return TaskQ45K<true>(b, t, x, acc);
  }
};
template <>
struct Format<8> {  // Q8_0, eight 34-byte blocks per 256 values
  static constexpr int kBlockBytes = 272;
  static constexpr int kTasks = 8;
  static __device__ __forceinline__ float Task(const uint8_t* b, int t, const float* x, float acc) {
    return TaskQ80(b, t, x, acc);
  }
};

/* Row r reads x + (r / rows_per_x) * k. Rows from m1 on come from w2 and go
 * to y2 (row r - m1). With kSwiGLU, w_up holds a second matrix of the same
 * shape whose rows are taken in the same pass, and row r stores
 * silu(min(gate, limit)) * clamp(up, +-limit). */
template <int kType, int S, bool kSwiGLU>
__global__ void __launch_bounds__(kWave* kWaves)
    gemv_kernel(const uint8_t* __restrict__ w, const uint8_t* __restrict__ w_up, float limit,
                const float* __restrict__ x, float* __restrict__ y, uint32_t m, uint32_t k, uint32_t rows_per_x,
                const uint8_t* __restrict__ w2, float* __restrict__ y2, uint32_t m1) {
  using F = Format<kType>;
  constexpr int kRowsPerWave = kWave / F::kTasks;
  constexpr int kGroups = kWaves / S;
  __shared__ float partial[kWaves][kWave];
  __shared__ float partial_up[kSwiGLU ? kWaves : 1][kWave];
  const int lane = threadIdx.x % kWave;
  const int wave = threadIdx.x / kWave;
  const int group = wave / S;
  const int slice = wave % S;
  const int t = lane % F::kTasks;
  const uint32_t row = (blockIdx.x * kGroups + group) * kRowsPerWave + lane / F::kTasks;
  const uint32_t blocks_per_row = k / 256;
  float acc = 0.0f;
  float acc_up = 0.0f;
  const bool second = row >= m1;
  if (row < m) {
    const uint64_t offset = (uint64_t)(second ? row - m1 : row) * blocks_per_row * F::kBlockBytes;
    const uint8_t* base = (second ? w2 : w) + offset;
    const float* xr = x + (uint64_t)(row / rows_per_x) * k;
#pragma unroll 2
    for (uint32_t blk = slice; blk < blocks_per_row; blk += S) {
      acc = F::Task(base + (uint64_t)blk * F::kBlockBytes, t, xr + blk * 256, acc);
      if constexpr (kSwiGLU) acc_up = F::Task(w_up + offset + (uint64_t)blk * F::kBlockBytes, t, xr + blk * 256, acc_up);
    }
  }
#pragma unroll
  for (int offset = F::kTasks / 2; offset > 0; offset >>= 1) {
    acc += __shfl_xor(acc, offset, kWave);
    if constexpr (kSwiGLU) acc_up += __shfl_xor(acc_up, offset, kWave);
  }
  if constexpr (S > 1) {
    partial[wave][lane] = acc;
    if constexpr (kSwiGLU) partial_up[wave][lane] = acc_up;
    __syncthreads();
    if (slice != 0) return;
#pragma unroll
    for (int s = 1; s < S; ++s) {
      acc += partial[wave + s][lane];
      if constexpr (kSwiGLU) acc_up += partial_up[wave + s][lane];
    }
  }
  if (t != 0 || row >= m) return;
  if constexpr (kSwiGLU) {
    float g = acc;
    float u = acc_up;
    if (limit > 0.0f) {
      u = fminf(fmaxf(u, -limit), limit);
      g = fminf(g, limit);
    }
    y[row] = g / (1.0f + expf(-g)) * u;
  } else if (second) {
    y2[row - m1] = acc;
  } else {
    y[row] = acc;
  }
}

/* The second matrix of a pair (w2, y2 from row m1), or none (m1 = m). */
struct Second {
  const void* w;
  float* y;
  uint32_t m1;
};

template <int kType, int S>
void launch(const void* w, const void* w_up, float limit, const float* x, float* y, uint32_t m, uint32_t k,
            uint32_t rows_per_x, Second second) {
  constexpr uint32_t kRows = (kWave / Format<kType>::kTasks) * (kWaves / S);
  const uint32_t blocks = (m + kRows - 1) / kRows;
  const auto* w2 = static_cast<const uint8_t*>(second.w);
  if (w_up)
    gemv_kernel<kType, S, true><<<blocks, kWave * kWaves>>>(static_cast<const uint8_t*>(w),
                                                             static_cast<const uint8_t*>(w_up), limit, x, y, m, k,
                                                             rows_per_x, w2, second.y, second.m1);
  else
    gemv_kernel<kType, S, false><<<blocks, kWave * kWaves>>>(static_cast<const uint8_t*>(w), nullptr, 0.0f, x, y, m,
                                                              k, rows_per_x, w2, second.y, second.m1);
}

template <int kType>
int dispatch(const void* w, const void* w_up, float limit, const float* x, float* y, uint32_t m, uint32_t k,
             uint32_t rows_per_x, int slices, Second second) {
  switch (slices) {
  case 1:
    launch<kType, 1>(w, w_up, limit, x, y, m, k, rows_per_x, second);
    break;
  case 2:
    launch<kType, 2>(w, w_up, limit, x, y, m, k, rows_per_x, second);
    break;
  case 4:
    launch<kType, 4>(w, w_up, limit, x, y, m, k, rows_per_x, second);
    break;
  case 8:
    launch<kType, 8>(w, w_up, limit, x, y, m, k, rows_per_x, second);
    break;
  default:
    return 1;
  }
  const hipError_t err = hipGetLastError();
  if (err == hipSuccess) return 0;
  fprintf(stderr, "ds41: gemv launch failed: %s\n", hipGetErrorString(err));
  return 1;
}

}  // namespace

namespace {

int run(int type, const void* w, const void* w_up, float limit, const float* x, float* y, uint32_t m, uint32_t k,
        uint32_t rows_per_x, int slices, Second second = {nullptr, nullptr, 0}) {
  if (k % 256 != 0 || m == 0) return 1;
  if (rows_per_x == 0) rows_per_x = m;
  if (slices <= 0) slices = 8;
  if (!second.w) second.m1 = m;
  switch (type) {
  case 13:
    return dispatch<13>(w, w_up, limit, x, y, m, k, rows_per_x, slices, second);
  case 8:
    return dispatch<8>(w, w_up, limit, x, y, m, k, rows_per_x, slices, second);
  default:
    return 1;
  }
}

}  // namespace

extern "C" int ds41_v41_gemv(int type, const void* w, const float* x, float* y, uint32_t m, uint32_t k,
                             uint32_t rows_per_x, int slices) {
  return run(type, w, nullptr, 0.0f, x, y, m, k, rows_per_x, slices);
}

extern "C" int ds41_v41_gemv_pair(int type, const void* w1, float* y1, uint32_t m1, const void* w2, float* y2,
                                  uint32_t m2, const float* x, uint32_t k, int slices) {
  if (!w2 || m2 == 0) return 1;
  return run(type, w1, nullptr, 0.0f, x, y1, m1 + m2, k, 0, slices, Second{w2, y2, m1});
}

extern "C" int ds41_v41_gemv_swiglu(int type, const void* gate, const void* up, const float* x, float* y, uint32_t m,
                                    uint32_t k, float limit, int slices) {
  if (!up) return 1;
  return run(type, gate, up, limit, x, y, m, k, 0, slices);
}
