// Device helpers for the quantization simulator (tools/gemma4/quant_sim),
// built by fastops.py against the ROCm runtime PyTorch loads.
//
// SyrkAccumulate: H += diag(c) Xs^T Xs diag(c) for one calibration row
// block, Xs given transposed as [K][T] binary16 (columns scaled to max
// |x| = 1, so products
// cannot overflow), c [K] FP32. Products accumulate in FP32 across every
// token on v_wmma_f32_16x16x16_f16 and the scaled tile is added to H in
// FP32: only the inputs are rounded. Only tiles with j >= i are written;
// the caller mirrors the upper triangle.
#include <hip/hip_runtime.h>

#include <cstdint>

namespace {

typedef _Float16 h16 __attribute__((ext_vector_type(16)));
typedef float f8 __attribute__((ext_vector_type(8)));

constexpr int kTile = 128;    // output tile (channels i, j)
constexpr int kSteps = 32;    // tokens per LDS stage
constexpr int kThreads = 256; // 8 waves: 4 (i) x 2 (j), each 32 x 64

/// LDS: [channel][token] for one 128-channel panel, a lane's 16 tokens
/// contiguous; rows padded by 8 halves against bank conflicts.
constexpr int kLd = kSteps + 8;

__global__ __launch_bounds__(kThreads) void SyrkKernel(
    const _Float16* __restrict__ xs, const float* __restrict__ col,
    float* __restrict__ h, int tokens, int k) {
  __shared__ __attribute__((aligned(16))) _Float16 pa[kTile * kLd];
  __shared__ __attribute__((aligned(16))) _Float16 pb[kTile * kLd];
  // Upper-triangular tile index -> (bi, bj), bj >= bi.
  int t = blockIdx.x;
  int bi = 0;
  const int nb = k / kTile;
  while (t >= nb - bi) {
    t -= nb - bi;
    ++bi;
  }
  const int bj = bi + t;
  const int i0 = bi * kTile, j0 = bj * kTile;
  const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
  const int lr = lane & 15, hi = lane >> 4;
  const int wi = wave & 3, wj = wave >> 2;

  f8 acc[2][4];
#pragma unroll
  for (int a = 0; a < 2; ++a)
#pragma unroll
    for (int b = 0; b < 4; ++b) acc[a][b] = (f8){};

  for (int t0 = 0; t0 < tokens; t0 += kSteps) {
    // Stage: 128 channels x 32 tokens per panel from the [K][T] input;
    // thread -> 8 consecutive tokens (16 bytes) of one channel.
#pragma unroll
    for (int rep = 0; rep < 2; ++rep) {
      const int idx = tid + rep * kThreads;  // 0..511
      const int ch = idx >> 2;               // 0..127
      const int tok = (idx & 3) * 8;         // 0, 8, 16, 24
      typedef _Float16 h8 __attribute__((ext_vector_type(8)));
      *reinterpret_cast<h8*>(&pa[ch * kLd + tok]) = *reinterpret_cast<const h8*>(
          xs + static_cast<std::size_t>(i0 + ch) * tokens + t0 + tok);
      *reinterpret_cast<h8*>(&pb[ch * kLd + tok]) = *reinterpret_cast<const h8*>(
          xs + static_cast<std::size_t>(j0 + ch) * tokens + t0 + tok);
    }
    __syncthreads();
#pragma unroll
    for (int s = 0; s < kSteps; s += 16) {
      h16 av[2], bv[4];
#pragma unroll
      for (int a = 0; a < 2; ++a)
        av[a] = *reinterpret_cast<const h16*>(&pa[(wi * 32 + a * 16 + lr) * kLd + s]);
#pragma unroll
      for (int b = 0; b < 4; ++b)
        bv[b] = *reinterpret_cast<const h16*>(&pb[(wj * 64 + b * 16 + lr) * kLd + s]);
#pragma unroll
      for (int a = 0; a < 2; ++a)
#pragma unroll
        for (int b = 0; b < 4; ++b)
          acc[a][b] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(av[a], bv[b], acc[a][b]);
    }
    __syncthreads();
  }
  // Lane (hi, lr) holds rows 2r + hi (channel i) of column lr (channel j).
#pragma unroll
  for (int a = 0; a < 2; ++a) {
#pragma unroll
    for (int b = 0; b < 4; ++b) {
      const int j = j0 + wj * 64 + b * 16 + lr;
      const float cj = col[j];
#pragma unroll
      for (int r = 0; r < 8; ++r) {
        const int i = i0 + wi * 32 + a * 16 + 2 * r + hi;
        h[static_cast<std::size_t>(i) * k + j] += acc[a][b][r] * col[i] * cj;
      }
    }
  }
}

}  // namespace

extern "C" int SyrkAccumulate(const void* xs, const float* col, float* h,
                              int tokens, int k, hipStream_t stream) {
  if (k % kTile != 0 || tokens % kSteps != 0) return 1;
  const int nb = k / kTile;
  const int tiles = nb * (nb + 1) / 2;
  SyrkKernel<<<tiles, kThreads, 0, stream>>>(
      static_cast<const _Float16*>(xs), col, h, tokens, k);
  return hipGetLastError() == hipSuccess ? 0 : 2;
}
