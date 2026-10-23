#include "src/models/gemma4/kernels/rocm/w4a4.hpp"

#include <hip/hip_fp16.h>

namespace gufo::models::gemma4::rocm {
namespace {

typedef int v2i __attribute__((ext_vector_type(2)));
typedef int v4i __attribute__((ext_vector_type(4)));
typedef int v8i __attribute__((ext_vector_type(8)));

std::size_t CodeBytes(std::uint32_t padded, std::uint32_t k) {
  return std::size_t{k / 128} * padded * 64;
}

/// Lane k of the lane's own row of 16 (DPP row_share).
__device__ __forceinline__ float RowShare(float v, int k) {
  const int x = __float_as_int(v);
  int r;
  switch (k) {
    case 0:
      r = __builtin_amdgcn_update_dpp(0, x, 0x150, 0xf, 0xf, false);
      break;
    case 2:
      r = __builtin_amdgcn_update_dpp(0, x, 0x152, 0xf, 0xf, false);
      break;
    case 4:
      r = __builtin_amdgcn_update_dpp(0, x, 0x154, 0xf, 0xf, false);
      break;
    case 6:
      r = __builtin_amdgcn_update_dpp(0, x, 0x156, 0xf, 0xf, false);
      break;
    case 8:
      r = __builtin_amdgcn_update_dpp(0, x, 0x158, 0xf, 0xf, false);
      break;
    case 10:
      r = __builtin_amdgcn_update_dpp(0, x, 0x15a, 0xf, 0xf, false);
      break;
    case 12:
      r = __builtin_amdgcn_update_dpp(0, x, 0x15c, 0xf, 0xf, false);
      break;
    case 14:
      r = __builtin_amdgcn_update_dpp(0, x, 0x15e, 0xf, 0xf, false);
      break;
    default:
      __builtin_unreachable();
  }
  return __int_as_float(r);
}

/// One wave per (row, 128-block): lane l holds rotated values 4l..4l+3.
__global__ __launch_bounds__(256) void W4A4QuantizeKernel(
    const __half* __restrict__ x, const float* __restrict__ vec,
    std::uint8_t* __restrict__ xq, float* __restrict__ xs, int rows, int padded,
    int k) {
  const int lane = threadIdx.x & 31;
  const int group = blockIdx.x * 8 + (threadIdx.x >> 5);
  const int blocks = k / 128;
  if (group >= padded * blocks) {
    return;
  }
  const int t = group / blocks;
  const int b = group % blocks;
  const std::size_t g = static_cast<std::size_t>(b) * padded + t;
  if (t >= rows) {
    *reinterpret_cast<std::uint16_t*>(xq + g * 64 + lane * 2) = 0;
    if (lane == 0) {
      xs[g] = 0.0F;
    }
    return;
  }
  const __half* src = x + static_cast<std::size_t>(t) * k + b * 128 + lane * 4;
  const float2 lo = __half22float2(*reinterpret_cast<const __half2*>(src));
  const float2 hi = __half22float2(*reinterpret_cast<const __half2*>(src + 2));
  const float4 sg = *reinterpret_cast<const float4*>(vec + b * 128 + lane * 4);
  float v[4] = {lo.x * sg.x, lo.y * sg.y, hi.x * sg.z, hi.y * sg.w};
  // Butterflies of span 1 and 2 inside the lane, 4 to 64 across lanes.
  const float a0 = v[0] + v[1];
  const float a1 = v[0] - v[1];
  const float a2 = v[2] + v[3];
  const float a3 = v[2] - v[3];
  v[0] = a0 + a2;
  v[1] = a1 + a3;
  v[2] = a0 - a2;
  v[3] = a1 - a3;
#pragma unroll
  for (int mask = 1; mask < 32; mask <<= 1) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const float p = __shfl_xor(v[i], mask, 32);
      v[i] = (lane & mask) != 0 ? p - v[i] : v[i] + p;
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
  // The clip with the least squared error; the first of equal ones.
  constexpr float kClip[4] = {1.0F, 0.9F, 0.8F, 0.7F};
  float best = 0.0F;
  float best_err = 0.0F;
#pragma unroll
  for (int c = 0; c < 4; ++c) {
    const float s = amax * kClip[c] / 7.0F;
    float err = 0.0F;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const float q =
          s > 0.0F ? fminf(fmaxf(rintf(v[i] / s), -8.0F), 7.0F) : 0.0F;
      const float e = q * s - v[i];
      err += e * e;
    }
#pragma unroll
    for (int mask = 16; mask >= 1; mask >>= 1) {
      err += __shfl_xor(err, mask, 32);
    }
    if (c == 0 || err < best_err) {
      best = s;
      best_err = err;
    }
  }
  std::uint32_t packed = 0;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int q =
        best > 0.0F
            ? static_cast<int>(fminf(fmaxf(rintf(v[i] / best), -8.0F), 7.0F))
            : 0;
    packed |= (static_cast<std::uint32_t>(q) & 0xFU) << (4 * i);
  }
  *reinterpret_cast<std::uint16_t*>(xq + g * 64 + lane * 2) =
      static_cast<std::uint16_t>(packed);
  if (lane == 0) {
    xs[g] = best;
  }
}

/// One block computes TM weight rows x TN tokens; waves WM x WN, each
/// NA x NC 16x16 fragments. K walks in 128-value blocks, double-buffered in
/// LDS in fragment-major order (each 16x16 operand fragment one 128-byte
/// run). Rows past `rows` (tile padding) are not stored.
template<int TM, int TN, int WM, int WN>
__global__ __launch_bounds__(32 * WM * WN) void W4A4GemmKernel(
    const std::uint8_t* __restrict__ wq, const __half* __restrict__ ws,
    const std::uint8_t* __restrict__ xq, const float* __restrict__ xs,
    float* __restrict__ y, int m, int k, int padded, int rows) {
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
  // Token tiles fastest: the blocks in flight share a few weight row tiles
  // and the activations stay in the MALL.
  const int col0 = blockIdx.x * TN;
  const int row0 = blockIdx.y * TM;
  const int blocks = k / 128;

  // Staging: thread t loads 32 bytes (K16 slices 4h..4h+3 of one row).
  auto stage_load = [&](int b, v4i(&ra)[kAVec][2], v4i(&rx)[kXVec][2]) {
    const std::uint8_t* ga = wq + (static_cast<std::size_t>(b) * m + row0) * 64;
    const std::uint8_t* gx =
        xq + (static_cast<std::size_t>(b) * padded + col0) * 64;
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
  auto frag_off = [](int r, int s) {
    return ((r >> 4) * 8 + s) * 128 + (r & 15) * 8;
  };
  auto stage_store = [&](std::uint8_t* slot, const v4i(&ra)[kAVec][2],
                         const v4i(&rx)[kXVec][2]) {
#pragma unroll
    for (int i = 0; i < kAVec; ++i) {
      const int idx = tid + i * kThreads;
      const int r = idx >> 1;
      const int s0 = (idx & 1) * 4;
      const v4i a = ra[i][0];
      const v4i b = ra[i][1];
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
      const v4i a = rx[i][0];
      const v4i b = rx[i][1];
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
      sx[c] = xs[static_cast<std::size_t>(b) * padded + xcol + c * 16];
    }
  };

  float sum[NA][NC][8];
#pragma unroll
  for (int a = 0; a < NA; ++a) {
#pragma unroll
    for (int c = 0; c < NC; ++c) {
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        sum[a][c][j] = 0.0F;
      }
    }
  }

  v4i ra[kAVec][2];
  v4i rx[kXVec][2];
  float sa[NA];
  float sx[NC];
  float sa_n[NA];
  float sx_n[NC];
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
    const std::uint8_t* la = slot + wm * NA * 8 * 128 + lr * 8;
    const std::uint8_t* lx = slot + kABytes + wn * NC * 8 * 128 + lr * 8;
    // Flattened (a, s) steps, operands loaded one step ahead. The empty asm
    // statements pin the order: without them the WMMAs sink below every
    // step's loads and all operands stay live at once.
    auto load_a = [&](int i) {
      int o = static_cast<int>(la - lds) + i * 128;
      asm volatile("" : "+v"(o));
      return *reinterpret_cast<const v2i*>(lds + o);
    };
    auto load_x = [&](int s, v2i(&xv)[NC]) {
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
          for (int c = 0; c < NC; ++c) {
            xv[c] = xv_n[c];
          }
        }
      }
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const float wsc = RowShare(sa[a], 2 * j);
#pragma unroll
        for (int c = 0; c < NC; ++c) {
          sum[a][c][j] = __builtin_fmaf(
              wsc * sx[c], static_cast<float>(acc[c][j]), sum[a][c][j]);
        }
      }
      // This fragment's sums exist here, so its int32 chains end here.
#pragma unroll
      for (int c = 0; c < NC; ++c) {
        asm volatile(""
                     : "+v"(sum[a][c][0]), "+v"(sum[a][c][1]),
                       "+v"(sum[a][c][2]), "+v"(sum[a][c][3]),
                       "+v"(sum[a][c][4]), "+v"(sum[a][c][5]),
                       "+v"(sum[a][c][6]), "+v"(sum[a][c][7]));
      }
      __builtin_amdgcn_sched_barrier(0);
    }
    if (more) {
      stage_store(lds + ((b + 1) & 1) * kSlot, ra, rx);
#pragma unroll
      for (int a = 0; a < NA; ++a) {
        sa[a] = sa_n[a];
      }
#pragma unroll
      for (int c = 0; c < NC; ++c) {
        sx[c] = sx_n[c];
      }
    }
    __syncthreads();
  }
  // Lane (hi, lr): token xcol + 16c, rows 2j + hi of each fragment.
#pragma unroll
  for (int c = 0; c < NC; ++c) {
    const int token = xcol + c * 16;
    if (token >= rows) {
      continue;
    }
    float* yr =
        y + static_cast<std::size_t>(token) * m + row0 + wm * NA * 16 + hi;
#pragma unroll
    for (int a = 0; a < NA; ++a) {
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        yr[a * 16 + 2 * j] = sum[a][c][j];
      }
    }
  }
}

}  // namespace

std::size_t W4A4ActivationBytes(std::uint32_t rows, std::uint32_t k) noexcept {
  const std::uint32_t padded = W4A4PaddedRows(rows);
  return CodeBytes(padded, k) + std::size_t{k / 128} * padded * sizeof(float);
}

void LaunchW4A4Quantize(const void* x, const float* vec, void* xq,
                        std::uint32_t rows, std::uint32_t k,
                        hipStream_t stream) {
  const std::uint32_t padded = W4A4PaddedRows(rows);
  auto* codes = static_cast<std::uint8_t*>(xq);
  auto* scales = reinterpret_cast<float*>(codes + CodeBytes(padded, k));
  const std::size_t groups = std::size_t{padded} * (k / 128);
  W4A4QuantizeKernel<<<static_cast<unsigned>((groups + 7) / 8), 256, 0,
                       stream>>>(static_cast<const __half*>(x), vec, codes,
                                 scales, static_cast<int>(rows),
                                 static_cast<int>(padded), static_cast<int>(k));
}

bool LaunchW4A4Gemm(const void* wq, const void* ws, const void* xq, float* y,
                    std::uint32_t rows, std::uint32_t m, std::uint32_t k,
                    hipStream_t stream) {
  constexpr int kTm = 128;
  constexpr int kTn = 128;
  static_assert(kTn == kW4A4TileRows, "activation padding matches the tile");
  if (rows == 0 || m % kTm != 0 || k % 128 != 0) {
    return false;
  }
  const std::uint32_t padded = W4A4PaddedRows(rows);
  const auto* codes = static_cast<const std::uint8_t*>(xq);
  const auto* scales =
      reinterpret_cast<const float*>(codes + CodeBytes(padded, k));
  W4A4GemmKernel<kTm, kTn, 4, 2>
      <<<dim3(padded / kTn, m / kTm), 256, 0, stream>>>(
          static_cast<const std::uint8_t*>(wq), static_cast<const __half*>(ws),
          codes, scales, y, static_cast<int>(m), static_cast<int>(k),
          static_cast<int>(padded), static_cast<int>(rows));
  return true;
}

}  // namespace gufo::models::gemma4::rocm
