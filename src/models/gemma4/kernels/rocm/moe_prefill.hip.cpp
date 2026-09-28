// Routed expert GEMM for prefill over binary16 activations, for expert
// formats Flash-Next's routed binary16 GEMM does not decode (Q6_K). It reads
// the same routing layout: RoutedCompact's 16-padded expert buckets and a
// tile map of expert | tile << 16 entries.
//
// A block computes 128 output rows (eight waves of 16) for one tile of
// kTileTokens bucket rows. Per 32-value K block the tile's activations are
// staged in LDS (double buffered, one barrier per block) and every lane
// decodes its own weight row straight into binary16 WMMA fragments.
// Fragment layout (wave32 v_wmma_f32_16x16x16_f16): A holds row L%16 and B
// column L%16, each with 16 K values in both half-waves; C lane L holds
// column L%16, rows 2 i + L/16.
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstdint>

#include "src/models/gemma4/kernels/rocm/moe.hpp"

namespace gufo::models::gemma4::rocm {
namespace {

using v16h = __attribute__((__vector_size__(16 * sizeof(_Float16)))) _Float16;
using v8f = __attribute__((__vector_size__(8 * sizeof(float)))) float;

constexpr int kThreads = 256;
constexpr int kWave = 32;
constexpr int kRowsPerBlock = 128;
/// Halves per staged token row: 32 values plus padding against bank
/// conflicts on the fragment reads.
constexpr int kActStride = 40;

__device__ __forceinline__ v8f Wmma(v16h a, v16h b, v8f c) {
  return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
}

/// Four byte codes q (< 1024) as two half2 of (q - bias) * scale: the codes
/// become 1024 + q through the exponent byte 0x64, exactly.
__device__ __forceinline__ void CodesToHalves(std::uint32_t codes,
                                              __half2 offset, __half2 scale,
                                              __half2* out) {
  constexpr std::uint32_t kMagic = 0x64646464U;
  const std::uint32_t p0 = __builtin_amdgcn_perm(codes, kMagic, 0x01050004U);
  const std::uint32_t p1 = __builtin_amdgcn_perm(codes, kMagic, 0x03070206U);
  out[0] = __hmul2(__hsub2(__builtin_bit_cast(__half2, p0), offset), scale);
  out[1] = __hmul2(__hsub2(__builtin_bit_cast(__half2, p1), offset), scale);
}

/// Q6_K K block `kb` (32 values) of a row as two fragments. Block j = kb % 8
/// of a super-block is half n = j / 4, quarter s = j % 4: low (s < 2) or
/// high nibbles of ql[64 n + 32 (s % 2) + l], bits 2 s of qh[32 n + l],
/// scales sc[8 n + 2 s] (l < 16) and sc[8 n + 2 s + 1], offset 32.
__device__ __forceinline__ void DecodeQ6K(const std::uint8_t* row, int kb,
                                          v16h* lo, v16h* hi) {
  const std::uint8_t* block = row + (kb / 8) * 210;
  const int j = kb % 8;
  const int n = j / 4;
  const int s = j % 4;
  const uint4* ql =
      reinterpret_cast<const uint4*>(block + 64 * n + 32 * (s & 1));
  const uint4* qh = reinterpret_cast<const uint4*>(block + 128 + 32 * n);
  const float d = __half2float(*reinterpret_cast<const __half*>(block + 208));
  const auto* sc = reinterpret_cast<const std::int8_t*>(block + 192 + 8 * n);
  const int nibble = (s >> 1) * 4;
  const int high = 2 * s;
  const __half2 offset = __float2half2_rn(1056.0F);
#pragma unroll
  for (int part = 0; part < 2; ++part) {
    const uint4 l4 = ql[part];
    const uint4 h4 = qh[part];
    const std::uint32_t lw[4] = {l4.x, l4.y, l4.z, l4.w};
    const std::uint32_t hw[4] = {h4.x, h4.y, h4.z, h4.w};
    const __half2 scale =
        __float2half2_rn(d * static_cast<float>(sc[2 * s + part]));
    __half2 h[8];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const std::uint32_t codes = ((lw[i] >> nibble) & 0x0F0F0F0FU) |
                                  (((hw[i] >> high) & 0x03030303U) << 4U);
      CodesToHalves(codes, offset, scale, &h[2 * i]);
    }
    __builtin_memcpy(part == 0 ? lo : hi, h, 32);
  }
}

template<int kTileTokens>
__global__ void __launch_bounds__(kThreads)
    RoutedHalfQ6KKernel(const std::uint8_t* __restrict__ w,
                        const __half* __restrict__ x,
                        const std::int32_t* __restrict__ tiles,
                        const std::int32_t* __restrict__ pad_bounds,
                        const std::int32_t* __restrict__ rows_in,
                        const std::int32_t* __restrict__ rows_out,
                        float* __restrict__ out, __half* __restrict__ out_half,
                        std::uint32_t m, std::uint32_t k) {
  constexpr int kTokTiles = kTileTokens / 16;
  constexpr int kLoads = kTileTokens * 4;  // uint4 per staged K block
  __shared__
      __attribute__((aligned(16))) __half s_act[2][kTileTokens * kActStride];
  const std::int32_t tile = tiles[blockIdx.y];
  const int expert = tile & 0xFFFF;
  const int t_local = (tile >> 16) * kTileTokens;
  const int bucket_begin = pad_bounds[expert];
  const int bucket_rows = pad_bounds[expert + 1] - bucket_begin;
  if (t_local >= bucket_rows) {
    return;
  }
  const int tid = static_cast<int>(threadIdx.x);
  const int lane = tid % kWave;
  const int wave = tid / kWave;
  const int sub = lane & 15;
  const int half = lane >> 4;
  const std::uint32_t row =
      blockIdx.x * kRowsPerBlock + static_cast<std::uint32_t>(wave * 16 + sub);
  const std::size_t row_bytes = std::size_t{k} / 256 * 210;
  const std::uint8_t* w_row =
      w + (std::size_t{static_cast<std::uint32_t>(expert)} * m +
           (row < m ? row : m - 1)) *
              row_bytes;

  // Activation staging: thread tid < kLoads loads 8 values (a quarter of
  // the K block) of token tid / 4.
  const int a_tok = tid / 4;
  const int a_quarter = tid % 4;
  const uint4* a_src = nullptr;
  if (tid < kLoads && t_local + a_tok < bucket_rows) {
    const std::int32_t src = rows_in[bucket_begin + t_local + a_tok];
    if (src >= 0) {
      a_src = reinterpret_cast<const uint4*>(
                  x + std::size_t{static_cast<std::uint32_t>(src)} * k) +
              a_quarter;
    }
  }
  const auto load = [&](int kb) {
    return a_src != nullptr ? a_src[kb * 4] : make_uint4(0U, 0U, 0U, 0U);
  };

  v8f acc[kTokTiles];
#pragma unroll
  for (int j = 0; j < kTokTiles; ++j) {
    acc[j] = v8f{0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F};
  }
  const int blocks = static_cast<int>(k / 32);
  uint4 next = load(0);
  for (int kb = 0; kb < blocks; ++kb) {
    __half* stage = s_act[kb & 1];
    if (tid < kLoads) {
      *reinterpret_cast<uint4*>(stage + a_tok * kActStride + a_quarter * 8) =
          next;
    }
    __syncthreads();
    if (kb + 1 < blocks) {
      next = load(kb + 1);
    }
    v16h a_lo;
    v16h a_hi;
    DecodeQ6K(w_row, kb, &a_lo, &a_hi);
#pragma unroll
    for (int j = 0; j < kTokTiles; ++j) {
      const __half* b = stage + (j * 16 + sub) * kActStride;
      v16h b_lo;
      v16h b_hi;
      __builtin_memcpy(&b_lo, b, 32);
      __builtin_memcpy(&b_hi, b + 16, 32);
      acc[j] = Wmma(a_lo, b_lo, acc[j]);
      acc[j] = Wmma(a_hi, b_hi, acc[j]);
    }
  }
#pragma unroll
  for (int j = 0; j < kTokTiles; ++j) {
    const int t = t_local + j * 16 + sub;
    if (t >= bucket_rows) {
      continue;
    }
    const std::int32_t dst = rows_out[bucket_begin + t];
    if (dst < 0) {
      continue;
    }
    const std::size_t base = std::size_t{static_cast<std::uint32_t>(dst)} * m;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const std::uint32_t r =
          blockIdx.x * kRowsPerBlock +
          static_cast<std::uint32_t>(wave * 16 + 2 * i + half);
      if (r < m) {
        if (out_half != nullptr) {
          out_half[base + r] =
              __float2half(fminf(fmaxf(acc[j][i], -65504.0F), 65504.0F));
        } else {
          out[base + r] = acc[j][i];
        }
      }
    }
  }
}

}  // namespace

bool LaunchRoutedHalfGemm(ExpertFormat format, const void* w, const void* x,
                          const std::int32_t* tiles, std::uint32_t n_tiles,
                          std::uint32_t tile_rows,
                          const std::int32_t* pad_bounds,
                          const std::int32_t* rows_in,
                          const std::int32_t* rows_out, float* out,
                          void* out_half, std::uint32_t m, std::uint32_t k,
                          hipStream_t stream) {
  if (format != ExpertFormat::kQ6_K || tile_rows != 48 || k % 256 != 0 ||
      n_tiles == 0 || (out == nullptr) == (out_half == nullptr)) {
    return false;
  }
  RoutedHalfQ6KKernel<48>
      <<<dim3((m + kRowsPerBlock - 1) / kRowsPerBlock, n_tiles), kThreads, 0,
         stream>>>(static_cast<const std::uint8_t*>(w),
                   static_cast<const __half*>(x), tiles, pad_bounds, rows_in,
                   rows_out, out, static_cast<__half*>(out_half), m, k);
  return true;
}

}  // namespace gufo::models::gemma4::rocm
