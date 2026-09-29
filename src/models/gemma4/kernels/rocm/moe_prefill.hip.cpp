// Routed expert GEMM for prefill over binary16 activations for the K-quant
// expert formats (Q4_K, Q5_K, Q6_K). It reads the routing layout of
// Flash-Next's routed GEMM: RoutedCompact's 16-padded expert buckets and a
// tile map of expert | tile << 16 entries.
//
// A block computes 128 output rows (eight waves of 16) for one tile of
// kTileTokens bucket rows, skipping the tile's 16-row parts past the bucket.
// A stage is half a super-block (128 K values, four 32-value K blocks): two
// threads per row fetch its code bytes and scale header one stage ahead in
// registers, neighbouring threads on neighbouring bytes, and commit them with
// the tile's activations to LDS, the scales already as binary16. Each lane
// then decodes its own row's K blocks straight into binary16 WMMA fragments.
// Fragment layout (wave32 v_wmma_f32_16x16x16_f16): A holds row L%16 and B
// column L%16, each with 16 K values in both half-waves; C lane L holds
// column L%16, rows 2 i + L/16.
//
// Q4_K and Q5_K weights are q * (d sc) - dmin m, evaluated as one binary16
// FMA per pair with the two factors rounded to binary16, as in Flash-Next's
// routed GEMM, so the two produce the same products; Q6_K weights are
// (q - 32) * (d sc).
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
/// K values per stage: half a super-block.
constexpr int kStageK = 128;

/// Bytes per staged row: 64 low-bit bytes, 32 high-bit bytes (Q5_K, Q6_K)
/// and 16 bytes of binary16 scales. 80 and 112 bytes (20 and 28 dwords) put
/// the 16 rows a fragment read touches on distinct bank groups.
template<ExpertFormat F>
constexpr int kRowStride = F == ExpertFormat::kQ4_K ? 80 : 112;

template<ExpertFormat F>
constexpr int kBlockBytes = F == ExpertFormat::kQ4_K   ? 144
                            : F == ExpertFormat::kQ5_K ? 176
                                                       : 210;
/// Halves per staged token row: 128 values plus padding (68 dwords) against
/// bank conflicts on the fragment reads.
constexpr int kActStride = kStageK + 8;

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

/// Four byte codes q (< 1024) as two half2 of q * scale + bias.
__device__ __forceinline__ void CodesToHalvesAffine(std::uint32_t codes,
                                                    __half2 scale, __half2 bias,
                                                    __half2* out) {
  constexpr std::uint32_t kMagic = 0x64646464U;
  const __half2 magic = __float2half2_rn(-1024.0F);
  const std::uint32_t p0 = __builtin_amdgcn_perm(codes, kMagic, 0x01050004U);
  const std::uint32_t p1 = __builtin_amdgcn_perm(codes, kMagic, 0x03070206U);
  out[0] =
      __hfma2(__hadd2(__builtin_bit_cast(__half2, p0), magic), scale, bias);
  out[1] =
      __hfma2(__hadd2(__builtin_bit_cast(__half2, p1), magic), scale, bias);
}

/// Q4_K / Q5_K K block s (< 4) of a staged half super-block n as two
/// fragments: low (s even) or high nibbles of qs[32 (s / 2) + l], plus bit
/// 4 n + s of qh[l] as 16 (Q5_K), under sub-block s's (scale, bias).
template<ExpertFormat F>
__device__ __forceinline__ void DecodeQ45K(const uint4 (&qs)[4],
                                           const uint4 (&qh)[2],
                                           const __half2 (&scale)[4], int n,
                                           int s, v16h* lo, v16h* hi) {
  const int shift = 4 * (s & 1);
  const int bit = 4 * n + s;
  const __half2 sc = __low2half2(scale[s]);
  const __half2 bias = __high2half2(scale[s]);
#pragma unroll
  for (int part = 0; part < 2; ++part) {
    const uint4 q4 = qs[2 * (s >> 1) + part];
    const uint4 h4 = qh[part];
    const std::uint32_t qw[4] = {q4.x, q4.y, q4.z, q4.w};
    const std::uint32_t hw[4] = {h4.x, h4.y, h4.z, h4.w};
    __half2 h[8];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      std::uint32_t codes = (qw[i] >> shift) & 0x0F0F0F0FU;
      if constexpr (F == ExpertFormat::kQ5_K) {
        codes |= ((hw[i] >> bit) & 0x01010101U) << 4U;
      }
      CodesToHalvesAffine(codes, sc, bias, &h[2 * i]);
    }
    if (part == 0) {
      __builtin_memcpy(lo, h, 32);
    } else {
      __builtin_memcpy(hi, h, 32);
    }
  }
}

/// Q4_K / Q5_K scale and min of sub-block sb (< 8) from the 12 packed
/// 6-bit bytes that follow d and dmin in the header.
__device__ __forceinline__ void ScaleMinK4(const uint4& header, int sb,
                                           std::uint32_t* sc,
                                           std::uint32_t* mn) {
  const std::uint32_t words[4] = {header.x, header.y, header.z, header.w};
  const auto byte = [&](int i) {
    return (words[i / 4] >> (8 * (i % 4))) & 0xFFU;
  };
  // scales[i] is header byte 4 + i.
  if (sb < 4) {
    *sc = byte(4 + sb) & 0x3FU;
    *mn = byte(8 + sb) & 0x3FU;
  } else {
    *sc = (byte(8 + sb) & 0x0FU) | ((byte(sb) >> 6U) << 4U);
    *mn = (byte(8 + sb) >> 4U) | ((byte(4 + sb) >> 6U) << 4U);
  }
}

/// Q6_K K block s (< 4) of a staged half super-block as two fragments: low
/// (s < 2) or high nibbles of ql[32 (s % 2) + l], bits 2 s of qh[l], scales
/// 2 s (l < 16) and 2 s + 1, offset 32.
__device__ __forceinline__ void DecodeQ6K(const uint4 (&ql)[4],
                                          const uint4 (&qh)[2],
                                          const __half2 (&scale)[4], int s,
                                          v16h* lo, v16h* hi) {
  const int nibble = (s >> 1) * 4;
  const int high = 2 * s;
  const __half2 offset = __float2half2_rn(1056.0F);
#pragma unroll
  for (int part = 0; part < 2; ++part) {
    const uint4 l4 = ql[2 * (s & 1) + part];
    const uint4 h4 = qh[part];
    const std::uint32_t lw[4] = {l4.x, l4.y, l4.z, l4.w};
    const std::uint32_t hw[4] = {h4.x, h4.y, h4.z, h4.w};
    const __half2 sc =
        part == 0 ? __low2half2(scale[s]) : __high2half2(scale[s]);
    __half2 h[8];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const std::uint32_t codes = ((lw[i] >> nibble) & 0x0F0F0F0FU) |
                                  (((hw[i] >> high) & 0x03030303U) << 4U);
      CodesToHalves(codes, offset, sc, &h[2 * i]);
    }
    if (part == 0) {
      __builtin_memcpy(lo, h, 32);
    } else {
      __builtin_memcpy(hi, h, 32);
    }
  }
}

template<ExpertFormat F, int kTileTokens>
__global__ void __launch_bounds__(kThreads)
    RoutedHalfKQuantKernel(const std::uint8_t* __restrict__ w,
                           const __half* __restrict__ x,
                           const std::int32_t* __restrict__ tiles,
                           const std::int32_t* __restrict__ pad_bounds,
                           const std::int32_t* __restrict__ rows_in,
                           const std::int32_t* __restrict__ rows_out,
                           float* __restrict__ out,
                           __half* __restrict__ out_half, std::uint32_t m,
                           std::uint32_t k) {
  constexpr bool kQ6 = F == ExpertFormat::kQ6_K;
  constexpr bool kQ4 = F == ExpertFormat::kQ4_K;
  constexpr int kStride = kRowStride<F>;
  constexpr int kScaleChunk = kStride / 16 - 1;
  constexpr int kTokTiles = kTileTokens / 16;
  // uint4 activation chunks per stage (16 per token) and per thread.
  constexpr int kActChunks = kTileTokens * kStageK / 8;
  constexpr int kActPer = (kActChunks + kThreads - 1) / kThreads;
  __shared__
      __attribute__((aligned(16))) std::uint8_t s_rows[kRowsPerBlock * kStride];
  __shared__
      __attribute__((aligned(16))) __half s_act[kTileTokens * kActStride];
  const std::int32_t tile = tiles[blockIdx.y];
  const int expert = tile & 0xFFFF;
  const int t_local = (tile >> 16) * kTileTokens;
  const int bucket_begin = pad_bounds[expert];
  const int bucket_rows = pad_bounds[expert + 1] - bucket_begin;
  if (t_local >= bucket_rows) {
    return;
  }
  const int live_tiles = min(kTokTiles, (bucket_rows - t_local + 15) / 16);
  const int tid = static_cast<int>(threadIdx.x);
  const int lane = tid % kWave;
  const int wave = tid / kWave;
  const int sub = lane & 15;
  const int half = lane >> 4;
  const std::size_t row_bytes = std::size_t{k} / 256 * kBlockBytes<F>;

  // Weight fetch: thread tid moves row tid / 2 of the block, half its code
  // chunks each (part 0 first), and part 1 commits the scales as binary16.
  // Q6_K: ql 0-47 | ql 48-63, qh 0-31; Q5_K: qs 0-47 | qs 48-63, qh 0-31;
  // Q4_K: qs 0-31 | qs 32-63.
  const int f_row = tid >> 1;
  const int f_part = tid & 1;
  const std::uint32_t f_global = blockIdx.x * kRowsPerBlock + f_row;
  const bool f_live = f_global < m;
  const std::uint8_t* f_ptr =
      w + (std::size_t{static_cast<std::uint32_t>(expert)} * m +
           (f_live ? f_global : m - 1)) *
              row_bytes;
  uint4 f_code0;
  uint4 f_code1;
  uint4 f_code2;
  uint4 f_header;  // Q4_K / Q5_K: d, dmin and the packed scales
  uint2 f_sc;      // Q6_K: the half's eight scales
  float f_d = 0.0F;
  int f_half = 0;
  const auto fetch_weights = [&](int stage) {
    const std::uint8_t* block = f_ptr + (stage / 2) * kBlockBytes<F>;
    const int n = stage % 2;
    f_half = n;
    if constexpr (kQ6) {
      // The qh bytes at 128 + 32 n follow ql 48-63 after 64 - 32 n bytes.
      const std::uint8_t* first = block + 64 * n + 48 * f_part;
      const int gap = f_part * (64 - 32 * n);
      f_code0 = *reinterpret_cast<const uint4*>(first);
      f_code1 = *reinterpret_cast<const uint4*>(first + 16 + gap);
      f_code2 = *reinterpret_cast<const uint4*>(first + 32 + gap);
      f_sc = *reinterpret_cast<const uint2*>(block + 192 + 8 * n);
      f_d = __half2float(*reinterpret_cast<const __half*>(block + 208));
    } else if constexpr (kQ4) {
      const std::uint8_t* first = block + 16 + 64 * n + 32 * f_part;
      f_code0 = *reinterpret_cast<const uint4*>(first);
      f_code1 = *reinterpret_cast<const uint4*>(first + 16);
      f_header = *reinterpret_cast<const uint4*>(block);
    } else {
      // Q5_K: header, qh[32], qs[128].
      const std::uint8_t* first = block + 48 + 64 * n + 48 * f_part;
      f_code0 = *reinterpret_cast<const uint4*>(first);
      f_code1 = *reinterpret_cast<const uint4*>(f_part != 0 ? block + 16
                                                            : first + 16);
      f_code2 = *reinterpret_cast<const uint4*>(f_part != 0 ? block + 32
                                                            : first + 32);
      f_header = *reinterpret_cast<const uint4*>(block);
    }
  };
  const auto commit_weights = [&] {
    auto* dst = reinterpret_cast<uint4*>(s_rows + f_row * kStride);
    if constexpr (kQ4) {
      dst[2 * f_part] = f_code0;
      dst[2 * f_part + 1] = f_code1;
    } else {
      dst[3 * f_part] = f_code0;
      dst[3 * f_part + 1] = f_code1;
      dst[3 * f_part + 2] = f_code2;
    }
    if (f_part == 1) {
      std::uint32_t packed[4];
      if constexpr (kQ6) {
        const float d = f_live ? f_d : 0.0F;
        const std::uint32_t sw[2] = {f_sc.x, f_sc.y};
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const auto lo = static_cast<std::int8_t>(sw[i / 2] >> (16 * (i % 2)));
          const auto hi =
              static_cast<std::int8_t>(sw[i / 2] >> (16 * (i % 2) + 8));
          // d * sc in FP32, then binary16.
          packed[i] = __builtin_bit_cast(
              std::uint32_t,
              __halves2half2(__float2half_rn(d * static_cast<float>(lo)),
                             __float2half_rn(d * static_cast<float>(hi))));
        }
      } else {
        const __half2 dm = __builtin_bit_cast(__half2, f_header.x);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          std::uint32_t sc = 0;
          std::uint32_t mn = 0;
          ScaleMinK4(f_header, 4 * f_half + i, &sc, &mn);
          const float scale =
              f_live ? __low2float(dm) * static_cast<float>(sc) : 0.0F;
          const float offset =
              f_live ? __high2float(dm) * static_cast<float>(mn) : 0.0F;
          packed[i] = __builtin_bit_cast(std::uint32_t,
                                         __floats2half2_rn(scale, -offset));
        }
      }
      dst[kScaleChunk] = make_uint4(packed[0], packed[1], packed[2], packed[3]);
    }
  };

  // Activation fetch: chunk c = tid + 256 i is token c / 16, eight values at
  // K offset 8 (c % 16) of the stage; a_src is that chunk's offset in x at
  // stage 0, or -1 for a padding row.
  std::int32_t a_src[kActPer];
#pragma unroll
  for (int i = 0; i < kActPer; ++i) {
    const int chunk = tid + i * kThreads;
    const int t = chunk / 16;
    a_src[i] = -1;
    if (chunk < kActChunks && t_local + t < bucket_rows) {
      const std::int32_t src = rows_in[bucket_begin + t_local + t];
      if (src >= 0) {
        a_src[i] = src * static_cast<std::int32_t>(k) + (chunk % 16) * 8;
      }
    }
  }
  uint4 a_data[kActPer];
  const auto fetch_act = [&](int stage) {
#pragma unroll
    for (int i = 0; i < kActPer; ++i) {
      a_data[i] =
          a_src[i] >= 0
              ? *reinterpret_cast<const uint4*>(x + a_src[i] + stage * kStageK)
              : make_uint4(0U, 0U, 0U, 0U);
    }
  };
  const auto commit_act = [&] {
#pragma unroll
    for (int i = 0; i < kActPer; ++i) {
      const int chunk = tid + i * kThreads;
      if (chunk < kActChunks) {
        *reinterpret_cast<uint4*>(s_act + (chunk / 16) * kActStride +
                                  (chunk % 16) * 8) = a_data[i];
      }
    }
  };

  v8f acc[kTokTiles];
#pragma unroll
  for (int j = 0; j < kTokTiles; ++j) {
    acc[j] = v8f{0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F};
  }
  const int stages = static_cast<int>(k / kStageK);
  fetch_weights(0);
  fetch_act(0);
  for (int stage = 0; stage < stages; ++stage) {
    commit_weights();
    commit_act();
    __syncthreads();
    if (stage + 1 < stages) {
      fetch_weights(stage + 1);
      fetch_act(stage + 1);
    }
    const auto* row =
        reinterpret_cast<const uint4*>(s_rows + (wave * 16 + sub) * kStride);
    const uint4 q[4] = {row[0], row[1], row[2], row[3]};
    uint4 qh[2] = {};
    if constexpr (!kQ4) {
      qh[0] = row[4];
      qh[1] = row[5];
    }
    const uint4 sc4 = row[kScaleChunk];
    const __half2 scale[4] = {
        __builtin_bit_cast(__half2, sc4.x), __builtin_bit_cast(__half2, sc4.y),
        __builtin_bit_cast(__half2, sc4.z), __builtin_bit_cast(__half2, sc4.w)};
    const int n = stage % 2;
#pragma unroll
    for (int s = 0; s < 4; ++s) {
      v16h a_lo;
      v16h a_hi;
      if constexpr (kQ6) {
        DecodeQ6K(q, qh, scale, s, &a_lo, &a_hi);
      } else {
        DecodeQ45K<F>(q, qh, scale, n, s, &a_lo, &a_hi);
      }
#pragma unroll
      for (int j = 0; j < kTokTiles; ++j) {
        if (j < live_tiles) {
          const __half* b = s_act + (j * 16 + sub) * kActStride + s * 32;
          v16h b_lo;
          v16h b_hi;
          __builtin_memcpy(&b_lo, b, 32);
          __builtin_memcpy(&b_hi, b + 16, 32);
          acc[j] = Wmma(a_lo, b_lo, acc[j]);
          acc[j] = Wmma(a_hi, b_hi, acc[j]);
        }
      }
    }
    __syncthreads();
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
  if (tile_rows != 96 || k % 256 != 0 || n_tiles == 0 ||
      (out == nullptr) == (out_half == nullptr)) {
    return false;
  }
  const dim3 grid((m + kRowsPerBlock - 1) / kRowsPerBlock, n_tiles);
  const auto* wb = static_cast<const std::uint8_t*>(w);
  const auto* xh = static_cast<const __half*>(x);
  auto* oh = static_cast<__half*>(out_half);
  switch (format) {
    case ExpertFormat::kQ4_K:
      RoutedHalfKQuantKernel<ExpertFormat::kQ4_K, 96>
          <<<grid, kThreads, 0, stream>>>(wb, xh, tiles, pad_bounds, rows_in,
                                          rows_out, out, oh, m, k);
      return true;
    case ExpertFormat::kQ5_K:
      RoutedHalfKQuantKernel<ExpertFormat::kQ5_K, 96>
          <<<grid, kThreads, 0, stream>>>(wb, xh, tiles, pad_bounds, rows_in,
                                          rows_out, out, oh, m, k);
      return true;
    case ExpertFormat::kQ6_K:
      RoutedHalfKQuantKernel<ExpertFormat::kQ6_K, 96>
          <<<grid, kThreads, 0, stream>>>(wb, xh, tiles, pad_bounds, rows_in,
                                          rows_out, out, oh, m, k);
      return true;
    default:
      return false;
  }
}

}  // namespace gufo::models::gemma4::rocm
