#ifndef GUFO_MODELS_GEMMA4_KERNELS_ROCM_W4A4_HPP_
#define GUFO_MODELS_GEMMA4_KERNELS_ROCM_W4A4_HPP_

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

namespace gufo::models::gemma4::rocm {

/// Int4 prefill projections (the --prefill-sidecar copy; layout in
/// docs/models/gemma-4-31b/WEIGHT-FORMATS.md): inputs rotate as
/// x' = H(vec * x) per 128 values (H the orthonormal Walsh-Hadamard
/// transform) and quantize per token and 128 values to int4; weights are
/// int4 codes [k/128][m][64 bytes] with binary16 scales [k/128][m], in the
/// same rotated basis. Products run on v_wmma_i32_16x16x16_iu4.

/// Token rows of one GEMM tile: activations are staged for `rows` rounded
/// up to it.
inline constexpr std::uint32_t kW4A4TileRows = 128;

[[nodiscard]] constexpr std::uint32_t W4A4PaddedRows(
    std::uint32_t rows) noexcept {
  return (rows + kW4A4TileRows - 1) / kW4A4TileRows * kW4A4TileRows;
}

/// Bytes of the quantized activations of `rows` rows of width `k`: codes
/// [k/128][padded rows][64] then FP32 scales [k/128][padded rows].
[[nodiscard]] std::size_t W4A4ActivationBytes(std::uint32_t rows,
                                              std::uint32_t k) noexcept;

/// Rotates and quantizes binary16 rows x [rows][k] (k % 128 == 0) into
/// `xq` (W4A4ActivationBytes): per 128 rotated values the scale
/// amax * r / 7 with the least squared error over r in {1, 0.9, 0.8, 0.7},
/// codes round(x' / scale) clamped to [-8, 7]. Padding rows get zero codes
/// and scales.
void LaunchW4A4Quantize(const void* x, const float* vec, void* xq,
                        std::uint32_t rows, std::uint32_t k,
                        hipStream_t stream);

/// y[r][m] = sum over 128-blocks b of ws[b][m] * xs[b][r] * (codes dot)
/// for the quantized rows of LaunchW4A4Quantize; y rows are `m` FP32 wide.
/// Returns false without launching when m or k is not a multiple of 128.
[[nodiscard]] bool LaunchW4A4Gemm(const void* wq, const void* ws,
                                  const void* xq, float* y, std::uint32_t rows,
                                  std::uint32_t m, std::uint32_t k,
                                  hipStream_t stream);

}  // namespace gufo::models::gemma4::rocm

#endif  // GUFO_MODELS_GEMMA4_KERNELS_ROCM_W4A4_HPP_
