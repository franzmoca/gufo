#ifndef GUFO_MODELS_GEMMA4_KERNELS_ROCM_GEMV_HPP_
#define GUFO_MODELS_GEMMA4_KERNELS_ROCM_GEMV_HPP_

#include <hip/hip_runtime.h>

#include <cstdint>

namespace gufo::models::gemma4::rocm {

enum class GemvFormat : std::uint8_t { kQ4_K, kQ5_K, kQ6_K };

/// y[m] = W x for K-quant W ([m][k] in GGUF blocks) and an FP32 activation
/// row: the autoregressive decode projection. Returns false without
/// launching when the shape is unsupported.
[[nodiscard]] bool LaunchKQuantGemv(GemvFormat format, const void* w,
                                    const float* x, float* y, std::uint32_t m,
                                    std::uint32_t k, hipStream_t stream);

}  // namespace gufo::models::gemma4::rocm

#endif  // GUFO_MODELS_GEMMA4_KERNELS_ROCM_GEMV_HPP_
