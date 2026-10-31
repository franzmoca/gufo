#ifndef GUFO_MODELS_DEEPSEEK_V41_FLASH_RUNTIME_TENSOR_TYPES_H_
#define GUFO_MODELS_DEEPSEEK_V41_FLASH_RUNTIME_TENSOR_TYPES_H_

#include <cstdint>

inline constexpr uint32_t DS41_TENSOR_F32 = 0;
inline constexpr uint32_t DS41_TENSOR_F16 = 1;
inline constexpr uint32_t DS41_TENSOR_Q8_0 = 8;
inline constexpr uint32_t DS41_TENSOR_Q2_K = 10;
inline constexpr uint32_t DS41_TENSOR_Q4_K = 12;
inline constexpr uint32_t DS41_TENSOR_Q5_K = 13;
inline constexpr uint32_t DS41_TENSOR_Q6_K = 14;
inline constexpr uint32_t DS41_TENSOR_IQ2_XXS = 16;
inline constexpr uint32_t DS41_TENSOR_IQ2_XS = 17;
inline constexpr uint32_t DS41_TENSOR_IQ3_XXS = 18;
inline constexpr uint32_t DS41_TENSOR_IQ1_S = 19;
inline constexpr uint32_t DS41_TENSOR_IQ4_NL = 20;
inline constexpr uint32_t DS41_TENSOR_IQ3_S = 21;
inline constexpr uint32_t DS41_TENSOR_IQ2_S = 22;
inline constexpr uint32_t DS41_TENSOR_I32 = 26;
inline constexpr uint32_t DS41_TENSOR_I64 = 27;
inline constexpr uint32_t DS41_TENSOR_IQ1_M = 29;
inline constexpr uint32_t DS41_TENSOR_BF16 = 30;
inline constexpr uint32_t DS41_TENSOR_MXFP4 = 39;
/* unsloth's MXFP8 Engram tables: an E8M0 exponent, then 32 E4M3 values.
 * Upstream ggml gives type 43 to unscaled F8_E4M3; the table's byte size
 * tells the two apart. */
inline constexpr uint32_t DS41_TENSOR_MXFP8 = 43;

#endif  // GUFO_MODELS_DEEPSEEK_V41_FLASH_RUNTIME_TENSOR_TYPES_H_
