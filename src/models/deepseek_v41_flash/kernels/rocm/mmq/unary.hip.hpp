#pragma once
#include "common.hip.hpp"

#define CUDA_NEG_BLOCK_SIZE 256
#define CUDA_STEP_BLOCK_SIZE 256
#define CUDA_GELU_BLOCK_SIZE 256
#define CUDA_SILU_BLOCK_SIZE 256
#define CUDA_SILU_BACK_BLOCK_SIZE 256
#define CUDA_TANH_BLOCK_SIZE 256
#define CUDA_RELU_BLOCK_SIZE 256
#define CUDA_SIGMOID_BLOCK_SIZE 256
#define CUDA_HARDSIGMOID_BLOCK_SIZE 256
#define CUDA_EXP_BLOCK_SIZE 256
#define CUDA_HARDSWISH_BLOCK_SIZE 256
#define CUDA_SQR_BLOCK_SIZE 256
#define CUDA_SQRT_BLOCK_SIZE 256
#define CUDA_SIN_BLOCK_SIZE 256
#define CUDA_COS_BLOCK_SIZE 256
#define CUDA_GLU_BLOCK_SIZE 256
#define CUDA_XIELU_BLOCK_SIZE 256

void ds41_ggml_hip_op_abs(ds41_ggml_hip_context & ctx, ggml_tensor * dst);

void ds41_ggml_hip_op_sgn(ds41_ggml_hip_context & ctx, ggml_tensor * dst);

void ds41_ggml_hip_op_neg(ds41_ggml_hip_context & ctx, ggml_tensor * dst);

void ds41_ggml_hip_op_step(ds41_ggml_hip_context & ctx, ggml_tensor * dst);

void ds41_ggml_hip_op_gelu(ds41_ggml_hip_context & ctx, ggml_tensor * dst);

void ds41_ggml_hip_op_silu(ds41_ggml_hip_context & ctx, ggml_tensor * dst);

void ds41_ggml_hip_op_silu_back(ds41_ggml_hip_context & ctx, ggml_tensor * dst);

void ds41_ggml_hip_op_gelu_erf(ds41_ggml_hip_context & ctx, ggml_tensor * dst);

void ds41_ggml_hip_op_gelu_quick(ds41_ggml_hip_context & ctx, ggml_tensor * dst);

void ds41_ggml_hip_op_tanh(ds41_ggml_hip_context & ctx, ggml_tensor * dst);

void ds41_ggml_hip_op_relu(ds41_ggml_hip_context & ctx, ggml_tensor * dst);

void ds41_ggml_hip_op_sigmoid(ds41_ggml_hip_context & ctx, ggml_tensor * dst);

void ds41_ggml_hip_op_hardsigmoid(ds41_ggml_hip_context & ctx, ggml_tensor * dst);

void ds41_ggml_hip_op_exp(ds41_ggml_hip_context & ctx, ggml_tensor * dst);

void ds41_ggml_hip_op_hardswish(ds41_ggml_hip_context & ctx, ggml_tensor * dst);

void ds41_ggml_hip_op_leaky_relu(ds41_ggml_hip_context & ctx, ggml_tensor * dst);

void ds41_ggml_hip_op_sqr(ds41_ggml_hip_context & ctx, ggml_tensor * dst);

void ds41_ggml_hip_op_sqrt(ds41_ggml_hip_context & ctx, ggml_tensor * dst);

void ds41_ggml_hip_op_sin(ds41_ggml_hip_context & ctx, ggml_tensor * dst);

void ds41_ggml_hip_op_cos(ds41_ggml_hip_context & ctx, ggml_tensor * dst);

void ds41_ggml_hip_op_log(ds41_ggml_hip_context & ctx, ggml_tensor * dst);

void ds41_ggml_hip_op_expm1(ds41_ggml_hip_context & ctx, ggml_tensor * dst);

void ds41_ggml_hip_op_softplus(ds41_ggml_hip_context & ctx, ggml_tensor * dst);

void ds41_ggml_hip_op_elu(ds41_ggml_hip_context & ctx, ggml_tensor * dst);

void ds41_ggml_hip_op_floor(ds41_ggml_hip_context & ctx, ggml_tensor * dst);

void ds41_ggml_hip_op_ceil(ds41_ggml_hip_context & ctx, ggml_tensor * dst);

void ds41_ggml_hip_op_round(ds41_ggml_hip_context & ctx, ggml_tensor * dst);

void ds41_ggml_hip_op_trunc(ds41_ggml_hip_context & ctx, ggml_tensor * dst);

void ds41_ggml_hip_op_reglu(ds41_ggml_hip_context & ctx, ggml_tensor * dst);

void ds41_ggml_hip_op_geglu(ds41_ggml_hip_context & ctx, ggml_tensor * dst);

void ds41_ggml_hip_op_swiglu(ds41_ggml_hip_context & ctx, ggml_tensor * dst);

void ds41_ggml_hip_op_swiglu_oai(ds41_ggml_hip_context & ctx, ggml_tensor * dst);

void ds41_ggml_hip_op_geglu_erf(ds41_ggml_hip_context & ctx, ggml_tensor * dst);

void ds41_ggml_hip_op_geglu_quick(ds41_ggml_hip_context & ctx, ggml_tensor * dst);

void ds41_ggml_hip_op_xielu(ds41_ggml_hip_context & ctx, ggml_tensor * dst);

void ds41_ggml_hip_op_unary_mul(ds41_ggml_hip_context & ctx, ggml_tensor * unary_node, ggml_tensor * mul_node);

void ds41_ggml_hip_op_relu_sqr(ds41_ggml_hip_context & ctx, ggml_tensor * relu_node, ggml_tensor * sqr_node);

__device__ __forceinline__ float ds41_ggml_hip_op_silu_single(float x) {
    return x / (1.0f + expf(-x));
}

__device__ __forceinline__ float ds41_ggml_hip_op_gelu_single(float x) {
    const float GELU_COEF_A    = 0.044715f;
    const float SQRT_2_OVER_PI = 0.79788456080286535587989211986876f;

    return 0.5f * x * (1.0f + tanhf(SQRT_2_OVER_PI * x * (1.0f + GELU_COEF_A * x * x)));
}

__device__ __forceinline__ float ds41_ggml_hip_op_swiglu_oai_single(float x, float g, float alpha = 1.702f, float limit = 7.0f) {
    x = fminf(x, limit);
    g = fmaxf(fminf(g, limit), -limit);

    float out_glu = x / (1.0f + expf(-x * alpha));
    out_glu = out_glu * (1.0f + g);
    return out_glu;
}
