#pragma once

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include "src/models/deepseek_v41_flash/runtime/model.h"

// These kernels return only eight corrected logits per request. No additional
// full-vocabulary logits/probability buffer is needed.
__device__ static float dspark_q8_element(const unsigned char* weights,
                                          uint32_t i) {
  const auto* block = weights + (uint64_t)(i / 32) * 34;
  return __half2float(*reinterpret_cast<const __half*>(block)) *
         reinterpret_cast<const int8_t*>(block + 2)[i % 32];
}

__device__ static uint64_t dspark_candidate_key(float value, uint32_t id) {
  if (!isfinite(value))
    return 0;
  uint32_t bits = __float_as_uint(value);
  if ((bits & 0x7fffffffu) == 0)
    bits = 0;  // Stable signed-zero tie-break.
  const uint32_t ordered = (bits & 0x80000000u) ? ~bits : bits | 0x80000000u;
  return ((uint64_t)ordered << 32) | (uint32_t)~id;
}

__device__ static uint64_t dspark_max_key(uint64_t key, uint64_t* scratch) {
  const auto tid = threadIdx.x;
  scratch[tid] = key;
  __syncthreads();
  for (uint32_t stride = 128; stride; stride >>= 1) {
    if (tid < stride)
      scratch[tid] = max(scratch[tid], scratch[tid + stride]);
    __syncthreads();
  }
  const auto best = scratch[0];
  __syncthreads();
  return best;
}

__global__ static void dspark_markov_candidates_kernel(
    uint64_t* partial, const float* logits, uint64_t logits_stride,
    const int32_t* previous, const unsigned char* w1, const unsigned char* w2,
    uint32_t vocab, uint32_t rank) {
  const uint32_t request = blockIdx.y, tid = threadIdx.x;
  const uint32_t id = blockIdx.x * 256 + tid;
  const uint32_t rank_blocks = rank / 32;
  __shared__ float state[256];
  __shared__ uint64_t scratch[256];
  if (tid < rank)
    state[tid] = dspark_q8_element(
        w1 + (uint64_t)previous[request] * rank_blocks * 34, tid);
  __syncthreads();
  uint64_t key = 0;
  if (id < vocab) {
    const auto* weights = w2 + (uint64_t)id * rank_blocks * 34;
    float acc = 0;
    // Keep the qualified greedy kernel's accumulation order.
    for (uint32_t block = 0; block < rank_blocks; ++block) {
      const auto* qblock = weights + (uint64_t)block * 34;
      const float scale = __half2float(*(const __half*)qblock);
      const auto* quants = reinterpret_cast<const int8_t*>(qblock + 2);
      float sum = 0;
#pragma unroll
      for (uint32_t lane = 0; lane < 32; ++lane)
        sum += (float)quants[lane] * state[block * 32 + lane];
      acc += scale * sum;
    }
    key = dspark_candidate_key(logits[request * logits_stride + id] + acc, id);
  }
  const uint64_t base =
      ((uint64_t)request * gridDim.x + blockIdx.x) * DS41_DSPARK_CANDIDATES;
  for (uint32_t k = 0; k < DS41_DSPARK_CANDIDATES; ++k) {
    const auto best = dspark_max_key(key, scratch);
    if (tid == 0)
      partial[base + k] = best;
    if (key == best)
      key = 0;
  }
}

__global__ static void dspark_merge_candidates_kernel(
    ds41_dspark_candidates* out, const uint64_t* partial, uint32_t blocks,
    const float* hidden, uint64_t hidden_stride, uint32_t hidden_width,
    const int32_t* previous, const unsigned char* w1,
    const unsigned char* confidence, uint32_t rank) {
  const uint32_t request = blockIdx.x, tid = threadIdx.x;
  __shared__ uint64_t scratch[256];
  __shared__ float sums[256];
  float sum = 0;
  const auto* embedding = w1 + (uint64_t)previous[request] * (rank / 32) * 34;
  for (uint32_t i = tid; i < hidden_width + rank; i += 256) {
    const float value = i < hidden_width
                            ? hidden[(uint64_t)request * hidden_stride + i]
                            : dspark_q8_element(embedding, i - hidden_width);
    sum += value * dspark_q8_element(confidence, i);
  }
  sums[tid] = sum;
  __syncthreads();
  for (uint32_t stride = 128; stride; stride >>= 1) {
    if (tid < stride)
      sums[tid] += sums[tid + stride];
    __syncthreads();
  }
  // DS4 uses the pre-norm hidden state, with no projection bias.
  if (tid == 0)
    out[request].confidence = 1.F / (1.F + expf(-sums[0]));
  const auto* keys =
      partial + (uint64_t)request * blocks * DS41_DSPARK_CANDIDATES;
  uint64_t ceiling = UINT64_MAX;
  for (uint32_t k = 0; k < DS41_DSPARK_CANDIDATES; ++k) {
    uint64_t key = 0;
    for (uint32_t i = tid; i < blocks * DS41_DSPARK_CANDIDATES; i += 256)
      if (keys[i] < ceiling)
        key = max(key, keys[i]);
    const auto best = dspark_max_key(key, scratch);
    if (tid == 0) {
      const uint32_t ordered = best >> 32;
      const uint32_t bits =
          (ordered & 0x80000000u) ? ordered & 0x7fffffffu : ~ordered;
      out[request].ids[k] = best ? (int32_t)~(uint32_t)best : -1;
      out[request].logits[k] = best ? __uint_as_float(bits) : -INFINITY;
    }
    ceiling = best;
  }
}
