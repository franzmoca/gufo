// SPDX-License-Identifier: MIT
// Raw-pointer C ABI for DeepSeek's HIP quantized matrix kernels.
// Functions return zero on success and nonzero on failure. Device pointers
// remain caller-owned; every operation uses the supplied HIP stream.

#pragma once

#include <hip/hip_runtime.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// One-time init. Sets the current HIP device and triggers lazy population
// of the device-info singleton. Safe to call repeatedly.
//
//   device: HIP device ordinal (0 for the primary GPU).
// Returns 0 on success.
int ds41_mmq_init(int device);
void ds41_mmq_cleanup(void);

// ds4: routed MoE column-tile bound.
//
// The routed path sizes its column-tile grid from ncols_max, and without a
// better bound that has to be the whole chunk width: any one expert could in
// principle receive every token. Every block outside an expert's real bucket
// then exits immediately at the top of mul_mat_q, but it still costs a
// dispatch and a full 31.5 KiB shared-memory reservation, which caps residency
// at two workgroups per CU. At a 4,096-token chunk over 256 experts the mean
// bucket is n_tokens * n_expert_used / n_experts = 96 rows, so about 96% of the
// grid is empty.
//
// The caller already reads the per-expert assignment counts back to the host to
// build the Q2 down hot list, so it can pass the real maximum here. rows <= 0
// restores the conservative bound. Purely a launch-geometry bound: the blocks
// it removes did no work, so results are unchanged.
void ds41_mmq_set_routed_max_expert_rows(int rows);

// Column-tile width for the next routed gate/up pass, when the caller knows the
// whole bucket distribution and not just its maximum.
//
// The default rule takes the widest tile that minimizes the tile count for the
// chunk width, which is the right answer for a dense column range and the wrong
// one for a routed one. A tile executes all its columns whether or not the
// bucket fills them, so the cost is `sum_e ceil(c_e / w) * w` column-slots plus
// one weight-panel reload per tile: at a 4,096-token chunk the mean bucket is 96
// rows and the 80-column tile is nearly full, while at 512 tokens it is twelve
// rows and about 85% of the matrix-core issue is padding.
//
// cols <= 0, or a width the device cannot host, restores the default rule.
// Purely a tiling choice: each output element still walks K once in the same
// order, so results are unchanged.
void ds41_mmq_set_routed_tile_cols(int cols);
// V4.1: a tile width per expert. expert_class (device, one byte per expert)
// assigns each expert one of `count` classes; routed launches then run once
// per class with widths[i] columns, over experts of class i only, bounded by
// max_rows[i] (classes with max_rows[i] == 0 are skipped). count 0 or a null
// pointer turns it off. The host arrays are copied. Tile widths change the
// matrix-core accumulation split, so outputs may differ in the last ulp.
void ds41_mmq_set_routed_classes(const uint8_t * expert_class, int count,
                                 const int * widths, const int * max_rows);
// Fills expert_class[n_experts] (host), widths[3] and max_rows[3] from the
// per-expert row counts; returns the class count (3), 0 on bad input.
int ds41_mmq_routed_classes_for_counts(const unsigned int * counts, int n_experts,
                                       uint8_t * expert_class, int * widths,
                                       int * max_rows);

// The width the model above picks for one per-expert assignment-count array.
// Returns 0 when the counts are unusable, which the setter treats as "default".
int ds41_mmq_routed_tile_cols_for_counts(const unsigned int *counts,
                                        int n_experts);

// V4.1 Flash: ds41_mmq_<type>_dense / _moe for a run-time weight type
// (Q8_0, Q5_K, IQ1_S, IQ1_M, IQ2_XXS or IQ2_S). Same layouts as below.
int ds41_mmq_dense_typed(int type, const void * W, const float * X_f32,
                         float * out_f32, int M, int N, int K,
                         hipStream_t stream);
int ds41_mmq_moe_typed(int type, const void * W, const float * X_f32,
                       const int32_t * ids, float * out_f32, int M, int K,
                       int n_tokens, int n_experts, int n_expert_used,
                       hipStream_t stream);
// Two [M1, K] and [M2, K] weights of one type over the same X ([N, K]),
// quantized once: out1 = X W1^T ([N, M1]), out2 = X W2^T ([N, M2]).
int ds41_mmq_dense_pair_typed(int type, const void * W1, float * out1, int M1,
                              const void * W2, float * out2, int M2,
                              const float * X_f32, int N, int K, hipStream_t stream);
// `groups` [M, K] weights stored back to back, each applied to its own K
// columns of every X row ([N, groups * K]); out is [N, groups * M]. One input
// quantization and one launch (an MMQ channel per group).
int ds41_mmq_dense_grouped_typed(int type, const void * W, const float * X_f32,
                                 float * out_f32, int M, int N, int K, int groups,
                                 hipStream_t stream);
// Routed gate and up of one weight type over the same rows (input quantized
// once): gate_f32 receives the gate rows, and the up pass stores
// mid = router_weights[row] * silu(min(gate, clamp)) * clamp(up, +-clamp)
// (no clamp when clamp <= 1e-6; non-finite gate/up read as 0).
int ds41_mmq_moe_swiglu_typed(int type, const void * W_gate, const void * W_up,
                              const float * X_f32, const int32_t * ids,
                              const float * router_weights, float * gate_f32,
                              float * mid_f32, int M, int K, int n_tokens,
                              int n_experts, int n_expert_used, float clamp,
                              hipStream_t stream);

// Set the thread-local stream that the internal cuda pool uses for
// hipMallocAsync / hipFreeAsync.  Defaults to hipStreamPerThread.
// Use the graph capture stream so pool
// allocations land on the captured stream and don't invalidate capture.
// Pass NULL to reset to hipStreamPerThread.
void ds41_pool_set_stream(hipStream_t stream);
/* Frees the buffers the MMQ pool keeps for reuse. */
void ds41_pool_release_cached(void);

// V4.1 Flash matvecs (mmvq, N <= 8) over a weight type chosen at run time
// (ggml type codes), with FP32 activations and outputs and K a multiple of
// 256. Dense: out [N, M] = X [N, K] @ W^T. MoE: ids [n_tokens, n_expert_used]
// select the experts of W, out [token * n_expert_used + slot, M].
int ds41_mmq_dense_vec_typed(int type, const void* W, const float* X_f32,
                             float* out_f32, int M, int N, int K,
                             hipStream_t stream);
// `groups` matvecs in one launch: X [N, groups, K], W the groups' [M, K]
// blocks back to back, out [N, groups, M]. N <= 8.
int ds41_mmq_grouped_vec_typed(int type, const void* W, const float* X_f32,
                               float* out_f32, int M, int N, int K, int groups,
                               hipStream_t stream);
int ds41_mmq_moe_vec_typed(int type, const void* W, const float* X_f32,
                           const int32_t* ids, float* out_f32, int M, int K,
                           int n_tokens, int n_experts, int n_expert_used,
                           hipStream_t stream);

#ifdef __cplusplus
} // extern "C"
#endif
