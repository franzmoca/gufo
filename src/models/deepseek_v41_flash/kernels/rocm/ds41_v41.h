/* DeepSeek V4.1 Flash single-token decode kernels.
 *
 * Each entry mirrors one operation of the official inference/model.py (and
 * the tools/ds41 oracle) on fp32 device buffers. All launches use the legacy
 * stream, like the rest of the backend. Functions return 0 on success. */
#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* out[r] = sum_k W[r, k] * x[k]; W is row-major BF16 or F32. */
int ds41_v41_matvec_bf16(const uint16_t* w, const float* x, float* out, uint32_t rows, uint32_t k);
int ds41_v41_matvec_f32(const float* w, const float* x, float* out, uint32_t rows, uint32_t k);

/* out[r] = weight * x[r] * rsqrt(mean(x[r]^2) + eps) for `rows` rows of n; weight may be NULL. */
int ds41_v41_rms_norm(const float* x, const float* weight, float* out, uint32_t rows, uint32_t n, float eps);

/* Rotate adjacent pairs of the last `rope_dim` values of each of `rows` rows
 * of `row_dim` at position `pos`; inv_freq has rope_dim/2 entries. */
int ds41_v41_rope_tail(float* x, uint32_t rows, uint32_t row_dim, uint32_t rope_dim, const float* inv_freq,
                       uint32_t pos, int inverse);

/* In-place fake quantization (quantize then dequantize).
 * FP8 E4M3 with a power-of-two scale per 32 values (act_quant, ue8m0).
 * FP4 E2M1 per `block` values with an E4M3 scale (e4m3_scale) or a
 * power-of-two scale. */
int ds41_v41_fp8_fake_quant(float* x, uint32_t n);
int ds41_v41_fp4_fake_quant(float* x, uint32_t n, uint32_t block, int e4m3_scale);

/* mHC. mixes[24] = (fn . flatten(h)) * rsqrt(mean(h^2) + eps). */
int ds41_v41_hc_mixes(const float* fn, const float* h, float* mixes, uint32_t hc, uint32_t dim, float eps);
/* Split mixes into pre[hc], post[hc] and Sinkhorn-normalized comb[hc*hc]. */
int ds41_v41_hc_sinkhorn(const float* mixes, const float* scale, const float* base, float* pre, float* post,
                         float* comb, uint32_t hc, uint32_t iters, float eps);
/* out[i] = sum_c pre[c] * h[c, i]. */
int ds41_v41_hc_collapse(const float* h, const float* pre, float* out, uint32_t hc, uint32_t dim);
/* out[b, i] = post[b] * y[i] + sum_a comb[a, b] * residual[a, i]. */
int ds41_v41_hc_expand(const float* y, const float* residual, const float* post, const float* comb, float* out,
                       uint32_t hc, uint32_t dim);

/* Ratio-r compressor: store (kv, score) in slot; when emit, write the
 * softmax(score)-weighted sum over the r slots to out. */
int ds41_v41_compress_pool(const float* kv, const float* score, float* state_kv, float* state_score, uint32_t slot,
                           uint32_t ratio, int emit, float* out, uint32_t dim);

/* score[t] = sum_h relu(q_h . k_t) * w_h * w_scale for t < n. */
int ds41_v41_indexer_score(const float* q, const float* keys, const float* w, float w_scale, float* score,
                           uint32_t n, uint32_t heads, uint32_t head_dim);
/* Candidate blocks: max over `block`-row blocks, newest block forced in, keep
 * the top `topk_blocks`; score[t] = -inf outside kept blocks writes `mask`
 * (1 keep / 0 drop) for t < n. */
int ds41_v41_candidate_mask(const float* score, uint8_t* mask, float* block_scratch, uint32_t* select_scratch,
                            uint32_t n, uint32_t block, uint32_t topk_blocks);
int ds41_v41_apply_mask(float* score, const uint8_t* mask, uint32_t n);
/* Indices of the k largest scores (k <= n), in any order. */
int ds41_v41_topk(const float* score, uint32_t* out, uint32_t n, uint32_t k, uint32_t* scratch, uint64_t scratch_bytes);

/* Sparse attention for one query over `n_window` ring rows and the `n_comp`
 * compressed rows comp[idx[j]]; keys equal values; sinks join the softmax
 * denominator. q/out are [heads, head_dim]. */
int ds41_v41_attention(const float* q, const float* window, uint32_t n_window, const float* comp,
                       const uint32_t* idx, uint32_t n_comp, const float* sinks, float* out, uint32_t heads,
                       uint32_t head_dim, float scale);

/* Routing: scores = sqrt(softplus(logits)); top-k of scores + bias; weights
 * normalized and multiplied by route_scale. */
int ds41_v41_route(const float* logits, const float* bias, int32_t* ids, float* weights, uint32_t n_experts,
                   uint32_t k, float route_scale);

/* out[s, i] = weight[s] * silu(min(gate, limit)) * clamp(up, -limit, limit); weight may be NULL. */
int ds41_v41_swiglu(const float* gate, const float* up, const float* weight, float* out, uint32_t rows,
                    uint32_t n, float limit);
/* out[i] = extra[i] + sum_r x[r, i]; extra may be NULL. */
int ds41_v41_sum_rows(const float* x, const float* extra, float* out, uint32_t rows, uint32_t n);

/* Engram: kv holds hc keys then one value of `dim` each. Per copy c,
 * h_c += sigmoid(signed_sqrt(dot)) * value. q/k are BF16 [hc, dim]. */
int ds41_v41_engram_gate(float* h, const float* kv, const uint16_t* q, const uint16_t* k, uint32_t hc,
                         uint32_t dim, float eps);

#ifdef __cplusplus
}
#endif
