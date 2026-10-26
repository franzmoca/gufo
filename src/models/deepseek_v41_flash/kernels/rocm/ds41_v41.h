/* DeepSeek V4.1 Flash decode and prefill kernels.
 *
 * Each entry mirrors one operation of the official inference/model.py (and
 * the tools/ds41 oracle) on fp32 device buffers, for `n` consecutive tokens
 * at positions pos0 .. pos0 + n - 1 (row-major, one row per token). All
 * launches use the legacy stream, like the rest of the backend. Functions
 * return 0 on success.
 *
 * Compressed-attention lengths follow the official causal rule: at position
 * p a ratio-r layer sees (p + 1) / r compressed rows. */
#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* out[t, r] = sum_k W[r, k] * x[t, k]; W is row-major BF16 or F32. */
int ds41_v41_matmul_bf16(const uint16_t* w, const float* x, float* out, uint32_t rows, uint32_t k, uint32_t n);
/* F16 weights, the same per-row sums as F32 weights (the conversion is exact). */
int ds41_v41_matmul_f16(const uint16_t* w, const float* x, float* out, uint32_t rows, uint32_t k, uint32_t n);
int ds41_v41_matmul_f32(const float* w, const float* x, float* out, uint32_t rows, uint32_t k, uint32_t n);

/* Decode projections y[c, r] = W[r] . x_c for Q5_K (13) or Q8_0 (8) W
 * ([m][k] in GGUF blocks, k % 256 == 0) and FP32 activations, for n tokens
 * (1 to DS41_V41_GEMV_MAX_TOKENS) read once per weight: row r of token c
 * reads x + c * x_stride + (r / rows_per_x) * k (rows_per_x 0: every row
 * reads x + c * x_stride) and writes y + c * y_stride + r. `slices` (1, 2, 4,
 * 8; <= 0 picks the default) waves share a row group. Every token sees the
 * arithmetic of a single-token launch. Returns 1 for other types. */
#define DS41_V41_GEMV_MAX_TOKENS 8
int ds41_v41_gemv(int type, const void* w, const float* x, float* y, uint32_t m, uint32_t k, uint32_t rows_per_x,
                  int slices, uint32_t n, uint64_t x_stride, uint32_t y_stride);
/* Two matrices of one type and width k over the same x in one launch:
 * y1 = W1 x (m1 rows), y2 = W2 x (m2 rows), for n tokens of k floats (y1, y2
 * rows m1, m2 floats). */
int ds41_v41_gemv_pair(int type, const void* w1, float* y1, uint32_t m1, const void* w2, float* y2, uint32_t m2,
                       const float* x, uint32_t k, int slices, uint32_t n);
/* Gate and up matrices of the same type and shape over one x, fused with the
 * SwiGLU of ds41_v41_swiglu (no weight): y = silu(min(gate x, limit)) *
 * clamp(up x, +-limit), for n tokens of k floats (y rows m floats). */
int ds41_v41_gemv_swiglu(int type, const void* gate, const void* up, const float* x, float* y, uint32_t m, uint32_t k,
                         float limit, int slices, uint32_t n);
/* Decode routed experts in IQ1_S, IQ1_M, IQ2_XXS or IQ2_S with FP32
 * activations: y[s, 0:m] = W[ids[s]] x_s for s < slots, x_s = x + (s /
 * slots_per_x) * x_slot (x_slot 0 shares one input); expert e's blocks start
 * at w + e * expert_bytes, and a negative id writes zeros. Every slot sees the
 * same arithmetic. Returns 1 for other types. */
int ds41_v41_gemv_experts(int type, const void* w, uint64_t expert_bytes, const int32_t* ids, uint32_t slots,
                          const float* x, uint32_t x_slot, uint32_t slots_per_x, float* y, uint32_t m, uint32_t k);
/* Gate and up of the same experts (one type, same layout) over the inputs of
 * ds41_v41_gemv_experts, fused with the SwiGLU of ds41_v41_swiglu: y[s, r] =
 * route_w[s] * silu(min(gate, limit)) * clamp(up, +-limit); route_w may be
 * NULL. */
int ds41_v41_gemv_experts_swiglu(int type, const void* gate, const void* up, uint64_t expert_bytes,
                                 const int32_t* ids, uint32_t slots, const float* x, uint32_t x_slot,
                                 uint32_t slots_per_x, const float* route_w, float limit, float* y, uint32_t m,
                                 uint32_t k);

/* Small batches of routed experts, each expert's weights decoded once for
 * all slots routed to it. ds41_v41_expert_plan groups the `slots` (<=
 * DS41_V41_EXPERT_PLAN_SLOTS) ids by expert into `plan`
 * (DS41_V41_EXPERT_PLAN_INTS(slots) device ints). The grouped GEMV then
 * computes ds41_v41_gemv_experts (w_up NULL) or, for the IQ1 and IQ2_XXS
 * types, ds41_v41_gemv_experts_swiglu over the planned slots; no expert may
 * occur more than `cols` (<= DS41_V41_EXPERT_PLAN_COLS) times, which top-k
 * routing of `cols` tokens guarantees. Each slot's result equals the
 * ungrouped launch bit for bit. */
#define DS41_V41_EXPERT_PLAN_SLOTS 64
#define DS41_V41_EXPERT_PLAN_COLS 8
#define DS41_V41_EXPERT_PLAN_INTS(slots) (1 + (slots) * (2 + DS41_V41_EXPERT_PLAN_COLS))
int ds41_v41_expert_plan(const int32_t* ids, uint32_t slots, int32_t* plan);
int ds41_v41_gemv_experts_grouped(int type, const void* w, const void* w_up, const float* route_w, float limit,
                                  uint64_t expert_bytes, const int32_t* plan, uint32_t slots, uint32_t cols,
                                  const float* x, uint32_t x_slot, uint32_t slots_per_x, float* y, uint32_t m,
                                  uint32_t k);

/* Batched dense projections (Q5_K) on the FP16 matrix cores: for each group
 * z < groups, out[t, z m + r] = W_z[r] . x[t, z k : z k + k] over n tokens,
 * W_z the z-th [m][k] matrix stored back to back; x rows hold groups * k
 * floats, rounded to FP16 as they are staged, and out rows groups * m;
 * non-finite outputs read as 0. Returns 1 for other types, m not a multiple
 * of 256 or k not a multiple of 256. */
int ds41_v41_dense_f16(int type, const void* w, const float* x, float* out, uint32_t m, uint32_t n, uint32_t k,
                       uint32_t groups);
/* Batched routed experts (IQ1_S, IQ1_M, IQ2_XXS) on the FP16 matrix cores, with FP16
 * activations x. rows lists the slots grouped by expert, expert e owning
 * rows[bounds[e] .. bounds[e + 1]), and tiles[0:n_tiles] names 64 of them
 * each as expert | index << 16 (see ds41_v41_routed_plan). A slot s reads x
 * row s / x_div. With w_up, y[s, r] = route_w[s] * silu(min(gate, limit)) *
 * clamp(up, +-limit), FP16 if y_half, else FP32; without it, y (FP32)
 * [s, r] = (W x)[r]. Non-finite products read as 0. Returns 1 for other
 * types or k not a multiple of 256. */
int ds41_v41_routed_f16(int type, const void* w, const void* w_up, uint64_t expert_bytes, const uint16_t* x,
                        uint32_t x_div, const int32_t* rows, const int32_t* bounds, const int32_t* tiles,
                        uint32_t n_tiles, const float* route_w, float limit, void* y, int y_half, uint32_t m,
                        uint32_t k);
/* y[0:n] = x[0:n] rounded to FP16 (bits). */
int ds41_v41_to_half(const float* x, uint16_t* y, uint64_t n);
/* Host side: rows[slots], bounds[n_experts + 1] and tiles (at most
 * slots / 64 + n_experts entries) from the slot experts ids[slots]; negative
 * ids are skipped. Returns the tile count. */
uint32_t ds41_v41_routed_plan(const int32_t* ids, uint32_t slots, uint32_t n_experts, int32_t* rows,
                              int32_t* bounds, int32_t* tiles);

/* dst[r, 0:cols] = src[r, 0:cols] for `rows` rows with the given strides (floats). */
int ds41_v41_copy_rows(float* dst, uint64_t dst_stride, const float* src, uint64_t src_stride, uint32_t rows,
                       uint32_t cols);

/* out[r] = weight * x[r] * rsqrt(mean(x[r]^2) + eps) for `rows` rows of n; weight may be NULL. */
int ds41_v41_rms_norm(const float* x, const float* weight, float* out, uint32_t rows, uint32_t n, float eps);

/* Rotate adjacent pairs of the last `rope_dim` values of each of `rows` rows
 * of `row_dim`. Row r sits at position pos0 + (r / rows_per_pos) * pos_step;
 * inv_freq has rope_dim/2 entries. */
int ds41_v41_rope_tail(float* x, uint32_t rows, uint32_t row_dim, uint32_t rope_dim, const float* inv_freq,
                       uint32_t pos0, uint32_t rows_per_pos, uint32_t pos_step, int inverse);

/* In-place fake quantization (quantize then dequantize). rope_fp8 applies
 * ds41_v41_rope_tail (same arguments) and then FP8 E4M3 with a power-of-two
 * scale per 32 values (act_quant, ue8m0); row_dim and rope_dim are multiples
 * of 32. fp4 quantizes n values as FP4 E2M1 per `block` values with an E4M3
 * scale (e4m3_scale) or a power-of-two scale. */
int ds41_v41_rope_fp8(float* x, uint32_t rows, uint32_t row_dim, uint32_t rope_dim, const float* inv_freq,
                      uint32_t pos0, uint32_t rows_per_pos, uint32_t pos_step, int inverse);
int ds41_v41_fp4_fake_quant(float* x, uint32_t n, uint32_t block, int e4m3_scale);
/* FP4 rows: E2M1 codes two per byte (even value in the low nibble), dim / 2
 * bytes per row, and one scale byte per `block` values: the E4M3FN encoding
 * (e4m3_scale) or the float exponent field of a power-of-two scale. The
 * compressed KV rows use block 16 with E4M3 scales, the index-K rows block 32
 * with power-of-two scales. */
typedef struct ds41_fp4_rows {
  const uint8_t* codes;
  const uint8_t* scales;
} ds41_fp4_rows;
/* The same quantization of x packed into codes/scales (exact: decoding
 * returns the quantized floats); x is kept. */
int ds41_v41_fp4_store(const float* x, uint8_t* codes, uint8_t* scales, uint32_t n, uint32_t block, int e4m3_scale);
/* qr = RMSNorm(qa) * q_weight, and kvn = RMSNorm(kv) * kv_weight followed by
 * ds41_v41_rope_fp8 (token t at pos0 + t), for n tokens in one launch. */
int ds41_v41_q_kv_norm(const float* qa, const float* q_weight, float* qr, uint32_t q_dim, const float* kv,
                       const float* kv_weight, float* kvn, uint32_t kv_dim, uint32_t rope_dim, const float* inv_freq,
                       uint32_t pos0, uint32_t n, float eps);

/* mHC per token. mixes[t, 24] = (fn . flatten(h[t])) * rsqrt(mean(h[t]^2) + eps). */
int ds41_v41_hc_mixes(const float* fn, const float* h, float* mixes, uint32_t hc, uint32_t dim, float eps,
                      uint32_t n);
/* Split mixes into pre[t, hc], post[t, hc] and Sinkhorn-normalized comb[t, hc*hc]. */
int ds41_v41_hc_sinkhorn(const float* mixes, const float* scale, const float* base, float* pre, float* post,
                         float* comb, uint32_t hc, uint32_t iters, float eps, uint32_t n);
/* hc_mixes followed by hc_sinkhorn (iters, hc_eps); small batches run as one kernel. */
int ds41_v41_hc_mix(const float* fn, const float* h, const float* scale, const float* base, float* mixes, float* pre,
                    float* post, float* comb, uint32_t hc, uint32_t dim, float eps, uint32_t iters, float hc_eps,
                    uint32_t n);
/* x[t, i] = sum_c pre[t, c] * h[t, c, i]; out[t] = weight * RMSNorm(x[t]). */
int ds41_v41_hc_collapse_norm(const float* h, const float* pre, const float* weight, float* out, uint32_t hc,
                              uint32_t dim, float eps, uint32_t n);
/* Batched hc_mixes and hc_collapse_norm over one read of the state: with y,
 * the state is first ds41_v41_hc_expand(y, h, post, comb) and stored to
 * h_out, otherwise it is h. mixes come from fn, xn from pre and norm_w.
 * Returns 1 for hc != 4 or dim not a multiple of 256 up to 8192. */
int ds41_v41_hc_state_pass(const float* y, const float* post, const float* comb, const float* h, float* h_out,
                           const float* pre, const float* norm_w, float* xn, const float* fn, float* mixes,
                           uint32_t hc, uint32_t dim, float eps, uint32_t n);
/* out[t, b, i] = post[t, b] * y[t, i] + sum_a comb[t, a, b] * residual[t, a, i]. */
int ds41_v41_hc_expand(const float* y, const float* residual, const float* post, const float* comb, float* out,
                       uint32_t hc, uint32_t dim, uint32_t n);
/* ds41_v41_hc_expand with y[t] = extra[t] + sum_r x[t, r, :]; extra may be NULL. */
int ds41_v41_hc_expand_sum(const float* x, const float* extra, uint32_t rows, const float* residual, const float* post,
                           const float* comb, float* out, uint32_t hc, uint32_t dim, uint32_t n);

/* Ratio-r compressor over n tokens: token t stores (kv[t], score[t]) in
 * frontier slot (pos0 + t) % r; each token that completes a group appends
 * the softmax(score)-weighted sum over the r slots to out, in order. */
int ds41_v41_compress_pool(const float* kv, const float* score, float* state_kv, float* state_score, uint32_t pos0,
                           uint32_t n, uint32_t ratio, float* out, uint32_t dim);

/* score[t, j] = sum_h relu(q[t, h] . k_j) * w[t, h] * w_scale for the
 * j < (pos0 + t + 1) / ratio compressed rows (FP4 index-K rows); rows have
 * score_stride floats. */
int ds41_v41_indexer_score(const float* q, ds41_fp4_rows keys, const float* w, float w_scale, float* score,
                           uint64_t score_stride, uint32_t pos0, uint32_t n, uint32_t ratio, uint32_t heads,
                           uint32_t head_dim);
/* Small-batch (1-4 rows), 32-head, D128 scores restricted to 8-row
 * candidate blocks. The mask must select min(candidate_stride, visible
 * blocks) blocks. Scratch holds n * candidate_stride uint32 block ids.
 * Unselected scores become -inf; selected scores keep dense FP32 numerics. */
int ds41_v41_indexer_score_candidates(const float* q, ds41_fp4_rows keys, const float* w, float w_scale,
                                      float* score, uint64_t score_stride, const uint32_t* mask,
                                      uint64_t mask_stride, uint32_t* candidates, uint32_t candidate_stride,
                                      uint32_t pos0, uint32_t n, uint32_t ratio);
/* Candidate blocks per token: max over `block`-row blocks, newest block
 * forced in, keep the top `topk_blocks`; writes the keep bit of block b as
 * bit b % 32 of mask[t, b / 32], mask_stride words per token. block_scratch
 * holds n * block_stride floats (at least one per block), select_scratch
 * n * topk_blocks. */
int ds41_v41_candidate_mask(const float* score, uint64_t score_stride, uint32_t* mask, uint64_t mask_stride,
                            float* block_scratch, uint64_t block_stride, uint32_t* select_scratch, uint32_t pos0, uint32_t n,
                            uint32_t ratio, uint32_t block, uint32_t topk_blocks);
/* score[t, j] = -inf where the keep bit of block j / block is clear. */
int ds41_v41_apply_mask(float* score, uint64_t score_stride, const uint32_t* mask, uint64_t mask_stride, uint32_t pos0,
                        uint32_t n, uint32_t ratio, uint32_t block);
/* out[t, 0:k_t] = indices of the k_t = min(k, len_t) largest scores of token
 * t, len_t = ceil(((pos0 + t + 1) / ratio) / div); rows of out hold k. */
int ds41_v41_topk(const float* score, uint64_t score_stride, uint32_t* out, uint32_t k, uint32_t pos0, uint32_t n,
                  uint32_t ratio, uint32_t div);

/* Sparse attention for n queries. Token t at p = pos0 + t attends the window
 * positions max(0, p - window + 1, min(floor, p)) .. p, read from `ring`
 * (slot pos % window, positions before pos0) and from `chunk` (row
 * pos - pos0), plus
 * min(k, (p + 1) / ratio) compressed rows comp[idx[t, j]] (FP4 KV rows; none when ratio
 * is 0). Keys equal values; sinks join the softmax denominator. q/out are
 * [n, heads, head_dim]. With inv_freq, q is first rotated as by
 * ds41_v41_rope_tail (rows_per_pos heads, step 1; q may be rotated in place)
 * and out gets ds41_v41_rope_fp8's inverse rotation and FP8 rounding
 * (rope_scratch: n * rope_dim floats).
 * Batches below DS41_V41_ATTN_SPLIT_TOKENS split the rows across blocks and
 * need `partial` with n * heads * DS41_V41_ATTN_PARTIAL_PER_HEAD floats. */
#define DS41_V41_ATTN_SPLIT_TOKENS 8u
#define DS41_V41_ATTN_PARTIAL_PER_HEAD (20u * 516u)
int ds41_v41_attention(float* q, const float* ring, const float* chunk, uint32_t window, uint32_t floor,
                       ds41_fp4_rows comp, const uint32_t* idx, uint32_t k, uint32_t ratio, const float* sinks,
                       float* out, float* partial, uint32_t pos0, uint32_t n, uint32_t heads, uint32_t head_dim,
                       float scale, const float* inv_freq, uint32_t rope_dim, float* rope_scratch);

/* Routing per token: scores = sqrt(softplus(logits)); top-k of scores + bias;
 * weights normalized and multiplied by route_scale. ids/weights are [n, k]. */
int ds41_v41_route(const float* logits, const float* bias, int32_t* ids, float* weights, uint32_t n_experts,
                   uint32_t k, float route_scale, uint32_t n);

/* out[s, i] = weight[s] * silu(min(gate, limit)) * clamp(up, -limit, limit); weight may be NULL. */
int ds41_v41_swiglu(const float* gate, const float* up, const float* weight, float* out, uint32_t rows,
                    uint32_t n, float limit);

/* Engram: kv[t] holds hc keys then one value of `dim` each. Per copy c,
 * h[t, c] += sigmoid(signed_sqrt(dot)) * value. q/k are BF16 [hc, dim]. */
int ds41_v41_engram_gate(float* h, const float* kv, const uint16_t* q, const uint16_t* k, uint32_t hc,
                         uint32_t dim, float eps, uint32_t n);

/* DSpark block attention: each of the n_block queries attends the ring rows
 * of positions first .. first + n_ring - 1 (slot pos % window) and all
 * n_block rows of block_kv, with sinks; keys equal values. No rotation. */
int ds41_v41_dspark_attention(const float* q, const float* ring, uint32_t first, uint32_t n_ring, uint32_t window,
                              const float* block_kv, uint32_t n_block, const float* sinks, float* out, uint32_t heads,
                              uint32_t head_dim, float scale);
/* out[t * out_stride + i] = mean over the hc streams of h[t, :, i]. */
int ds41_v41_hc_mean(const float* h, float* out, uint64_t out_stride, uint32_t hc, uint32_t dim, uint32_t n);
enum { DS41_V41_ROW_F32, DS41_V41_ROW_BF16, DS41_V41_ROW_F16 };
/* out[0:rank] = row tokens[index] of an F32, BF16 or F16 table (DS41_V41_ROW_*). */
int ds41_v41_gather_row(const void* table, int format, const int32_t* tokens, uint32_t index, float* out,
                        uint32_t rank);
/* tokens[index] = the first index of the largest finite x[v] + bias[v]
 * (bias may be NULL), or -1 when none is finite. */
int ds41_v41_argmax_bias(const float* x, const float* bias, uint32_t n, int32_t* tokens, uint32_t index);

#ifdef __cplusplus
}
#endif
