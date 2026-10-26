/* DeepSeek V4.1 Flash DSpark drafter (internal).
 *
 * Follows the official inference/model.py DSparkBlock: the hc-mean of the
 * target's residual stream entering each target layer is concatenated,
 * projected and normalized (main_x); each of the three stages keeps a
 * 128-row ring of its main KV rows (kv_norm(wkv(main_x)), roped at the row's
 * position, FP8). A draft embeds [anchor, noise x (block - 1)] at the next
 * positions, runs the stages (block rows attend every ring row and every
 * block row, with sinks; 128-expert routed MoE), collapses the hc streams
 * through the target's output head and adds the Markov bias of the previous
 * token row by row. */
#pragma once

#include <cstdint>
#include <vector>

#include "model_data_internal.h"

struct ds41_v41_proposal;
struct ds41_v41_sample_result;

struct ds41_decoder;
struct ds41_engine;
struct ds41_dspark_sampler;

inline constexpr uint32_t DS41_DSPARK_STAGES = 3;
inline constexpr uint32_t DS41_DSPARK_BLOCK = 5;
inline constexpr uint32_t DS41_DSPARK_TARGET_LAYERS = 3;
inline constexpr uint32_t DS41_DSPARK_EXPERTS = 128;
inline constexpr uint32_t DS41_DSPARK_EXPERTS_USED = 3;
inline constexpr uint32_t DS41_DSPARK_MARKOV_RANK = 256;
inline constexpr int DS41_DSPARK_NOISE_TOKEN = 128799;
inline constexpr uint32_t DS41_DSPARK_TARGET_LAYER[DS41_DSPARK_TARGET_LAYERS] = {37, 38, 39};

/* A device tensor: `rows` rows of `k` values of GGUF `type`. */
struct ds41_dspark_tensor {
  const void* p = nullptr;
  uint32_t type = 0;
  uint32_t k = 0;
  uint64_t rows = 0;
};

struct ds41_dspark_stage {
  ds41_dspark_tensor hc_attn_fn, hc_attn_scale, hc_attn_base, attn_norm;
  ds41_dspark_tensor q_a, q_a_norm, q_b, kv, kv_a_norm, sinks, o_a, o_b;
  ds41_dspark_tensor hc_ffn_fn, hc_ffn_scale, hc_ffn_base, ffn_norm;
  ds41_dspark_tensor gate_inp, probs_b, gate_shexp, up_shexp, down_shexp;
  ds41_dspark_tensor gate_exps, up_exps, down_exps;
};

/* Host binding of one GGUF tensor before upload; F16 tensors convert to F32. */
struct ds41_dspark_source {
  ds41_dspark_tensor* dst;
  const ds41_tensor* src;
};

struct ds41_dspark_model {
  ds41_model file;
  std::vector<ds41_dspark_source> sources;
  void* device = nullptr;
  uint64_t device_bytes = 0;
  ds41_dspark_stage stage[DS41_DSPARK_STAGES];
  ds41_dspark_tensor fc, main_norm, out_norm, markov_w1, markov_w2;
  /* Optional learned acceptance head: sigmoid(w . [pre-norm hidden; W1 row]
   * + bias). p is null when the drafter ships none. */
  ds41_dspark_tensor confidence, confidence_bias;
  struct ds41_dspark_scratch* scratch = nullptr;
};

/* Uploads the bound sources into one device allocation and allocates the
 * engine-wide draft scratch. Returns false on failure. */
bool ds41_dspark_upload(ds41_dspark_model* m);
void ds41_dspark_free(ds41_dspark_model* m);

/* Per-session drafter state: the stage rings, valid for positions
 * [ring_start, ring_end). */
struct ds41_dspark_state;
ds41_dspark_state* ds41_dspark_state_create(ds41_engine* engine, ds41_decoder* decoder);
void ds41_dspark_state_destroy(ds41_dspark_state* s);
void ds41_dspark_state_reset(ds41_dspark_state* s, uint32_t pos);
uint32_t ds41_dspark_state_end(const ds41_dspark_state* s);
/* The device rings ([stage][128][head_dim] floats) and their positions. */
float* ds41_dspark_state_ring(ds41_dspark_state* s, uint64_t* bytes);
void ds41_dspark_state_range(const ds41_dspark_state* s, uint32_t* start, uint32_t* end);
bool ds41_dspark_state_set_range(ds41_dspark_state* s, uint32_t start, uint32_t end);
/* Appends the main KV rows of the decoder's captured features of positions
 * [pos0, pos0 + rows); rows that do not continue the ring restart it. */
bool ds41_dspark_inject(ds41_dspark_state* s, uint32_t pos0, uint32_t rows);
/* Draft count tokens (1..DS41_DSPARK_BLOCK) after `anchor`, the token at
 * position ds41_dspark_state_end. Writes exactly count IDs. An optional
 * sampler proposes from post-Markov candidates (writing q's top probability
 * per row to `top`, if given); otherwise each row takes its argmax. */
bool ds41_dspark_draft(ds41_dspark_state* s, int anchor, int32_t* drafts, uint32_t count,
                       const ds41_dspark_sampler* sampler = nullptr, float* top = nullptr,
                       float* confidence = nullptr);
/* The session's last sampled draft's device proposals, one per drafted row. */
const ds41_v41_proposal* ds41_dspark_proposals(const ds41_dspark_state* s);
/* Whether the drafter has a usable learned acceptance head; without one,
 * ds41_dspark_draft writes -1 confidences. */
bool ds41_dspark_has_confidence(const ds41_dspark_model* m);
/* The last draft's target-head logits before the Markov bias, DS41_DSPARK_BLOCK rows (tests). */
bool ds41_dspark_copy_logits(const ds41_engine* engine, float* out);
/* Ring bytes, and their copy to/from host memory (with ring_start/end). */
uint64_t ds41_dspark_state_bytes(const ds41_dspark_state* s);
bool ds41_dspark_state_save(const ds41_dspark_state* s, std::vector<uint8_t>* out);
bool ds41_dspark_state_load(ds41_dspark_state* s, const uint8_t* data, uint64_t size, uint64_t* consumed);

/* Decoder hooks (v41_decode.cpp). Capture keeps the hc-mean features of the
 * target layers for the last rows of every evaluation. */
inline constexpr uint32_t DS41_DSPARK_FEATURE_ROWS = 128;
bool ds41_decoder_enable_features(ds41_decoder* d);
/* ds41_v41_sample_row on row `row` of the last evaluation's logits, while
 * resident (evaluations of at most 64 logit rows). */
bool ds41_decoder_sample(ds41_decoder* d, uint32_t row, const ds41_v41_proposal* q, float temperature, double u,
                         double u2, ds41_v41_sample_result* out);
/* Device features [rows][3 * dim] of positions [*pos0, *pos0 + *rows). */
const float* ds41_decoder_features(const ds41_decoder* d, uint32_t* pos0, uint32_t* rows);
/* Row-major out[t] = W x[t] for a [rows, k] weight of GGUF `type`. */
bool ds41_v41_linear(uint32_t type, const void* w, uint32_t rows, uint32_t k, const float* x, float* out, uint32_t n);
/* The target's output head and token embedding. */
bool ds41_decoder_output_head(const ds41_decoder* d, const float* x, float* logits, uint32_t n);
bool ds41_decoder_embed(const ds41_decoder* d, const int* tokens, uint32_t n, float* host_rows);
const float* ds41_decoder_freq_plain(const ds41_decoder* d);
