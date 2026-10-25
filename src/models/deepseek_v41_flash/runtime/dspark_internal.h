#ifndef GUFO_MODELS_DEEPSEEK_V41_FLASH_RUNTIME_DSPARK_INTERNAL_H_
#define GUFO_MODELS_DEEPSEEK_V41_FLASH_RUNTIME_DSPARK_INTERNAL_H_

#include <cstddef>
#include <cstdint>

#include "model_data_internal.h"

/*
 * DeepSeek V4 Flash DSpark support model.
 *
 * DSpark is DeepSeek's own speculative drafter for this checkpoint. It is a
 * separate GGUF holding a short stack of full DS4 blocks (MLA attention, routed
 * MoE, hyper-connections) plus a low-rank Markov path selector. Each request's
 * verification budget adapts to observed acceptance and calibrated cost.
 * DSpark is unrelated to the
 * Qwen DFlash and DFlash-2 drafters, which use block diffusion over a single
 * fused draft layer.
 *
 * One draft pass proposes a whole block of `block_size` tokens:
 *
 *   1. Feature injection - the target's hidden states at `target_layer_ids` are
 *      concatenated, projected by `main_proj`, normalized by `main_norm`, and
 *      turned into per-stage KV rows. This is what lets a three-block drafter
 *      track a forty-three-block target.
 *   2. Non-causal block attention - the block enters as
 *      [target_next_token, MASK, MASK, ...] embedded with the target's tied
 *      token embeddings. Every block row attends to the injected KV *and* to
 *      all other block rows, so the whole block is produced by one forward
 *      pass instead of `block_size` sequential steps.
 *   3. Path selection - base logits come from the target LM head; the Markov
 *      head then adds W2 * W1[previous token] so the block follows one
 *      coherent trajectory rather than independent per-position argmaxes.
 */

inline constexpr uint32_t DS41_DSPARK_MAX_STAGES = 8;
inline constexpr uint32_t DS41_DSPARK_RETRY_TOKENS = 64;
inline constexpr uint32_t DS41_DSPARK_MAX_BLOCK = 16;
inline constexpr uint32_t DS41_DSPARK_MAX_TARGET_LAYERS = 8;

struct ds41_dspark_attention_span {
  uint32_t start = 0;
  uint32_t count = 0;
};

// Official DSparkAttention sees the last 128 committed target rows and the
// entire draft block. Physical ring capacity also accommodates prefill writes;
// it does not determine the attention window. Draft row zero starts at
// prefix_tokens, immediately after the independently injected target prefix.
inline constexpr ds41_dspark_attention_span ds41_dspark_visible_span(
    uint32_t prefix_tokens, uint32_t draft_rows, uint32_t cache_capacity) {
  if (draft_rows == 0 || draft_rows > cache_capacity ||
      prefix_tokens > UINT32_MAX - draft_rows) {
    return {};
  }
  const uint32_t past = prefix_tokens < DS41_N_SWA ? prefix_tokens : DS41_N_SWA;
  if (past > cache_capacity - draft_rows) {
    return {};
  }
  return {(prefix_tokens - past) % cache_capacity, past + draft_rows};
}

/* One request verifies its drafted block plus the target correction row.
 * A coordinator lazily grows to eight such independent request blocks. */
inline constexpr uint32_t DS41_SPEC_SESSION_MAX_ROWS =
    DS41_DSPARK_MAX_BLOCK + 1u;
inline constexpr uint32_t DS41_SPEC_MAX_ROWS =
    8u * DS41_SPEC_SESSION_MAX_ROWS;

// Per-request controller state is durable for an exact snapshot continuation.
// A new request reusing the prefix resets it while retaining the support KV.
struct ds41_dspark_request_state {
  uint64_t drafted = 0;
  uint64_t accepted = 0;
  uint64_t support_drafted = 0;
  uint64_t support_accepted = 0;
  uint64_t positional_accepted = 0;
  uint64_t anchors = 0;
  uint64_t full_blocks = 0;
  uint64_t steps = 0;
  uint64_t skipped = 0;
  uint64_t policy_start_drafted = 0;
  uint64_t policy_start_accepted = 0;
  uint64_t policy_start_steps = 0;
  uint32_t skip_remaining = 0;
  uint32_t concurrent_width = 3;
  uint32_t policy_concurrency = 1;
  uint32_t probe_cycles = 0;
  uint32_t probe_emitted = 0;
  uint32_t probe_cost = 0;
  bool plain_only = false;
  bool force_plain_request = false;
  /* Sampled token rejected by the last cycle, emitted as the next anchor.
   * Never persisted: a restored prefix starts a new request. */
  int pending_anchor = -1;
};

struct ds41_dspark_stage_weights {
    /* Present on the first stage only: fuses the target's sampled layers. */
    ds41_tensor *main_proj;
    ds41_tensor *main_norm;

    /* Present on the last stage only. */
    ds41_tensor *norm;
    ds41_tensor *hc_head_base;
    ds41_tensor *hc_head_fn;
    ds41_tensor *hc_head_scale;
    ds41_tensor *markov_w1;
    ds41_tensor* markov_w2;
    ds41_tensor* confidence;
    /* Every stage is a DS4 block with no compressor and no indexer, so it
     * binds into the same layout the target's ratio-0 layers use. */
    ds41_layer_weights block;
};

struct ds41_dspark_model {
    ds41_model *model;
    uint32_t n_stages;
    uint32_t block_size;
    uint32_t markov_rank;
    uint32_t noise_token_id;
    uint32_t n_target_layers;
    uint32_t target_layer_ids[DS41_DSPARK_MAX_TARGET_LAYERS];
    ds41_dspark_stage_weights stage[DS41_DSPARK_MAX_STAGES];
};

/* Loads and strictly validates a DSpark support GGUF, then makes its tensor
 * spans GPU-visible through the selected residency policy. Returns 0 on
 * success. */
int ds41_dspark_open(ds41_dspark_model **out, const char *path);
void ds41_dspark_close(ds41_dspark_model *dspark);

/* Total concatenated target feature width consumed by main_proj. */
uint32_t ds41_dspark_feature_width(const ds41_dspark_model *dspark);

#endif  // GUFO_MODELS_DEEPSEEK_V41_FLASH_RUNTIME_DSPARK_INTERNAL_H_
