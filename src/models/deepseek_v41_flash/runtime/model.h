#ifndef GUFO_MODELS_DEEPSEEK_V41_FLASH_RUNTIME_MODEL_H_
#define GUFO_MODELS_DEEPSEEK_V41_FLASH_RUNTIME_MODEL_H_

#include <cstddef>
#include <cstdint>
#include <vector>

struct ds41_tokens {
  int* v;
  int len;
  int cap;
};

struct ds41_engine;
struct ds41_session;

using ds41_session_cancel_fn = bool (*)(void* user_data);

struct ds41_engine_options {
  const char* model_path;
  /* Optional DSpark support model. NULL leaves speculative decoding off and
   * the engine byte-for-byte identical to a non-speculative build. */
  const char* dspark_model_path;
  /* Engram GGUF. NULL opens deepseek41.engram.file beside the first shard. */
  const char* engram_path;
  /* Parse, validate and bind the artifacts without uploading any weights. */
  bool validate_only;
  /* Most prompt tokens per prefill batch, DS41_MIN_PREFILL_BATCH to
   * DS41_MAX_PREFILL_BATCH; 0 takes DS41_DEFAULT_PREFILL_BATCH. Larger
   * batches fill the routed expert tiles better for about 0.6 MiB of
   * activations per token. */
  uint32_t prefill_batch;
  /* Prefill runs every decoder layer over every prompt token instead of the
   * decoder SWA bounded replay (ds41_decoder_eval_prompt). */
  bool exact_prefill;
};

inline constexpr uint32_t DS41_MIN_PREFILL_BATCH = 16;
inline constexpr uint32_t DS41_DEFAULT_PREFILL_BATCH = 2048;
inline constexpr uint32_t DS41_MAX_PREFILL_BATCH = 4096;

struct ds41_session_snapshot {
  uint8_t* ptr;
  uint64_t len;
  uint64_t cap;
};

struct ds41_session_batch_item {
  ds41_session* session;
  int token;
};

inline constexpr uint32_t DS41_DSPARK_CANDIDATES = 8;

struct ds41_dspark_candidates {
  int32_t ids[DS41_DSPARK_CANDIDATES];
  float logits[DS41_DSPARK_CANDIDATES];
  float confidence;
};

/* Request sampler for a DSpark cycle. NULL keeps the greedy argmax cycle.
 *
 * `sample` draws the next token from one row of target logits with the
 * request's sampling configuration and RNG. Every drawn token is eventually
 * emitted in draw order (accepted rows now, a rejected row as the next cycle's
 * anchor), so the callee may record it as request history immediately.
 * `accept` reports a token the runtime emits without drawing it: the anchor
 * carried over from the previous cycle's rejected row.
 *
 * Optional paired callbacks `propose`/`verify` implement stochastic proposals.
 * `propose` retains the normalized q actually used to draw this position;
 * `verify` returns either that draft (accepted) or an exact residual draw.
 * Each callback and its probability rows belong to one request and one cycle.
 * Greedy-with-penalties and legacy callers retain the point-mass path. */
struct ds41_dspark_sampler {
  void* ctx;
  int (*sample)(void* ctx, const float* logits, uint32_t vocabulary_size);
  void (*accept)(void* ctx, int token);
  int (*propose)(void* ctx, uint32_t row,
                 const ds41_dspark_candidates* candidates) = nullptr;
  int (*verify)(void* ctx, uint32_t row, const float* logits,
                uint32_t vocabulary_size, int draft) = nullptr;
  // Accept an already-computed target argmax only when it is also the exact
  // request selection. A false result must leave sampler state unchanged.
  bool (*try_accept_argmax)(void* ctx, int token) = nullptr;
};

struct ds41_session_dspark_batch_item {
  ds41_session* session;
  int* emitted;
  int emitted_cap;
  uint32_t max_draft_tokens;
  int* n_emitted;
  const ds41_dspark_sampler* sampler;
  bool stop_at_eos = false;
};

inline constexpr int DS41_SESSION_SYNC_INTERRUPTED = 2;
inline constexpr std::uint32_t DS41_SESSION_PAYLOAD_VERSION = 5;

int ds41_engine_open(ds41_engine** out, const ds41_engine_options* options);
void ds41_engine_close(ds41_engine* engine);
int ds41_engine_vocab_size(const ds41_engine* engine);
const char* ds41_engine_model_name(const ds41_engine* engine);

void ds41_tokens_free(ds41_tokens* tokens);
void ds41_tokenize_text(ds41_engine* engine, const char* text,
                        ds41_tokens* out);
void ds41_tokenize_rendered_chat(ds41_engine* engine, const char* text,
                                 ds41_tokens* out);
void ds41_encode_chat_prompt(ds41_engine* engine, const char* system,
                             const char* prompt, ds41_tokens* out);
void ds41_chat_begin(ds41_engine* engine, ds41_tokens* tokens);
void ds41_chat_append_message(ds41_engine* engine, ds41_tokens* tokens,
                              const char* role, const char* content);
void ds41_chat_append_assistant_prefix(ds41_engine* engine,
                                       ds41_tokens* tokens);
char* ds41_token_text(ds41_engine* engine, int token, size_t* length);
int ds41_token_eos(const ds41_engine* engine);
bool ds41_token_is_stop(const ds41_engine* engine, int token);

int ds41_session_create(ds41_session** out, ds41_engine* engine,
                        int context_size, bool use_dspark);
void ds41_session_free(ds41_session* session);
bool ds41_session_dspark_enabled(const ds41_session* session);
/* Evaluates the prompt from the longest evaluated prefix. The prompt is the
 * first part of a prompt ending at prompt_end (0: prompt->len), which
 * decides the decoder's bounded replay (ds41_decoder_eval_prompt; none with
 * the engine's exact_prefill); a state synced to a shorter part is complete
 * only for that same end. */
int ds41_session_sync(ds41_session* session, const ds41_tokens* prompt,
                      int prompt_end, char* error, size_t error_capacity);
int ds41_session_argmax(const ds41_session* session);
int ds41_session_sample(const ds41_session* session, float temperature,
                        int top_k, float top_p, float min_p,
                        uint64_t* rng_state);
int ds41_session_eval(ds41_session* session, int token, char* error,
                      size_t error_capacity);
int ds41_sessions_eval_batch(const ds41_session_batch_item* items,
                             size_t item_count, char* error,
                             size_t error_capacity);
void ds41_session_begin_request(ds41_session* session);
void ds41_session_prepare_batch_execution(ds41_session* session);
bool ds41_engine_has_dspark(const ds41_engine* engine);
bool ds41_engine_exact_prefill(const ds41_engine* engine);
int ds41_sessions_dspark_step_batch(const ds41_session_dspark_batch_item* items,
                                    size_t item_count, char* error,
                                    size_t error_capacity);
/* A speculative step leaves its last emitted token unevaluated (the next
 * step's anchor). Evaluates it, so plain steps can continue; a no-op without
 * one. ds41_session_eval refuses while such a token is outstanding. */
int ds41_session_flush(ds41_session* session, char* error,
                       size_t error_capacity);
void ds41_session_dspark_stats(const ds41_session* session, uint64_t* drafted,
                               uint64_t* accepted, uint64_t* support_drafted,
                               uint64_t* support_accepted,
                               uint64_t* positional_accepted, uint64_t* anchors,
                               uint64_t* full_blocks, uint64_t* steps,
                               uint64_t* skipped, uint32_t* context_tokens);
/* Prompt-lookup copies verified by DSpark steps. */
void ds41_session_lookup_stats(const ds41_session* session, uint64_t* cycles,
                               uint64_t* copied, uint64_t* accepted);
/* The prompt-lookup state the token history cannot rebuild
 * (PromptLookup::State); setting it fails unless it fits the session's
 * tokens. */
void ds41_session_lookup_state(const ds41_session* session, uint32_t* resume,
                               uint32_t* request_start);
int ds41_session_set_lookup_state(ds41_session* session, uint32_t resume,
                                  uint32_t request_start);
int ds41_session_copy_logits(const ds41_session* session, float* out,
                             int capacity);
int ds41_session_save_snapshot(const ds41_session* session,
                               ds41_session_snapshot* snapshot, char* error,
                               size_t error_capacity);
int ds41_session_load_snapshot(ds41_session* session,
                               const ds41_session_snapshot* snapshot,
                               char* error, size_t error_capacity);
void ds41_session_snapshot_free(ds41_session_snapshot* snapshot);
void ds41_session_set_cancel(ds41_session* session,
                             ds41_session_cancel_fn callback, void* user_data);
void ds41_session_invalidate(ds41_session* session);
int ds41_session_pos(const ds41_session* session);
int ds41_session_ctx(const ds41_session* session);
uint32_t ds41_session_prefill_capacity(const ds41_session* session);
uint64_t ds41_session_payload_bytes(const ds41_session* session);

/* Single-sequence V4.1 decoder from position 0. Prompts run layer-major in
 * batches of up to ds41_decoder_batch_capacity tokens; decode is a batch of
 * one. */
struct ds41_decoder;
/* Called after each block with the [hc, dim] residual streams and the routed
 * expert ids of that layer for the batch's last token, in host memory. */
typedef void (*ds41_decoder_layer_fn)(void* ctx, uint32_t layer,
                                      const float* hidden,
                                      const int32_t* experts);
ds41_decoder* ds41_decoder_create(ds41_engine* engine, uint32_t context);
void ds41_decoder_destroy(ds41_decoder* decoder);
uint32_t ds41_decoder_batch_capacity(const ds41_decoder* decoder);
/* Evaluates `n` tokens (n <= batch capacity) at the next positions. Writes
 * vocabulary logits for the last `logit_rows` tokens (0 to n) to `logits`,
 * row-major; logits may be NULL when logit_rows is 0. */
int ds41_decoder_eval(ds41_decoder* decoder, const int* tokens, uint32_t n,
                      float* logits, uint32_t logit_rows,
                      ds41_decoder_layer_fn layer_fn, void* ctx);
/* Decoder SWA bounded replay, as in DeepSeek's DeepInfer: of a prompt ending
 * at prompt_end, the decoder layers run only the last
 * DS41_DECODER_REPLAY_ROWS positions (the 128-row window and 12 rows of
 * backtracking) and their windows start there; the first decoder layer
 * still compresses every position for the global KV and index keys.
 * ds41_decoder_eval_prompt evaluates `n` tokens of such a prompt
 * (pos + n <= prompt_end); its logit rows and DSpark features come from the
 * replayed positions only, and the window state is complete once the
 * evaluation reaches prompt_end. */
#define DS41_DECODER_REPLAY_ROWS 140u
int ds41_decoder_eval_prompt(ds41_decoder* decoder, const int* tokens,
                             uint32_t n, uint32_t prompt_end, float* logits,
                             uint32_t logit_rows);
/* Rows a speculative batch may hold. */
#define DS41_DECODER_MAX_SPECULATIVE 16
/* ds41_decoder_eval of `n` tokens (n <= DS41_DECODER_MAX_SPECULATIVE) with
 * logits for every row, whose window rows and compressor inputs stay pending
 * until ds41_decoder_commit keeps a prefix of them. The decoder's position
 * advances by n meanwhile; no other evaluation may run before the commit. */
int ds41_decoder_eval_speculative(ds41_decoder* decoder, const int* tokens,
                                  uint32_t n, float* logits);
/* Keeps the first `keep` (1 to n) tokens of the pending speculative batch,
 * leaving the state of an evaluation of exactly those tokens. */
int ds41_decoder_commit(ds41_decoder* decoder, uint32_t keep);
/* ds41_decoder_eval of one token. */
int ds41_decoder_step(ds41_decoder* decoder, int token, float* logits,
                      ds41_decoder_layer_fn layer_fn, void* ctx);
uint32_t ds41_decoder_pos(const ds41_decoder* decoder);
uint32_t ds41_decoder_context(const ds41_decoder* decoder);
/* Returns to position 0; rows past the position are never read again. */
int ds41_decoder_reset(ds41_decoder* decoder);
/* Appends the decoder state to `out`; load consumes it from `data`. */
uint64_t ds41_decoder_state_bytes(const ds41_decoder* decoder);
int ds41_decoder_save(const ds41_decoder* decoder, std::vector<uint8_t>* out);
int ds41_decoder_load(ds41_decoder* decoder, const uint8_t* data, uint64_t size,
                      uint64_t* consumed);

/* Decoder state arrays, for the component cache. Row parts hold one row per
 * compression ratio tokens of a KV source layer; the others are whole. */
enum ds41_state_part {
  DS41_STATE_WINDOW,      /* sliding-window ring, every layer */
  DS41_STATE_COMP,        /* compressed KV, FP4 codes */
  DS41_STATE_COMP_SCALE,  /* compressed KV, FP4 scales */
  DS41_STATE_INDEX_K,     /* index K, FP4 codes */
  DS41_STATE_INDEX_SCALE, /* index K, FP4 scales */
  DS41_STATE_POOL_KV,     /* compressor frontier */
  DS41_STATE_POOL_SCORE,
};
/* Device memory of a layer's part, with the bytes of one row (row parts) or
 * of the whole part in *bytes; NULL when the layer has no such part. */
void* ds41_decoder_state_part(ds41_decoder* decoder, uint32_t layer, int part,
                              uint64_t* bytes);
/* Tokens per row of a KV source layer's row parts; 0 for other layers. */
uint32_t ds41_layer_state_ratio(uint32_t layer);
/* Adopts state restored into the decoder's arrays at `n` tokens, rebuilding
 * the Engram n-gram history from the tokens. */
int ds41_decoder_adopt(ds41_decoder* decoder, const int* tokens, uint32_t n);

/* Positions [first, end) are about to be evaluated (rows written). */
typedef void (*ds41_session_write_fn)(void* user_data, uint32_t first,
                                      uint32_t end);
void ds41_session_set_write_hook(ds41_session* session,
                                 ds41_session_write_fn callback,
                                 void* user_data);
ds41_decoder* ds41_session_decoder(ds41_session* session);
/* Whether the session holds evaluated state (false after a failed step). */
bool ds41_session_valid(const ds41_session* session);
/* The evaluated tokens and the frontier logits (vocabulary floats). */
const int* ds41_session_tokens(const ds41_session* session);
float* ds41_session_logits(ds41_session* session);
/* Starts a restore of `n` tokens: invalidates the session and returns its
 * token buffer, resized to n, for the caller to fill with the logits. */
int* ds41_session_begin_restore(ds41_session* session, uint32_t n);
/* Adopts the restored tokens, logits and decoder state. */
int ds41_session_finish_restore(ds41_session* session);
/* The drafter's device rings and the positions [start, end) they hold
 * (null and 0 without a drafter). Restoring a range needs end equal to the
 * session position; finish_restore empties it first. */
uint64_t ds41_dspark_ring_bytes(void);
void* ds41_session_dspark_ring(ds41_session* session, uint64_t* bytes);
int ds41_session_dspark_range(const ds41_session* session, uint32_t* start, uint32_t* end);
int ds41_session_dspark_set_range(ds41_session* session, uint32_t start, uint32_t end);

#endif  // GUFO_MODELS_DEEPSEEK_V41_FLASH_RUNTIME_MODEL_H_
