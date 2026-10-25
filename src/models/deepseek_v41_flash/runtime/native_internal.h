#ifndef GUFO_MODELS_DEEPSEEK_V41_FLASH_RUNTIME_NATIVE_INTERNAL_H_
#define GUFO_MODELS_DEEPSEEK_V41_FLASH_RUNTIME_NATIVE_INTERNAL_H_

#include <cstddef>
#include <cstdint>

#include "model.h"

struct ds41_dspark_model;
struct ds41_engram_tables;
struct ds41_dspark_request_state;
struct ds41_model;
struct ds41_rocm_graph;
struct ds41_vocab;
struct ds41_weights;

struct ds41_string_view {
  const char* ptr;
  uint64_t len;
};

struct ds41_string_iterator {
  const ds41_model* model;
  uint64_t pos;
  uint64_t remaining;
};

struct ds41_rocm_batch_item {
  ds41_rocm_graph* graph;
  int token;
  uint32_t position;
  float* logits;
};

struct ds41_rocm_verify_item {
  ds41_rocm_graph* graph;
  const ds41_tokens* tokens;
  uint32_t start;
  uint32_t n_tokens;
  uint32_t logical_n_tokens;
  int32_t* row_tops;
  float* frontier_logits;
  /* Optional: receives all `n_tokens` verified rows (row-major, vocab
   * floats each). The frontier gather is skipped for such items because the
   * caller decides acceptance by sampling those rows. */
  float* row_logits;
  /* Greedy items: the token the request stops at, or -1. The frontier gather
   * stops before an accepted stop token exactly as the CPU commit does. */
  int stop_token = -1;
};

/* Greedy acceptance of a verified DSpark block: tokens[0] is the anchor and
 * row_tops[i] the target argmax after tokens[i]. Accepts while the argmax
 * reproduces the next token and stops before an accepted stop_token (-1 for
 * none), so frontier row accepted - 1 proposes it as the next anchor. The CPU
 * commit and the GPU frontier gather both use it and must agree. */
inline uint32_t ds41_dspark_greedy_accept(const int32_t* row_tops,
                                          const int32_t* tokens,
                                          uint32_t n_tokens, int stop_token) {
  uint32_t accepted = 1;
  while (accepted < n_tokens && row_tops[accepted - 1] == tokens[accepted] &&
         tokens[accepted] != stop_token) {
    accepted++;
  }
  return accepted;
}

struct ds41_rocm_dspark_draft_item {
  ds41_rocm_graph* graph;
  int target_next_token;
  uint32_t position;
  uint32_t max_draft_tokens;
  int32_t* tokens;
  uint32_t* n_tokens;
  const ds41_dspark_sampler* sampler = nullptr;
  size_t concurrency = 1;
};

struct ds41_engine {
  ds41_model* model;
  ds41_vocab* vocab;
  ds41_weights* weights;
  ds41_dspark_model* dspark;
  ds41_engram_tables* engram;
  ds41_rocm_graph* batch_workspace;
  bool rocm_ready;
  uint32_t prefill_batch;  // most tokens per prefill batch
};

bool ds41_model_string_array_begin(const ds41_model* model, const char* key,
                                   ds41_string_iterator* iterator);
bool ds41_model_string_array_next(ds41_string_iterator* iterator,
                                  ds41_string_view* value);

ds41_vocab* ds41_vocab_create(const ds41_model* model);
void ds41_vocab_destroy(ds41_vocab* vocab);
void ds41_vocab_tokenize(const ds41_vocab* vocab, const char* text,
                         ds41_tokens* tokens);
void ds41_vocab_tokenize_rendered_chat(const ds41_vocab* vocab,
                                       const char* text, ds41_tokens* tokens);
void ds41_vocab_encode_chat(const ds41_vocab* vocab, const char* system,
                            const char* prompt, ds41_tokens* tokens);
void ds41_vocab_chat_begin(const ds41_vocab* vocab, ds41_tokens* tokens);
void ds41_vocab_chat_append_message(const ds41_vocab* vocab,
                                    ds41_tokens* tokens, const char* role,
                                    const char* content);
void ds41_vocab_chat_append_assistant_prefix(const ds41_vocab* vocab,
                                             ds41_tokens* tokens);
char* ds41_vocab_token_text(const ds41_vocab* vocab, int token, size_t* length);
int ds41_vocab_size(const ds41_vocab* vocab);
int ds41_vocab_eos(const ds41_vocab* vocab);
bool ds41_vocab_is_stop(const ds41_vocab* vocab, int token);

void ds41_tokens_push(ds41_tokens* tokens, int token);
void ds41_tokens_copy(ds41_tokens* destination, const ds41_tokens* source);
bool ds41_tokens_starts_with(const ds41_tokens* tokens,
                             const ds41_tokens* prefix);

int ds41_sample_argmax(const float* logits, uint32_t vocabulary_size);
int ds41_sample_top_p_min_p(const float* logits, uint32_t vocabulary_size,
                            float temperature, int top_k, float top_p,
                            float min_p, uint64_t* rng_state);

uint32_t ds41_rocm_graph_prefill_capacity(int context_size);
uint32_t ds41_rocm_graph_resume_prefill_min_tokens(void);
ds41_rocm_graph* ds41_rocm_graph_create(ds41_engine* engine, int context_size,
                                        uint32_t prefill_capacity);
void ds41_rocm_graph_destroy(ds41_rocm_graph* graph);
bool ds41_rocm_graph_reset(ds41_rocm_graph* graph);
bool ds41_rocm_graph_prefill(ds41_rocm_graph* graph, ds41_engine* engine,
                             const ds41_tokens* prompt, float* logits);
bool ds41_rocm_graph_prefill_range(ds41_rocm_graph* graph, ds41_engine* engine,
                                   const ds41_tokens* prompt, uint32_t start,
                                   uint32_t token_count, float* logits);
bool ds41_rocm_graph_eval(ds41_rocm_graph* graph, ds41_engine* engine,
                          int token, uint32_t position, float* logits);
bool ds41_rocm_graph_eval_batch(ds41_engine* engine,
                                const ds41_rocm_batch_item* items,
                                size_t item_count);
/* DSpark speculative verification.
 *
 * ds41_rocm_graph_verify_suffix scores `n_tokens` candidates in one batched
 * pass and returns the target's greedy continuation after each candidate row.
 * The caller decides how many rows to commit and must restore the saved
 * frontier whenever it commits fewer rows than it verified. */
bool ds41_rocm_graph_spec_prepare(ds41_rocm_graph* graph, ds41_engine* engine,
                                  uint32_t rows_cap);
uint32_t ds41_rocm_graph_spec_rows_cap(const ds41_rocm_graph* graph);
bool ds41_rocm_graph_spec_frontier_save(ds41_rocm_graph* graph);
bool ds41_rocm_graph_spec_frontier_restore(ds41_rocm_graph* graph);
bool ds41_rocm_graph_spec_frontier_commit_prefix(ds41_rocm_graph* graph,
                                                 uint32_t prefix_len);
bool ds41_rocm_graph_verify_suffix(ds41_rocm_graph* graph, ds41_engine* engine,
                                   const ds41_tokens* tokens, uint32_t start,
                                   uint32_t n_tokens, int32_t* row_tops);
bool ds41_rocm_graph_verify_batch(ds41_engine* engine,
                                  const ds41_rocm_verify_item* items,
                                  size_t item_count);
bool ds41_rocm_graph_read_spec_logits_row(const ds41_rocm_graph* graph,
                                          uint32_t row, float* logits);

/* DSpark drafting.
 *
 * Attaching a support model allocates the drafter's rings and enables target
 * feature capture. `_inject` turns captured features into ring rows for the
 * given absolute positions; `_draft` proposes one block for [pos0, pos0 +
 * block_size).
 */
bool ds41_rocm_graph_dspark_attach(ds41_rocm_graph* graph, ds41_engine* engine,
                                   const ds41_dspark_model* dspark);
uint32_t ds41_rocm_graph_dspark_block_size(const ds41_rocm_graph* graph);
void ds41_rocm_graph_dspark_set_capture_enabled(ds41_rocm_graph* graph,
                                                bool enabled);
bool ds41_rocm_graph_dspark_capture_ready(const ds41_rocm_graph* graph);
void ds41_rocm_graph_dspark_capture_reset(ds41_rocm_graph* graph);
uint32_t ds41_rocm_graph_dspark_batch_capture_rows(const ds41_rocm_graph* graph,
                                                   uint32_t* start);
bool ds41_rocm_graph_dspark_inject(ds41_rocm_graph* graph, uint32_t pos0,
                                   uint32_t n_rows);
uint32_t ds41_rocm_graph_dspark_context_len(const ds41_rocm_graph* graph);
void ds41_rocm_graph_dspark_truncate_context(ds41_rocm_graph* graph,
                                             uint32_t length);
bool ds41_rocm_graph_dspark_draft(ds41_rocm_graph* graph, ds41_engine* engine,
                                  int target_next_token, uint32_t pos0,
                                  int32_t* tokens_out, uint32_t* n_out,
                                  const ds41_dspark_sampler* sampler = nullptr,
                                  uint32_t max_tokens = UINT32_MAX,
                                  size_t concurrency = 1);
bool ds41_rocm_graph_dspark_draft_head_batch(
    ds41_engine* engine, const ds41_rocm_dspark_draft_item* items,
    size_t item_count);

uint64_t ds41_rocm_graph_snapshot_bytes(const ds41_rocm_graph* graph,
                                        const ds41_tokens* checkpoint);
int ds41_rocm_graph_save_snapshot(const ds41_rocm_graph* graph,
                                  const ds41_tokens* checkpoint,
                                  const float* logits,
                                  uint32_t prefill_capacity, int context_size,
                                  const ds41_dspark_request_state* state,
                                  ds41_session_snapshot* snapshot, char* error,
                                  size_t error_capacity);
int ds41_rocm_graph_load_snapshot(ds41_rocm_graph* graph,
                                  ds41_tokens* checkpoint, float* logits,
                                  uint32_t prefill_capacity, int context_size,
                                  ds41_dspark_request_state* state,
                                  const ds41_session_snapshot* snapshot,
                                  char* error, size_t error_capacity);

#endif  // GUFO_MODELS_DEEPSEEK_V41_FLASH_RUNTIME_NATIVE_INTERNAL_H_
