#ifndef GUFO_MODELS_DEEPSEEK_V41_FLASH_RUNTIME_NATIVE_INTERNAL_H_
#define GUFO_MODELS_DEEPSEEK_V41_FLASH_RUNTIME_NATIVE_INTERNAL_H_

#include <cstddef>
#include <cstdint>

#include "../kernels/rocm/ds41_v41.h"
#include "model.h"
#include "model_data_internal.h"

struct ds41_dspark_model;
struct ds41_engram_tables;
struct ds41_expert_store;
struct ds41_model;
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

struct ds41_engine {
  ds41_model* model;
  ds41_vocab* vocab;
  ds41_weights* weights;
  ds41_dspark_model* dspark;
  ds41_engram_tables* engram;
  ds41_expert_store* expert_store;
  char* expert_histogram;  // optional seed ranking beside the model (owned)
  bool rocm_ready;
  uint32_t prefill_batch;  // most tokens per prefill batch
  bool exact_prefill;      // no decoder SWA bounded replay
  float* image_bias;       // device [layer][expert], or NULL without vision
  /* Direction adapter edits per layer: [0] the attention output, [1] the
   * FFN output (shared expert, then the whole sum). k = 0 when none. */
  ds41_v41_steer steer[DS41_N_LAYER][2];
  float* steer_vectors;  // device, owns every steer[][] vector
  /* LoRA terms per layer and DS41_LORA_* target (rank 0 when none). */
  struct ds41_lora_site {
    const float* a;
    const float* b;
    uint32_t rank, in, out;
    float scale;
  } lora[DS41_N_LAYER][DS41_N_LORA_TARGETS];
  float* lora_buf;      // device, owns every lora[][] half
  float* lora_t;        // device scratch, lora_t_rows rows of lora_max_rank
  uint32_t lora_t_rows;
  uint32_t lora_max_rank;
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

#endif  // GUFO_MODELS_DEEPSEEK_V41_FLASH_RUNTIME_NATIVE_INTERNAL_H_
