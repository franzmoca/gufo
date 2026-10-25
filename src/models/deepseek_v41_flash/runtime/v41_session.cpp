/* DeepSeek V4.1 Flash request sessions over the single-token decoder.
 *
 * A session owns one decoder and the tokens it has evaluated. Sync reuses
 * the evaluated prefix when the prompt extends it and otherwise starts over;
 * snapshots carry the tokens, frontier logits and decoder state. Speculative
 * decoding is not available for this model yet. */
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "model.h"
#include "native_internal.h"

int ds41_sample_argmax(const float* logits, uint32_t vocabulary_size);
int ds41_sample_top_p_min_p(const float* logits, uint32_t vocabulary_size, float temperature, int top_k,
                            float top_p, float min_p, uint64_t* rng_state);

struct ds41_session {
  ds41_engine* engine = nullptr;
  ds41_decoder* decoder = nullptr;
  int context = 0;
  std::vector<int> tokens;
  std::vector<float> logits;
  bool valid = false;
  ds41_session_cancel_fn cancel = nullptr;
  void* cancel_data = nullptr;
};

namespace {

constexpr uint64_t kPayloadMagic = 0x3153534531345344ull;  // "DS41ESS1"

void set_error(char* error, size_t capacity, const char* message) {
  if (error && capacity) snprintf(error, capacity, "%s", message);
}

uint32_t vocab(const ds41_session* s) { return (uint32_t)ds41_engine_vocab_size(s->engine); }

bool cancelled(const ds41_session* s) { return s->cancel && s->cancel(s->cancel_data); }

bool step(ds41_session* s, int token, char* error, size_t capacity) {
  if ((int)s->tokens.size() + 1 >= s->context) {
    set_error(error, capacity, "context is full");
    return false;
  }
  if (ds41_decoder_step(s->decoder, token, s->logits.data(), nullptr, nullptr) != 0) {
    set_error(error, capacity, "ROCm decode step failed");
    s->valid = false;
    return false;
  }
  s->tokens.push_back(token);
  s->valid = true;
  return true;
}

}  // namespace

int ds41_session_create(ds41_session** out, ds41_engine* engine, int context_size, bool use_dspark) {
  if (!out) return 1;
  *out = nullptr;
  if (!engine || context_size <= 1 || use_dspark) return 1;
  auto* s = new ds41_session();
  s->engine = engine;
  s->context = context_size;
  s->decoder = ds41_decoder_create(engine, (uint32_t)context_size);
  if (!s->decoder) {
    delete s;
    return 1;
  }
  s->logits.resize(vocab(s));
  *out = s;
  return 0;
}

void ds41_session_free(ds41_session* s) {
  if (!s) return;
  ds41_decoder_destroy(s->decoder);
  delete s;
}

bool ds41_session_dspark_enabled(const ds41_session*) { return false; }

int ds41_session_sync(ds41_session* s, const ds41_tokens* prompt, char* error, size_t capacity) {
  if (!s || !prompt || !prompt->v || prompt->len <= 0 || prompt->len >= s->context) {
    set_error(error, capacity, "prompt exceeds context");
    return 1;
  }
  const int n_vocab = (int)vocab(s);
  if (std::any_of(prompt->v, prompt->v + prompt->len, [n_vocab](int t) { return t < 0 || t >= n_vocab; })) {
    set_error(error, capacity, "prompt contains an invalid token ID");
    return 1;
  }
  const size_t n = (size_t)prompt->len;
  const bool extends = s->valid && s->tokens.size() <= n && std::equal(s->tokens.begin(), s->tokens.end(), prompt->v);
  if (!extends || s->tokens.size() == n) {
    if (extends) return 0; /* already at this prompt; its logits are current */
    if (ds41_decoder_reset(s->decoder) != 0) {
      set_error(error, capacity, "ROCm decoder reset failed");
      return 1;
    }
    s->tokens.clear();
    s->valid = false;
  }
  for (size_t i = s->tokens.size(); i < n; i++) {
    if (cancelled(s)) {
      set_error(error, capacity, "prefill cancelled");
      return DS41_SESSION_SYNC_INTERRUPTED;
    }
    if (!step(s, prompt->v[i], error, capacity)) return 1;
  }
  return 0;
}

int ds41_session_argmax(const ds41_session* s) {
  if (!s || !s->valid) return -1;
  return ds41_sample_argmax(s->logits.data(), vocab(s));
}

int ds41_session_sample(const ds41_session* s, float temperature, int top_k, float top_p, float min_p,
                        uint64_t* rng_state) {
  if (!s || !s->valid || !rng_state) return -1;
  return ds41_sample_top_p_min_p(s->logits.data(), vocab(s), temperature, top_k, top_p, min_p, rng_state);
}

int ds41_session_eval(ds41_session* s, int token, char* error, size_t capacity) {
  if (!s || token < 0 || token >= (int)vocab(s)) {
    set_error(error, capacity, "invalid token");
    return 1;
  }
  return step(s, token, error, capacity) ? 0 : 1;
}

int ds41_sessions_eval_batch(const ds41_session_batch_item* items, size_t count, char* error, size_t capacity) {
  for (size_t i = 0; i < count; i++) {
    if (ds41_session_eval(items[i].session, items[i].token, error, capacity) != 0) return 1;
  }
  return 0;
}

void ds41_session_begin_request(ds41_session*) {}
void ds41_session_prepare_batch_execution(ds41_session*) {}

int ds41_sessions_dspark_step_batch(const ds41_session_dspark_batch_item*, size_t, char* error, size_t capacity) {
  set_error(error, capacity, "DSpark is not available for DeepSeek V4.1 Flash");
  return 1;
}

int ds41_session_dspark_take_pending_anchor(ds41_session*) { return -1; }

void ds41_session_dspark_stats(const ds41_session* s, uint64_t* drafted, uint64_t* accepted,
                               uint64_t* support_drafted, uint64_t* support_accepted, uint64_t* positional_accepted,
                               uint64_t* anchors, uint64_t* full_blocks, uint64_t* steps, uint64_t* skipped,
                               uint32_t* context_tokens) {
  for (uint64_t* v : {drafted, accepted, support_drafted, support_accepted, positional_accepted, anchors,
                      full_blocks, steps, skipped})
    if (v) *v = 0;
  if (context_tokens) *context_tokens = s ? (uint32_t)s->tokens.size() : 0;
}

int ds41_session_copy_logits(const ds41_session* s, float* out, int capacity) {
  if (!s || !s->valid || !out || capacity < (int)vocab(s)) return 0;
  memcpy(out, s->logits.data(), s->logits.size() * sizeof(float));
  return (int)vocab(s);
}

int ds41_session_save_snapshot(const ds41_session* s, ds41_session_snapshot* snapshot, char* error,
                               size_t capacity) {
  if (!s || !snapshot || !s->valid) {
    set_error(error, capacity, "no evaluated state to save");
    return 1;
  }
  std::vector<uint8_t> payload;
  const uint64_t header[3] = {kPayloadMagic, (uint64_t)s->tokens.size(), (uint64_t)s->logits.size()};
  auto append = [&](const void* p, size_t bytes) {
    payload.insert(payload.end(), (const uint8_t*)p, (const uint8_t*)p + bytes);
  };
  append(header, sizeof(header));
  append(s->tokens.data(), s->tokens.size() * sizeof(int));
  append(s->logits.data(), s->logits.size() * sizeof(float));
  if (ds41_decoder_save(s->decoder, &payload) != 0) {
    set_error(error, capacity, "ROCm state copy failed");
    return 1;
  }
  snapshot->ptr = (uint8_t*)malloc(payload.size());
  if (!snapshot->ptr) {
    set_error(error, capacity, "out of memory");
    return 1;
  }
  memcpy(snapshot->ptr, payload.data(), payload.size());
  snapshot->len = snapshot->cap = payload.size();
  return 0;
}

int ds41_session_load_snapshot(ds41_session* s, const ds41_session_snapshot* snapshot, char* error,
                               size_t capacity) {
  uint64_t header[3];
  if (!s || !snapshot || !snapshot->ptr || snapshot->len < sizeof(header)) {
    set_error(error, capacity, "invalid snapshot");
    return 1;
  }
  memcpy(header, snapshot->ptr, sizeof(header));
  const uint64_t fixed = sizeof(header) + header[1] * sizeof(int) + header[2] * sizeof(float);
  if (header[0] != kPayloadMagic || header[2] != vocab(s) || header[1] == 0 || header[1] >= (uint64_t)s->context ||
      snapshot->len < fixed) {
    set_error(error, capacity, "snapshot does not match this session");
    return 1;
  }
  uint64_t consumed = 0;
  if (ds41_decoder_load(s->decoder, snapshot->ptr + fixed, snapshot->len - fixed, &consumed) != 0 ||
      fixed + consumed != snapshot->len || ds41_decoder_pos(s->decoder) != header[1]) {
    s->valid = false;
    s->tokens.clear();
    (void)ds41_decoder_reset(s->decoder);
    set_error(error, capacity, "snapshot state does not match this session");
    return 1;
  }
  const uint8_t* at = snapshot->ptr + sizeof(header);
  s->tokens.assign((const int*)at, (const int*)at + header[1]);
  memcpy(s->logits.data(), at + header[1] * sizeof(int), header[2] * sizeof(float));
  s->valid = true;
  return 0;
}

void ds41_session_snapshot_free(ds41_session_snapshot* snapshot) {
  if (!snapshot) return;
  free(snapshot->ptr);
  *snapshot = ds41_session_snapshot{};
}

void ds41_session_set_cancel(ds41_session* s, ds41_session_cancel_fn callback, void* user_data) {
  if (!s) return;
  s->cancel = callback;
  s->cancel_data = user_data;
}

void ds41_session_invalidate(ds41_session* s) {
  if (!s) return;
  s->valid = false;
  s->tokens.clear();
}

int ds41_session_pos(const ds41_session* s) { return s ? (int)s->tokens.size() : 0; }
int ds41_session_ctx(const ds41_session* s) { return s ? s->context : 0; }
uint32_t ds41_session_prefill_capacity(const ds41_session* s) { return s ? (uint32_t)s->context : 0; }

uint64_t ds41_session_payload_bytes(const ds41_session* s) {
  if (!s || !s->valid) return 0;
  return 3 * sizeof(uint64_t) + s->tokens.size() * sizeof(int) + s->logits.size() * sizeof(float) +
         ds41_decoder_state_bytes(s->decoder);
}
