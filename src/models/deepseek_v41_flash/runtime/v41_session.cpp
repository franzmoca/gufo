/* DeepSeek V4.1 Flash request sessions over the decoder.
 *
 * A session owns one decoder and the tokens it has evaluated. Sync reuses
 * the evaluated prefix when the prompt extends it and otherwise starts over,
 * evaluating the rest in decoder-sized batches; snapshots carry the tokens,
 * frontier logits and decoder state. With a DSpark drafter, a step drafts a
 * block after the anchor, verifies it in one speculative evaluation and keeps
 * the accepted prefix (ds41_decoder_commit); the drafter's rings follow every
 * evaluated position. */
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "model.h"
#include "native_internal.h"
#include "v41_dspark.h"

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
  ds41_session_write_fn before_write = nullptr;
  void* write_data = nullptr;
  ds41_dspark_state* dspark = nullptr;
  int pending = -1;  // a sampled token a rejected row drew, emitted next
  static constexpr float kAcceptPrior = 0.7f;
  /* Plain steps before a one-draft cycle re-measures acceptance: from the
   * first, doubling while the policy keeps returning to plain steps. */
  static constexpr uint32_t kProbeFirst = 4, kProbeMax = 64;
  /* Draft length policy: drafts per cycle maximize the expected emitted
   * tokens over the cycle's time. accept[k] estimates P(draft k + 1 is
   * accepted | the drafts before it were); cycle_ms[w] is a cycle of w drafts
   * (draft plus verification), cycle_ms[0] a plain step. Both start from
   * gfx1151 measurements at 2K context and follow the session; estimates of
   * positions past the drafted ones relax toward the prior, so longer drafts
   * are retried. */
  struct {
    float accept[DS41_DSPARK_BLOCK] = {kAcceptPrior, kAcceptPrior, kAcceptPrior, kAcceptPrior, kAcceptPrior};
    float cycle_ms[DS41_DSPARK_BLOCK + 1] = {42.4f, 69.6f, 83.6f, 95.5f, 106.1f, 117.2f};
    uint32_t plain = 0;   // plain steps since the last cycle
    uint32_t probe = kProbeFirst;  // plain steps before a one-draft cycle re-measures
  } policy;
  std::vector<float> spec_logits;
  struct {
    uint64_t drafted = 0, accepted = 0, support_drafted = 0, support_accepted = 0, positional = 0;
    uint64_t anchors = 0, full_blocks = 0, steps = 0;
  } stats;
};

namespace {

constexpr uint64_t kPayloadMagic = 0x3153534531345344ull;  // "DS41ESS1"

void set_error(char* error, size_t capacity, const char* message) {
  if (error && capacity) snprintf(error, capacity, "%s", message);
}

uint32_t vocab(const ds41_session* s) { return (uint32_t)ds41_engine_vocab_size(s->engine); }

bool cancelled(const ds41_session* s) { return s->cancel && s->cancel(s->cancel_data); }

/* Evaluates n tokens; the frontier logits are those of the last one. */
bool eval(ds41_session* s, const int* tokens, uint32_t n, char* error, size_t capacity) {
  if ((int)(s->tokens.size() + n) >= s->context) {
    set_error(error, capacity, "context is full");
    return false;
  }
  const uint32_t first = (uint32_t)s->tokens.size();
  if (s->before_write) s->before_write(s->write_data, first, first + n);
  if (ds41_decoder_eval(s->decoder, tokens, n, s->logits.data(), 1, nullptr, nullptr) != 0) {
    set_error(error, capacity, "ROCm evaluation failed");
    s->valid = false;
    return false;
  }
  s->tokens.insert(s->tokens.end(), tokens, tokens + n);
  s->valid = true;
  /* The drafter keeps the newest rows' main KV. */
  const uint32_t rows = std::min(n, DS41_DSPARK_FEATURE_ROWS);
  if (s->dspark && !ds41_dspark_inject(s->dspark, first + n - rows, rows)) {
    set_error(error, capacity, "DSpark feature injection failed");
    s->valid = false;
    return false;
  }
  return true;
}

}  // namespace

int ds41_session_create(ds41_session** out, ds41_engine* engine, int context_size, bool use_dspark) {
  if (!out) return 1;
  *out = nullptr;
  if (!engine || context_size <= 1 || (use_dspark && !ds41_engine_has_dspark(engine))) return 1;
  auto* s = new ds41_session();
  s->engine = engine;
  s->context = context_size;
  s->decoder = ds41_decoder_create(engine, (uint32_t)context_size);
  if (s->decoder && use_dspark) s->dspark = ds41_dspark_state_create(engine, s->decoder);
  if (!s->decoder || (use_dspark && !s->dspark)) {
    ds41_decoder_destroy(s->decoder);
    delete s;
    return 1;
  }
  s->logits.resize(vocab(s));
  if (s->dspark) s->spec_logits.resize((uint64_t)(DS41_DSPARK_BLOCK + 1) * vocab(s));
  *out = s;
  return 0;
}

void ds41_session_free(ds41_session* s) {
  if (!s) return;
  ds41_dspark_state_destroy(s->dspark);
  ds41_decoder_destroy(s->decoder);
  delete s;
}

bool ds41_session_dspark_enabled(const ds41_session* s) { return s && s->dspark; }

void* ds41_session_dspark_ring(ds41_session* s, uint64_t* bytes) {
  *bytes = 0;
  return s && s->dspark ? ds41_dspark_state_ring(s->dspark, bytes) : nullptr;
}

int ds41_session_dspark_range(const ds41_session* s, uint32_t* start, uint32_t* end) {
  if (!s || !s->dspark) return 1;
  ds41_dspark_state_range(s->dspark, start, end);
  return 0;
}

int ds41_session_dspark_set_range(ds41_session* s, uint32_t start, uint32_t end) {
  return s && s->dspark && s->valid && end == s->tokens.size() && ds41_dspark_state_set_range(s->dspark, start, end)
             ? 0
             : 1;
}

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
    s->pending = -1;
    if (s->dspark) ds41_dspark_state_reset(s->dspark, 0);
  }
  const size_t batch = ds41_decoder_batch_capacity(s->decoder);
  for (size_t i = s->tokens.size(); i < n;) {
    if (cancelled(s)) {
      set_error(error, capacity, "prefill cancelled");
      return DS41_SESSION_SYNC_INTERRUPTED;
    }
    const uint32_t count = (uint32_t)std::min(batch, n - i);
    if (!eval(s, prompt->v + i, count, error, capacity)) return 1;
    i += count;
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
  return eval(s, &token, 1, error, capacity) ? 0 : 1;
}

int ds41_sessions_eval_batch(const ds41_session_batch_item* items, size_t count, char* error, size_t capacity) {
  for (size_t i = 0; i < count; i++) {
    if (ds41_session_eval(items[i].session, items[i].token, error, capacity) != 0) return 1;
  }
  return 0;
}

/* Acceptance depends on the request (content, sampling); cycle times do not. */
void ds41_session_begin_request(ds41_session* s) {
  if (!s) return;
  std::fill(std::begin(s->policy.accept), std::end(s->policy.accept), ds41_session::kAcceptPrior);
  s->policy.plain = 0;
  s->policy.probe = ds41_session::kProbeFirst;
}
void ds41_session_prepare_batch_execution(ds41_session*) {}

namespace {

constexpr float kPolicyRate = 0.1f, kPolicyRelax = 0.02f;

uint32_t draft_width(ds41_session* s, uint32_t limit) {
  auto& p = s->policy;
  uint32_t best_w = 0;
  float best = 1.0f / p.cycle_ms[0], expected = 1.0f, reach = 1.0f;
  for (uint32_t w = 1; w <= limit; w++) {
    reach *= p.accept[w - 1];
    expected += reach;
    if (expected / p.cycle_ms[w] > best) {
      best = expected / p.cycle_ms[w];
      best_w = w;
    }
  }
  if (best_w > 0) {
    p.probe = ds41_session::kProbeFirst;
  } else if (limit > 0 && ++p.plain >= p.probe) {
    best_w = 1;
    p.probe = std::min(2 * p.probe, ds41_session::kProbeMax);
  }
  if (best_w > 0) p.plain = 0;
  return best_w;
}

void ema(float& v, float x, float rate = kPolicyRate) { v += rate * (x - v); }

/* The token that starts the next block: the frontier argmax (greedy), the
 * token a rejected row already drew, or a draw from the frontier logits. */
int dspark_anchor(ds41_session* s, const ds41_dspark_sampler* sampler) {
  if (!sampler) return ds41_sample_argmax(s->logits.data(), vocab(s));
  const int pending = s->pending;
  s->pending = -1;
  if (pending >= 0) {
    if (sampler->accept) sampler->accept(sampler->ctx, pending);
    return pending;
  }
  return sampler->sample(sampler->ctx, s->logits.data(), vocab(s));
}

/* One cycle: emits the anchor and the accepted drafts. Drafts are the
 * drafter's greedy path; a sampled request draws every verified row with its
 * own sampler (point-mass proposals), so the output distribution is exactly
 * the target's. */
int dspark_step(const ds41_session_dspark_batch_item& item, char* error, size_t capacity) {
  ds41_session* s = item.session;
  const ds41_dspark_sampler* sampler = item.sampler;
  if (!s || !s->dspark || !s->valid || !item.emitted || item.emitted_cap < 1 || !item.n_emitted ||
      (sampler && !sampler->sample)) {
    set_error(error, capacity, "invalid DSpark step request");
    return 1;
  }
  *item.n_emitted = 0;
  if ((int)s->tokens.size() + 1 >= s->context) {
    set_error(error, capacity, "context is full");
    return 1;
  }
  const uint32_t n_vocab = vocab(s);
  const int anchor = dspark_anchor(s, sampler);
  if (anchor < 0 || anchor >= (int)n_vocab) {
    set_error(error, capacity, "DSpark target logits contain no finite values");
    s->valid = false;
    return 1;
  }
  int* out = item.emitted;
  if (item.stop_at_eos && ds41_token_is_stop(s->engine, anchor)) {
    out[(*item.n_emitted)++] = anchor;
    return 0;
  }
  const uint32_t first = (uint32_t)s->tokens.size();
  const uint32_t room = (uint32_t)(s->context - (int)first - 2);
  const uint32_t limit =
      std::min({(uint32_t)DS41_DSPARK_BLOCK, item.max_draft_tokens, (uint32_t)item.emitted_cap - 1, room});
  const uint32_t tail = draft_width(s, limit);
  int32_t drafts[DS41_DSPARK_BLOCK];
  const bool trace = getenv("GUFO_DEEPSEEK41_DSPARK_TRACE") != nullptr;
  const auto ms = [](auto a, auto b) { return (float)std::chrono::duration<double, std::milli>(b - a).count(); };
  const auto t0 = std::chrono::steady_clock::now();
  if (tail == 0 || ds41_dspark_state_end(s->dspark) != first || !ds41_dspark_draft(s->dspark, anchor, drafts)) {
    if (!eval(s, &anchor, 1, error, capacity)) return 1;
    if (tail == 0) ema(s->policy.cycle_ms[0], ms(t0, std::chrono::steady_clock::now()));
    out[(*item.n_emitted)++] = anchor;
    return 0;
  }
  int block[DS41_DSPARK_BLOCK + 1];
  block[0] = anchor;
  std::copy(drafts, drafts + tail, block + 1);
  const uint32_t m = tail + 1;
  if (s->before_write) s->before_write(s->write_data, first, first + m);
  const auto t1 = std::chrono::steady_clock::now();
  float* rows = s->spec_logits.data();
  if (ds41_decoder_eval_speculative(s->decoder, block, m, rows) != 0) {
    set_error(error, capacity, "DSpark verification failed");
    s->valid = false;
    return 1;
  }
  /* Row i - 1 verifies block[i]; the first mismatch ends the accepted prefix. */
  const int stop = item.stop_at_eos ? ds41_token_eos(s->engine) : -1;
  uint32_t accepted = 1;
  uint32_t positional = 0;
  bool mismatch = false;
  for (uint32_t i = 1; i < m; i++) {
    const float* row = rows + (uint64_t)(i - 1) * n_vocab;
    const int top = ds41_sample_argmax(row, n_vocab);
    positional += top == block[i];
    if (mismatch) continue;
    int drawn = top;
    if (sampler && !(sampler->try_accept_argmax && top >= 0 && sampler->try_accept_argmax(sampler->ctx, top)))
      drawn = sampler->sample(sampler->ctx, row, n_vocab);
    if (drawn != block[i] || (item.stop_at_eos && ds41_token_is_stop(s->engine, drawn)) || block[i] == stop) {
      /* A sampled draw is the request's next token; greedy re-takes the frontier argmax. */
      if (sampler) s->pending = drawn;
      mismatch = true;
      continue;
    }
    accepted++;
  }
  if (ds41_decoder_commit(s->decoder, accepted) != 0) {
    set_error(error, capacity, "DSpark commit failed");
    s->valid = false;
    return 1;
  }
  s->tokens.insert(s->tokens.end(), block, block + accepted);
  memcpy(s->logits.data(), rows + (uint64_t)(accepted - 1) * n_vocab, n_vocab * sizeof(float));
  if (!ds41_dspark_inject(s->dspark, first, accepted)) {
    set_error(error, capacity, "DSpark feature injection failed");
    s->valid = false;
    return 1;
  }
  for (uint32_t i = 0; i < accepted; i++) out[(*item.n_emitted)++] = block[i];
  auto& st = s->stats;
  st.drafted += m;
  st.accepted += accepted;
  st.support_drafted += tail;
  st.support_accepted += accepted - 1;
  st.positional += positional;
  st.anchors++;
  st.full_blocks += accepted == m;
  st.steps++;
  /* Drafts after the first rejection say nothing about their acceptance. */
  auto& p = s->policy;
  for (uint32_t k = 0; k < std::min(accepted, tail); k++) ema(p.accept[k], k + 1 < accepted ? 1.0f : 0.0f);
  for (uint32_t k = tail; k < DS41_DSPARK_BLOCK; k++) ema(p.accept[k], ds41_session::kAcceptPrior, kPolicyRelax);
  const auto t2 = std::chrono::steady_clock::now();
  ema(p.cycle_ms[tail], ms(t0, t2));
  if (trace)
    fprintf(stderr, "ds41: DSpark pos=%u drafted=%u accepted=%u draft=%.1fms verify=%.1fms\n", first, tail,
            accepted - 1, ms(t0, t1), ms(t1, t2));
  return 0;
}

}  // namespace

int ds41_sessions_dspark_step_batch(const ds41_session_dspark_batch_item* items, size_t count, char* error,
                                    size_t capacity) {
  for (size_t i = 0; i < count; i++)
    if (dspark_step(items[i], error, capacity) != 0) return 1;
  return 0;
}

int ds41_session_dspark_take_pending_anchor(ds41_session* s) {
  if (!s) return -1;
  const int pending = s->pending;
  s->pending = -1;
  return pending;
}

void ds41_session_dspark_stats(const ds41_session* s, uint64_t* drafted, uint64_t* accepted,
                               uint64_t* support_drafted, uint64_t* support_accepted, uint64_t* positional_accepted,
                               uint64_t* anchors, uint64_t* full_blocks, uint64_t* steps, uint64_t* skipped,
                               uint32_t* context_tokens) {
  const auto set = [](uint64_t* v, uint64_t x) {
    if (v) *v = x;
  };
  const bool on = s && s->dspark;
  set(drafted, on ? s->stats.drafted : 0);
  set(accepted, on ? s->stats.accepted : 0);
  set(support_drafted, on ? s->stats.support_drafted : 0);
  set(support_accepted, on ? s->stats.support_accepted : 0);
  set(positional_accepted, on ? s->stats.positional : 0);
  set(anchors, on ? s->stats.anchors : 0);
  set(full_blocks, on ? s->stats.full_blocks : 0);
  set(steps, on ? s->stats.steps : 0);
  set(skipped, 0);
  if (context_tokens)
    *context_tokens = !s ? 0 : on ? ds41_dspark_state_end(s->dspark) : (uint32_t)s->tokens.size();
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
  if (ds41_decoder_save(s->decoder, &payload) != 0 || (s->dspark && !ds41_dspark_state_save(s->dspark, &payload))) {
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
  uint64_t consumed = 0, drafter = 0;
  const bool loaded = ds41_decoder_load(s->decoder, snapshot->ptr + fixed, snapshot->len - fixed, &consumed) == 0;
  if (loaded && s->dspark &&
      !ds41_dspark_state_load(s->dspark, snapshot->ptr + fixed + consumed, snapshot->len - fixed - consumed, &drafter))
    drafter = UINT64_MAX;
  if (!loaded || drafter == UINT64_MAX || fixed + consumed + drafter != snapshot->len ||
      ds41_decoder_pos(s->decoder) != header[1]) {
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
  s->pending = -1;
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
  s->pending = -1;
  if (s->dspark) ds41_dspark_state_reset(s->dspark, 0);
}

void ds41_session_set_write_hook(ds41_session* s, ds41_session_write_fn callback, void* user_data) {
  if (!s) return;
  s->before_write = callback;
  s->write_data = user_data;
}

ds41_decoder* ds41_session_decoder(ds41_session* s) { return s ? s->decoder : nullptr; }

bool ds41_session_valid(const ds41_session* s) { return s && s->valid; }

const int* ds41_session_tokens(const ds41_session* s) { return s ? s->tokens.data() : nullptr; }

float* ds41_session_logits(ds41_session* s) { return s ? s->logits.data() : nullptr; }

int* ds41_session_begin_restore(ds41_session* s, uint32_t n) {
  if (!s || n == 0 || n >= (uint32_t)s->context) return nullptr;
  s->valid = false;
  s->tokens.resize(n);
  return s->tokens.data();
}

int ds41_session_finish_restore(ds41_session* s) {
  if (!s || s->tokens.empty()) return 1;
  const int n_vocab = (int)vocab(s);
  if (std::any_of(s->tokens.begin(), s->tokens.end(), [n_vocab](int t) { return t < 0 || t >= n_vocab; }) ||
      ds41_decoder_adopt(s->decoder, s->tokens.data(), (uint32_t)s->tokens.size()) != 0)
    return 1;
  /* Restored state carries no drafter rows; they refill as tokens are evaluated. */
  if (s->dspark) ds41_dspark_state_reset(s->dspark, (uint32_t)s->tokens.size());
  s->pending = -1;
  s->valid = true;
  return 0;
}

int ds41_session_pos(const ds41_session* s) { return s ? (int)s->tokens.size() : 0; }
int ds41_session_ctx(const ds41_session* s) { return s ? s->context : 0; }
uint32_t ds41_session_prefill_capacity(const ds41_session* s) { return s ? (uint32_t)s->context : 0; }

uint64_t ds41_session_payload_bytes(const ds41_session* s) {
  if (!s || !s->valid) return 0;
  return 3 * sizeof(uint64_t) + s->tokens.size() * sizeof(int) + s->logits.size() * sizeof(float) +
         ds41_decoder_state_bytes(s->decoder) + (s->dspark ? ds41_dspark_state_bytes(s->dspark) : 0);
}
