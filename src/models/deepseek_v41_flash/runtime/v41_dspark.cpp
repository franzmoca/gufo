/* DeepSeek V4.1 Flash DSpark drafter; see v41_dspark.h. */
#include "v41_dspark.h"

#include <fcntl.h>
#include <hip/hip_runtime_api.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#include "ds41_v41.h"
#include "mmq/ds41_mmq.h"
#include "native_internal.h"
#include "tensor_types.h"

namespace {

constexpr uint32_t kHc = DS41_N_HC;
constexpr uint32_t kDim = DS41_N_EMBD;
constexpr uint32_t kHeadDim = DS41_N_HEAD_DIM;
constexpr uint32_t kRot = DS41_N_ROT;
constexpr uint32_t kQDim = DS41_N_HEAD * DS41_N_HEAD_DIM;
constexpr uint32_t kGroupDim = kQDim / DS41_N_OUT_GROUP;
constexpr uint32_t kMixes = 2 * kHc + kHc * kHc;
constexpr uint32_t kFeatureDim = DS41_DSPARK_TARGET_LAYERS * kDim;
constexpr uint32_t kRows = DS41_DSPARK_BLOCK;
constexpr uint32_t kRouted = DS41_DSPARK_EXPERTS_USED;
constexpr uint64_t kRingFloats = (uint64_t)DS41_N_SWA * kHeadDim;
constexpr float kSwigluLimit = DS41_SWIGLU_CLAMP_EXP;

float half_to_float(uint16_t h) {
  const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
  const uint32_t exp = (h >> 10) & 0x1fu;
  uint32_t mant = h & 0x3ffu;
  uint32_t bits;
  if (exp == 0) {
    if (mant == 0) {
      bits = sign;
    } else {
      int e = -1;
      do {
        e++;
        mant <<= 1;
      } while ((mant & 0x400u) == 0);
      bits = sign | (uint32_t)(127 - 15 - e) << 23 | (mant & 0x3ffu) << 13;
    }
  } else if (exp == 31) {
    bits = sign | 0x7f800000u | mant << 13;
  } else {
    bits = sign | (exp + 127 - 15) << 23 | mant << 13;
  }
  float f;
  memcpy(&f, &bits, sizeof(f));
  return f;
}

}  // namespace

/* Draft scratch, shared by the sessions of one engine (evaluations are serialized). */
struct ds41_dspark_scratch {
  std::vector<void*> owned;
  float *h = nullptr, *h2 = nullptr, *xn = nullptr, *mixes = nullptr, *pre[2] = {}, *post = nullptr;
  float *comb = nullptr, *ffn_post = nullptr, *ffn_comb = nullptr;
  float *qa = nullptr, *qr = nullptr, *q = nullptr, *kv = nullptr, *kvn = nullptr, *o = nullptr, *oa = nullptr;
  float *attn_out = nullptr, *router = nullptr, *route_w = nullptr;
  int32_t* route_ids = nullptr;
  float *g = nullptr, *u = nullptr, *mid = nullptr, *down = nullptr, *sg = nullptr, *su = nullptr, *sm = nullptr;
  float *sd = nullptr, *logits = nullptr, *markov_x = nullptr, *bias = nullptr;
  int32_t* tokens = nullptr;
  float *main_raw = nullptr, *main_x = nullptr, *main_kv = nullptr, *main_kvn = nullptr;
  ~ds41_dspark_scratch() {
    for (void* p : owned) (void)hipFree(p);
  }
  template <typename T>
  bool alloc(T** p, uint64_t count) {
    void* v = nullptr;
    if (hipMalloc(&v, count * sizeof(T)) != hipSuccess) return false;
    owned.push_back(v);
    *p = static_cast<T*>(v);
    return true;
  }
};

struct ds41_dspark_state {
  ds41_engine* engine = nullptr;
  ds41_decoder* decoder = nullptr;
  float* ring = nullptr;  // [stage][DS41_N_SWA][head_dim], slot pos % DS41_N_SWA
  uint32_t ring_start = 0, ring_end = 0;
  std::vector<float> host_rows;
};

namespace {

bool lin(const ds41_dspark_tensor& t, const float* x, float* out, uint32_t n) {
  return ds41_v41_linear(t.type, t.p, (uint32_t)t.rows, t.k, x, out, n);
}

const float* f32(const ds41_dspark_tensor& t) { return static_cast<const float*>(t.p); }

bool gemv_type(uint32_t type) { return type == DS41_TENSOR_Q5_K || type == DS41_TENSOR_Q8_0; }

bool alloc_scratch(ds41_dspark_scratch& s) {
  const uint64_t r = kRows;
  const uint64_t hd = (uint64_t)kHc * kDim;
  const uint64_t f = DS41_DSPARK_FEATURE_ROWS;
  return s.alloc(&s.h, r * hd) && s.alloc(&s.h2, r * hd) && s.alloc(&s.xn, r * kDim) &&
         s.alloc(&s.mixes, r * kMixes) && s.alloc(&s.pre[0], r * kHc) && s.alloc(&s.pre[1], r * kHc) &&
         s.alloc(&s.post, r * kHc) && s.alloc(&s.comb, r * kHc * kHc) && s.alloc(&s.ffn_post, r * kHc) &&
         s.alloc(&s.ffn_comb, r * kHc * kHc) && s.alloc(&s.qa, r * DS41_N_LORA_Q) &&
         s.alloc(&s.qr, r * DS41_N_LORA_Q) && s.alloc(&s.q, r * kQDim) && s.alloc(&s.kv, r * kHeadDim) &&
         s.alloc(&s.kvn, r * kHeadDim) && s.alloc(&s.o, r * kQDim) &&
         s.alloc(&s.oa, r * DS41_N_OUT_GROUP * DS41_N_LORA_O) && s.alloc(&s.attn_out, r * kDim) &&
         s.alloc(&s.router, r * DS41_DSPARK_EXPERTS) && s.alloc(&s.route_w, r * kRouted) &&
         s.alloc(&s.route_ids, r * kRouted) && s.alloc(&s.g, r * kRouted * DS41_N_FF_EXP) &&
         s.alloc(&s.u, r * kRouted * DS41_N_FF_EXP) && s.alloc(&s.mid, r * kRouted * DS41_N_FF_EXP) &&
         s.alloc(&s.down, r * kRouted * kDim) && s.alloc(&s.sg, r * DS41_N_FF_EXP) &&
         s.alloc(&s.su, r * DS41_N_FF_EXP) && s.alloc(&s.sm, r * DS41_N_FF_EXP) && s.alloc(&s.sd, r * kDim) &&
         s.alloc(&s.logits, r * DS41_N_VOCAB) && s.alloc(&s.markov_x, DS41_DSPARK_MARKOV_RANK) &&
         s.alloc(&s.bias, DS41_N_VOCAB) && s.alloc(&s.tokens, r + 1) && s.alloc(&s.main_raw, f * kDim) &&
         s.alloc(&s.main_x, f * kDim) && s.alloc(&s.main_kv, f * kHeadDim) && s.alloc(&s.main_kvn, f * kHeadDim);
}

/* Stores rows of positions pos0 .. pos0 + n - 1 into a ring, slot pos % DS41_N_SWA. */
bool store_ring(float* ring, const float* rows, uint32_t pos0, uint32_t n) {
  const uint32_t first = n > DS41_N_SWA ? n - DS41_N_SWA : 0;
  for (uint32_t t = first; t < n;) {
    const uint32_t slot = (pos0 + t) % DS41_N_SWA;
    const uint32_t run = std::min(n - t, DS41_N_SWA - slot);
    if (hipMemcpyAsync(ring + (uint64_t)slot * kHeadDim, rows + (uint64_t)t * kHeadDim,
                       (uint64_t)run * kHeadDim * sizeof(float), hipMemcpyDeviceToDevice, 0) != hipSuccess)
      return false;
    t += run;
  }
  return true;
}

/* The DSparkAttention of one stage for the block rows in xn at positions pos .. pos + kRows - 1. */
bool attention(const ds41_dspark_stage& st, ds41_dspark_scratch& s, const float* ring, uint32_t first, uint32_t n_ring,
               uint32_t pos, const float* freq) {
  const uint32_t n = kRows;
  const float eps = DS41_RMS_EPS;
  if (!lin(st.q_a, s.xn, s.qa, n) || !lin(st.kv, s.xn, s.kv, n) ||
      ds41_v41_q_kv_norm(s.qa, f32(st.q_a_norm), s.qr, DS41_N_LORA_Q, s.kv, f32(st.kv_a_norm), s.kvn, kHeadDim,
                         kRot, freq, pos, n, eps) ||
      !lin(st.q_b, s.qr, s.q, n) ||
      ds41_v41_rope_tail(s.q, n * DS41_N_HEAD, kHeadDim, kRot, freq, pos, DS41_N_HEAD, 1, 0) ||
      ds41_v41_dspark_attention(s.q, ring, first, n_ring, DS41_N_SWA, s.kvn, n, f32(st.sinks), s.o, DS41_N_HEAD,
                                kHeadDim, 1.0f / sqrtf((float)kHeadDim)) ||
      ds41_v41_rope_fp8(s.o, n * DS41_N_HEAD, kHeadDim, kRot, freq, pos, DS41_N_HEAD, 1, 1))
    return false;
  /* Grouped LoRA output: group g maps head slice g of every row. */
  const uint32_t o_rows = DS41_N_OUT_GROUP * DS41_N_LORA_O;
  if (gemv_type(st.o_a.type)) {
    if (ds41_v41_gemv((int)st.o_a.type, st.o_a.p, s.o, s.oa, o_rows, kGroupDim, DS41_N_LORA_O, 0, n, kQDim, o_rows))
      return false;
  } else if (ds41_mmq_grouped_vec_typed((int)st.o_a.type, st.o_a.p, s.o, s.oa, DS41_N_LORA_O, (int)n, kGroupDim,
                                        DS41_N_OUT_GROUP, 0)) {
    return false;
  }
  return lin(st.o_b, s.oa, s.attn_out, n);
}

bool moe(const ds41_dspark_stage& st, ds41_dspark_scratch& s) {
  const uint32_t n = kRows;
  const int experts = (int)DS41_DSPARK_EXPERTS;
  if (!lin(st.gate_inp, s.xn, s.router, n) ||
      ds41_v41_route(s.router, f32(st.probs_b), s.route_ids, s.route_w, DS41_DSPARK_EXPERTS, kRouted,
                     DS41_EXPERT_WEIGHT_SCALE, n) ||
      ds41_mmq_moe_vec_typed((int)st.gate_exps.type, st.gate_exps.p, s.xn, s.route_ids, s.g, DS41_N_FF_EXP, kDim,
                             (int)n, experts, (int)kRouted, 0) ||
      ds41_mmq_moe_vec_typed((int)st.up_exps.type, st.up_exps.p, s.xn, s.route_ids, s.u, DS41_N_FF_EXP, kDim, (int)n,
                             experts, (int)kRouted, 0) ||
      ds41_v41_swiglu(s.g, s.u, s.route_w, s.mid, n * kRouted, DS41_N_FF_EXP, kSwigluLimit) ||
      ds41_mmq_moe_vec_typed((int)st.down_exps.type, st.down_exps.p, s.mid, s.route_ids, s.down, kDim, DS41_N_FF_EXP,
                             (int)(n * kRouted), experts, 1, 0))
    return false;
  if (st.gate_shexp.type == st.up_shexp.type && gemv_type(st.gate_shexp.type)) {
    if (ds41_v41_gemv_swiglu((int)st.gate_shexp.type, st.gate_shexp.p, st.up_shexp.p, s.xn, s.sm, DS41_N_FF_EXP,
                             kDim, kSwigluLimit, 0, n))
      return false;
  } else if (!lin(st.gate_shexp, s.xn, s.sg, n) || !lin(st.up_shexp, s.xn, s.su, n) ||
             ds41_v41_swiglu(s.sg, s.su, nullptr, s.sm, n, DS41_N_FF_EXP, kSwigluLimit)) {
    return false;
  }
  return lin(st.down_shexp, s.sm, s.sd, n);
}

bool hc_mix(const ds41_dspark_tensor& fn, const ds41_dspark_tensor& scale, const ds41_dspark_tensor& base,
            ds41_dspark_scratch& s, const float* h, float* pre, float* post, float* comb) {
  return ds41_v41_hc_mix(f32(fn), h, f32(scale), f32(base), s.mixes, pre, post, comb, kHc, kDim, DS41_RMS_EPS,
                         DS41_N_HC_SINKHORN_ITER, DS41_HC_EPS, kRows) == 0;
}

}  // namespace

bool ds41_dspark_upload(ds41_dspark_model* m) {
  /* F16 tensors widen to F32, except the Markov tables, which their gather and matmul read as F16. */
  const auto widen = [m](const ds41_dspark_source& src) {
    return src.src->type == DS41_TENSOR_F16 && src.dst != &m->markov_w1 && src.dst != &m->markov_w2;
  };
  constexpr uint64_t kAlign = 256;
  constexpr uint64_t kChunk = 64ull << 20;
  std::vector<uint64_t> offsets;
  uint64_t total = 0;
  for (const ds41_dspark_source& src : m->sources) {
    const uint64_t bytes = widen(src) ? src.src->elements * sizeof(float) : src.src->bytes;
    offsets.push_back(total);
    total += (bytes + kAlign - 1) / kAlign * kAlign;
  }
  if (hipMalloc(&m->device, total) != hipSuccess) {
    m->device = nullptr;
    return false;
  }
  m->device_bytes = total;
  void* staging = nullptr;
  if (hipHostMalloc(&staging, kChunk) != hipSuccess) return false;
  const int fd = m->file.shard_fd[0];
  bool ok = true;
  std::vector<uint16_t> half;
  std::vector<float> full;
  for (size_t i = 0; i < m->sources.size() && ok; i++) {
    const ds41_tensor* t = m->sources[i].src;
    ds41_dspark_tensor* dst = m->sources[i].dst;
    uint8_t* out = static_cast<uint8_t*>(m->device) + offsets[i];
    const off_t at = (off_t)(t->abs_offset - m->file.shard_offset[0]);
    if (widen(m->sources[i])) {
      half.resize(t->elements);
      full.resize(t->elements);
      ok = pread(fd, half.data(), t->bytes, at) == (ssize_t)t->bytes;
      for (uint64_t j = 0; ok && j < t->elements; j++) full[j] = half_to_float(half[j]);
      ok = ok && hipMemcpy(out, full.data(), t->elements * sizeof(float), hipMemcpyHostToDevice) == hipSuccess;
      dst->type = DS41_TENSOR_F32;
    } else {
      for (uint64_t done = 0; ok && done < t->bytes; done += kChunk) {
        const uint64_t bytes = std::min(kChunk, t->bytes - done);
        ok = pread(fd, staging, bytes, at + (off_t)done) == (ssize_t)bytes &&
             hipMemcpy(out + done, staging, bytes, hipMemcpyHostToDevice) == hipSuccess;
      }
    }
    (void)posix_fadvise(fd, at, (off_t)t->bytes, POSIX_FADV_DONTNEED);
    dst->p = out;
  }
  (void)hipHostFree(staging);
  if (ok) {
    m->scratch = new ds41_dspark_scratch();
    ok = alloc_scratch(*m->scratch);
  }
  return ok;
}

void ds41_dspark_free(ds41_dspark_model* m) {
  if (!m) return;
  delete m->scratch;
  m->scratch = nullptr;
  if (m->device) (void)hipFree(m->device);
  m->device = nullptr;
}

ds41_dspark_state* ds41_dspark_state_create(ds41_engine* engine, ds41_decoder* decoder) {
  if (!engine || !engine->dspark || !engine->dspark->device || !decoder || !ds41_decoder_enable_features(decoder))
    return nullptr;
  auto* s = new ds41_dspark_state();
  s->engine = engine;
  s->decoder = decoder;
  void* ring = nullptr;
  if (hipMalloc(&ring, DS41_DSPARK_STAGES * kRingFloats * sizeof(float)) != hipSuccess) {
    delete s;
    return nullptr;
  }
  s->ring = static_cast<float*>(ring);
  s->host_rows.resize((uint64_t)kRows * kDim);
  return s;
}

void ds41_dspark_state_destroy(ds41_dspark_state* s) {
  if (!s) return;
  (void)hipFree(s->ring);
  delete s;
}

void ds41_dspark_state_reset(ds41_dspark_state* s, uint32_t pos) {
  s->ring_start = s->ring_end = pos;
}

uint32_t ds41_dspark_state_end(const ds41_dspark_state* s) { return s->ring_end; }

uint64_t ds41_dspark_ring_bytes(void) { return DS41_DSPARK_STAGES * kRingFloats * sizeof(float); }

float* ds41_dspark_state_ring(ds41_dspark_state* s, uint64_t* bytes) {
  *bytes = ds41_dspark_ring_bytes();
  return s->ring;
}

void ds41_dspark_state_range(const ds41_dspark_state* s, uint32_t* start, uint32_t* end) {
  *start = s->ring_start;
  *end = s->ring_end;
}

bool ds41_dspark_state_set_range(ds41_dspark_state* s, uint32_t start, uint32_t end) {
  if (start > end) return false;
  s->ring_start = start;
  s->ring_end = end;
  return true;
}

bool ds41_dspark_inject(ds41_dspark_state* s, uint32_t pos0, uint32_t rows) {
  if (rows == 0) return true;
  uint32_t fpos0 = 0, frows = 0;
  const float* features = ds41_decoder_features(s->decoder, &fpos0, &frows);
  if (!features || rows > DS41_DSPARK_FEATURE_ROWS || pos0 < fpos0 || pos0 + rows > fpos0 + frows) return false;
  const ds41_dspark_model* m = s->engine->dspark;
  ds41_dspark_scratch& sc = *m->scratch;
  const float* freq = ds41_decoder_freq_plain(s->decoder);
  const float eps = DS41_RMS_EPS;
  /* main_x = main_norm(main_proj(main_hidden)); each stage's ring row is
   * kv_norm(wkv(main_x)), roped at its position and rounded to FP8. */
  if (!lin(m->fc, features + (uint64_t)(pos0 - fpos0) * kFeatureDim, sc.main_raw, rows) ||
      ds41_v41_rms_norm(sc.main_raw, f32(m->main_norm), sc.main_x, rows, kDim, eps))
    return false;
  for (uint32_t st = 0; st < DS41_DSPARK_STAGES; st++) {
    const ds41_dspark_stage& stage = m->stage[st];
    if (!lin(stage.kv, sc.main_x, sc.main_kv, rows) ||
        ds41_v41_rms_norm(sc.main_kv, f32(stage.kv_a_norm), sc.main_kvn, rows, kHeadDim, eps) ||
        ds41_v41_rope_fp8(sc.main_kvn, rows, kHeadDim, kRot, freq, pos0, 1, 1, 0) ||
        !store_ring(s->ring + st * kRingFloats, sc.main_kvn, pos0, rows))
      return false;
  }
  if (pos0 != s->ring_end) s->ring_start = pos0;
  s->ring_end = pos0 + rows;
  return hipDeviceSynchronize() == hipSuccess;
}

bool ds41_dspark_draft(ds41_dspark_state* s, int anchor, int32_t* drafts, uint32_t count) {
  if (count == 0 || count > kRows) return false;
  const ds41_dspark_model* m = s->engine->dspark;
  ds41_dspark_scratch& sc = *m->scratch;
  const uint32_t pos = s->ring_end;  // the anchor's position
  if (pos == s->ring_start) return false;
  const uint32_t first = std::max(s->ring_start, pos > DS41_N_SWA ? pos - DS41_N_SWA : 0u);
  const uint32_t n_ring = pos - first;
  const float* freq = ds41_decoder_freq_plain(s->decoder);
  const uint32_t n = kRows;
  const uint64_t hd = (uint64_t)kHc * kDim;

  /* Block rows [anchor, noise ...], replicated over the hc streams; the first stage's pre-mix is one-hot. */
  int tokens[kRows];
  tokens[0] = anchor;
  for (uint32_t t = 1; t < n; t++) tokens[t] = DS41_DSPARK_NOISE_TOKEN;
  float one_hot[kRows * kHc] = {};
  for (uint32_t t = 0; t < n; t++) one_hot[t * kHc] = 1.0f;
  bool ok = ds41_decoder_embed(s->decoder, tokens, n, s->host_rows.data()) &&
            hipMemcpy(sc.xn, s->host_rows.data(), (uint64_t)n * kDim * sizeof(float), hipMemcpyHostToDevice) ==
                hipSuccess &&
            hipMemcpy(sc.pre[0], one_hot, sizeof(one_hot), hipMemcpyHostToDevice) == hipSuccess &&
            hipMemcpy(sc.tokens, tokens, sizeof(int32_t), hipMemcpyHostToDevice) == hipSuccess;
  for (uint32_t c = 0; c < kHc && ok; c++)
    ok = hipMemcpy2DAsync(sc.h + c * kDim, hd * sizeof(float), sc.xn, kDim * sizeof(float), kDim * sizeof(float), n,
                          hipMemcpyDeviceToDevice, 0) == hipSuccess;

  /* Shifted mHC as in the decoder: attention takes pre[0] and leaves its mix
   * in pre[1] for the FFN, whose own mix returns to pre[0]. */
  for (uint32_t st = 0; st < DS41_DSPARK_STAGES && ok; st++) {
    const ds41_dspark_stage& l = m->stage[st];
    ok = hc_mix(l.hc_attn_fn, l.hc_attn_scale, l.hc_attn_base, sc, sc.h, sc.pre[1], sc.post, sc.comb) &&
         ds41_v41_hc_collapse_norm(sc.h, sc.pre[0], f32(l.attn_norm), sc.xn, kHc, kDim, DS41_RMS_EPS, n) == 0 &&
         attention(l, sc, s->ring + st * kRingFloats, first, n_ring, pos, freq) &&
         ds41_v41_hc_expand(sc.attn_out, sc.h, sc.post, sc.comb, sc.h2, kHc, kDim, n) == 0 &&
         hc_mix(l.hc_ffn_fn, l.hc_ffn_scale, l.hc_ffn_base, sc, sc.h2, sc.pre[0], sc.ffn_post, sc.ffn_comb) &&
         ds41_v41_hc_collapse_norm(sc.h2, sc.pre[1], f32(l.ffn_norm), sc.xn, kHc, kDim, DS41_RMS_EPS, n) == 0 &&
         moe(l, sc) &&
         ds41_v41_hc_expand_sum(sc.down, sc.sd, kRouted, sc.h2, sc.ffn_post, sc.ffn_comb, sc.h, kHc, kDim, n) == 0;
  }
  /* Target head, then the Markov path: row i adds W2 W1[token i] and takes the argmax as token i + 1. */
  ok = ok && ds41_v41_hc_collapse_norm(sc.h, sc.pre[0], f32(m->out_norm), sc.xn, kHc, kDim, DS41_RMS_EPS, n) == 0 &&
       ds41_decoder_output_head(s->decoder, sc.xn, sc.logits, count);
  const int format = m->markov_w1.type == DS41_TENSOR_BF16  ? DS41_V41_ROW_BF16
                     : m->markov_w1.type == DS41_TENSOR_F16 ? DS41_V41_ROW_F16
                                                            : DS41_V41_ROW_F32;
  // Later Markov rows depend only on earlier proposals; unused suffixes
  // cannot affect the requested prefix. The transformer body keeps all five rows.
  for (uint32_t i = 0; i < count && ok; i++)
    ok = ds41_v41_gather_row(m->markov_w1.p, format, sc.tokens, i, sc.markov_x, DS41_DSPARK_MARKOV_RANK) == 0 &&
         lin(m->markov_w2, sc.markov_x, sc.bias, 1) &&
         ds41_v41_argmax_bias(sc.logits + (uint64_t)i * DS41_N_VOCAB, sc.bias, DS41_N_VOCAB, sc.tokens, i + 1) == 0;
  ok = ok && hipMemcpy(drafts, sc.tokens + 1, count * sizeof(int32_t), hipMemcpyDeviceToHost) == hipSuccess;
  for (uint32_t i = 0; i < count && ok; i++) ok = drafts[i] >= 0;
  return ok;
}

uint64_t ds41_dspark_state_bytes(const ds41_dspark_state*) {
  return 2 * sizeof(uint64_t) + DS41_DSPARK_STAGES * kRingFloats * sizeof(float);
}

bool ds41_dspark_state_save(const ds41_dspark_state* s, std::vector<uint8_t>* out) {
  const uint64_t header[2] = {s->ring_start, s->ring_end};
  out->insert(out->end(), (const uint8_t*)header, (const uint8_t*)(header + 2));
  const size_t at = out->size();
  const uint64_t bytes = DS41_DSPARK_STAGES * kRingFloats * sizeof(float);
  out->resize(at + bytes);
  return hipMemcpy(out->data() + at, s->ring, bytes, hipMemcpyDeviceToHost) == hipSuccess;
}

bool ds41_dspark_state_load(ds41_dspark_state* s, const uint8_t* data, uint64_t size, uint64_t* consumed) {
  const uint64_t bytes = ds41_dspark_state_bytes(s);
  uint64_t header[2];
  if (size < bytes) return false;
  memcpy(header, data, sizeof(header));
  if (header[0] > header[1] || header[1] > UINT32_MAX ||
      hipMemcpy(s->ring, data + sizeof(header), bytes - sizeof(header), hipMemcpyHostToDevice) != hipSuccess)
    return false;
  s->ring_start = (uint32_t)header[0];
  s->ring_end = (uint32_t)header[1];
  *consumed = bytes;
  return true;
}

bool ds41_dspark_copy_logits(const ds41_engine* engine, float* out) {
  const ds41_dspark_model* m = engine && engine->dspark ? engine->dspark : nullptr;
  return m && m->scratch &&
         hipMemcpy(out, m->scratch->logits, (uint64_t)kRows * DS41_N_VOCAB * sizeof(float), hipMemcpyDeviceToHost) ==
             hipSuccess;
}
