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
constexpr uint32_t kMaxRows = kRows * DS41_DSPARK_MAX_BATCH;
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
  int32_t* expert_plan = nullptr;  // ds41_v41_expert_plan groups of a block's slots
  float *g = nullptr, *u = nullptr, *mid = nullptr, *down = nullptr, *sg = nullptr, *su = nullptr, *sm = nullptr;
  float *sd = nullptr, *logits = nullptr, *markov_x = nullptr, *bias = nullptr;
  int32_t *tokens = nullptr, *candidate_ids = nullptr;
  float* candidate_logits = nullptr;
  uint32_t* draws = nullptr;
  float* markov_v = nullptr;
  float* confidence = nullptr;
  int32_t* markov_i = nullptr;
  ds41_v41_proposal* proposals = nullptr;
  float *main_raw = nullptr, *main_x = nullptr, *main_kv = nullptr, *main_kvn = nullptr;
  float* head_x = nullptr;     // the drafted rows' final norms, compacted for the target head
  std::vector<float> host_rows;  // embedded block rows
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
  /* The last sampled draft's proposals, kept past other sessions' drafts. */
  ds41_v41_proposal* proposals = nullptr;
};

namespace {

int slices(const ds41_dspark_tensor& t) { return t.k <= 2304 ? 2 : (t.k <= 5120 && t.rows <= 4096 ? 4 : 8); }

bool lin(const ds41_dspark_tensor& t, const float* x, float* out, uint32_t n) {
  // The drafter's Q4_K rows take the FP32 GEMV (one weight read for every row)
  // rather than Q8_1 MMVQ; the target keeps its own dispatch.
  if (t.type == DS41_TENSOR_Q4_K && n <= 8 && t.k % 256 == 0)
    return ds41_v41_gemv((int)t.type, t.p, x, out, (uint32_t)t.rows, t.k, 0, slices(t), n, t.k, (uint32_t)t.rows) ==
           0;
  return ds41_v41_linear(t.type, t.p, (uint32_t)t.rows, t.k, x, out, n);
}

/* lin over the blocks of a batch, every row as in its own block's launch:
 * the Q4_K GEMV keeps each token's arithmetic for up to
 * DS41_V41_GEMV_MAX_TOKENS tokens and the FP32 matmul for any count; other
 * types run block by block. */
bool lin_rows(const ds41_dspark_tensor& t, const float* x, float* out, uint32_t n) {
  if (n <= kRows) return lin(t, x, out, n);
  const bool q4 = t.type == DS41_TENSOR_Q4_K && t.k % 256 == 0;
  if (t.type == DS41_TENSOR_F32) return lin(t, x, out, n);
  const uint32_t chunk = q4 ? DS41_V41_GEMV_MAX_TOKENS : kRows;
  for (uint32_t r = 0; r < n; r += chunk) {
    const uint32_t m = std::min(chunk, n - r);
    const float* xr = x + (uint64_t)r * t.k;
    float* yr = out + (uint64_t)r * t.rows;
    if (q4 ? ds41_v41_gemv((int)t.type, t.p, xr, yr, (uint32_t)t.rows, t.k, 0, slices(t), m, t.k,
                           (uint32_t)t.rows) != 0
           : !lin(t, xr, yr, m))
      return false;
  }
  return true;
}

const float* f32(const ds41_dspark_tensor& t) { return static_cast<const float*>(t.p); }

bool gemv_type(uint32_t type) {
  return type == DS41_TENSOR_Q5_K || type == DS41_TENSOR_Q8_0 || type == DS41_TENSOR_Q4_K;
}

bool alloc_scratch(ds41_dspark_scratch& s) {
  const uint64_t r = kMaxRows;
  s.host_rows.resize(r * kDim);
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
         s.alloc(&s.route_ids, r * kRouted) &&
         s.alloc(&s.expert_plan, DS41_V41_EXPERT_PLAN_INTS(DS41_V41_EXPERT_PLAN_COLS * kRouted)) && s.alloc(&s.g, r * kRouted * DS41_N_FF_EXP) &&
         s.alloc(&s.u, r * kRouted * DS41_N_FF_EXP) && s.alloc(&s.mid, r * kRouted * DS41_N_FF_EXP) &&
         s.alloc(&s.down, r * kRouted * kDim) && s.alloc(&s.sg, r * DS41_N_FF_EXP) &&
         s.alloc(&s.su, r * DS41_N_FF_EXP) && s.alloc(&s.sm, r * DS41_N_FF_EXP) && s.alloc(&s.sd, r * kDim) &&
         s.alloc(&s.logits, r * DS41_N_VOCAB) && s.alloc(&s.markov_x, DS41_DSPARK_MARKOV_RANK) &&
         s.alloc(&s.bias, DS41_N_VOCAB) && s.alloc(&s.tokens, (uint64_t)(kRows + 1) * DS41_DSPARK_MAX_BATCH) && s.alloc(&s.candidate_ids, DS41_DSPARK_CANDIDATES) &&
         s.alloc(&s.candidate_logits, DS41_DSPARK_CANDIDATES) &&
         s.alloc(&s.draws, r) && s.alloc(&s.proposals, r) && s.alloc(&s.markov_v, DS41_V41_MARKOV_CANDIDATES) &&
         s.alloc(&s.confidence, r) &&
         s.alloc(&s.markov_i, DS41_V41_MARKOV_CANDIDATES) && s.alloc(&s.main_raw, f * kDim) &&
         s.alloc(&s.main_x, f * kDim) && s.alloc(&s.main_kv, f * kHeadDim) && s.alloc(&s.main_kvn, f * kHeadDim) &&
         s.alloc(&s.head_x, r * kDim);
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

/* One drafting session of a batch: block rows [row0, row0 + kRows) at
 * positions pos .. pos + kRows - 1, attending to ring rows [first, pos). */
struct Job {
  ds41_dspark_draft_request* req = nullptr;
  uint32_t row0 = 0, pos = 0, first = 0, n_ring = 0;
  const float* freq = nullptr;
  bool stochastic = false;
  float temperature = 1.0f;
};

/* The DSparkAttention of one stage for every job's block rows in xn: the
 * projections run over all rows, the positional phases per job. */
bool attention(const ds41_dspark_stage& st, ds41_dspark_scratch& s, uint32_t stage, const std::vector<Job>& jobs) {
  const uint32_t n = (uint32_t)jobs.size() * kRows;
  const float eps = DS41_RMS_EPS;
  if (!lin_rows(st.q_a, s.xn, s.qa, n) || !lin_rows(st.kv, s.xn, s.kv, n)) return false;
  for (const Job& j : jobs)
    if (ds41_v41_q_kv_norm(s.qa + (uint64_t)j.row0 * DS41_N_LORA_Q, f32(st.q_a_norm),
                           s.qr + (uint64_t)j.row0 * DS41_N_LORA_Q, nullptr, DS41_N_LORA_Q,
                           s.kv + (uint64_t)j.row0 * kHeadDim, f32(st.kv_a_norm), s.kvn + (uint64_t)j.row0 * kHeadDim,
                           kHeadDim, kRot, j.freq, j.pos, kRows, eps))
      return false;
  if (!lin_rows(st.q_b, s.qr, s.q, n)) return false;
  for (const Job& j : jobs) {
    float* q = s.q + (uint64_t)j.row0 * kQDim;
    float* o = s.o + (uint64_t)j.row0 * kQDim;
    if (ds41_v41_rope_tail(q, kRows * DS41_N_HEAD, kHeadDim, kRot, j.freq, j.pos, DS41_N_HEAD, 1, 0) ||
        ds41_v41_dspark_attention(q, j.req->state->ring + stage * kRingFloats, j.first, j.n_ring, DS41_N_SWA,
                                  s.kvn + (uint64_t)j.row0 * kHeadDim, kRows, f32(st.sinks), o, DS41_N_HEAD, kHeadDim,
                                  1.0f / sqrtf((float)kHeadDim)) ||
        ds41_v41_rope_fp8(o, kRows * DS41_N_HEAD, kHeadDim, kRot, j.freq, j.pos, DS41_N_HEAD, 1, 1))
      return false;
  }
  /* Grouped LoRA output: group g maps head slice g of every row. */
  const uint32_t o_rows = DS41_N_OUT_GROUP * DS41_N_LORA_O;
  const bool gemv = gemv_type(st.o_a.type);
  const uint32_t chunk = gemv ? DS41_V41_GEMV_MAX_TOKENS : kRows;
  for (uint32_t r = 0; r < n; r += chunk) {
    const uint32_t m = std::min(chunk, n - r);
    const float* o = s.o + (uint64_t)r * kQDim;
    float* oa = s.oa + (uint64_t)r * o_rows;
    if (gemv ? ds41_v41_gemv((int)st.o_a.type, st.o_a.p, o, oa, o_rows, kGroupDim, DS41_N_LORA_O, 0, m, kQDim, o_rows)
             : ds41_mmq_grouped_vec_typed((int)st.o_a.type, st.o_a.p, o, oa, DS41_N_LORA_O, (int)m, kGroupDim,
                                          DS41_N_OUT_GROUP, 0))
      return false;
  }
  return lin_rows(st.o_b, s.oa, s.attn_out, n);
}

/* The routed experts' FP32-activation GEMV (ds41_expert_gemv.hip.cpp) covers
 * the drafter's Q2_K and Q4_K layers: one fused gate/up/SwiGLU launch and
 * one down launch replace the three mmvq passes and their Q8_1 rounding,
 * streaming the weights at the dense GEMV's bandwidth. */
static int expert_block_bytes(uint32_t type) { return type == 10 ? 84 : type == 12 ? 144 : 0; }

bool moe(const ds41_dspark_stage& st, ds41_dspark_scratch& s, uint32_t n) {
  const int experts = (int)DS41_DSPARK_EXPERTS;
  const int gu_type = (int)st.gate_exps.type;
  const uint64_t gu_bytes = (uint64_t)DS41_N_FF_EXP * kDim / 256 * expert_block_bytes(gu_type);
  const uint64_t down_bytes = (uint64_t)kDim * DS41_N_FF_EXP / 256 * expert_block_bytes(st.down_exps.type);
  const bool gemv_experts =
      gu_type == (int)st.up_exps.type && gu_type == (int)st.down_exps.type && expert_block_bytes(gu_type) != 0;
  if (!lin_rows(st.gate_inp, s.xn, s.router, n) ||
      ds41_v41_route(s.router, f32(st.probs_b), s.route_ids, s.route_w, DS41_DSPARK_EXPERTS, kRouted,
                     DS41_EXPERT_WEIGHT_SCALE, n))
    return false;
  /* Up to 16 rows read each routed expert once for all its slots. */
  const uint32_t slots = n * kRouted;
  if (gemv_experts && n > 1 && n <= DS41_V41_EXPERT_PLAN_COLS) {
    if (ds41_v41_expert_plan(s.route_ids, slots, s.expert_plan) ||
        ds41_v41_gemv_experts_grouped(gu_type, st.gate_exps.p, st.up_exps.p, s.route_w, kSwigluLimit, gu_bytes,
                                      s.expert_plan, slots, n, s.xn, kDim, kRouted, s.mid, DS41_N_FF_EXP, kDim,
                                      nullptr, nullptr, 1) ||
        ds41_v41_gemv_experts_grouped((int)st.down_exps.type, st.down_exps.p, nullptr, nullptr, 0.0f, down_bytes,
                                      s.expert_plan, slots, n, s.mid, DS41_N_FF_EXP, 1, s.down, kDim,
                                      DS41_N_FF_EXP, nullptr, nullptr, 1))
      return false;
  } else if (gemv_experts) {
    if (ds41_v41_gemv_experts_swiglu(gu_type, st.gate_exps.p, st.up_exps.p, gu_bytes, s.route_ids, n * kRouted,
                                     s.xn, kDim, kRouted, s.route_w, kSwigluLimit, s.mid, DS41_N_FF_EXP, kDim,
                                     nullptr, nullptr, 1) ||
        ds41_v41_gemv_experts((int)st.down_exps.type, st.down_exps.p, down_bytes, s.route_ids, n * kRouted, s.mid,
                              DS41_N_FF_EXP, 1, s.down, kDim, DS41_N_FF_EXP, nullptr, 1))
      return false;
  } else {
    /* MMVQ block by block: each block's launch is its own pass's. */
    for (uint32_t r = 0; r < n; r += kRows) {
      const uint64_t slot = (uint64_t)r * kRouted;
      const uint64_t ff = slot * DS41_N_FF_EXP;
      if (ds41_mmq_moe_vec_typed(gu_type, st.gate_exps.p, s.xn + (uint64_t)r * kDim, s.route_ids + slot, s.g + ff,
                                 DS41_N_FF_EXP, kDim, (int)kRows, experts, (int)kRouted, 0) ||
          ds41_mmq_moe_vec_typed((int)st.up_exps.type, st.up_exps.p, s.xn + (uint64_t)r * kDim, s.route_ids + slot,
                                 s.u + ff, DS41_N_FF_EXP, kDim, (int)kRows, experts, (int)kRouted, 0) ||
          ds41_v41_swiglu(s.g + ff, s.u + ff, s.route_w + slot, s.mid + ff, kRows * kRouted, DS41_N_FF_EXP,
                          kSwigluLimit) ||
          ds41_mmq_moe_vec_typed((int)st.down_exps.type, st.down_exps.p, s.mid + ff, s.route_ids + slot,
                                 s.down + slot * kDim, kDim, DS41_N_FF_EXP, (int)(kRows * kRouted), experts, 1, 0))
        return false;
    }
  }
  if (st.gate_shexp.type == st.up_shexp.type && gemv_type(st.gate_shexp.type)) {
    for (uint32_t r = 0; r < n; r += DS41_V41_GEMV_MAX_TOKENS)
      if (ds41_v41_gemv_swiglu((int)st.gate_shexp.type, st.gate_shexp.p, st.up_shexp.p, s.xn + (uint64_t)r * kDim,
                               s.sm + (uint64_t)r * DS41_N_FF_EXP, DS41_N_FF_EXP, kDim, kSwigluLimit, 0,
                               std::min<uint32_t>(DS41_V41_GEMV_MAX_TOKENS, n - r)))
        return false;
  } else if (!lin_rows(st.gate_shexp, s.xn, s.sg, n) || !lin_rows(st.up_shexp, s.xn, s.su, n) ||
             ds41_v41_swiglu(s.sg, s.su, nullptr, s.sm, n, DS41_N_FF_EXP, kSwigluLimit)) {
    return false;
  }
  return lin_rows(st.down_shexp, s.sm, s.sd, n);
}

/* In launches of at most 16 rows, each on hc_mix's per-token kernel as
 * the drafter's own block is. */
bool hc_mix(const ds41_dspark_tensor& fn, const ds41_dspark_tensor& scale, const ds41_dspark_tensor& base,
            ds41_dspark_scratch& s, const float* h, float* pre, float* post, float* comb, uint32_t n) {
  constexpr uint32_t kMixRows = 16;
  for (uint32_t r = 0; r < n; r += kMixRows)
    if (ds41_v41_hc_mix(f32(fn), h + (uint64_t)r * kHc * kDim, f32(scale), f32(base), s.mixes + (uint64_t)r * kMixes,
                        pre + (uint64_t)r * kHc, post + (uint64_t)r * kHc, comb + (uint64_t)r * kHc * kHc, kHc, kDim,
                        DS41_RMS_EPS, DS41_N_HC_SINKHORN_ITER, DS41_HC_EPS, std::min(kMixRows, n - r)))
      return false;
  return true;
}

}  // namespace

bool ds41_dspark_has_confidence(const ds41_dspark_model* m) {
  return m && m->confidence.p && m->confidence.type == DS41_TENSOR_F32 &&
         m->confidence.k == kDim + DS41_DSPARK_MARKOV_RANK &&
         (!m->confidence_bias.p || m->confidence_bias.type == DS41_TENSOR_F32);
}

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
  if (const hipError_t err = hipMalloc(&m->device, total); err != hipSuccess) {
    (void)hipGetLastError();
    size_t free_b = 0, total_b = 0;
    (void)hipMemGetInfo(&free_b, &total_b);
    fprintf(stderr,
            "ds41: DSpark drafter needs %.2f GiB of GPU memory, %.2f GiB free of %.2f GiB after the backbone: %s\n",
            (double)total / 1073741824.0, (double)free_b / 1073741824.0, (double)total_b / 1073741824.0,
            hipGetErrorString(err));
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
  void *ring = nullptr, *proposals = nullptr;
  if (hipMalloc(&ring, DS41_DSPARK_STAGES * kRingFloats * sizeof(float)) != hipSuccess ||
      hipMalloc(&proposals, kRows * sizeof(ds41_v41_proposal)) != hipSuccess) {
    (void)hipFree(ring);
    delete s;
    return nullptr;
  }
  s->ring = static_cast<float*>(ring);
  s->proposals = static_cast<ds41_v41_proposal*>(proposals);
  return s;
}

void ds41_dspark_state_destroy(ds41_dspark_state* s) {
  if (!s) return;
  (void)hipFree(s->ring);
  (void)hipFree(s->proposals);
  delete s;
}

void ds41_dspark_state_reset(ds41_dspark_state* s, uint32_t pos) {
  s->ring_start = s->ring_end = pos;
}

uint32_t ds41_dspark_state_end(const ds41_dspark_state* s) { return s->ring_end; }

const ds41_v41_proposal* ds41_dspark_proposals(const ds41_dspark_state* s) {
  return s->proposals;
}

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

bool ds41_dspark_draft(ds41_dspark_state* s, int anchor, int32_t* drafts, uint32_t count,
                       const ds41_dspark_sampler* sampler, float* top, float* confidence) {
  ds41_dspark_draft_request request;
  request.state = s;
  request.anchor = anchor;
  request.drafts = drafts;
  request.count = count;
  request.sampler = sampler;
  request.top = top;
  request.confidence = confidence;
  return ds41_dspark_draft_batch(&request, 1) && request.drafted;
}

bool ds41_dspark_draft_batch(ds41_dspark_draft_request* requests, uint32_t count) {
  if (count > DS41_DSPARK_MAX_BATCH) return false;
  std::vector<Job> jobs;
  for (uint32_t i = 0; i < count; i++) {
    ds41_dspark_draft_request& r = requests[i];
    r.drafted = false;
    ds41_dspark_state* s = r.state;
    if (!s || r.count == 0 || r.count > kRows || s->ring_end == s->ring_start) continue;
    if (!jobs.empty() && s->engine->dspark != jobs[0].req->state->engine->dspark) return false;
    for (const Job& j : jobs)
      if (j.req->state == s) return false;
    Job j;
    j.req = &r;
    j.row0 = (uint32_t)jobs.size() * kRows;
    j.pos = s->ring_end;  // the anchor's position
    j.first = std::max(s->ring_start, j.pos > DS41_N_SWA ? j.pos - DS41_N_SWA : 0u);
    j.n_ring = j.pos - j.first;
    j.freq = ds41_decoder_freq_plain(s->decoder);
    jobs.push_back(j);
  }
  if (jobs.empty()) return true;
  const ds41_dspark_model* m = jobs[0].req->state->engine->dspark;
  ds41_dspark_scratch& sc = *m->scratch;
  const uint32_t n = (uint32_t)jobs.size() * kRows;
  const uint64_t hd = (uint64_t)kHc * kDim;
  constexpr uint32_t kTokenStride = kRows + 1;  // a job's anchor and its drafts in sc.tokens

  /* Block rows [anchor, noise ...], replicated over the hc streams; the first stage's pre-mix is one-hot.
   * Sampled requests draw each proposal on the device from the biased top
   * eight, so the next Markov row reads it without a host round trip; the
   * request RNG supplies the draws up front and q returns once per cycle. */
  std::vector<int32_t> tokens((uint64_t)jobs.size() * kTokenStride, -1);
  std::vector<float> one_hot((uint64_t)n * kHc, 0.0f);
  std::vector<uint32_t> draws(n, 0);
  bool ok = true;
  bool any_stochastic = false;
  for (Job& j : jobs) {
    const ds41_dspark_draft_request& r = *j.req;
    int block[kRows];
    block[0] = r.anchor;
    for (uint32_t t = 1; t < kRows; t++) block[t] = DS41_DSPARK_NOISE_TOKEN;
    for (uint32_t t = 0; t < kRows; t++) one_hot[(uint64_t)(j.row0 + t) * kHc] = 1.0f;
    tokens[(uint64_t)(j.row0 / kRows) * kTokenStride] = r.anchor;
    j.stochastic = r.sampler && r.sampler->draws && r.sampler->retain;
    if (j.stochastic) {
      j.temperature = r.sampler->draws(r.sampler->ctx, r.count, draws.data() + j.row0);
      any_stochastic = true;
    }
    ok = ok && ds41_decoder_embed(r.state->decoder, block, kRows, sc.host_rows.data() + (uint64_t)j.row0 * kDim);
  }
  ok = ok && (!any_stochastic ||
              hipMemcpy(sc.draws, draws.data(), n * sizeof(uint32_t), hipMemcpyHostToDevice) == hipSuccess) &&
       hipMemcpy(sc.xn, sc.host_rows.data(), (uint64_t)n * kDim * sizeof(float), hipMemcpyHostToDevice) ==
           hipSuccess &&
       hipMemcpy(sc.pre[0], one_hot.data(), one_hot.size() * sizeof(float), hipMemcpyHostToDevice) == hipSuccess &&
       hipMemcpy(sc.tokens, tokens.data(), tokens.size() * sizeof(int32_t), hipMemcpyHostToDevice) == hipSuccess;
  for (uint32_t c = 0; c < kHc && ok; c++)
    ok = hipMemcpy2DAsync(sc.h + c * kDim, hd * sizeof(float), sc.xn, kDim * sizeof(float), kDim * sizeof(float), n,
                          hipMemcpyDeviceToDevice, 0) == hipSuccess;

  /* Shifted mHC as in the decoder: attention takes pre[0] and leaves its mix
   * in pre[1] for the FFN, whose own mix returns to pre[0]. */
  for (uint32_t st = 0; st < DS41_DSPARK_STAGES && ok; st++) {
    const ds41_dspark_stage& l = m->stage[st];
    ok = hc_mix(l.hc_attn_fn, l.hc_attn_scale, l.hc_attn_base, sc, sc.h, sc.pre[1], sc.post, sc.comb, n) &&
         ds41_v41_hc_collapse_norm(sc.h, sc.pre[0], f32(l.attn_norm), sc.xn, kHc, kDim, DS41_RMS_EPS, n) == 0 &&
         attention(l, sc, st, jobs) && ds41_v41_hc_expand(sc.attn_out, sc.h, sc.post, sc.comb, sc.h2, kHc, kDim, n) == 0 &&
         hc_mix(l.hc_ffn_fn, l.hc_ffn_scale, l.hc_ffn_base, sc, sc.h2, sc.pre[0], sc.ffn_post, sc.ffn_comb, n) &&
         ds41_v41_hc_collapse_norm(sc.h2, sc.pre[1], f32(l.ffn_norm), sc.xn, kHc, kDim, DS41_RMS_EPS, n) == 0 &&
         moe(l, sc, n) &&
         ds41_v41_hc_expand_sum(sc.down, sc.sd, kRouted, sc.h2, sc.ffn_post, sc.ffn_comb, sc.h, kHc, kDim, n) == 0;
  }
  /* Target head over each job's drafted rows (compacted for several jobs),
   * in launches of at most 8 rows: the decode GEMV's per-token arithmetic. */
  ok = ok && ds41_v41_hc_collapse_norm(sc.h, sc.pre[0], f32(m->out_norm), sc.xn, kHc, kDim, DS41_RMS_EPS, n) == 0;
  std::vector<uint32_t> logits_row0(jobs.size());
  uint32_t head_rows = 0;
  for (uint32_t i = 0; i < jobs.size() && ok; i++) {
    const Job& j = jobs[i];
    logits_row0[i] = head_rows;
    if (jobs.size() > 1)
      ok = hipMemcpyAsync(sc.head_x + (uint64_t)head_rows * kDim, sc.xn + (uint64_t)j.row0 * kDim,
                          (uint64_t)j.req->count * kDim * sizeof(float), hipMemcpyDeviceToDevice, 0) == hipSuccess;
    head_rows += j.req->count;
  }
  const float* head_x = jobs.size() > 1 ? sc.head_x : sc.xn;
  constexpr uint32_t kHeadRows = 8;
  for (uint32_t r = 0; r < head_rows && ok; r += kHeadRows)
    ok = ds41_decoder_output_head(jobs[0].req->state->decoder, head_x + (uint64_t)r * kDim,
                                  sc.logits + (uint64_t)r * DS41_N_VOCAB, std::min(kHeadRows, head_rows - r));
  const int format = m->markov_w1.type == DS41_TENSOR_BF16  ? DS41_V41_ROW_BF16
                     : m->markov_w1.type == DS41_TENSOR_F16 ? DS41_V41_ROW_F16
                                                            : DS41_V41_ROW_F32;
  // Later Markov rows depend only on earlier proposals; unused suffixes
  // cannot affect the requested prefix. The transformer body keeps all five rows.
  // The learned acceptance head reads each row's Markov input.
  const bool has_confidence = ds41_dspark_has_confidence(m);
  // FP16 W2 rows of the rank fuse the projection into the selection.
  const bool fused = m->markov_w2.type == DS41_TENSOR_F16 && m->markov_w2.k == DS41_DSPARK_MARKOV_RANK &&
                     m->markov_w2.rows == DS41_N_VOCAB && (uintptr_t)m->markov_w2.p % 16 == 0;
  const auto* w2 = static_cast<const uint16_t*>(m->markov_w2.p);
  for (uint32_t ji = 0; ji < jobs.size() && ok; ji++) {
    const Job& j = jobs[ji];
    const ds41_dspark_draft_request& r = *j.req;
    const bool learned = r.confidence && has_confidence;
    int32_t* job_tokens = sc.tokens + (uint64_t)(j.row0 / kRows) * kTokenStride;
    for (uint32_t i = 0; i < r.count && ok; i++) {
      const float* row = sc.logits + (uint64_t)(logits_row0[ji] + i) * DS41_N_VOCAB;
      const uint32_t at = j.row0 + i;
      ok = ds41_v41_gather_row(m->markov_w1.p, format, job_tokens, i, sc.markov_x, DS41_DSPARK_MARKOV_RANK) == 0 &&
           (!learned || ds41_v41_dspark_confidence(sc.h + (uint64_t)j.row0 * hd, sc.pre[0] + (uint64_t)j.row0 * kHc,
                                                   sc.markov_x, f32(m->confidence),
                                                   m->confidence_bias.p ? f32(m->confidence_bias) : nullptr, kDim,
                                                   DS41_DSPARK_MARKOV_RANK, i, sc.confidence + j.row0) == 0);
      if (!ok) break;
      if (fused)
        ok = (j.stochastic ? ds41_v41_markov_propose(w2, sc.markov_x, row, DS41_N_VOCAB, sc.markov_v, sc.markov_i,
                                                     sc.candidate_ids, sc.candidate_logits, sc.draws + at,
                                                     j.temperature, sc.proposals + at, job_tokens + i + 1)
                           : ds41_v41_markov_argmax(w2, sc.markov_x, row, DS41_N_VOCAB, sc.markov_v, sc.markov_i,
                                                    job_tokens, i + 1)) == 0;
      else
        ok = lin(m->markov_w2, sc.markov_x, sc.bias, 1) &&
             (j.stochastic ? ds41_v41_propose_top8(row, sc.bias, DS41_N_VOCAB, sc.candidate_ids, sc.candidate_logits,
                                                   sc.draws + at, j.temperature, sc.proposals + at, job_tokens + i + 1)
                           : ds41_v41_argmax_bias(row, sc.bias, DS41_N_VOCAB, job_tokens, i + 1)) == 0;
    }
  }
  /* One download each of the drafted tokens, proposals and confidences. */
  std::vector<ds41_v41_proposal> q(any_stochastic ? n : 0);
  std::vector<float> conf(n);
  ok = ok && hipMemcpy(tokens.data(), sc.tokens, tokens.size() * sizeof(int32_t), hipMemcpyDeviceToHost) == hipSuccess &&
       (!any_stochastic || hipMemcpy(q.data(), sc.proposals, n * sizeof(q[0]), hipMemcpyDeviceToHost) == hipSuccess);
  bool any_learned = false;
  for (const Job& j : jobs) any_learned = any_learned || (j.req->confidence && has_confidence);
  ok = ok && (!any_learned || hipMemcpy(conf.data(), sc.confidence, n * sizeof(float), hipMemcpyDeviceToHost) ==
                                  hipSuccess);
  /* A job without finite candidates fails alone, as its own draft would. */
  for (Job& j : jobs) {
    if (!ok) break;
    ds41_dspark_draft_request& r = *j.req;
    bool valid = true;
    if (j.stochastic) {
      ok = hipMemcpyAsync(r.state->proposals, sc.proposals + j.row0, r.count * sizeof(q[0]), hipMemcpyDeviceToDevice,
                          0) == hipSuccess;
      for (uint32_t i = 0; i < r.count && ok && valid; i++) {
        const ds41_v41_proposal& p = q[j.row0 + i];
        valid = p.token >= 0;  // fewer than eight finite candidates
        if (valid) r.sampler->retain(r.sampler->ctx, i, p.ids, p.masses, p.size, p.token);
        if (r.top) r.top[i] = (float)p.masses[0] / (float)(1u << 24);
        r.drafts[i] = p.token;
      }
    } else {
      for (uint32_t i = 0; i < r.count; i++) r.drafts[i] = tokens[(uint64_t)(j.row0 / kRows) * kTokenStride + 1 + i];
    }
    if (!valid) continue;
    if (r.confidence) {
      if (has_confidence)
        std::copy_n(conf.data() + j.row0, r.count, r.confidence);
      else
        std::fill_n(r.confidence, r.count, -1.0f);
    }
    for (uint32_t i = 0; i < r.count && valid; i++) valid = r.drafts[i] >= 0;
    r.drafted = valid;
  }
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

bool ds41_dspark_copy_logits(const ds41_engine* engine, float* out, uint32_t row0, uint32_t rows) {
  const ds41_dspark_model* m = engine && engine->dspark ? engine->dspark : nullptr;
  return m && m->scratch && row0 + rows <= kMaxRows &&
         hipMemcpy(out, m->scratch->logits + (uint64_t)row0 * DS41_N_VOCAB, (uint64_t)rows * DS41_N_VOCAB * sizeof(float),
                   hipMemcpyDeviceToHost) == hipSuccess;
}
