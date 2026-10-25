/* DeepSeek V4.1 Flash single-token decoder.
 *
 * One token per step, following the official inference/model.py decode path:
 * Engram (layers 1 and 14), shifted mHC, CSA2 KV/index sharing, sparse
 * attention with sinks, and the routed + shared MoE. The quantized
 * projections run on the vendored mmvq kernels; everything else runs on the
 * plain kernels in kernels/rocm/ds41_v41.hip.cpp. */
#include <hip/hip_runtime_api.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#include "ds41_v41.h"
#include "mmq/ds41_mmq.h"
#include "resident_api.h"
#include "model.h"
#include "model_data_internal.h"
#include "native_internal.h"
#include "tensor_types.h"

namespace {

constexpr uint32_t kHc = DS41_N_HC;
constexpr uint32_t kDim = DS41_N_EMBD;
constexpr uint32_t kHeadDim = DS41_N_HEAD_DIM;
constexpr uint32_t kRot = DS41_N_ROT;
constexpr uint32_t kQDim = DS41_N_HEAD * DS41_N_HEAD_DIM;
constexpr uint32_t kIndexQDim = DS41_N_INDEXER_HEAD * DS41_N_INDEXER_HEAD_DIM;
constexpr uint32_t kGroupDim = DS41_N_HEAD_DIM * (DS41_N_HEAD / DS41_N_OUT_GROUP);
constexpr uint32_t kMixes = 2 * kHc + kHc * kHc;
constexpr uint32_t kEngramIn = DS41_ENGRAM_COLS * DS41_ENGRAM_HEAD_DIM;
constexpr uint32_t kEngramOut = kDim * (kHc + 1);
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

uint16_t read_u16(const uint8_t* p) { return (uint16_t)(p[0] | p[1] << 8); }

void scale_min_k4(int j, const uint8_t* q, uint8_t* d, uint8_t* m) {
  if (j < 4) {
    *d = q[j] & 63;
    *m = q[j + 4] & 63;
  } else {
    *d = (q[j + 4] & 0xf) | ((q[j - 4] >> 6) << 4);
    *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
  }
}

/* Host dequantization of one row; enough for the embedding and Engram rows. */
bool dequant_row(uint32_t type, const uint8_t* src, float* out, uint32_t n) {
  switch (type) {
  case DS41_TENSOR_F32:
    memcpy(out, src, n * sizeof(float));
    return true;
  case DS41_TENSOR_Q8_0:
    for (uint32_t b = 0; b < n / 32; b++, src += 34) {
      const float d = half_to_float(read_u16(src));
      for (int i = 0; i < 32; i++) out[b * 32 + i] = d * (float)(int8_t)src[2 + i];
    }
    return true;
  case DS41_TENSOR_IQ4_NL: {
    static const int8_t kValues[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
    for (uint32_t b = 0; b < n / 32; b++, src += 18) {
      const float d = half_to_float(read_u16(src));
      for (int i = 0; i < 16; i++) {
        out[b * 32 + i] = d * kValues[src[2 + i] & 0xf];
        out[b * 32 + 16 + i] = d * kValues[src[2 + i] >> 4];
      }
    }
    return true;
  }
  case DS41_TENSOR_Q5_K:
    for (uint32_t b = 0; b < n / 256; b++, src += 176) {
      const float d = half_to_float(read_u16(src));
      const float dmin = half_to_float(read_u16(src + 2));
      const uint8_t* scales = src + 4;
      const uint8_t* qh = src + 16;
      const uint8_t* ql = src + 48;
      float* y = out + b * 256;
      uint8_t u1 = 1;
      uint8_t u2 = 2;
      for (int j = 0, is = 0; j < 256; j += 64, is += 2) {
        uint8_t sc;
        uint8_t m;
        scale_min_k4(is, scales, &sc, &m);
        const float d1 = d * sc;
        const float m1 = dmin * m;
        scale_min_k4(is + 1, scales, &sc, &m);
        const float d2 = d * sc;
        const float m2 = dmin * m;
        for (int l = 0; l < 32; l++) *y++ = d1 * ((ql[l] & 0xf) + (qh[l] & u1 ? 16 : 0)) - m1;
        for (int l = 0; l < 32; l++) *y++ = d2 * ((ql[l] >> 4) + (qh[l] & u2 ? 16 : 0)) - m2;
        ql += 32;
        u1 <<= 2;
        u2 <<= 2;
      }
    }
    return true;
  default:
    return false;
  }
}

/* Official precompute_freqs_cis inverse frequencies, YaRN when original_ctx > 0. */
std::vector<float> rope_inv_freq(float base, uint64_t original_ctx) {
  const uint32_t dim = kRot;
  std::vector<float> freqs(dim / 2);
  for (uint32_t i = 0; i < dim / 2; i++) freqs[i] = 1.0f / powf(base, (float)(2 * i) / (float)dim);
  if (original_ctx > 0) {
    auto corrected = [&](double rotations) {
      return dim * std::log((double)original_ctx / (rotations * 2 * M_PI)) / (2 * std::log((double)base));
    };
    const double low = std::max(std::floor(corrected(DS41_ROPE_YARN_BETA_FAST)), 0.0);
    const double high = std::min(std::ceil(corrected(DS41_ROPE_YARN_BETA_SLOW)), (double)dim - 1);
    for (uint32_t i = 0; i < dim / 2; i++) {
      const float ramp =
          std::clamp(((float)i - (float)low) / (float)std::max(high - low, 1e-3), 0.0f, 1.0f);
      const float smooth = 1.0f - ramp;
      freqs[i] = freqs[i] / DS41_ROPE_SCALE_FACTOR * (1.0f - smooth) + freqs[i] * smooth;
    }
  }
  return freqs;
}

struct LayerState {
  float* window = nullptr;    // [DS41_N_SWA, head_dim] ring
  float* comp = nullptr;      // [context / ratio, head_dim] (KV sources)
  float* index_k = nullptr;   // [context / ratio, indexer head_dim] (KV sources)
  float* pool_kv = nullptr;   // [ratio, head_dim] compressor frontier
  float* pool_score = nullptr;
};

}  // namespace

struct ds41_decoder {
  ds41_engine* engine = nullptr;
  uint32_t context = 0;
  uint32_t pos = 0;
  std::vector<void*> allocations;
  LayerState layer[DS41_N_LAYER];
  float* freq_plain = nullptr;
  float* freq_compress = nullptr;

  float *h = nullptr, *h2 = nullptr, *x = nullptr, *xn = nullptr;
  float *mixes = nullptr, *pre[2] = {}, *post = nullptr, *comb = nullptr, *ffn_post = nullptr, *ffn_comb = nullptr;
  float *qa = nullptr, *qr = nullptr, *q = nullptr, *kv = nullptr, *kvn = nullptr;
  float *ckv = nullptr, *cscore = nullptr, *craw = nullptr, *latent = nullptr;
  float *ik = nullptr, *ikn = nullptr, *iq = nullptr, *iw = nullptr, *scores = nullptr, *block_scores = nullptr;
  uint8_t* candidates = nullptr;
  uint32_t *topk = nullptr, *block_select = nullptr;
  float *o = nullptr, *oa = nullptr, *attn_out = nullptr;
  float *router = nullptr, *route_w = nullptr;
  int32_t* route_ids = nullptr;
  float *g6 = nullptr, *u6 = nullptr, *mid6 = nullptr, *down6 = nullptr;
  float *sg = nullptr, *su = nullptr, *sm = nullptr, *sd = nullptr, *moe_out = nullptr;
  float *engram_in = nullptr, *engram_kv = nullptr, *logits = nullptr;

  uint32_t n_topk = 0;
  int comp_layer = -1;
  std::vector<int64_t> engram_history;
  std::vector<float> host_h;
  std::vector<int32_t> host_ids;
};

namespace {

template <typename T>
T* alloc(ds41_decoder* d, uint64_t count) {
  void* p = nullptr;
  if (hipMalloc(&p, count * sizeof(T)) != hipSuccess) return nullptr;
  (void)hipMemset(p, 0, count * sizeof(T));
  d->allocations.push_back(p);
  return static_cast<T*>(p);
}

const ds41_model* model(const ds41_decoder* d) { return d->engine->model; }

const void* dev(const ds41_decoder* d, const ds41_tensor* t) {
  return ds41_gpu_model_ptr(model(d)->map, t->abs_offset, t->bytes);
}

const uint8_t* host(const ds41_decoder* d, const ds41_tensor* t) { return model(d)->map + t->abs_offset; }

/* out = W x for a [rows, k] weight of any supported storage type. */
bool linear(const ds41_decoder* d, const ds41_tensor* w, const float* x, float* out) {
  const uint32_t k = (uint32_t)w->dim[0];
  const uint32_t rows = (uint32_t)(w->elements / k);
  const void* p = dev(d, w);
  if (!p) return false;
  switch (w->type) {
  case DS41_TENSOR_BF16:
    return ds41_v41_matvec_bf16((const uint16_t*)p, x, out, rows, k) == 0;
  case DS41_TENSOR_F32:
    return ds41_v41_matvec_f32((const float*)p, x, out, rows, k) == 0;
  default:
    return ds41_mmq_dense_vec_typed((int)w->type, p, x, out, (int)rows, 1, (int)k, 0) == 0;
  }
}

const float* f32(const ds41_decoder* d, const ds41_tensor* t) { return (const float*)dev(d, t); }

const float* layer_freq(const ds41_decoder* d, uint32_t il) {
  return ds41_layer_compress_ratio(il) ? d->freq_compress : d->freq_plain;
}

/* Engram row ids for the newest token; mirrors NgramHashState.forward. */
void engram_hash(ds41_decoder* d, int token, int64_t (*rows)[DS41_ENGRAM_COLS]) {
  const ds41_weights* w = d->engine->weights;
  const int32_t* token_map = (const int32_t*)host(d, w->engram_token_map);
  const int64_t* primes = (const int64_t*)host(d, w->engram_primes);
  const int64_t* offsets = (const int64_t*)host(d, w->engram_offsets);
  const int64_t* multipliers = (const int64_t*)host(d, w->engram_multipliers);
  const int64_t pad = token_map[DS41_ENGRAM_PAD_TOKEN];
  d->engram_history.push_back(token_map[token]);
  const int64_t p = (int64_t)d->engram_history.size() - 1;
  int64_t grams[DS41_ENGRAM_MAX_NGRAM];
  bool blocked = false;
  for (int64_t shift = 0; shift < DS41_ENGRAM_MAX_NGRAM; shift++) {
    blocked = blocked || p < shift;
    grams[shift] = blocked ? pad : d->engram_history[p - shift];
  }
  for (uint32_t layer = 0; layer < DS41_N_ENGRAM_LAYER; layer++) {
    const int64_t* mult = multipliers + layer * DS41_ENGRAM_MAX_NGRAM;
    int64_t rolling = grams[0] * mult[0];
    for (uint32_t i = 1; i < DS41_ENGRAM_MAX_NGRAM; i++) {
      rolling ^= grams[i] * mult[i];
      for (uint32_t head = 0; head < DS41_ENGRAM_HEADS; head++) {
        const uint32_t col = (i - 1) * DS41_ENGRAM_HEADS + head;
        rows[layer][col] = rolling % primes[layer * DS41_ENGRAM_COLS + col] + offsets[layer * DS41_ENGRAM_COLS + col];
      }
    }
  }
}

bool engram(ds41_decoder* d, uint32_t il, const int64_t* rows) {
  const ds41_layer_weights* l = &d->engine->weights->layer[il];
  const ds41_engram_tables* e = d->engine->engram;
  const ds41_tensor* table = l->engram_embd;
  const uint64_t base = table->abs_offset - e->file.shard_offset[0];
  std::vector<uint8_t> raw(e->row_bytes);
  std::vector<float> in(kEngramIn);
  for (uint32_t c = 0; c < DS41_ENGRAM_COLS; c++) {
    const off_t at = (off_t)(base + (uint64_t)rows[c] * e->row_bytes);
    if (pread(e->file.shard_fd[0], raw.data(), raw.size(), at) != (ssize_t)raw.size()) {
      fprintf(stderr, "ds41: Engram row read failed\n");
      return false;
    }
    if (!dequant_row(table->type, raw.data(), in.data() + c * DS41_ENGRAM_HEAD_DIM, DS41_ENGRAM_HEAD_DIM))
      return false;
  }
  if (hipMemcpy(d->engram_in, in.data(), in.size() * sizeof(float), hipMemcpyHostToDevice) != hipSuccess) return false;
  if (!linear(d, l->engram_wkv, d->engram_in, d->engram_kv)) return false;
  return ds41_v41_engram_gate(d->h, d->engram_kv, (const uint16_t*)dev(d, l->engram_q),
                              (const uint16_t*)dev(d, l->engram_k), kHc, kDim, DS41_RMS_EPS) == 0;
}

bool hc_mix(ds41_decoder* d, const ds41_tensor* fn, const ds41_tensor* scale, const ds41_tensor* base,
            const float* h, float* pre, float* post, float* comb) {
  return ds41_v41_hc_mixes(f32(d, fn), h, d->mixes, kHc, kDim, DS41_RMS_EPS) == 0 &&
         ds41_v41_hc_sinkhorn(d->mixes, f32(d, scale), f32(d, base), pre, post, comb, kHc,
                              DS41_N_HC_SINKHORN_ITER, DS41_HC_EPS) == 0;
}

bool copy_row(float* dst, const float* src, uint32_t n) {
  return hipMemcpyAsync(dst, src, n * sizeof(float), hipMemcpyDeviceToDevice, 0) == hipSuccess;
}

bool attention(ds41_decoder* d, uint32_t il) {
  const ds41_layer_weights* l = &d->engine->weights->layer[il];
  LayerState& s = d->layer[il];
  const uint32_t pos = d->pos;
  const uint32_t ratio = ds41_layer_compress_ratio(il);
  const float* freq = layer_freq(d, il);
  const float eps = DS41_RMS_EPS;

  if (!linear(d, l->attn_q_a, d->xn, d->qa) ||
      ds41_v41_rms_norm(d->qa, f32(d, l->attn_q_a_norm), d->qr, 1, DS41_N_LORA_Q, eps) ||
      !linear(d, l->attn_q_b, d->qr, d->q) || ds41_v41_rope_tail(d->q, DS41_N_HEAD, kHeadDim, kRot, freq, pos, 0))
    return false;

  if (!linear(d, l->attn_kv, d->xn, d->kv) ||
      ds41_v41_rms_norm(d->kv, f32(d, l->attn_kv_a_norm), d->kvn, 1, kHeadDim, eps) ||
      ds41_v41_rope_tail(d->kvn, 1, kHeadDim, kRot, freq, pos, 0) || ds41_v41_fp8_fake_quant(d->kvn, kHeadDim) ||
      !copy_row(s.window + (uint64_t)(pos % DS41_N_SWA) * kHeadDim, d->kvn, kHeadDim))
    return false;
  const uint32_t n_window = std::min(pos + 1, DS41_N_SWA);

  const float* comp = nullptr;
  uint32_t n_comp = 0;
  if (ratio) {
    const uint32_t compress_len = (pos + 1) / ratio;
    const uint32_t group_pos = pos + 1 - ratio;
    bool emitted = false;
    if (ds41_layer_is_kv_source(il)) {
      if (!linear(d, l->attn_compressor_kv, d->xn, d->ckv)) return false;
      if (ratio == 1) {
        if (ds41_v41_rms_norm(d->ckv, f32(d, l->attn_compressor_norm), d->latent, 1, kHeadDim, eps)) return false;
        emitted = true;
      } else {
        emitted = (pos + 1) % ratio == 0;
        if (!linear(d, l->attn_compressor_gate, d->xn, d->cscore) ||
            ds41_v41_compress_pool(d->ckv, d->cscore, s.pool_kv, s.pool_score, pos % ratio, ratio, emitted, d->craw,
                                   kHeadDim))
          return false;
        if (emitted && ds41_v41_rms_norm(d->craw, f32(d, l->attn_compressor_norm), d->latent, 1, kHeadDim, eps))
          return false;
      }
      d->comp_layer = (int)il;
    }
    const LayerState& source = d->layer[d->comp_layer];
    if (ds41_layer_is_index_source(il)) {
      d->n_topk = 0;
      if (compress_len > 0) {
        if (emitted) {
          if (!linear(d, l->indexer_attn_k, d->latent, d->ik) ||
              ds41_v41_rms_norm(d->ik, f32(d, l->indexer_k_norm), d->ikn, 1, DS41_N_INDEXER_HEAD_DIM, eps) ||
              ds41_v41_rope_tail(d->ikn, 1, DS41_N_INDEXER_HEAD_DIM, kRot, freq, group_pos, 0) ||
              ds41_v41_fp4_fake_quant(d->ikn, DS41_N_INDEXER_HEAD_DIM, 32, 0) ||
              !copy_row(s.index_k + (uint64_t)(pos / ratio) * DS41_N_INDEXER_HEAD_DIM, d->ikn,
                        DS41_N_INDEXER_HEAD_DIM))
            return false;
        }
        const float w_scale = 1.0f / sqrtf((float)DS41_N_INDEXER_HEAD_DIM) / sqrtf((float)DS41_N_INDEXER_HEAD);
        if (!linear(d, l->indexer_attn_q_b, d->qr, d->iq) ||
            ds41_v41_rope_tail(d->iq, DS41_N_INDEXER_HEAD, DS41_N_INDEXER_HEAD_DIM, kRot, freq, pos, 0) ||
            ds41_v41_fp4_fake_quant(d->iq, kIndexQDim, 32, 0) || !linear(d, l->indexer_proj, d->xn, d->iw) ||
            ds41_v41_indexer_score(d->iq, source.index_k, d->iw, w_scale, d->scores, compress_len,
                                   DS41_N_INDEXER_HEAD, DS41_N_INDEXER_HEAD_DIM))
          return false;
        if (il == DS41_CANDIDATE_SOURCE_LAYER) {
          if (ds41_v41_candidate_mask(d->scores, d->candidates, d->block_scores, d->block_select, compress_len,
                                      DS41_CANDIDATE_BLOCK_SIZE, DS41_CANDIDATE_TOPK_BLOCKS))
            return false;
        } else if (il > DS41_CANDIDATE_SOURCE_LAYER) {
          if (ds41_v41_apply_mask(d->scores, d->candidates, compress_len)) return false;
        }
        d->n_topk = std::min(DS41_N_INDEXER_TOP_K, compress_len);
        if (ds41_v41_topk(d->scores, d->topk, compress_len, d->n_topk, nullptr, 0)) return false;
      }
    }
    if (emitted) {
      if (ds41_v41_rope_tail(d->latent, 1, kHeadDim, kRot, freq, group_pos, 0) ||
          ds41_v41_fp4_fake_quant(d->latent, kHeadDim, 16, 1) ||
          !copy_row(s.comp + (uint64_t)(pos / ratio) * kHeadDim, d->latent, kHeadDim))
        return false;
    }
    comp = source.comp;
    n_comp = d->n_topk;
  }

  if (ds41_v41_attention(d->q, s.window, n_window, comp, d->topk, n_comp, f32(d, l->attn_sinks), d->o, DS41_N_HEAD,
                         kHeadDim, 1.0f / sqrtf((float)kHeadDim)) ||
      ds41_v41_rope_tail(d->o, DS41_N_HEAD, kHeadDim, kRot, freq, pos, 1) || ds41_v41_fp8_fake_quant(d->o, kQDim))
    return false;

  const ds41_tensor* wo_a = l->attn_output_a;
  const uint64_t group_bytes = wo_a->bytes / DS41_N_OUT_GROUP;
  const uint8_t* wo_a_dev = (const uint8_t*)dev(d, wo_a);
  if (!wo_a_dev) return false;
  for (uint32_t g = 0; g < DS41_N_OUT_GROUP; g++) {
    if (ds41_mmq_dense_vec_typed((int)wo_a->type, wo_a_dev + g * group_bytes, d->o + (uint64_t)g * kGroupDim,
                                 d->oa + (uint64_t)g * DS41_N_LORA_O, (int)DS41_N_LORA_O, 1, (int)kGroupDim, 0))
      return false;
  }
  return linear(d, l->attn_output_b, d->oa, d->attn_out);
}

bool moe(ds41_decoder* d, uint32_t il) {
  const ds41_layer_weights* l = &d->engine->weights->layer[il];
  const int used = (int)DS41_N_EXPERT_USED;
  if (!linear(d, l->ffn_gate_inp, d->xn, d->router) ||
      ds41_v41_route(d->router, f32(d, l->ffn_exp_probs_b), d->route_ids, d->route_w, DS41_N_EXPERT, used,
                     DS41_EXPERT_WEIGHT_SCALE))
    return false;
  const void* gate = dev(d, l->ffn_gate_exps);
  const void* up = dev(d, l->ffn_up_exps);
  const void* down = dev(d, l->ffn_down_exps);
  if (!gate || !up || !down) return false;
  if (ds41_mmq_moe_vec_typed((int)l->ffn_gate_exps->type, gate, d->xn, d->route_ids, d->g6, DS41_N_FF_EXP, kDim, 1,
                             DS41_N_EXPERT, used, 0) ||
      ds41_mmq_moe_vec_typed((int)l->ffn_up_exps->type, up, d->xn, d->route_ids, d->u6, DS41_N_FF_EXP, kDim, 1,
                             DS41_N_EXPERT, used, 0) ||
      ds41_v41_swiglu(d->g6, d->u6, d->route_w, d->mid6, used, DS41_N_FF_EXP, kSwigluLimit) ||
      ds41_mmq_moe_vec_typed((int)l->ffn_down_exps->type, down, d->mid6, d->route_ids, d->down6, kDim, DS41_N_FF_EXP,
                             used, DS41_N_EXPERT, 1, 0))
    return false;
  if (!linear(d, l->ffn_gate_shexp, d->xn, d->sg) || !linear(d, l->ffn_up_shexp, d->xn, d->su) ||
      ds41_v41_swiglu(d->sg, d->su, nullptr, d->sm, 1, DS41_N_FF_EXP, kSwigluLimit) ||
      !linear(d, l->ffn_down_shexp, d->sm, d->sd))
    return false;
  return ds41_v41_sum_rows(d->down6, d->sd, d->moe_out, used, kDim) == 0;
}

bool block(ds41_decoder* d, uint32_t il, int prev, int next) {
  const ds41_layer_weights* l = &d->engine->weights->layer[il];
  const float eps = DS41_RMS_EPS;
  /* Attention uses the pre-mix of the previous FFN (shifted mHC). */
  if (!hc_mix(d, l->hc_attn_fn, l->hc_attn_scale, l->hc_attn_base, d->h, d->pre[next], d->post, d->comb) ||
      ds41_v41_hc_collapse(d->h, d->pre[prev], d->x, kHc, kDim) ||
      ds41_v41_rms_norm(d->x, f32(d, l->attn_norm), d->xn, 1, kDim, eps) || !attention(d, il) ||
      ds41_v41_hc_expand(d->attn_out, d->h, d->post, d->comb, d->h2, kHc, kDim))
    return false;
  /* The FFN uses this layer's attention pre-mix; its own pre-mix feeds the next attention. */
  if (!hc_mix(d, l->hc_ffn_fn, l->hc_ffn_scale, l->hc_ffn_base, d->h2, d->pre[prev], d->ffn_post, d->ffn_comb) ||
      ds41_v41_hc_collapse(d->h2, d->pre[next], d->x, kHc, kDim) ||
      ds41_v41_rms_norm(d->x, f32(d, l->ffn_norm), d->xn, 1, kDim, eps) || !moe(d, il) ||
      ds41_v41_hc_expand(d->moe_out, d->h2, d->ffn_post, d->ffn_comb, d->h, kHc, kDim))
    return false;
  return true;
}

}  // namespace

ds41_decoder* ds41_decoder_create(ds41_engine* engine, uint32_t context) {
  if (!engine || !engine->rocm_ready || context == 0) return nullptr;
  auto* d = new ds41_decoder();
  d->engine = engine;
  d->context = context;
  bool ok = true;
  for (uint32_t il = 0; il < DS41_N_LAYER && ok; il++) {
    LayerState& s = d->layer[il];
    s.window = alloc<float>(d, (uint64_t)DS41_N_SWA * kHeadDim);
    ok = s.window != nullptr;
    const uint32_t ratio = ds41_layer_compress_ratio(il);
    if (ok && ds41_layer_is_kv_source(il)) {
      const uint64_t rows = context / ratio + 1;
      s.comp = alloc<float>(d, rows * kHeadDim);
      s.index_k = alloc<float>(d, rows * DS41_N_INDEXER_HEAD_DIM);
      s.pool_kv = alloc<float>(d, (uint64_t)ratio * kHeadDim);
      s.pool_score = alloc<float>(d, (uint64_t)ratio * kHeadDim);
      ok = s.comp && s.index_k && s.pool_kv && s.pool_score;
    }
  }
  const std::vector<float> plain = rope_inv_freq(DS41_ROPE_FREQ_BASE, 0);
  const std::vector<float> compress = rope_inv_freq(DS41_COMPRESS_ROPE_FREQ_BASE, DS41_ROPE_ORIG_CTX);
  d->freq_plain = alloc<float>(d, plain.size());
  d->freq_compress = alloc<float>(d, compress.size());
  ok = ok && d->freq_plain && d->freq_compress &&
       hipMemcpy(d->freq_plain, plain.data(), plain.size() * 4, hipMemcpyHostToDevice) == hipSuccess &&
       hipMemcpy(d->freq_compress, compress.data(), compress.size() * 4, hipMemcpyHostToDevice) == hipSuccess;

  const uint32_t hd = kHc * kDim;
  struct Buf {
    float** p;
    uint64_t n;
  } floats[] = {
      {&d->h, hd},
      {&d->h2, hd},
      {&d->x, kDim},
      {&d->xn, kDim},
      {&d->mixes, kMixes},
      {&d->pre[0], kHc},
      {&d->pre[1], kHc},
      {&d->post, kHc},
      {&d->comb, kHc * kHc},
      {&d->ffn_post, kHc},
      {&d->ffn_comb, kHc * kHc},
      {&d->qa, DS41_N_LORA_Q},
      {&d->qr, DS41_N_LORA_Q},
      {&d->q, kQDim},
      {&d->kv, kHeadDim},
      {&d->kvn, kHeadDim},
      {&d->ckv, kHeadDim},
      {&d->cscore, kHeadDim},
      {&d->craw, kHeadDim},
      {&d->latent, kHeadDim},
      {&d->ik, DS41_N_INDEXER_HEAD_DIM},
      {&d->ikn, DS41_N_INDEXER_HEAD_DIM},
      {&d->iq, kIndexQDim},
      {&d->iw, DS41_N_INDEXER_HEAD},
      {&d->scores, (uint64_t)context + 1},
      {&d->block_scores, (uint64_t)context / DS41_CANDIDATE_BLOCK_SIZE + 2},
      {&d->o, kQDim},
      {&d->oa, (uint64_t)DS41_N_OUT_GROUP * DS41_N_LORA_O},
      {&d->attn_out, kDim},
      {&d->router, DS41_N_EXPERT},
      {&d->route_w, DS41_N_EXPERT_USED},
      {&d->g6, (uint64_t)DS41_N_EXPERT_USED * DS41_N_FF_EXP},
      {&d->u6, (uint64_t)DS41_N_EXPERT_USED * DS41_N_FF_EXP},
      {&d->mid6, (uint64_t)DS41_N_EXPERT_USED * DS41_N_FF_EXP},
      {&d->down6, (uint64_t)DS41_N_EXPERT_USED * kDim},
      {&d->sg, DS41_N_FF_EXP},
      {&d->su, DS41_N_FF_EXP},
      {&d->sm, DS41_N_FF_EXP},
      {&d->sd, kDim},
      {&d->moe_out, kDim},
      {&d->engram_in, kEngramIn},
      {&d->engram_kv, kEngramOut},
      {&d->logits, DS41_N_VOCAB},
  };
  for (const Buf& b : floats) {
    if (!ok) break;
    *b.p = alloc<float>(d, b.n);
    ok = *b.p != nullptr;
  }
  if (ok) {
    d->candidates = alloc<uint8_t>(d, (uint64_t)context + 1);
    d->topk = alloc<uint32_t>(d, DS41_N_INDEXER_TOP_K);
    d->block_select = alloc<uint32_t>(d, DS41_CANDIDATE_TOPK_BLOCKS);
    d->route_ids = alloc<int32_t>(d, DS41_N_EXPERT_USED);
    ok = d->candidates && d->topk && d->block_select && d->route_ids;
  }
  if (!ok) {
    fprintf(stderr, "ds41: decoder allocation failed\n");
    ds41_decoder_destroy(d);
    return nullptr;
  }
  d->host_h.resize(hd);
  d->host_ids.resize(DS41_N_EXPERT_USED);
  return d;
}

void ds41_decoder_destroy(ds41_decoder* d) {
  if (!d) return;
  for (void* p : d->allocations) (void)hipFree(p);
  delete d;
}

uint32_t ds41_decoder_pos(const ds41_decoder* d) { return d ? d->pos : 0; }

uint32_t ds41_decoder_context(const ds41_decoder* d) { return d ? d->context : 0; }

namespace {

/* Every device range that carries state across steps, with its live bytes. */
template <typename Fn>
bool for_each_state(const ds41_decoder* d, uint32_t pos, Fn&& fn) {
  for (uint32_t il = 0; il < DS41_N_LAYER; il++) {
    const LayerState& s = d->layer[il];
    const uint32_t ratio = ds41_layer_compress_ratio(il);
    if (!fn(s.window, (uint64_t)DS41_N_SWA * kHeadDim * sizeof(float))) return false;
    if (!s.comp) continue;
    const uint64_t rows = pos / ratio;
    if (!fn(s.comp, rows * kHeadDim * sizeof(float)) ||
        !fn(s.index_k, rows * DS41_N_INDEXER_HEAD_DIM * sizeof(float)) ||
        !fn(s.pool_kv, (uint64_t)ratio * kHeadDim * sizeof(float)) ||
        !fn(s.pool_score, (uint64_t)ratio * kHeadDim * sizeof(float)))
      return false;
  }
  return true;
}

constexpr uint64_t kStateMagic = 0x3176615453313453ull;  // "S4STav1"

}  // namespace

uint64_t ds41_decoder_state_bytes(const ds41_decoder* d) {
  if (!d) return 0;
  uint64_t bytes = 4 * sizeof(uint64_t) + d->engram_history.size() * sizeof(int64_t);
  (void)for_each_state(d, d->pos, [&](void*, uint64_t n) {
    bytes += n;
    return true;
  });
  return bytes;
}

int ds41_decoder_save(const ds41_decoder* d, std::vector<uint8_t>* out) {
  if (!d || !out) return 1;
  const uint64_t header[4] = {kStateMagic, d->pos, (uint64_t)(int64_t)d->comp_layer,
                              (uint64_t)d->engram_history.size()};
  out->insert(out->end(), (const uint8_t*)header, (const uint8_t*)(header + 4));
  out->insert(out->end(), (const uint8_t*)d->engram_history.data(),
              (const uint8_t*)(d->engram_history.data() + d->engram_history.size()));
  return for_each_state(d, d->pos, [&](void* p, uint64_t bytes) {
           const size_t at = out->size();
           out->resize(at + bytes);
           return hipMemcpy(out->data() + at, p, bytes, hipMemcpyDeviceToHost) == hipSuccess;
         })
             ? 0
             : 1;
}

int ds41_decoder_load(ds41_decoder* d, const uint8_t* data, uint64_t size, uint64_t* consumed) {
  if (!d || !data || size < 32) return 1;
  uint64_t header[4];
  memcpy(header, data, sizeof(header));
  if (header[0] != kStateMagic || header[1] > d->context || header[3] != header[1]) return 1;
  uint64_t at = sizeof(header);
  if (size - at < header[3] * sizeof(int64_t)) return 1;
  std::vector<int64_t> history(header[3]);
  memcpy(history.data(), data + at, header[3] * sizeof(int64_t));
  at += header[3] * sizeof(int64_t);
  const bool ok = for_each_state(d, (uint32_t)header[1], [&](void* p, uint64_t bytes) {
    if (size - at < bytes) return false;
    const bool copied = hipMemcpy(p, data + at, bytes, hipMemcpyHostToDevice) == hipSuccess;
    at += bytes;
    return copied;
  });
  if (!ok) return 1;
  d->pos = (uint32_t)header[1];
  d->comp_layer = (int)(int64_t)header[2];
  d->n_topk = 0;
  d->engram_history = std::move(history);
  if (consumed) *consumed = at;
  return 0;
}

int ds41_decoder_reset(ds41_decoder* d) {
  if (!d) return 1;
  for (uint32_t il = 0; il < DS41_N_LAYER; il++) {
    const LayerState& s = d->layer[il];
    const uint32_t ratio = ds41_layer_compress_ratio(il);
    if (hipMemset(s.window, 0, (size_t)DS41_N_SWA * kHeadDim * sizeof(float)) != hipSuccess) return 1;
    if (s.pool_kv && (hipMemset(s.pool_kv, 0, (size_t)ratio * kHeadDim * sizeof(float)) != hipSuccess ||
                      hipMemset(s.pool_score, 0, (size_t)ratio * kHeadDim * sizeof(float)) != hipSuccess))
      return 1;
  }
  d->pos = 0;
  d->n_topk = 0;
  d->comp_layer = -1;
  d->engram_history.clear();
  return 0;
}

int ds41_decoder_step(ds41_decoder* d, int token, float* logits, ds41_decoder_layer_fn layer_fn, void* ctx) {
  if (!d || token < 0 || token >= (int)DS41_N_VOCAB || d->pos >= d->context) return 1;
  const ds41_weights* w = d->engine->weights;

  int64_t engram_rows[DS41_N_ENGRAM_LAYER][DS41_ENGRAM_COLS];
  engram_hash(d, token, engram_rows);

  /* Embedding, replicated over the hc residual streams; layer 0 pre-mix is one-hot. */
  const ds41_tensor* embd = w->token_embd;
  const uint64_t row_bytes = embd->bytes / DS41_N_VOCAB;
  std::vector<float> row(kDim);
  if (!dequant_row(embd->type, host(d, embd) + (uint64_t)token * row_bytes, row.data(), kDim)) return 1;
  for (uint32_t c = 0; c < kHc; c++) memcpy(d->host_h.data() + c * kDim, row.data(), kDim * sizeof(float));
  const float one_hot[kHc] = {1.0f};
  if (hipMemcpy(d->h, d->host_h.data(), d->host_h.size() * sizeof(float), hipMemcpyHostToDevice) != hipSuccess ||
      hipMemcpy(d->pre[0], one_hot, sizeof(one_hot), hipMemcpyHostToDevice) != hipSuccess)
    return 1;

  int prev = 0;
  for (uint32_t il = 0; il < DS41_N_LAYER; il++) {
    const int slot = ds41_layer_engram_slot(il);
    if (slot >= 0 && !engram(d, il, engram_rows[slot])) return 1;
    if (!block(d, il, prev, prev ^ 1)) return 1;
    /* block() leaves this layer's FFN pre-mix in pre[prev]. */
    if (layer_fn) {
      if (hipMemcpy(d->host_h.data(), d->h, d->host_h.size() * sizeof(float), hipMemcpyDeviceToHost) != hipSuccess ||
          hipMemcpy(d->host_ids.data(), d->route_ids, d->host_ids.size() * sizeof(int32_t), hipMemcpyDeviceToHost) !=
              hipSuccess)
        return 1;
      layer_fn(ctx, il, d->host_h.data(), d->host_ids.data());
    }
  }
  if (ds41_v41_hc_collapse(d->h, d->pre[prev], d->x, kHc, kDim) ||
      ds41_v41_rms_norm(d->x, f32(d, w->output_norm), d->xn, 1, kDim, DS41_RMS_EPS) || !linear(d, w->output, d->xn, d->logits))
    return 1;
  if (logits && hipMemcpy(logits, d->logits, DS41_N_VOCAB * sizeof(float), hipMemcpyDeviceToHost) != hipSuccess) return 1;
  if (hipDeviceSynchronize() != hipSuccess) return 1;
  d->pos++;
  return 0;
}
