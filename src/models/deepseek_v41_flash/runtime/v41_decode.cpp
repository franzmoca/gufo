/* DeepSeek V4.1 Flash decoder.
 *
 * Follows the official inference/model.py: Engram (layers 1 and 14), shifted
 * mHC, CSA2 KV/index sharing, sparse attention with sinks, and the routed +
 * shared MoE. A batch of tokens runs layer-major, so prompts share every
 * weight read; decode is a batch of one. Quantized projections run on the
 * vendored mmvq (up to 8 tokens) and mmq kernels; everything else runs on
 * the kernels in kernels/rocm/ds41_v41.hip.cpp. */
#include <hip/hip_runtime_api.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <atomic>
#include <cstring>
#include <future>
#include <thread>
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
constexpr uint32_t kEngramPinnedTokens = 16;
constexpr uint64_t kEngramRowsPerThread = 4;
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
  float* window = nullptr;    // [DS41_N_SWA, head_dim] ring, slot pos % DS41_N_SWA
  float* comp = nullptr;      // [context / ratio, head_dim] (KV sources)
  float* index_k = nullptr;   // [context / ratio, indexer head_dim] (KV sources)
  float* pool_kv = nullptr;   // [ratio, head_dim] compressor frontier
  float* pool_score = nullptr;
};

/* Batch capacity: up to 2048 tokens, so the routed expert tiles average 32
 * rows per expert, fewer when the activations (about 1 MiB per token) would
 * leave less than kDeviceReserve of device memory free. On gfx1151 the
 * reported free memory runs about 1 GiB below MemAvailable, so this keeps
 * about 2 GiB of RAM for the MMQ scratch and the rest of the system. */
constexpr uint32_t kMaxBatch = 2048;
constexpr uint64_t kDeviceReserve = 1ull << 30;

/* Indexer score rows (context + 1 floats per token) run in groups of up to
 * this many tokens, bounding the score buffer to 256 MiB at any context. */
uint32_t score_rows(uint64_t score_stride) {
  const uint64_t rows = (256ull << 20) / (score_stride * sizeof(float));
  return (uint32_t)std::clamp<uint64_t>(rows / 16 * 16, 16, kMaxBatch);
}

constexpr uint32_t kLogitRows = 64;  // output projection rows per pass

}  // namespace

struct ds41_decoder {
  ds41_engine* engine = nullptr;
  uint32_t context = 0;
  uint32_t capacity = 0;
  uint32_t score_rows = 0;  // tokens per indexer score group
  uint32_t pos = 0;
  uint64_t score_stride = 0;
  uint64_t block_stride = 0;
  std::vector<void*> allocations;
  LayerState layer[DS41_N_LAYER];
  float* freq_plain = nullptr;
  float* freq_compress = nullptr;

  /* Per-token activations, `capacity` rows each. */
  float *h = nullptr, *h2 = nullptr, *xn = nullptr;
  float *mixes = nullptr, *pre[2] = {}, *post = nullptr, *comb = nullptr, *ffn_post = nullptr, *ffn_comb = nullptr;
  float *qa = nullptr, *qr = nullptr, *q = nullptr, *kv = nullptr, *kvn = nullptr;
  float *ckv = nullptr, *cscore = nullptr, *craw = nullptr, *latent = nullptr;
  float *ik = nullptr, *ikn = nullptr, *iq = nullptr, *iw = nullptr;
  float *scores = nullptr, *block_scores = nullptr;  // score_rows rows each
  uint8_t* candidates = nullptr;                     // [capacity, block_stride] candidate-block mask
  uint32_t *topk = nullptr, *block_select = nullptr;
  float *attn_partial = nullptr;
  float *o = nullptr, *o_group = nullptr, *oa_group = nullptr, *oa = nullptr, *attn_out = nullptr;
  float *router = nullptr, *route_w = nullptr;
  int32_t* route_ids = nullptr;
  float *g6 = nullptr, *u6 = nullptr, *mid6 = nullptr, *down6 = nullptr;
  float *sg = nullptr, *su = nullptr, *sm = nullptr, *sd = nullptr;
  float *engram_in = nullptr, *engram_kv = nullptr, *logits = nullptr;
  /* Pinned staging for the Engram rows of batches up to kEngramPinnedTokens,
   * per Engram layer, so the upload does not wait for the queue to drain.
   * Each evaluation ends with a device synchronize, so the next one finds
   * the buffers idle. */
  float* engram_pinned[DS41_N_ENGRAM_LAYER] = {};

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

bool mmq_batched(uint32_t type) {
  switch (type) {
  case DS41_TENSOR_Q8_0:
  case DS41_TENSOR_Q5_K:
  case DS41_TENSOR_IQ1_S:
  case DS41_TENSOR_IQ1_M:
  case DS41_TENSOR_IQ2_XXS:
  case DS41_TENSOR_IQ2_S:
    return true;
  default:
    return false;
  }
}

/* Waves per row group of the single-token GEMV, from a slice sweep over the
 * V4.1 shapes: short rows split fewer ways, so more rows run per block. */
int gemv_slices(uint32_t rows, uint32_t k) {
  if (k <= 2304) return 2;
  if (k <= 5120 && rows <= 4096) return 4;
  return 8;
}

bool gemv_type(uint32_t type) { return type == DS41_TENSOR_Q5_K || type == DS41_TENSOR_Q8_0; }

/* out[t] = W x[t] for n tokens and a [rows, k] weight at `p` of `type`. */
bool linear_raw(uint32_t type, const void* p, uint32_t rows, uint32_t k, const float* x, float* out, uint32_t n) {
  if (!p) return false;
  switch (type) {
  case DS41_TENSOR_BF16:
    return ds41_v41_matmul_bf16((const uint16_t*)p, x, out, rows, k, n) == 0;
  case DS41_TENSOR_F32:
    return ds41_v41_matmul_f32((const float*)p, x, out, rows, k, n) == 0;
  default:
    /* One token: FP32 activations, no Q8_1 pass. */
    if (n == 1 && gemv_type(type)) return ds41_v41_gemv((int)type, p, x, out, rows, k, 0, gemv_slices(rows, k)) == 0;
    if (n > 8 && mmq_batched(type)) return ds41_mmq_dense_typed((int)type, p, x, out, (int)rows, (int)n, (int)k, 0) == 0;
    for (uint32_t t = 0; t < n; t += 8) {
      const uint32_t cols = std::min(8u, n - t);
      if (ds41_mmq_dense_vec_typed((int)type, p, x + (uint64_t)t * k, out + (uint64_t)t * rows, (int)rows, (int)cols,
                                   (int)k, 0))
        return false;
    }
    return true;
  }
}

bool linear(const ds41_decoder* d, const ds41_tensor* w, const float* x, float* out, uint32_t n) {
  const uint32_t k = (uint32_t)w->dim[0];
  return linear_raw(w->type, dev(d, w), (uint32_t)(w->elements / k), k, x, out, n);
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

/* Reads and dequantizes the n * DS41_ENGRAM_COLS table rows of layer il into
 * out. Reads run on parallel threads: each is a small random read, so
 * throughput comes from queue depth. Touches only immutable engine state, so
 * it can run while the GPU works on earlier layers. */
bool engram_read(const ds41_decoder* d, uint32_t il, const int64_t* rows, uint32_t n, float* out) {
  const ds41_layer_weights* l = &d->engine->weights->layer[il];
  const ds41_engram_tables* e = d->engine->engram;
  const ds41_tensor* table = l->engram_embd;
  const uint64_t base = table->abs_offset - e->file.shard_offset[0];
  const uint64_t count = (uint64_t)n * DS41_ENGRAM_COLS;
  std::atomic<uint64_t> next{0};
  std::atomic<bool> ok{true};
  auto worker = [&] {
    std::vector<uint8_t> raw(e->row_bytes);
    for (uint64_t i = next++; i < count && ok; i = next++) {
      const off_t at = (off_t)(base + (uint64_t)rows[i] * e->row_bytes);
      if (pread(e->file.shard_fd[0], raw.data(), raw.size(), at) != (ssize_t)raw.size() ||
          !dequant_row(table->type, raw.data(), out + i * DS41_ENGRAM_HEAD_DIM, DS41_ENGRAM_HEAD_DIM))
        ok = false;
    }
  };
  const uint32_t threads = (uint32_t)std::min<uint64_t>(32, (count + kEngramRowsPerThread - 1) / kEngramRowsPerThread);
  std::vector<std::thread> pool;
  for (uint32_t t = 1; t < threads; t++) pool.emplace_back(worker);
  worker();
  for (auto& t : pool) t.join();
  if (!ok) fprintf(stderr, "ds41: Engram row read failed\n");
  return ok;
}

/* Adds layer il's Engram term from rows read by engram_read. */
bool engram(ds41_decoder* d, uint32_t il, const float* rows, uint32_t n) {
  const ds41_layer_weights* l = &d->engine->weights->layer[il];
  const uint64_t bytes = (uint64_t)n * DS41_ENGRAM_COLS * DS41_ENGRAM_HEAD_DIM * sizeof(float);
  const hipError_t err = n <= kEngramPinnedTokens
                             ? hipMemcpyAsync(d->engram_in, rows, bytes, hipMemcpyHostToDevice, 0)
                             : hipMemcpy(d->engram_in, rows, bytes, hipMemcpyHostToDevice);
  if (err != hipSuccess) return false;
  if (!linear(d, l->engram_wkv, d->engram_in, d->engram_kv, n)) return false;
  return ds41_v41_engram_gate(d->h, d->engram_kv, (const uint16_t*)dev(d, l->engram_q),
                              (const uint16_t*)dev(d, l->engram_k), kHc, kDim, DS41_RMS_EPS, n) == 0;
}

bool hc_mix(ds41_decoder* d, const ds41_tensor* fn, const ds41_tensor* scale, const ds41_tensor* base,
            const float* h, float* pre, float* post, float* comb, uint32_t n) {
  return ds41_v41_hc_mix(f32(d, fn), h, f32(d, scale), f32(d, base), d->mixes, pre, post, comb, kHc, kDim,
                         DS41_RMS_EPS, DS41_N_HC_SINKHORN_ITER, DS41_HC_EPS, n) == 0;
}

bool copy_dev(float* dst, const float* src, uint64_t count) {
  return count == 0 ||
         hipMemcpyAsync(dst, src, count * sizeof(float), hipMemcpyDeviceToDevice, 0) == hipSuccess;
}

/* Keeps the newest DS41_N_SWA of the batch's window rows in the ring. */
bool store_window(LayerState& s, const float* rows, uint32_t pos0, uint32_t n) {
  const uint32_t first = n > DS41_N_SWA ? n - DS41_N_SWA : 0;
  for (uint32_t t = first; t < n;) {
    const uint32_t slot = (pos0 + t) % DS41_N_SWA;
    const uint32_t run = std::min(n - t, DS41_N_SWA - slot);
    if (!copy_dev(s.window + (uint64_t)slot * kHeadDim, rows + (uint64_t)t * kHeadDim, (uint64_t)run * kHeadDim))
      return false;
    t += run;
  }
  return true;
}

bool attention(ds41_decoder* d, uint32_t il, uint32_t n) {
  const ds41_layer_weights* l = &d->engine->weights->layer[il];
  LayerState& s = d->layer[il];
  const uint32_t pos0 = d->pos;
  const uint32_t ratio = ds41_layer_compress_ratio(il);
  const float* freq = layer_freq(d, il);
  const float eps = DS41_RMS_EPS;

  /* q_a and kv share the input; one token takes both in one launch. */
  const ds41_tensor* wqa = l->attn_q_a;
  const ds41_tensor* wkv = l->attn_kv;
  const bool pair = n == 1 && wqa->type == wkv->type && wqa->dim[0] == wkv->dim[0] && gemv_type(wqa->type);
  if (pair) {
    const uint32_t k = (uint32_t)wqa->dim[0];
    const uint32_t m = (uint32_t)((wqa->elements + wkv->elements) / k);
    if (ds41_v41_gemv_pair((int)wqa->type, dev(d, wqa), d->qa, DS41_N_LORA_Q, dev(d, wkv), d->kv, kHeadDim, d->xn, k,
                           gemv_slices(m, k)))
      return false;
  } else if (!linear(d, wqa, d->xn, d->qa, n) || !linear(d, wkv, d->xn, d->kv, n)) {
    return false;
  }
  if (ds41_v41_q_kv_norm(d->qa, f32(d, l->attn_q_a_norm), d->qr, DS41_N_LORA_Q, d->kv, f32(d, l->attn_kv_a_norm),
                         d->kvn, kHeadDim, kRot, freq, pos0, n, eps) ||
      !linear(d, l->attn_q_b, d->qr, d->q, n) ||
      ds41_v41_rope_tail(d->q, n * DS41_N_HEAD, kHeadDim, kRot, freq, pos0, DS41_N_HEAD, 1, 0))
    return false;

  const float* comp = nullptr;
  if (ratio) {
    /* Compressed rows R0 .. R0 + emitted - 1 complete inside this batch. */
    const uint32_t r0 = pos0 / ratio;
    const uint32_t emitted = (pos0 + n) / ratio - r0;
    const bool source_layer = ds41_layer_is_kv_source(il);
    if (source_layer) {
      if (!linear(d, l->attn_compressor_kv, d->xn, d->ckv, n)) return false;
      if (ratio == 1) {
        if (ds41_v41_rms_norm(d->ckv, f32(d, l->attn_compressor_norm), d->latent, n, kHeadDim, eps)) return false;
      } else if (!linear(d, l->attn_compressor_gate, d->xn, d->cscore, n) ||
                 ds41_v41_compress_pool(d->ckv, d->cscore, s.pool_kv, s.pool_score, pos0, n, ratio, d->craw,
                                        kHeadDim) ||
                 ds41_v41_rms_norm(d->craw, f32(d, l->attn_compressor_norm), d->latent, emitted, kHeadDim, eps)) {
        return false;
      }
      d->comp_layer = (int)il;
    }
    const LayerState& source = d->layer[d->comp_layer];
    if (ds41_layer_is_index_source(il)) {
      if (source_layer && emitted &&
          (!linear(d, l->indexer_attn_k, d->latent, d->ik, emitted) ||
           ds41_v41_rms_norm(d->ik, f32(d, l->indexer_k_norm), d->ikn, emitted, DS41_N_INDEXER_HEAD_DIM, eps) ||
           ds41_v41_rope_tail(d->ikn, emitted, DS41_N_INDEXER_HEAD_DIM, kRot, freq, r0 * ratio, 1, ratio, 0) ||
           ds41_v41_fp4_fake_quant(d->ikn, emitted * DS41_N_INDEXER_HEAD_DIM, 32, 0) ||
           !copy_dev(s.index_k + (uint64_t)r0 * DS41_N_INDEXER_HEAD_DIM, d->ikn,
                     (uint64_t)emitted * DS41_N_INDEXER_HEAD_DIM)))
        return false;
      const float w_scale = 1.0f / sqrtf((float)DS41_N_INDEXER_HEAD_DIM) / sqrtf((float)DS41_N_INDEXER_HEAD);
      if (!linear(d, l->indexer_attn_q_b, d->qr, d->iq, n) ||
          ds41_v41_rope_tail(d->iq, n * DS41_N_INDEXER_HEAD, DS41_N_INDEXER_HEAD_DIM, kRot, freq, pos0,
                             DS41_N_INDEXER_HEAD, 1, 0) ||
          ds41_v41_fp4_fake_quant(d->iq, n * kIndexQDim, 32, 0) || !linear(d, l->indexer_proj, d->xn, d->iw, n))
        return false;
      for (uint32_t t0 = 0; t0 < n; t0 += d->score_rows) {
        const uint32_t m = std::min(d->score_rows, n - t0);
        const uint32_t p0 = pos0 + t0;
        uint8_t* mask = d->candidates + t0 * d->block_stride;
        if (ds41_v41_indexer_score(d->iq + (uint64_t)t0 * kIndexQDim, source.index_k,
                                   d->iw + (uint64_t)t0 * DS41_N_INDEXER_HEAD, w_scale, d->scores, d->score_stride,
                                   p0, m, ratio, DS41_N_INDEXER_HEAD, DS41_N_INDEXER_HEAD_DIM))
          return false;
        if (il == DS41_CANDIDATE_SOURCE_LAYER) {
          if (ds41_v41_candidate_mask(d->scores, d->score_stride, mask, d->block_scores, d->block_stride,
                                      d->block_select, p0, m, ratio, DS41_CANDIDATE_BLOCK_SIZE,
                                      DS41_CANDIDATE_TOPK_BLOCKS))
            return false;
        } else if (il > DS41_CANDIDATE_SOURCE_LAYER &&
                   ds41_v41_apply_mask(d->scores, d->score_stride, mask, d->block_stride, p0, m, ratio,
                                       DS41_CANDIDATE_BLOCK_SIZE)) {
          return false;
        }
        if (ds41_v41_topk(d->scores, d->score_stride, d->topk + (uint64_t)t0 * DS41_N_INDEXER_TOP_K,
                          DS41_N_INDEXER_TOP_K, p0, m, ratio, 1))
          return false;
      }
    }
    if (source_layer && emitted &&
        (ds41_v41_rope_tail(d->latent, emitted, kHeadDim, kRot, freq, r0 * ratio, 1, ratio, 0) ||
         ds41_v41_fp4_fake_quant(d->latent, emitted * kHeadDim, 16, 1) ||
         !copy_dev(s.comp + (uint64_t)r0 * kHeadDim, d->latent, (uint64_t)emitted * kHeadDim)))
      return false;
    comp = source.comp;
  }

  if (ds41_v41_attention(d->q, s.window, d->kvn, DS41_N_SWA, comp, d->topk, DS41_N_INDEXER_TOP_K, ratio,
                         f32(d, l->attn_sinks), d->o, d->attn_partial, pos0, n, DS41_N_HEAD, kHeadDim,
                         1.0f / sqrtf((float)kHeadDim)) ||
      !store_window(s, d->kvn, pos0, n) ||
      ds41_v41_rope_fp8(d->o, n * DS41_N_HEAD, kHeadDim, kRot, freq, pos0, DS41_N_HEAD, 1, 1))
    return false;

  /* Grouped LoRA output: group g maps head slice g of every token. */
  const ds41_tensor* wo_a = l->attn_output_a;
  const uint64_t group_bytes = wo_a->bytes / DS41_N_OUT_GROUP;
  const uint8_t* wo_a_dev = (const uint8_t*)dev(d, wo_a);
  if (!wo_a_dev) return false;
  /* Small batches take all groups in one matvec launch. */
  if (n == 1 && gemv_type(wo_a->type)) {
    if (ds41_v41_gemv((int)wo_a->type, wo_a_dev, d->o, d->oa, DS41_N_OUT_GROUP * DS41_N_LORA_O, kGroupDim,
                      DS41_N_LORA_O, gemv_slices(DS41_N_OUT_GROUP * DS41_N_LORA_O, kGroupDim)))
      return false;
    return linear(d, l->attn_output_b, d->oa, d->attn_out, n);
  }
  if (n <= 8 && mmq_batched(wo_a->type)) {
    if (ds41_mmq_grouped_vec_typed((int)wo_a->type, wo_a_dev, d->o, d->oa, DS41_N_LORA_O, (int)n, kGroupDim,
                                   DS41_N_OUT_GROUP, 0))
      return false;
    return linear(d, l->attn_output_b, d->oa, d->attn_out, n);
  }
  for (uint32_t g = 0; g < DS41_N_OUT_GROUP; g++) {
    const float* in = d->o + (uint64_t)g * kGroupDim;
    float* out = d->oa + (uint64_t)g * DS41_N_LORA_O;
    if (n > 1) {
      if (ds41_v41_copy_rows(d->o_group, kGroupDim, in, kQDim, n, kGroupDim)) return false;
      in = d->o_group;
      out = d->oa_group;
    }
    if (!linear_raw(wo_a->type, wo_a_dev + g * group_bytes, DS41_N_LORA_O, kGroupDim, in, out, n)) return false;
    if (n > 1 && ds41_v41_copy_rows(d->oa + (uint64_t)g * DS41_N_LORA_O, (uint64_t)DS41_N_OUT_GROUP * DS41_N_LORA_O,
                                    d->oa_group, DS41_N_LORA_O, n, DS41_N_LORA_O))
      return false;
  }
  return linear(d, l->attn_output_b, d->oa, d->attn_out, n);
}

/* Bytes per 256 values of the routed-expert types the FP32-activation expert
 * GEMV decodes; 0 for others. */
uint64_t expert_block_bytes(int type) {
  switch ((uint32_t)type) {
  case DS41_TENSOR_IQ1_S:
    return 50;
  case DS41_TENSOR_IQ1_M:
    return 56;
  case DS41_TENSOR_IQ2_XXS:
    return 66;
  case DS41_TENSOR_IQ2_S:
    return 82;
  default:
    return 0;
  }
}

/* Single-token expert projection through the FP32-activation expert GEMV
 * where it measured faster than mmvq: IQ1 always, IQ2 with one input per slot
 * (down). Returns false to fall back to mmvq. */
bool expert_gemv(int type, const void* w, const int32_t* ids, uint32_t slots, const float* x, bool per_slot, float* y,
                 uint32_t m, uint32_t k, int* status) {
  const uint64_t block = expert_block_bytes(type);
  const bool iq1 = type == (int)DS41_TENSOR_IQ1_S || type == (int)DS41_TENSOR_IQ1_M;
  if (block == 0 || (!per_slot && !iq1)) return false;
  *status = ds41_v41_gemv_experts(type, w, (uint64_t)m * k / 256 * block, ids, slots, x, per_slot ? k : 0, y, m, k);
  return true;
}

bool moe(ds41_decoder* d, uint32_t il, uint32_t n) {
  const ds41_layer_weights* l = &d->engine->weights->layer[il];
  const int used = (int)DS41_N_EXPERT_USED;
  if (!linear(d, l->ffn_gate_inp, d->xn, d->router, n) ||
      ds41_v41_route(d->router, f32(d, l->ffn_exp_probs_b), d->route_ids, d->route_w, DS41_N_EXPERT, used,
                     DS41_EXPERT_WEIGHT_SCALE, n))
    return false;
  const void* gate = dev(d, l->ffn_gate_exps);
  const void* up = dev(d, l->ffn_up_exps);
  const void* down = dev(d, l->ffn_down_exps);
  if (!gate || !up || !down) return false;
  const int gate_type = (int)l->ffn_gate_exps->type;
  const int up_type = (int)l->ffn_up_exps->type;
  const int down_type = (int)l->ffn_down_exps->type;
  const int rows = (int)n * used;
  /* Batched passes size their tiles from the real expert buckets: at 512
   * tokens over 384 experts the mean bucket is 8 rows. */
  struct RoutedBounds {
    ~RoutedBounds() {
      ds41_mmq_set_routed_max_expert_rows(0);
      ds41_mmq_set_routed_tile_cols(0);
    }
  } routed_bounds;
  if (n > 1) {
    std::vector<int32_t> ids((size_t)rows);
    std::vector<unsigned int> counts(DS41_N_EXPERT, 0u);
    if (hipMemcpy(ids.data(), d->route_ids, ids.size() * sizeof(int32_t), hipMemcpyDeviceToHost) != hipSuccess)
      return false;
    unsigned int most = 0;
    for (const int32_t e : ids)
      if (e >= 0 && e < (int32_t)DS41_N_EXPERT) most = std::max(most, ++counts[(size_t)e]);
    ds41_mmq_set_routed_max_expert_rows((int)most);
    ds41_mmq_set_routed_tile_cols(ds41_mmq_routed_tile_cols_for_counts(counts.data(), DS41_N_EXPERT));
  }
  /* Single tokens take gate, up and SwiGLU in one pass when gate and up share
   * a type (IQ2_XXS too: the pair runs as fast as two mmvq launches). */
  const bool fused_pair = gate_type == up_type && (gate_type == (int)DS41_TENSOR_IQ1_S ||
                                                   gate_type == (int)DS41_TENSOR_IQ1_M ||
                                                   gate_type == (int)DS41_TENSOR_IQ2_XXS);
  if (n == 1 && fused_pair) {
    if (ds41_v41_gemv_experts_swiglu(gate_type, gate, up,
                                     (uint64_t)DS41_N_FF_EXP * kDim / 256 * expert_block_bytes(gate_type),
                                     d->route_ids, used, d->xn, d->route_w, kSwigluLimit, d->mid6, DS41_N_FF_EXP,
                                     kDim))
      return false;
  } else if (n == 1) {
    int status = 0;
    if (!expert_gemv(gate_type, gate, d->route_ids, used, d->xn, false, d->g6, DS41_N_FF_EXP, kDim, &status))
      status = ds41_mmq_moe_vec_typed(gate_type, gate, d->xn, d->route_ids, d->g6, DS41_N_FF_EXP, kDim, 1,
                                      DS41_N_EXPERT, used, 0);
    if (status) return false;
    if (!expert_gemv(up_type, up, d->route_ids, used, d->xn, false, d->u6, DS41_N_FF_EXP, kDim, &status))
      status = ds41_mmq_moe_vec_typed(up_type, up, d->xn, d->route_ids, d->u6, DS41_N_FF_EXP, kDim, 1, DS41_N_EXPERT,
                                      used, 0);
    if (status) return false;
  } else if (gate_type == up_type) {
    /* Batches quantize the input once and store SwiGLU from the up pass. */
    if (ds41_mmq_moe_swiglu_typed(gate_type, gate, up, d->xn, d->route_ids, d->route_w, d->g6, d->mid6,
                                  DS41_N_FF_EXP, kDim, (int)n, DS41_N_EXPERT, used, kSwigluLimit, 0))
      return false;
  } else if (ds41_mmq_moe_typed(gate_type, gate, d->xn, d->route_ids, d->g6, DS41_N_FF_EXP, kDim, (int)n,
                                DS41_N_EXPERT, used, 0) ||
             ds41_mmq_moe_typed(up_type, up, d->xn, d->route_ids, d->u6, DS41_N_FF_EXP, kDim, (int)n, DS41_N_EXPERT,
                                used, 0)) {
    return false;
  }
  const bool mid_done = (n == 1 && fused_pair) || (n > 1 && gate_type == up_type);
  if (!mid_done && ds41_v41_swiglu(d->g6, d->u6, d->route_w, d->mid6, rows, DS41_N_FF_EXP, kSwigluLimit))
    return false;
  /* Each (token, slot) row multiplies its own expert's down projection. */
  if (n == 1) {
    int status = 0;
    if (!expert_gemv(down_type, down, d->route_ids, rows, d->mid6, true, d->down6, kDim, DS41_N_FF_EXP, &status))
      status = ds41_mmq_moe_vec_typed(down_type, down, d->mid6, d->route_ids, d->down6, kDim, DS41_N_FF_EXP, rows,
                                      DS41_N_EXPERT, 1, 0);
    if (status) return false;
  } else if (ds41_mmq_moe_typed(down_type, down, d->mid6, d->route_ids, d->down6, kDim, DS41_N_FF_EXP, rows,
                                DS41_N_EXPERT, 1, 0)) {
    return false;
  }
  const ds41_tensor* sg = l->ffn_gate_shexp;
  const ds41_tensor* su = l->ffn_up_shexp;
  if (n == 1 && sg->type == su->type && sg->elements == su->elements && gemv_type(sg->type)) {
    const uint32_t k = (uint32_t)sg->dim[0];
    const uint32_t m = (uint32_t)(sg->elements / k);
    if (ds41_v41_gemv_swiglu((int)sg->type, dev(d, sg), dev(d, su), d->xn, d->sm, m, k, kSwigluLimit,
                             gemv_slices(m, k)))
      return false;
  } else if (!linear(d, sg, d->xn, d->sg, n) || !linear(d, su, d->xn, d->su, n) ||
             ds41_v41_swiglu(d->sg, d->su, nullptr, d->sm, n, DS41_N_FF_EXP, kSwigluLimit)) {
    return false;
  }
  /* The routed rows (down6) and the shared output (sd) are summed by block()'s hc expand. */
  return linear(d, l->ffn_down_shexp, d->sm, d->sd, n);
}

bool block(ds41_decoder* d, uint32_t il, int prev, int next, uint32_t n) {
  const ds41_layer_weights* l = &d->engine->weights->layer[il];
  const float eps = DS41_RMS_EPS;
  /* Attention uses the pre-mix of the previous FFN (shifted mHC). */
  if (!hc_mix(d, l->hc_attn_fn, l->hc_attn_scale, l->hc_attn_base, d->h, d->pre[next], d->post, d->comb, n) ||
      ds41_v41_hc_collapse_norm(d->h, d->pre[prev], f32(d, l->attn_norm), d->xn, kHc, kDim, eps, n) ||
      !attention(d, il, n) ||
      ds41_v41_hc_expand(d->attn_out, d->h, d->post, d->comb, d->h2, kHc, kDim, n))
    return false;
  /* The FFN uses this layer's attention pre-mix; its own pre-mix feeds the next attention. */
  if (!hc_mix(d, l->hc_ffn_fn, l->hc_ffn_scale, l->hc_ffn_base, d->h2, d->pre[prev], d->ffn_post, d->ffn_comb, n) ||
      ds41_v41_hc_collapse_norm(d->h2, d->pre[next], f32(d, l->ffn_norm), d->xn, kHc, kDim, eps, n) ||
      !moe(d, il, n) ||
      ds41_v41_hc_expand_sum(d->down6, d->sd, DS41_N_EXPERT_USED, d->h2, d->ffn_post, d->ffn_comb, d->h, kHc, kDim,
                             n))
    return false;
  return true;
}

}  // namespace

ds41_decoder* ds41_decoder_create(ds41_engine* engine, uint32_t context) {
  if (!engine || !engine->rocm_ready || context == 0) return nullptr;
  auto* d = new ds41_decoder();
  d->engine = engine;
  d->context = context;
  d->score_stride = (uint64_t)context + 1;
  d->block_stride = (uint64_t)context / DS41_CANDIDATE_BLOCK_SIZE + 2;
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

  const uint64_t hd = (uint64_t)kHc * kDim;
  const uint64_t routed = (uint64_t)DS41_N_EXPERT_USED * DS41_N_FF_EXP;
  struct Buf {
    float** p;
    uint64_t n;
  };
  const auto float_buffers = [&](uint64_t cap, uint64_t rows) -> std::vector<Buf> {
    return {
        {&d->h, cap * hd},
        {&d->h2, cap * hd},
        {&d->xn, cap * kDim},
        {&d->mixes, cap * kMixes},
        {&d->pre[0], cap * kHc},
        {&d->pre[1], cap * kHc},
        {&d->post, cap * kHc},
        {&d->comb, cap * kHc * kHc},
        {&d->ffn_post, cap * kHc},
        {&d->ffn_comb, cap * kHc * kHc},
        {&d->qa, cap * DS41_N_LORA_Q},
        {&d->qr, cap * DS41_N_LORA_Q},
        {&d->q, cap * kQDim},
        {&d->kv, cap * kHeadDim},
        {&d->kvn, cap * kHeadDim},
        {&d->ckv, cap * kHeadDim},
        {&d->cscore, cap * kHeadDim},
        {&d->craw, cap * kHeadDim},
        {&d->latent, cap * kHeadDim},
        {&d->ik, cap * DS41_N_INDEXER_HEAD_DIM},
        {&d->ikn, cap * DS41_N_INDEXER_HEAD_DIM},
        {&d->iq, cap * kIndexQDim},
        {&d->iw, cap * DS41_N_INDEXER_HEAD},
        {&d->scores, rows * d->score_stride},
        {&d->block_scores, rows * d->block_stride},
        {&d->o, cap * kQDim},
        {&d->o_group, cap * kGroupDim},
        {&d->oa_group, cap * DS41_N_LORA_O},
        {&d->oa, cap * DS41_N_OUT_GROUP * DS41_N_LORA_O},
        {&d->attn_out, cap * kDim},
        {&d->router, cap * DS41_N_EXPERT},
        {&d->route_w, cap * DS41_N_EXPERT_USED},
        {&d->g6, cap * routed},
        {&d->u6, cap * routed},
        {&d->mid6, cap * routed},
        {&d->down6, cap * DS41_N_EXPERT_USED * kDim},
        {&d->sg, cap * DS41_N_FF_EXP},
        {&d->su, cap * DS41_N_FF_EXP},
        {&d->sm, cap * DS41_N_FF_EXP},
        {&d->sd, cap * kDim},
        {&d->engram_in, cap * kEngramIn},
        {&d->engram_kv, cap * kEngramOut},
        {&d->logits, (uint64_t)kLogitRows * DS41_N_VOCAB},
      {&d->attn_partial, (uint64_t)(DS41_V41_ATTN_SPLIT_TOKENS - 1) * DS41_N_HEAD * DS41_V41_ATTN_PARTIAL_PER_HEAD},
    };
  };
  const auto float_count = [&](uint64_t cap, uint64_t rows) {
    uint64_t total = 0;
    for (const Buf& b : float_buffers(cap, rows)) total += b.n;
    return total;
  };
  /* Size the batch from the memory left after the KV state: per token, the
   * float activations, the candidate mask, top-k and route ids, and host_h. */
  const uint32_t rows = score_rows(d->score_stride);
  const uint64_t per_token = (float_count(1, rows) - float_count(0, rows)) * sizeof(float) + d->block_stride +
                             DS41_N_INDEXER_TOP_K * sizeof(uint32_t) + DS41_N_EXPERT_USED * sizeof(int32_t) +
                             hd * sizeof(float);
  const uint64_t fixed = float_count(0, rows) * sizeof(float) + rows * DS41_CANDIDATE_TOPK_BLOCKS * sizeof(uint32_t);
  size_t free_bytes = 0, total_bytes = 0;
  if (ok && hipMemGetInfo(&free_bytes, &total_bytes) != hipSuccess) ok = false;
  const uint64_t budget = free_bytes > kDeviceReserve + fixed ? free_bytes - kDeviceReserve - fixed : 0;
  d->capacity = (uint32_t)std::clamp<uint64_t>(budget / per_token / 16 * 16, 16, kMaxBatch);
  d->score_rows = std::min(rows, d->capacity);
  const uint64_t cap = d->capacity;
  for (const Buf& b : float_buffers(cap, d->score_rows)) {
    if (!ok) break;
    *b.p = alloc<float>(d, b.n);
    ok = *b.p != nullptr;
  }
  if (ok) {
    d->candidates = alloc<uint8_t>(d, cap * d->block_stride);
    d->topk = alloc<uint32_t>(d, cap * DS41_N_INDEXER_TOP_K);
    d->block_select = alloc<uint32_t>(d, (uint64_t)d->score_rows * DS41_CANDIDATE_TOPK_BLOCKS);
    d->route_ids = alloc<int32_t>(d, cap * DS41_N_EXPERT_USED);
    ok = d->candidates && d->topk && d->block_select && d->route_ids;
  }
  if (!ok) {
    fprintf(stderr, "ds41: decoder allocation failed\n");
    ds41_decoder_destroy(d);
    return nullptr;
  }
  for (uint32_t slot = 0; slot < DS41_N_ENGRAM_LAYER; slot++) {
    if (hipHostMalloc(&d->engram_pinned[slot], (uint64_t)kEngramPinnedTokens * kEngramIn * sizeof(float)) !=
        hipSuccess) {
      fprintf(stderr, "ds41: decoder allocation failed\n");
      ds41_decoder_destroy(d);
      return nullptr;
    }
  }
  d->host_h.resize(cap * hd);
  d->host_ids.resize(DS41_N_EXPERT_USED);
  return d;
}

void ds41_decoder_destroy(ds41_decoder* d) {
  if (!d) return;
  for (void* p : d->allocations) (void)hipFree(p);
  for (float* p : d->engram_pinned)
    if (p) (void)hipHostFree(p);
  delete d;
}

uint32_t ds41_decoder_batch_capacity(const ds41_decoder* d) { return d ? d->capacity : 0; }

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
  d->comp_layer = -1;
  d->engram_history.clear();
  return 0;
}

int ds41_decoder_eval(ds41_decoder* d, const int* tokens, uint32_t n, float* logits, uint32_t logit_rows,
                      ds41_decoder_layer_fn layer_fn, void* ctx) {
  if (!d || !tokens || n == 0 || n > d->capacity || logit_rows > n || (logit_rows && !logits) ||
      d->pos + n > d->context)
    return 1;
  for (uint32_t t = 0; t < n; t++)
    if (tokens[t] < 0 || tokens[t] >= (int)DS41_N_VOCAB) return 1;
  const ds41_weights* w = d->engine->weights;
  const uint64_t hd = (uint64_t)kHc * kDim;
  const size_t history = d->engram_history.size();

  /* Engram rows, [layer][token][column]. */
  std::vector<int64_t> engram_rows((uint64_t)DS41_N_ENGRAM_LAYER * n * DS41_ENGRAM_COLS);
  for (uint32_t t = 0; t < n; t++) {
    int64_t rows[DS41_N_ENGRAM_LAYER][DS41_ENGRAM_COLS];
    engram_hash(d, tokens[t], rows);
    for (uint32_t layer = 0; layer < DS41_N_ENGRAM_LAYER; layer++)
      memcpy(&engram_rows[((uint64_t)layer * n + t) * DS41_ENGRAM_COLS], rows[layer], sizeof(rows[layer]));
  }

  /* Every Engram layer's rows are read while the GPU runs the layers before it. */
  std::vector<float> engram_host[DS41_N_ENGRAM_LAYER];
  float* engram_out[DS41_N_ENGRAM_LAYER];
  std::future<bool> engram_reads[DS41_N_ENGRAM_LAYER];
  for (uint32_t il = 0; il < DS41_N_LAYER; il++) {
    const int slot = ds41_layer_engram_slot(il);
    if (slot < 0) continue;
    engram_out[slot] = d->engram_pinned[slot];
    if (n > kEngramPinnedTokens) {
      engram_host[slot].resize((uint64_t)n * DS41_ENGRAM_COLS * DS41_ENGRAM_HEAD_DIM);
      engram_out[slot] = engram_host[slot].data();
    }
    engram_reads[slot] = std::async(std::launch::async, engram_read, d, il,
                                    &engram_rows[(uint64_t)slot * n * DS41_ENGRAM_COLS], n, engram_out[slot]);
  }

  /* Embedding, replicated over the hc residual streams; layer 0 pre-mix is one-hot. */
  const ds41_tensor* embd = w->token_embd;
  const uint64_t row_bytes = embd->bytes / DS41_N_VOCAB;
  std::vector<float> one_hot((uint64_t)n * kHc, 0.0f);
  bool ok = true;
  for (uint32_t t = 0; t < n && ok; t++) {
    float* row = d->host_h.data() + t * hd;
    ok = dequant_row(embd->type, host(d, embd) + (uint64_t)tokens[t] * row_bytes, row, kDim);
    for (uint32_t c = 1; c < kHc; c++) memcpy(row + c * kDim, row, kDim * sizeof(float));
    one_hot[(uint64_t)t * kHc] = 1.0f;
  }
  ok = ok && hipMemcpy(d->h, d->host_h.data(), n * hd * sizeof(float), hipMemcpyHostToDevice) == hipSuccess &&
       hipMemcpy(d->pre[0], one_hot.data(), one_hot.size() * sizeof(float), hipMemcpyHostToDevice) == hipSuccess;

  const int prev = 0;
  for (uint32_t il = 0; il < DS41_N_LAYER && ok; il++) {
    const int slot = ds41_layer_engram_slot(il);
    if (slot >= 0) ok = engram_reads[slot].get() && engram(d, il, engram_out[slot], n);
    /* block() leaves this layer's FFN pre-mix in pre[prev]. */
    ok = ok && block(d, il, prev, prev ^ 1, n);
    if (ok && layer_fn) {
      ok = hipMemcpy(d->host_h.data(), d->h + (n - 1) * hd, hd * sizeof(float), hipMemcpyDeviceToHost) ==
               hipSuccess &&
           hipMemcpy(d->host_ids.data(), d->route_ids + (uint64_t)(n - 1) * DS41_N_EXPERT_USED,
                     d->host_ids.size() * sizeof(int32_t), hipMemcpyDeviceToHost) == hipSuccess;
      if (ok) layer_fn(ctx, il, d->host_h.data(), d->host_ids.data());
    }
  }

  /* Final mix and norm for the requested rows, then the vocabulary projection in passes. */
  const uint32_t first = n - logit_rows;
  if (ok && logit_rows) {
    ok = ds41_v41_hc_collapse_norm(d->h + first * hd, d->pre[prev] + (uint64_t)first * kHc, f32(d, w->output_norm),
                                   d->xn, kHc, kDim, DS41_RMS_EPS, logit_rows) == 0;
    for (uint32_t r = 0; r < logit_rows && ok; r += kLogitRows) {
      const uint32_t rows = std::min(kLogitRows, logit_rows - r);
      ok = linear(d, w->output, d->xn + (uint64_t)r * kDim, d->logits, rows) &&
           hipMemcpy(logits + (uint64_t)r * DS41_N_VOCAB, d->logits, (uint64_t)rows * DS41_N_VOCAB * sizeof(float),
                     hipMemcpyDeviceToHost) == hipSuccess;
    }
  }
  ok = ok && hipDeviceSynchronize() == hipSuccess;
  if (!ok) {
    /* The device state is partial; callers reset or reload before reuse. */
    d->engram_history.resize(history);
    return 1;
  }
  d->pos += n;
  return 0;
}

int ds41_decoder_step(ds41_decoder* d, int token, float* logits, ds41_decoder_layer_fn layer_fn, void* ctx) {
  return ds41_decoder_eval(d, &token, 1, logits, logits ? 1 : 0, layer_fn, ctx);
}
