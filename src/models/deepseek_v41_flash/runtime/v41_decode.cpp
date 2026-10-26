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
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include "ds41_v41.h"
#include "mmq/ds41_mmq.h"
#include "resident_api.h"
#include "model.h"
#include "model_data_internal.h"
#include "native_internal.h"
#include "tensor_types.h"
#include "v41_dspark.h"

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
/* Bytes per FP4 compressed row (ds41_fp4_rows): KV block 16, index-K block 32. */
constexpr uint32_t kCompCodeBytes = kHeadDim / 2, kCompScaleBytes = kHeadDim / 16;
constexpr uint32_t kIndexCodeBytes = DS41_N_INDEXER_HEAD_DIM / 2, kIndexScaleBytes = DS41_N_INDEXER_HEAD_DIM / 32;
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
  /* KV sources: [context / ratio] compressed KV and index-K rows as FP4 (ds41_fp4_rows). */
  uint8_t *comp = nullptr, *comp_scale = nullptr;
  uint8_t *index_k = nullptr, *index_scale = nullptr;
  float* pool_kv = nullptr;   // [ratio, head_dim] compressor frontier
  float* pool_score = nullptr;
};

/* Batch capacity: up to the engine's prefill batch (2048 tokens by default,
 * where the routed expert tiles average 32 rows per expert), fewer when the
 * activations (about 0.6 MiB per token) would leave less than kDeviceReserve
 * of device memory free. On gfx1151 the
 * reported free memory runs about 1 GiB below MemAvailable, so this keeps
 * about 2 GiB of RAM for the MMQ scratch and the rest of the system. */
constexpr uint32_t kMaxBatch = DS41_MAX_PREFILL_BATCH;
constexpr uint64_t kDeviceReserve = 1ull << 30;

/* Indexer score rows (context + 1 floats per token) run in groups of up to
 * this many tokens, bounding the score buffer to 256 MiB at any context. */
uint32_t score_rows(uint64_t score_stride) {
  const uint64_t rows = (256ull << 20) / (score_stride * sizeof(float));
  return (uint32_t)std::clamp<uint64_t>(rows / 16 * 16, 16, kMaxBatch);
}

constexpr uint32_t kLogitRows = 64;  // output projection rows per pass

/* Batch scratch, `capacity` rows per activation. No scratch value outlives
 * an evaluation and the engine serializes evaluations, so every decoder of
 * one engine and context shares one set. */
struct Activations {
  uint32_t capacity = 0;
  uint32_t score_rows = 0;  // tokens per indexer score group
  float *h = nullptr, *h2 = nullptr, *xn = nullptr;
  float *mixes = nullptr, *pre[2] = {}, *post = nullptr, *comb = nullptr, *ffn_post = nullptr, *ffn_comb = nullptr;
  float *qa = nullptr, *qr = nullptr, *q = nullptr, *kv = nullptr, *kvn = nullptr;
  float *ckv = nullptr, *cscore = nullptr, *craw = nullptr, *latent = nullptr;
  float *ik = nullptr, *ikn = nullptr, *iq = nullptr, *iw = nullptr;
  float *scores = nullptr, *block_scores = nullptr;  // score_rows rows each
  uint32_t* candidates = nullptr;  // [capacity, mask_words] candidate-block bits
  uint32_t *topk = nullptr, *block_select = nullptr;
  float *attn_partial = nullptr;
  float *o = nullptr, *o_group = nullptr, *oa_group = nullptr, *oa = nullptr, *attn_out = nullptr;
  float* rope = nullptr;  // [cap, kRot / 2] (cos, sin) for the batched attention
  float *router = nullptr, *route_w = nullptr;
  int32_t* route_ids = nullptr;
  uint8_t* expert_class = nullptr;  // routed tile-width class per expert
  int32_t* routed_plan = nullptr;   // ds41_v41_routed_plan rows, bounds and tiles
  int32_t* expert_plan = nullptr;   // ds41_v41_expert_plan groups of a small batch
  float *g6 = nullptr, *u6 = nullptr, *mid6 = nullptr, *down6 = nullptr;
  float *sg = nullptr, *su = nullptr, *sm = nullptr, *sd = nullptr;
  float *engram_in = nullptr, *engram_kv = nullptr, *logits = nullptr;
  /* Pinned staging for the Engram rows of batches up to kEngramPinnedTokens,
   * per Engram layer, so the upload does not wait for the queue to drain.
   * Each evaluation ends with a device synchronize, so the next one finds
   * the buffers idle. */
  float* engram_pinned[DS41_N_ENGRAM_LAYER] = {};
};

struct SharedActivations {
  ds41_engine* engine = nullptr;
  uint32_t context = 0;
  Activations views;
  std::vector<void*> device;
  ~SharedActivations() {
    for (void* p : device) (void)hipFree(p);
    for (float* p : views.engram_pinned)
      if (p) (void)hipHostFree(p);
  }
};

std::mutex g_shared_mutex;
std::vector<std::weak_ptr<SharedActivations>> g_shared;

}  // namespace

struct ds41_decoder : Activations {
  ds41_engine* engine = nullptr;
  uint32_t context = 0;
  uint32_t pos = 0;
  uint64_t score_stride = 0;
  uint64_t block_stride = 0;
  uint64_t mask_words = 0;
  std::vector<void*> allocations;  // decoder state
  std::shared_ptr<SharedActivations> shared;
  LayerState layer[DS41_N_LAYER];
  float* freq_plain = nullptr;
  float* freq_compress = nullptr;

  int comp_layer = -1;
  /* During a bounded-replay prompt evaluation, the decoder layers' first
   * visible window position; 0 otherwise. */
  uint32_t window_floor = 0;
  /* A speculative batch of spec_n rows from spec_pos0 holds every layer's
   * window rows and the KV source layers' compressor inputs (kv, score)
   * here until ds41_decoder_commit; spec_n is 0 otherwise. spec_eval marks
   * the evaluation of that batch. */
  uint32_t spec_pos0 = 0, spec_n = 0;
  bool spec_eval = false;
  float* spec_window = nullptr;  // [layer][DS41_DECODER_MAX_SPECULATIVE][head_dim]
  float* spec_pool = nullptr;    // [layer][kv, score][DS41_DECODER_MAX_SPECULATIVE][head_dim]
  /* DSpark target features: [DS41_DSPARK_FEATURE_ROWS][3 * dim] hc-means of
   * the target layers' inputs for positions feature_pos0 .. + feature_rows. */
  float* features = nullptr;
  uint32_t feature_pos0 = 0, feature_rows = 0;
  std::vector<int64_t> engram_history;
  std::vector<float> host_h;
  std::vector<int32_t> host_ids;
};

namespace {

template <typename T>
T* alloc(std::vector<void*>& owned, uint64_t count) {
  void* p = nullptr;
  if (hipMalloc(&p, count * sizeof(T)) != hipSuccess) return nullptr;
  (void)hipMemset(p, 0, count * sizeof(T));
  owned.push_back(p);
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

/* Batches of Q5_K rows run on the FP16 matrix cores (ds41_v41_dense_f16). */
bool dense_f16(uint32_t type, uint32_t rows, uint32_t k, uint32_t n) {
  return n > 8 && type == DS41_TENSOR_Q5_K && rows % 256 == 0 && k % 256 == 0;
}

/* out[t] = W x[t] for n tokens and a [rows, k] weight at `p` of `type`. */
bool linear_raw(uint32_t type, const void* p, uint32_t rows, uint32_t k, const float* x, float* out, uint32_t n) {
  if (!p) return false;
  switch (type) {
  case DS41_TENSOR_BF16:
    return ds41_v41_matmul_bf16((const uint16_t*)p, x, out, rows, k, n) == 0;
  case DS41_TENSOR_F32:
    return ds41_v41_matmul_f32((const float*)p, x, out, rows, k, n) == 0;
  case DS41_TENSOR_F16:
    return ds41_v41_matmul_f16((const uint16_t*)p, x, out, rows, k, n) == 0;
  default:
    /* Up to eight tokens: FP32 activations, no Q8_1 pass, each weight read once. */
    if (n <= DS41_V41_GEMV_MAX_TOKENS && gemv_type(type))
      return ds41_v41_gemv((int)type, p, x, out, rows, k, 0, gemv_slices(rows, k), n, k, rows) == 0;
    if (dense_f16(type, rows, k, n)) return ds41_v41_dense_f16((int)type, p, x, out, rows, n, k, 1) == 0;
    if (n > 8 && mmq_batched(type))
      return ds41_mmq_dense_typed((int)type, p, x, out, (int)rows, (int)n, (int)k, 0) == 0;
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

/* out1 = W1 x and out2 = W2 x; batches of one MMQ type quantize x once. */
bool linear_pair(const ds41_decoder* d, const ds41_tensor* w1, float* out1, const ds41_tensor* w2, float* out2,
                 const float* x, uint32_t n) {
  const uint32_t k = (uint32_t)w1->dim[0];
  if (n > 8 && w1->type == w2->type && w2->dim[0] == k && mmq_batched(w1->type) &&
      !dense_f16(w1->type, (uint32_t)(w1->elements / k), k, n)) {
    const void* p1 = dev(d, w1);
    const void* p2 = dev(d, w2);
    return p1 && p2 &&
           ds41_mmq_dense_pair_typed((int)w1->type, p1, out1, (int)(w1->elements / k), p2, out2,
                                     (int)(w2->elements / k), x, (int)n, (int)k, 0) == 0;
  }
  return linear(d, w1, x, out1, n) && linear(d, w2, x, out2, n);
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

/* Layer il's pending rows of a speculative buffer; `part` selects kv or score of spec_pool. */
float* spec_rows(const ds41_decoder* d, float* base, uint32_t il, uint32_t part) {
  const uint64_t parts = base == d->spec_pool ? 2 : 1;
  return base + ((uint64_t)il * parts + part) * DS41_DECODER_MAX_SPECULATIVE * kHeadDim;
}

/* Stores the compressed rows r0 .. r0 + rows - 1 of KV source layer il from
 * their unrotated latent in d->latent: the index keys (index sources) and
 * the rotated FP4 KV rows. */
bool store_compressed(ds41_decoder* d, uint32_t il, uint32_t r0, uint32_t rows, uint32_t ratio) {
  const ds41_layer_weights* l = &d->engine->weights->layer[il];
  LayerState& s = d->layer[il];
  const float* freq = layer_freq(d, il);
  const float eps = DS41_RMS_EPS;
  if (ds41_layer_is_index_source(il) &&
      (!linear(d, l->indexer_attn_k, d->latent, d->ik, rows) ||
       ds41_v41_rms_norm(d->ik, f32(d, l->indexer_k_norm), d->ikn, rows, DS41_N_INDEXER_HEAD_DIM, eps) ||
       ds41_v41_rope_tail(d->ikn, rows, DS41_N_INDEXER_HEAD_DIM, kRot, freq, r0 * ratio, 1, ratio, 0) ||
       ds41_v41_fp4_store(d->ikn, s.index_k + (uint64_t)r0 * kIndexCodeBytes,
                          s.index_scale + (uint64_t)r0 * kIndexScaleBytes, rows * DS41_N_INDEXER_HEAD_DIM, 32, 0)))
    return false;
  return ds41_v41_rope_tail(d->latent, rows, kHeadDim, kRot, freq, r0 * ratio, 1, ratio, 0) == 0 &&
         ds41_v41_fp4_store(d->latent, s.comp + (uint64_t)r0 * kCompCodeBytes,
                            s.comp_scale + (uint64_t)r0 * kCompScaleBytes, rows * kHeadDim, 16, 1) == 0;
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
  const bool pair = n <= DS41_V41_GEMV_MAX_TOKENS && wqa->type == wkv->type && wqa->dim[0] == wkv->dim[0] &&
                    gemv_type(wqa->type);
  if (pair) {
    const uint32_t k = (uint32_t)wqa->dim[0];
    const uint32_t m = (uint32_t)((wqa->elements + wkv->elements) / k);
    if (ds41_v41_gemv_pair((int)wqa->type, dev(d, wqa), d->qa, DS41_N_LORA_Q, dev(d, wkv), d->kv, kHeadDim, d->xn, k,
                           gemv_slices(m, k), n))
      return false;
  } else if (!linear_pair(d, wqa, d->qa, wkv, d->kv, d->xn, n)) {
    return false;
  }
  if (ds41_v41_q_kv_norm(d->qa, f32(d, l->attn_q_a_norm), d->qr, DS41_N_LORA_Q, d->kv, f32(d, l->attn_kv_a_norm),
                         d->kvn, kHeadDim, kRot, freq, pos0, n, eps) ||
      !linear(d, l->attn_q_b, d->qr, d->q, n))
    return false;

  ds41_fp4_rows comp{};
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
                 (d->spec_eval && (!copy_dev(spec_rows(d, d->spec_pool, il, 0), d->ckv, (uint64_t)n * kHeadDim) ||
                                !copy_dev(spec_rows(d, d->spec_pool, il, 1), d->cscore, (uint64_t)n * kHeadDim))) ||
                 ds41_v41_compress_pool(d->ckv, d->cscore, s.pool_kv, s.pool_score, pos0, n, ratio, d->craw,
                                        kHeadDim) ||
                 ds41_v41_rms_norm(d->craw, f32(d, l->attn_compressor_norm), d->latent, emitted, kHeadDim, eps)) {
        return false;
      }
      if (emitted && !store_compressed(d, il, r0, emitted, ratio)) return false;
      d->comp_layer = (int)il;
    }
    const LayerState& source = d->layer[d->comp_layer];
    if (ds41_layer_is_index_source(il)) {
      const float w_scale = 1.0f / sqrtf((float)DS41_N_INDEXER_HEAD_DIM) / sqrtf((float)DS41_N_INDEXER_HEAD);
      if (!linear(d, l->indexer_attn_q_b, d->qr, d->iq, n) ||
          ds41_v41_rope_tail(d->iq, n * DS41_N_INDEXER_HEAD, DS41_N_INDEXER_HEAD_DIM, kRot, freq, pos0,
                             DS41_N_INDEXER_HEAD, 1, 0) ||
          ds41_v41_fp4_fake_quant(d->iq, n * kIndexQDim, 32, 0) || !linear(d, l->indexer_proj, d->xn, d->iw, n))
        return false;
      for (uint32_t t0 = 0; t0 < n; t0 += d->score_rows) {
        const uint32_t m = std::min(d->score_rows, n - t0);
        const uint32_t p0 = pos0 + t0;
        uint32_t* mask = d->candidates + t0 * d->mask_words;
        if (ds41_v41_indexer_score(d->iq + (uint64_t)t0 * kIndexQDim, {source.index_k, source.index_scale},
                                   d->iw + (uint64_t)t0 * DS41_N_INDEXER_HEAD, w_scale, d->scores, d->score_stride,
                                   p0, m, ratio, DS41_N_INDEXER_HEAD, DS41_N_INDEXER_HEAD_DIM))
          return false;
        if (il == DS41_CANDIDATE_SOURCE_LAYER) {
          if (ds41_v41_candidate_mask(d->scores, d->score_stride, mask, d->mask_words, d->block_scores, d->block_stride,
                                      d->block_select, p0, m, ratio, DS41_CANDIDATE_BLOCK_SIZE,
                                      DS41_CANDIDATE_TOPK_BLOCKS))
            return false;
        } else if (il > DS41_CANDIDATE_SOURCE_LAYER &&
                   ds41_v41_apply_mask(d->scores, d->score_stride, mask, d->mask_words, p0, m, ratio,
                                       DS41_CANDIDATE_BLOCK_SIZE)) {
          return false;
        }
        if (ds41_v41_topk(d->scores, d->score_stride, d->topk + (uint64_t)t0 * DS41_N_INDEXER_TOP_K,
                          DS41_N_INDEXER_TOP_K, p0, m, ratio, 1))
          return false;
      }
    }
    comp = {source.comp, source.comp_scale};
  }

  const uint32_t floor = il >= DS41_FIRST_DECODER_LAYER ? d->window_floor : 0;
  if (ds41_v41_attention(d->q, s.window, d->kvn, DS41_N_SWA, floor, comp, d->topk, DS41_N_INDEXER_TOP_K, ratio,
                         f32(d, l->attn_sinks), d->o, d->attn_partial, pos0, n, DS41_N_HEAD, kHeadDim,
                         1.0f / sqrtf((float)kHeadDim), freq, kRot, d->rope) ||
      !(d->spec_eval ? copy_dev(spec_rows(d, d->spec_window, il, 0), d->kvn, (uint64_t)n * kHeadDim)
                  : store_window(s, d->kvn, pos0, n)))
    return false;

  /* Grouped LoRA output: group g maps head slice g of every token. */
  const ds41_tensor* wo_a = l->attn_output_a;
  const uint64_t group_bytes = wo_a->bytes / DS41_N_OUT_GROUP;
  const uint8_t* wo_a_dev = (const uint8_t*)dev(d, wo_a);
  if (!wo_a_dev) return false;
  /* Small batches take all groups in one matvec launch. */
  if (n <= DS41_V41_GEMV_MAX_TOKENS && gemv_type(wo_a->type)) {
    if (ds41_v41_gemv((int)wo_a->type, wo_a_dev, d->o, d->oa, DS41_N_OUT_GROUP * DS41_N_LORA_O, kGroupDim,
                      DS41_N_LORA_O, gemv_slices(DS41_N_OUT_GROUP * DS41_N_LORA_O, kGroupDim), n, kQDim,
                      DS41_N_OUT_GROUP * DS41_N_LORA_O))
      return false;
    return linear(d, l->attn_output_b, d->oa, d->attn_out, n);
  }
  if (n <= 8 && mmq_batched(wo_a->type)) {
    if (ds41_mmq_grouped_vec_typed((int)wo_a->type, wo_a_dev, d->o, d->oa, DS41_N_LORA_O, (int)n, kGroupDim,
                                   DS41_N_OUT_GROUP, 0))
      return false;
    return linear(d, l->attn_output_b, d->oa, d->attn_out, n);
  }
  /* Batches take all groups in one launch: the FP16 kernel's grid groups,
   * or MMQ channels with one input quantization. */
  if (dense_f16(wo_a->type, DS41_N_LORA_O, kGroupDim, n)) {
    if (ds41_v41_dense_f16((int)wo_a->type, wo_a_dev, d->o, d->oa, DS41_N_LORA_O, n, kGroupDim, DS41_N_OUT_GROUP))
      return false;
    return linear(d, l->attn_output_b, d->oa, d->attn_out, n);
  }
  if (n > 8 && mmq_batched(wo_a->type)) {
    if (ds41_mmq_dense_grouped_typed((int)wo_a->type, wo_a_dev, d->o, d->oa, DS41_N_LORA_O, (int)n, kGroupDim,
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
/* Batched routed projections on the FP16 matrix cores (ds41_v41_routed_f16). */
bool routed_f16_type(int type) {
  return type == (int)DS41_TENSOR_IQ1_S || type == (int)DS41_TENSOR_IQ1_M || type == (int)DS41_TENSOR_IQ2_XXS;
}

/* ds41_v41_routed_plan's rows, bounds and tiles for batches up to cap tokens. */
uint64_t routed_plan_ints(uint64_t cap) {
  const uint64_t slots = cap * DS41_N_EXPERT_USED;
  return slots + (DS41_N_EXPERT + 1) + (slots / 64 + DS41_N_EXPERT);
}

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
  *status = ds41_v41_gemv_experts(type, w, (uint64_t)m * k / 256 * block, ids, slots, x, per_slot ? k : 0, 1, y, m, k);
  return true;
}

/* Batches of 2 to this many tokens take the routed experts through the
 * grouped expert GEMV: each expert's weights are read once for its tokens,
 * with the single-token arithmetic and no host round trip for a plan. */
constexpr uint32_t kSmallMoeTokens = DS41_V41_EXPERT_PLAN_COLS;
static_assert(kSmallMoeTokens * DS41_N_EXPERT_USED <= DS41_V41_EXPERT_PLAN_SLOTS);

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
      ds41_mmq_set_routed_classes(nullptr, 0, nullptr, nullptr);
    }
  } routed_bounds;
  const bool grouped = n > 1 && n <= kSmallMoeTokens && expert_block_bytes(gate_type) &&
                       expert_block_bytes(up_type) && expert_block_bytes(down_type);
  const bool small = n == 1 || grouped;
  if (grouped && ds41_v41_expert_plan(d->route_ids, (uint32_t)rows, d->expert_plan)) return false;
  /* y = W x over the planned slots (with w_up: the routed SwiGLU into mid6). */
  const auto grouped_gemv = [&](int type, const void* w, const void* w_up, const float* x, uint32_t x_slot,
                                uint32_t slots_per_x, float* y, uint32_t m, uint32_t k) {
    return ds41_v41_gemv_experts_grouped(type, w, w_up, w_up ? d->route_w : nullptr, w_up ? kSwigluLimit : 0.0f,
                                         (uint64_t)m * k / 256 * expert_block_bytes(type), d->expert_plan,
                                         (uint32_t)rows, n, x, x_slot, slots_per_x, y, m, k) == 0;
  };
  const bool f16_gate = !small && gate_type == up_type && routed_f16_type(gate_type);
  const bool f16_down = !small && routed_f16_type(down_type);
  const int32_t* plan_rows = d->routed_plan;
  const int32_t* plan_bounds = plan_rows + rows;
  const int32_t* plan_tiles = plan_bounds + DS41_N_EXPERT + 1;
  uint32_t n_tiles = 0;
  /* FP16 copies of the input and of the SwiGLU rows, in the buffers the
   * MMQ path would use for the gate and up rows. */
  auto* x16 = reinterpret_cast<uint16_t*>(d->g6);
  auto* mid16 = reinterpret_cast<uint16_t*>(d->u6);
  if (!small) {
    std::vector<int32_t> ids((size_t)rows);
    std::vector<unsigned int> counts(DS41_N_EXPERT, 0u);
    if (hipMemcpy(ids.data(), d->route_ids, ids.size() * sizeof(int32_t), hipMemcpyDeviceToHost) != hipSuccess)
      return false;
    unsigned int most = 0;
    for (const int32_t e : ids)
      if (e >= 0 && e < (int32_t)DS41_N_EXPERT) most = std::max(most, ++counts[(size_t)e]);
    ds41_mmq_set_routed_max_expert_rows((int)most);
    ds41_mmq_set_routed_tile_cols(ds41_mmq_routed_tile_cols_for_counts(counts.data(), DS41_N_EXPERT));
    /* Each expert's tiles take the width that suits its bucket (16 to 64 columns). */
    uint8_t classes[DS41_N_EXPERT];
    int widths[3], class_rows[3];
    const int n_classes =
        ds41_mmq_routed_classes_for_counts(counts.data(), DS41_N_EXPERT, classes, widths, class_rows);
    if (n_classes > 0) {
      if (hipMemcpy(d->expert_class, classes, sizeof(classes), hipMemcpyHostToDevice) != hipSuccess) return false;
      ds41_mmq_set_routed_classes(d->expert_class, n_classes, widths, class_rows);
    }
    if (f16_gate || f16_down) {
      std::vector<int32_t> plan(routed_plan_ints(n));
      int32_t* bounds = plan.data() + rows;
      n_tiles = ds41_v41_routed_plan(ids.data(), (uint32_t)rows, DS41_N_EXPERT, plan.data(), bounds,
                                     bounds + DS41_N_EXPERT + 1);
      const size_t used_ints = (size_t)rows + DS41_N_EXPERT + 1 + n_tiles;
      if (n_tiles == 0 || hipMemcpy(d->routed_plan, plan.data(), used_ints * sizeof(int32_t),
                                    hipMemcpyHostToDevice) != hipSuccess)
        return false;
    }
  }
  /* Small batches take gate, up and SwiGLU in one pass when gate and up share
   * a type (IQ2_XXS too: the pair runs as fast as two mmvq launches). */
  const bool fused_pair = gate_type == up_type && (gate_type == (int)DS41_TENSOR_IQ1_S ||
                                                   gate_type == (int)DS41_TENSOR_IQ1_M ||
                                                   gate_type == (int)DS41_TENSOR_IQ2_XXS);
  if (grouped && gate_type == up_type) {
    if (!grouped_gemv(gate_type, gate, up, d->xn, kDim, used, d->mid6, DS41_N_FF_EXP, kDim)) return false;
  } else if (grouped) {
    if (!grouped_gemv(gate_type, gate, nullptr, d->xn, kDim, used, d->g6, DS41_N_FF_EXP, kDim) ||
        !grouped_gemv(up_type, up, nullptr, d->xn, kDim, used, d->u6, DS41_N_FF_EXP, kDim))
      return false;
  } else if (n == 1 && fused_pair) {
    if (ds41_v41_gemv_experts_swiglu(gate_type, gate, up,
                                     (uint64_t)DS41_N_FF_EXP * kDim / 256 * expert_block_bytes(gate_type),
                                     d->route_ids, used, d->xn, 0, 1, d->route_w, kSwigluLimit, d->mid6,
                                     DS41_N_FF_EXP, kDim))
      return false;
  } else if (small) {
    int status = 0;
    if (!expert_gemv(gate_type, gate, d->route_ids, rows, d->xn, false, d->g6, DS41_N_FF_EXP, kDim, &status))
      status = ds41_mmq_moe_vec_typed(gate_type, gate, d->xn, d->route_ids, d->g6, DS41_N_FF_EXP, kDim, (int)n,
                                      DS41_N_EXPERT, used, 0);
    if (status) return false;
    if (!expert_gemv(up_type, up, d->route_ids, rows, d->xn, false, d->u6, DS41_N_FF_EXP, kDim, &status))
      status = ds41_mmq_moe_vec_typed(up_type, up, d->xn, d->route_ids, d->u6, DS41_N_FF_EXP, kDim, (int)n,
                                      DS41_N_EXPERT, used, 0);
    if (status) return false;
  } else if (f16_gate) {
    const uint64_t expert_bytes = (uint64_t)DS41_N_FF_EXP * kDim / 256 * expert_block_bytes(gate_type);
    if (ds41_v41_to_half(d->xn, x16, (uint64_t)n * kDim) ||
        ds41_v41_routed_f16(gate_type, gate, up, expert_bytes, x16, (uint32_t)used, plan_rows, plan_bounds,
                            plan_tiles, n_tiles, d->route_w, kSwigluLimit, f16_down ? (void*)mid16 : (void*)d->mid6,
                            f16_down, DS41_N_FF_EXP, kDim))
      return false;
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
  const bool mid_done =
      (n == 1 && fused_pair) || (grouped && gate_type == up_type) || (!small && gate_type == up_type);
  if (!mid_done && ds41_v41_swiglu(d->g6, d->u6, d->route_w, d->mid6, rows, DS41_N_FF_EXP, kSwigluLimit))
    return false;
  /* An FP16 down pass after an MMQ gate/up pass takes its rows in FP16. */
  if (f16_down && !f16_gate && ds41_v41_to_half(d->mid6, mid16, (uint64_t)rows * DS41_N_FF_EXP)) return false;
  /* Each (token, slot) row multiplies its own expert's down projection. */
  if (grouped) {
    if (!grouped_gemv(down_type, down, nullptr, d->mid6, DS41_N_FF_EXP, 1, d->down6, kDim, DS41_N_FF_EXP))
      return false;
  } else if (small) {
    int status = 0;
    if (!expert_gemv(down_type, down, d->route_ids, rows, d->mid6, true, d->down6, kDim, DS41_N_FF_EXP, &status))
      status = ds41_mmq_moe_vec_typed(down_type, down, d->mid6, d->route_ids, d->down6, kDim, DS41_N_FF_EXP, rows,
                                      DS41_N_EXPERT, 1, 0);
    if (status) return false;
  } else if (f16_down) {
    const uint64_t expert_bytes = (uint64_t)DS41_N_FF_EXP * kDim / 256 * expert_block_bytes(down_type);
    if (ds41_v41_routed_f16(down_type, down, nullptr, expert_bytes, mid16, 1, plan_rows, plan_bounds, plan_tiles,
                            n_tiles, nullptr, 0.0f, d->down6, 0, kDim, DS41_N_FF_EXP))
      return false;
  } else if (ds41_mmq_moe_typed(down_type, down, d->mid6, d->route_ids, d->down6, kDim, DS41_N_FF_EXP, rows,
                                DS41_N_EXPERT, 1, 0)) {
    return false;
  }
  const ds41_tensor* sg = l->ffn_gate_shexp;
  const ds41_tensor* su = l->ffn_up_shexp;
  if (n <= DS41_V41_GEMV_MAX_TOKENS && sg->type == su->type && sg->elements == su->elements && gemv_type(sg->type)) {
    const uint32_t k = (uint32_t)sg->dim[0];
    const uint32_t m = (uint32_t)(sg->elements / k);
    if (ds41_v41_gemv_swiglu((int)sg->type, dev(d, sg), dev(d, su), d->xn, d->sm, m, k, kSwigluLimit,
                             gemv_slices(m, k), n))
      return false;
  } else if (!linear_pair(d, sg, d->sg, su, d->su, d->xn, n) ||
             ds41_v41_swiglu(d->sg, d->su, nullptr, d->sm, n, DS41_N_FF_EXP, kSwigluLimit)) {
    return false;
  }
  /* The routed rows (down6) and the shared output (sd) are summed by block()'s hc expand. */
  return linear(d, l->ffn_down_shexp, d->sm, d->sd, n);
}

/* Tokens from which block() fuses the batched state passes. */
constexpr uint32_t kStatePassTokens = 16;

bool block(ds41_decoder* d, uint32_t il, int prev, int next, uint32_t n) {
  const ds41_layer_weights* l = &d->engine->weights->layer[il];
  const float eps = DS41_RMS_EPS;
  /* Batches read each state once for its mixes and collapse (the FFN's
   * after the attention expand); the collapse uses the pre-mix that was
   * known before the mix. */
  if (n >= kStatePassTokens) {
    const auto sinkhorn = [&](const ds41_tensor* scale, const ds41_tensor* base, float* pre, float* post,
                              float* comb) {
      return ds41_v41_hc_sinkhorn(d->mixes, f32(d, scale), f32(d, base), pre, post, comb, kHc,
                                  DS41_N_HC_SINKHORN_ITER, DS41_HC_EPS, n) == 0;
    };
    return ds41_v41_hc_state_pass(nullptr, nullptr, nullptr, d->h, nullptr, d->pre[prev], f32(d, l->attn_norm),
                                  d->xn, f32(d, l->hc_attn_fn), d->mixes, kHc, kDim, eps, n) == 0 &&
           sinkhorn(l->hc_attn_scale, l->hc_attn_base, d->pre[next], d->post, d->comb) && attention(d, il, n) &&
           ds41_v41_hc_state_pass(d->attn_out, d->post, d->comb, d->h, d->h2, d->pre[next], f32(d, l->ffn_norm),
                                  d->xn, f32(d, l->hc_ffn_fn), d->mixes, kHc, kDim, eps, n) == 0 &&
           sinkhorn(l->hc_ffn_scale, l->hc_ffn_base, d->pre[prev], d->ffn_post, d->ffn_comb) && moe(d, il, n) &&
           ds41_v41_hc_expand_sum(d->down6, d->sd, DS41_N_EXPERT_USED, d->h2, d->ffn_post, d->ffn_comb, d->h, kHc,
                                  kDim, n) == 0;
  }
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

/* Allocates the batch scratch, sizing the batch from the memory left after
 * the decoder state. */
std::shared_ptr<SharedActivations> make_activations(ds41_engine* engine, uint32_t context, uint64_t score_stride,
                                                    uint64_t block_stride) {
  auto sh = std::make_shared<SharedActivations>();
  sh->engine = engine;
  sh->context = context;
  Activations& a = sh->views;
  const uint64_t hd = (uint64_t)kHc * kDim;
  const uint64_t routed = (uint64_t)DS41_N_EXPERT_USED * DS41_N_FF_EXP;
  /* Buffers of one phase are dead outside it, so the three phases overlay
   * one region: attention (dead after its hc expand), the FFN (dead after
   * the closing hc expand) and the Engram add. */
  enum Phase { kOwn, kAttention, kFfn, kEngram, kPhases };
  struct Buf {
    float** p;
    uint64_t n;
    Phase phase = kOwn;
  };
  const auto float_buffers = [&](uint64_t cap, uint64_t rows) -> std::vector<Buf> {
    return {
        {&a.h, cap * hd},
        {&a.h2, cap * hd},
        {&a.xn, cap * kDim},
        {&a.mixes, cap * kMixes},
        {&a.pre[0], cap * kHc},
        {&a.pre[1], cap * kHc},
        {&a.post, cap * kHc},
        {&a.comb, cap * kHc * kHc},
        {&a.ffn_post, cap * kHc},
        {&a.ffn_comb, cap * kHc * kHc},
        {&a.qa, cap * DS41_N_LORA_Q, kAttention},
        {&a.qr, cap * DS41_N_LORA_Q, kAttention},
        {&a.q, cap * kQDim, kAttention},
        {&a.kv, cap * kHeadDim, kAttention},
        {&a.kvn, cap * kHeadDim, kAttention},
        {&a.ckv, cap * kHeadDim, kAttention},
        {&a.cscore, cap * kHeadDim, kAttention},
        {&a.craw, cap * kHeadDim, kAttention},
        {&a.latent, cap * kHeadDim, kAttention},
        {&a.ik, cap * DS41_N_INDEXER_HEAD_DIM, kAttention},
        {&a.ikn, cap * DS41_N_INDEXER_HEAD_DIM, kAttention},
        {&a.iq, cap * kIndexQDim, kAttention},
        {&a.iw, cap * DS41_N_INDEXER_HEAD, kAttention},
        {&a.scores, rows * score_stride},
        {&a.block_scores, rows * block_stride},
        {&a.o, cap * kQDim, kAttention},
        {&a.o_group, cap * kGroupDim, kAttention},
        {&a.oa_group, cap * DS41_N_LORA_O, kAttention},
        {&a.oa, cap * DS41_N_OUT_GROUP * DS41_N_LORA_O, kAttention},
        {&a.attn_out, cap * kDim, kAttention},
        {&a.rope, cap * kRot, kAttention},
        {&a.router, cap * DS41_N_EXPERT, kFfn},
        {&a.route_w, cap * DS41_N_EXPERT_USED, kFfn},
        {&a.g6, cap * routed, kFfn},
        {&a.u6, cap * routed, kFfn},
        {&a.mid6, cap * routed, kFfn},
        {&a.down6, cap * DS41_N_EXPERT_USED * kDim, kFfn},
        {&a.sg, cap * DS41_N_FF_EXP, kFfn},
        {&a.su, cap * DS41_N_FF_EXP, kFfn},
        {&a.sm, cap * DS41_N_FF_EXP, kFfn},
        {&a.sd, cap * kDim, kFfn},
        {&a.engram_in, cap * kEngramIn, kEngram},
        {&a.engram_kv, cap * kEngramOut, kEngram},
        {&a.logits, (uint64_t)kLogitRows * DS41_N_VOCAB},
        {&a.attn_partial, (uint64_t)(DS41_V41_ATTN_SPLIT_TOKENS - 1) * DS41_N_HEAD * DS41_V41_ATTN_PARTIAL_PER_HEAD},
    };
  };
  const auto padded = [](uint64_t n) { return (n + 63) / 64 * 64; };  // 256-byte offsets in the region
  /* Floats of the separate buffers and of the shared region. */
  const auto counts = [&](uint64_t cap, uint64_t rows, uint64_t* region) {
    uint64_t own = 0, phase[kPhases] = {};
    for (const Buf& b : float_buffers(cap, rows))
      if (b.phase == kOwn)
        own += b.n;
      else
        phase[b.phase] += padded(b.n);
    *region = std::max({phase[kAttention], phase[kFfn], phase[kEngram]});
    return own;
  };
  const auto float_count = [&](uint64_t cap, uint64_t rows) {
    uint64_t region = 0;
    const uint64_t own = counts(cap, rows, &region);
    return own + region;
  };
  /* Size the batch from the memory left after the KV state: per token, the
   * float activations, the candidate mask, top-k and route ids, and the
   * host embedding rows. */
  const uint32_t rows = score_rows(score_stride);
  const uint64_t per_token = (float_count(1, rows) - float_count(0, rows)) * sizeof(float) +
                             (block_stride + 31) / 32 * sizeof(uint32_t) +
                             DS41_N_INDEXER_TOP_K * sizeof(uint32_t) + DS41_N_EXPERT_USED * sizeof(int32_t) +
                             (routed_plan_ints(1) - routed_plan_ints(0)) * sizeof(int32_t) + kDim * sizeof(float);
  const uint64_t fixed = float_count(0, rows) * sizeof(float) + rows * DS41_CANDIDATE_TOPK_BLOCKS * sizeof(uint32_t);
  size_t free_bytes = 0, total_bytes = 0;
  bool ok = hipMemGetInfo(&free_bytes, &total_bytes) == hipSuccess;
  const uint64_t budget = free_bytes > kDeviceReserve + fixed ? free_bytes - kDeviceReserve - fixed : 0;
  a.capacity = (uint32_t)std::clamp<uint64_t>(budget / per_token / 16 * 16, 16, engine->prefill_batch / 16 * 16);
  a.score_rows = std::min(rows, a.capacity);
  const uint64_t cap = a.capacity;
  uint64_t region_floats = 0;
  (void)counts(cap, a.score_rows, &region_floats);
  float* region = ok ? alloc<float>(sh->device, region_floats) : nullptr;
  ok = ok && region;
  uint64_t offset[kPhases] = {};
  for (const Buf& b : float_buffers(cap, a.score_rows)) {
    if (!ok) break;
    if (b.phase == kOwn) {
      *b.p = alloc<float>(sh->device, b.n);
      ok = *b.p != nullptr;
    } else {
      *b.p = region + offset[b.phase];
      offset[b.phase] += padded(b.n);
    }
  }
  if (ok) {
    a.candidates = alloc<uint32_t>(sh->device, cap * ((block_stride + 31) / 32));
    a.topk = alloc<uint32_t>(sh->device, cap * DS41_N_INDEXER_TOP_K);
    a.block_select = alloc<uint32_t>(sh->device, (uint64_t)a.score_rows * DS41_CANDIDATE_TOPK_BLOCKS);
    a.route_ids = alloc<int32_t>(sh->device, cap * DS41_N_EXPERT_USED);
    a.expert_class = alloc<uint8_t>(sh->device, DS41_N_EXPERT);
    a.routed_plan = alloc<int32_t>(sh->device, routed_plan_ints(cap));
    a.expert_plan = alloc<int32_t>(sh->device, DS41_V41_EXPERT_PLAN_INTS(DS41_V41_EXPERT_PLAN_SLOTS));
    ok = a.candidates && a.topk && a.block_select && a.route_ids && a.expert_class && a.routed_plan && a.expert_plan;
  }
  for (uint32_t slot = 0; slot < DS41_N_ENGRAM_LAYER && ok; slot++)
    ok = hipHostMalloc(&a.engram_pinned[slot], (uint64_t)kEngramPinnedTokens * kEngramIn * sizeof(float)) ==
         hipSuccess;
  return ok ? sh : nullptr;
}

/* The scratch of a live decoder with this engine and context, or a new one. */
std::shared_ptr<SharedActivations> shared_activations(ds41_engine* engine, uint32_t context, uint64_t score_stride,
                                                      uint64_t block_stride) {
  const std::lock_guard<std::mutex> lock(g_shared_mutex);
  std::erase_if(g_shared, [](const std::weak_ptr<SharedActivations>& w) { return w.expired(); });
  for (const auto& w : g_shared) {
    auto sh = w.lock();
    if (sh && sh->engine == engine && sh->context == context) return sh;
  }
  auto sh = make_activations(engine, context, score_stride, block_stride);
  if (sh) g_shared.push_back(sh);
  return sh;
}

}  // namespace

ds41_decoder* ds41_decoder_create(ds41_engine* engine, uint32_t context) {
  if (!engine || !engine->rocm_ready || context == 0) return nullptr;
  auto* d = new ds41_decoder();
  d->engine = engine;
  d->context = context;
  d->score_stride = (uint64_t)context + 1;
  d->block_stride = (uint64_t)context / DS41_CANDIDATE_BLOCK_SIZE + 2;
  d->mask_words = (d->block_stride + 31) / 32;
  bool ok = true;
  for (uint32_t il = 0; il < DS41_N_LAYER && ok; il++) {
    LayerState& s = d->layer[il];
    s.window = alloc<float>(d->allocations, (uint64_t)DS41_N_SWA * kHeadDim);
    ok = s.window != nullptr;
    const uint32_t ratio = ds41_layer_compress_ratio(il);
    if (ok && ds41_layer_is_kv_source(il)) {
      const uint64_t rows = context / ratio + 1;
      s.comp = alloc<uint8_t>(d->allocations, rows * kCompCodeBytes);
      s.comp_scale = alloc<uint8_t>(d->allocations, rows * kCompScaleBytes);
      s.index_k = alloc<uint8_t>(d->allocations, rows * kIndexCodeBytes);
      s.index_scale = alloc<uint8_t>(d->allocations, rows * kIndexScaleBytes);
      s.pool_kv = alloc<float>(d->allocations, (uint64_t)ratio * kHeadDim);
      s.pool_score = alloc<float>(d->allocations, (uint64_t)ratio * kHeadDim);
      ok = s.comp && s.comp_scale && s.index_k && s.index_scale && s.pool_kv && s.pool_score;
    }
  }
  const uint64_t spec_floats = (uint64_t)DS41_N_LAYER * DS41_DECODER_MAX_SPECULATIVE * kHeadDim;
  d->spec_window = ok ? alloc<float>(d->allocations, spec_floats) : nullptr;
  d->spec_pool = ok ? alloc<float>(d->allocations, 2 * spec_floats) : nullptr;
  ok = ok && d->spec_window && d->spec_pool;
  const std::vector<float> plain = rope_inv_freq(DS41_ROPE_FREQ_BASE, 0);
  const std::vector<float> compress = rope_inv_freq(DS41_COMPRESS_ROPE_FREQ_BASE, DS41_ROPE_ORIG_CTX);
  d->freq_plain = alloc<float>(d->allocations, plain.size());
  d->freq_compress = alloc<float>(d->allocations, compress.size());
  ok = ok && d->freq_plain && d->freq_compress &&
       hipMemcpy(d->freq_plain, plain.data(), plain.size() * 4, hipMemcpyHostToDevice) == hipSuccess &&
       hipMemcpy(d->freq_compress, compress.data(), compress.size() * 4, hipMemcpyHostToDevice) == hipSuccess;

  if (ok) {
    d->shared = shared_activations(engine, context, d->score_stride, d->block_stride);
    ok = d->shared != nullptr;
  }
  if (!ok) {
    fprintf(stderr, "ds41: decoder allocation failed\n");
    ds41_decoder_destroy(d);
    return nullptr;
  }
  static_cast<Activations&>(*d) = d->shared->views;
  d->host_h.resize(std::max<uint64_t>((uint64_t)d->capacity * kDim, (uint64_t)kHc * kDim));
  d->host_ids.resize(DS41_N_EXPERT_USED);
  return d;
}

void ds41_decoder_destroy(ds41_decoder* d) {
  if (!d) return;
  for (void* p : d->allocations) (void)hipFree(p);
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
    if (!fn(s.comp, rows * kCompCodeBytes) || !fn(s.comp_scale, rows * kCompScaleBytes) ||
        !fn(s.index_k, rows * kIndexCodeBytes) || !fn(s.index_scale, rows * kIndexScaleBytes) ||
        !fn(s.pool_kv, (uint64_t)ratio * kHeadDim * sizeof(float)) ||
        !fn(s.pool_score, (uint64_t)ratio * kHeadDim * sizeof(float)))
      return false;
  }
  return true;
}

constexpr uint64_t kStateMagic = 0x3376615453313453ull;  // "S4STav3": FP4 compressed rows

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
  if (!d || !out || d->spec_n) return 1;
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
  d->spec_n = 0;
  d->feature_rows = 0;
  d->comp_layer = (int)(int64_t)header[2];
  d->engram_history = std::move(history);
  if (consumed) *consumed = at;
  return 0;
}

uint32_t ds41_layer_state_ratio(uint32_t layer) {
  return layer < DS41_N_LAYER && ds41_layer_is_kv_source(layer) ? ds41_layer_compress_ratio(layer) : 0;
}

void* ds41_decoder_state_part(ds41_decoder* d, uint32_t layer, int part, uint64_t* bytes) {
  if (!d || layer >= DS41_N_LAYER || !bytes) return nullptr;
  LayerState& s = d->layer[layer];
  const uint64_t frontier = (uint64_t)ds41_layer_state_ratio(layer) * kHeadDim * sizeof(float);
  switch (part) {
  case DS41_STATE_WINDOW:
    *bytes = (uint64_t)DS41_N_SWA * kHeadDim * sizeof(float);
    return s.window;
  case DS41_STATE_COMP:
    *bytes = kCompCodeBytes;
    return s.comp;
  case DS41_STATE_COMP_SCALE:
    *bytes = kCompScaleBytes;
    return s.comp_scale;
  case DS41_STATE_INDEX_K:
    *bytes = kIndexCodeBytes;
    return s.index_k;
  case DS41_STATE_INDEX_SCALE:
    *bytes = kIndexScaleBytes;
    return s.index_scale;
  case DS41_STATE_POOL_KV:
    *bytes = frontier;
    return s.pool_kv;
  case DS41_STATE_POOL_SCORE:
    *bytes = frontier;
    return s.pool_score;
  default:
    return nullptr;
  }
}

int ds41_decoder_adopt(ds41_decoder* d, const int* tokens, uint32_t n) {
  if (!d || n > d->context || (n && !tokens)) return 1;
  const int32_t* token_map = (const int32_t*)host(d, d->engine->weights->engram_token_map);
  d->engram_history.resize(n);
  for (uint32_t i = 0; i < n; i++) {
    if (tokens[i] < 0 || tokens[i] >= DS41_N_VOCAB) return 1;
    d->engram_history[i] = token_map[tokens[i]];
  }
  d->spec_n = 0;
  /* Every evaluation leaves the last KV source layer here. */
  d->comp_layer = -1;
  for (uint32_t il = 0; n && il < DS41_N_LAYER; il++)
    if (ds41_layer_is_kv_source(il)) d->comp_layer = (int)il;
  d->pos = n;
  d->feature_rows = 0;
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
  d->spec_n = 0;
  d->feature_rows = 0;
  d->comp_layer = -1;
  d->engram_history.clear();
  return 0;
}

namespace {

static_assert(DS41_FIRST_DECODER_LAYER == DS41_CANDIDATE_SOURCE_LAYER &&
                  DS41_KV_SOURCE_LAYERS[DS41_N_KV_SOURCE - 1] == DS41_FIRST_DECODER_LAYER,
              "bounded replay compresses the first decoder layer's rows");
static_assert(DS41_ENGRAM_LAYERS[DS41_N_ENGRAM_LAYER - 1] < DS41_FIRST_DECODER_LAYER &&
                  DS41_DSPARK_TARGET_LAYER[0] >= DS41_FIRST_DECODER_LAYER,
              "Engram runs every row, DSpark features the replayed ones");

/* The first `skip` rows reach the first decoder layer but no decoder layer
 * runs them: that layer only compresses them (global KV and index keys).
 * The remaining rows move to the front of the batch. */
bool replay_boundary(ds41_decoder* d, uint32_t il, int prev, uint32_t skip, uint32_t n) {
  const ds41_layer_weights* l = &d->engine->weights->layer[il];
  const uint64_t hd = (uint64_t)kHc * kDim;
  const uint32_t kept = n - skip;
  /* The attention input as block() derives it for this many rows. */
  const bool collapsed =
      skip >= kStatePassTokens
          ? ds41_v41_hc_state_pass(nullptr, nullptr, nullptr, d->h, nullptr, d->pre[prev], f32(d, l->attn_norm),
                                   d->xn, f32(d, l->hc_attn_fn), d->mixes, kHc, kDim, DS41_RMS_EPS, skip) == 0
          : ds41_v41_hc_collapse_norm(d->h, d->pre[prev], f32(d, l->attn_norm), d->xn, kHc, kDim, DS41_RMS_EPS,
                                      skip) == 0;
  if (!collapsed || !linear(d, l->attn_compressor_kv, d->xn, d->ckv, skip) ||
      ds41_v41_rms_norm(d->ckv, f32(d, l->attn_compressor_norm), d->latent, skip, kHeadDim, DS41_RMS_EPS) ||
      !store_compressed(d, il, d->pos, skip, 1))
    return false;
  /* h2 and the other pre-mix are free between blocks. */
  if (!copy_dev(d->h2, d->h + skip * hd, kept * hd) || !copy_dev(d->h, d->h2, kept * hd) ||
      !copy_dev(d->pre[prev ^ 1], d->pre[prev] + (uint64_t)skip * kHc, (uint64_t)kept * kHc) ||
      !copy_dev(d->pre[prev], d->pre[prev ^ 1], (uint64_t)kept * kHc))
    return false;
  d->pos += skip;
  return true;
}

/* Evaluates n tokens; the decoder layers run the positions from keep_from
 * on (bounded replay; 0 runs every row), whose windows start there. */
int eval_rows(ds41_decoder* d, const int* tokens, uint32_t n, uint32_t keep_from, float* logits,
              uint32_t logit_rows, ds41_decoder_layer_fn layer_fn, void* ctx) {
  if (!d || !tokens || n == 0 || n > d->capacity || logit_rows > n || (logit_rows && !logits) ||
      d->pos + n > d->context || d->spec_n)
    return 1;
  const uint32_t pos0 = d->pos;
  const uint32_t skip = std::clamp(keep_from, pos0, pos0 + n) - pos0;
  const uint32_t kept = n - skip;
  if (logit_rows > kept) return 1;
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
    ok = dequant_row(embd->type, host(d, embd) + (uint64_t)tokens[t] * row_bytes, d->host_h.data() + t * kDim, kDim);
    one_hot[(uint64_t)t * kHc] = 1.0f;
  }
  /* One upload into xn (free until the first block), then a device copy per stream. */
  ok = ok && hipMemcpy(d->xn, d->host_h.data(), (uint64_t)n * kDim * sizeof(float), hipMemcpyHostToDevice) ==
                 hipSuccess;
  for (uint32_t c = 0; c < kHc && ok; c++)
    ok = hipMemcpy2DAsync(d->h + c * kDim, hd * sizeof(float), d->xn, kDim * sizeof(float), kDim * sizeof(float), n,
                          hipMemcpyDeviceToDevice, 0) == hipSuccess;
  ok = ok &&
       hipMemcpy(d->pre[0], one_hot.data(), one_hot.size() * sizeof(float), hipMemcpyHostToDevice) == hipSuccess;

  const int prev = 0;
  const uint32_t feature_rows = std::min(kept, DS41_DSPARK_FEATURE_ROWS);
  d->feature_rows = 0;
  d->window_floor = keep_from;
  uint32_t rows = n;  // rows of the current layer
  for (uint32_t il = 0; il < DS41_N_LAYER && ok; il++) {
    if (il == DS41_FIRST_DECODER_LAYER && skip) {
      ok = replay_boundary(d, il, prev, skip, n);
      rows = kept;
      if (!ok || !kept) break;
    }
    const int slot = ds41_layer_engram_slot(il);
    if (slot >= 0) ok = engram_reads[slot].get() && engram(d, il, engram_out[slot], n);
    if (d->features) {
      for (uint32_t f = 0; f < DS41_DSPARK_TARGET_LAYERS && ok; f++)
        if (il == DS41_DSPARK_TARGET_LAYER[f])
          ok = ds41_v41_hc_mean(d->h + (uint64_t)(rows - feature_rows) * hd, d->features + (uint64_t)f * kDim,
                                (uint64_t)DS41_DSPARK_TARGET_LAYERS * kDim, kHc, kDim, feature_rows) == 0;
    }
    /* block() leaves this layer's FFN pre-mix in pre[prev]. */
    ok = ok && block(d, il, prev, prev ^ 1, rows);
    if (ok && layer_fn) {
      ok = hipMemcpy(d->host_h.data(), d->h + (rows - 1) * hd, hd * sizeof(float), hipMemcpyDeviceToHost) ==
               hipSuccess &&
           hipMemcpy(d->host_ids.data(), d->route_ids + (uint64_t)(rows - 1) * DS41_N_EXPERT_USED,
                     d->host_ids.size() * sizeof(int32_t), hipMemcpyDeviceToHost) == hipSuccess;
      if (ok) layer_fn(ctx, il, d->host_h.data(), d->host_ids.data());
    }
  }
  d->window_floor = 0;

  /* Final mix and norm for the requested rows, then the vocabulary projection in passes. */
  const uint32_t first = rows - logit_rows;
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
    d->pos = pos0;
    return 1;
  }
  if (d->features) {
    d->feature_pos0 = pos0 + n - feature_rows;
    d->feature_rows = feature_rows;
  }
  d->pos = pos0 + n;
  return 0;
}

}  // namespace

int ds41_decoder_eval(ds41_decoder* d, const int* tokens, uint32_t n, float* logits, uint32_t logit_rows,
                      ds41_decoder_layer_fn layer_fn, void* ctx) {
  return eval_rows(d, tokens, n, 0, logits, logit_rows, layer_fn, ctx);
}

int ds41_decoder_eval_prompt(ds41_decoder* d, const int* tokens, uint32_t n, uint32_t prompt_end, float* logits,
                             uint32_t logit_rows) {
  if (!d || prompt_end < d->pos + n) return 1;
  const uint32_t keep_from = prompt_end > DS41_DECODER_REPLAY_ROWS ? prompt_end - DS41_DECODER_REPLAY_ROWS : 0;
  return eval_rows(d, tokens, n, keep_from, logits, logit_rows, nullptr, nullptr);
}

int ds41_decoder_eval_speculative(ds41_decoder* d, const int* tokens, uint32_t n, float* logits) {
  if (!d || d->spec_n || n == 0 || n > DS41_DECODER_MAX_SPECULATIVE) return 1;
  d->spec_pos0 = d->pos;
  d->spec_eval = true;
  const int rc = ds41_decoder_eval(d, tokens, n, logits, n, nullptr, nullptr);
  d->spec_eval = false;
  if (rc == 0) d->spec_n = n;
  return rc;
}

int ds41_decoder_commit(ds41_decoder* d, uint32_t keep) {
  if (!d || !d->spec_n || keep == 0 || keep > d->spec_n) return 1;
  const uint32_t pos0 = d->spec_pos0;
  const uint32_t end = pos0 + keep;
  d->spec_n = 0;
  bool ok = true;
  for (uint32_t il = 0; il < DS41_N_LAYER && ok; il++) {
    LayerState& s = d->layer[il];
    ok = store_window(s, spec_rows(d, d->spec_window, il, 0), pos0, keep);
    const uint32_t ratio = ds41_layer_compress_ratio(il);
    if (!ok || !s.pool_kv || ratio < 2) continue;
    /* The frontier holds the positions of the kept tokens' unfinished group,
     * slot pos % ratio. Ratio 2 keeps at most the last kept token, which is
     * in the batch; older pending positions would need the earlier frontier. */
    for (uint32_t pos = end - end % ratio; pos < end && ok; pos++) {
      if (pos < pos0) return 1;
      const uint64_t row = (uint64_t)(pos - pos0) * kHeadDim;
      const uint64_t slot = (uint64_t)(pos % ratio) * kHeadDim;
      ok = copy_dev(s.pool_kv + slot, spec_rows(d, d->spec_pool, il, 0) + row, kHeadDim) &&
           copy_dev(s.pool_score + slot, spec_rows(d, d->spec_pool, il, 1) + row, kHeadDim);
    }
  }
  ok = ok && hipDeviceSynchronize() == hipSuccess;
  if (!ok) return 1;
  d->pos = end;
  d->engram_history.resize(end);
  return 0;
}

int ds41_decoder_step(ds41_decoder* d, int token, float* logits, ds41_decoder_layer_fn layer_fn, void* ctx) {
  return ds41_decoder_eval(d, &token, 1, logits, logits ? 1 : 0, layer_fn, ctx);
}

bool ds41_decoder_enable_features(ds41_decoder* d) {
  if (!d) return false;
  if (!d->features)
    d->features =
        alloc<float>(d->allocations, (uint64_t)DS41_DSPARK_FEATURE_ROWS * DS41_DSPARK_TARGET_LAYERS * kDim);
  d->feature_rows = 0;
  return d->features != nullptr;
}

const float* ds41_decoder_features(const ds41_decoder* d, uint32_t* pos0, uint32_t* rows) {
  *pos0 = d->feature_pos0;
  *rows = d->feature_rows;
  return d->features;
}

bool ds41_v41_linear(uint32_t type, const void* w, uint32_t rows, uint32_t k, const float* x, float* out, uint32_t n) {
  return linear_raw(type, w, rows, k, x, out, n);
}

bool ds41_decoder_output_head(const ds41_decoder* d, const float* x, float* logits, uint32_t n) {
  return linear(d, d->engine->weights->output, x, logits, n);
}

bool ds41_decoder_embed(const ds41_decoder* d, const int* tokens, uint32_t n, float* host_rows) {
  const ds41_tensor* embd = d->engine->weights->token_embd;
  const uint64_t row_bytes = embd->bytes / DS41_N_VOCAB;
  for (uint32_t t = 0; t < n; t++)
    if (tokens[t] < 0 || tokens[t] >= (int)DS41_N_VOCAB ||
        !dequant_row(embd->type, host(d, embd) + (uint64_t)tokens[t] * row_bytes, host_rows + (uint64_t)t * kDim,
                     kDim))
      return false;
  return true;
}

const float* ds41_decoder_freq_plain(const ds41_decoder* d) { return d->freq_plain; }
