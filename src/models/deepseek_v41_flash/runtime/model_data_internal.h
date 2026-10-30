#ifndef GUFO_MODELS_DEEPSEEK_V41_FLASH_RUNTIME_MODEL_DATA_INTERNAL_H_
#define GUFO_MODELS_DEEPSEEK_V41_FLASH_RUNTIME_MODEL_DATA_INTERNAL_H_

#include <cstddef>
#include <cstdint>

#include "model.h"
#include "tensor_types.h"

inline constexpr float DS41_NEG_INF = -1.0e30F;
inline constexpr float DS41_POS_INF = 1.0e30F;

/* DeepSeek V4.1 Flash backbone shape. Every ISTA GSQ-RCO build shares it; the
 * loader validates each value against the GGUF metadata. */
inline constexpr uint32_t DS41_N_LAYER = 40;
inline constexpr uint32_t DS41_N_EMBD = 5120;
inline constexpr uint32_t DS41_N_VOCAB = 129280;
inline constexpr uint32_t DS41_N_HEAD = 64;
inline constexpr uint32_t DS41_N_HEAD_KV = 1;
inline constexpr uint32_t DS41_N_HEAD_DIM = 512;
inline constexpr uint32_t DS41_N_VALUE_DIM = 512;
inline constexpr uint32_t DS41_N_ROT = 64;
inline constexpr uint32_t DS41_N_OUT_GROUP = 8;
inline constexpr uint32_t DS41_N_LORA_Q = 1280;
inline constexpr uint32_t DS41_N_LORA_O = 1024;
inline constexpr uint32_t DS41_N_EXPERT = 384;
inline constexpr uint32_t DS41_N_EXPERT_USED = 6;
inline constexpr uint32_t DS41_N_EXPERT_SHARED = 1;
inline constexpr uint32_t DS41_N_FF_EXP = 2304;
inline constexpr uint32_t DS41_N_SWA = 128;
inline constexpr uint32_t DS41_N_INDEXER_HEAD = 32;
inline constexpr uint32_t DS41_N_INDEXER_HEAD_DIM = 128;
inline constexpr uint32_t DS41_N_INDEXER_TOP_K = 512;
inline constexpr uint32_t DS41_N_HC = 4;
inline constexpr uint32_t DS41_N_HC_SINKHORN_ITER = 20;
inline constexpr uint32_t DS41_MAX_DIMS = 8;
inline constexpr uint32_t DS41_MAX_LAYER = DS41_N_LAYER;
inline constexpr uint32_t DS41_MAX_SHARDS = 16;

/* CSA2 cross-layer sharing: compressed KV and index keys come from the latest
 * KV source; top-k selections from the latest index source. */
inline constexpr uint32_t DS41_N_KV_SOURCE = 4;
inline constexpr uint32_t DS41_KV_SOURCE_LAYERS[DS41_N_KV_SOURCE] = {2, 8, 14,
                                                                     20};
inline constexpr uint32_t DS41_N_INDEX_SOURCE = 8;
inline constexpr uint32_t DS41_INDEX_SOURCE_LAYERS[DS41_N_INDEX_SOURCE] = {
    2, 8, 14, 20, 24, 28, 32, 36};
inline constexpr uint32_t DS41_CANDIDATE_SOURCE_LAYER = 20;
/* YoCo: layers from here on (the decoder half) take their compressed KV from
 * this layer, the first with compression ratio 1. */
inline constexpr uint32_t DS41_FIRST_DECODER_LAYER = 20;
inline constexpr uint32_t DS41_CANDIDATE_BLOCK_SIZE = 8;
inline constexpr uint32_t DS41_CANDIDATE_TOPK_BLOCKS = 2048;

/* Engram n-gram lookup (layers 1 and 14). */
inline constexpr uint32_t DS41_N_ENGRAM_LAYER = 2;
inline constexpr uint32_t DS41_ENGRAM_LAYERS[DS41_N_ENGRAM_LAYER] = {1, 14};
inline constexpr uint32_t DS41_ENGRAM_MAX_NGRAM = 4;
inline constexpr uint32_t DS41_ENGRAM_HEADS = 8;
inline constexpr uint32_t DS41_ENGRAM_HEAD_DIM = 256;
inline constexpr uint32_t DS41_ENGRAM_COLS =
    (DS41_ENGRAM_MAX_NGRAM - 1) * DS41_ENGRAM_HEADS;
inline constexpr uint32_t DS41_ENGRAM_COMPRESSED_VOCAB = 99092;
inline constexpr uint32_t DS41_ENGRAM_PAD_TOKEN = 2;

inline constexpr char DS41_MODEL_SHAPE_NAME[] = "DeepSeek V4.1 Flash";
inline constexpr float DS41_RMS_EPS = 1.0e-20F;
inline constexpr float DS41_HC_EPS = 1.0e-6F;
inline constexpr float DS41_EXPERT_WEIGHT_SCALE = 1.5F;
inline constexpr float DS41_SWIGLU_CLAMP_EXP = 10.0F;
inline constexpr float DS41_ROPE_FREQ_BASE = 10000.0F;
inline constexpr float DS41_ROPE_SCALE_FACTOR = 16.0F;
inline constexpr float DS41_ROPE_YARN_BETA_FAST = 32.0F;
inline constexpr float DS41_ROPE_YARN_BETA_SLOW = 1.0F;
inline constexpr float DS41_COMPRESS_ROPE_FREQ_BASE = 160000.0F;
inline constexpr uint64_t DS41_ROPE_ORIG_CTX = 65536;

struct ds41_str {
  const char* ptr;
  uint64_t len;
};

struct ds41_kv {
  ds41_str key;
  uint32_t type;
  uint64_t value_pos;
};

struct ds41_tensor {
  ds41_str name;
  uint32_t ndim;
  uint64_t dim[DS41_MAX_DIMS];
  uint32_t type;
  uint64_t rel_offset;
  uint64_t abs_offset;
  uint64_t elements;
  uint64_t bytes;
};

/* A split GGUF is mapped as one contiguous read-only range: shard i starts at
 * shard_offset[i], so tensor abs_offset addresses the whole model and the
 * resident upload path sees a single map. */
struct ds41_model {
  int fd;
  const uint8_t* map;
  uint64_t size;
  uint32_t n_shards;
  int shard_fd[DS41_MAX_SHARDS];
  uint64_t shard_offset[DS41_MAX_SHARDS];
  uint64_t shard_size[DS41_MAX_SHARDS];
  uint32_t version;
  uint64_t n_kv;
  uint64_t n_tensors;
  uint64_t alignment;
  uint64_t tensor_data_pos;
  uint64_t max_tensor_bytes;
  ds41_kv* kv;
  ds41_tensor* tensors;
};

struct ds41_layer_weights {
  ds41_tensor* hc_attn_fn;
  ds41_tensor* hc_attn_scale;
  ds41_tensor* hc_attn_base;
  ds41_tensor* attn_norm;
  ds41_tensor* attn_q_a;
  ds41_tensor* attn_q_a_norm;
  ds41_tensor* attn_q_b;
  ds41_tensor* attn_kv;
  ds41_tensor* attn_kv_a_norm;
  ds41_tensor* attn_sinks;
  ds41_tensor* attn_output_a;
  ds41_tensor* attn_output_b;
  /* KV sources only; the gate exists for ratio 2. */
  ds41_tensor* attn_compressor_kv;
  ds41_tensor* attn_compressor_gate;
  ds41_tensor* attn_compressor_norm;
  /* Index sources; attn_k / k_norm only on KV sources (index keys). */
  ds41_tensor* indexer_attn_q_b;
  ds41_tensor* indexer_proj;
  ds41_tensor* indexer_attn_k;
  ds41_tensor* indexer_k_norm;
  /* Engram layers; their table rows live in the Engram GGUF. engram_gate
   * is the F32 product of the checkpoint's BF16 q and k. */
  ds41_tensor* engram_gate;
  ds41_tensor* engram_kv;
  ds41_tensor* hc_ffn_fn;
  ds41_tensor* hc_ffn_scale;
  ds41_tensor* hc_ffn_base;
  ds41_tensor* ffn_norm;
  ds41_tensor* ffn_gate_inp;
  ds41_tensor* ffn_exp_probs_b;
  /* The gate's bias for image positions (bias_vl). */
  ds41_tensor* ffn_exp_probs_b_vl;
  ds41_tensor* ffn_gate_exps;
  ds41_tensor* ffn_up_exps;
  ds41_tensor* ffn_down_exps;
  ds41_tensor* ffn_gate_shexp;
  ds41_tensor* ffn_up_shexp;
  ds41_tensor* ffn_down_shexp;
};

struct ds41_weights {
  ds41_tensor* token_embd;
  ds41_tensor* output_norm;
  ds41_tensor* output;
  ds41_layer_weights layer[DS41_MAX_LAYER];
  /* Device [layer][expert][embd]: F32 routers whose values are all BF16
   * (Unsloth stores the BF16 router widened) narrowed back, or NULL. The
   * F32 tensors then stay on the host. */
  uint16_t* router_bf16;
  /* Engram n-gram hash constants (metadata). Each head hashes into its own
   * bucket range of the shared table: head_offsets are absolute rows. */
  int32_t engram_token_map[DS41_N_VOCAB];
  int64_t engram_head_vocab[DS41_N_ENGRAM_LAYER * DS41_ENGRAM_COLS];
  int64_t engram_head_offsets[DS41_N_ENGRAM_LAYER * DS41_ENGRAM_COLS];
  int64_t engram_multipliers[DS41_N_ENGRAM_LAYER * DS41_ENGRAM_MAX_NGRAM];
};

/* The Engram GGUF, the last shard of a build: one table of every Engram
 * layer's rows, kept on disk and read row by row. */
struct ds41_engram_tables {
  ds41_model file;
  const ds41_tensor* table;
  uint64_t row_bytes;
};

using token_vec = ds41_tokens;

uint32_t ds41_layer_compress_ratio(uint32_t layer);
bool ds41_layer_is_kv_source(uint32_t layer);
bool ds41_layer_is_index_source(uint32_t layer);
/* Index into DS41_ENGRAM_LAYERS, or -1 when the layer has no Engram. */
int ds41_layer_engram_slot(uint32_t layer);
void* ds41_xcalloc(size_t count, size_t size);
void* ds41_xmalloc(size_t size);
void* ds41_xrealloc(void* pointer, size_t size);
double ds41_now_seconds(void);
uint64_t ds41_align_up(uint64_t value, uint64_t alignment);
const char* ds41_tensor_type_name(uint32_t type);
uint64_t ds41_routed_expert_row_bytes(const ds41_tensor* tensor);

#endif  // GUFO_MODELS_DEEPSEEK_V41_FLASH_RUNTIME_MODEL_DATA_INTERNAL_H_
