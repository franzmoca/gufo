/* DeepSeek V4.1 Flash GGUF loading, validation, and resident ROCm weight
 * binding. */

#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <math.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/file.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include <hip/hip_runtime_api.h>

#include <algorithm>
#include <array>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include "../kernels/rocm/resident_api.h"
#include "expert_store.h"
#include "model.h"
#include "model_data_internal.h"
#include "native_internal.h"
#include "v41_dspark.h"
#include "src/cli/serve/logging.hpp"
#include "src/core/mapped_prefetch.hpp"

static uint32_t g_ds41_compress_ratios[DS41_N_LAYER] = {0};

static int g_ds41_lock_fd = -1;

/* =========================================================================
 * GGUF Quant Block Formats.
 * =========================================================================
 *
 * These layouts and IQ2 tables match the GGUF quantized tensor format,
 * reduced to only the formats ds4.c currently reads:
 *   - Q2_K routed down experts
 *   - Q4_K routed experts in the high-memory variant
 *   - IQ2_XXS routed gate/up experts
 */
#define QK_K 256

struct block_q2_K {
  uint8_t scales[QK_K / 16];
  uint8_t qs[QK_K / 4];
  uint16_t d;
  uint16_t dmin;
};

struct block_q4_K {
  uint16_t d;
  uint16_t dmin;
  uint8_t scales[12];
  uint8_t qs[QK_K / 2];
};

struct block_iq2_xxs {
  uint16_t d;
  uint16_t qs[QK_K / 8];
};

static_assert(sizeof(block_q2_K) == 84);
static_assert(sizeof(block_q4_K) == 144);
static_assert(sizeof(block_iq2_xxs) == 66);

/* =========================================================================
 * Shared Helpers, Allocation Guards, and Cursor Reads.
 * =========================================================================
 *
 * This section holds process-wide utilities used by all later stages:
 * fatal-error helpers, allocation wrappers, and the small byte cursor used to
 * parse GGUF metadata.
 */

#define DS41_GGUF_MAGIC 0x46554747u /* "GGUF", little endian. */
struct ds41_cursor {
  const uint8_t* base;
  uint64_t size;
  uint64_t pos;
  char error[256];
};

[[noreturn]] static void ds41_die(const char* msg) {
  throw std::runtime_error(msg);
}

/* Attention compression is read from GGUF metadata after validating that it
 * matches the exact layout expected for the loaded model shape. */
uint32_t ds41_layer_compress_ratio(uint32_t il) {
  if (il >= DS41_N_LAYER)
    ds41_die("DeepSeek4 layer index is outside the loaded model layout");
  return g_ds41_compress_ratios[il];
}

static uint32_t ds41_expected_layer_compress_ratio(uint32_t il) {
  if (il >= DS41_N_LAYER)
    ds41_die("DeepSeek4 layer index is outside the loaded model layout");
  if (il < 2)
    return 0;
  return il < DS41_CANDIDATE_SOURCE_LAYER ? 2u : 1u;
}

bool ds41_layer_is_kv_source(uint32_t il) {
  for (uint32_t layer : DS41_KV_SOURCE_LAYERS) {
    if (layer == il)
      return true;
  }
  return false;
}

bool ds41_layer_is_index_source(uint32_t il) {
  for (uint32_t layer : DS41_INDEX_SOURCE_LAYERS) {
    if (layer == il)
      return true;
  }
  return false;
}

int ds41_layer_engram_slot(uint32_t il) {
  for (uint32_t i = 0; i < DS41_N_ENGRAM_LAYER; i++) {
    if (DS41_ENGRAM_LAYERS[i] == il)
      return (int)i;
  }
  return -1;
}

static void ds41_die_errno(const char* what, const char* path) {
  fprintf(stderr, "ds41: %s '%s': %s\n", what, path, strerror(errno));
  throw std::runtime_error("incompatible or invalid DeepSeek artifact");
}

static bool ds41_streq(ds41_str s, const char* z) {
  size_t n = strlen(z);
  return s.len == n && memcmp(s.ptr, z, n) == 0;
}

static bool ds41_str_eq(ds41_str a, ds41_str b) {
  return a.len == b.len && memcmp(a.ptr, b.ptr, a.len) == 0;
}

static uint64_t hash_bytes(const void* ptr, uint64_t len) {
  const auto* p = static_cast<const uint8_t*>(ptr);
  uint64_t h = 1469598103934665603ull;
  for (uint64_t i = 0; i < len; i++) {
    h ^= p[i];
    h *= 1099511628211ull;
  }
  return h;
}

void* ds41_xcalloc(size_t n, size_t size) {
  void* p = calloc(n, size);
  if (!p)
    ds41_die("out of memory");
  return p;
}

void* ds41_xmalloc(size_t size) {
  void* p = malloc(size);
  if (!p)
    ds41_die("out of memory");
  return p;
}

void* ds41_xrealloc(void* ptr, size_t size) {
  void* p = realloc(ptr, size);
  if (!p)
    ds41_die("out of memory");
  return p;
}

double ds41_now_seconds(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + (double)ts.tv_nsec * 1.0e-9;
}

static void cursor_error(ds41_cursor* c, const char* msg) {
  if (c->error[0] == '\0') {
    snprintf(c->error, sizeof(c->error), "%s at byte %" PRIu64, msg, c->pos);
  }
}

static bool cursor_has(ds41_cursor* c, uint64_t n) {
  if (n > c->size || c->pos > c->size - n) {
    cursor_error(c, "truncated GGUF file");
    return false;
  }
  return true;
}

static bool cursor_read(ds41_cursor* c, void* dst, uint64_t n) {
  if (!cursor_has(c, n))
    return false;
  memcpy(dst, c->base + c->pos, (size_t)n);
  c->pos += n;
  return true;
}

static bool cursor_skip(ds41_cursor* c, uint64_t n) {
  if (!cursor_has(c, n))
    return false;
  c->pos += n;
  return true;
}

static bool cursor_u32(ds41_cursor* c, uint32_t* v) {
  return cursor_read(c, v, sizeof(*v));
}

static bool cursor_u64(ds41_cursor* c, uint64_t* v) {
  return cursor_read(c, v, sizeof(*v));
}

static bool cursor_string(ds41_cursor* c, ds41_str* s) {
  uint64_t len;
  if (!cursor_u64(c, &len))
    return false;
  if (!cursor_has(c, len))
    return false;
  s->ptr = (const char*)(c->base + c->pos);
  s->len = len;
  c->pos += len;
  return true;
}

uint64_t ds41_align_up(uint64_t value, uint64_t alignment) {
  uint64_t rem = value % alignment;
  return rem == 0 ? value : value + alignment - rem;
}

/* =========================================================================
 * GGUF Parsing and Model Mapping.
 * =========================================================================
 *
 * The loader maps the model once, records metadata/tensor descriptors, and
 * leaves tensor bytes in place.  Inference code accesses weights by adding
 * tensor offsets to the mapping instead of copying the GGUF into private
 * structures.
 */

enum {
  GGUF_VALUE_UINT8 = 0,
  GGUF_VALUE_INT8 = 1,
  GGUF_VALUE_UINT16 = 2,
  GGUF_VALUE_INT16 = 3,
  GGUF_VALUE_UINT32 = 4,
  GGUF_VALUE_INT32 = 5,
  GGUF_VALUE_FLOAT32 = 6,
  GGUF_VALUE_BOOL = 7,
  GGUF_VALUE_STRING = 8,
  GGUF_VALUE_ARRAY = 9,
  GGUF_VALUE_UINT64 = 10,
  GGUF_VALUE_INT64 = 11,
  GGUF_VALUE_FLOAT64 = 12,
};

struct gguf_type_info {
  const char* name;
  uint32_t block_elems;
  uint32_t block_bytes;
};

static constexpr auto gguf_types = [] {
  std::array<gguf_type_info, 40> types{};
  types[0] = {"f32", 1, 4};
  types[1] = {"f16", 1, 2};
  types[2] = {"q4_0", 32, 18};
  types[3] = {"q4_1", 32, 20};
  types[6] = {"q5_0", 32, 22};
  types[7] = {"q5_1", 32, 24};
  types[8] = {"q8_0", 32, 34};
  types[9] = {"q8_1", 32, 40};
  types[10] = {"q2_k", 256, 84};
  types[11] = {"q3_k", 256, 110};
  types[12] = {"q4_k", 256, 144};
  types[13] = {"q5_k", 256, 176};
  types[14] = {"q6_k", 256, 210};
  types[15] = {"q8_k", 256, 292};
  types[16] = {"iq2_xxs", 256, 66};
  types[17] = {"iq2_xs", 256, 74};
  types[18] = {"iq3_xxs", 256, 98};
  types[19] = {"iq1_s", 256, 50};
  types[20] = {"iq4_nl", 32, 18};
  types[21] = {"iq3_s", 256, 110};
  types[22] = {"iq2_s", 256, 82};
  types[23] = {"iq4_xs", 256, 136};
  types[24] = {"i8", 1, 1};
  types[25] = {"i16", 1, 2};
  types[26] = {"i32", 1, 4};
  types[27] = {"i64", 1, 8};
  types[28] = {"f64", 1, 8};
  types[29] = {"iq1_m", 256, 56};
  types[30] = {"bf16", 1, 2};
  types[39] = {"mxfp4", 32, 17};
  return types;
}();

static uint64_t scalar_value_size(uint32_t type) {
  switch (type) {
    case GGUF_VALUE_UINT8:
    case GGUF_VALUE_INT8:
    case GGUF_VALUE_BOOL:
      return 1;
    case GGUF_VALUE_UINT16:
    case GGUF_VALUE_INT16:
      return 2;
    case GGUF_VALUE_UINT32:
    case GGUF_VALUE_INT32:
    case GGUF_VALUE_FLOAT32:
      return 4;
    case GGUF_VALUE_UINT64:
    case GGUF_VALUE_INT64:
    case GGUF_VALUE_FLOAT64:
      return 8;
    default:
      return 0;
  }
}

static bool skip_value(ds41_cursor* c, uint32_t type, int depth) {
  if (depth > 8) {
    cursor_error(c, "metadata array nesting is too deep");
    return false;
  }

  uint64_t scalar = scalar_value_size(type);
  if (scalar != 0)
    return cursor_skip(c, scalar);

  if (type == GGUF_VALUE_STRING) {
    ds41_str ignored;
    return cursor_string(c, &ignored);
  }

  if (type == GGUF_VALUE_ARRAY) {
    uint32_t item_type;
    uint64_t len;

    if (!cursor_u32(c, &item_type))
      return false;
    if (!cursor_u64(c, &len))
      return false;

    uint64_t item_size = scalar_value_size(item_type);
    if (item_size != 0) {
      if (len > UINT64_MAX / item_size) {
        cursor_error(c, "metadata array is too large");
        return false;
      }
      return cursor_skip(c, len * item_size);
    }

    for (uint64_t i = 0; i < len; i++) {
      if (!skip_value(c, item_type, depth + 1))
        return false;
    }
    return true;
  }

  cursor_error(c, "unknown GGUF metadata type");
  return false;
}

static const gguf_type_info* tensor_type(uint32_t type) {
  if (type >= gguf_types.size() || gguf_types[type].name == nullptr) {
    return nullptr;
  }
  return &gguf_types[type];
}

const char* ds41_tensor_type_name(uint32_t type) {
  const gguf_type_info* info = tensor_type(type);
  return info ? info->name : "unknown";
}

static bool tensor_nbytes(uint32_t type, uint64_t elements, uint64_t* bytes) {
  const gguf_type_info* info = tensor_type(type);
  if (!info || info->block_elems == 0)
    return false;
  uint64_t blocks = (elements + info->block_elems - 1) / info->block_elems;
  if (blocks > UINT64_MAX / info->block_bytes)
    return false;
  *bytes = blocks * info->block_bytes;
  return true;
}

static ds41_cursor cursor_at(const ds41_model* m, uint64_t pos) {
  ds41_cursor c{m->map, m->size, pos, {0}};
  return c;
}

static ds41_kv* model_find_kv(const ds41_model* m, const char* key) {
  for (uint64_t i = 0; i < m->n_kv; i++) {
    if (ds41_streq(m->kv[i].key, key))
      return &m->kv[i];
  }
  return nullptr;
}

/* A u32 value; a non-negative int32 also qualifies (llama.cpp writes layer
 * ids that way). */
static bool model_get_u32(const ds41_model* m, const char* key, uint32_t* out) {
  ds41_kv* kv = model_find_kv(m, key);
  if (!kv || (kv->type != GGUF_VALUE_UINT32 && kv->type != GGUF_VALUE_INT32))
    return false;
  ds41_cursor c = cursor_at(m, kv->value_pos);
  if (!cursor_u32(&c, out))
    return false;
  return kv->type == GGUF_VALUE_UINT32 || (int32_t)*out >= 0;
}

static bool model_get_u64(const ds41_model* m, const char* key, uint64_t* out) {
  ds41_kv* kv = model_find_kv(m, key);
  if (!kv)
    return false;
  ds41_cursor c = cursor_at(m, kv->value_pos);
  if (kv->type == GGUF_VALUE_UINT64) {
    return cursor_u64(&c, out);
  }
  if (kv->type == GGUF_VALUE_UINT32) {
    uint32_t v = 0;
    if (!cursor_u32(&c, &v))
      return false;
    *out = v;
    return true;
  }
  return false;
}

static bool model_get_f32(const ds41_model* m, const char* key, float* out) {
  ds41_kv* kv = model_find_kv(m, key);
  if (!kv)
    return false;
  ds41_cursor c = cursor_at(m, kv->value_pos);
  if (kv->type == GGUF_VALUE_FLOAT32) {
    return cursor_read(&c, out, sizeof(*out));
  }
  if (kv->type == GGUF_VALUE_FLOAT64) {
    double v = 0.0;
    if (!cursor_read(&c, &v, sizeof(v)))
      return false;
    *out = (float)v;
    return true;
  }
  if (kv->type == GGUF_VALUE_UINT32) {
    uint32_t v = 0;
    if (!cursor_u32(&c, &v))
      return false;
    *out = (float)v;
    return true;
  }
  if (kv->type == GGUF_VALUE_INT32) {
    int32_t v = 0;
    if (!cursor_read(&c, &v, sizeof(v)))
      return false;
    *out = (float)v;
    return true;
  }
  return false;
}

static bool model_get_bool(const ds41_model* m, const char* key, bool* out) {
  ds41_kv* kv = model_find_kv(m, key);
  if (!kv || kv->type != GGUF_VALUE_BOOL)
    return false;
  ds41_cursor c = cursor_at(m, kv->value_pos);
  uint8_t v = 0;
  if (!cursor_read(&c, &v, sizeof(v)))
    return false;
  *out = v != 0;
  return true;
}

struct ds41_array_ref {
  uint32_t type;
  uint64_t len;
  uint64_t data_pos;
};

static bool model_get_array(const ds41_model* m, const char* key,
                            ds41_array_ref* out) {
  ds41_kv* kv = model_find_kv(m, key);
  if (!kv || kv->type != GGUF_VALUE_ARRAY)
    return false;

  ds41_cursor c = cursor_at(m, kv->value_pos);
  if (!cursor_u32(&c, &out->type))
    return false;
  if (!cursor_u64(&c, &out->len))
    return false;
  out->data_pos = c.pos;
  return true;
}

bool ds41_model_string_array_begin(const ds41_model* model, const char* key,
                                   ds41_string_iterator* iterator) {
  if (!model || !key || !iterator)
    return false;

  ds41_array_ref array;
  if (!model_get_array(model, key, &array) || array.type != GGUF_VALUE_STRING) {
    return false;
  }
  *iterator = ds41_string_iterator{model, array.data_pos, array.len};
  return true;
}

bool ds41_model_string_array_next(ds41_string_iterator* iterator,
                                  ds41_string_view* value) {
  if (!iterator || !value || !iterator->model || iterator->remaining == 0) {
    return false;
  }

  ds41_cursor cursor = cursor_at(iterator->model, iterator->pos);
  ds41_str string;
  if (!cursor_string(&cursor, &string))
    return false;
  iterator->pos = cursor.pos;
  iterator->remaining--;
  *value = ds41_string_view{string.ptr, string.len};
  return true;
}

static void model_close(ds41_model* m) {
  if (!m)
    return;
  free(m->kv);
  free(m->tensors);
  if (m->map)
    munmap((void*)m->map, (size_t)m->size);
  for (uint32_t i = 0; i < m->n_shards; i++) {
    if (m->shard_fd[i] >= 0)
      close(m->shard_fd[i]);
  }
  memset(m, 0, sizeof(*m));
  m->fd = -1;
}

/* Read one shard's GGUF metadata table. Values stay in the mapping; only the
 * first shard's keys are kept, the others carry split bookkeeping. */
static void parse_metadata(ds41_model* m, ds41_cursor* c, uint64_t n_kv,
                           bool keep) {
  if (keep) {
    m->kv = static_cast<ds41_kv*>(
        calloc(static_cast<size_t>(n_kv), sizeof(m->kv[0])));
    if (!m->kv)
      ds41_die("out of memory while allocating metadata table");
    m->n_kv = n_kv;
    m->alignment = 32;
  }
  uint64_t alignment = 32;
  for (uint64_t i = 0; i < n_kv; i++) {
    ds41_kv kv{};
    if (!cursor_string(c, &kv.key))
      ds41_die(c->error);
    if (!cursor_u32(c, &kv.type))
      ds41_die(c->error);
    kv.value_pos = c->pos;
    if (ds41_streq(kv.key, "general.alignment") &&
        kv.type == GGUF_VALUE_UINT32) {
      ds41_cursor tmp = *c;
      uint32_t value;
      if (cursor_u32(&tmp, &value) && value != 0)
        alignment = value;
    }
    if (!skip_value(c, kv.type, 0))
      ds41_die(c->error);
    if (keep)
      m->kv[i] = kv;
  }
  if (keep)
    m->alignment = alignment;
  if (alignment != m->alignment)
    ds41_die("GGUF shards use different alignments");
}

/* Append one shard's tensor directory, converting relative offsets to
 * offsets in the combined mapping. Tensor bytes are never copied here. */
static void parse_tensors(ds41_model* m, ds41_cursor* c, uint32_t shard,
                          uint64_t n_tensors) {
  const uint64_t first = m->n_tensors;
  ds41_tensor* grown = static_cast<ds41_tensor*>(
      realloc(m->tensors,
              static_cast<size_t>(first + n_tensors) * sizeof(m->tensors[0])));
  if (!grown)
    ds41_die("out of memory while allocating tensor table");
  m->tensors = grown;
  memset(m->tensors + first, 0,
         static_cast<size_t>(n_tensors) * sizeof(m->tensors[0]));
  m->n_tensors = first + n_tensors;

  for (uint64_t i = first; i < m->n_tensors; i++) {
    ds41_tensor* t = &m->tensors[i];
    if (!cursor_string(c, &t->name))
      ds41_die(c->error);
    if (!cursor_u32(c, &t->ndim))
      ds41_die(c->error);
    if (t->ndim == 0 || t->ndim > DS41_MAX_DIMS) {
      ds41_die("tensor has an unsupported number of dimensions");
    }
    t->elements = 1;
    for (uint32_t d = 0; d < t->ndim; d++) {
      if (!cursor_u64(c, &t->dim[d]))
        ds41_die(c->error);
      if (t->dim[d] != 0 && t->elements > UINT64_MAX / t->dim[d]) {
        ds41_die("tensor element count overflow");
      }
      t->elements *= t->dim[d];
    }
    if (!cursor_u32(c, &t->type))
      ds41_die(c->error);
    if (!cursor_u64(c, &t->rel_offset))
      ds41_die(c->error);
    if (!tensor_nbytes(t->type, t->elements, &t->bytes)) {
      gufo::server::Logger::LogFormatted(
          gufo::server::LogLevel::kWarn, "ds41",
          "warning: tensor %.*s has unsupported GGUF type %u", (int)t->name.len,
          t->name.ptr, t->type);
    }
  }

  const uint64_t base = m->shard_offset[shard];
  const uint64_t limit = base + m->shard_size[shard];
  const uint64_t data_pos = base + ds41_align_up(c->pos - base, m->alignment);
  if (shard == 0)
    m->tensor_data_pos = data_pos;
  for (uint64_t i = first; i < m->n_tensors; i++) {
    ds41_tensor* t = &m->tensors[i];
    if (t->rel_offset > UINT64_MAX - data_pos)
      ds41_die("tensor offset overflow");
    t->abs_offset = data_pos + t->rel_offset;
    if (t->abs_offset > limit || t->bytes > limit - t->abs_offset) {
      ds41_die("tensor points outside its GGUF shard");
    }
    if (t->bytes > m->max_tensor_bytes)
      m->max_tensor_bytes = t->bytes;
  }
}

/* Shard paths for "<stem>-00001-of-0000N.gguf"; a plain path is one shard. */
static std::vector<std::string> model_shard_paths(const char* path) {
  const std::string first(path);
  static constexpr char kSuffix[] = "-00001-of-";
  const size_t at = first.rfind(kSuffix);
  if (at == std::string::npos ||
      first.size() != at + strlen(kSuffix) + 5 + strlen(".gguf") ||
      first.compare(first.size() - 5, 5, ".gguf") != 0) {
    return {first};
  }
  const std::string count_text = first.substr(at + strlen(kSuffix), 5);
  const int count = atoi(count_text.c_str());
  if (count < 1 || count > (int)DS41_MAX_SHARDS)
    ds41_die("unsupported GGUF shard count");
  std::vector<std::string> paths;
  for (int i = 1; i <= count; i++) {
    char index[8];
    snprintf(index, sizeof(index), "%05d", i);
    paths.push_back(first.substr(0, at) + "-" + index + "-of-" + count_text +
                    ".gguf");
  }
  return paths;
}

/* Open every shard and map them back to back into one reserved range. A
 * 2 MiB gap separates shards so merged upload spans never cross a file. */
static void model_open(ds41_model* m, const std::vector<std::string>& paths) {
  memset(m, 0, sizeof(*m));
  m->fd = -1;
  static constexpr uint64_t kShardGap = 2ull << 20;

  uint64_t total = 0;
  for (size_t i = 0; i < paths.size(); i++) {
    m->shard_fd[i] = -1;
  }
  m->n_shards = (uint32_t)paths.size();
  for (size_t i = 0; i < paths.size(); i++) {
    const int fd = open(paths[i].c_str(), O_RDONLY | O_CLOEXEC);
    if (fd == -1) {
      ds41_die_errno("cannot open model", paths[i].c_str());
      ds41_die("cannot open model shard");
    }
    m->shard_fd[i] = fd;
    struct stat st;
    if (fstat(fd, &st) == -1)
      ds41_die("cannot stat model shard");
    if (st.st_size < 32)
      ds41_die("model shard is too small to be GGUF");
    m->shard_offset[i] = total;
    m->shard_size[i] = (uint64_t)st.st_size;
    total = ds41_align_up(total + (uint64_t)st.st_size, kShardGap) + kShardGap;
  }
  m->fd = m->shard_fd[0];

  void* region = mmap(NULL, (size_t)total, PROT_NONE,
                      MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
  if (region == MAP_FAILED)
    ds41_die("cannot reserve the model address range");
  m->map = static_cast<const uint8_t*>(region);
  m->size = total;
  for (uint32_t i = 0; i < m->n_shards; i++) {
    void* at = (uint8_t*)region + m->shard_offset[i];
    void* mapped = mmap(at, (size_t)m->shard_size[i], PROT_READ,
                        MAP_SHARED | MAP_FIXED, m->shard_fd[i], 0);
    if (mapped == MAP_FAILED) {
      ds41_die_errno("cannot mmap model", paths[i].c_str());
      ds41_die("cannot mmap model shard");
    }
  }

  for (uint32_t i = 0; i < m->n_shards; i++) {
    ds41_cursor c{
        m->map, m->shard_offset[i] + m->shard_size[i], m->shard_offset[i], {0}};
    uint32_t magic, version;
    uint64_t n_tensors, n_kv;
    if (!cursor_u32(&c, &magic))
      ds41_die(c.error);
    if (magic != DS41_GGUF_MAGIC)
      ds41_die("model shard is not a GGUF file");
    if (!cursor_u32(&c, &version))
      ds41_die(c.error);
    if (version != 3)
      ds41_die("only GGUF v3 is supported");
    if (!cursor_u64(&c, &n_tensors))
      ds41_die(c.error);
    if (!cursor_u64(&c, &n_kv))
      ds41_die(c.error);
    if (i == 0)
      m->version = version;
    parse_metadata(m, &c, n_kv, i == 0);
    parse_tensors(m, &c, i, n_tensors);
  }
}

static ds41_tensor* model_find_tensor(const ds41_model* m, const char* name) {
  const size_t len = strlen(name);
  for (uint64_t i = 0; i < m->n_tensors; i++) {
    if (m->tensors[i].name.len == len &&
        memcmp(m->tensors[i].name.ptr, name, len) == 0) {
      return &m->tensors[i];
    }
  }
  return NULL;
}

struct accelerator_tensor_span {
  uint64_t off;
  uint64_t end;
};

/* Spans never split a tensor: device lookups need each tensor in one range. */
static uint64_t accelerator_rocm_preload_span_bytes(const ds41_model* m) {
  const uint64_t floor = 1024ull * 1048576ull;
  return m->max_tensor_bytes > floor ? m->max_tensor_bytes : floor;
}

/* The token embedding is dequantized on the host one row per token, so it
 * stays in the mapped file instead of taking device memory. */
static bool tensor_device_resident(const ds41_tensor* t, const ds41_weights* weights, bool streaming) {
  static const char kHostOnly[] = "token_embd.weight";
  if (t->name.len == sizeof(kHostOnly) - 1 && memcmp(t->name.ptr, kHostOnly, sizeof(kHostOnly) - 1) == 0)
    return false;
  if (streaming && weights)
    for (const ds41_layer_weights& l : weights->layer)
      if (t == l.ffn_gate_exps || t == l.ffn_up_exps || t == l.ffn_down_exps) return false;
  if (weights && weights->router_bf16)
    for (const ds41_layer_weights& l : weights->layer)
      if (t == l.ffn_gate_inp) return false;
  return true;
}

constexpr uint64_t kRouterElements = (uint64_t)DS41_N_EMBD * DS41_N_EXPERT;

/* Narrows F32 routers to BF16 on the device when that is exact, so they run
 * the BF16 router kernels at half the bytes. Leaves router_bf16 NULL (the
 * F32 path) for BF16 routers or any value BF16 cannot hold. */
static bool narrow_f32_routers(ds41_weights* w, const ds41_model* m) {
  for (uint32_t il = 0; il < DS41_N_LAYER; il++)
    if (w->layer[il].ffn_gate_inp->type != DS41_TENSOR_F32) return true;
  std::vector<uint16_t> narrow((uint64_t)DS41_N_LAYER * kRouterElements);
  for (uint32_t il = 0; il < DS41_N_LAYER; il++) {
    const uint8_t* src = m->map + w->layer[il].ffn_gate_inp->abs_offset;
    uint16_t* dst = narrow.data() + (uint64_t)il * kRouterElements;
    for (uint64_t i = 0; i < kRouterElements; i++) {
      uint32_t bits;
      memcpy(&bits, src + 4 * i, 4);
      if (bits & 0xffffu) return true;
      dst[i] = (uint16_t)(bits >> 16);
    }
  }
  const uint64_t bytes = narrow.size() * sizeof(uint16_t);
  void* p = NULL;
  if (hipMalloc(&p, bytes) != hipSuccess) return false;
  if (hipMemcpy(p, narrow.data(), bytes, hipMemcpyHostToDevice) != hipSuccess) {
    (void)hipFree(p);
    return false;
  }
  w->router_bf16 = (uint16_t*)p;
  return true;
}

static bool accelerator_cache_model_tensor_spans(const ds41_model* m,
                                                 const ds41_weights* weights,
                                                 bool streaming,
                                                 uint64_t* cached_out) {
  std::vector<accelerator_tensor_span> spans;
  spans.reserve(static_cast<size_t>(m->n_tensors));
  for (uint64_t i = 0; i < m->n_tensors; i++) {
    const ds41_tensor* t = &m->tensors[i];
    if (t->bytes == 0 || !tensor_device_resident(t, weights, streaming))
      continue;
    if (t->abs_offset > m->size || t->bytes > m->size - t->abs_offset) {
      return false;
    }
    spans.push_back(
        accelerator_tensor_span{t->abs_offset, t->abs_offset + t->bytes});
  }
  std::sort(spans.begin(), spans.end(),
            [](const auto& left, const auto& right) {
              return left.off < right.off ||
                     (left.off == right.off && left.end < right.end);
            });

  const uint64_t max_span = accelerator_rocm_preload_span_bytes(m);
  uint64_t cached = 0;
  uint64_t merged = 0;
  for (size_t i = 0; i < spans.size();) {
    uint64_t off = spans[i].off;
    uint64_t end = spans[i].end;
    i++;
    while (i < spans.size() && spans[i].off <= end + 65536u &&
           spans[i].end - off <= max_span) {
      if (spans[i].end > end)
        end = spans[i].end;
      i++;
    }
    while (off < end) {
      uint64_t chunk_end = end;
      if (chunk_end - off > max_span)
        chunk_end = off + max_span;
      char label[96];
      snprintf(label, sizeof(label), "tensor-span:%" PRIu64, merged);
      if (ds41_gpu_cache_model_range(m->map, m->size, off, chunk_end - off,
                                     label) == 0) {
        fprintf(stderr,
                "ds41: accelerator failed to cache model tensor span %" PRIu64
                " at offset %" PRIu64 "\n",
                merged, off);
        return false;
      }
      cached += chunk_end - off;
      merged++;
      off = chunk_end;
    }
  }
  if (cached_out)
    *cached_out = cached;
  return true;
}

static bool accelerator_cache_model_tensors(const ds41_model* m, const ds41_weights* weights, bool streaming) {
  if (!m || !m->map || m->size == 0)
    return false;

  const double t0 = ds41_now_seconds();
  uint64_t cached = 0;
  const bool cache_ok = accelerator_cache_model_tensor_spans(m, weights, streaming, &cached);
  const bool upload_ok = ds41_gpu_release_model_staging() != 0;
  if (!cache_ok || !upload_ok)
    return false;
  if (cached != 0) {
    const double t1 = ds41_now_seconds();
    gufo::server::Logger::LogFormatted(
        gufo::server::LogLevel::kInfo, "ds4",
        "ROCm startup model cache prepared %.2f GiB of tensor spans in "
        "%.3fs",
        (double)cached / 1073741824.0, t1 - t0);
  }
  return true;
}

/* =========================================================================
 * Fixed Weight Binding and Model Validation.
 * =========================================================================
 *
 * The GGUF tensor directory is converted into a DS4-specific pointer table.
 * After this section, the rest of the program addresses tensors by semantic
 * fields such as layer->attn_q_a or layer->ffn_gate_exps rather than by string
 * lookup.  Shape validation is intentionally strict.
 */

static uint32_t required_u32(const ds41_model* m, const char* key) {
  uint32_t v = 0;
  if (!model_get_u32(m, key, &v)) {
    fprintf(stderr, "ds41: required metadata key is missing: %s\n", key);
    throw std::runtime_error("incompatible or invalid DeepSeek artifact");
  }
  return v;
}

static float required_f32(const ds41_model* m, const char* key) {
  float v = 0.0f;
  if (!model_get_f32(m, key, &v)) {
    fprintf(stderr, "ds41: required metadata key is missing: %s\n", key);
    throw std::runtime_error("incompatible or invalid DeepSeek artifact");
  }
  return v;
}

static bool required_bool(const ds41_model* m, const char* key) {
  bool v = false;
  if (!model_get_bool(m, key, &v)) {
    fprintf(stderr, "ds41: required metadata key is missing: %s\n", key);
    throw std::runtime_error("incompatible or invalid DeepSeek artifact");
  }
  return v;
}

static ds41_tensor* required_tensor(const ds41_model* m, const char* name) {
  ds41_tensor* t = model_find_tensor(m, name);
  if (!t) {
    fprintf(stderr, "ds41: required tensor is missing: %s\n", name);
    throw std::runtime_error("incompatible or invalid DeepSeek artifact");
  }
  return t;
}

static ds41_tensor* tensor_by_namef(const ds41_model* m, const char* fmt,
                                    uint32_t layer) {
  char name[128];
  int n = snprintf(name, sizeof(name), fmt, layer);
  if (n < 0 || (size_t)n >= sizeof(name))
    ds41_die("tensor name is too long");
  return model_find_tensor(m, name);
}

static ds41_tensor* required_tensorf(const ds41_model* m, const char* fmt,
                                     uint32_t layer) {
  char name[128];
  int n = snprintf(name, sizeof(name), fmt, layer);
  if (n < 0 || (size_t)n >= sizeof(name))
    ds41_die("tensor name is too long");
  return required_tensor(m, name);
}

static void tensor_expect_layout(const ds41_tensor* t, uint32_t type,
                                 uint32_t ndim, uint64_t d0, uint64_t d1,
                                 uint64_t d2) {
  if (!t)
    ds41_die("internal error: missing tensor while validating layout");
  if (t->type != type) {
    fprintf(stderr, "ds41: tensor %.*s has type %s, expected %s\n",
            (int)t->name.len, t->name.ptr, ds41_tensor_type_name(t->type),
            ds41_tensor_type_name(type));
    throw std::runtime_error("incompatible or invalid DeepSeek artifact");
  }
  if (t->ndim != ndim) {
    fprintf(stderr, "ds41: tensor %.*s has %u dimensions, expected %u\n",
            (int)t->name.len, t->name.ptr, t->ndim, ndim);
    throw std::runtime_error("incompatible or invalid DeepSeek artifact");
  }

  const uint64_t want[3] = {d0, d1, d2};
  for (uint32_t i = 0; i < ndim; i++) {
    if (t->dim[i] == want[i])
      continue;
    fprintf(stderr,
            "ds41: tensor %.*s has dim[%u]=%" PRIu64 ", expected %" PRIu64 "\n",
            (int)t->name.len, t->name.ptr, i, t->dim[i], want[i]);
    throw std::runtime_error("incompatible or invalid DeepSeek artifact");
  }
}

static void tensor_expect_optional(const ds41_tensor* t, uint32_t type,
                                   uint32_t ndim, uint64_t d0, uint64_t d1,
                                   uint64_t d2) {
  if (t)
    tensor_expect_layout(t, type, ndim, d0, d1, d2);
}

static bool tensor_is_routed_expert_type(uint32_t type) {
  switch (type) {
    case DS41_TENSOR_IQ1_S:
    case DS41_TENSOR_IQ1_M:
    case DS41_TENSOR_IQ2_XXS:
    case DS41_TENSOR_IQ2_XS:
    case DS41_TENSOR_IQ2_S:
    case DS41_TENSOR_IQ3_XXS:
    case DS41_TENSOR_IQ3_S:
    case DS41_TENSOR_Q2_K:
    case DS41_TENSOR_Q4_K:
      return true;
    default:
      return false;
  }
}

/* Weight types the dense projections may use; the graph dispatches on them. */
static bool tensor_is_dense_type(uint32_t type) {
  return type == DS41_TENSOR_Q8_0 || type == DS41_TENSOR_Q5_K ||
         type == DS41_TENSOR_Q6_K || type == DS41_TENSOR_BF16 ||
         type == DS41_TENSOR_F16;
}

uint64_t ds41_routed_expert_row_bytes(const ds41_tensor* t) {
  const gguf_type_info* info = tensor_type(t->type);
  if (!info || (t->dim[0] % info->block_elems) != 0)
    ds41_die("routed expert row is not block aligned");
  return (t->dim[0] / info->block_elems) * info->block_bytes;
}

[[noreturn]] static void tensor_reject(const ds41_tensor* t, const char* what) {
  fprintf(stderr, "ds41: tensor %.*s %s\n", (int)t->name.len, t->name.ptr,
          what);
  throw std::runtime_error("incompatible or invalid DeepSeek V4.1 artifact");
}

static void tensor_expect_dims(const ds41_tensor* t, uint32_t ndim, uint64_t d0,
                               uint64_t d1, uint64_t d2) {
  if (!t)
    ds41_die("internal error: missing tensor while validating layout");
  if (t->ndim != ndim)
    tensor_reject(t, "has the wrong number of dimensions");
  const uint64_t want[3] = {d0, d1, d2};
  for (uint32_t i = 0; i < ndim; i++) {
    if (t->dim[i] != want[i]) {
      fprintf(stderr,
              "ds41: tensor %.*s has dim[%u]=%" PRIu64 ", expected %" PRIu64
              "\n",
              (int)t->name.len, t->name.ptr, i, t->dim[i], want[i]);
      throw std::runtime_error(
          "incompatible or invalid DeepSeek V4.1 artifact");
    }
  }
}

static void tensor_expect(const ds41_tensor* t, uint32_t type, uint32_t ndim,
                          uint64_t d0, uint64_t d1 = 0, uint64_t d2 = 0) {
  tensor_expect_dims(t, ndim, d0, d1, d2);
  if (t->type != type) {
    fprintf(stderr, "ds41: tensor %.*s has type %s, expected %s\n",
            (int)t->name.len, t->name.ptr, ds41_tensor_type_name(t->type),
            ds41_tensor_type_name(type));
    throw std::runtime_error("incompatible or invalid DeepSeek V4.1 artifact");
  }
}

static void tensor_expect_dense(const ds41_tensor* t, uint32_t ndim,
                                uint64_t d0, uint64_t d1, uint64_t d2 = 0) {
  tensor_expect_dims(t, ndim, d0, d1, d2);
  if (!tensor_is_dense_type(t->type))
    tensor_reject(t, "has an unsupported dense weight type");
}

static void tensor_expect_routed_expert(const ds41_tensor* t, uint64_t d0,
                                        uint64_t d1) {
  tensor_expect_dims(t, 3, d0, d1, DS41_N_EXPERT);
  if (!tensor_is_routed_expert_type(t->type))
    tensor_reject(t, "has an unsupported routed expert type");
}

/* Verify every tensor type and dimension the V4.1 graph uses. After this,
 * inference code relies on the fixed constants and the per-tensor types. */
static void weights_validate_layout(const ds41_weights* w) {
  const uint64_t hc_dim = (uint64_t)DS41_N_EMBD * DS41_N_HC;
  const uint64_t hc_mix_dim = 2u * DS41_N_HC + (uint64_t)DS41_N_HC * DS41_N_HC;
  const uint64_t q_dim = (uint64_t)DS41_N_HEAD * DS41_N_HEAD_DIM;
  const uint64_t group_dim =
      (uint64_t)DS41_N_HEAD_DIM * (DS41_N_HEAD / DS41_N_OUT_GROUP);
  const uint64_t out_low_dim = (uint64_t)DS41_N_OUT_GROUP * DS41_N_LORA_O;
  const uint64_t index_q_dim =
      (uint64_t)DS41_N_INDEXER_HEAD * DS41_N_INDEXER_HEAD_DIM;
  const uint64_t engram_in = (uint64_t)DS41_ENGRAM_COLS * DS41_ENGRAM_HEAD_DIM;
  const uint64_t engram_out = (uint64_t)DS41_N_EMBD * (DS41_N_HC + 1);

  tensor_expect_dense(w->token_embd, 2, DS41_N_EMBD, DS41_N_VOCAB);
  tensor_expect(w->output_norm, DS41_TENSOR_F32, 1, DS41_N_EMBD);
  tensor_expect_dense(w->output, 2, DS41_N_EMBD, DS41_N_VOCAB);

  for (uint32_t il = 0; il < DS41_N_LAYER; il++) {
    const ds41_layer_weights* l = &w->layer[il];
    const uint32_t ratio = ds41_layer_compress_ratio(il);

    tensor_expect(l->hc_attn_fn, DS41_TENSOR_F32, 2, hc_dim, hc_mix_dim);
    tensor_expect(l->hc_attn_scale, DS41_TENSOR_F32, 1, 3);
    tensor_expect(l->hc_attn_base, DS41_TENSOR_F32, 1, hc_mix_dim);
    tensor_expect(l->attn_norm, DS41_TENSOR_F32, 1, DS41_N_EMBD);
    tensor_expect_dense(l->attn_q_a, 2, DS41_N_EMBD, DS41_N_LORA_Q);
    tensor_expect(l->attn_q_a_norm, DS41_TENSOR_F32, 1, DS41_N_LORA_Q);
    tensor_expect_dense(l->attn_q_b, 2, DS41_N_LORA_Q, q_dim);
    tensor_expect_dense(l->attn_kv, 2, DS41_N_EMBD, DS41_N_HEAD_DIM);
    tensor_expect(l->attn_kv_a_norm, DS41_TENSOR_F32, 1, DS41_N_HEAD_DIM);
    tensor_expect(l->attn_sinks, DS41_TENSOR_F32, 1, DS41_N_HEAD);
    tensor_expect_dense(l->attn_output_a, 2, group_dim, out_low_dim);
    tensor_expect_dense(l->attn_output_b, 2, out_low_dim, DS41_N_EMBD);

    if (ds41_layer_is_kv_source(il)) {
      tensor_expect(l->attn_compressor_kv, DS41_TENSOR_BF16, 2, DS41_N_EMBD,
                    DS41_N_HEAD_DIM);
      tensor_expect(l->attn_compressor_norm, DS41_TENSOR_F32, 1,
                    DS41_N_HEAD_DIM);
      if (ratio > 1) {
        tensor_expect(l->attn_compressor_gate, DS41_TENSOR_BF16, 2, DS41_N_EMBD,
                      DS41_N_HEAD_DIM);
      } else if (l->attn_compressor_gate) {
        tensor_reject(l->attn_compressor_gate,
                      "is unexpected on a ratio-1 layer");
      }
      tensor_expect(l->indexer_attn_k, DS41_TENSOR_BF16, 2, DS41_N_HEAD_DIM,
                    DS41_N_INDEXER_HEAD_DIM);
      tensor_expect(l->indexer_k_norm, DS41_TENSOR_F32, 1,
                    DS41_N_INDEXER_HEAD_DIM);
    }
    if (ds41_layer_is_index_source(il)) {
      tensor_expect(l->indexer_attn_q_b, DS41_TENSOR_BF16, 2, DS41_N_LORA_Q,
                    index_q_dim);
      tensor_expect(l->indexer_proj, DS41_TENSOR_BF16, 2, DS41_N_EMBD,
                    DS41_N_INDEXER_HEAD);
    }
    if (ds41_layer_engram_slot(il) >= 0) {
      tensor_expect(l->engram_gate, DS41_TENSOR_F32, 2, DS41_N_EMBD,
                    DS41_N_HC);
      tensor_expect_dense(l->engram_kv, 2, engram_in, engram_out);
    }

    tensor_expect(l->hc_ffn_fn, DS41_TENSOR_F32, 2, hc_dim, hc_mix_dim);
    tensor_expect(l->hc_ffn_scale, DS41_TENSOR_F32, 1, 3);
    tensor_expect(l->hc_ffn_base, DS41_TENSOR_F32, 1, hc_mix_dim);
    tensor_expect(l->ffn_norm, DS41_TENSOR_F32, 1, DS41_N_EMBD);
    /* The router is BF16 in the ISTA builds and F32 in Unsloth's. */
    tensor_expect_dims(l->ffn_gate_inp, 2, DS41_N_EMBD, DS41_N_EXPERT, 0);
    if (l->ffn_gate_inp->type != DS41_TENSOR_BF16 &&
        l->ffn_gate_inp->type != DS41_TENSOR_F32)
      tensor_reject(l->ffn_gate_inp, "has an unsupported router type");
    tensor_expect(l->ffn_exp_probs_b, DS41_TENSOR_F32, 1, DS41_N_EXPERT);
    tensor_expect(l->ffn_exp_probs_b_vl, DS41_TENSOR_F32, 1, DS41_N_EXPERT);
    tensor_expect_routed_expert(l->ffn_gate_exps, DS41_N_EMBD, DS41_N_FF_EXP);
    tensor_expect_routed_expert(l->ffn_up_exps, DS41_N_EMBD, DS41_N_FF_EXP);
    tensor_expect_routed_expert(l->ffn_down_exps, DS41_N_FF_EXP, DS41_N_EMBD);
    if (l->ffn_gate_exps->type != l->ffn_up_exps->type) {
      fprintf(stderr,
              "ds41: routed gate/up experts use different quant types in layer "
              "%u\n",
              il);
      throw std::runtime_error(
          "incompatible or invalid DeepSeek V4.1 artifact");
    }
    tensor_expect_dense(l->ffn_gate_shexp, 2, DS41_N_EMBD, DS41_N_FF_EXP);
    tensor_expect_dense(l->ffn_up_shexp, 2, DS41_N_EMBD, DS41_N_FF_EXP);
    tensor_expect_dense(l->ffn_down_shexp, 2, DS41_N_FF_EXP, DS41_N_EMBD);
  }
}

static void config_expect_u32(const char* name, uint32_t got,
                              uint32_t expected) {
  if (got == expected)
    return;
  fprintf(stderr, "ds41: expected %s=%u for %s, got %u\n", name, expected,
          DS41_MODEL_SHAPE_NAME, got);
  throw std::runtime_error("incompatible or invalid DeepSeek V4.1 artifact");
}

static void config_expect_f32(const char* name, float got, float expected) {
  const float scale = fabsf(expected) > 1.0f ? fabsf(expected) : 1.0f;
  if (fabsf(got - expected) <= scale * 1.0e-6f || got == expected)
    return;
  fprintf(stderr, "ds41: expected %s=%.9g for %s, got %.9g\n", name,
          (double)expected, DS41_MODEL_SHAPE_NAME, (double)got);
  throw std::runtime_error("incompatible or invalid DeepSeek V4.1 artifact");
}

static void config_expect_bool(const char* name, bool got, bool expected) {
  if (got == expected)
    return;
  fprintf(stderr, "ds41: expected %s=%s for %s\n", name,
          expected ? "true" : "false", DS41_MODEL_SHAPE_NAME);
  throw std::runtime_error("incompatible or invalid DeepSeek V4.1 artifact");
}

static std::string required_string(const ds41_model* m, const char* key) {
  ds41_kv* kv = model_find_kv(m, key);
  ds41_str value{};
  if (kv && kv->type == GGUF_VALUE_STRING) {
    ds41_cursor c = cursor_at(m, kv->value_pos);
    if (cursor_string(&c, &value))
      return std::string(value.ptr, value.len);
  }
  fprintf(stderr, "ds41: required string metadata key is missing: %s\n", key);
  throw std::runtime_error("incompatible or invalid DeepSeek V4.1 artifact");
}

/* Read an integer array (u32/i32/u64/i64) and require it to equal `want`,
 * optionally followed by zeros up to its length. */
static void config_expect_int_array(const ds41_model* m, const char* key,
                                    const uint64_t* want, uint32_t count,
                                    bool zero_tail, uint32_t* out = nullptr) {
  ds41_array_ref arr;
  if (!model_get_array(m, key, &arr) || scalar_value_size(arr.type) == 0 ||
      arr.type == GGUF_VALUE_FLOAT32 || arr.type == GGUF_VALUE_FLOAT64 ||
      arr.len < count || (!zero_tail && arr.len != count)) {
    fprintf(
        stderr,
        "ds41: integer array metadata %s is missing or has the wrong length\n",
        key);
    throw std::runtime_error("incompatible or invalid DeepSeek V4.1 artifact");
  }
  ds41_cursor c = cursor_at(m, arr.data_pos);
  const uint64_t width = scalar_value_size(arr.type);
  for (uint64_t i = 0; i < arr.len; i++) {
    uint64_t raw = 0;
    if (!cursor_read(&c, &raw, width))
      ds41_die(c.error);
    if (width == 4 && arr.type == GGUF_VALUE_INT32 && (int32_t)raw < 0)
      ds41_die("negative metadata value");
    const uint64_t expected = i < count ? want[i] : 0;
    if (raw != expected) {
      fprintf(stderr,
              "ds41: %s[%" PRIu64 "]=%" PRIu64 ", expected %" PRIu64
              " for %s\n",
              key, i, raw, expected, DS41_MODEL_SHAPE_NAME);
      throw std::runtime_error(
          "incompatible or invalid DeepSeek V4.1 artifact");
    }
    if (out && i < count)
      out[i] = (uint32_t)raw;
  }
}

static void config_expect_f32_array(const ds41_model* m, const char* key,
                                    float want) {
  ds41_array_ref arr;
  if (!model_get_array(m, key, &arr) ||
      (arr.type != GGUF_VALUE_FLOAT32 && arr.type != GGUF_VALUE_FLOAT64) ||
      arr.len < DS41_N_LAYER) {
    fprintf(stderr, "ds41: float array metadata %s is missing or too short\n",
            key);
    throw std::runtime_error("incompatible or invalid DeepSeek V4.1 artifact");
  }
  ds41_cursor c = cursor_at(m, arr.data_pos);
  for (uint32_t i = 0; i < DS41_N_LAYER; i++) {
    float got = 0.0f;
    if (arr.type == GGUF_VALUE_FLOAT32) {
      if (!cursor_read(&c, &got, sizeof(got)))
        ds41_die(c.error);
    } else {
      double v = 0.0;
      if (!cursor_read(&c, &v, sizeof(v)))
        ds41_die(c.error);
      got = (float)v;
    }
    config_expect_f32(key, got, want);
  }
}

/* Validate every metadata value that affects V4.1 semantics. */
static void config_validate_model(const ds41_model* m) {
  if (required_string(m, "general.architecture") != "deepseek41") {
    ds41_die("model architecture is not deepseek41");
  }
  struct U32Key {
    const char* key;
    uint32_t want;
  };
  static constexpr U32Key kU32[] = {
      {"deepseek41.block_count", DS41_N_LAYER},
      {"deepseek41.embedding_length", DS41_N_EMBD},
      {"deepseek41.attention.head_count", DS41_N_HEAD},
      {"deepseek41.attention.head_count_kv", DS41_N_HEAD_KV},
      {"deepseek41.attention.key_length", DS41_N_HEAD_DIM},
      {"deepseek41.attention.value_length", DS41_N_VALUE_DIM},
      {"deepseek41.rope.dimension_count", DS41_N_ROT},
      {"deepseek41.attention.q_lora_rank", DS41_N_LORA_Q},
      {"deepseek41.attention.output_lora_rank", DS41_N_LORA_O},
      {"deepseek41.attention.output_group_count", DS41_N_OUT_GROUP},
      {"deepseek41.attention.sliding_window", DS41_N_SWA},
      {"deepseek41.expert_count", DS41_N_EXPERT},
      {"deepseek41.expert_used_count", DS41_N_EXPERT_USED},
      {"deepseek41.expert_feed_forward_length", DS41_N_FF_EXP},
      {"deepseek41.expert_shared_count", DS41_N_EXPERT_SHARED},
      {"deepseek41.hash_layer_count", 0},
      {"deepseek41.attention.indexer.head_count", DS41_N_INDEXER_HEAD},
      {"deepseek41.attention.indexer.key_length", DS41_N_INDEXER_HEAD_DIM},
      {"deepseek41.attention.indexer.top_k", DS41_N_INDEXER_TOP_K},
      {"deepseek41.attention.indexer.candidate_block_size",
       DS41_CANDIDATE_BLOCK_SIZE},
      {"deepseek41.attention.indexer.candidate_top_k_blocks",
       DS41_CANDIDATE_TOPK_BLOCKS},
      {"deepseek41.attention.indexer.candidate_source_layer",
       DS41_CANDIDATE_SOURCE_LAYER},
      {"deepseek41.hyper_connection.count", DS41_N_HC},
      {"deepseek41.hyper_connection.sinkhorn_iterations",
       DS41_N_HC_SINKHORN_ITER},
      {"deepseek41.embedding_length_per_layer_input", DS41_ENGRAM_HEAD_DIM},
      {"deepseek41.ple.heads_per_ngram", DS41_ENGRAM_HEADS},
      {"deepseek41.ple.ngram_size", DS41_ENGRAM_MAX_NGRAM},
      {"deepseek41.engram.pad_token_id", DS41_ENGRAM_PAD_TOKEN},
      /* sqrtsoftplus scoring (llama.cpp's LLAMA_EXPERT_GATING_FUNC_TYPE). */
      {"deepseek41.expert_gating_func", 4},
  };
  for (const U32Key& k : kU32)
    config_expect_u32(k.key, required_u32(m, k.key), k.want);
  uint32_t groups = 0;
  model_get_u32(m, "deepseek41.expert_group_count", &groups);
  config_expect_u32("expert_group_count", groups, 0);
  config_expect_bool("expert_weights_norm",
                     required_bool(m, "deepseek41.expert_weights_norm"), true);

  uint64_t ratios[DS41_N_LAYER];
  for (uint32_t il = 0; il < DS41_N_LAYER; il++)
    ratios[il] = ds41_expected_layer_compress_ratio(il);
  config_expect_int_array(m, "deepseek41.attention.compress_ratios", ratios,
                          DS41_N_LAYER, true, g_ds41_compress_ratios);
  uint64_t list[DS41_N_INDEX_SOURCE];
  for (uint32_t i = 0; i < DS41_N_KV_SOURCE; i++)
    list[i] = DS41_KV_SOURCE_LAYERS[i];
  config_expect_int_array(m, "deepseek41.attention.kv_source_layers", list,
                          DS41_N_KV_SOURCE, false);
  for (uint32_t i = 0; i < DS41_N_INDEX_SOURCE; i++)
    list[i] = DS41_INDEX_SOURCE_LAYERS[i];
  config_expect_int_array(m, "deepseek41.attention.indexer.source_layers",
                          list, DS41_N_INDEX_SOURCE, false);
  for (uint32_t i = 0; i < DS41_N_ENGRAM_LAYER; i++)
    list[i] = DS41_ENGRAM_LAYERS[i];
  config_expect_int_array(m, "deepseek41.ple.layers", list,
                          DS41_N_ENGRAM_LAYER, false);
  config_expect_f32_array(m, "deepseek41.swiglu_clamp_exp",
                          DS41_SWIGLU_CLAMP_EXP);
  config_expect_f32_array(m, "deepseek41.swiglu_clamp_shexp",
                          DS41_SWIGLU_CLAMP_EXP);

  uint64_t rope_orig_ctx = 0;
  model_get_u64(m, "deepseek41.rope.scaling.original_context_length",
                &rope_orig_ctx);
  if (rope_orig_ctx != DS41_ROPE_ORIG_CTX)
    ds41_die("unexpected rope.scaling.original_context_length");
  config_expect_f32("rope.freq_base",
                    required_f32(m, "deepseek41.rope.freq_base"),
                    DS41_ROPE_FREQ_BASE);
  config_expect_f32("rope.scaling.factor",
                    required_f32(m, "deepseek41.rope.scaling.factor"),
                    DS41_ROPE_SCALE_FACTOR);
  config_expect_f32("rope.scaling.yarn_beta_fast",
                    required_f32(m, "deepseek41.rope.scaling.yarn_beta_fast"),
                    DS41_ROPE_YARN_BETA_FAST);
  config_expect_f32("rope.scaling.yarn_beta_slow",
                    required_f32(m, "deepseek41.rope.scaling.yarn_beta_slow"),
                    DS41_ROPE_YARN_BETA_SLOW);
  config_expect_f32(
      "attention.compress_rope_freq_base",
      required_f32(m, "deepseek41.attention.compress_rope_freq_base"),
      DS41_COMPRESS_ROPE_FREQ_BASE);
  config_expect_f32("expert_weights_scale",
                    required_f32(m, "deepseek41.expert_weights_scale"),
                    DS41_EXPERT_WEIGHT_SCALE);
  config_expect_f32(
      "attention.layer_norm_rms_epsilon",
      required_f32(m, "deepseek41.attention.layer_norm_rms_epsilon"),
      DS41_RMS_EPS);
  config_expect_f32("hyper_connection.epsilon",
                    required_f32(m, "deepseek41.hyper_connection.epsilon"),
                    DS41_HC_EPS);
}

/* Read an integer array metadata value of exactly `count` entries. */
static void required_int_array(const ds41_model* m, const char* key,
                               int64_t* out, uint64_t count) {
  ds41_array_ref arr;
  const bool found = model_get_array(m, key, &arr);
  const uint64_t width = found ? scalar_value_size(arr.type) : 0;
  if (width == 0 || arr.type == GGUF_VALUE_FLOAT32 ||
      arr.type == GGUF_VALUE_FLOAT64 || arr.len != count) {
    fprintf(
        stderr,
        "ds41: integer array metadata %s is missing or has the wrong length\n",
        key);
    throw std::runtime_error("incompatible or invalid DeepSeek V4.1 artifact");
  }
  ds41_cursor c = cursor_at(m, arr.data_pos);
  for (uint64_t i = 0; i < count; i++) {
    uint64_t raw = 0;
    if (!cursor_read(&c, &raw, width))
      ds41_die(c.error);
    out[i] = arr.type == GGUF_VALUE_INT32 ? (int64_t)(int32_t)raw : (int64_t)raw;
  }
}

/* The Engram n-gram hash constants: token map, per-head bucket counts and
 * absolute table offsets, and the per-layer odd multipliers. */
static void engram_constants_load(ds41_weights* w, const ds41_model* m) {
  static_assert(DS41_N_ENGRAM_LAYER * DS41_ENGRAM_COLS >=
                DS41_N_ENGRAM_LAYER * DS41_ENGRAM_MAX_NGRAM);
  std::vector<int64_t> values(DS41_N_VOCAB);
  required_int_array(m, "deepseek41.engram.token_map", values.data(),
                     DS41_N_VOCAB);
  int64_t top = -1;
  for (uint32_t i = 0; i < DS41_N_VOCAB; i++) {
    if (values[i] < 0 || values[i] >= DS41_ENGRAM_COMPRESSED_VOCAB)
      ds41_die("deepseek41.engram.token_map is outside the compressed vocabulary");
    w->engram_token_map[i] = (int32_t)values[i];
    top = std::max(top, values[i]);
  }
  if (top + 1 != DS41_ENGRAM_COMPRESSED_VOCAB)
    ds41_die("deepseek41.engram.token_map does not cover the compressed vocabulary");

  constexpr uint32_t heads = DS41_N_ENGRAM_LAYER * DS41_ENGRAM_COLS;
  required_int_array(m, "deepseek41.ple.head_vocab_sizes", w->engram_head_vocab,
                     heads);
  required_int_array(m, "deepseek41.ple.head_offsets", w->engram_head_offsets,
                     heads);
  int64_t row = 0;
  for (uint32_t i = 0; i < heads; i++) {
    if (w->engram_head_vocab[i] <= 0 || w->engram_head_offsets[i] != row)
      ds41_die("deepseek41.ple.head_offsets do not follow head_vocab_sizes");
    row += w->engram_head_vocab[i];
  }
  required_int_array(m, "deepseek41.ple.layer_multipliers",
                     w->engram_multipliers,
                     DS41_N_ENGRAM_LAYER * DS41_ENGRAM_MAX_NGRAM);
  for (int64_t multiplier : w->engram_multipliers)
    if (multiplier <= 0 || multiplier % 2 == 0)
      ds41_die("deepseek41.ple.layer_multipliers must be odd and positive");
}

/* Bind tensor names once into the fixed V4.1 layer layout. */
static void weights_bind(ds41_weights* w, const ds41_model* m) {
  memset(w, 0, sizeof(*w));
  w->token_embd = required_tensor(m, "token_embd.weight");
  w->output_norm = required_tensor(m, "output_norm.weight");
  w->output = required_tensor(m, "output.weight");

  for (uint32_t il = 0; il < DS41_N_LAYER; il++) {
    ds41_layer_weights* l = &w->layer[il];
    l->hc_attn_fn = required_tensorf(m, "blk.%u.hc_attn_fn.weight", il);
    l->hc_attn_scale = required_tensorf(m, "blk.%u.hc_attn_scale.weight", il);
    l->hc_attn_base = required_tensorf(m, "blk.%u.hc_attn_base.weight", il);
    l->attn_norm = required_tensorf(m, "blk.%u.attn_norm.weight", il);
    l->attn_q_a = required_tensorf(m, "blk.%u.attn_q_a.weight", il);
    l->attn_q_a_norm = required_tensorf(m, "blk.%u.attn_q_a_norm.weight", il);
    l->attn_q_b = required_tensorf(m, "blk.%u.attn_q_b.weight", il);
    l->attn_kv = required_tensorf(m, "blk.%u.attn_kv.weight", il);
    l->attn_kv_a_norm = required_tensorf(m, "blk.%u.attn_kv_a_norm.weight", il);
    l->attn_sinks = required_tensorf(m, "blk.%u.attn_sinks.weight", il);
    l->attn_output_a = required_tensorf(m, "blk.%u.attn_output_a.weight", il);
    l->attn_output_b = required_tensorf(m, "blk.%u.attn_output_b.weight", il);
    if (ds41_layer_is_kv_source(il)) {
      l->attn_compressor_kv =
          required_tensorf(m, "blk.%u.attn_compressor_kv.weight", il);
      l->attn_compressor_norm =
          required_tensorf(m, "blk.%u.attn_compressor_norm.weight", il);
      l->attn_compressor_gate =
          tensor_by_namef(m, "blk.%u.attn_compressor_gate.weight", il);
      l->indexer_attn_k =
          required_tensorf(m, "blk.%u.indexer.attn_k.weight", il);
      l->indexer_k_norm =
          required_tensorf(m, "blk.%u.indexer.k_norm.weight", il);
    }
    if (ds41_layer_is_index_source(il)) {
      l->indexer_attn_q_b =
          required_tensorf(m, "blk.%u.indexer.attn_q_b.weight", il);
      l->indexer_proj = required_tensorf(m, "blk.%u.indexer.proj.weight", il);
    }
    if (ds41_layer_engram_slot(il) >= 0) {
      l->engram_gate = required_tensorf(m, "blk.%u.engram_gate.weight", il);
      l->engram_kv = required_tensorf(m, "blk.%u.engram_kv.weight", il);
    }
    l->hc_ffn_fn = required_tensorf(m, "blk.%u.hc_ffn_fn.weight", il);
    l->hc_ffn_scale = required_tensorf(m, "blk.%u.hc_ffn_scale.weight", il);
    l->hc_ffn_base = required_tensorf(m, "blk.%u.hc_ffn_base.weight", il);
    l->ffn_norm = required_tensorf(m, "blk.%u.ffn_norm.weight", il);
    l->ffn_gate_inp = required_tensorf(m, "blk.%u.ffn_gate_inp.weight", il);
    l->ffn_exp_probs_b = required_tensorf(m, "blk.%u.exp_probs_b.bias", il);
    l->ffn_exp_probs_b_vl =
        required_tensorf(m, "blk.%u.exp_probs_b_vl.bias", il);
    l->ffn_gate_exps = required_tensorf(m, "blk.%u.ffn_gate_exps.weight", il);
    l->ffn_up_exps = required_tensorf(m, "blk.%u.ffn_up_exps.weight", il);
    l->ffn_down_exps = required_tensorf(m, "blk.%u.ffn_down_exps.weight", il);
    l->ffn_gate_shexp = required_tensorf(m, "blk.%u.ffn_gate_shexp.weight", il);
    l->ffn_up_shexp = required_tensorf(m, "blk.%u.ffn_up_shexp.weight", il);
    l->ffn_down_shexp = required_tensorf(m, "blk.%u.ffn_down_shexp.weight", il);
  }
  weights_validate_layout(w);
  engram_constants_load(w, m);
}

static void weights_free(ds41_weights* w) {
  if (w)
    memset(w, 0, sizeof(*w));
}

/* =========================================================================
 * DSpark Drafter.
 * =========================================================================
 *
 * Two GGUF layouts of the official mtp.* weights: "dflash" (block tensors
 * blk.N.*, fc, enc.output_norm, markov_w1/w2) and "deepseek41-dflash-draft"
 * (dflash.fc, dflash.hidden_norm, dflash.dspark.markov.w1/w2). Dense
 * projections may be any type the GEMV paths take; F16 tensors are converted
 * to F32 at upload.
 */

static void dspark_bind(ds41_dspark_model* d, ds41_dspark_tensor* dst,
                        const ds41_tensor* t, uint32_t k, uint64_t rows) {
  if (t->dim[0] != k || t->elements != (uint64_t)k * rows) {
    fprintf(stderr, "ds41: DSpark tensor %.*s has an unexpected shape\n",
            (int)t->name.len, t->name.ptr);
    throw std::runtime_error("incompatible DSpark drafter");
  }
  dst->k = k;
  dst->rows = rows;
  dst->type = t->type;
  d->sources.push_back({dst, t});
}

static void dspark_open(ds41_dspark_model* d, const char* path) {
  ds41_model* m = &d->file;
  model_open(m, model_shard_paths(path));
  const std::string arch = required_string(m, "general.architecture");
  const bool lucebox = arch == "deepseek41-dflash-draft";
  if (arch != "dflash" && !lucebox)
    ds41_die("DSpark drafter architecture is not dflash");
  const auto key = [&](const char* name) { return arch + "." + name; };
  config_expect_u32("dspark block_count",
                    required_u32(m, key("block_count").c_str()),
                    DS41_DSPARK_STAGES);
  config_expect_u32("dspark embedding_length",
                    required_u32(m, key("embedding_length").c_str()),
                    DS41_N_EMBD);
  config_expect_u32("dspark expert_count",
                    required_u32(m, key("expert_count").c_str()),
                    DS41_DSPARK_EXPERTS);
  config_expect_u32("dspark expert_used_count",
                    required_u32(m, key("expert_used_count").c_str()),
                    DS41_DSPARK_EXPERTS_USED);
  config_expect_u32("dspark sliding_window",
                    required_u32(m, key("attention.sliding_window").c_str()),
                    DS41_N_SWA);
  config_expect_f32("dspark rope.freq_base",
                    required_f32(m, key("rope.freq_base").c_str()),
                    DS41_ROPE_FREQ_BASE);
  config_expect_f32("dspark expert_weights_scale",
                    required_f32(m, key("expert_weights_scale").c_str()),
                    DS41_EXPERT_WEIGHT_SCALE);
  const std::string block_key =
      lucebox ? key("dflash.block_size") : key("block_size");
  config_expect_u32("dspark block_size", required_u32(m, block_key.c_str()),
                    DS41_DSPARK_BLOCK);
  const uint64_t targets[DS41_DSPARK_TARGET_LAYERS] = {
      DS41_DSPARK_TARGET_LAYER[0], DS41_DSPARK_TARGET_LAYER[1],
      DS41_DSPARK_TARGET_LAYER[2]};
  config_expect_int_array(
      m,
      (lucebox ? key("dflash.target_layer_ids") : key("target_layers")).c_str(),
      targets, DS41_DSPARK_TARGET_LAYERS, false);
  uint32_t mask = 0;
  if (model_get_u32(m, key("dflash.mask_token_id").c_str(), &mask))
    config_expect_u32("dspark mask_token_id", mask, DS41_DSPARK_NOISE_TOKEN);

  const uint32_t dim = DS41_N_EMBD;
  const uint32_t q_dim = DS41_N_HEAD * DS41_N_HEAD_DIM;
  const uint32_t mixes = 2 * DS41_N_HC + DS41_N_HC * DS41_N_HC;
  for (uint32_t s = 0; s < DS41_DSPARK_STAGES; s++) {
    ds41_dspark_stage* st = &d->stage[s];
    const auto bind = [&](ds41_dspark_tensor* dst, const char* fmt, uint32_t k,
                          uint64_t rows) {
      dspark_bind(d, dst, required_tensorf(m, fmt, s), k, rows);
    };
    bind(&st->hc_attn_fn, "blk.%u.hc_attn_fn.weight", DS41_N_HC * dim, mixes);
    bind(&st->hc_attn_scale, "blk.%u.hc_attn_scale.weight", 3, 1);
    bind(&st->hc_attn_base, "blk.%u.hc_attn_base.weight", mixes, 1);
    bind(&st->attn_norm, "blk.%u.attn_norm.weight", dim, 1);
    bind(&st->q_a, "blk.%u.attn_q_a.weight", dim, DS41_N_LORA_Q);
    bind(&st->q_a_norm, "blk.%u.attn_q_a_norm.weight", DS41_N_LORA_Q, 1);
    bind(&st->q_b, "blk.%u.attn_q_b.weight", DS41_N_LORA_Q, q_dim);
    bind(&st->kv, "blk.%u.attn_kv.weight", dim, DS41_N_HEAD_DIM);
    bind(&st->kv_a_norm, "blk.%u.attn_kv_a_norm.weight", DS41_N_HEAD_DIM, 1);
    bind(&st->sinks, "blk.%u.attn_sinks.weight", DS41_N_HEAD, 1);
    bind(&st->o_a, "blk.%u.attn_output_a.weight", q_dim / DS41_N_OUT_GROUP,
         (uint64_t)DS41_N_OUT_GROUP * DS41_N_LORA_O);
    bind(&st->o_b, "blk.%u.attn_output_b.weight",
         DS41_N_OUT_GROUP * DS41_N_LORA_O, dim);
    bind(&st->hc_ffn_fn, "blk.%u.hc_ffn_fn.weight", DS41_N_HC * dim, mixes);
    bind(&st->hc_ffn_scale, "blk.%u.hc_ffn_scale.weight", 3, 1);
    bind(&st->hc_ffn_base, "blk.%u.hc_ffn_base.weight", mixes, 1);
    bind(&st->ffn_norm, "blk.%u.ffn_norm.weight", dim, 1);
    bind(&st->gate_inp, "blk.%u.ffn_gate_inp.weight", dim, DS41_DSPARK_EXPERTS);
    bind(&st->probs_b, "blk.%u.exp_probs_b.bias", DS41_DSPARK_EXPERTS, 1);
    bind(&st->gate_shexp, "blk.%u.ffn_gate_shexp.weight", dim, DS41_N_FF_EXP);
    bind(&st->up_shexp, "blk.%u.ffn_up_shexp.weight", dim, DS41_N_FF_EXP);
    bind(&st->down_shexp, "blk.%u.ffn_down_shexp.weight", DS41_N_FF_EXP, dim);
    bind(&st->gate_exps, "blk.%u.ffn_gate_exps.weight", dim,
         (uint64_t)DS41_N_FF_EXP * DS41_DSPARK_EXPERTS);
    bind(&st->up_exps, "blk.%u.ffn_up_exps.weight", dim,
         (uint64_t)DS41_N_FF_EXP * DS41_DSPARK_EXPERTS);
    bind(&st->down_exps, "blk.%u.ffn_down_exps.weight", DS41_N_FF_EXP,
         (uint64_t)dim * DS41_DSPARK_EXPERTS);
  }
  const auto bind = [&](ds41_dspark_tensor* dst, const char* name, uint32_t k,
                        uint64_t rows) {
    dspark_bind(d, dst, required_tensor(m, name), k, rows);
  };
  bind(&d->fc, lucebox ? "dflash.fc.weight" : "fc.weight",
       DS41_DSPARK_TARGET_LAYERS * dim, dim);
  bind(&d->main_norm,
       lucebox ? "dflash.hidden_norm.weight" : "enc.output_norm.weight", dim,
       1);
  bind(&d->out_norm, "output_norm.weight", dim, 1);
  bind(&d->markov_w1, lucebox ? "dflash.dspark.markov.w1" : "markov_w1.weight",
       DS41_DSPARK_MARKOV_RANK, DS41_N_VOCAB);
  bind(&d->markov_w2, lucebox ? "dflash.dspark.markov.w2" : "markov_w2.weight",
       DS41_DSPARK_MARKOV_RANK, DS41_N_VOCAB);
  // The acceptance head is optional: hidden then Markov-rank inputs.
  if (const ds41_tensor* w = model_find_tensor(
          m, lucebox ? "dflash.dspark.confidence.weight"
                     : "confidence_head.proj.weight")) {
    dspark_bind(d, &d->confidence, w, dim + DS41_DSPARK_MARKOV_RANK, 1);
    if (const ds41_tensor* b =
            model_find_tensor(m, "dflash.dspark.confidence.bias"))
      dspark_bind(d, &d->confidence_bias, b, 1, 1);
  }
}

/* Free CMA pages count as available memory but only hold movable pages, so
 * the resident weights and every later GPU allocation must fit beside them.
 * The kernel's KHO scratch area (Ubuntu 26.04) can make that several GiB. */
static void warn_free_cma(void) {
  FILE* f = fopen("/proc/meminfo", "r");
  if (!f) return;
  char line[256];
  unsigned long long cma_kib = 0;
  while (fgets(line, sizeof(line), f))
    if (sscanf(line, "CmaFree: %llu kB", &cma_kib) == 1) break;
  fclose(f);
  if (cma_kib < (1ull << 20)) return;
  gufo::server::Logger::LogFormatted(
      gufo::server::LogLevel::kWarn, "ds41",
      "%.2f GiB of the available memory is free CMA (CmaFree in /proc/meminfo), which GPU memory cannot use; "
      "the resident model and its sessions need that memory. If it is the kernel's KHO scratch area "
      "(Ubuntu 26.04), boot with kho=off",
      (double)cma_kib / 1048576.0);
}

/* =========================================================================
 * Engram Tables.
 * =========================================================================
 *
 * The two n-gram tables (about 110 GB at IQ4_NL) never become resident. They
 * stay mapped read-only and rows are read on demand; the kernel is told the
 * access pattern is random so read-ahead does not fill memory.
 */

static void engram_open(ds41_engram_tables* e, const ds41_weights* w,
                        const ds41_model* backbone, const std::string& path) {
  model_open(&e->file, {path});
  const ds41_model* m = &e->file;
  /* The Engram file has no architecture of its own; it belongs to the build
   * through the split bookkeeping and the bucket counts of the hash. */
  if (m->n_tensors != 1 ||
      required_u32(m, "split.tensors.count") !=
          required_u32(backbone, "split.tensors.count")) {
    fprintf(stderr, "ds41: %s is not the Engram shard of this build\n",
            path.c_str());
    throw std::runtime_error("incompatible or invalid DeepSeek V4.1 artifact");
  }
  int64_t rows = 0;
  for (int64_t n : w->engram_head_vocab)
    rows += n;
  ds41_tensor* t = required_tensor(m, "per_layer_token_embd.weight");
  tensor_expect_dims(t, 2, DS41_ENGRAM_HEAD_DIM, (uint64_t)rows, 0);
  if (t->type != DS41_TENSOR_IQ4_NL && t->type != DS41_TENSOR_Q8_0)
    tensor_reject(t, "has an unsupported Engram row type");
  e->table = t;
  e->row_bytes = t->bytes / (uint64_t)rows;
  for (uint32_t i = 0; i < m->n_shards; i++) {
    (void)posix_fadvise(m->shard_fd[i], 0, 0, POSIX_FADV_RANDOM);
    (void)madvise((void*)(m->map + m->shard_offset[i]),
                  (size_t)m->shard_size[i], MADV_RANDOM);
  }
}

/* Builds published before the llama.cpp layout name the Engram file in
 * their metadata and use the old key names. */
static void require_current_layout(const ds41_model* m, const char* path) {
  if (model_find_kv(m, "deepseek41.engram.file") == nullptr &&
      model_find_kv(m, "deepseek41.attention.kv_source_layers") != nullptr)
    return;
  fprintf(stderr,
          "ds41: %s uses the GGUF layout ISTA-DASLab published before "
          "2026-10-27. Convert it with convert_ds41_upstream.py from the model "
          "repository (README_DEEPSEEK.md)\n",
          path);
  throw std::runtime_error("incompatible or invalid DeepSeek V4.1 artifact");
}

static void engram_close(ds41_engram_tables* e) {
  if (!e)
    return;
  model_close(&e->file);
  free(e);
}

/* Bytes of the tensors uploaded to the device; nothing in the Engram file is
 * resident. */
static uint64_t model_resident_bytes(const ds41_model* m, const ds41_weights* weights = nullptr,
                                     bool streaming = false) {
  uint64_t bytes = 0;
  for (uint64_t i = 0; i < m->n_tensors; i++)
    if (tensor_device_resident(&m->tensors[i], weights, streaming))
      bytes += m->tensors[i].bytes;
  return bytes;
}

/* Match ds41_dspark_upload's widened F16 allocation for the placement plan. */
static uint64_t dspark_resident_bytes(const ds41_dspark_model* dspark) {
  if (!dspark) return 0;
  uint64_t total = 0;
  for (const ds41_dspark_source& source : dspark->sources) {
    const bool widen = source.src->type == DS41_TENSOR_F16 && source.dst != &dspark->markov_w1 &&
                       source.dst != &dspark->markov_w2;
    const uint64_t bytes = widen ? source.src->elements * sizeof(float) : source.src->bytes;
    total += (bytes + 255) / 256 * 256;
  }
  return total;
}

static void ds41_release_instance_lock(void) {
  if (g_ds41_lock_fd >= 0) {
    close(g_ds41_lock_fd);
    g_ds41_lock_fd = -1;
  }
}

/* Refuse to start a second ds4 process.  The model can map tens of GiB, so a
 * stale accidental second run is more dangerous than a normal CLI error. */
static void ds41_acquire_instance_lock(void) {
  const char* path = getenv("DS41_LOCK_FILE");
  if (!path || !path[0])
    path = "/tmp/ds4.lock";

  const int fd = open(path, O_RDWR | O_CREAT, 0600);
  if (fd < 0) {
    fprintf(stderr, "ds41: failed to open lock file %s: %s\n", path,
            strerror(errno));
    throw std::runtime_error("DeepSeek instance lock unavailable");
  }
  (void)fcntl(fd, F_SETFD, FD_CLOEXEC);

  if (flock(fd, LOCK_EX | LOCK_NB) != 0) {
    if (errno == EWOULDBLOCK) {
      char buf[64];
      const ssize_t n = pread(fd, buf, sizeof(buf) - 1, 0);
      long owner = -1;
      if (n > 0) {
        buf[n] = '\0';
        char* end = NULL;
        owner = strtol(buf, &end, 10);
      }
      if (owner > 0) {
        fprintf(stderr,
                "ds41: another ds4 process is already running (pid %ld); "
                "refusing to start\n",
                owner);
      } else {
        fprintf(stderr,
                "ds41: another ds4 process is already running; refusing to "
                "start\n");
      }
      close(fd);
      throw std::runtime_error("DeepSeek instance lock unavailable");
    }
    fprintf(stderr, "ds41: failed to lock %s: %s\n", path, strerror(errno));
    close(fd);
    throw std::runtime_error("DeepSeek instance lock unavailable");
  }

  if (ftruncate(fd, 0) != 0) {
    fprintf(stderr, "ds41: failed to truncate lock file %s: %s\n", path,
            strerror(errno));
    close(fd);
    throw std::runtime_error("DeepSeek instance lock unavailable");
  }
  dprintf(fd, "%ld\n", (long)getpid());
  g_ds41_lock_fd = fd;
}

int ds41_engine_open(ds41_engine** out, const ds41_engine_options* opt) {
  if (out)
    *out = nullptr;
  if (!out || !opt || !opt->model_path || !opt->model_path[0])
    return 1;
  if (opt->prefill_batch != 0 &&
      (opt->prefill_batch < DS41_MIN_PREFILL_BATCH ||
       opt->prefill_batch > DS41_MAX_PREFILL_BATCH)) {
    fprintf(stderr, "ds41: prefill batch %u is outside %u..%u\n",
            opt->prefill_batch, DS41_MIN_PREFILL_BATCH,
            DS41_MAX_PREFILL_BATCH);
    return 1;
  }
  if (opt->expert_cache_bytes && !opt->stream_experts) {
    fprintf(stderr, "ds41: an expert cache size needs expert streaming\n");
    return 1;
  }
  static std::mutex load_mutex;
  const std::lock_guard lock(load_mutex);
  if (g_ds41_lock_fd >= 0) {
    fprintf(stderr, "ds41: a DeepSeek engine is already open\n");
    return 1;
  }
  ds41_engine* e = nullptr;
  try {
    ds41_acquire_instance_lock();

    e = static_cast<ds41_engine*>(ds41_xcalloc(1, sizeof(ds41_engine)));
    e->model = static_cast<ds41_model*>(ds41_xcalloc(1, sizeof(ds41_model)));
    e->model->fd = -1;
    e->prefill_batch = opt->prefill_batch ? opt->prefill_batch
                                          : DS41_DEFAULT_PREFILL_BATCH;
    e->exact_prefill = opt->exact_prefill;
    e->weights =
        static_cast<ds41_weights*>(ds41_xcalloc(1, sizeof(ds41_weights)));

    /* The last shard of a build is the Engram file all builds share. */
    std::vector<std::string> shards = model_shard_paths(opt->model_path);
    const std::string engram_path = shards.back();
    if (shards.size() > 1)
      shards.pop_back();
    model_open(e->model, shards);
    require_current_layout(e->model, opt->model_path);
    if (shards.size() == 1 && engram_path == shards[0])
      ds41_die("the model path must be the first shard of a split build");
    e->vocab = ds41_vocab_create(e->model);
    config_validate_model(e->model);
    weights_bind(e->weights, e->model);
    {
      e->engram = static_cast<ds41_engram_tables*>(
          ds41_xcalloc(1, sizeof(ds41_engram_tables)));
      engram_open(e->engram, e->weights, e->model, engram_path);
      gufo::server::Logger::LogFormatted(
          gufo::server::LogLevel::kInfo, "ds41",
          "%s: %u backbone shard(s), %.2f GiB device-eligible weights; Engram %s "
          "(%.2f GiB, %" PRIu64 " B rows, read from disk)",
          DS41_MODEL_SHAPE_NAME, e->model->n_shards,
          (double)model_resident_bytes(e->model) / 1073741824.0,
          engram_path.c_str(),
          (double)model_resident_bytes(&e->engram->file) / 1073741824.0,
          e->engram->row_bytes);
    }
    warn_free_cma();
    if (opt->dspark_model_path && opt->dspark_model_path[0]) {
      e->dspark = new ds41_dspark_model();
      dspark_open(e->dspark, opt->dspark_model_path);
    }
    if (opt->validate_only) {
      *out = e;
      return 0;
    }

    e->rocm_ready =
        true;  // Cleanup is needed once backend initialization starts.
    if (!ds41_gpu_init()) {
      fprintf(stderr, "ds41: ROCm backend unavailable; aborting startup\n");
      ds41_engine_close(e);
      *out = NULL;
      return 1;
    }

    /* Streaming is opt-in: a silent fallback would turn a misconfigured host
     * (a small GTT limit, free CMA) into a much slower server. */
    const bool streaming = opt->stream_experts;
    if (!narrow_f32_routers(e->weights, e->model)) {
      fprintf(stderr, "ds41: failed to upload the BF16 routers\n");
      ds41_engine_close(e);
      *out = NULL;
      return 1;
    }
    const uint64_t available = ds41_expert_available_memory();
    const uint64_t resident =
        model_resident_bytes(e->model, e->weights, streaming) +
        (e->weights->router_bf16 ? (uint64_t)DS41_N_LAYER * kRouterElements * sizeof(uint16_t) : 0);
    const uint64_t needed = resident + dspark_resident_bytes(e->dspark) + (2ull << 30);
    if (!streaming && available != 0 && needed > available) {
      char drafter[48] = "";
      if (e->dspark)
        snprintf(drafter, sizeof(drafter), ", %.2f GiB drafter", (double)dspark_resident_bytes(e->dspark) / 1073741824.0);
      fprintf(stderr,
              "ds41: the model needs %.2f GiB (%.2f GiB weights%s, 2 GiB for sessions) but %.2f GiB of memory "
              "is available, excluding free CMA. Free memory, check the GTT limit and CMA (README_DEEPSEEK.md, "
              "host setup), or pass --stream-experts to stream routed experts from the SSD, more slowly\n",
              (double)needed / 1073741824.0, (double)resident / 1073741824.0, drafter,
              (double)available / 1073741824.0);
      ds41_engine_close(e);
      *out = NULL;
      return 1;
    }
    if (streaming) {
      gufo::server::Logger::LogFormatted(
          gufo::server::LogLevel::kInfo, "ds41",
          "streaming routed experts: %.2f GiB dense weights, %.2f GiB available unified memory",
          (double)resident / 1073741824.0, (double)available / 1073741824.0);
    }
    ds41_gpu_set_resident_weight_bytes(resident);
    (void)ds41_gpu_set_model_shards(e->model->shard_fd, e->model->shard_offset,
                                    e->model->shard_size, e->model->n_shards);
    if (!ds41_gpu_set_model_map_range(
            e->model->map, e->model->size, e->model->tensor_data_pos,
            e->model->size - e->model->tensor_data_pos,
            e->model->max_tensor_bytes)) {
      fprintf(stderr,
              "ds41: ROCm failed to map model views; aborting startup. "
              "This is commonly caused by insufficient memory or accelerator "
              "VM budget.\n");
      ds41_engine_close(e);
      *out = NULL;
      return 1;
    }
    if (!accelerator_cache_model_tensors(e->model, e->weights, streaming)) {
      fprintf(stderr, "ds41: ROCm failed to prepare startup model cache\n");
      ds41_engine_close(e);
      *out = NULL;
      return 1;
    }
    if (e->dspark) {
      if (!ds41_dspark_upload(e->dspark)) {
        fprintf(stderr, "ds41: DSpark drafter upload failed\n");
        ds41_engine_close(e);
        *out = NULL;
        return 1;
      }
      gufo::server::Logger::LogFormatted(
          gufo::server::LogLevel::kInfo, "ds41",
          "DSpark drafter %s: %.2f GiB resident", opt->dspark_model_path,
          (double)e->dspark->device_bytes / 1073741824.0);
    }

    if (streaming) {
      e->expert_store = ds41_expert_store_create(e->model, e->weights, opt->expert_cache_bytes);
      if (!e->expert_store) {
        fprintf(stderr, "ds41: failed to initialize the streamed expert store\n");
        ds41_engine_close(e);
        *out = nullptr;
        return 1;
      }
      const std::string histogram = std::string(opt->model_path) + ".expert-histogram";
      const bool custom = access(histogram.c_str(), F_OK) == 0;
      if (custom && !ds41_expert_store_seed(e->expert_store, histogram.c_str())) {
        fprintf(stderr, "ds41: failed to load expert histogram %s; using the built-in ranking\n", histogram.c_str());
        if (!ds41_expert_store_seed_default(e->expert_store))
          fprintf(stderr, "ds41: failed to warm the expert cache\n");
      } else if (!custom && !ds41_expert_store_seed_default(e->expert_store)) {
        fprintf(stderr, "ds41: failed to warm the expert cache\n");
      }
    }

    *out = e;
    return 0;
  } catch (const std::exception& error) {
    fprintf(stderr, "ds41: model load failed: %s\n", error.what());
    ds41_engine_close(e);
    if (!e)
      ds41_release_instance_lock();
    return 1;
  }
}

bool ds41_engine_exact_prefill(const ds41_engine* e) {
    return e && e->exact_prefill;
}

int ds41_engine_enable_image_routing(ds41_engine* e) {
    if (!e || !e->rocm_ready)
        return 1;
    if (e->image_bias)
        return 0;
    std::vector<float> bias((uint64_t)DS41_N_LAYER * DS41_N_EXPERT);
    for (uint32_t il = 0; il < DS41_N_LAYER; il++)
        memcpy(bias.data() + (uint64_t)il * DS41_N_EXPERT,
               e->model->map + e->weights->layer[il].ffn_exp_probs_b_vl->abs_offset,
               DS41_N_EXPERT * sizeof(float));
    for (float value : bias)
        if (!std::isfinite(value))
            return 1;
    const uint64_t bytes = bias.size() * sizeof(float);
    void* p = NULL;
    if (hipMalloc(&p, bytes) != hipSuccess)
        return 1;
    if (hipMemcpy(p, bias.data(), bytes, hipMemcpyHostToDevice) != hipSuccess) {
        (void)hipFree(p);
        return 1;
    }
    e->image_bias = (float*)p;
    return 0;
}

bool ds41_engine_has_image_bias(const ds41_engine* e) {
    return e && e->image_bias;
}

bool ds41_engine_has_dspark(const ds41_engine* e) {
  return e && e->dspark && e->dspark->device;
}

int ds41_engine_vocab_size(const ds41_engine* e) {
  return e ? ds41_vocab_size(e->vocab) : 0;
}

const char* ds41_engine_model_name(const ds41_engine* e) {
  (void)e;
  return DS41_MODEL_SHAPE_NAME;
}

void ds41_engine_close(ds41_engine* e) {
  if (!e)
    return;
  if (e->image_bias)
    (void)hipFree(e->image_bias);
  if (e->weights && e->weights->router_bf16)
    (void)hipFree(e->weights->router_bf16);
  ds41_expert_store_log(e->expert_store);
  ds41_expert_store_destroy(e->expert_store);
  weights_free(e->weights);
  if (e->dspark) {
    ds41_dspark_free(e->dspark);
    model_close(&e->dspark->file);
    delete e->dspark;
    e->dspark = NULL;
  }
  engram_close(e->engram);
  e->engram = NULL;
  ds41_vocab_destroy(e->vocab);
  model_close(e->model);
  if (e->rocm_ready)
    ds41_gpu_cleanup();
  ds41_release_instance_lock();
  free(e->weights);
  free(e->model);
  free(e);
}
