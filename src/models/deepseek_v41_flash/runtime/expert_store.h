#ifndef GUFO_MODELS_DEEPSEEK_V41_FLASH_RUNTIME_EXPERT_STORE_H_
#define GUFO_MODELS_DEEPSEEK_V41_FLASH_RUNTIME_EXPERT_STORE_H_

#include <cstdint>

struct ds41_expert_store;
struct ds41_model;
struct ds41_weights;

struct ds41_expert_store_stats {
  uint64_t uses = 0;
  uint64_t hits = 0;
  uint64_t bytes_read = 0;
  uint64_t tokens = 0;
  uint64_t stall_ns = 0;
  uint64_t cache_bytes = 0;
};

/* Available unified host memory, including reclaimable file cache. */
uint64_t ds41_expert_available_memory();

/* What ds41_expert_store_prepare stages for the layer, for work overlapped
 * with the projection reads. The id rows keep their route positions: hit_ids
 * holds the resident experts with misses as -1, miss_ids the read experts
 * with hits as -1, so a launch over one array zeroes exactly the rows the
 * other launch fills. table[p] holds each expert's device weight base (its
 * cache slot, or the stage row a read just filled), so kernels read cached
 * experts without a staging copy. */
struct ds41_expert_overlap {
  uint32_t layer;
  uint32_t rows;            // n_tokens * DS41_N_EXPERT_USED route rows
  uint32_t miss_rows;       // rows served by reads this call
  const int32_t* hit_ids;   // device rows; null when rows exceeds the buffer
  const int32_t* miss_ids;  // device rows; null when rows exceeds the buffer
  const void* table[3];     // device [DS41_N_EXPERT] weight bases
};

/* The model and its shard descriptors outlive the store. cache_limit_bytes
 * bounds the pool when nonzero; zero sizes it from available device memory. */
ds41_expert_store* ds41_expert_store_create(const ds41_model* model,
                                            const ds41_weights* weights,
                                            uint64_t cache_limit_bytes = 0);
void ds41_expert_store_destroy(ds41_expert_store* store);

/* Prime the cache from the built-in V4.1 ranking or an optional
 * DS41_EXPERT_HISTOGRAM_V1 text file. Priming is startup work and does not
 * count as inference in the statistics. */
bool ds41_expert_store_seed_default(ds41_expert_store* store);
bool ds41_expert_store_seed(ds41_expert_store* store, const char* path);

/* Routes are device [n_tokens][6]. The returned gate, up and down tables map
 * each expert index to its device weights and stay valid until the next call.
 * Engine evaluations are serialized, including joint verification. */
bool ds41_expert_store_prepare(ds41_expert_store* store, uint32_t layer,
                               const int32_t* routes, uint32_t n_tokens,
                               const void** gate, const void** up,
                               const void** down,
                               bool (*overlap)(void*, const ds41_expert_overlap*) = nullptr,
                               void* overlap_context = nullptr);
ds41_expert_store_stats ds41_expert_store_get_stats(
    const ds41_expert_store* store);
void ds41_expert_store_log(const ds41_expert_store* store);

#endif  // GUFO_MODELS_DEEPSEEK_V41_FLASH_RUNTIME_EXPERT_STORE_H_
