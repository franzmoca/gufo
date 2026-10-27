#ifndef GUFO_MODELS_DEEPSEEK_V41_FLASH_RUNTIME_EXPERT_STORE_H_
#define GUFO_MODELS_DEEPSEEK_V41_FLASH_RUNTIME_EXPERT_STORE_H_

#include <cstdint>

#include <hip/hip_runtime_api.h>

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
  uint64_t prefetch_bytes = 0;         // staged by background prefetch jobs
  uint64_t prefetch_dropped_bytes = 0; // prefetched bytes no prepare consumed
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
  const int32_t* hit_ids;   // device rows; null unless the batch both hits and
                            // misses within the buffer
  const int32_t* miss_ids;  // as hit_ids
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

/* Queue a background read of the given uncached experts of `layer` into the
 * stage set the layer's next prepare consumes, without displacing anything.
 * The reads run only while no demand phase does; the next prepare for the
 * layer adopts what is still pending, and any other prepare drops it.
 * `wide` marks a prefill-scale expert set, which only a batch-scale prepare
 * adopts. Best effort: returns false when the layer is not the next
 * prepare's target or successor, has no uncached listed experts, or the job
 * was refused. */
bool ds41_expert_store_prefetch(ds41_expert_store* store, uint32_t layer,
                                const int32_t* experts, uint32_t count,
                                bool wide = false);

/* Records a router prediction whose ids `done` signals in decoder-owned
 * pinned memory; a store thread submits the prefetch when the event lands.
 * Returns false when the record was refused (best effort by design). */
bool ds41_expert_store_predict(ds41_expert_store* store, uint32_t layer,
                               hipEvent_t done, const int32_t* ids, uint32_t count,
                               bool wide);

/* Waits until every recorded prediction has been submitted, so the ids and
 * event may be released. */
void ds41_expert_store_predict_flush(ds41_expert_store* store);

/* True when no background prefetch work is queued or running. */
bool ds41_expert_store_prefetch_idle(const ds41_expert_store* store);
ds41_expert_store_stats ds41_expert_store_get_stats(
    const ds41_expert_store* store);
void ds41_expert_store_log(const ds41_expert_store* store);

#endif  // GUFO_MODELS_DEEPSEEK_V41_FLASH_RUNTIME_EXPERT_STORE_H_
