#ifndef GUFO_MODELS_DEEPSEEK_V41_FLASH_ROCM_RESIDENT_API_H_
#define GUFO_MODELS_DEEPSEEK_V41_FLASH_ROCM_RESIDENT_API_H_

#include <cstdint>

/* Device setup and resident weight upload for the V4.1 runtime. Kernels are
 * declared in ds41_v41.h. */

#ifdef __cplusplus
extern "C" {
#endif

/* Selects the device and initializes the vendored MMQ tier. */
int ds41_gpu_init(void);
/* Frees every uploaded weight range and the MMQ tier. */
void ds41_gpu_cleanup(void);
/* Bytes of weights the next load uploads to the device, which decide their
 * placement; 0 takes the whole registered mapping. Streamed experts and
 * disk-read tensors are not uploaded. */
void ds41_gpu_set_resident_weight_bytes(uint64_t bytes);
/* Registers the combined model mapping; weight ranges are uploaded later by
 * ds41_gpu_cache_model_range. */
int ds41_gpu_set_model_map_range(const void* model_map, uint64_t model_size,
                                 uint64_t map_offset, uint64_t map_size,
                                 uint64_t max_tensor_bytes);
/* The GGUF shards inside the combined mapping, read through their files. */
int ds41_gpu_set_model_shards(const int* fds, const uint64_t* offsets,
                              const uint64_t* sizes, uint32_t count);
/* Uploads a weight range unless it is already resident. */
int ds41_gpu_cache_model_range(const void* model_map, uint64_t model_size,
                               uint64_t offset, uint64_t bytes,
                               const char* label);
/* Waits for pending uploads and frees the staging buffers. */
int ds41_gpu_release_model_staging(void);
/* Device address of a model range, uploaded first if it is not resident yet;
 * NULL if the upload fails. */
const void* ds41_gpu_model_ptr(const void* model_map, uint64_t offset,
                               uint64_t bytes);

#ifdef __cplusplus
}
#endif

#endif  // GUFO_MODELS_DEEPSEEK_V41_FLASH_ROCM_RESIDENT_API_H_
