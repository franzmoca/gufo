/* Device setup and resident weight upload for the V4.1 runtime: weight spans
 * are allocated once (device heap or managed memory, see
 * gufo::hip::WeightAllocator) and filled from the GGUF shard files. */

#include <hip/hip_runtime.h>
#include <inttypes.h>
#include <unistd.h>

#include <cstdint>
#include <ctime>
#include <cstdio>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "mmq/ds41_mmq.h"
#include "resident_api.h"
#include "src/cli/serve/logging.hpp"
#include "src/core/hip/managed_alloc.hpp"
#include "src/core/hip/weight_upload.hpp"

#define DS41_GPU_LOG_PREFIX "ds41: ROCm "

namespace {

struct ModelShard {
  int fd;
  uint64_t offset;
  uint64_t size;
};

struct ModelRange {
  const void* host_base;
  uint64_t offset;
  uint64_t bytes;
  char* device_ptr;
};

const void* g_model_host_base;
uint64_t g_model_registered_size;
/* Weights this load uploads (ds41_gpu_set_resident_weight_bytes); 0 means
 * the whole registered mapping. */
uint64_t g_model_resident_weight_bytes;
/* Placement for this model load, decided once from the registered weight size
 * against the device heap. */
std::unique_ptr<gufo::hip::WeightAllocator> g_model_weight_alloc;
int g_model_fd = -1;
const void* g_model_fd_host_base;
std::vector<ModelShard> g_model_shards;
int g_model_cache_full;
std::unique_ptr<gufo::hip::WeightUpload> g_model_uploader;
std::vector<ModelRange> g_model_ranges;
std::vector<void*> g_model_allocations;
std::unordered_map<uint64_t, size_t> g_model_range_by_offset;
uint64_t g_model_range_bytes;
uint64_t g_model_load_progress_next;
double g_model_load_progress_last;
int g_model_load_progress_started;
int g_model_load_progress_tty;

int hip_ok(hipError_t err, const char* what) {
  if (err == hipSuccess)
    return 1;
  gufo::server::Logger::LogFormatted(gufo::server::LogLevel::kError, "ds4",
                                     "%s failed: %s", what,
                                     hipGetErrorString(err));
  return 0;
}

double hip_wall_sec() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + (double)ts.tv_nsec * 1.0e-9;
}

void hip_model_load_progress_reset() {
  g_model_load_progress_next = 0;
  g_model_load_progress_last = 0.0;
  g_model_load_progress_started = 0;
  g_model_load_progress_tty = 0;
}

void hip_model_load_progress_note(uint64_t cached_bytes) {
  const double now = hip_wall_sec();
  if (!g_model_load_progress_started) {
    g_model_load_progress_started = 1;
    g_model_load_progress_tty = isatty(STDERR_FILENO) != 0;
    g_model_load_progress_next = (g_model_load_progress_tty ? 2ull : 16ull) *
                                 1024ull * 1024ull * 1024ull;
    g_model_load_progress_last = now;
    gufo::server::Logger::LogFormatted(
        gufo::server::LogLevel::kInfo, "ds4",
        "loading model tensors into device cache");
  }

  if (cached_bytes < g_model_load_progress_next &&
      now - g_model_load_progress_last <
          (g_model_load_progress_tty ? 2.0 : 10.0)) {
    return;
  }

  // One milestone line per step instead of an in-place carriage return: the
  // shared logger owns the line format and applies the level threshold.
  gufo::server::Logger::LogFormatted(gufo::server::LogLevel::kInfo, "ds4",
                                     "loading model tensors %.2f GiB cached",
                                     (double)cached_bytes / 1073741824.0);
  g_model_load_progress_last = now;
  const uint64_t step =
      (g_model_load_progress_tty ? 2ull : 16ull) * 1024ull * 1024ull * 1024ull;
  while (g_model_load_progress_next <= cached_bytes) {
    g_model_load_progress_next += step;
  }
}

/* On a unified-memory machine the driver's device heap is a fraction of system
 * RAM (about 61 of 122 GiB on gfx1151), so a checkpoint whose weights exceed
 * it cannot be allocated device-only even though the system has the memory
 * free. Managed memory is ordinary system memory the GPU can also access, and
 * is not bounded by that heap.
 *
 * Decide once from the registered weight size against the heap, rather than
 * discovering the limit one failed span at a time: a per-span fallback drains
 * the heap to its last byte and leaves nothing for the session's device-only
 * scratch tensors. Oversized weights therefore go wholly to managed memory,
 * leaving the heap for session state.
 *
 * gufo::hip::WeightAllocator makes the decision and allocates coarse-grain
 * managed memory, so the GPU caches weight pages the way it caches hipMalloc
 * memory. */
uint64_t hip_model_weight_bytes() {
  return g_model_resident_weight_bytes ? g_model_resident_weight_bytes
                                       : g_model_registered_size;
}

gufo::hip::WeightAllocator* hip_model_weight_allocator() {
  if (!g_model_weight_alloc) {
    g_model_weight_alloc =
        std::make_unique<gufo::hip::WeightAllocator>(hip_model_weight_bytes());
    if (g_model_weight_alloc->placement() == gufo::hip::Placement::kManaged) {
      // A heap far below system memory usually means the GTT limit kept its
      // default of half the RAM. Managed weights are pageable system memory:
      // the kernel may swap them, and later device allocations (the DSpark
      // drafter, session scratch) find the heap too small.
      size_t free_b = 0, total_b = 0;
      (void)hipMemGetInfo(&free_b, &total_b);
      gufo::server::Logger::LogFormatted(
          gufo::server::LogLevel::kWarn, "ds4",
          "weights (%.2f GiB) do not fit the GPU memory heap (%.2f GiB free "
          "of %.2f GiB) above its %.2f GiB dispatch reserve; placing them in "
          "pageable managed memory, where the kernel may swap them and later "
          "GPU allocations can fail. Raise the GTT limit so the heap covers "
          "system memory: kernel parameters ttm.pages_limit=32505856 "
          "(amdttm.pages_limit with AMD's DKMS driver) and "
          "amdgpu.gttsize=126976 on a 128 GB Strix Halo",
          (double)hip_model_weight_bytes() / 1073741824.0,
          (double)free_b / 1073741824.0, (double)total_b / 1073741824.0,
          (double)gufo::hip::kDispatchReserve / 1073741824.0);
    }
  }
  return g_model_weight_alloc.get();
}

char* hip_model_range_alloc(uint64_t bytes, const char* what) {
  if (bytes == 0)
    return NULL;
  if (g_model_cache_full)
    return NULL;
  // The loader already merges adjacent tensors into large upload spans.
  // Allocate each span exactly: packing them into larger fixed blocks leaves
  // several GiB unused, reducing memory available to concurrent sessions.
  void* dev = NULL;
  hipError_t err = hip_model_weight_allocator()->Allocate(&dev, (size_t)bytes);
  if (err != hipSuccess) {
    fprintf(stderr,
            DS41_GPU_LOG_PREFIX
            "model span alloc failed for %s (%.2f MiB): %s\n",
            what ? what : "weights", (double)bytes / 1048576.0,
            hipGetErrorString(err));
    (void)hipGetLastError();
    g_model_cache_full = 1;
    return NULL;
  }
  g_model_allocations.push_back(dev);
  return (char*)dev;
}

/* Allocates and uploads one weight range from its shard file. */
const char* hip_model_range_ptr_from_fd(const void* model_map, uint64_t offset,
                                        uint64_t bytes, const char* what,
                                        bool wait) {
  if (g_model_fd < 0 || bytes == 0)
    return NULL;
  if (g_model_fd_host_base != NULL && model_map != g_model_fd_host_base)
    return NULL;
  char* dev = hip_model_range_alloc(bytes, what);
  if (!dev) {
    return NULL;
  }
  std::string error;
  if (!g_model_uploader) {
    std::vector<gufo::core::GgufMappedRegion> regions;
    for (const ModelShard& shard : g_model_shards) {
      regions.push_back(gufo::core::GgufMappedRegion{
          .data = (const uint8_t*)model_map + shard.offset,
          .size = static_cast<size_t>(shard.size),
          .file_descriptor = shard.fd,
      });
    }
    g_model_uploader = gufo::hip::WeightUpload::Create(regions, &error);
  }
  uint32_t shard = 0;
  while (shard < g_model_shards.size() &&
         offset >= g_model_shards[shard].offset + g_model_shards[shard].size) {
    shard++;
  }
  if (shard == g_model_shards.size() || offset < g_model_shards[shard].offset ||
      bytes >
          g_model_shards[shard].offset + g_model_shards[shard].size - offset) {
    fprintf(stderr,
            DS41_GPU_LOG_PREFIX "model range %" PRIu64 "+%" PRIu64
                                " crosses a GGUF shard\n",
            offset, bytes);
    return NULL;
  }
  if (!g_model_uploader ||
      !g_model_uploader->Copy(shard, offset - g_model_shards[shard].offset,
                              bytes, dev, &error) ||
      (wait && !g_model_uploader->Finish(&error))) {
    fprintf(stderr, DS41_GPU_LOG_PREFIX "model upload failed for %s: %s\n",
            what ? what : "weights", error.c_str());
    return NULL;
  }

  g_model_ranges.push_back({model_map, offset, bytes, dev});
  g_model_range_by_offset[offset] = g_model_ranges.size() - 1u;
  g_model_range_bytes += bytes;
  hip_model_load_progress_note(g_model_range_bytes);
  return (const char*)dev;
}

const char* hip_model_range_ptr(const void* model_map, uint64_t offset,
                                uint64_t bytes, const char* what) {
  if (bytes == 0)
    return (const char*)model_map + offset;

  if (model_map != g_model_host_base) {
    fprintf(stderr,
            DS41_GPU_LOG_PREFIX
            "model range rejected for %s: unregistered model mapping\n",
            what ? what : "weights");
    return NULL;
  }

  const uint64_t end = offset + bytes;
  auto exact = g_model_range_by_offset.find(offset);
  if (exact != g_model_range_by_offset.end()) {
    const ModelRange& r = g_model_ranges[exact->second];
    if (r.host_base == model_map && end >= offset && bytes <= r.bytes)
      return r.device_ptr;
  }
  for (const ModelRange& r : g_model_ranges) {
    if (r.host_base == model_map && offset >= r.offset && end >= offset &&
        end <= r.offset + r.bytes) {
      return r.device_ptr + (offset - r.offset);
    }
  }

  return hip_model_range_ptr_from_fd(model_map, offset, bytes, what, true);
}

int hip_model_range_is_cached(const void* model_map, uint64_t offset,
                              uint64_t bytes) {
  if (bytes == 0)
    return 1;

  const uint64_t end = offset + bytes;
  if (end < offset)
    return 0;
  for (const ModelRange& r : g_model_ranges) {
    if (r.host_base == model_map && offset >= r.offset &&
        end <= r.offset + r.bytes) {
      return 1;
    }
  }
  return 0;
}

void hip_model_range_release_all() {
  for (void* allocation : g_model_allocations) {
    (void)hipFree(allocation);
  }
  g_model_allocations.clear();
  g_model_ranges.clear();
  g_model_range_by_offset.clear();
  g_model_range_bytes = 0;
  g_model_weight_alloc.reset();
  hip_model_load_progress_reset();
}

int set_model_map(const void* model_map, uint64_t model_size) {
  if (!model_map || model_size == 0)
    return 0;
  if (g_model_host_base == model_map && g_model_registered_size == model_size)
    return 1;
  hip_model_range_release_all();
  g_model_host_base = model_map;
  g_model_registered_size = model_size;
  g_model_cache_full = 0;
  if (g_model_fd >= 0 && g_model_fd_host_base == NULL) {
    g_model_fd_host_base = model_map;
  }
  return 1;
}

}  // namespace

extern "C" void ds41_gpu_set_resident_weight_bytes(uint64_t bytes) {
  g_model_resident_weight_bytes = bytes;
}

extern "C" int ds41_gpu_init(void) {
  int dev = 0;
  if (!hip_ok(hipSetDevice(dev), "set device"))
    return 0;
  int gfx1151 = 0;
  hipDeviceProp_t prop;
  if (hipGetDeviceProperties(&prop, dev) == hipSuccess) {
    gfx1151 = prop.major == 11 && prop.minor == 5;
    gufo::server::Logger::LogFormatted(gufo::server::LogLevel::kInfo, "ds4",
                                       "backend initialized on %s (sm_%d%d)",
                                       prop.name, prop.major, prop.minor);
  }
  const bool mmq_ready = gfx1151 && ds41_mmq_init(dev) == 0;
  gufo::server::Logger::LogFormatted(gufo::server::LogLevel::kInfo, "ds4",
                                     "native MMQ %s",
                                     mmq_ready ? "enabled" : "unavailable");
  return 1;
}

extern "C" void ds41_gpu_cleanup(void) {
  g_model_uploader.reset();
  (void)hipDeviceSynchronize();
  ds41_mmq_cleanup();
  hip_model_range_release_all();
  g_model_host_base = NULL;
  g_model_registered_size = 0;
  g_model_resident_weight_bytes = 0;
  g_model_weight_alloc.reset();
  g_model_fd = -1;
  g_model_shards.clear();
  g_model_cache_full = 0;
}

extern "C" int ds41_gpu_set_model_map_range(const void* model_map,
                                            uint64_t model_size,
                                            uint64_t map_offset,
                                            uint64_t map_size,
                                            uint64_t max_tensor_bytes) {
  (void)max_tensor_bytes;
  if (!model_map || model_size == 0 || map_offset > model_size ||
      map_size > model_size - map_offset) {
    return 0;
  }
  // Ranges are uploaded per tensor span by ds41_gpu_cache_model_range, so
  // registering the mapping copies nothing.
  return set_model_map(model_map, model_size);
}

extern "C" const void* ds41_gpu_model_ptr(const void* model_map,
                                          uint64_t offset, uint64_t bytes) {
  return hip_model_range_ptr(model_map, offset, bytes, "tensor");
}

extern "C" int ds41_gpu_set_model_shards(const int* fds,
                                         const uint64_t* offsets,
                                         const uint64_t* sizes,
                                         uint32_t count) {
  if (!fds || !offsets || !sizes || count == 0)
    return 0;
  g_model_shards.clear();
  for (uint32_t i = 0; i < count; i++) {
    g_model_shards.push_back({fds[i], offsets[i], sizes[i]});
  }
  g_model_uploader.reset();
  g_model_fd = fds[0];
  g_model_fd_host_base = g_model_host_base;
  return 1;
}

extern "C" int ds41_gpu_release_model_staging(void) {
  std::string error;
  const bool ok = !g_model_uploader || g_model_uploader->Finish(&error);
  if (!ok)
    fprintf(stderr, DS41_GPU_LOG_PREFIX "model upload failed: %s\n",
            error.c_str());
  g_model_uploader.reset();
  return ok;
}

extern "C" int ds41_gpu_cache_model_range(const void* model_map,
                                          uint64_t model_size, uint64_t offset,
                                          uint64_t bytes, const char* label) {
  if (!model_map || bytes == 0)
    return 1;
  if (offset > model_size || bytes > model_size - offset)
    return 0;
  if (hip_model_range_is_cached(model_map, offset, bytes))
    return 1;
  if (!hip_model_range_ptr_from_fd(model_map, offset, bytes,
                                   label ? label : "model_tensor", false))
    return 0;
  return hip_model_range_is_cached(model_map, offset, bytes);
}
