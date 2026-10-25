// SPDX-License-Identifier: MIT
// Model-private HIP runtime helpers for the imported ggml matrix templates.

#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <mutex>

#include "common.hip.hpp"  // pulls in ds41_ggml_stubs.h via redirect headers

// Device properties are queried once and shared by the matrix dispatchers.
const ds41_ggml_hip_device_info& ds41_ggml_hip_info() {
  static ds41_ggml_hip_device_info info;
  static std::once_flag once;
  std::call_once(once, [] {
    int count = 0;
    hipError_t err = hipGetDeviceCount(&count);
    if (err != hipSuccess) {
      fprintf(stderr, "ds41_ggml_hip_info: hipGetDeviceCount failed: %s\n",
              hipGetErrorString(err));
      count = 0;
    }
    if (count > DS41_GGML_HIP_MAX_DEVICES)
      count = DS41_GGML_HIP_MAX_DEVICES;
    info.device_count = count;
    for (int i = 0; i < count; i++) {
      hipDeviceProp_t p;
      DS41_HIP_CHECK(hipGetDeviceProperties(&p, i));
      unsigned arch = 0;
      if (sscanf(p.gcnArchName, "gfx%x", &arch) != 1) {
        fprintf(stderr,
                "ds41_ggml_hip_info: unsupported HIP architecture '%s'\n",
                p.gcnArchName);
      }
      info.devices[i].cc = DS41_GGML_HIP_CC_OFFSET_AMD + (int)arch;
      info.devices[i].nsm = p.multiProcessorCount;
      info.devices[i].smpb = p.sharedMemPerBlock;
      info.devices[i].smpbo = p.sharedMemPerBlockOptin;
      info.devices[i].integrated = p.integrated != 0;
      info.devices[i].vmm = false;
      info.devices[i].vmm_granularity = 0;
      info.devices[i].total_vram = p.totalGlobalMem;
      info.devices[i].warp_size = p.warpSize;
      info.devices[i].supports_cooperative_launch = p.cooperativeLaunch != 0;
    }
  });
  return info;
}

// ds41_ggml_hip_get_device / ds41_ggml_hip_set_device are declared (not
// defined) in common.hip.hpp. We provide thin wrappers.

int ds41_ggml_hip_get_device() {
  int dev = 0;
  hipGetDevice(&dev);
  return dev;
}

void ds41_ggml_hip_set_device(int device) {
  int cur = -1;
  hipGetDevice(&cur);
  if (cur != device) {
    DS41_HIP_CHECK(hipSetDevice(device));
  }
}

int64_t ds41_ggml_time_us() {
  using clock = std::chrono::steady_clock;
  static const auto t0 = clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(clock::now() -
                                                               t0)
      .count();
}

// ----------------------------------------------------------------------------
// ds41_ggml_hip_error: invoked by the DS41_HIP_CHECK / DS41_HIPBLAS_CHECK
// macros defined in common.hip.hpp on the error path. Marked [[noreturn]] in
// the declaration (common.hip.hpp:155) - abort() satisfies that contract.
// ----------------------------------------------------------------------------

[[noreturn]] void ds41_ggml_hip_error(const char* stmt, const char* func,
                                      const char* file, int line,
                                      const char* msg) {
  fprintf(stderr, "HIP error: %s\n  call: %s\n  in: %s at %s:%d\n", msg, stmt,
          func, file, line);
  fflush(stderr);
  abort();
}

// ----------------------------------------------------------------------------
// Concrete pool wrapping hipMallocAsync / hipFreeAsync.
// ----------------------------------------------------------------------------

namespace {

/* Thread-local stream that ds41_naive_pool uses for hipMallocAsync /
 * hipFreeAsync.  Defaults to hipStreamPerThread (preserves prior
 * behaviour).  Step 8 / CUDA Graphs sets this to the capture stream
 * before allocating, so the alloc node lives on the captured stream
 * and capture is not invalidated.  Set via ds41_pool_set_stream() from
 * ds41_mmq.cu wrappers; an explicit stream=0 means the legacy default
 * stream so pool ops stay ordered with legacy-stream kernels. */
static thread_local hipStream_t t_ds41_pool_stream = hipStreamPerThread;

}  // anonymous namespace

extern "C" void ds41_pool_set_stream(hipStream_t stream) {
  t_ds41_pool_stream = stream;
}

extern "C" hipStream_t ds41_pool_get_stream(void) {
  return t_ds41_pool_stream;
}

namespace {

struct ds41_naive_pool : public ds41_ggml_hip_pool {
  int device;

  explicit ds41_naive_pool(int device) : device(device) {}

  void* alloc(size_t size, size_t* actual_size) override {
    ds41_ggml_hip_set_device(device);
    void* ptr = nullptr;
    DS41_HIP_CHECK(hipMallocAsync(&ptr, size, t_ds41_pool_stream));
    if (actual_size)
      *actual_size = size;
    return ptr;
  }

  void free(void* ptr, size_t /*size*/) override {
    if (!ptr)
      return;
    ds41_ggml_hip_set_device(device);
    DS41_HIP_CHECK(hipFreeAsync(ptr, t_ds41_pool_stream));
  }
};

}  // anonymous namespace

std::unique_ptr<ds41_ggml_hip_pool> ds41_ggml_hip_context::new_pool_for_device(
    int device, int /*stream_no*/) {
  return std::unique_ptr<ds41_ggml_hip_pool>(new ds41_naive_pool(device));
}

ds41_ggml_hip_context::~ds41_ggml_hip_context() {
  if (copy_event) {
    hipEventDestroy(copy_event);
    copy_event = nullptr;
  }
  // streams[][], hipblas_handles[], and pools[][] are owned-by-value
  // (hipStream_t and hipblasHandle_t are opaque handles - destroying the
  // context "should" tear them down, but in our shim ds4 manages streams
  // externally and we leave them alone. The pools auto-destruct via
  // unique_ptr.).
}
