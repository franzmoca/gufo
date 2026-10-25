#include "backend.h"

#include <hipblaslt/hipblaslt.h>

/* The process-wide logger lets the loading lines below honor --log-level
 * instead of writing straight to stderr. */
#include "src/cli/serve/logging.hpp"

/* Vendored llama.cpp quantized-matmul tier. It owns the routed IQ2 gate/up and
 * dense Q8 prefill kernels; see mmq/VENDOR.md. */
#include "mmq/ds41_mmq.h"

/* MMQ can reuse a producer-emitted Q8_1 activation instead of quantizing its
 * own. This backend keeps no such registry, so MMQ always takes its regular
 * activation-quantize prelude. */
extern "C" int ds41_hip_q8_fold_take_q81(const void* src, uint64_t in_dim,
                                         const void** q81) {
  (void)src;
  (void)in_dim;
  if (q81)
    *q81 = NULL;
  return 0;
}

#define FULL_WARP_MASK 0xFFFFFFFFFFFFFFFFULL
#define MASK_T uint64_t
#define DS41_GPU_BACKEND_NAME "ROCm"
#define DS41_GPU_LOG_PREFIX "ds41: ROCm "
#define DS41_GPU_BLAS_NAME "hipBLAS"

#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <limits.h>
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include <algorithm>
#include <unordered_map>
#include <vector>

#include "resident_api.h"
#include "src/core/hip/managed_alloc.hpp"
#include "src/core/hip/weight_upload.hpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

enum {
  /* attention_decode_mixed_kernel stores raw-window scores plus visible
   * compressed scores in shared memory.  The host routes larger unmasked
   * decode calls to the online attention kernel so this fixed buffer never
   * becomes an out-of-bounds write at long context. */
  DS41_ROCM_ATTENTION_SCORE_CAP = 8192u,
  DS41_ROCM_ATTENTION_RAW_SCORE_CAP = 256u
};

struct ds41_gpu_tensor {
  void* ptr;
  uint64_t bytes;
  int owner;
};

#include "detail/ds41_rocm_attention.hip.hpp"
#include "detail/ds41_rocm_attention_launch.hip.hpp"
#include "detail/ds41_rocm_common.hip.hpp"
#include "detail/ds41_rocm_compressor.hip.hpp"
#include "detail/ds41_rocm_embedding_launch.hip.hpp"
#include "detail/ds41_rocm_fp8_kv.hip.hpp"
#include "detail/ds41_rocm_fp8_kv_launch.hip.hpp"
#include "detail/ds41_rocm_hc.hip.hpp"
#include "detail/ds41_rocm_hc_output_launch.hip.hpp"
#include "detail/ds41_rocm_indexer.hip.hpp"
#include "detail/ds41_rocm_matmul.hip.hpp"
#include "detail/ds41_rocm_misc_launch.hip.hpp"
#include "detail/ds41_rocm_moe.hip.hpp"
#include "detail/ds41_rocm_moe_launch.hip.hpp"
#include "detail/ds41_rocm_norm_rope.hip.hpp"
#include "detail/ds41_rocm_output.hip.hpp"
#include "detail/ds41_rocm_q8.hip.hpp"
#include "detail/ds41_rocm_router.hip.hpp"
#include "detail/ds41_rocm_runtime.hip.hpp"
#include "detail/ds41_rocm_shared_expert.hip.hpp"
