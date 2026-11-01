// SPDX-License-Identifier: MIT
// ds41_mmq.hip.cpp - host wrapper around llama.cpp's vendored mul_mat_q kernels.
//
// The fused target-prefill MoE dispatcher (ds41_mmq_fused_down,
// ds41_swiglu_weighted_f32, the fused_down branches of
// ds41_mmq_moe_pair_impl, and the ds41_mmq_iq2_xxs_q2_K_moe_fused_* entry
// points) is Portions Copyright (c) 2026 Marco Palaferri (MIT), adapted
// from xangel82/DS4-GB10-GX10-DSpark-CUDA commit 910501e (v0.5 inc-9).
//
// Implements the public ds41_mmq_* entry points and explicitly instantiates
// the mul_mat_q_case<T> template for each quant type the caller needs.
//
#include "ds41_mmq.h"

#include "common.hip.hpp"
#include "mmq.hip.hpp"
#include "quantize.hip.hpp"
#include "mmid.hip.hpp"
#include "ds41_mmq_d2r.hip.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>

// ----------------------------------------------------------------------------
// Init
// ----------------------------------------------------------------------------

static int    g_routed_max_expert_rows = 0;
static int    g_routed_tile_cols = 0;
/* ds41: per-expert tile-width classes, see ds41_mmq_set_routed_classes. */
constexpr int kRoutedClassesMax = 4;
static const uint8_t *g_routed_class_dev = nullptr;
static int    g_routed_class_count = 0;
static int    g_routed_class_width[kRoutedClassesMax] = {};
static int    g_routed_class_rows[kRoutedClassesMax] = {};

struct ds41_hip_persistent_scratch {
    void *ptr = nullptr;
    size_t bytes = 0;
    std::mutex mutex;
};

static ds41_hip_persistent_scratch g_hip_dense_q81_scratch;
static ds41_hip_persistent_scratch g_hip_pair_map_scratch;

static void *ds41_hip_persistent_scratch_reserve_locked(
        ds41_hip_persistent_scratch *scratch,
        size_t bytes,
        size_t granularity,
        hipStream_t stream,
        const char *tag) {
    if (!scratch || bytes == 0 || stream != (hipStream_t)0) return nullptr;
    if (scratch->ptr && scratch->bytes >= bytes) return scratch->ptr;
    if (granularity == 0 || bytes > SIZE_MAX - (granularity - 1u)) {
        return nullptr;
    }

    const size_t reserve =
        ((bytes + granularity - 1u) / granularity) * granularity;
    hipError_t err = hipStreamSynchronize(stream);
    if (err != hipSuccess) {
        fprintf(stderr, "%s: scratch resize synchronization failed: %s\n",
                tag, hipGetErrorString(err));
        return nullptr;
    }

    void *next = nullptr;
    err = hipMalloc(&next, reserve);
    if (err != hipSuccess) {
        fprintf(stderr, "%s: persistent scratch allocation of %zu bytes failed: %s\n",
                tag, reserve, hipGetErrorString(err));
        (void)hipGetLastError();
        return nullptr;
    }
    if (scratch->ptr) (void)hipFree(scratch->ptr);
    scratch->ptr = next;
    scratch->bytes = reserve;
    return scratch->ptr;
}

extern "C" void ds41_mmq_set_routed_max_expert_rows(int rows) {
    g_routed_max_expert_rows = rows > 0 ? rows : 0;
}

extern "C" void ds41_mmq_set_routed_tile_cols(int cols) {
    g_routed_tile_cols = cols > 0 ? cols : 0;
}

extern "C" void ds41_mmq_set_routed_classes(const uint8_t *expert_class, int count, const int *widths,
                                            const int *max_rows) {
    const bool on = expert_class && widths && max_rows && count > 0 && count <= kRoutedClassesMax;
    g_routed_class_dev = on ? expert_class : nullptr;
    g_routed_class_count = on ? count : 0;
    for (int i = 0; i < g_routed_class_count; i++) {
        g_routed_class_width[i] = widths[i];
        g_routed_class_rows[i] = max_rows[i];
    }
}

/* Routed launches: one per tile-width class over its experts when the caller
 * set classes, otherwise one launch over every expert. */
template <ggml_type type>
static void ds41_mmq_routed_case(ds41_ggml_hip_context &ctx, mmq_args args, hipStream_t stream) {
    if (g_routed_class_count == 0 || !args.ids_dst || args.use_stream_k) {
        mul_mat_q_case<type>(ctx, args, stream);
        return;
    }
    for (int i = 0; i < g_routed_class_count; i++) {
        if (g_routed_class_rows[i] <= 0) continue;
        args.expert_class = g_routed_class_dev;
        args.expert_class_id = i;
        args.mmq_x_request = g_routed_class_width[i];
        args.ncols_grid_max = g_routed_class_rows[i];
        mul_mat_q_case<type>(ctx, args, stream);
    }
}

extern "C" int ds41_mmq_routed_tile_cols_for_counts(
        const unsigned int *counts, int n_experts) {
    /* Pick the column-tile width that minimizes what the routed gate/up pass
     * actually issues. A tile is executed whole, so expert e with c_e rows
     * costs `ceil(c_e / w)` tiles of `w` columns each, and every one of those
     * tiles also re-reads e's weight panel. The padding term therefore falls
     * with w while the weight term rises, and the balance depends on the
     * distribution rather than on its maximum: at a 4,096-token chunk the mean
     * bucket is 96 rows and 80 is right, at 512 tokens it is twelve and 80
     * spends 85% of its issue on padding.
     *
     * `kPanel` prices one weight-panel reload in column-equivalents. It was
     * fitted, not assumed: the real per-expert counts were dumped for every
     * layer at 256, 512, 1,024 and 2,048 tokens and scored against a
     * forced-width sweep of the same four chunks. Over P in 0..128, P = 16 is
     * the single constant closest to the per-width optimum -- 0.85% mean and
     * 3.4% worst-case shortfall, against 1.7% mean at P = 8 or 32 and 2.6% at
     * P = 48. No constant reproduces the whole curve: the optimum is 16 columns
     * at 512 tokens and 48 at 1,024, and this cost function is linear in
     * `tiles * (w + P)`, so it cannot separate those two. The caller therefore
     * runs it only where it is validated; see
     * DS41_ROCM_ROUTED_TILE_MODEL_ROWS. */
    if (!counts || n_experts <= 0) return 0;
    constexpr int kPanel = 16;
    int best_cols = 0;
    long long best_cost = 0;
    for (int w = 16; w <= 80; w += 16) {
        long long cost = 0;
        for (int e = 0; e < n_experts; e++) {
            const unsigned int c = counts[e];
            if (c == 0u) continue;
            cost += (long long)((c + (unsigned int)w - 1u) / (unsigned int)w) *
                    (long long)(w + kPanel);
        }
        if (best_cols == 0 || cost < best_cost) {
            best_cols = w;
            best_cost = cost;
        }
    }
    return best_cols;
}

extern "C" int ds41_mmq_routed_classes_for_counts(const unsigned int *counts, int n_experts,
                                                  uint8_t *expert_class, int *widths, int *max_rows) {
    /* ds41: a tile width per expert instead of one per launch. Expert e with
     * c_e rows costs ceil(c_e / w) tiles of (w + kPanel) column-equivalents,
     * kPanel pricing the per-tile weight-panel unpack. Fitted on V4.1 Flash
     * (IQ1_S/IQ1_M/IQ2_XXS experts, 2048-token batches): forced widths of 16,
     * 32, 48 and 64 gave tile cost ~ 2.6e-6 s per column plus 5.4e-5 s per
     * tile summed over a layer's three projections, i.e. kPanel ~ 21. 48 is
     * left out: it measured slower than that line. Adding 128-column tiles
     * or moving kPanel within 14-28 measured within noise. */
    constexpr int kN = 3;
    constexpr int kWidths[kN] = {16, 32, 64};
    constexpr int kPanel = 21;
    if (!counts || !expert_class || !widths || !max_rows || n_experts <= 0) return 0;
    for (int i = 0; i < kN; i++) {
        widths[i] = kWidths[i];
        max_rows[i] = 0;
    }
    for (int e = 0; e < n_experts; e++) {
        const long long c = counts[e];
        int best = 0;
        long long best_cost = -1;
        for (int i = 0; i < kN; i++) {
            const long long cost = (c + kWidths[i] - 1) / kWidths[i] * (kWidths[i] + kPanel);
            if (best_cost < 0 || cost < best_cost) {
                best = i;
                best_cost = cost;
            }
        }
        expert_class[e] = (uint8_t)best;
        if ((int)c > max_rows[best]) max_rows[best] = (int)c;
    }
    return kN;
}


/* flat-pool p5b: the direct fused gate/up path stages its input Q8 through
 * the ids_src1 column->token map inside the D2R kernel, so the activation
 * quantize runs once per TOKEN (n_tokens rows) instead of once per
 * assignment slot (n_tokens * top-k rows and bytes).  Bit-identical by
 * construction: the quantize is row-local, so a gathered slot for token t
 * holds exactly the bytes of compact row t; the kernel consumes the same
 * blocks in the same tile order through the indirection. The two buffers were
 * byte-compared in situ during rollout (bad=0). */
/* Column floor for the D2R down path. */
static constexpr int64_t D2R_MIN_COLS = 1024;

extern "C" int ds41_mmq_init(int device) {
    if (device < 0) {
        fprintf(stderr, "ds41_mmq_init: invalid device %d\n", device);
        return -1;
    }
    ds41_ggml_hip_set_device(device);
    // Trigger lazy population of the device-info singleton.
    const auto & info = ds41_ggml_hip_info();
    if (info.device_count == 0) {
        fprintf(stderr, "ds41_mmq_init: no HIP devices found\n");
        return -1;
    }
    if (device >= info.device_count) {
        fprintf(stderr, "ds41_mmq_init: device %d out of range (have %d)\n",
                device, info.device_count);
        return -1;
    }

    return 0;
}

extern "C" void ds41_mmq_cleanup(void) {
    (void)hipDeviceSynchronize();
    ds41_pool_release_cached();
    (void)hipDeviceSynchronize();
    {
        std::lock_guard<std::mutex> lock(g_hip_dense_q81_scratch.mutex);
        if (g_hip_dense_q81_scratch.ptr) {
            (void)hipFree(g_hip_dense_q81_scratch.ptr);
            g_hip_dense_q81_scratch.ptr = nullptr;
            g_hip_dense_q81_scratch.bytes = 0;
        }
    }
    {
        std::lock_guard<std::mutex> lock(g_hip_pair_map_scratch.mutex);
        if (g_hip_pair_map_scratch.ptr) {
            (void)hipFree(g_hip_pair_map_scratch.ptr);
            g_hip_pair_map_scratch.ptr = nullptr;
            g_hip_pair_map_scratch.bytes = 0;
        }
    }
}

// ----------------------------------------------------------------------------
// Dense matmul implementation, shared across all three quant types.
//
// Computes  out[col, row] = sum_k W[row, k] * X[k, col]   with W in the
// type-specific block layout and X / out in F32 (X K-innermost row-major,
// out column-major out[col*M + row]).
//
// Mirrors upstream mmq.cu:154-159 (the no-ids branch) but builds mmq_args
// from plain pointers + shape ints instead of ggml_tensor introspection.
// ----------------------------------------------------------------------------

// Per-device singleton context. Owns the pool for stream-K fixup scratch.
// Phase 4 will make this per-stream as well; for now a single context per
// device is sufficient for the dense path.
namespace {

__global__ static void ds41_mmq_sanitize_f32_kernel(float *p, uint64_t n) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float v = p[i];
    if (!isfinite(v)) p[i] = 0.0f;
}

static void ds41_mmq_sanitize_f32(float *p, uint64_t n, hipStream_t stream) {
    if (!p || n == 0) return;
    ds41_mmq_sanitize_f32_kernel<<<(unsigned)((n + 255u) / 256u), 256, 0, stream>>>(p, n);
}

ds41_ggml_hip_context * get_ctx_for_device(int device) {
    static ds41_ggml_hip_context * cached[DS41_GGML_HIP_MAX_DEVICES] = {};
    if (device < 0 || device >= DS41_GGML_HIP_MAX_DEVICES) return nullptr;
    if (!cached[device]) {
        cached[device] = new ds41_ggml_hip_context(device);
    }
    return cached[device];
}

template <ggml_type type>
bool ds41_mmq_k_tile_supported(const char *tag, int K, int cc) {
    if constexpr (type == GGML_TYPE_MXFP4 || type == GGML_TYPE_NVFP4) {
        if (blackwell_mma_available(cc) && K % MMQ_ITER_K_FP4 != 0) {
            fprintf(stderr,
                    "%s: Blackwell FP4 K=%d must be a multiple of %d\n",
                    tag, K, MMQ_ITER_K_FP4);
            return false;
        }
    }
    return true;
}

template <ggml_type type>
int ds41_mmq_dense_impl(
        const char  * tag,
        const void  * W,
        const float * X_f32,
        float       * out_f32,
        int           M,
        int           N,
        int           K,
        hipStream_t  stream,
        /* ds41: `groups` independent [M, K] weights stored back to back; token
         * rows of X hold the groups' K inputs side by side and rows of out the
         * groups' M outputs (one MMQ channel per group). */
        int           groups = 1,
        /* ds41: further [M_extra[i], K] weights of the same type applied to
         * the same X (quantized once), each into its own out_extra[i]. */
        int           extra = 0,
        const void  * const * W_extra = nullptr,
        float       * const * out_extra = nullptr,
        const int   * M_extra = nullptr) {

    if (!W || !X_f32 || !out_f32 || groups <= 0 || (extra > 0 && (groups != 1 || !W_extra || !out_extra || !M_extra))) {
        fprintf(stderr, "%s: null pointer\n", tag);
        return -1;
    }
    if (K <= 0 || M <= 0 || N <= 0) {
        fprintf(stderr, "%s: bad shape M=%d N=%d K=%d\n", tag, M, N, K);
        return -1;
    }
    if (K % 256 != 0) {
        // mmq requires K to be a multiple of the largest super-block size
        // it sees during the inner tile loop, which is QK_K=256.
        fprintf(stderr, "%s: K=%d must be a multiple of 256\n", tag, K);
        return -1;
    }

    const int dev = ds41_ggml_hip_get_device();
    const int cc  = ds41_ggml_hip_info().devices[dev].cc;
    if (!ds41_mmq_k_tile_supported<type>(tag, K, cc)) return -1;

    ds41_ggml_hip_context * ctx = get_ctx_for_device(dev);
    if (!ctx) {
        fprintf(stderr, "%s: failed to get HIP context for device %d\n", tag, dev);
        return -1;
    }

    /* Order the pool's hipMallocAsync/hipFreeAsync on the SAME
     * stream the kernels below launch on.  The pool defaults to
     * hipStreamPerThread; with kernels on the legacy stream the RAII free is
     * ordered on an EMPTY stream, so the driver can recycle/remap the scratch
     * while the in-flight quantize/GEMM still reads it -> intermittent illegal
     * access under shape churn (the batched-draft early-step crash).  The vec
     * impls already do this (graph-capture fix); the batched impls were missed.
     */
    ds41_pool_set_stream(stream);

    // 1. Quantize F32 activations into the format consumed by MMQ. Blackwell
    //    MXFP4 uses native FP4 tensor cores; other paths use MMQ Q8_1.
    const int64_t ne00         = K;
    const int64_t ne10_padded  = GGML_PAD((int64_t)K, MATRIX_ROW_PADDING);
    const int64_t ne11         = N;
    const int64_t ne12         = groups;
    const int64_t ne13         = 1;

    const bool use_native_fp4 =
        type == GGML_TYPE_MXFP4 && blackwell_mma_available(cc);
    const size_t y_block_size = use_native_fp4
        ? sizeof(block_fp4_mmq) : sizeof(block_q8_1_mmq);
    const size_t y_values_per_block = use_native_fp4
        ? QK_FP4_MMQ : 4 * QK8_1;
    const size_t nbytes_src1_q8_1 =
        ne13 * ne12 * ne11 * ne10_padded * y_block_size /
            y_values_per_block +
        get_mmq_x_max_host(cc) * sizeof(block_q8_1_mmq);

    ds41_ggml_hip_pool_alloc<char> src1_q8_1_pool;
    char *src1_q8_1 = nullptr;
    std::unique_lock<std::mutex> dense_scratch_lock;
    if (stream == (hipStream_t)0) {
        dense_scratch_lock =
            std::unique_lock<std::mutex>(g_hip_dense_q81_scratch.mutex);
        src1_q8_1 = (char *)ds41_hip_persistent_scratch_reserve_locked(
            &g_hip_dense_q81_scratch,
            nbytes_src1_q8_1,
            64u * 1024u * 1024u,
            stream,
            tag);
    }
    if (!src1_q8_1) {
        src1_q8_1 = src1_q8_1_pool.alloc(
            ctx->pool(), nbytes_src1_q8_1);
    }

    // S1.1a fix: the mmq Y (activation) buffer is over-allocated for the kernel's
    // tail-tile reads (the +mmq_x_max blocks above), and ne11 columns may not fill
    // the final column tile -- but ds41_quantize_mmq_q8_1_hip only writes the ne11 valid
    // columns.  The mmq kernel (mmq.hip.hpp:3528) unconditionally loads the full column
    // tile, reading the never-written tail.  Pool allocs reuse stale device memory,
    // so that tail is non-deterministic: any allocator/stream perturbation (e.g. an
    // MTP draft's hipMalloc) changes it and flips a near-threshold argmax in the
    // batched forward (confirmed by compute-sanitizer --tool initcheck on a PRO6000
    // / sm_120: 4-byte uninitialized __global__ read in mul_mat_q_process_tile).
    // The tail's dot-products are masked out by write_back, so only their
    // non-determinism matters; zero the buffer so the tail is a deterministic zero
    // (a zero q8_1 block contributes 0 to the dot product).

    if (use_native_fp4) {
        ds41_quantize_mmq_fp4_hip(
            X_f32, /*ids=*/nullptr, (void *)src1_q8_1,
            type, /*ne00=*/K, /*s11=*/(int64_t)K, /*s12=*/0, /*s13=*/0,
            /*ne0=*/ne10_padded, /*ne1=*/ne11, /*ne2=*/ne12, /*ne3=*/ne13,
            stream);
    } else {
        ds41_quantize_mmq_q8_1_hip(
            X_f32, /*ids=*/nullptr, (void *)src1_q8_1,
            type, /*ne00=*/K, /*s11=*/(int64_t)K * groups, /*s12=*/groups > 1 ? (int64_t)K : 0, /*s13=*/0,
            /*ne0=*/ne10_padded, /*ne1=*/ne11, /*ne2=*/ne12, /*ne3=*/ne13,
            stream);
    }

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        fprintf(stderr, "%s: quantize failed: %s\n", tag, hipGetErrorString(err));
        return -2;
    }

    // 2. Build mmq_args. stride_row_x is in WEIGHT BLOCKS per row, which
    //    is K / blck_size(type). Q8_0 has block size 32; Q2_K and IQ2_XXS
    //    are K-quants with block size 256.
    const int64_t blck   = ggml_blck_size(type);
    const int64_t s01    = (int64_t)K / blck;
    const int64_t s1     = (int64_t)M * groups;
    const int64_t s12    = ne11 * ne10_padded * y_block_size /
                           (y_values_per_block * sizeof(int));
    const int64_t s13    = ne12 * s12;

    const bool use_stream_k =
        (DS41_GGML_HIP_CC_IS_NVIDIA(cc) && ds41_ggml_hip_highest_compiled_arch(cc) >= DS41_GGML_HIP_CC_VOLTA) ||
        DS41_GGML_HIP_CC_IS_CDNA(cc);

    mmq_args args = {
        /*x=*/(const char *)W,
        /*type_x=*/type,
        /*y=*/(const int *)src1_q8_1,
        /*ids_dst=*/nullptr,
        /*expert_bounds=*/nullptr,
        /*dst=*/out_f32,
        /*ncols_x=*/ne00,    /*nrows_x=*/(int64_t)M,    /*ncols_dst=*/ne11,
        /*stride_row_x=*/s01,/*ncols_y=*/ne11,          /*nrows_dst=*/s1,
        /*nchannels_x=*/groups, /*nchannels_y=*/groups,
        /*stride_channel_x=*/(int64_t)M * s01, /*stride_channel_y=*/s12, /*stride_channel_dst=*/(int64_t)M,
        /*nsamples_x=*/1,    /*nsamples_y=*/1,
        /*stride_sample_x=*/0, /*stride_sample_y=*/s13, /*stride_sample_dst=*/0,
        /*use_stream_k=*/use_stream_k,
        /*ncols_max=*/ne11,
    };
    /* Tiled launches sanitize at the store (see mul_mat_q_case). */
    const bool inline_sanitize = !use_stream_k;
    args.sanitize_output = inline_sanitize;

    mul_mat_q_case<type>(*ctx, args, stream);

    err = hipGetLastError();
    if (err != hipSuccess) {
        fprintf(stderr, "%s: mul_mat_q_case launch failed: %s\n", tag, hipGetErrorString(err));
        return -3;
    }
    if (!inline_sanitize) {
        ds41_mmq_sanitize_f32(out_f32, (uint64_t)s1 * (uint64_t)N, stream);
    }
    for (int i = 0; i < extra; i++) {
        if (!W_extra[i] || !out_extra[i] || M_extra[i] <= 0) return -1;
        args.x = (const char *)W_extra[i];
        args.dst = out_extra[i];
        args.nrows_x = M_extra[i];
        args.nrows_dst = M_extra[i];
        mul_mat_q_case<type>(*ctx, args, stream);
        err = hipGetLastError();
        if (err != hipSuccess) {
            fprintf(stderr, "%s: mul_mat_q_case launch failed: %s\n", tag, hipGetErrorString(err));
            return -3;
        }
        if (!inline_sanitize) {
            ds41_mmq_sanitize_f32(out_extra[i], (uint64_t)M_extra[i] * (uint64_t)N, stream);
        }
    }
    return 0;
} // anonymous namespace

}

// ----------------------------------------------------------------------------
// MoE matmul implementation, shared across all three quant types.
//
// Mirrors upstream mmq.cu:163-222 (the ids != nullptr branch).  Caller
// provides:
//   - per-expert weights stacked contiguously
//   - per-token activations [n_tokens, K]
//   - routing table ids[t, s] = expert id
// The wrapper invokes:
//   1. ds41_ggml_hip_launch_mm_ids_helper to build (ids_src1, ids_dst,
//      expert_bounds) - permutations that sort assignments by expert.
//   2. ds41_quantize_mmq_q8_1_hip with ids_src1 - gathers and quantizes the
//      activation into the expert-major flat layout.
//   3. mul_mat_q_case<type> with ids_dst + expert_bounds - the matmul.
// ----------------------------------------------------------------------------

namespace {

template <ggml_type type>
int ds41_mmq_moe_impl(
        const char    * tag,
        const void    * W,
        const float   * X_f32,
        const int32_t * ids,
        float         * out_f32,
        int             M,
        int             K,
        int             n_tokens,
        int             n_experts,
        int             n_expert_used,
        hipStream_t    stream,
        /* ds4 (P4 Inc3): optional aligned-SoA artifact; when non-null the mmq
         * kernel loads tiles from it directly and W is ignored (see mmq_args). */
        const char    * x_soa      = NULL,
        int64_t         soa_blocks = 0,
        /* ds4 (P3): false skips the whole-buffer nonfinite pass; only valid
         * when every consumer sanitizes at read (the routed-MoE swiglu/sum
         * kernels do). */
        bool            sanitize_out = true) {

    if (!W || !X_f32 || !ids || !out_f32) {
        fprintf(stderr, "%s: null pointer\n", tag);
        return -1;
    }
    if (M <= 0 || K <= 0 || n_tokens <= 0 || n_experts <= 0 || n_expert_used <= 0) {
        fprintf(stderr, "%s: bad shape M=%d K=%d ntok=%d nexp=%d nused=%d\n",
                tag, M, K, n_tokens, n_experts, n_expert_used);
        return -1;
    }
    if (K % 256 != 0) {
        fprintf(stderr, "%s: K=%d must be a multiple of 256\n", tag, K);
        return -1;
    }
    if (n_expert_used > n_experts) {
        fprintf(stderr, "%s: n_expert_used=%d > n_experts=%d\n", tag, n_expert_used, n_experts);
        return -1;
    }

    const int dev = ds41_ggml_hip_get_device();
    const int cc  = ds41_ggml_hip_info().devices[dev].cc;
    if (!ds41_mmq_k_tile_supported<type>(tag, K, cc)) return -1;

    ds41_ggml_hip_context * ctx = get_ctx_for_device(dev);
    if (!ctx) {
        fprintf(stderr, "%s: failed to get HIP context for device %d\n", tag, dev);
        return -1;
    }

    ds41_pool_set_stream(stream);  /* task #22: pool ops must be stream-ordered with the kernels (see ds41_mmq_dense_impl) */

    const int64_t ne_get_rows  = (int64_t)n_tokens * n_expert_used;
    const int64_t ne00         = K;
    const int64_t ne10_padded  = GGML_PAD((int64_t)K, MATRIX_ROW_PADDING);
    const int64_t ne11         = 1;             // src1 rows per channel (one per token)
    const int64_t ne12         = n_tokens;      // src1 channels (= tokens)
    const int64_t blck         = ggml_blck_size(type);
    const int64_t s01          = (int64_t)K / blck;
    const int64_t s02          = (int64_t)M * s01;   // per-expert weight stride in blocks

    // 1. Build the expert-major work map.
    ds41_ggml_hip_pool_alloc<int32_t> ids_src1(ctx->pool(), ne_get_rows);
    ds41_ggml_hip_pool_alloc<int32_t> ids_dst(ctx->pool(), ne_get_rows);
    ds41_ggml_hip_pool_alloc<int32_t> expert_bounds(ctx->pool(), n_experts + 1);

    // mm_ids_helper compacts - it only writes ids_src1
    // entries for in-range router ids and drops invalid ones (the router's NaN
    // path emits -1 by design), so with any dropped id the tail of ids_src1
    // stays unwritten pool memory.  quantize_mmq_q8_1's grid covers all
    // ne_get_rows rows and gathers x rows via ids_src1[i1] unconditionally
    // (quantize.cu:304), so a stale/garbage tail entry becomes a wild OOB read
    // (the intermittent batched-draft illegal access; B200 memcheck-convicted).
    // Zero both id maps so unwritten tail slots gather/scatter row 0 instead:
    // those lanes' output is never consumed (the mmq write-back loop is
    // expert_bounds-bounded), the cost is a few KB of memset on-stream.
    hipMemsetAsync(ids_src1.get(), 0, ne_get_rows * sizeof(int32_t), stream);
    hipMemsetAsync(ids_dst.get(),  0, ne_get_rows * sizeof(int32_t), stream);

    // si1 = stride between tokens in the ids tensor, in elements. Our ids is
    // contiguous [n_tokens, n_expert_used] so si1 = n_expert_used.
    // sis1 = stride between src1 channels in row-units. With ne11=1, sis1=1
    //        means each "channel" of src1 is one row of K floats.
    const int si1  = n_expert_used;
    const int sis1 = 1;

    // The smem mm_ids_helper uses n_tokens * 4 bytes of dynamic shared memory;
    // the down matmul reaches here with n_tokens = assignments (6x the forward
    // width), so 8192-row prefill chunks pass 48384 "tokens" > cap.  P5: past
    // the cap the launcher dispatches the bit-identical two-pass global
    // variant instead (mmid.cu mm_ids_helper_global) — refusing here used to
    // throw the WHOLE MoE block (including gate/up mmq work) onto the legacy
    // expert-tile fallback, the W8192 prefill cliff.
    ds41_ggml_hip_launch_mm_ids_helper(
        ids, ids_src1.get(), ids_dst.get(), expert_bounds.get(),
        n_experts, n_tokens, n_expert_used, /*nchannels_y=*/(int)ne11, si1, sis1, stream);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        fprintf(stderr, "%s: mm_ids_helper failed: %s\n", tag, hipGetErrorString(err));
        return -2;
    }

    // 2. Gather + quantize activations. Native Blackwell MXFP4 consumes FP4;
    //    all other MMQ kernels consume Q8_1.
    const bool use_native_fp4 =
        type == GGML_TYPE_MXFP4 && blackwell_mma_available(cc);
    const size_t y_block_size = use_native_fp4
        ? sizeof(block_fp4_mmq) : sizeof(block_q8_1_mmq);
    const size_t y_values_per_block = use_native_fp4
        ? QK_FP4_MMQ : 4 * QK8_1;
    const size_t nbytes_src1_q8_1 =
        ne_get_rows * ne10_padded * y_block_size / y_values_per_block +
        get_mmq_x_max_host(cc) * sizeof(block_q8_1_mmq);
    ds41_ggml_hip_pool_alloc<char> src1_q8_1(ctx->pool(), nbytes_src1_q8_1);

    // S1.1a fix (same as the dense path): the mmq Y buffer is over-allocated for the
    // kernel's tail-tile reads and ne_get_rows columns need not fill the final mmq
    // column tile, but quantize only writes the valid columns.  The mmq kernel
    // (mmq.hip.hpp:3528) unconditionally loads the full tile, reading the never-written
    // tail from stale pool memory -> allocator-perturbation-dependent garbage in the
    // (write_back-masked) tail lanes -> non-deterministic batched-forward output.
    // Zero it so the masked-out tail is a deterministic zero.

    // src1 logical [K, ne11=1, ne12=n_tokens, ne13=1] - K innermost, then
    // one row per channel, channels = tokens.
    const int64_t s11_src = (int64_t)K;                                 // stride between rows of a channel
    const int64_t s12_src = (int64_t)K * ne11;                          // stride between channels = K*1
    const int64_t s13_src = (int64_t)K * ne11 * ne12;                   // stride between samples

    if (use_native_fp4) {
        ds41_quantize_mmq_fp4_hip(
            X_f32, ids_src1.get(), (void *)src1_q8_1.get(),
            type, /*ne00=*/K, s11_src, s12_src, s13_src,
            /*ne0=*/ne10_padded, /*ne1=*/ne_get_rows, /*ne2=*/1, /*ne3=*/1,
            stream);
    } else {
        ds41_quantize_mmq_q8_1_hip(
            X_f32, ids_src1.get(), (void *)src1_q8_1.get(),
            type, /*ne00=*/K, s11_src, s12_src, s13_src,
            /*ne0=*/ne10_padded, /*ne1=*/ne_get_rows, /*ne2=*/1, /*ne3=*/1,
            stream);
    }

    err = hipGetLastError();
    if (err != hipSuccess) {
        fprintf(stderr, "%s: MMQ activation quantize failed: %s\n",
                tag, hipGetErrorString(err));
        return -3;
    }

    // 3. Build mmq_args for the MoE path.
    //
    // dst layout convention matches upstream's MoE branch
    // (mmq.cu:215-220): dst is interpreted as [M, n_expert_used, n_tokens]
    // with M innermost and n_expert_used as the second dim that mmq writes
    // through ids_dst.  s1 = M (the column stride in the flat dst buffer
    // mmq writes into).  The output is column-major: out[col*M + row].
    const int64_t s1            = (int64_t)M;
    // stride_channel_y per upstream: ne11 * ne10_padded * sizeof(block_q8_1)
    //                                     / (QK8_1 * sizeof(int))
    // In MoE mode the kernel zeroes out the channel-stride contribution to
    // offset_y after reading expert_bounds, so the value is permissive -
    // but we set it consistently with upstream.
    const int64_t s12_mmq = ne11 * ne10_padded * y_block_size /
                            (y_values_per_block * sizeof(int));
    const int64_t s13_mmq = ne12 * s12_mmq;

    const bool use_stream_k =
        (DS41_GGML_HIP_CC_IS_NVIDIA(cc) && ds41_ggml_hip_highest_compiled_arch(cc) >= DS41_GGML_HIP_CC_VOLTA) ||
        DS41_GGML_HIP_CC_IS_CDNA(cc);

    if (type == GGML_TYPE_Q2_K && x_soa != nullptr &&
        K % 256 == 0 && M % 2 == 0 && ne_get_rows >= D2R_MIN_COLS) {
        static int d2r_avail_cc = -1;
        static int d2r_avail = 0;
        if (d2r_avail_cc != cc) {
            d2r_avail_cc = cc;
            d2r_avail = ds41_mmq_q2_K_moe_d2r_available(cc) ? 1 : 0;
        }
        if (d2r_avail) {
            const size_t d2r_work_bytes =
                ds41_mmq_q2_K_moe_d2r_scratch_bytes(ne_get_rows, n_experts);
            if (d2r_work_bytes != 0) {
                ds41_ggml_hip_pool_alloc<char> d2r_work(ctx->pool(), d2r_work_bytes);
                const int d2r_rc = ds41_mmq_q2_K_moe_d2r_launch(
                    x_soa, soa_blocks, src1_q8_1.get(), ids_dst.get(), expert_bounds.get(),
                    out_f32, M, K, ne_get_rows, n_experts, d2r_work.get(), d2r_work_bytes,
                    stream);
                if (d2r_rc == 0) {
                    return 0;
                }
            }
        }
    }

    mmq_args args = {
        /*x=*/(const char *)W,
        /*type_x=*/type,
        /*y=*/(const int *)src1_q8_1.get(),
        /*ids_dst=*/ids_dst.get(),
        /*expert_bounds=*/expert_bounds.get(),
        /*dst=*/out_f32,
        /*ncols_x=*/ne00,
        /*nrows_x=*/(int64_t)M,
        /*ncols_dst=*/ne_get_rows,
        /*stride_row_x=*/s01,
        /*ncols_y=*/ne_get_rows,
        /*nrows_dst=*/s1,
        /*nchannels_x=*/(int64_t)n_experts,
        /*nchannels_y=*/(int64_t)n_experts,
        /*stride_channel_x=*/s02,
        /*stride_channel_y=*/s12_mmq,
        /*stride_channel_dst=*/(int64_t)0,
        /*nsamples_x=*/1,
        /*nsamples_y=*/1,
        /*stride_sample_x=*/0,
        /*stride_sample_y=*/s13_mmq,
        /*stride_sample_dst=*/0,
        /*use_stream_k=*/use_stream_k,
        /*ncols_max=*/ne_get_rows,
        /*x_soa=*/x_soa,
        /*soa_blocks=*/soa_blocks,
    };
    /* The caller's bucket bounds, as on the routed pair path (see
     * ds41_mmq_set_routed_max_expert_rows / _tile_cols). */
    args.ncols_grid_max = g_routed_max_expert_rows;
    args.mmq_x_request = g_routed_tile_cols;

    ds41_mmq_routed_case<type>(*ctx, args, stream);

    err = hipGetLastError();
    if (err != hipSuccess) {
        fprintf(stderr, "%s: mul_mat_q_case (moe) launch failed: %s\n", tag, hipGetErrorString(err));
        return -4;
    }
    /* Tiled launches of these types sanitize at the store (see mul_mat_q_case). */
    if (sanitize_out && (use_stream_k || type == GGML_TYPE_Q8_0)) {
        ds41_mmq_sanitize_f32(out_f32, (uint64_t)M * (uint64_t)ne_get_rows, stream);
    }
    return 0;
}

struct ds41_mmq_fused_down {
    const void  * W;
    const char  * W_soa;
    int64_t       soa_blocks;
    const float * router_weights;
    float       * mid_f32;
    float       * out;
    int           out_dim;
    float         clamp;
    bool          direct_gateup_q8;
    void        * input_q8_scratch;
    size_t        input_q8_scratch_bytes;
    void        * q8_scratch;
    size_t        q8_scratch_bytes;
    void        * work_scratch;
    size_t        work_scratch_bytes;
    /* flat-pool p5c: producer-emitted token-compact q8 of X (see the
     * fused_direct_soa doc in ds41_mmq.h); NULL = quantize internally. */
    const void  * input_q8_ext;
    size_t        input_q8_ext_bytes;
};

struct ds41_mmq_swiglu_epilogue {
    const float * router_weights;
    float       * mid_f32;
    half        * mid_f16;
    float         clamp;
};

static bool ds41_mmq_take_scratch(
        void *base, size_t capacity, size_t *offset,
        size_t bytes, size_t alignment, void **result) {
    if (!base || !offset || !result || alignment == 0 ||
        (alignment & (alignment - 1)) != 0) {
        return false;
    }
    if (*offset > capacity) return false;
    const uintptr_t address = (uintptr_t)base + *offset;
    const size_t padding = (size_t)(-(uintptr_t)address) & (alignment - 1);
    if (padding > capacity - *offset) return false;
    const size_t aligned = *offset + padding;
    if (bytes > capacity - aligned) return false;
    *result = (char *)base + aligned;
    *offset = aligned + bytes;
    return true;
}

// Produce the weighted SwiGLU rows in their canonical pair-major order. The
// proven upstream quantizer below gathers them through the already available
// ids_dst map, so gate/up and down share one expert-major schedule without a
// second mm_ids_helper.
static __global__ void ds41_swiglu_weighted_f32(
        const float * __restrict__ gate,
        const float * __restrict__ up,
        const float * __restrict__ router_weights,
        float * __restrict__ mid,
        uint64_t n,
        int K,
        float clamp) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const uint64_t pair = i / (uint64_t)K;
    float g = isfinite(gate[i]) ? gate[i] : 0.0f;
    float u = isfinite(up[i]) ? up[i] : 0.0f;
    if (clamp > 1.0e-6f) {
        g = fminf(g, clamp);
        u = fminf(fmaxf(u, -clamp), clamp);
    }
    mid[i] = (g / (1.0f + expf(-g))) * u * router_weights[pair];
}

// Paired MoE: one helper + one quantize covers both weights.  Internal
// structure mirrors ds41_mmq_moe_impl above; the only differences are the
// two W pointers, the two output pointers, and the second mul_mat_q_case
// launch with a fresh (x, dst) pair.
template<ggml_type type, bool token_expert_unique = false>
int ds41_mmq_moe_pair_impl(
    const char* tag, const void* W_a, const void* W_b, const float* X_f32,
    const int32_t* ids, float* out_a, float* out_b, int M, int K, int n_tokens,
    int n_experts, int n_expert_used, hipStream_t stream,
    /* ds4 (P4 Inc3): optional aligned-SoA artifacts for W_a / W_b (same
     * shape, so one block count); see ds41_mmq_moe_impl. */
    const char* xa_soa = NULL, const char* xb_soa = NULL,
    int64_t soa_blocks = 0,
    /* ds4 (P3): see ds41_mmq_moe_impl. */
    bool sanitize_out = true, const ds41_mmq_fused_down* fused_down = nullptr,
    const ds41_mmq_swiglu_epilogue* swiglu_epilogue = nullptr) {
  const bool direct_gateup_q8 =
      fused_down != nullptr && fused_down->direct_gateup_q8;
  if (!W_a || !W_b || !X_f32 || !ids ||
      (!direct_gateup_q8 && (!out_a || (!out_b && !swiglu_epilogue)))) {
    fprintf(stderr, "%s: null pointer\n", tag);
    return -1;
  }
  if (M <= 0 || K <= 0 || n_tokens <= 0 || n_experts <= 0 ||
      n_expert_used <= 0) {
    fprintf(stderr, "%s: bad shape M=%d K=%d ntok=%d nexp=%d nused=%d\n", tag,
            M, K, n_tokens, n_experts, n_expert_used);
    return -1;
  }
  if (K % 256 != 0) {
    fprintf(stderr, "%s: K=%d must be a multiple of 256\n", tag, K);
    return -1;
  }
  if (n_expert_used > n_experts) {
    fprintf(stderr, "%s: n_expert_used=%d > n_experts=%d\n", tag, n_expert_used,
            n_experts);
    return -1;
  }
  if (fused_down &&
      (type != GGML_TYPE_IQ2_XXS || !fused_down->W ||
       !fused_down->router_weights ||
       (!direct_gateup_q8 && !fused_down->mid_f32) ||
       (direct_gateup_q8 &&
        (!xa_soa || !xb_soa || !fused_down->W_soa ||
         !fused_down->input_q8_scratch ||
         fused_down->input_q8_scratch_bytes == 0 || !fused_down->q8_scratch ||
         fused_down->q8_scratch_bytes == 0 || !fused_down->work_scratch ||
         fused_down->work_scratch_bytes == 0)) ||
       !fused_down->out || fused_down->out_dim <= 0 || M % 256 != 0)) {
    fprintf(stderr, "%s: invalid fused Q2_K down configuration\n", tag);
    return -1;
  }
  if (swiglu_epilogue &&
      (fused_down || !swiglu_epilogue->router_weights || !swiglu_epilogue->mid_f32)) {
    fprintf(stderr, "%s: invalid weighted SwiGLU epilogue\n", tag);
    return -1;
  }

  const int dev = ds41_ggml_hip_get_device();
  const int cc = ds41_ggml_hip_info().devices[dev].cc;
  if (!ds41_mmq_k_tile_supported<type>(tag, K, cc))
    return -1;

  ds41_ggml_hip_context* ctx = get_ctx_for_device(dev);
  if (!ctx) {
    fprintf(stderr, "%s: failed to get HIP context for device %d\n", tag, dev);
    return -1;
  }

  ds41_pool_set_stream(stream); /* task #22: pool ops must be stream-ordered with
                                  the kernels (see ds41_mmq_dense_impl) */

  const int64_t ne_get_rows = (int64_t)n_tokens * n_expert_used;
  const int64_t ne00 = K;
  const int64_t ne10_padded = GGML_PAD((int64_t)K, MATRIX_ROW_PADDING);
  const int64_t ne11 = 1;
  const int64_t ne12 = n_tokens;
  const int64_t blck = ggml_blck_size(type);
  const int64_t s01 = (int64_t)K / blck;
  const int64_t s02 = (int64_t)M * s01;

  ds41_ggml_hip_pool_alloc<int32_t> ids_src1_alloc;
  ds41_ggml_hip_pool_alloc<int32_t> ids_dst_alloc;
  ds41_ggml_hip_pool_alloc<int32_t> expert_bounds_alloc;
  int32_t* ids_src1 = nullptr;
  int32_t* ids_dst = nullptr;
  int32_t* expert_bounds = nullptr;
    std::unique_lock<std::mutex> pair_map_scratch_lock;
    void *direct_work = nullptr;
    size_t direct_work_bytes = 0;

    const bool use_native_fp4 =
        type == GGML_TYPE_MXFP4 && blackwell_mma_available(cc);
    const size_t y_block_size = use_native_fp4
        ? sizeof(block_fp4_mmq) : sizeof(block_q8_1_mmq);
    const size_t y_values_per_block = use_native_fp4
        ? QK_FP4_MMQ : 4 * QK8_1;
    const size_t nbytes_src1_q8_1 =
        ne_get_rows * ne10_padded * y_block_size / y_values_per_block +
        get_mmq_x_max_host(cc) * sizeof(block_q8_1_mmq);
    size_t direct_down_q8_bytes = 0;
    if (direct_gateup_q8) {
        const int64_t down_ne10_padded = GGML_PAD((int64_t)M, MATRIX_ROW_PADDING);
        direct_down_q8_bytes =
            (size_t)ne_get_rows * (size_t)down_ne10_padded *
                sizeof(block_q8_1) / QK8_1 +
            (size_t)get_mmq_x_max_host(cc) * sizeof(block_q8_1_mmq);
        const size_t gateup_work_bytes =
            ds41_mmq_iq2_xxs_moe_d2r_fused_scratch_bytes(
                ne_get_rows, n_experts);
        const size_t down_work_bytes =
            ds41_mmq_q2_K_moe_d2r_scratch_bytes(ne_get_rows, n_experts);
        if (fused_down->input_q8_scratch_bytes < nbytes_src1_q8_1) return -91;
        if (fused_down->q8_scratch_bytes < direct_down_q8_bytes) return -92;
        if (gateup_work_bytes == 0 || down_work_bytes == 0) return -93;
        if (ne_get_rows < D2R_MIN_COLS) return -95;
        if (!ds41_mmq_iq2_xxs_moe_d2r_available(cc) ||
            !ds41_mmq_q2_K_moe_d2r_available(cc)) {
            return -96;
        }

        size_t offset = 0;
        void *ids_src1_raw = nullptr;
        void *ids_dst_raw = nullptr;
        void *expert_bounds_raw = nullptr;
        direct_work_bytes = gateup_work_bytes > down_work_bytes
            ? gateup_work_bytes : down_work_bytes;
        if (!ds41_mmq_take_scratch(
                fused_down->work_scratch, fused_down->work_scratch_bytes,
                &offset, (size_t)ne_get_rows * sizeof(int32_t), 256,
                &ids_src1_raw) ||
            !ds41_mmq_take_scratch(
                fused_down->work_scratch, fused_down->work_scratch_bytes,
                &offset, (size_t)ne_get_rows * sizeof(int32_t), 256,
                &ids_dst_raw) ||
            !ds41_mmq_take_scratch(
                fused_down->work_scratch, fused_down->work_scratch_bytes,
                &offset, (size_t)(n_experts + 1) * sizeof(int32_t), 256,
                &expert_bounds_raw) ||
            !ds41_mmq_take_scratch(
                fused_down->work_scratch, fused_down->work_scratch_bytes,
                &offset, direct_work_bytes, 256, &direct_work)) {
            return -97;
        }
        ids_src1 = (int32_t *)ids_src1_raw;
        ids_dst = (int32_t *)ids_dst_raw;
        expert_bounds = (int32_t *)expert_bounds_raw;
    } else {
        if (stream == (hipStream_t)0 &&
            (uint64_t)ne_get_rows <=
                (SIZE_MAX / sizeof(int32_t) - (size_t)n_experts - 1u) / 2u) {
            pair_map_scratch_lock =
                std::unique_lock<std::mutex>(g_hip_pair_map_scratch.mutex);
            const size_t map_elems =
                2u * (size_t)ne_get_rows + (size_t)n_experts + 1u;
            int32_t *map = (int32_t *)
                ds41_hip_persistent_scratch_reserve_locked(
                    &g_hip_pair_map_scratch,
                    map_elems * sizeof(int32_t),
                    1024u * 1024u,
                    stream,
                    tag);
            if (map) {
                ids_src1 = map;
                ids_dst = map + ne_get_rows;
                expert_bounds = ids_dst + ne_get_rows;
            }
        }
        if (!ids_src1) {
            ids_src1 = ids_src1_alloc.alloc(ctx->pool(), ne_get_rows);
            ids_dst = ids_dst_alloc.alloc(ctx->pool(), ne_get_rows);
            expert_bounds =
                expert_bounds_alloc.alloc(ctx->pool(), n_experts + 1);
        }
    }

    const int si1  = n_expert_used;
    const int sis1 = 1;

    // Same cap guard as ds41_mmq_moe_impl (see comment there): past the smem
    // cap the launcher takes the bit-identical global variant (P5).
    hipError_t err = hipSuccess;
    {
      // Zero the ID maps, as in ds41_mmq_moe_impl
      // so entries dropped by mm_ids_helper never expose stale pool memory.
      hipMemsetAsync(ids_src1, 0, ne_get_rows * sizeof(int32_t), stream);
      hipMemsetAsync(ids_dst, 0, ne_get_rows * sizeof(int32_t), stream);
      ds41_ggml_hip_launch_mm_ids_helper(
          ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens,
          n_expert_used, /*nchannels_y=*/(int)ne11, si1, sis1, stream);

      err = hipGetLastError();
      if (err != hipSuccess) {
        fprintf(stderr, "%s: mm_ids_helper failed: %s\n", tag,
                hipGetErrorString(err));
        return -2;
      }
    }

    const bool use_stream_k =
        (DS41_GGML_HIP_CC_IS_NVIDIA(cc) && ds41_ggml_hip_highest_compiled_arch(cc) >= DS41_GGML_HIP_CC_VOLTA) ||
        DS41_GGML_HIP_CC_IS_CDNA(cc);
    if (swiglu_epilogue && use_stream_k) {
        return -13;
    }
    /* True top-k assignments cannot select one expert twice for a token, so
     * no expert bucket can exceed n_tokens rows. Keep the conservative
     * gathered-row bound for generic MMQ callers, including DSpark/MTP. */
    const int64_t routed_ncols_max = fused_down || token_expert_unique
        ? (int64_t)n_tokens
        : ne_get_rows;
    /* The caller may know the largest expert bucket (see
     * ds41_mmq_set_routed_max_expert_rows). ncols_max keeps selecting the tile
     * shape from the chunk width; only the column-tile grid shrinks. */
    const int64_t routed_ncols_grid =
        g_routed_max_expert_rows > 0 &&
        (int64_t)g_routed_max_expert_rows < routed_ncols_max
            ? (int64_t)g_routed_max_expert_rows
            : 0;

    /* The materialized path stream-frees gate/up Q8_1 before allocating the
     * down Q8_1. The direct path needs both simultaneously, but writes down
     * Q8_1 into caller-owned gate scratch instead of growing the CUDA pool. */
    {
    ds41_ggml_hip_pool_alloc<char> src1_q8_1_alloc;
    char *src1_q8_1 = nullptr;
    if (direct_gateup_q8) {
        src1_q8_1 = (char *)fused_down->input_q8_scratch;
    } else {
        src1_q8_1 = src1_q8_1_alloc.alloc(ctx->pool(), nbytes_src1_q8_1);
    }

    // S1.1a fix (same as the dense/moe paths): zero the over-allocated mmq Y buffer
    // so the kernel's unconditional masked-out tail-tile read (mmq.hip.hpp:3528) returns
    // a deterministic zero instead of allocator-perturbation-dependent stale memory.
    const int64_t s11_src = (int64_t)K;
    const int64_t s12_src = (int64_t)K * ne11;
    const int64_t s13_src = (int64_t)K * ne11 * ne12;
    /* p5b: token-compact input quantize on the direct fused path (the
     * materialized paths below keep the slot-gathered form: their pair/mmq
     * consumers address Y by assignment slot).
     * p5c: a producer-emitted token-compact buffer replaces even the
     * compact quantize — same layout by construction (ib = kseg*n_tokens
     * + row), so the p5b indirection consumes it unchanged. */
    const bool moe_yind = direct_gateup_q8;
    const void *input_q8_ext = nullptr;
    if (moe_yind && fused_down && fused_down->input_q8_ext) {
        const size_t ext_need =
            (size_t)n_tokens * (size_t)ne10_padded * sizeof(block_q8_1) / QK8_1;
        if (fused_down->input_q8_ext_bytes >= ext_need) {
            input_q8_ext = fused_down->input_q8_ext;
        }
    }
    if (input_q8_ext) {
        src1_q8_1 = (char *)const_cast<void *>(input_q8_ext);
        static bool ext_logged = false;
        if (!ext_logged) {
            ext_logged = true;
            fprintf(stderr, "ds41: moe gateup consuming producer q8 "
                    "(flat-pool p5c, first n_tokens=%d)\n", n_tokens);
        }
    }
    const int64_t quant_rows = moe_yind ? (int64_t)n_tokens : ne_get_rows;
    if (!input_q8_ext) {
      if (use_native_fp4) {
        ds41_quantize_mmq_fp4_hip(X_f32, moe_yind ? nullptr : ids_src1,
                              (void*)src1_q8_1, type, /*ne00=*/K, s11_src,
                              s12_src, s13_src,
                              /*ne0=*/ne10_padded, /*ne1=*/quant_rows,
                              /*ne2=*/1, /*ne3=*/1, stream);
      } else {
        ds41_quantize_mmq_q8_1_hip(X_f32, moe_yind ? nullptr : ids_src1,
                               (void*)src1_q8_1, type, /*ne00=*/K, s11_src,
                               s12_src, s13_src,
                               /*ne0=*/ne10_padded, /*ne1=*/quant_rows,
                               /*ne2=*/1, /*ne3=*/1, stream);
      }

      err = hipGetLastError();
      if (err != hipSuccess) {
        fprintf(stderr, "%s: MMQ activation quantize failed: %s\n", tag,
                hipGetErrorString(err));
        return -3;
      }
    }

    if (direct_gateup_q8) {
        if (moe_yind) {
            static bool yind_logged = false;
            if (!yind_logged) {
                yind_logged = true;
                fprintf(stderr, "ds41: moe gateup y-indirect q8 staging engage "
                        "(flat-pool p5b, first n_tokens=%d n_assign=%lld)\n",
                        n_tokens, (long long)ne_get_rows);
            }
        }
        const size_t gateup_work_bytes =
            ds41_mmq_iq2_xxs_moe_d2r_fused_scratch_bytes(
                ne_get_rows, n_experts);
        if (gateup_work_bytes == 0) {
            return -9;
        }
        {
          const int d2r_rc = ds41_mmq_iq2_xxs_moe_d2r_fused_launch(
              xa_soa, xb_soa, soa_blocks, src1_q8_1,
              moe_yind ? ids_src1 : nullptr, moe_yind ? n_tokens : 0, ids_dst,
              expert_bounds, fused_down->router_weights, fused_down->q8_scratch,
              M, K, ne_get_rows, n_experts, fused_down->clamp, direct_work,
              gateup_work_bytes, stream);
          if (d2r_rc != 0) {
            return -10;
          }
        }

        const size_t down_work_bytes =
            ds41_mmq_q2_K_moe_d2r_scratch_bytes(ne_get_rows, n_experts);
        if (down_work_bytes == 0) {
            return -11;
        }
        {
          const int down_rc = ds41_mmq_q2_K_moe_d2r_launch(
              fused_down->W_soa, fused_down->soa_blocks, fused_down->q8_scratch,
              ids_dst, expert_bounds, fused_down->out, fused_down->out_dim, M,
              ne_get_rows, n_experts, direct_work, down_work_bytes, stream);
          if (down_rc != 0) {
            return -12;
          }
        }
        return 0;
    }

    const int64_t s1      = (int64_t)M;
    const int64_t s12_mmq = ne11 * ne10_padded * y_block_size /
                            (y_values_per_block * sizeof(int));
    const int64_t s13_mmq = ne12 * s12_mmq;

    bool gate_up_done = false;
    if (!swiglu_epilogue &&
        type == GGML_TYPE_IQ2_XXS && xa_soa != nullptr && xb_soa != nullptr &&
        K % 256 == 0 &&
        ne_get_rows >= D2R_MIN_COLS) {
        static int d2r_iq2_avail_cc = -1;
        static int d2r_iq2_avail = 0;
        if (d2r_iq2_avail_cc != cc) {
            d2r_iq2_avail_cc = cc;
            d2r_iq2_avail = ds41_mmq_iq2_xxs_moe_d2r_available(cc) ? 1 : 0;
        }
        if (d2r_iq2_avail) {
            const size_t d2r_work_bytes =
                ds41_mmq_iq2_xxs_moe_d2r_pair_scratch_bytes(ne_get_rows, n_experts);
            if (d2r_work_bytes != 0) {
                ds41_ggml_hip_pool_alloc<char> d2r_work(ctx->pool(), d2r_work_bytes);

                const int d2r_rc = ds41_mmq_iq2_xxs_moe_d2r_pair_launch(
                        xa_soa, xb_soa, soa_blocks, src1_q8_1, ids_dst,
                        expert_bounds, out_a, out_b, M, K, ne_get_rows, n_experts,
                        d2r_work.get(), d2r_work_bytes, stream);
                if (d2r_rc == 0) {
                    gate_up_done = true;
                }
            }
        }
    }

    if (!gate_up_done) {
    mmq_args args = {
        /*x=*/(const char *)W_a,
        /*type_x=*/type,
        /*y=*/(const int *)src1_q8_1,
        /*ids_dst=*/ids_dst,
        /*expert_bounds=*/expert_bounds,
        /*dst=*/out_a,
        /*ncols_x=*/ne00,
        /*nrows_x=*/(int64_t)M,
        /*ncols_dst=*/ne_get_rows,
        /*stride_row_x=*/s01,
        /*ncols_y=*/ne_get_rows,
        /*nrows_dst=*/s1,
        /*nchannels_x=*/(int64_t)n_experts,
        /*nchannels_y=*/(int64_t)n_experts,
        /*stride_channel_x=*/s02,
        /*stride_channel_y=*/s12_mmq,
        /*stride_channel_dst=*/(int64_t)0,
        /*nsamples_x=*/1,
        /*nsamples_y=*/1,
        /*stride_sample_x=*/0,
        /*stride_sample_y=*/s13_mmq,
        /*stride_sample_dst=*/0,
        /*use_stream_k=*/use_stream_k,
        /*ncols_max=*/routed_ncols_max,
        /*x_soa=*/xa_soa,
        /*soa_blocks=*/soa_blocks,
    };
    args.ncols_grid_max = routed_ncols_grid;
    args.mmq_x_request = g_routed_tile_cols;

    {
      ds41_mmq_routed_case<type>(*ctx, args, stream);
      err = hipGetLastError();
      if (err != hipSuccess) {
        fprintf(stderr, "%s: mul_mat_q_case (pair a) launch failed: %s\n", tag,
                hipGetErrorString(err));
        return -4;
      }
    }

    // Second matmul over the same activation buffer and same routing map.
    args.x     = (const char *)W_b;
    args.dst   = out_b;
    args.x_soa = xb_soa;
    if (swiglu_epilogue) {
        args.epilogue_gate = out_a;
        args.epilogue_weights = swiglu_epilogue->router_weights;
        args.epilogue_mid = swiglu_epilogue->mid_f32;
        args.epilogue_mid_h = swiglu_epilogue->mid_f16;
        args.epilogue_clamp = swiglu_epilogue->clamp;
    }
    {
      ds41_mmq_routed_case<type>(*ctx, args, stream);
      err = hipGetLastError();
      if (err != hipSuccess) {
        fprintf(stderr, "%s: mul_mat_q_case (pair b) launch failed: %s\n", tag,
                hipGetErrorString(err));
        return -5;
      }
    }
    }
    }

    if (fused_down) {
        const int64_t down_ne10_padded = GGML_PAD((int64_t)M, MATRIX_ROW_PADDING);
        const size_t logical_q8_bytes =
            (size_t)ne_get_rows * (size_t)down_ne10_padded * sizeof(block_q8_1) / QK8_1;
        const size_t tail_q8_bytes =
            (size_t)get_mmq_x_max_host(cc) * sizeof(block_q8_1_mmq);
        ds41_ggml_hip_pool_alloc<char> down_q8_1(
            ctx->pool(), logical_q8_bytes + tail_q8_bytes);

        const uint64_t mid_values = (uint64_t)ne_get_rows * (uint64_t)M;
        {
          ds41_swiglu_weighted_f32<<<(uint32_t)((mid_values + 255u) / 256u), 256,
                                    0, stream>>>(
              out_a, out_b, fused_down->router_weights, fused_down->mid_f32,
              mid_values, M, fused_down->clamp);
          err = hipGetLastError();
          if (err != hipSuccess) {
            fprintf(stderr, "%s: weighted SwiGLU launch failed: %s\n", tag,
                    hipGetErrorString(err));
            return -6;
          }

          ds41_quantize_mmq_q8_1_hip(
              fused_down->mid_f32, ids_dst, (void*)down_q8_1.get(),
              GGML_TYPE_Q2_K, /*ne00=*/M, /*s01=*/M,
              /*s02=*/(int64_t)M, /*s03=*/(int64_t)M * ne_get_rows,
              /*ne0=*/down_ne10_padded, /*ne1=*/ne_get_rows,
              /*ne2=*/1, /*ne3=*/1, stream);
          err = hipGetLastError();
          if (err != hipSuccess) {
            fprintf(stderr, "%s: down ds41_quantize_mmq_q8_1_hip failed: %s\n", tag,
                    hipGetErrorString(err));
            return -7;
          }
        }

        const int64_t down_s01 = (int64_t)M / ggml_blck_size(GGML_TYPE_Q2_K);
        const int64_t down_s02 = (int64_t)fused_down->out_dim * down_s01;
        const int64_t down_s12 =
            down_ne10_padded * sizeof(block_q8_1) / (QK8_1 * sizeof(int));
        mmq_args down_args = {
            /*x=*/(const char *)fused_down->W,
            /*type_x=*/GGML_TYPE_Q2_K,
            /*y=*/(const int *)down_q8_1.get(),
            /*ids_dst=*/ids_dst,
            /*expert_bounds=*/expert_bounds,
            /*dst=*/fused_down->out,
            /*ncols_x=*/(int64_t)M,
            /*nrows_x=*/(int64_t)fused_down->out_dim,
            /*ncols_dst=*/ne_get_rows,
            /*stride_row_x=*/down_s01,
            /*ncols_y=*/ne_get_rows,
            /*nrows_dst=*/(int64_t)fused_down->out_dim,
            /*nchannels_x=*/(int64_t)n_experts,
            /*nchannels_y=*/(int64_t)n_experts,
            /*stride_channel_x=*/down_s02,
            /*stride_channel_y=*/down_s12,
            /*stride_channel_dst=*/(int64_t)0,
            /*nsamples_x=*/1,
            /*nsamples_y=*/1,
            /*stride_sample_x=*/0,
            /*stride_sample_y=*/ne_get_rows * down_s12,
            /*stride_sample_dst=*/0,
            /*use_stream_k=*/use_stream_k,
            /*ncols_max=*/routed_ncols_max,
            /*x_soa=*/fused_down->W_soa,
            /*soa_blocks=*/fused_down->soa_blocks,
        };
        down_args.ncols_grid_max = routed_ncols_grid;
        bool down_done = false;
        if (fused_down->W_soa != nullptr &&
            ne_get_rows >= D2R_MIN_COLS &&
            ds41_mmq_q2_K_moe_d2r_available(cc)) {
            const size_t work_bytes =
                ds41_mmq_q2_K_moe_d2r_scratch_bytes(ne_get_rows, n_experts);
            if (work_bytes != 0u) {
                ds41_ggml_hip_pool_alloc<char> work(ctx->pool(), work_bytes);

                down_done = ds41_mmq_q2_K_moe_d2r_launch(
                        fused_down->W_soa,
                        fused_down->soa_blocks,
                        down_q8_1.get(),
                        ids_dst,
                        expert_bounds,
                        fused_down->out,
                        fused_down->out_dim,
                        M,
                        ne_get_rows,
                        n_experts,
                        work.get(),
                        work_bytes,
                        stream) == 0;
            }
        }
        if (!down_done) {
          mul_mat_q_case<GGML_TYPE_Q2_K>(*ctx, down_args, stream);
          err = hipGetLastError();
          if (err != hipSuccess) {
            fprintf(stderr, "%s: fused Q2_K down launch failed: %s\n", tag,
                    hipGetErrorString(err));
            return -8;
          }
        }
    }
    if (sanitize_out && !swiglu_epilogue) {
        ds41_mmq_sanitize_f32(out_a, (uint64_t)M * (uint64_t)ne_get_rows, stream);
        ds41_mmq_sanitize_f32(out_b, (uint64_t)M * (uint64_t)ne_get_rows, stream);
    }
    return 0;
} // anonymous namespace

}

// ----------------------------------------------------------------------------
// MMVQ entry points.
//
// mmvq is upstream's matrix-vector matmul family, optimised for the
// n_tokens <= MMVQ_MAX_BATCH_SIZE=8 regime. Unlike mmq it consumes the
// CANONICAL block_q8_1 layout (via ds41_quantize_row_q8_1_hip), not the
// interleaved block_q8_1_mmq that ds41_quantize_mmq_q8_1_hip produces.
//
// The single-W _moe_vec entries cover:
//   - the down matmul at decode (treating [n_tokens=1, n_expert_used=6]
//     as [n_tokens=6, n_expert_used=1])
//   - dense attention projections at decode (n_tokens=1, no MoE)
//   - any small-batch path where mmvq's per-token grid wins over mmq's
//     tile-based approach
// ----------------------------------------------------------------------------

#include "mmvq.hip.hpp"

namespace {

template <ggml_type type>
int ds41_mmq_moe_vec_impl(
        const char    * tag,
        const void    * W,
        const float   * X_f32,
        const int32_t * ids,
        float         * out_f32,
        int             M,
        int             K,
        int             n_tokens,
        int             n_experts,
        int             n_expert_used,
        hipStream_t    stream) {

    if (!W || !X_f32 || !ids || !out_f32) {
        fprintf(stderr, "%s: null pointer\n", tag);
        return -1;
    }
    if (M <= 0 || K <= 0 || n_tokens <= 0 || n_experts <= 0 || n_expert_used <= 0) {
        fprintf(stderr, "%s: bad shape M=%d K=%d ntok=%d nexp=%d nused=%d\n",
                tag, M, K, n_tokens, n_experts, n_expert_used);
        return -1;
    }
    if (K % 256 != 0) {
        fprintf(stderr, "%s: K=%d must be a multiple of 256\n", tag, K);
        return -1;
    }
    if (n_expert_used > n_experts) {
        fprintf(stderr, "%s: n_expert_used=%d > n_experts=%d\n", tag, n_expert_used, n_experts);
        return -1;
    }
    // mmvq's per-arch batch cap. ncols_dst as computed below is
    // max(n_tokens, n_expert_used) depending on which dim we route into.
    // We follow upstream's convention: ne_y = n_tokens, ne_dst = n_expert_used.
    // So ncols_dst = n_tokens and nchannels_dst = n_expert_used.
    // FD Inc2a: n_tokens beyond the per-launch column cap no longer rejects;
    // the launch loop below splits the column dim into capped chunks.

    const int dev = ds41_ggml_hip_get_device();
    ds41_ggml_hip_context * ctx = get_ctx_for_device(dev);
    if (!ctx) {
        fprintf(stderr, "%s: failed to get HIP context for device %d\n", tag, dev);
        return -1;
    }

    // Route the pool's hipMallocAsync / hipFreeAsync through the same
    // stream the caller uses for kernel launches so allocations do not
    // invalidate graph capture.
    ds41_pool_set_stream(stream);

    // 1. Quantize X into CANONICAL Q8_1 (NOT the MMQ-interleaved variant).
    //    Layout: [ne13=1, ne12=n_tokens, ne11=1, ne10_padded blocks].
    const int64_t ne10_padded = GGML_PAD((int64_t)K, MATRIX_ROW_PADDING);
    const size_t nbytes_q8_1 =
        (size_t)n_tokens * ne10_padded * sizeof(block_q8_1) / QK8_1;
    ds41_ggml_hip_pool_alloc<char> src1_q8_1_pool;
    src1_q8_1_pool.alloc(ctx->pool(), nbytes_q8_1);
    char* src1_q8_1_ptr = src1_q8_1_pool.get();

    // s11 = stride between rows of an src1 channel in source-float units.
    //       Logical src1 [K, ne11=1, ne12=n_tokens, ne13=1] - K innermost.
    // s12 = stride between channels = K * ne11 = K.
    // s13 = stride between samples = K * ne11 * ne12 = K * n_tokens.
    ds41_quantize_row_q8_1_hip(
        X_f32, /*ids=*/nullptr, (void *)src1_q8_1_ptr,
        type, /*ne00=*/K,
        /*s11=*/(int64_t)K, /*s12=*/(int64_t)K, /*s13=*/(int64_t)K * n_tokens,
        /*ne0=*/ne10_padded, /*ne1=*/1, /*ne2=*/n_tokens, /*ne3=*/1,
        stream);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        fprintf(stderr, "%s: ds41_quantize_row_q8_1_hip failed: %s\n",
                tag, hipGetErrorString(err));
        return -2;
    }

    // 2. mmvq stride setup. Mirror upstream's ds41_ggml_hip_mul_mat_vec_q
    //    dispatch (mmvq.cu:1101-1136).
    //
    //    For MoE (ids != nullptr): per the dispatch math at line 1121-1130,
    //      ncols_dst          = ne2  = n_tokens
    //      nchannels_y        = ne11 = 1
    //      nchannels_dst      = ne1  = n_expert_used
    //      stride_col_y       = s12  = ne11 * (ne10_padded / QK8_1)
    //      stride_col_dst     = s2   = n_expert_used * M (token stride in dst)
    //      stride_channel_y   = s11  = ne10_padded / QK8_1
    //      stride_channel_dst = s1   = M (channel/slot stride in dst)
    //      ids_stride         = stride between rows of ids[] tensor
    //
    //    FD Inc2a stride fix: stride_col_dst was previously M, same as the
    //    channel stride.  That was invisible while every caller degenerated
    //    one dim (gate/up at n_tokens=1: col index always 0; down at
    //    n_expert_used=1: channel index always 0, and 1 * M == M keeps it
    //    bit-identical here).  At n_tokens >= 2 with n_expert_used > 1 the
    //    multi-token MoE kernel writes dst[chan*s1 + col*s2 + row], and
    //    equal strides collide (token=0,slot=1) with (token=1,slot=0).
    //    s2 = n_expert_used * M yields the row-major
    //    [token * n_expert_used + slot, M] layout the swiglu consumer
    //    expects.
    const int64_t blck      = ggml_blck_size(type);
    const int64_t s01_row   = (int64_t)K / blck;            // weight row stride in blocks
    const int64_t s02_chan  = (int64_t)M * s01_row;         // expert-stack stride
    const int64_t s11_y     = ne10_padded / QK8_1;          // src1 channel stride in blocks
    const int64_t s12_y     = (int64_t)1 * s11_y;           // ne11 * s11
    const int64_t s1_dst    = (int64_t)M;                   // dst channel (slot) stride
    const int64_t s2_dst    = (int64_t)n_expert_used * M;   // dst col (token) stride

    // ids_stride: stride between rows of the ids tensor in int32 elements.
    // Caller passes ids[t * n_expert_used + s], so stride between tokens
    // is n_expert_used.
    const int ids_stride = n_expert_used;

    ds41_ggml_hip_mm_fusion_args_device fusion = {};

    /* mmvq stores every (token, slot) output, zeroing non-finite values. */

    // FD Inc2a: one mmvq launch serves at most col_cap columns -- the moe
    // kernel runs one warp per column (block.y = ncols_dst) under
    // __launch_bounds__ baked per COMPILED arch + type
    // (get_mmvq_mmid_max_batch_for_device).  The runtime device cc can
    // exceed the compiled arch (CUDA_ARCH= builds run default-arch PTX on
    // newer GPUs), so the host cap MUST be looked up at the compiled arch:
    // asking the runtime cc says 8 where the compiled bounds say 7 (e.g.
    // Q2_K builds at turing_plus -> 7*warp_size threads) and the launch
    // dies with cudaErrorInvalidValue.  Wider batches run as
    // ceil(n_tokens / col_cap) launches; every per-column stride (vy, ids,
    // dst) is uniform, so a chunk is plain pointer offsets.  The single
    // quantize above already covers all columns.
    const int cc      = ds41_ggml_hip_info().devices[dev].cc;
    const int col_cap = ds41_get_mmvq_mmid_max_batch(type, ds41_ggml_hip_highest_compiled_arch(cc));

    for (int c0 = 0; c0 < n_tokens; c0 += col_cap) {
        const int ncols = (n_tokens - c0 < col_cap) ? (n_tokens - c0) : col_cap;
        mul_mat_vec_q_switch_type(
            /*vx=*/W, /*type_x=*/type,
            /*vy=*/(const void *)(src1_q8_1_ptr + (size_t)c0 * s12_y * sizeof(block_q8_1)),
            /*ids=*/ids + (size_t)c0 * ids_stride, /*fusion=*/fusion,
            /*dst=*/out_f32 + (int64_t)c0 * s2_dst,
            /*ncols_x=*/K, /*nrows_x=*/M, /*ncols_dst=*/ncols,
            /*stride_row_x=*/(int)s01_row,
            /*stride_col_y=*/(int)s12_y,
            /*stride_col_dst=*/(int)s2_dst,
            /*nchannels_x=*/n_experts,
            /*nchannels_y=*/1,
            /*nchannels_dst=*/n_expert_used,
            /*stride_channel_x=*/(int)s02_chan,
            /*stride_channel_y=*/(int)s11_y,
            /*stride_channel_dst=*/(int)s1_dst,
            /*nsamples_x=*/1, /*nsamples_dst=*/1,
            /*stride_sample_x=*/0, /*stride_sample_y=*/0, /*stride_sample_dst=*/0,
            /*ids_stride=*/ids_stride, stream);

        err = hipGetLastError();
        if (err != hipSuccess) {
            fprintf(stderr, "%s: mul_mat_vec_q_switch_type launch failed: %s (cols %d..%d cap %d)\n",
                    tag, hipGetErrorString(err), c0, c0 + ncols - 1, col_cap);
            return -3;
        }
    }

    return 0;
}

template <ggml_type type>
int ds41_mmq_dense_vec_impl(
        const char  * tag,
        const void  * W,
        const float * X_f32,
        float       * out_f32,
        int           M,
        int           N,
        int           K,
        hipStream_t  stream,
        /* ds41: `groups` independent matvecs in one launch. X is
         * [N, groups, K], W holds the groups' [M, K] blocks back to back and
         * out is [N, groups, M]. */
        int           groups = 1) {

    if (!W || !X_f32 || !out_f32) {
        fprintf(stderr, "%s: null pointer\n", tag);
        return -1;
    }
    if (M <= 0 || N <= 0 || K <= 0) {
        fprintf(stderr, "%s: bad shape M=%d N=%d K=%d\n", tag, M, N, K);
        return -1;
    }
    if (K % 256 != 0) {
        fprintf(stderr, "%s: K=%d must be a multiple of 256\n", tag, K);
        return -1;
    }
    if (N > MMVQ_MAX_BATCH_SIZE) {
        fprintf(stderr, "%s: N=%d exceeds MMVQ_MAX_BATCH_SIZE=%d\n",
                tag, N, MMVQ_MAX_BATCH_SIZE);
        return -1;
    }

    const int dev = ds41_ggml_hip_get_device();
    ds41_ggml_hip_context * ctx = get_ctx_for_device(dev);
    if (!ctx) {
        fprintf(stderr, "%s: failed to get HIP context for device %d\n", tag, dev);
        return -1;
    }

    // Route the pool's hipMallocAsync through the caller-supplied stream
    // for graph capture compatibility.  See ds41_mmq_moe_vec_impl.
    ds41_pool_set_stream(stream);

    // Dense: no MoE, ids=null. Layout [K, N, 1, 1] for src1.
    const int64_t ne10_padded = GGML_PAD((int64_t)K, MATRIX_ROW_PADDING);
    const size_t  nbytes_q8_1 = (size_t)groups * N * ne10_padded *
                                sizeof(block_q8_1) / QK8_1;
    ds41_ggml_hip_pool_alloc<char> src1_q8_1(ctx->pool(), nbytes_q8_1);

    // Dense src1 layout: K innermost, N next; ne11=N, ne12=1, ne13=1.
    ds41_quantize_row_q8_1_hip(
        X_f32, /*ids=*/nullptr, (void *)src1_q8_1.get(),
        type, /*ne00=*/K,
        /*s11=*/(int64_t)K * groups, /*s12=*/(int64_t)K, /*s13=*/(int64_t)K * N * groups,
        /*ne0=*/ne10_padded, /*ne1=*/N, /*ne2=*/groups, /*ne3=*/1,
        stream);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        fprintf(stderr, "%s: ds41_quantize_row_q8_1_hip failed: %s\n",
                tag, hipGetErrorString(err));
        return -2;
    }

    // Dense (no ids): per upstream dispatch (mmvq.cu:1121-1127),
    //   ncols_dst          = ne1  = N
    //   nchannels_y        = ne12 = 1
    //   nchannels_dst      = ne2  = 1
    //   stride_col_y       = s11  = ne10_padded / QK8_1
    //   stride_channel_y   = s12  = N * (ne10_padded / QK8_1)
    const int64_t blck      = ggml_blck_size(type);
    const int64_t s01_row   = (int64_t)K / blck;
    const int64_t s11_y     = ne10_padded / QK8_1;
    const int64_t s12_y     = (int64_t)N * s11_y;
    const int64_t s1_dst    = (int64_t)M * groups;

    ds41_ggml_hip_mm_fusion_args_device fusion = {};

    /* mmvq stores every output, zeroing non-finite values; no clear or
     * sanitize pass. */
    mul_mat_vec_q_switch_type(
        /*vx=*/W, /*type_x=*/type,
        /*vy=*/(const void *)src1_q8_1.get(),
        /*ids=*/nullptr, /*fusion=*/fusion,
        /*dst=*/out_f32,
        /*ncols_x=*/K, /*nrows_x=*/M, /*ncols_dst=*/N,
        /*stride_row_x=*/(int)s01_row,
        /*stride_col_y=*/(int)s11_y,
        /*stride_col_dst=*/(int)s1_dst,
        /*nchannels_x=*/groups,
        /*nchannels_y=*/groups,
        /*nchannels_dst=*/groups,
        /*stride_channel_x=*/(int)(s01_row * M),
        /*stride_channel_y=*/(int)s12_y,
        /*stride_channel_dst=*/M,
        /*nsamples_x=*/1, /*nsamples_dst=*/1,
        /*stride_sample_x=*/0, /*stride_sample_y=*/0, /*stride_sample_dst=*/0,
        /*ids_stride=*/0, stream);

    err = hipGetLastError();
    if (err != hipSuccess) {
        fprintf(stderr, "%s: mul_mat_vec_q_switch_type (dense) launch failed: %s\n",
                tag, hipGetErrorString(err));
        return -3;
    }
    return 0;
}

} // anonymous namespace

// V4.1 Flash: per-tensor weight types chosen at load time. mmvq dispatches on
// the type at run time; the template parameter only selects the activation
// quantization layout, which is the canonical Q8_1 for every type here.
#define DS41_MMQ_TYPED_CASES(X) \
  X(GGML_TYPE_Q8_0)             \
  X(GGML_TYPE_Q5_K)             \
  X(GGML_TYPE_Q6_K)             \
  X(GGML_TYPE_IQ1_S)            \
  X(GGML_TYPE_IQ1_M)            \
  X(GGML_TYPE_IQ2_XXS)          \
  X(GGML_TYPE_IQ2_XS)           \
  X(GGML_TYPE_IQ2_S)            \
  X(GGML_TYPE_IQ3_XXS)          \
  X(GGML_TYPE_IQ3_S)            \
  X(GGML_TYPE_IQ4_NL)           \
  X(GGML_TYPE_Q2_K)             \
  X(GGML_TYPE_Q4_K)             \
  X(GGML_TYPE_MXFP4)

extern "C" int ds41_mmq_dense_vec_typed(int type, const void* W, const float* X,
                                        float* out, int M, int N, int K,
                                        hipStream_t stream) {
  switch ((ggml_type)type) {
#define DS41_CASE(T) \
  case T:            \
    return ds41_mmq_dense_vec_impl<T>("ds41_mmq_dense_vec_typed", W, X, out, M, N, K, stream);
    DS41_MMQ_TYPED_CASES(DS41_CASE)
#undef DS41_CASE
    default:
      fprintf(stderr, "ds41_mmq_dense_vec_typed: unsupported type %d\n", type);
      return -1;
  }
}

extern "C" int ds41_mmq_grouped_vec_typed(int type, const void* W, const float* X,
                                          float* out, int M, int N, int K,
                                          int groups, hipStream_t stream) {
  if (groups <= 0) return -1;
  switch ((ggml_type)type) {
#define DS41_CASE(T)                                                                     \
  case T:                                                                                \
    return ds41_mmq_dense_vec_impl<T>("ds41_mmq_grouped_vec_typed", W, X, out, M, N, K, \
                                      stream, groups);
    DS41_MMQ_TYPED_CASES(DS41_CASE)
#undef DS41_CASE
    default:
      fprintf(stderr, "ds41_mmq_grouped_vec_typed: unsupported type %d\n", type);
      return -1;
  }
}

extern "C" int ds41_mmq_moe_vec_typed(int type, const void* W, const float* X,
                                      const int32_t* ids, float* out, int M,
                                      int K, int n_tokens, int n_experts,
                                      int n_expert_used, hipStream_t stream) {
  switch ((ggml_type)type) {
#define DS41_CASE(T)                                                                  \
  case T:                                                                             \
    return ds41_mmq_moe_vec_impl<T>("ds41_mmq_moe_vec_typed", W, X, ids, out, M, K, \
                                    n_tokens, n_experts, n_expert_used, stream);
    DS41_MMQ_TYPED_CASES(DS41_CASE)
#undef DS41_CASE
    default:
      fprintf(stderr, "ds41_mmq_moe_vec_typed: unsupported type %d\n", type);
      return -1;
  }
}

// V4.1 Flash batched GEMMs on the mmq tiles of each weight type. The tiles
// for types outside the V4 set are instantiated in ds41_mmq_v41_types.hip.cpp.
#define DS41_MMQ_BATCH_CASES(X) \
  X(GGML_TYPE_Q8_0)             \
  X(GGML_TYPE_Q5_K)             \
  X(GGML_TYPE_Q6_K)             \
  X(GGML_TYPE_IQ1_S)            \
  X(GGML_TYPE_IQ1_M)            \
  X(GGML_TYPE_IQ2_XXS)          \
  X(GGML_TYPE_IQ2_XS)           \
  X(GGML_TYPE_IQ2_S)

extern "C" int ds41_mmq_dense_typed(int type, const void* W, const float* X,
                                    float* out, int M, int N, int K,
                                    hipStream_t stream) {
  switch ((ggml_type)type) {
#define DS41_CASE(T) \
  case T:            \
    return ds41_mmq_dense_impl<T>("ds41_mmq_dense_typed", W, X, out, M, N, K, stream);
    DS41_MMQ_BATCH_CASES(DS41_CASE)
#undef DS41_CASE
    default:
      fprintf(stderr, "ds41_mmq_dense_typed: unsupported type %d\n", type);
      return -1;
  }
}

extern "C" int ds41_mmq_dense_pair_typed(int type, const void* W1, float* out1, int M1,
                                        const void* W2, float* out2, int M2, const float* X,
                                        int N, int K, hipStream_t stream) {
  const void* const w_extra[1] = {W2};
  float* const out_extra[1] = {out2};
  const int m_extra[1] = {M2};
  switch ((ggml_type)type) {
#define DS41_CASE(T)                                                                          \
  case T:                                                                                     \
    return ds41_mmq_dense_impl<T>("ds41_mmq_dense_pair_typed", W1, X, out1, M1, N, K, stream, 1, 1, \
                                  w_extra, out_extra, m_extra);
    DS41_MMQ_BATCH_CASES(DS41_CASE)
#undef DS41_CASE
    default:
      fprintf(stderr, "ds41_mmq_dense_pair_typed: unsupported type %d\n", type);
      return -1;
  }
}

extern "C" int ds41_mmq_dense_grouped_typed(int type, const void* W, const float* X,
                                           float* out, int M, int N, int K, int groups,
                                           hipStream_t stream) {
  switch ((ggml_type)type) {
#define DS41_CASE(T) \
  case T:            \
    return ds41_mmq_dense_impl<T>("ds41_mmq_dense_grouped_typed", W, X, out, M, N, K, stream, groups);
    DS41_MMQ_BATCH_CASES(DS41_CASE)
#undef DS41_CASE
    default:
      fprintf(stderr, "ds41_mmq_dense_grouped_typed: unsupported type %d\n", type);
      return -1;
  }
}

extern "C" int ds41_mmq_moe_typed(int type, const void* W, const float* X,
                                  const int32_t* ids, float* out, int M, int K,
                                  int n_tokens, int n_experts,
                                  int n_expert_used, hipStream_t stream) {
  switch ((ggml_type)type) {
#define DS41_CASE(T)                                                                \
  case T:                                                                           \
    return ds41_mmq_moe_impl<T>("ds41_mmq_moe_typed", W, X, ids, out, M, K, n_tokens, \
                                n_experts, n_expert_used, stream);
    DS41_MMQ_BATCH_CASES(DS41_CASE)
#undef DS41_CASE
    default:
      fprintf(stderr, "ds41_mmq_moe_typed: unsupported type %d\n", type);
      return -1;
  }
}

extern "C" int ds41_mmq_moe_swiglu_typed(int type, const void* W_gate, const void* W_up,
                                         const float* X, const int32_t* ids,
                                         const float* router_weights, float* gate,
                                         float* mid, int M, int K, int n_tokens,
                                         int n_experts, int n_expert_used, float clamp,
                                         hipStream_t stream) {
  const ds41_mmq_swiglu_epilogue epilogue = {router_weights, mid, nullptr, clamp};
  switch ((ggml_type)type) {
#define DS41_CASE(T)                                                                 \
  case T:                                                                            \
    return ds41_mmq_moe_pair_impl<T>("ds41_mmq_moe_swiglu_typed", W_gate, W_up, X, ids, \
                                     gate, nullptr, M, K, n_tokens, n_experts,       \
                                     n_expert_used, stream, nullptr, nullptr, 0,     \
                                     /*sanitize_out=*/false, nullptr, &epilogue);
    DS41_MMQ_BATCH_CASES(DS41_CASE)
#undef DS41_CASE
    default:
      fprintf(stderr, "ds41_mmq_moe_swiglu_typed: unsupported type %d\n", type);
      return -1;
  }
}

// Explicit instantiations. One per quant type the public API exposes.
// Each instantiation drags in the load_tiles_<type> + vec_dot_<type>_*
// device functions from mmq.hip.hpp, so the .o objects below contain everything
// needed to link against the public C entries.
template void mul_mat_q_case<GGML_TYPE_Q8_0>(
    ds41_ggml_hip_context & ctx, const mmq_args & args, hipStream_t stream);
template void mul_mat_q_case<GGML_TYPE_Q2_K>(
    ds41_ggml_hip_context & ctx, const mmq_args & args, hipStream_t stream);
template void mul_mat_q_case<GGML_TYPE_IQ2_XXS>(
    ds41_ggml_hip_context & ctx, const mmq_args & args, hipStream_t stream);
template void mul_mat_q_case<GGML_TYPE_Q4_K>(
    ds41_ggml_hip_context & ctx, const mmq_args & args, hipStream_t stream);
template void mul_mat_q_case<GGML_TYPE_MXFP4>(
    ds41_ggml_hip_context & ctx, const mmq_args & args, hipStream_t stream);
