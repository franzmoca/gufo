// Prefill attention for Gemma 4 on the gfx1151 WMMA matrix cores.
//
// Tiling. A block of eight wave32 waves covers kRowBlocks row blocks of 16
// query rows each; kHeads heads sharing one KV head times kQueryBlocks
// query blocks form those rows, so every staged K/V tile serves all of them.
// Each row block is split across kWavesPerRow waves by head dimension: a
// wave keeps a 128-dim slice of its rows' Q fragments and O accumulators in
// registers (64 + 64 VGPRs for both hd256 and hd512).
//
// Per 16-key tile: stage K ([key][dim]) and V transposed ([dim][key]) in
// LDS, reading the ring slot of each key for sliding layers; every wave adds
// its slice's partial scores to LDS; one wave per row block runs the online
// softmax over the summed scores (window, causal and key-limit masks per
// element) and writes the binary16 probabilities; every wave rescales its O
// slice and accumulates P V.
//
// Fragment layout (wave32 v_wmma_f32_16x16x16_f16): A holds row L%16 and 16
// contiguous k, B holds column L%16 and 16 contiguous k, and C element i is
// row 2i + L/16, column L%16; both half-waves carry the same operands.
#include "src/models/gemma4/kernels/rocm/attention_wmma.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>

namespace gufo::models::gemma4::rocm {
namespace {

using v16h = __attribute__((__vector_size__(16 * sizeof(_Float16)))) _Float16;
using v8f = __attribute__((__vector_size__(8 * sizeof(float)))) float;

constexpr std::uint32_t kKeys = 16;
constexpr std::uint32_t kSlice = 128;  // head dims per wave
constexpr std::uint32_t kSliceSteps = kSlice / 16;

__device__ __forceinline__ v8f Wmma(v16h a, v16h b, v8f c) {
  return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
}

__device__ __forceinline__ v16h LoadFrag(const __half* p) {
  union {
    v16h f;
    uint4 u[2];
  } cvt;
  cvt.u[0] = *reinterpret_cast<const uint4*>(p);
  cvt.u[1] = *reinterpret_cast<const uint4*>(p + 8);
  return cvt.f;
}

struct Bounds {
  std::uint32_t lo;
  std::uint32_t hi;
};

__device__ inline Bounds KeyBounds(const AttentionArgs& a,
                                   std::uint32_t position) {
  const std::uint32_t hi = min(position + 1, a.key_limit);
  std::uint32_t lo = 0;
  if (a.window != 0 && position + 1 > a.window) {
    lo = position + 1 - a.window;
  }
  return {min(lo, hi), hi};
}

template<std::uint32_t D, std::uint32_t kHeads, std::uint32_t kQueryBlocks>
__global__ void __launch_bounds__(256)
    WmmaPrefillAttentionKernel(AttentionArgs a) {
  constexpr std::uint32_t kWavesPerRow = D / kSlice;
  constexpr std::uint32_t kRowBlocks = kHeads * kQueryBlocks;
  static_assert(kRowBlocks * kWavesPerRow == 8, "eight waves per block");
  constexpr std::uint32_t kKStride = D + 8;       // K rows, halves
  constexpr std::uint32_t kVtStride = kKeys + 8;  // V^T rows, halves
  constexpr std::uint32_t kQueries = kQueryBlocks * 16;

  __shared__ __align__(16) __half k_lds[kKeys * kKStride];
  __shared__ __align__(16) __half vt_lds[D * kVtStride];
  __shared__ float s_lds[kRowBlocks][kWavesPerRow][16][17];
  __shared__ __align__(16) __half p_lds[kRowBlocks][16][kKeys + 8];
  __shared__ float row_scale[kRowBlocks][16];
  __shared__ float row_sum[kRowBlocks][16];

  const std::uint32_t tid = threadIdx.x;
  const std::uint32_t lane = tid & 31U;
  const std::uint32_t wave = tid >> 5U;
  const std::uint32_t sub = lane & 15U;
  const std::uint32_t half_id = lane >> 4U;
  const std::uint32_t rb = wave / kWavesPerRow;
  const std::uint32_t part = wave % kWavesPerRow;
  const std::uint32_t dim0 = part * kSlice;

  const std::uint32_t query_start = blockIdx.x * kQueries;
  const std::uint32_t head_base = blockIdx.y * kHeads;
  const std::uint32_t gqa = a.heads / a.kv_heads;
  const std::uint32_t kv_head = head_base / gqa;
  const std::uint32_t head = head_base + rb % kHeads;
  const std::uint32_t row0 = query_start + (rb / kHeads) * 16;  // this rb

  const auto position_of = [&](std::uint32_t row) {
    return a.shared_position ? a.first_position : a.first_position + row;
  };
  // Keys the whole block may need.
  const std::uint32_t last_row = min(query_start + kQueries, a.rows) - 1;
  const std::uint32_t block_lo =
      KeyBounds(a, position_of(query_start)).lo & ~(kKeys - 1);
  const std::uint32_t block_hi = KeyBounds(a, position_of(last_row)).hi;

  // Q slice fragments: row `sub` of this row block, dims [dim0, dim0 + 128).
  v16h q_frag[kSliceSteps];
  {
    const std::uint32_t row = row0 + sub;
    const bool live = row < a.rows;
    const float* q =
        a.q + (static_cast<std::size_t>(row) * a.heads + head) * D + dim0;
#pragma unroll
    for (std::uint32_t ks = 0; ks < kSliceSteps; ++ks) {
#pragma unroll
      for (std::uint32_t v = 0; v < 16; v += 4) {
        const float4 f = live
                             ? *reinterpret_cast<const float4*>(q + ks * 16 + v)
                             : make_float4(0.0F, 0.0F, 0.0F, 0.0F);
        q_frag[ks][v + 0] = static_cast<_Float16>(f.x);
        q_frag[ks][v + 1] = static_cast<_Float16>(f.y);
        q_frag[ks][v + 2] = static_cast<_Float16>(f.z);
        q_frag[ks][v + 3] = static_cast<_Float16>(f.w);
      }
    }
  }
  v8f o_acc[kSliceSteps] = {};

  // Online softmax state, owned by lane pairs of the part-0 wave: lane l
  // covers row l / 2 and keys [8 (l % 2), 8 (l % 2) + 8) of each tile.
  float running_max = -INFINITY;
  float running_sum = 0.0F;
  const std::uint32_t sm_row = lane >> 1U;
  const std::uint32_t sm_seg = lane & 1U;
  const std::uint32_t sm_query = row0 + sm_row;
  const Bounds sm_keys = KeyBounds(a, position_of(min(sm_query, a.rows - 1)));

  const std::size_t kv_stride = static_cast<std::size_t>(a.kv_heads) * D;
  const auto* k_cache = reinterpret_cast<const __half*>(a.k_cache);
  const auto* v_cache = reinterpret_cast<const __half*>(a.v_cache);
  const auto slot_of = [&](std::uint32_t key) {
    return a.ring != 0 ? key % a.ring : key;
  };

  for (std::uint32_t key0 = block_lo; key0 < block_hi; key0 += kKeys) {
    __syncthreads();
    // Stage K row-major and V transposed; lane-major keys keep the V^T
    // writes on distinct banks.
    for (std::uint32_t idx = tid; idx < kKeys * (D / 8); idx += 256) {
      const std::uint32_t key = idx % kKeys;
      const std::uint32_t d8 = (idx / kKeys) * 8;
      const std::uint32_t position = key0 + key;
      uint4 kv = make_uint4(0U, 0U, 0U, 0U);
      uint4 vv = make_uint4(0U, 0U, 0U, 0U);
      if (position < block_hi) {
        const std::size_t at =
            static_cast<std::size_t>(slot_of(position)) * kv_stride +
            static_cast<std::size_t>(kv_head) * D + d8;
        kv = *reinterpret_cast<const uint4*>(k_cache + at);
        vv = *reinterpret_cast<const uint4*>(v_cache + at);
      }
      *reinterpret_cast<uint4*>(&k_lds[key * kKStride + d8]) = kv;
      const auto* vh = reinterpret_cast<const __half*>(&vv);
#pragma unroll
      for (std::uint32_t i = 0; i < 8; ++i) {
        vt_lds[(d8 + i) * kVtStride + key] = vh[i];
      }
    }
    __syncthreads();

    // Partial scores over this wave's dim slice.
    {
      v8f s_acc = {};
#pragma unroll
      for (std::uint32_t ks = 0; ks < kSliceSteps; ++ks) {
        const v16h k_frag = LoadFrag(&k_lds[sub * kKStride + dim0 + ks * 16]);
        s_acc = Wmma(q_frag[ks], k_frag, s_acc);
      }
#pragma unroll
      for (std::uint32_t i = 0; i < 8; ++i) {
        s_lds[rb][part][2 * i + half_id][sub] = s_acc[i];
      }
    }
    __syncthreads();

    if (part == 0) {
      const std::uint32_t position_row = sm_query;
      float vals[8];
      float tile_max = -INFINITY;
#pragma unroll
      for (std::uint32_t m = 0; m < 8; ++m) {
        const std::uint32_t col = sm_seg * 8 + m;
        const std::uint32_t key = key0 + col;
        float s = 0.0F;
#pragma unroll
        for (std::uint32_t p = 0; p < kWavesPerRow; ++p) {
          s += s_lds[rb][p][sm_row][col];
        }
        const bool valid =
            position_row < a.rows && key >= sm_keys.lo && key < sm_keys.hi;
        vals[m] = valid ? s : -INFINITY;
        tile_max = fmaxf(tile_max, vals[m]);
      }
      tile_max = fmaxf(tile_max, __shfl_xor(tile_max, 1, 32));
      const float next_max = fmaxf(running_max, tile_max);
      const float scale =
          next_max == -INFINITY ? 1.0F : expf(running_max - next_max);
      float tile_sum = 0.0F;
#pragma unroll
      for (std::uint32_t m = 0; m < 8; ++m) {
        const float w = vals[m] == -INFINITY ? 0.0F : expf(vals[m] - next_max);
        tile_sum += w;
        p_lds[rb][sm_row][sm_seg * 8 + m] = __float2half(w);
      }
      tile_sum += __shfl_xor(tile_sum, 1, 32);
      running_max = next_max;
      running_sum = running_sum * scale + tile_sum;
      if (sm_seg == 0) {
        row_scale[rb][sm_row] = scale;
      }
    }
    __syncthreads();

    float scale[8];
#pragma unroll
    for (std::uint32_t i = 0; i < 8; ++i) {
      scale[i] = row_scale[rb][2 * i + half_id];
    }
    const v16h p_frag = LoadFrag(&p_lds[rb][sub][0]);
#pragma unroll
    for (std::uint32_t t = 0; t < kSliceSteps; ++t) {
#pragma unroll
      for (std::uint32_t i = 0; i < 8; ++i) {
        o_acc[t][i] *= scale[i];
      }
      const v16h v_frag = LoadFrag(&vt_lds[(dim0 + t * 16 + sub) * kVtStride]);
      o_acc[t] = Wmma(p_frag, v_frag, o_acc[t]);
    }
  }

  if (part == 0 && sm_seg == 0) {
    row_sum[rb][sm_row] = running_sum;
  }
  __syncthreads();
#pragma unroll
  for (std::uint32_t i = 0; i < 8; ++i) {
    const std::uint32_t r = 2 * i + half_id;
    const std::uint32_t row = row0 + r;
    if (row >= a.rows) {
      continue;
    }
    const float l = row_sum[rb][r];
    const float inv = l > 0.0F ? 1.0F / l : 0.0F;
    float* out = a.out + (static_cast<std::size_t>(row) * a.heads + head) * D +
                 dim0 + sub;
#pragma unroll
    for (std::uint32_t t = 0; t < kSliceSteps; ++t) {
      out[t * 16] = o_acc[t][i] * inv;
    }
  }
}

}  // namespace

bool LaunchWmmaPrefillAttention(const AttentionArgs& a, hipStream_t stream) {
  const std::uint32_t gqa = a.heads / a.kv_heads;
  if (gqa % 2 != 0 || a.heads % 2 != 0) {
    return false;
  }
  if (a.head_dim == 256) {
    // Two heads x two 16-query blocks, two waves per row block.
    const dim3 grid((a.rows + 31) / 32, a.heads / 2);
    WmmaPrefillAttentionKernel<256, 2, 2><<<grid, 256, 0, stream>>>(a);
    return true;
  }
  if (a.head_dim == 512) {
    // Two heads x one 16-query block, four waves per row block.
    const dim3 grid((a.rows + 15) / 16, a.heads / 2);
    WmmaPrefillAttentionKernel<512, 2, 1><<<grid, 256, 0, stream>>>(a);
    return true;
  }
  return false;
}

}  // namespace gufo::models::gemma4::rocm
