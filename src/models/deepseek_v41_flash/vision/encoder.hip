#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <openssl/evp.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cmath>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/core/hip/host_allocations.hpp"
#include "src/core/hip/managed_alloc.hpp"
#include "src/models/deepseek_v41_flash/vision/encoder.hpp"
#include "src/models/qwen/hip/ops/gemm.hpp"

namespace gufo::models::deepseek_v41_flash::vision {
namespace {
using Bf16 = hip_bfloat16;
constexpr unsigned kHidden = 1024;
constexpr unsigned kHeads = 16;
constexpr unsigned kHeadDim = kHidden / kHeads;
constexpr unsigned kFfn = 2816;
constexpr unsigned kLayers = 32;
constexpr unsigned kPatchDim = 3 * kPatchSize * kPatchSize;
constexpr unsigned kMergeDim = kHidden * kDownsample * kDownsample;
/// Rotary frequencies per axis: half of the head's rotated half.
constexpr unsigned kRopeFreqs = kHeadDim / 4;
constexpr unsigned kThreads = 256;
constexpr unsigned kTensors = 298;

void Check(hipError_t error) {
  if (error != hipSuccess)
    throw std::runtime_error(hipGetErrorString(error));
}
void CheckBlas(hipblasStatus_t status) {
  if (status != HIPBLAS_STATUS_SUCCESS) {
    throw std::runtime_error("DeepSeek V4.1 vision hipBLAS failure: " +
                             std::to_string(status));
  }
}

unsigned Blocks(std::size_t n) {
  return static_cast<unsigned>((n + kThreads - 1) / kThreads);
}

struct Allocation {
  void* data{nullptr};
  Allocation() = default;
  explicit Allocation(std::size_t bytes) { Check(hipMalloc(&data, bytes)); }
  static Allocation Weight(std::size_t bytes) {
    // The device heap when its reserve permits: movable managed pages could
    // evict text inference queues while no image request runs.
    Allocation allocation;
    Check(gufo::hip::AllocateDevice(&allocation.data, bytes));
    return allocation;
  }
  ~Allocation() {
    if (data != nullptr)
      (void)hipFree(data);
  }
  Allocation(const Allocation&) = delete;
  Allocation& operator=(const Allocation&) = delete;
  Allocation(Allocation&& other) noexcept
      : data(std::exchange(other.data, nullptr)) {}
  Allocation& operator=(Allocation&& other) noexcept {
    if (this != &other) {
      if (data != nullptr)
        (void)hipFree(data);
      data = std::exchange(other.data, nullptr);
    }
    return *this;
  }
  template<class T>
  T* As() const {
    return static_cast<T*>(data);
  }
};

// Copies through one bounded pinned buffer: pageable async copies would leave
// the sidecar's pages registered in ROCr's SVM cache.
class PinnedUpload {
public:
  explicit PinnedUpload(hipStream_t stream, std::size_t bytes = kBytes)
      : capacity_(std::min(bytes, kBytes)), stream_(stream) {
    Check(allocations_.Allocate(&data_, capacity_));
  }
  PinnedUpload(const PinnedUpload&) = delete;
  PinnedUpload& operator=(const PinnedUpload&) = delete;

  void Copy(void* destination, const void* source, std::size_t bytes) {
    auto* out = static_cast<std::byte*>(destination);
    const auto* in = static_cast<const std::byte*>(source);
    for (std::size_t offset = 0; offset < bytes;) {
      const auto count = std::min(capacity_, bytes - offset);
      std::memcpy(data_, in + offset, count);
      Check(hipMemcpyAsync(out + offset, data_, count, hipMemcpyHostToDevice,
                           stream_));
      Check(hipStreamSynchronize(stream_));
      offset += count;
    }
  }

private:
  static constexpr std::size_t kBytes = 16ULL << 20;
  gufo::hip::HostAllocations allocations_;
  void* data_{nullptr};
  std::size_t capacity_;
  hipStream_t stream_;
};

__device__ __forceinline__ float Round(float x) {
  return static_cast<float>(Bf16(x));
}

__device__ float BlockSum(float value, float* scratch) {
  scratch[threadIdx.x] = value;
  __syncthreads();
  for (unsigned stride = kThreads / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride)
      scratch[threadIdx.x] += scratch[threadIdx.x + stride];
    __syncthreads();
  }
  const float sum = scratch[0];
  __syncthreads();
  return sum;
}

// load_image's normalization and patch order: patch (y, x) holds
// [channel][row][column] of its 14x14 pixels.
__global__ void Patchify(const std::uint8_t* pixels, Bf16* patches,
                         unsigned width, unsigned grid_width, unsigned count) {
  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count * kPatchDim)
    return;
  const unsigned patch = i / kPatchDim;
  const unsigned element = i % kPatchDim;
  const unsigned channel = element / (kPatchSize * kPatchSize);
  const unsigned row = element / kPatchSize % kPatchSize;
  const unsigned column = element % kPatchSize;
  const unsigned y = patch / grid_width * kPatchSize + row;
  const unsigned x = patch % grid_width * kPatchSize + column;
  const float value =
      static_cast<float>(pixels[(std::size_t{y} * width + x) * 3 + channel]) /
      255.0F;
  patches[i] = Bf16((value - 0.5F) / 0.5F);
}

// A BF16 linear output: its FP32 accumulation plus bias, rounded once.
__global__ void BiasRound(float* output, const float* bias, unsigned size,
                          unsigned width) {
  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size)
    output[i] = Round(output[i] + (bias != nullptr ? bias[i % width] : 0.0F));
}

// x + linear(...), both BF16.
__global__ void Residual(float* hidden, const float* projection,
                         const float* bias, unsigned size, unsigned width) {
  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
    const float out =
        Round(projection[i] + (bias != nullptr ? bias[i % width] : 0.0F));
    hidden[i] = Round(hidden[i] + out);
  }
}

__global__ void RmsNorm(const float* input, Bf16* output, const float* weight) {
  __shared__ float scratch[kThreads];
  const float* row = input + std::size_t{blockIdx.x} * kHidden;
  float sum = 0;
  for (unsigned j = threadIdx.x; j < kHidden; j += kThreads)
    sum += row[j] * row[j];
  const float scale = rsqrtf(BlockSum(sum, scratch) / kHidden + 1e-6F);
  for (unsigned j = threadIdx.x; j < kHidden; j += kThreads)
    output[std::size_t{blockIdx.x} * kHidden + j] =
        Bf16(weight[j] * (row[j] * scale));
}

// wqkv's BF16 output split into head-major q, k [head][padded][64] and
// transposed v [head][64][padded], the layouts FusedAttention reads; q and k
// take the 2D rotary embedding (apply_rotary) over the whole head: dimension
// d pairs with d + 32, whose frequency j < 16 follows the patch row and
// j >= 16 its column.
__global__ void QkvRope(const float* input, const float* bias,
                        const float* inv_freq, Bf16* q, Bf16* k, Bf16* vt,
                        unsigned count, unsigned padded, unsigned grid_width) {
  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count * kHidden)
    return;
  const unsigned token = i / kHidden;
  const unsigned head = i % kHidden / kHeadDim;
  const unsigned d = i % kHeadDim;
  const unsigned pair = d % (kHeadDim / 2);
  const unsigned position =
      pair < kRopeFreqs ? token / grid_width : token % grid_width;
  const float angle =
      static_cast<float>(position) * inv_freq[pair % kRopeFreqs];
  const float cosine = cosf(angle);
  const float sine = sinf(angle);
  const unsigned first = head * kHeadDim + pair;
  const unsigned second = first + kHeadDim / 2;
  const float* row = input + std::size_t{token} * 3 * kHidden;
  const auto value = [&](unsigned column) {
    return Round(row[column] + bias[column]);
  };
  const float q0 = value(first), q1 = value(second);
  const float k0 = value(kHidden + first), k1 = value(kHidden + second);
  const std::size_t target =
      (std::size_t{head} * padded + token) * kHeadDim + d;
  const bool low = d < kHeadDim / 2;
  q[target] = Bf16(low ? q0 * cosine - q1 * sine : q1 * cosine + q0 * sine);
  k[target] = Bf16(low ? k0 * cosine - k1 * sine : k1 * cosine + k0 * sine);
  vt[(std::size_t{head} * kHeadDim + d) * padded + token] =
      Bf16(value(2 * kHidden + head * kHeadDim + d));
}

using Bf16x16 = __attribute__((__vector_size__(16 * sizeof(short)))) short;
using Float8 = __attribute__((__vector_size__(8 * sizeof(float)))) float;
constexpr unsigned kDimSteps = kHeadDim / 16;
constexpr unsigned kAttentionWaves = 8;
constexpr unsigned kAttentionQueries = 16 * kAttentionWaves;
constexpr unsigned kKeyTile = 64;

__device__ __forceinline__ Bf16x16 LoadBf16x16(const Bf16* p) {
  union {
    Bf16x16 h;
    uint4 u[2];
  } value;
  value.u[0] = *reinterpret_cast<const uint4*>(p);
  value.u[1] = *reinterpret_cast<const uint4*>(p + 8);
  return value.h;
}

// BF16 bits rounded to nearest even, without the conversion's branches; NaN
// stays quiet NaN.
__device__ __forceinline__ unsigned Bf16Bits(float x) {
  const unsigned bits = __float_as_uint(x);
  const unsigned rounded = (bits + 0x7fffU + ((bits >> 16) & 1U)) >> 16;
  return x != x ? (bits >> 16) | 0x40U : rounded;
}

/// Bidirectional attention with an online softmax, as the fused kernels of
/// scaled_dot_product_attention: each wave owns 16 queries of one head and
/// walks 64-key tiles (four 16-key WMMA blocks per softmax update), which
/// the block's waves stage in LDS together. BF16 scores on the matrix cores, FP32 scaled scores and softmax state, BF16
/// probabilities times V accumulated in FP32. The wave computes S^T = K Q^T
/// and O^T = V^T P^T (wave32 v_wmma_f32_16x16x16_bf16: A holds row L%16 and
/// 16 contiguous k, B column L%16 and 16 contiguous k, and accumulator
/// element i is row 2i + L/16, column L%16), so lane L keeps query L%16's
/// softmax state and half its keys in registers; the half-waves swap their
/// probabilities for the B operand. Writes token-major BF16 rows.
/// `padded` is a multiple of kAttentionQueries and kKeyTile; keys past
/// `count` are masked, and their padded V columns must be finite.
__global__ void __launch_bounds__(32 * kAttentionWaves)
    FusedAttention(const Bf16* q, const Bf16* k, const Bf16* vt, Bf16* out,
                   unsigned count, unsigned padded) {
  constexpr float kScale = 0.125F;  // 1 / sqrt(head_dim)
  constexpr unsigned kBlocks = kKeyTile / 16;
  const unsigned lane = threadIdx.x % 32;
  const unsigned wave = threadIdx.x / 32;
  const unsigned head = blockIdx.y;
  // Waves past the last query still stage tiles; padded rows are zero.
  const unsigned q0 = (blockIdx.x * kAttentionWaves + wave) * 16;
  constexpr unsigned kPitch = kKeyTile + 8;  // shifts banks between rows
  static_assert(kHeadDim == kKeyTile);
  __shared__ __align__(16) Bf16 k_lds[kKeyTile][kPitch];
  __shared__ __align__(16) Bf16 v_lds[kHeadDim][kPitch];
  const unsigned row = lane % 16;
  const unsigned half = lane / 16;
  const Bf16* qh = q + std::size_t{head} * padded * kHeadDim;
  const Bf16* kh = k + std::size_t{head} * padded * kHeadDim;
  const Bf16* vh = vt + std::size_t{head} * kHeadDim * padded;
  Bf16x16 query[kDimSteps];
#pragma unroll
  for (unsigned s = 0; s < kDimSteps; ++s)
    query[s] = LoadBf16x16(qh + std::size_t{q0 + row} * kHeadDim + s * 16);
  Float8 acc[kDimSteps] = {};
  float m = -INFINITY, l = 0.0F;
  for (unsigned key0 = 0; key0 < count; key0 += kKeyTile) {
    __syncthreads();
    for (unsigned i = threadIdx.x; i < kKeyTile * kHeadDim / 8;
         i += 32 * kAttentionWaves) {
      const unsigned r = i / (kKeyTile / 8);
      const unsigned c = i % (kKeyTile / 8) * 8;
      *reinterpret_cast<uint4*>(&k_lds[r][c]) = *reinterpret_cast<const uint4*>(
          kh + std::size_t{key0 + r} * kHeadDim + c);
      *reinterpret_cast<uint4*>(&v_lds[r][c]) = *reinterpret_cast<const uint4*>(
          vh + std::size_t{r} * padded + key0 + c);
    }
    __syncthreads();
    Float8 score[kBlocks] = {};
#pragma unroll
    for (unsigned b = 0; b < kBlocks; ++b) {
#pragma unroll
      for (unsigned s = 0; s < kDimSteps; ++s) {
        score[b] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(
            LoadBf16x16(&k_lds[b * 16 + row][s * 16]), query[s], score[b]);
      }
    }
    float maximum = -INFINITY;
#pragma unroll
    for (unsigned b = 0; b < kBlocks; ++b) {
#pragma unroll
      for (unsigned i = 0; i < 8; ++i) {
        score[b][i] = key0 + b * 16 + 2 * i + half < count
                          ? score[b][i] * kScale
                          : -INFINITY;
        maximum = fmaxf(maximum, score[b][i]);
      }
    }
    maximum = fmaxf(maximum, __shfl_xor(maximum, 16, 32));
    const float next = fmaxf(m, maximum);
    const float alpha = expf(m - next);
    float sum = 0.0F;
#pragma unroll
    for (unsigned b = 0; b < kBlocks; ++b) {
      // Key 2i + half of the block, as BF16 pairs (keys 4t + half, 4t + 2 +
      // half); the other half-wave holds the remaining keys.
      unsigned mine[4];
#pragma unroll
      for (unsigned t = 0; t < 4; ++t) {
        const float p0 = expf(score[b][2 * t] - next);
        const float p1 = expf(score[b][2 * t + 1] - next);
        sum += p0;
        sum += p1;
        mine[t] = Bf16Bits(p0) | Bf16Bits(p1) << 16;
      }
      union {
        Bf16x16 h;
        unsigned u[8];
      } probabilities;
#pragma unroll
      for (unsigned t = 0; t < 4; ++t) {
        const unsigned other = __builtin_amdgcn_permlanex16(
            mine[t], mine[t], 0x76543210, 0xfedcba98, false, false);
        const unsigned even = half == 0 ? mine[t] : other;
        const unsigned odd = half == 0 ? other : mine[t];
        probabilities.u[2 * t] = __builtin_amdgcn_perm(odd, even, 0x05040100);
        probabilities.u[2 * t + 1] =
            __builtin_amdgcn_perm(odd, even, 0x07060302);
      }
#pragma unroll
      for (unsigned c = 0; c < kDimSteps; ++c) {
        if (b == 0) {
#pragma unroll
          for (unsigned i = 0; i < 8; ++i)
            acc[c][i] *= alpha;
        }
        acc[c] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(
            LoadBf16x16(&v_lds[c * 16 + row][b * 16]), probabilities.h,
            acc[c]);
      }
    }
    sum += __shfl_xor(sum, 16, 32);
    l = l * alpha + sum;
    m = next;
  }
  const float inverse = 1.0F / l;
  const unsigned token = q0 + row;
  if (token >= count)
    return;
  Bf16* const target = out + std::size_t{token} * kHidden + head * kHeadDim;
#pragma unroll
  for (unsigned c = 0; c < kDimSteps; ++c) {
#pragma unroll
    for (unsigned i = 0; i < 8; ++i)
      target[c * 16 + 2 * i + half] = Bf16(acc[c][i] * inverse);
  }
}

// w2(silu(gate) * up), each BF16 op rounded.
__global__ void SwiGlu(const float* gate, const float* up, Bf16* output,
                       unsigned size) {
  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= size)
    return;
  const float g = Round(gate[i]);
  const float silu = Round(g / (1.0F + expf(-g)));
  output[i] = Bf16(silu * Round(up[i]));
}

// The aligner's F.pad and F.unfold: token (y, x) of the 3x3-merged grid
// takes channel c of patch (3y + i, 3x + j) at column c * 9 + i * 3 + j;
// patches past the grid are zero.
__global__ void Unfold(const Bf16* input, Bf16* output, unsigned grid_height,
                       unsigned grid_width, unsigned merged_width,
                       unsigned count) {
  const std::size_t i = std::size_t{blockIdx.x} * blockDim.x + threadIdx.x;
  if (i >= std::size_t{count} * kMergeDim)
    return;
  const unsigned token = static_cast<unsigned>(i / kMergeDim);
  const unsigned column = static_cast<unsigned>(i % kMergeDim);
  const unsigned channel = column / (kDownsample * kDownsample);
  const unsigned tap = column % (kDownsample * kDownsample);
  const unsigned y = token / merged_width * kDownsample + tap / kDownsample;
  const unsigned x = token % merged_width * kDownsample + tap % kDownsample;
  output[i] = y < grid_height && x < grid_width
                  ? input[(std::size_t{y} * grid_width + x) * kHidden + channel]
                  : Bf16(0.0F);
}

__global__ void Gelu(const float* input, const float* bias, Bf16* output,
                     unsigned size, unsigned width) {
  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= size)
    return;
  const float x = Round(input[i] + bias[i % width]);
  output[i] = Bf16(0.5F * x * (1.0F + erff(x * 0.7071067811865475F)));
}

// merge_image_embeddings over one span: the delimiters' learned rows and the
// aligner rows in reading order, one IMAGE_NEW_LINE after each grid row.
__global__ void Span(const float* aligned, const float* bias,
                     const float* start, const float* newline, const float* end,
                     float* output, unsigned llm_width, unsigned rows) {
  const std::size_t i = std::size_t{blockIdx.x} * blockDim.x + threadIdx.x;
  if (i >= std::size_t{rows} * kEmbeddingWidth)
    return;
  const unsigned row = static_cast<unsigned>(i / kEmbeddingWidth);
  const unsigned column = static_cast<unsigned>(i % kEmbeddingWidth);
  float value;
  if (row == 0) {
    value = start[column];
  } else if (row == rows - 1) {
    value = end[column];
  } else {
    const unsigned grid_row = (row - 1) / (llm_width + 1);
    const unsigned slot = (row - 1) % (llm_width + 1);
    value = slot == llm_width
                ? newline[column]
                : aligned[(std::size_t{grid_row} * llm_width + slot) *
                              kEmbeddingWidth +
                          column] +
                      bias[column];
  }
  output[i] = Round(value);
}

std::uint16_t ToBf16(float value) {
  const auto bits = std::bit_cast<std::uint32_t>(value);
  return static_cast<std::uint16_t>((bits + 0x7fff + ((bits >> 16) & 1)) >> 16);
}

}  // namespace

struct Encoder::Impl {
  std::unique_ptr<core::GgufReader> reader;
  std::string identity;
  std::once_flag identity_once;
  std::atomic<std::size_t> resident_bytes{0};
  std::unordered_map<std::string, Allocation> weights;
  Allocation inv_freq;
  hipStream_t stream{nullptr};
  hipblasHandle_t blas{nullptr};
  std::unique_ptr<gufo::hip::HipblasLtGemm> lt;
  bool uploaded{false};

  explicit Impl(const std::filesystem::path& path) {
    std::string error;
    reader = core::GgufReader::OpenFile(path, &error);
    if (!reader)
      throw std::invalid_argument(error);
    const auto expect = [&](std::string_view key, std::uint64_t value) {
      if (reader->GetMetadataUint64(key) != value) {
        throw std::invalid_argument(
            "vision sidecar does not match DeepSeek V4.1 Flash: " +
            std::string(key));
      }
    };
    if (reader->GetMetadataString("general.architecture") != "clip" ||
        reader->GetMetadataString("clip.projector_type") != "deepseek41v" ||
        reader->GetMetadataBool("clip.has_vision_encoder") != true) {
      throw std::invalid_argument(
          "vision sidecar is not a DeepSeek V4.1 (deepseek41v) projector");
    }
    expect("clip.vision.projection_dim", kEmbeddingWidth);
    expect("clip.vision.patch_size", kPatchSize);
    expect("clip.vision.embedding_length", kHidden);
    expect("clip.vision.feed_forward_length", kFfn);
    expect("clip.vision.block_count", kLayers);
    expect("clip.vision.attention.head_count", kHeads);
    expect("clip.vision.projector.scale_factor", kDownsample);
    expect("clip.vision.image_min_pixels", kMinPixels);
    if (reader->GetMetadataFloat32(
            "clip.vision.attention.layer_norm_epsilon") != 1e-6F) {
      throw std::invalid_argument("unexpected DeepSeek V4.1 vision epsilon");
    }
    for (const auto key : {"clip.vision.image_mean", "clip.vision.image_std"}) {
      const auto* value = reader->FindMetadata(key);
      const auto* array =
          value ? std::get_if<std::vector<double>>(&value->value) : nullptr;
      if (!array || *array != std::vector<double>{0.5, 0.5, 0.5})
        throw std::invalid_argument("unsupported DeepSeek V4.1 normalization");
    }
    if (reader->GetMetadataString("deepseek41v.image_routing_bias.source")) {
      throw std::invalid_argument(
          "this vision sidecar carries the image routing bias, which the "
          "model now provides; pass the mmproj itself");
    }
    const auto require = [&](const std::string& name,
                             std::initializer_list<std::uint64_t> shape,
                             core::GgmlType type) {
      const auto* tensor = reader->FindTensor(name);
      if (tensor == nullptr ||
          tensor->dimensions != std::vector<std::uint64_t>(shape) ||
          tensor->type != type) {
        throw std::invalid_argument("invalid DeepSeek V4.1 vision tensor: " +
                                    name);
      }
      const auto bytes =
          tensor->ElementCount() * (type == core::GgmlType::kF32 ? 4 : 2);
      const auto address = reinterpret_cast<std::uintptr_t>(tensor->data);
      const bool mapped = std::any_of(
          reader->GetMappedRegions().begin(), reader->GetMappedRegions().end(),
          [&](const auto& region) {
            const auto begin = reinterpret_cast<std::uintptr_t>(region.data);
            return address >= begin && address - begin <= region.size &&
                   bytes <= region.size - (address - begin);
          });
      if (!mapped)
        throw std::invalid_argument("truncated DeepSeek V4.1 vision tensor: " +
                                    name);
    };
    constexpr auto kF32 = core::GgmlType::kF32;
    constexpr auto kBF16 = core::GgmlType::kBF16;
    for (unsigned layer = 0; layer < kLayers; ++layer) {
      const auto prefix = "v.blk." + std::to_string(layer) + ".";
      require(prefix + "ln1.weight", {kHidden}, kF32);
      require(prefix + "ln2.weight", {kHidden}, kF32);
      require(prefix + "attn_qkv.weight", {kHidden, 3 * kHidden}, kBF16);
      require(prefix + "attn_qkv.bias", {3 * kHidden}, kF32);
      require(prefix + "attn_out.weight", {kHidden, kHidden}, kBF16);
      require(prefix + "attn_out.bias", {kHidden}, kF32);
      require(prefix + "ffn_gate.weight", {kHidden, kFfn}, kBF16);
      require(prefix + "ffn_up.weight", {kHidden, kFfn}, kBF16);
      require(prefix + "ffn_down.weight", {kFfn, kHidden}, kBF16);
    }
    require("v.patch_embd.weight", {kPatchSize, kPatchSize, 3, kHidden}, kF32);
    require("v.patch_embd.bias", {kHidden}, kF32);
    require("v.post_ln.weight", {kHidden}, kF32);
    require("mm.1.weight", {kMergeDim, kEmbeddingWidth}, kBF16);
    require("mm.1.bias", {kEmbeddingWidth}, kF32);
    require("mm.2.weight", {kEmbeddingWidth, kEmbeddingWidth}, kBF16);
    require("mm.2.bias", {kEmbeddingWidth}, kF32);
    for (const auto* name : {"v.token_embd.img_start", "v.token_embd.img_end",
                             "v.image_newline"})
      require(name, {kEmbeddingWidth}, kF32);
    if (reader->GetTensorCount() != kTensors) {
      throw std::invalid_argument(
          "unexpected tensors in DeepSeek V4.1 vision sidecar");
    }
  }

  ~Impl() {
    if (stream != nullptr)
      (void)hipStreamSynchronize(stream);
    weights.clear();
    lt.reset();
    if (blas != nullptr)
      (void)hipblasDestroy(blas);
    if (stream != nullptr)
      (void)hipStreamDestroy(stream);
  }

  void Upload() {
    if (uploaded)
      return;
    if (stream == nullptr)
      Check(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    if (blas == nullptr)
      CheckBlas(hipblasCreate(&blas));
    CheckBlas(hipblasSetStream(blas, stream));
    if (lt == nullptr)
      lt = std::make_unique<gufo::hip::HipblasLtGemm>(
          gufo::hip::HipblasLtGemmOptions{.ignore_environment = true});
    weights.clear();
    std::size_t bytes_uploaded = 0;
    PinnedUpload upload(stream);
    for (const auto& tensor : reader->GetTensors()) {
      // The patch embedding is converted below.
      if (tensor.name == "v.patch_embd.weight")
        continue;
      const std::size_t bytes =
          tensor.ElementCount() * (tensor.type == core::GgmlType::kF32 ? 4 : 2);
      Allocation buffer = Allocation::Weight(bytes);
      upload.Copy(buffer.data, tensor.data, bytes);
      bytes_uploaded += bytes;
      weights.emplace(tensor.name, std::move(buffer));
    }
    // The checkpoint's patch embedding is BF16; the sidecar widened it.
    const auto* patch = static_cast<const float*>(
        reader->FindTensor("v.patch_embd.weight")->data);
    std::vector<std::uint16_t> converted(std::size_t{kHidden} * kPatchDim);
    for (std::size_t i = 0; i < converted.size(); ++i)
      converted[i] = ToBf16(patch[i]);
    Allocation buffer =
        Allocation::Weight(converted.size() * sizeof(std::uint16_t));
    upload.Copy(buffer.data, converted.data(),
                converted.size() * sizeof(std::uint16_t));
    bytes_uploaded += converted.size() * sizeof(std::uint16_t);
    weights.emplace("patch", std::move(buffer));
    // vision.get_vision_cos_sin's inverse frequencies, in FP32.
    std::array<float, kRopeFreqs> freqs{};
    for (unsigned i = 0; i < kRopeFreqs; ++i) {
      freqs[i] = 1.0F / std::pow(10000.0F, static_cast<float>(2 * i) /
                                               (2 * kRopeFreqs));
    }
    inv_freq = Allocation(sizeof(freqs));
    upload.Copy(inv_freq.data, freqs.data(), sizeof(freqs));
    Check(hipStreamSynchronize(stream));
    resident_bytes.store(bytes_uploaded);
    uploaded = true;
  }

  const float* FloatWeight(const std::string& name) const {
    return weights.at(name).As<float>();
  }

  /// output[rows][out] = input[rows][in] W^T, FP32 accumulation and output.
  void Gemm(const std::string& weight, const Bf16* input, float* output,
            unsigned rows, unsigned out, unsigned in) {
    if (lt->RunBf16(weights.at(weight).data, input, output, rows, out, in,
                    stream))
      return;
    constexpr float alpha = 1, beta = 0;
    CheckBlas(hipblasGemmEx(blas, HIPBLAS_OP_T, HIPBLAS_OP_N, out, rows, in,
                            &alpha, weights.at(weight).data, HIP_R_16BF, in,
                            input, HIP_R_16BF, in, &beta, output, HIP_R_32F,
                            out, HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT));
  }
};

Encoder::Encoder(const std::filesystem::path& path)
    : impl_(std::make_unique<Impl>(path)) {}
Encoder::~Encoder() = default;

const std::string& Encoder::identity() const {
  std::call_once(impl_->identity_once, [&] {
    const std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> hash(
        EVP_MD_CTX_new(), EVP_MD_CTX_free);
    if (!hash || EVP_DigestInit_ex(hash.get(), EVP_sha256(), nullptr) != 1)
      throw std::runtime_error("cannot initialize vision fingerprint");
    for (const auto& region : impl_->reader->GetMappedRegions()) {
      if (EVP_DigestUpdate(hash.get(), region.data, region.size) != 1)
        throw std::runtime_error("cannot hash vision sidecar");
    }
    std::array<unsigned char, 32> digest{};
    unsigned length = 0;
    if (EVP_DigestFinal_ex(hash.get(), digest.data(), &length) != 1 ||
        length != digest.size())
      throw std::runtime_error("cannot finish vision fingerprint");
    constexpr std::string_view hex = "0123456789abcdef";
    impl_->identity.reserve(digest.size() * 2);
    for (const auto byte : digest) {
      impl_->identity.push_back(hex[byte >> 4]);
      impl_->identity.push_back(hex[byte & 15]);
    }
  });
  return impl_->identity;
}

std::size_t Encoder::ResidentBytes() const noexcept {
  return impl_->resident_bytes.load();
}

void Encoder::Upload() {
  impl_->Upload();
}

Encoder::Embedding::~Embedding() {
  if (data_ != nullptr)
    (void)hipFree(data_);
}

std::shared_ptr<const Encoder::Embedding> Encoder::Encode(
    const core::Image& image, const ImageGrid& grid, const Observer& observer,
    const CancellationCheck& is_cancelled) {
  const auto check_cancelled = [&] {
    if (is_cancelled && is_cancelled())
      throw std::runtime_error("vision encoding cancelled");
  };
  check_cancelled();
  if (grid.width == 0 || grid.height == 0 || image.width != grid.width ||
      image.height != grid.height || grid.width % kPatchSize != 0 ||
      grid.height % kPatchSize != 0 ||
      grid.vit_width != grid.width / kPatchSize ||
      grid.vit_height != grid.height / kPatchSize ||
      grid.llm_width != (grid.vit_width + kDownsample - 1) / kDownsample ||
      grid.llm_height != (grid.vit_height + kDownsample - 1) / kDownsample ||
      grid.Tokens() > kMaxImageTokens ||
      image.pixels.size() != std::size_t{image.width} * image.height * 3) {
    throw std::invalid_argument(
        "DeepSeek V4.1 vision encoder requires a padded image of its grid");
  }
  impl_->Upload();
  const auto stream = impl_->stream;
  const unsigned count = grid.vit_width * grid.vit_height;
  const unsigned merged = grid.llm_width * grid.llm_height;
  Allocation pixels(image.pixels.size());
  Allocation hidden(std::size_t{count} * kHidden * sizeof(float));
  Allocation work(std::max(std::size_t{count} * 3 * kHidden,
                           std::size_t{merged} * kEmbeddingWidth) *
                  sizeof(float));
  Allocation work2(std::size_t{count} * kFfn * sizeof(float));
  Allocation normalized(std::max(std::size_t{count} * kFfn,
                                 std::size_t{merged} * kMergeDim) *
                        sizeof(Bf16));
  // Attention rows padded to whole key tiles; the padding stays zero.
  static_assert(kAttentionQueries % kKeyTile == 0);
  const unsigned padded =
      (count + kAttentionQueries - 1) / kAttentionQueries * kAttentionQueries;
  const std::size_t qkv_size = std::max(std::size_t{3} * padded * kHidden,
                                        std::size_t{merged} * kEmbeddingWidth);
  Allocation qkv(qkv_size * sizeof(Bf16));
  Check(hipMemsetAsync(qkv.data, 0, qkv_size * sizeof(Bf16), stream));
  Allocation attention(std::size_t{count} * kHidden * sizeof(Bf16));
  {
    PinnedUpload upload(stream, image.pixels.size());
    upload.Copy(pixels.data, image.pixels.data(), image.pixels.size());
  }
  const auto observe = [&](const std::string& stage, const float* data,
                           std::size_t size) {
    if (!observer)
      return;
    std::vector<float> copy(size);
    Check(hipMemcpyAsync(copy.data(), data, size * sizeof(float),
                         hipMemcpyDeviceToHost, stream));
    Check(hipStreamSynchronize(stream));
    observer(stage, copy);
  };
  auto* const patches = normalized.As<Bf16>();
  Patchify<<<Blocks(std::size_t{count} * kPatchDim), kThreads, 0, stream>>>(
      pixels.As<std::uint8_t>(), patches, grid.width, grid.vit_width, count);
  impl_->Gemm("patch", patches, hidden.As<float>(), count, kHidden, kPatchDim);
  BiasRound<<<Blocks(std::size_t{count} * kHidden), kThreads, 0, stream>>>(
      hidden.As<float>(), impl_->FloatWeight("v.patch_embd.bias"),
      count * kHidden, kHidden);
  observe("patch", hidden.As<float>(), std::size_t{count} * kHidden);
  auto* const q = qkv.As<Bf16>();
  auto* const k = q + std::size_t{padded} * kHidden;
  auto* const vt = k + std::size_t{padded} * kHidden;
  for (unsigned layer = 0; layer < kLayers; ++layer) {
    check_cancelled();
    const auto prefix = "v.blk." + std::to_string(layer) + ".";
    RmsNorm<<<count, kThreads, 0, stream>>>(
        hidden.As<float>(), normalized.As<Bf16>(),
        impl_->FloatWeight(prefix + "ln1.weight"));
    impl_->Gemm(prefix + "attn_qkv.weight", normalized.As<Bf16>(),
                work.As<float>(), count, 3 * kHidden, kHidden);
    QkvRope<<<Blocks(std::size_t{count} * kHidden), kThreads, 0, stream>>>(
        work.As<float>(), impl_->FloatWeight(prefix + "attn_qkv.bias"),
        impl_->inv_freq.As<float>(), q, k, vt, count, padded, grid.vit_width);
    FusedAttention<<<dim3(padded / kAttentionQueries, kHeads),
                     32 * kAttentionWaves, 0, stream>>>(
        q, k, vt, attention.As<Bf16>(), count, padded);
    impl_->Gemm(prefix + "attn_out.weight", attention.As<Bf16>(),
                work.As<float>(), count, kHidden, kHidden);
    Residual<<<Blocks(std::size_t{count} * kHidden), kThreads, 0, stream>>>(
        hidden.As<float>(), work.As<float>(),
        impl_->FloatWeight(prefix + "attn_out.bias"), count * kHidden, kHidden);
    RmsNorm<<<count, kThreads, 0, stream>>>(
        hidden.As<float>(), normalized.As<Bf16>(),
        impl_->FloatWeight(prefix + "ln2.weight"));
    impl_->Gemm(prefix + "ffn_gate.weight", normalized.As<Bf16>(),
                work.As<float>(), count, kFfn, kHidden);
    impl_->Gemm(prefix + "ffn_up.weight", normalized.As<Bf16>(),
                work2.As<float>(), count, kFfn, kHidden);
    SwiGlu<<<Blocks(std::size_t{count} * kFfn), kThreads, 0, stream>>>(
        work.As<float>(), work2.As<float>(), normalized.As<Bf16>(),
        count * kFfn);
    impl_->Gemm(prefix + "ffn_down.weight", normalized.As<Bf16>(),
                work.As<float>(), count, kHidden, kFfn);
    Residual<<<Blocks(std::size_t{count} * kHidden), kThreads, 0, stream>>>(
        hidden.As<float>(), work.As<float>(), nullptr, count * kHidden,
        kHidden);
    observe("layer" + std::to_string(layer), hidden.As<float>(),
            std::size_t{count} * kHidden);
  }
  // The final norm reuses the attention buffer (count x 1024 BF16).
  RmsNorm<<<count, kThreads, 0, stream>>>(hidden.As<float>(),
                                          attention.As<Bf16>(),
                                          impl_->FloatWeight("v.post_ln.weight"));
  Unfold<<<Blocks(std::size_t{merged} * kMergeDim), kThreads, 0, stream>>>(
      attention.As<Bf16>(), normalized.As<Bf16>(), grid.vit_height,
      grid.vit_width, grid.llm_width, merged);
  impl_->Gemm("mm.1.weight", normalized.As<Bf16>(), work.As<float>(), merged,
              kEmbeddingWidth, kMergeDim);
  Gelu<<<Blocks(std::size_t{merged} * kEmbeddingWidth), kThreads, 0,
         stream>>>(work.As<float>(), impl_->FloatWeight("mm.1.bias"),
                   qkv.As<Bf16>(), merged * kEmbeddingWidth, kEmbeddingWidth);
  impl_->Gemm("mm.2.weight", qkv.As<Bf16>(), work.As<float>(), merged,
              kEmbeddingWidth, kEmbeddingWidth);
  auto embedding = std::shared_ptr<Embedding>(new Embedding);
  embedding->rows_ = grid.Tokens();
  Check(hipMalloc(&embedding->data_, std::size_t{embedding->rows_} *
                                         kEmbeddingWidth * sizeof(float)));
  Span<<<Blocks(std::size_t{embedding->rows_} * kEmbeddingWidth), kThreads, 0,
         stream>>>(work.As<float>(), impl_->FloatWeight("mm.2.bias"),
                   impl_->FloatWeight("v.token_embd.img_start"),
                   impl_->FloatWeight("v.image_newline"),
                   impl_->FloatWeight("v.token_embd.img_end"),
                   embedding->data_, grid.llm_width, embedding->rows_);
  Check(hipGetLastError());
  Check(hipStreamSynchronize(stream));
  check_cancelled();
  observe("span", embedding->data_,
          std::size_t{embedding->rows_} * kEmbeddingWidth);
  return embedding;
}

}  // namespace gufo::models::deepseek_v41_flash::vision
