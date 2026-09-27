#include "src/models/gemma4/kernels/rocm/device_model.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>

#include "src/core/hip/weight_upload.hpp"

namespace gufo::models::gemma4::rocm {
namespace {

/// Quantized GEMM kernels read whole 256-element k-iterations and may
/// over-read the last row; every upload carries a zeroed tail so those reads
/// stay inside the allocation.
constexpr std::size_t kTailMargin = 4096;

struct Uploader {
  hip::WeightUpload& stager;
  std::vector<void*>& allocations;
  std::size_t& bytes;
  std::size_t& max_cols;
  std::string* error;
  std::uint32_t shard_base{0};
  bool ok{true};

  void Fail(const std::string& message) {
    if (ok && error != nullptr) {
      *error = message;
    }
    ok = false;
  }

  void* Allocate(std::size_t size, const std::string& what) {
    void* ptr = nullptr;
    if (hipMalloc(&ptr, size + kTailMargin) != hipSuccess) {
      Fail("hipMalloc failed for " + what + " (" + std::to_string(size) +
           " bytes)");
      return nullptr;
    }
    allocations.push_back(ptr);
    bytes += size + kTailMargin;
    (void)hipMemsetAsync(static_cast<std::uint8_t*>(ptr) + size, 0, kTailMargin,
                         nullptr);
    return ptr;
  }

  DeviceTensor Copy(const TensorRef& t) {
    DeviceTensor d;
    if (t.empty() || !ok) {
      return d;
    }
    const std::size_t size = t.SizeBytes();
    void* ptr = Allocate(size, std::string(t.name));
    if (ptr == nullptr) {
      return d;
    }
    if (!stager.Copy(shard_base + t.shard, t.file_offset, size, ptr, error)) {
      Fail("upload failed for " + std::string(t.name));
      return d;
    }
    d.data = ptr;
    d.type = t.type;
    d.cols = static_cast<std::uint32_t>(t.cols);
    d.rows = static_cast<std::uint32_t>(t.rows);
    if (t.type != core::GgmlType::kF32) {
      max_cols = std::max<std::size_t>(max_cols, t.cols);
    }
    return d;
  }

  float* Floats(const std::vector<float>& values, const char* what) {
    const std::size_t size = values.size() * sizeof(float);
    auto* ptr = static_cast<float*>(Allocate(size, what));
    if (ptr != nullptr && hipMemcpy(ptr, values.data(), size,
                                    hipMemcpyHostToDevice) != hipSuccess) {
      Fail(std::string("copy failed for ") + what);
    }
    return ptr;
  }

  DeviceLayer Layer(const LayerWeights& l) {
    DeviceLayer d;
    d.attn_norm = Copy(l.attn_norm);
    d.attn_q = Copy(l.attn_q);
    d.attn_k = Copy(l.attn_k);
    d.attn_v = Copy(l.attn_v);
    d.attn_q_norm = Copy(l.attn_q_norm);
    d.attn_k_norm = Copy(l.attn_k_norm);
    d.attn_output = Copy(l.attn_output);
    d.post_attn_norm = Copy(l.post_attn_norm);
    d.ffn_norm = Copy(l.ffn_norm);
    d.ffn_gate = Copy(l.ffn_gate);
    d.ffn_up = Copy(l.ffn_up);
    d.ffn_down = Copy(l.ffn_down);
    d.post_ffn_norm = Copy(l.post_ffn_norm);
    d.output_scale = l.output_scale;
    return d;
  }
};

}  // namespace

DeviceModel::~DeviceModel() {
  for (void* ptr : allocations_) {
    (void)hipFree(ptr);
  }
}

std::unique_ptr<DeviceModel> DeviceModel::Upload(
    const ModelWeights& weights, const core::GgufReader& reader,
    const DraftWeights* draft, const core::GgufReader* draft_reader,
    std::string* error_msg) {
  std::unique_ptr<DeviceModel> m(new DeviceModel());
  m->config_ = weights.config;
  m->vocab_ = weights.vocab_size;
  const auto regions = reader.GetMappedRegions();
  std::vector<core::GgufMappedRegion> shards(regions.begin(), regions.end());
  const auto shard_count = static_cast<std::uint32_t>(shards.size());
  if (draft != nullptr) {
    if (draft_reader == nullptr) {
      if (error_msg != nullptr) {
        *error_msg = "draft weights require their bound reader";
      }
      return nullptr;
    }
    const auto extra = draft_reader->GetMappedRegions();
    shards.insert(shards.end(), extra.begin(), extra.end());
  }
  auto stager = hip::WeightUpload::Create(shards, error_msg);
  if (!stager) {
    return nullptr;
  }
  Uploader up{*stager, m->allocations_, m->bytes_, m->max_cols_, error_msg};
  m->token_embd_ = up.Copy(weights.token_embd);
  m->output_ = weights.TiedOutput() ? m->token_embd_ : up.Copy(weights.output);
  m->output_norm_ = up.Copy(weights.output_norm);
  m->rope_factors_ = up.Floats(weights.rope_factors.values, "rope factors");
  {
    // A pair whose angle stays below 1e-12 rad at 2^24 positions rounds to
    // the identity in binary16 keys.
    const auto& factors = weights.rope_factors.values;
    const double scale =
        std::pow(static_cast<double>(m->config_.rope_theta_global),
                 -2.0 / m->config_.head_dim_global);
    for (std::size_t i = 0; i < factors.size(); ++i) {
      const double angle =
          16777216.0 * std::pow(scale, static_cast<double>(i)) / factors[i];
      if (angle >= 1e-12) {
        m->global_rope_pairs_ = static_cast<std::uint32_t>(i + 1);
      }
    }
  }
  m->layers_.reserve(weights.layers.size());
  for (const auto& l : weights.layers) {
    m->layers_.push_back(up.Layer(l));
    if (!up.ok) {
      return nullptr;
    }
  }
  if (draft != nullptr) {
    up.shard_base = shard_count;
    auto& d = m->draft_;
    d.config = draft->config;
    d.pre_projection = up.Copy(draft->pre_projection);
    d.post_projection = up.Copy(draft->post_projection);
    d.token_embd = up.Copy(draft->token_embd);
    d.output_norm = up.Copy(draft->output_norm);
    d.rope_factors =
        up.Floats(draft->rope_factors.values, "draft rope factors");
    for (const auto& l : draft->layers) {
      d.layers.push_back(up.Layer(l));
    }
    m->has_draft_ = true;
  }
  if (!up.ok || !stager->Finish(error_msg)) {
    return nullptr;
  }
  const hipError_t status = hipDeviceSynchronize();
  if (status != hipSuccess) {
    if (error_msg != nullptr) {
      *error_msg =
          std::string("weight upload failed: ") + hipGetErrorString(status);
    }
    return nullptr;
  }
  return m;
}

}  // namespace gufo::models::gemma4::rocm
