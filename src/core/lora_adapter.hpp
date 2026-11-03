#ifndef GUFO_CORE_LORA_ADAPTER_HPP_
#define GUFO_CORE_LORA_ADAPTER_HPP_

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace gufo::core {

/// One LoRA target of a GGUF adapter (`general.type = adapter`,
/// `adapter.type = lora`), in llama.cpp's layout: for base tensor W
/// (ne = [in, out] or [in, out, experts]) the pair `W.lora_a` (ne = [in,
/// rank(, experts)]) and `W.lora_b` (ne = [rank, out(, experts)]) adds
/// scale * B (A x) to W x, with scale = run-time scale * alpha / rank
/// (alpha = rank when the file has none or 0, as llama.cpp).
struct LoraTarget {
  std::string tensor;  // the base tensor, e.g. "blk.3.attn_output_b.weight"
  std::uint32_t rank = 0;
  std::uint32_t in = 0;
  std::uint32_t out = 0;
  std::uint32_t experts = 0;  // 0 for a dense matrix
  float scale = 1.0f;         // alpha / rank
  std::vector<float> a;       // [max(experts, 1)][rank][in]
  std::vector<float> b;       // [max(experts, 1)][out][rank]
};

struct LoraAdapter {
  /// gguf-sha256-v1 digest of the file (the prompt cache keys on it).
  std::string identity;
  std::vector<LoraTarget> targets;  // in file order of their lora_a
};

/// Loads a LoRA adapter for `architecture`: F32, F16 or BF16 halves,
/// converted to F32. Returns nullopt with *error_msg set otherwise.
[[nodiscard]] std::optional<LoraAdapter> LoadLoraAdapter(
    const std::filesystem::path& path, std::string_view architecture,
    std::string* error_msg = nullptr);

/// The adapter.type of a GGUF adapter, or empty when the file is not one.
[[nodiscard]] std::string AdapterType(const std::filesystem::path& path,
                                      std::string* error_msg = nullptr);

}  // namespace gufo::core

#endif  // GUFO_CORE_LORA_ADAPTER_HPP_
