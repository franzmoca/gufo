#ifndef GUFO_MODELS_DEEPSEEK_V41_FLASH_VISION_ENCODER_HPP_
#define GUFO_MODELS_DEEPSEEK_V41_FLASH_VISION_ENCODER_HPP_

#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <string_view>

#include "src/core/image.hpp"
#include "src/models/deepseek_v41_flash/vision/prompt.hpp"

namespace gufo::models::deepseek_v41_flash::vision {

inline constexpr std::uint32_t kEmbeddingWidth = 5120;

/// DeepSeek V4.1 Flash vision tower and aligner on the GPU (official
/// inference/vision.py): BF16 weights, activations rounded to BF16 where the
/// official BF16 forward rounds them, FP32 accumulation.
class Encoder {
public:
  /// Device rows [rows][kEmbeddingWidth] of FP32 language-model input
  /// embeddings for one image span, delimiters included.
  class Embedding {
  public:
    ~Embedding();
    Embedding(const Embedding&) = delete;
    Embedding& operator=(const Embedding&) = delete;
    [[nodiscard]] const float* data() const noexcept { return data_; }
    [[nodiscard]] std::uint32_t rows() const noexcept { return rows_; }

  private:
    friend class Encoder;
    Embedding() = default;
    float* data_{nullptr};
    std::uint32_t rows_{0};
  };

  using Observer =
      std::function<void(std::string_view stage, std::span<const float>)>;
  using CancellationCheck = std::function<bool()>;

  /// Validates the sidecar; weights stay on disk until Upload or Encode.
  explicit Encoder(const std::filesystem::path& path);
  ~Encoder();
  Encoder(const Encoder&) = delete;
  Encoder& operator=(const Encoder&) = delete;

  /// SHA-256 of the whole sidecar, computed once.
  [[nodiscard]] const std::string& identity() const;
  [[nodiscard]] std::size_t ResidentBytes() const noexcept;
  /// Upload the weights now instead of in the first Encode. Repeated calls
  /// keep the existing upload.
  void Upload();
  /// `pixels` is the padded image of `grid` (PadImage). Calls are serialized
  /// by the owner; scratch is sized by the image.
  [[nodiscard]] std::shared_ptr<const Embedding> Encode(
      const core::Image& pixels, const ImageGrid& grid,
      const Observer& observer = {},
      const CancellationCheck& is_cancelled = {});

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace gufo::models::deepseek_v41_flash::vision

#endif  // GUFO_MODELS_DEEPSEEK_V41_FLASH_VISION_ENCODER_HPP_
