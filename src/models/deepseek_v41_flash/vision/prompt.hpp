#ifndef GUFO_MODELS_DEEPSEEK_V41_FLASH_VISION_PROMPT_HPP_
#define GUFO_MODELS_DEEPSEEK_V41_FLASH_VISION_PROMPT_HPP_

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string_view>
#include <vector>

#include "src/core/image.hpp"

namespace gufo::models::deepseek_v41_flash::vision {

/// `<｜deepseek_image｜>`: the rendered placeholder and every position of an
/// image span (image_token_id).
inline constexpr int kImageToken = 129264;
inline constexpr std::uint32_t kPatchSize = 14;
/// The aligner merges 3x3 patches into one language-model token.
inline constexpr std::uint32_t kDownsample = 3;
/// Most language-model tokens of one image, its delimiters included.
inline constexpr std::uint32_t kMaxImageTokens = 1024;
/// Smaller images are enlarged to at least this many pixels.
inline constexpr std::uint32_t kMinPixels = 544 * 544;
inline constexpr std::string_view kPreprocessingVersion =
    "deepseek41-image-pad127-bicubic-rgb8-bf16-v1";

/// image_processor.plan_image_grid: the padded pixel size, its ViT patch grid
/// and the aligner's token grid.
struct ImageGrid {
  std::uint32_t width{0};
  std::uint32_t height{0};
  std::uint32_t vit_width{0};
  std::uint32_t vit_height{0};
  std::uint32_t llm_width{0};
  std::uint32_t llm_height{0};
  /// [IMAGE_START] + ([IMAGE] * llm_width + [IMAGE_NEW_LINE]) * llm_height +
  /// [IMAGE_END].
  [[nodiscard]] std::uint32_t Tokens() const noexcept {
    return llm_height * (llm_width + 1) + 2;
  }
  bool operator==(const ImageGrid&) const = default;
};

[[nodiscard]] ImageGrid PlanGrid(std::uint32_t width, std::uint32_t height);
/// load_image: ImageOps.pad to the grid's size, bicubic resize preserving the
/// aspect ratio, centred on gray (127, 127, 127).
[[nodiscard]] core::Image PadImage(const core::Image& image,
                                   const ImageGrid& grid);

struct PreparedImage {
  /// Padded RGB8 of grid.width x grid.height.
  std::shared_ptr<const core::Image> pixels;
  ImageGrid grid;
  /// First position of the span (its IMAGE_START).
  std::uint32_t offset{0};
  /// The placeholder's index in the rendered tokens.
  std::uint32_t rendered_offset{0};
  /// Covers every image up to and including this one: placement, pixels,
  /// preprocessing version and encoder identity.
  std::array<std::uint8_t, 32> prefix_identity{};
};

struct Prompt {
  std::vector<int> tokens;
  std::vector<PreparedImage> images;
  /// Identity of all images; empty for a text-only prompt.
  std::vector<std::uint8_t> cache_identity;
  /// Identity of the images starting before `token_count`.
  [[nodiscard]] std::span<const std::uint8_t> IdentityForPrefix(
      std::size_t token_count) const;
  /// Position in `tokens` of the rendered token at `rendered`.
  [[nodiscard]] std::size_t Expanded(std::size_t rendered) const;
};

/// prepare_vl_inputs: expands each placeholder of the rendered prompt into
/// its image's span; images are decoded, resized and padded here, once.
[[nodiscard]] Prompt Expand(
    std::span<const int> rendered,
    std::span<const std::shared_ptr<const std::vector<std::uint8_t>>> images,
    std::string_view encoder_identity, std::uint32_t max_context);

}  // namespace gufo::models::deepseek_v41_flash::vision

#endif  // GUFO_MODELS_DEEPSEEK_V41_FLASH_VISION_PROMPT_HPP_
