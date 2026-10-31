#include "src/models/deepseek_v41_flash/vision/prompt.hpp"

#include <algorithm>
#include <cfenv>
#include <cmath>
#include <stdexcept>
#include <string>

#include "src/core/crypto/sha256.hpp"

namespace gufo::models::deepseek_v41_flash::vision {
namespace {

void HashU32(crypto::Sha256Hasher& hash, std::uint32_t value) {
  const std::array<std::uint8_t, 4> bytes{
      static_cast<std::uint8_t>(value), static_cast<std::uint8_t>(value >> 8),
      static_cast<std::uint8_t>(value >> 16),
      static_cast<std::uint8_t>(value >> 24)};
  hash.Update(bytes);
}

void HashText(crypto::Sha256Hasher& hash, std::string_view text) {
  hash.Update(
      {reinterpret_cast<const std::uint8_t*>(text.data()), text.size()});
}

/// Python's round(): to nearest, ties to even.
std::int64_t RoundHalfEven(double value) {
  const int mode = std::fegetround();
  std::fesetround(FE_TONEAREST);
  const double rounded = std::nearbyint(value);
  std::fesetround(mode);
  return static_cast<std::int64_t>(rounded);
}

std::uint32_t CeilDiv(std::uint64_t a, std::uint64_t b) {
  return static_cast<std::uint32_t>((a + b - 1) / b);
}

struct Size {
  std::uint64_t height;
  std::uint64_t width;
};

/// llm_grid.
Size LlmGrid(Size pixels) {
  return {CeilDiv(pixels.height / kPatchSize, kDownsample),
          CeilDiv(pixels.width / kPatchSize, kDownsample)};
}

std::uint64_t NumImageTokens(Size grid) {
  return grid.height * (grid.width + 1) + 2;
}

/// solve_resize_ratio, in the same double arithmetic.
Size SolveResizeRatio(double height, double width) {
  const double r = height / width;
  const double max_w = std::sqrt((kMaxImageTokens - 2) / r + 0.25) - 0.5;
  const double max_h = max_w * r;
  constexpr std::uint64_t cell = kPatchSize * kDownsample;
  if (max_w < 1.0)
    return {(kMaxImageTokens - 2) / 2 * cell, cell};
  if (max_h < 1.0)
    return {cell, (kMaxImageTokens - 3) * cell};
  const double beta = std::min(std::floor(max_w) * cell / width,
                               std::floor(max_h) * cell / height);
  return {static_cast<std::uint64_t>(std::floor(height * beta / kPatchSize)) *
              kPatchSize,
          static_cast<std::uint64_t>(std::floor(width * beta / kPatchSize)) *
              kPatchSize};
}

// Pillow's Resample.c for 8-bit RGB with the bicubic filter (a = -0.5): per
// axis, double coefficients normalized to sum 1, then rounded to 22-bit
// fixed point; horizontal pass first, each pass rounded and clipped to 8 bits.
constexpr int kPrecisionBits = 32 - 8 - 2;

double BicubicFilter(double x) {
  constexpr double a = -0.5;
  if (x < 0.0)
    x = -x;
  if (x < 1.0)
    return ((a + 2.0) * x - (a + 3.0)) * x * x + 1;
  if (x < 2.0)
    return (((x - 5) * x + 8) * x - 4) * a;
  return 0.0;
}

struct Coefficients {
  std::vector<int> first;
  std::vector<int> count;
  std::vector<std::int32_t> weights;  // [output][size]
  int size{0};
};

Coefficients Precompute(std::uint32_t input, std::uint32_t output) {
  const double scale = static_cast<double>(input) / output;
  const double filter_scale = std::max(scale, 1.0);
  const double support = 2.0 * filter_scale;
  Coefficients c;
  c.size = static_cast<int>(std::ceil(support)) * 2 + 1;
  c.first.resize(output);
  c.count.resize(output);
  c.weights.assign(std::size_t{output} * c.size, 0);
  std::vector<double> k(c.size);
  for (std::uint32_t xx = 0; xx < output; ++xx) {
    const double center = (xx + 0.5) * scale;
    const double ss = 1.0 / filter_scale;
    int xmin = static_cast<int>(center - support + 0.5);
    if (xmin < 0)
      xmin = 0;
    int xmax = static_cast<int>(center + support + 0.5);
    if (xmax > static_cast<int>(input))
      xmax = static_cast<int>(input);
    xmax -= xmin;
    double ww = 0.0;
    for (int x = 0; x < xmax; ++x) {
      k[x] = BicubicFilter((x + xmin - center + 0.5) * ss);
      ww += k[x];
    }
    for (int x = 0; x < xmax; ++x) {
      if (ww != 0.0)
        k[x] /= ww;
      const double scaled = k[x] * (1 << kPrecisionBits);
      c.weights[std::size_t{xx} * c.size + x] =
          static_cast<std::int32_t>(k[x] < 0 ? -0.5 + scaled : 0.5 + scaled);
    }
    c.first[xx] = xmin;
    c.count[xx] = xmax;
  }
  return c;
}

std::uint8_t Clip8(std::int64_t value) {
  return static_cast<std::uint8_t>(
      std::clamp<std::int64_t>(value >> kPrecisionBits, 0, 255));
}

core::Image ResizeHorizontal(const core::Image& image, std::uint32_t width) {
  const auto c = Precompute(image.width, width);
  core::Image out{.width = width, .height = image.height};
  out.pixels.resize(std::size_t{width} * image.height * 3);
  for (std::uint32_t y = 0; y < image.height; ++y) {
    const std::uint8_t* row =
        image.pixels.data() + std::size_t{y} * image.width * 3;
    for (std::uint32_t x = 0; x < width; ++x) {
      const std::int32_t* k = c.weights.data() + std::size_t{x} * c.size;
      for (unsigned channel = 0; channel < 3; ++channel) {
        std::int64_t ss = std::int64_t{1} << (kPrecisionBits - 1);
        for (int i = 0; i < c.count[x]; ++i)
          ss += std::int64_t{row[(std::size_t{c.first[x]} + i) * 3 + channel]} *
                k[i];
        out.pixels[(std::size_t{y} * width + x) * 3 + channel] = Clip8(ss);
      }
    }
  }
  return out;
}

core::Image ResizeVertical(const core::Image& image, std::uint32_t height) {
  const auto c = Precompute(image.height, height);
  core::Image out{.width = image.width, .height = height};
  out.pixels.resize(std::size_t{image.width} * height * 3);
  const std::size_t stride = std::size_t{image.width} * 3;
  for (std::uint32_t y = 0; y < height; ++y) {
    const std::int32_t* k = c.weights.data() + std::size_t{y} * c.size;
    for (std::size_t x = 0; x < stride; ++x) {
      std::int64_t ss = std::int64_t{1} << (kPrecisionBits - 1);
      for (int i = 0; i < c.count[y]; ++i)
        ss +=
            std::int64_t{
                image.pixels[(std::size_t{c.first[y]} + i) * stride + x]} *
            k[i];
      out.pixels[std::size_t{y} * stride + x] = Clip8(ss);
    }
  }
  return out;
}

/// Image.resize(size, BICUBIC): an unchanged axis is not resampled.
core::Image ResizeBicubicPil(const core::Image& image, std::uint32_t width,
                             std::uint32_t height) {
  if (width == image.width && height == image.height)
    return image;
  if (width == image.width)
    return ResizeVertical(image, height);
  const core::Image horizontal = ResizeHorizontal(image, width);
  return height == image.height ? horizontal
                                : ResizeVertical(horizontal, height);
}

}  // namespace

ImageGrid PlanGrid(std::uint32_t width, std::uint32_t height) {
  if (width == 0 || height == 0)
    throw std::invalid_argument("image has no pixels");
  double w = width, h = height;
  if (w * h < kMinPixels) {
    const double ratio = std::pow(kMinPixels / (w * h), 0.5);
    w = std::trunc(w * ratio);
    h = std::trunc(h * ratio);
  }
  if (w < 1 || h < 1)
    throw std::invalid_argument("image is too narrow");
  Size best{CeilDiv(static_cast<std::uint64_t>(h), kPatchSize) * kPatchSize,
            CeilDiv(static_cast<std::uint64_t>(w), kPatchSize) * kPatchSize};
  Size grid = LlmGrid(best);
  if (NumImageTokens(grid) > kMaxImageTokens) {
    best = SolveResizeRatio(h, w);
    grid = LlmGrid(best);
    if (NumImageTokens(grid) > kMaxImageTokens || best.height == 0 ||
        best.width == 0)
      throw std::logic_error("image grid exceeds its token budget");
  }
  return {
      .width = static_cast<std::uint32_t>(best.width),
      .height = static_cast<std::uint32_t>(best.height),
      .vit_width = static_cast<std::uint32_t>(best.width / kPatchSize),
      .vit_height = static_cast<std::uint32_t>(best.height / kPatchSize),
      .llm_width = static_cast<std::uint32_t>(grid.width),
      .llm_height = static_cast<std::uint32_t>(grid.height),
  };
}

core::Image PadImage(const core::Image& image, const ImageGrid& grid) {
  // ImageOps.contain, then pad: the resized image is pasted centred.
  std::uint32_t width = grid.width, height = grid.height;
  const double image_ratio = static_cast<double>(image.width) / image.height;
  const double target_ratio = static_cast<double>(grid.width) / grid.height;
  if (image_ratio > target_ratio) {
    height = static_cast<std::uint32_t>(RoundHalfEven(
        static_cast<double>(image.height) / image.width * grid.width));
  } else if (image_ratio < target_ratio) {
    width = static_cast<std::uint32_t>(RoundHalfEven(
        static_cast<double>(image.width) / image.height * grid.height));
  }
  if (width == 0 || height == 0)
    throw std::invalid_argument("image resizes to no pixels");
  const core::Image resized = ResizeBicubicPil(image, width, height);
  if (width == grid.width && height == grid.height)
    return resized;
  core::Image out{.width = grid.width, .height = grid.height};
  out.pixels.assign(std::size_t{grid.width} * grid.height * 3, 127);
  const std::uint32_t x = width != grid.width
                              ? static_cast<std::uint32_t>(
                                    RoundHalfEven((grid.width - width) * 0.5))
                              : 0;
  const std::uint32_t y =
      width != grid.width ? 0
                          : static_cast<std::uint32_t>(
                                RoundHalfEven((grid.height - height) * 0.5));
  for (std::uint32_t row = 0; row < height; ++row) {
    std::copy_n(
        resized.pixels.begin() + std::size_t{row} * width * 3,
        std::size_t{width} * 3,
        out.pixels.begin() + (std::size_t{y + row} * grid.width + x) * 3);
  }
  return out;
}

std::span<const std::uint8_t> Prompt::IdentityForPrefix(
    std::size_t token_count) const {
  for (auto image = images.rbegin(); image != images.rend(); ++image) {
    if (image->offset < token_count)
      return image->prefix_identity;
  }
  return {};
}

std::size_t Prompt::Expanded(std::size_t rendered) const {
  std::size_t position = rendered;
  for (const auto& image : images) {
    if (image.rendered_offset < rendered)
      position += image.grid.Tokens() - 1;
  }
  return position;
}

Prompt Expand(
    std::span<const int> rendered,
    std::span<const std::shared_ptr<const std::vector<std::uint8_t>>> images,
    std::string_view encoder_identity, std::uint32_t max_context) {
  if (!images.empty() && encoder_identity.empty())
    throw std::invalid_argument(
        "image input requires DeepSeek V4.1's --mmproj vision sidecar");
  const auto placeholders = std::ranges::count(rendered, kImageToken);
  if (placeholders != static_cast<std::ptrdiff_t>(images.size()))
    throw std::invalid_argument(
        "rendered prompt has " + std::to_string(placeholders) +
        " image placeholders for " + std::to_string(images.size()) + " images");
  Prompt prompt;
  crypto::Sha256Hasher identity;
  HashText(identity, kPreprocessingVersion);
  HashText(identity, encoder_identity);
  std::size_t next = 0;
  for (std::size_t i = 0; i < rendered.size(); ++i) {
    if (rendered[i] != kImageToken) {
      prompt.tokens.push_back(rendered[i]);
      continue;
    }
    const auto& bytes = images[next++];
    if (!bytes)
      throw std::invalid_argument("image has no data");
    const core::Image decoded = core::DecodeImage(*bytes);
    const ImageGrid grid = PlanGrid(decoded.width, decoded.height);
    auto pixels = std::make_shared<const core::Image>(PadImage(decoded, grid));
    const auto offset = static_cast<std::uint32_t>(prompt.tokens.size());
    prompt.tokens.insert(prompt.tokens.end(), grid.Tokens(), kImageToken);
    HashU32(identity, offset);
    HashU32(identity, grid.width);
    HashU32(identity, grid.height);
    identity.Update(pixels->pixels);
    prompt.images.push_back({std::move(pixels), grid, offset,
                             static_cast<std::uint32_t>(i), identity.Digest()});
  }
  if (prompt.tokens.size() >= max_context)
    throw std::length_error(
        "prompt exceeds the " + std::to_string(max_context) +
        "-token context; increase --context or shorten the conversation");
  if (!prompt.images.empty()) {
    const auto digest = identity.Finish();
    prompt.cache_identity.assign(digest.begin(), digest.end());
  }
  return prompt;
}

}  // namespace gufo::models::deepseek_v41_flash::vision
