// DeepSeek V4.1 Flash image prompt: grid planning (official
// image_processor.plan_image_grid), padding and placeholder expansion.
#include <array>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/core/image.hpp"
#include "src/models/deepseek_v41_flash/vision/prompt.hpp"

namespace vision = gufo::models::deepseek_v41_flash::vision;

namespace {

int failures = 0;

void Expect(bool condition, const std::string& what) {
  if (!condition) {
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    ++failures;
  }
}

std::uint32_t Crc32(const std::uint8_t* data, std::size_t size,
                    std::uint32_t crc = 0) {
  crc = ~crc;
  for (std::size_t i = 0; i < size; ++i) {
    crc ^= data[i];
    for (int bit = 0; bit < 8; ++bit)
      crc = (crc >> 1) ^ (0xEDB88320U & (0U - (crc & 1U)));
  }
  return ~crc;
}

void PutU32(std::vector<std::uint8_t>& out, std::uint32_t value) {
  for (int shift = 24; shift >= 0; shift -= 8)
    out.push_back(static_cast<std::uint8_t>(value >> shift));
}

void Chunk(std::vector<std::uint8_t>& png, const char* type,
           const std::vector<std::uint8_t>& data) {
  PutU32(png, static_cast<std::uint32_t>(data.size()));
  const auto start = png.size();
  png.insert(png.end(), type, type + 4);
  png.insert(png.end(), data.begin(), data.end());
  PutU32(png, Crc32(png.data() + start, png.size() - start));
}

// An RGB8 PNG of a solid colour, with stored (uncompressed) deflate blocks.
std::shared_ptr<const std::vector<std::uint8_t>> SolidPng(
    std::uint32_t width, std::uint32_t height,
    std::array<std::uint8_t, 3> rgb) {
  std::vector<std::uint8_t> raw;
  for (std::uint32_t y = 0; y < height; ++y) {
    raw.push_back(0);  // filter: none
    for (std::uint32_t x = 0; x < width; ++x)
      raw.insert(raw.end(), rgb.begin(), rgb.end());
  }
  std::vector<std::uint8_t> z{0x78, 0x01};
  for (std::size_t at = 0; at < raw.size();) {
    const auto n = std::min<std::size_t>(65535, raw.size() - at);
    z.push_back(at + n == raw.size() ? 1 : 0);
    z.push_back(static_cast<std::uint8_t>(n));
    z.push_back(static_cast<std::uint8_t>(n >> 8));
    z.push_back(static_cast<std::uint8_t>(~n));
    z.push_back(static_cast<std::uint8_t>(~n >> 8));
    z.insert(z.end(), raw.begin() + static_cast<std::ptrdiff_t>(at),
             raw.begin() + static_cast<std::ptrdiff_t>(at + n));
    at += n;
  }
  std::uint32_t a = 1, b = 0;
  for (const auto byte : raw) {
    a = (a + byte) % 65521;
    b = (b + a) % 65521;
  }
  PutU32(z, b << 16 | a);
  std::vector<std::uint8_t> png{0x89, 'P', 'N', 'G', '\r', '\n', 0x1a, '\n'};
  std::vector<std::uint8_t> header;
  PutU32(header, width);
  PutU32(header, height);
  header.insert(header.end(), {8, 2, 0, 0, 0});
  Chunk(png, "IHDR", header);
  Chunk(png, "IDAT", z);
  Chunk(png, "IEND", {});
  return std::make_shared<const std::vector<std::uint8_t>>(std::move(png));
}

void TestGrids() {
  // Grids of the official image_processor (tools/ds41/vision_reference.py):
  // enlargement below 544^2 pixels, padding to 14-pixel patches, a merged
  // grid rounded up, and the 1024-token cap.
  struct Case {
    std::uint32_t width, height;
    vision::ImageGrid grid;
  } cases[] = {
      {2000, 300, {2002, 308, 143, 22, 48, 8}},
      {250, 1900, {252, 1904, 18, 136, 6, 46}},
      {97, 61, {686, 434, 49, 31, 17, 11}},
      {4000, 3000, {1512, 1134, 108, 81, 36, 27}},
  };
  for (const auto& c : cases) {
    const auto grid = vision::PlanGrid(c.width, c.height);
    Expect(grid == c.grid, "grid of " + std::to_string(c.width) + "x" +
                               std::to_string(c.height));
    Expect(grid.Tokens() <= vision::kMaxImageTokens, "token cap");
  }
  Expect(vision::PlanGrid(4000, 3000).Tokens() == 1001, "capped tokens");
}

void TestPad() {
  // A wide solid image keeps its colour and is centred on gray 127.
  const auto decoded =
      gufo::core::DecodeImage(*SolidPng(60, 20, {200, 10, 30}));
  const auto grid = vision::PlanGrid(60, 20);
  const auto padded = vision::PadImage(decoded, grid);
  Expect(padded.width == grid.width && padded.height == grid.height,
         "padded size");
  const auto at = [&](std::uint32_t x, std::uint32_t y, int c) {
    return padded.pixels[(std::size_t{y} * padded.width + x) * 3 + c];
  };
  Expect(at(0, 0, 0) == 127 && at(0, 0, 1) == 127 && at(0, 0, 2) == 127,
         "gray border");
  const auto cx = padded.width / 2, cy = padded.height / 2;
  Expect(at(cx, cy, 0) == 200 && at(cx, cy, 1) == 10 && at(cx, cy, 2) == 30,
         "colour kept");
}

void TestExpand() {
  const int image = vision::kImageToken;
  const std::vector<int> rendered{1, image, 2, 3, image, 4};
  const std::vector images{SolidPng(60, 20, {1, 2, 3}),
                           SolidPng(600, 40, {4, 5, 6})};
  const auto prompt = vision::Expand(rendered, images, "encoder", 1 << 20);
  Expect(prompt.images.size() == 2, "two images");
  const auto& first = prompt.images[0];
  const auto& second = prompt.images[1];
  Expect(first.offset == 1 && first.rendered_offset == 1, "first offset");
  Expect(second.rendered_offset == 4 &&
             second.offset == 1 + first.grid.Tokens() + 2,
         "second offset");
  Expect(prompt.tokens.size() ==
             rendered.size() - 2 + first.grid.Tokens() + second.grid.Tokens(),
         "expanded length");
  for (const auto* span : {&first, &second}) {
    for (std::uint32_t i = 0; i < span->grid.Tokens(); ++i)
      Expect(prompt.tokens[span->offset + i] == image, "span token");
  }
  Expect(prompt.tokens[first.offset + first.grid.Tokens()] == 2,
         "text after the first image");
  Expect(prompt.Expanded(0) == 0 && prompt.Expanded(1) == 1, "Expanded");
  Expect(prompt.Expanded(2) == first.offset + first.grid.Tokens(),
         "Expanded past an image");
  Expect(prompt.Expanded(5) == prompt.tokens.size() - 1, "Expanded end");
  // A prefix holds only the images that start inside it.
  Expect(prompt.IdentityForPrefix(first.offset).empty(), "no image prefix");
  const auto one = prompt.IdentityForPrefix(second.offset);
  Expect(std::equal(one.begin(), one.end(), first.prefix_identity.begin(),
                    first.prefix_identity.end()),
         "first-image prefix");
  Expect(first.prefix_identity != second.prefix_identity, "prefix identities");
  Expect(!prompt.cache_identity.empty(), "cache identity");

  // Identities follow the pixels and the encoder.
  const auto other = vision::Expand(
      rendered, std::vector{images[0], SolidPng(600, 40, {4, 5, 7})}, "encoder",
      1 << 20);
  Expect(other.images[0].prefix_identity == first.prefix_identity &&
             other.cache_identity != prompt.cache_identity,
         "pixel identity");
  const auto encoder = vision::Expand(rendered, images, "other", 1 << 20);
  Expect(encoder.cache_identity != prompt.cache_identity, "encoder identity");

  const auto throws = [](auto&& call) {
    try {
      call();
    } catch (const std::exception&) {
      return true;
    }
    return false;
  };
  Expect(throws([&] {
           (void)vision::Expand(rendered, std::vector{images[0]}, "encoder",
                                1 << 20);
         }),
         "placeholder count");
  Expect(throws([&] { (void)vision::Expand(rendered, images, "", 1 << 20); }),
         "missing encoder");
  Expect(throws([&] {
           (void)vision::Expand(
               rendered, images, "encoder",
               static_cast<std::uint32_t>(prompt.tokens.size()));
         }),
         "context limit");
  Expect(
      vision::Expand(std::vector<int>{1, 2}, {}, "", 8).cache_identity.empty(),
      "text-only prompt");
}

}  // namespace

int main() {
  try {
    TestGrids();
    TestPad();
    TestExpand();
  } catch (const std::exception& error) {
    std::fprintf(stderr, "FAIL: %s\n", error.what());
    return 1;
  }
  if (failures == 0)
    std::printf("ds41 vision prompt: ok\n");
  return failures == 0 ? 0 : 1;
}
