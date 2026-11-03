#include "src/core/direction_adapter.hpp"

#include <stdlib.h>

#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/core/platform/env.hpp"

namespace {

using gufo::core::GgmlType;
using gufo::core::GgufValueType;

void Expect(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    std::exit(1);
  }
}

struct Tensor {
  std::string name;
  std::vector<std::uint64_t> ne;
  std::vector<float> values;
  GgmlType type = GgmlType::kF32;
};

using Value = std::variant<std::string, float>;

std::vector<std::uint8_t> Build(
    const std::vector<std::pair<std::string, Value>>& metadata,
    const std::vector<Tensor>& tensors) {
  std::vector<std::uint8_t> out;
  auto append = [&out](const void* data, std::size_t size) {
    const auto* bytes = static_cast<const std::uint8_t*>(data);
    out.insert(out.end(), bytes, bytes + size);
  };
  auto pod = [&append](auto value) { append(&value, sizeof(value)); };
  auto string = [&](std::string_view value) {
    pod(static_cast<std::uint64_t>(value.size()));
    append(value.data(), value.size());
  };
  append("GGUF", 4);
  pod(std::uint32_t{3});
  pod(static_cast<std::uint64_t>(tensors.size()));
  pod(static_cast<std::uint64_t>(metadata.size()));
  for (const auto& [key, value] : metadata) {
    string(key);
    if (const auto* s = std::get_if<std::string>(&value)) {
      pod(static_cast<std::uint32_t>(GgufValueType::kString));
      string(*s);
    } else {
      pod(static_cast<std::uint32_t>(GgufValueType::kFloat32));
      pod(std::get<float>(value));
    }
  }
  std::uint64_t offset = 0;
  for (const auto& t : tensors) {
    string(t.name);
    pod(static_cast<std::uint32_t>(t.ne.size()));
    for (auto d : t.ne)
      pod(d);
    pod(static_cast<std::uint32_t>(t.type));
    pod(offset);
    offset += (t.values.size() * sizeof(float) + 31) / 32 * 32;
  }
  while (out.size() % 32)
    out.push_back(0);
  for (const auto& t : tensors) {
    append(t.values.data(), t.values.size() * sizeof(float));
    while (out.size() % 32)
      out.push_back(0);
  }
  return out;
}

constexpr std::uint32_t kWidth = 8, kLayers = 4;
constexpr std::array<std::string_view, 3> kSites = {"attn_out", "shexp_out",
                                                    "ffn_out"};

std::vector<std::pair<std::string, Value>> Metadata(
    std::string arch = "toy", std::string type = "direction",
    std::string mode = "project") {
  return {{"general.architecture", arch},
          {"general.type", std::string("adapter")},
          {"adapter.type", type},
          {"adapter.direction.mode", mode},
          {"adapter.direction.scale", 0.5f}};
}

std::vector<float> Ramp(std::size_t n, float base) {
  std::vector<float> v(n);
  for (std::size_t i = 0; i < n; ++i)
    v[i] = base + static_cast<float>(i);
  return v;
}

std::optional<gufo::core::DirectionAdapter> Load(
    const std::filesystem::path& dir, const std::vector<std::uint8_t>& image,
    std::string* error) {
  static int counter = 0;
  const auto path = dir / ("adapter" + std::to_string(counter++) + ".gguf");
  std::ofstream(path, std::ios::binary)
      .write(reinterpret_cast<const char*>(image.data()),
             static_cast<std::streamsize>(image.size()));
  return gufo::core::LoadDirectionAdapter(path,
                                          {.architecture = "toy",
                                           .width = kWidth,
                                           .layers = kLayers,
                                           .sites = kSites},
                                          error);
}

void ExpectRejected(const std::filesystem::path& dir,
                    const std::vector<std::uint8_t>& image,
                    std::string_view what, std::string_view needle) {
  std::string error;
  Expect(!Load(dir, image, &error), std::string(what) + " is rejected");
  Expect(error.find(needle) != std::string::npos,
         std::string(what) + ": error names the cause: " + error);
}

}  // namespace

int main() {
  char temp[] = "/tmp/gufo-direction-XXXXXX";
  Expect(mkdtemp(temp) != nullptr, "create test directory");
  const std::filesystem::path dir(temp);
  gufo::platform::SetEnv("XDG_CACHE_HOME", temp);

  const Tensor ffn{"blk.3.ffn_out.direction", {kWidth}, Ramp(kWidth, 1.0f)};
  const Tensor attn{
      "blk.1.attn_out.direction", {kWidth, 2}, Ramp(2 * kWidth, -4.0f)};
  const Tensor zero{
      "blk.0.shexp_out.direction", {kWidth, 1}, std::vector<float>(kWidth)};
  {
    std::string error;
    const auto adapter =
        Load(dir, Build(Metadata(), {ffn, attn, zero}), &error);
    Expect(adapter.has_value(), "valid adapter loads: " + error);
    Expect(adapter->mode == gufo::core::DirectionMode::kProject, "mode");
    Expect(adapter->scale == 0.5f, "scale");
    Expect(adapter->identity.size() == 64, "identity is a SHA-256 digest");
    Expect(adapter->sets.size() == 2, "the all-zero set is dropped");
    Expect(adapter->sets[0].layer == 1 && adapter->sets[0].site == 0 &&
               adapter->sets[0].count == 2 &&
               adapter->sets[0].vectors == attn.values,
           "sets are ordered by layer and keep their vectors");
    Expect(adapter->sets[1].layer == 3 && adapter->sets[1].site == 2 &&
               adapter->sets[1].count == 1,
           "one-dimensional tensors hold one vector");
  }
  {
    std::string error;
    const auto adapter =
        Load(dir, Build(Metadata("toy", "direction", "add"), {ffn}), &error);
    Expect(adapter && adapter->mode == gufo::core::DirectionMode::kAdd,
           "add mode: " + error);
  }
  ExpectRejected(dir, Build(Metadata("other"), {ffn}), "another model",
                 "not toy");
  ExpectRejected(dir, Build(Metadata("toy", "lora"), {ffn}), "a LoRA",
                 "not a direction adapter");
  ExpectRejected(dir, Build(Metadata("toy", "direction", "steer"), {ffn}),
                 "an unknown mode", "adapter.direction.mode");
  ExpectRejected(
      dir,
      Build(Metadata(),
            {{"blk.0.stream.direction", {kWidth}, Ramp(kWidth, 1.0f)}}),
      "an unknown site", "site 'stream'");
  ExpectRejected(
      dir,
      Build(Metadata(),
            {{"blk.4.ffn_out.direction", {kWidth}, Ramp(kWidth, 1.0f)}}),
      "a layer past the model", "past the model");
  ExpectRejected(
      dir,
      Build(
          Metadata(),
          {{"blk.0.ffn_out.direction", {kWidth + 1}, Ramp(kWidth + 1, 1.0f)}}),
      "another width", "ne = [8, k]");
  // The reader refuses repeated tensor names before the loader sees them.
  ExpectRejected(dir, Build(Metadata(), {ffn, ffn}), "duplicate sets",
                 "uplicate");
  ExpectRejected(dir, Build(Metadata(), {zero}), "only zero vectors",
                 "no nonzero");
  std::filesystem::remove_all(dir);
  std::cout << "direction adapter tests passed\n";
  return 0;
}
