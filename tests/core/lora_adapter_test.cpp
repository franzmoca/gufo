#include "src/core/lora_adapter.hpp"

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

// Binary16 bits of a normal value whose mantissa fits in 10 bits.
std::uint16_t Half(float value) {
  if (value == 0.0f)
    return 0;
  std::uint32_t bits;
  std::memcpy(&bits, &value, 4);
  const std::uint32_t exp = ((bits >> 23) & 0xff) - 127 + 15;
  return static_cast<std::uint16_t>(((bits >> 16) & 0x8000) | (exp << 10) |
                                    ((bits >> 13) & 0x3ff));
}

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
    const std::size_t width = t.type == GgmlType::kF16 ? 2 : sizeof(float);
    offset += (t.values.size() * width + 31) / 32 * 32;
  }
  while (out.size() % 32)
    out.push_back(0);
  for (const auto& t : tensors) {
    if (t.type == GgmlType::kF16) {
      for (float v : t.values)
        pod(Half(v));
    } else {
      append(t.values.data(), t.values.size() * sizeof(float));
    }
    while (out.size() % 32)
      out.push_back(0);
  }
  return out;
}

std::vector<std::pair<std::string, Value>> Metadata(
    std::string arch = "toy", std::string type = "lora",
    std::optional<float> alpha = 8.0f) {
  std::vector<std::pair<std::string, Value>> m = {
      {"general.architecture", arch},
      {"general.type", std::string("adapter")},
      {"adapter.type", type}};
  if (alpha)
    m.push_back({"adapter.lora.alpha", *alpha});
  return m;
}

std::vector<float> Ramp(std::size_t n, float base) {
  std::vector<float> v(n);
  for (std::size_t i = 0; i < n; ++i)
    v[i] = base + static_cast<float>(i);
  return v;
}

std::filesystem::path Write(const std::filesystem::path& dir,
                            const std::vector<std::uint8_t>& image) {
  static int counter = 0;
  const auto path = dir / ("adapter" + std::to_string(counter++) + ".gguf");
  std::ofstream(path, std::ios::binary)
      .write(reinterpret_cast<const char*>(image.data()),
             static_cast<std::streamsize>(image.size()));
  return path;
}

std::optional<gufo::core::LoraAdapter> Load(
    const std::filesystem::path& dir, const std::vector<std::uint8_t>& image,
    std::string* error) {
  return gufo::core::LoadLoraAdapter(Write(dir, image), "toy", error);
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
  char temp[] = "/tmp/gufo-lora-XXXXXX";
  Expect(mkdtemp(temp) != nullptr, "create test directory");
  const std::filesystem::path dir(temp);
  gufo::platform::SetEnv("XDG_CACHE_HOME", temp);

  // Dense: in 6, rank 2, out 5. Experts: in 4, rank 4, out 3, 2 experts.
  const Tensor dense_a{"blk.1.out.weight.lora_a", {6, 2}, Ramp(12, -6.0f)};
  const Tensor dense_b{
      "blk.1.out.weight.lora_b", {2, 5}, Ramp(10, 1.0f), GgmlType::kF16};
  const Tensor exps_a{
      "blk.0.down_exps.weight.lora_a", {4, 4, 2}, Ramp(32, 0.0f)};
  const Tensor exps_b{
      "blk.0.down_exps.weight.lora_b", {4, 3, 2}, Ramp(24, -12.0f)};
  {
    std::string error;
    const auto adapter = Load(
        dir, Build(Metadata(), {dense_a, exps_a, dense_b, exps_b}), &error);
    Expect(adapter.has_value(), "valid adapter loads: " + error);
    Expect(adapter->identity.size() == 64, "identity is a SHA-256 digest");
    Expect(adapter->targets.size() == 2, "one target per lora_a/lora_b pair");
    const auto& dense = adapter->targets[0];
    Expect(dense.tensor == "blk.1.out.weight" && dense.in == 6 &&
               dense.rank == 2 && dense.out == 5 && dense.experts == 0,
           "dense shape");
    Expect(dense.scale == 4.0f, "scale is alpha / rank");
    Expect(dense.a == dense_a.values && dense.b == dense_b.values,
           "F32 and F16 halves load as F32");
    const auto& exps = adapter->targets[1];
    Expect(exps.tensor == "blk.0.down_exps.weight" && exps.in == 4 &&
               exps.rank == 4 && exps.out == 3 && exps.experts == 2 &&
               exps.scale == 2.0f && exps.a == exps_a.values,
           "per-expert shape");
    Expect(gufo::core::AdapterType(
               Write(dir, Build(Metadata(), {dense_a, dense_b}))) == "lora",
           "AdapterType reads adapter.type");
  }
  for (const std::optional<float> alpha :
       {std::optional<float>(), std::optional<float>(0.0f)}) {
    std::string error;
    const auto adapter = Load(
        dir, Build(Metadata("toy", "lora", alpha), {dense_a, dense_b}), &error);
    Expect(adapter && adapter->targets[0].scale == 1.0f,
           "no alpha or alpha 0 scales by 1: " + error);
  }
  ExpectRejected(dir, Build(Metadata("other"), {dense_a, dense_b}),
                 "another model", "not toy");
  ExpectRejected(dir, Build(Metadata("toy", "direction"), {dense_a, dense_b}),
                 "a direction adapter", "not a LoRA adapter");
  ExpectRejected(dir, Build(Metadata(), {dense_a}), "a missing half",
                 "lacks its lora_b");
  ExpectRejected(
      dir,
      Build(Metadata(),
            {dense_a, {"blk.1.out.weight.lora_b", {3, 5}, Ramp(15, 1.0f)}}),
      "mismatched ranks", "ne = [in, rank");
  ExpectRejected(dir,
                 Build(Metadata(), {{"blk.1.out.weight", {6}, Ramp(6, 1.0f)}}),
                 "a stray tensor", "unexpected tensor");
  ExpectRejected(dir, Build(Metadata(), {}), "no tensors", "no LoRA tensors");
  std::string error;
  Expect(gufo::core::AdapterType(
             Write(dir, Build({{"general.architecture", std::string("toy")}},
                              {dense_a, dense_b})),
             &error)
                 .empty() &&
             error.find("not a GGUF adapter") != std::string::npos,
         "AdapterType rejects a model file: " + error);
  std::filesystem::remove_all(dir);
  std::cout << "lora adapter tests passed\n";
  return 0;
}
