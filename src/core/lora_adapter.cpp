#include "src/core/lora_adapter.hpp"

#include <bit>
#include <cmath>
#include <cstring>
#include <exception>
#include <map>
#include <string>

#include "src/core/gguf_identity.hpp"
#include "src/core/gguf_reader.hpp"

namespace gufo::core {
namespace {

bool Fail(std::string* error_msg, std::string message) {
  if (error_msg)
    *error_msg = std::move(message);
  return false;
}

float HalfToFloat(std::uint16_t h) {
  const std::uint32_t sign = (std::uint32_t)(h & 0x8000) << 16;
  std::uint32_t exp = (h >> 10) & 0x1f, mant = h & 0x3ff;
  std::uint32_t bits;
  if (exp == 0) {
    if (mant == 0) {
      bits = sign;
    } else {
      exp = 127 - 15 + 1;
      while (!(mant & 0x400)) {
        mant <<= 1;
        --exp;
      }
      bits = sign | (exp << 23) | ((mant & 0x3ff) << 13);
    }
  } else if (exp == 31) {
    bits = sign | 0x7f800000 | (mant << 13);
  } else {
    bits = sign | ((exp + 127 - 15) << 23) | (mant << 13);
  }
  return std::bit_cast<float>(bits);
}

bool ToFloat(const GgufTensorInfo& t, std::vector<float>* out) {
  const std::uint64_t n = t.ElementCount();
  out->resize(n);
  const auto* p = static_cast<const std::uint8_t*>(t.data);
  switch (t.type) {
    case GgmlType::kF32:
      if (t.size_bytes != n * 4)
        return false;
      std::memcpy(out->data(), p, n * 4);
      return true;
    case GgmlType::kF16:
    case GgmlType::kBF16:
      if (t.size_bytes != n * 2)
        return false;
      for (std::uint64_t i = 0; i < n; ++i) {
        std::uint16_t h;
        std::memcpy(&h, p + 2 * i, 2);
        (*out)[i] = t.type == GgmlType::kF16
                        ? HalfToFloat(h)
                        : std::bit_cast<float>((std::uint32_t)h << 16);
      }
      return true;
    default:
      return false;
  }
}

}  // namespace

std::string AdapterType(const std::filesystem::path& path,
                        std::string* error_msg) {
  std::string open_error;
  const auto reader = GgufReader::OpenFile(path, &open_error);
  if (!reader) {
    Fail(error_msg, path.string() + ": " + open_error);
    return {};
  }
  if (reader->GetMetadataString("general.type") != "adapter") {
    Fail(error_msg, path.string() +
                        " is not a GGUF adapter "
                        "(general.type = adapter)");
    return {};
  }
  return std::string(reader->GetMetadataString("adapter.type").value_or(""));
}

std::optional<LoraAdapter> LoadLoraAdapter(const std::filesystem::path& path,
                                           std::string_view architecture,
                                           std::string* error_msg) {
  const std::string file = path.string();
  std::string open_error;
  const auto reader = GgufReader::OpenFile(path, &open_error);
  if (!reader) {
    Fail(error_msg, file + ": " + open_error);
    return std::nullopt;
  }
  if (reader->GetMetadataString("general.type") != "adapter" ||
      reader->GetMetadataString("adapter.type") != "lora") {
    Fail(error_msg, file +
                        " is not a LoRA adapter (general.type = adapter, "
                        "adapter.type = lora)");
    return std::nullopt;
  }
  const auto arch = reader->GetMetadataString("general.architecture");
  if (arch != architecture) {
    Fail(error_msg, file + " is an adapter for " +
                        std::string(arch.value_or("an unnamed model")) +
                        ", not " + std::string(architecture));
    return std::nullopt;
  }
  std::optional<float> alpha;
  if (reader->FindMetadata("adapter.lora.alpha")) {
    alpha = reader->GetMetadataFloat32("adapter.lora.alpha");
    if (!alpha || !std::isfinite(*alpha)) {
      Fail(error_msg, file + ": adapter.lora.alpha must be a finite float32");
      return std::nullopt;
    }
  }
  struct Pair {
    const GgufTensorInfo* a = nullptr;
    const GgufTensorInfo* b = nullptr;
  };
  std::map<std::string, Pair> pairs;
  std::vector<std::string> order;
  for (const auto& t : reader->GetTensors()) {
    const std::string_view name = t.name;
    const bool is_a = name.ends_with(".lora_a"),
               is_b = name.ends_with(".lora_b");
    if (!is_a && !is_b) {
      Fail(error_msg, file + ": unexpected tensor " + std::string(name));
      return std::nullopt;
    }
    const std::string base(name.substr(0, name.size() - 7));
    auto [it, inserted] = pairs.try_emplace(base);
    if (inserted)
      order.push_back(base);
    (is_a ? it->second.a : it->second.b) = &t;
  }
  LoraAdapter adapter;
  for (const auto& base : order) {
    const Pair& p = pairs[base];
    if (!p.a || !p.b) {
      Fail(error_msg,
           file + ": " + base + " lacks its lora_" + (p.a ? "b" : "a"));
      return std::nullopt;
    }
    const auto& da = p.a->dimensions;
    const auto& db = p.b->dimensions;
    const bool experts = da.size() == 3;
    if (da.size() < 2 || da.size() > 3 || db.size() != da.size() ||
        da[1] == 0 || db[0] != da[1] || da[0] == 0 || db[1] == 0 ||
        (experts && (db[2] != da[2] || da[2] == 0))) {
      Fail(error_msg, file + ": " + base +
                          " halves are not ne = [in, rank(, experts)] and "
                          "[rank, out(, experts)]");
      return std::nullopt;
    }
    LoraTarget target;
    target.tensor = base;
    target.in = static_cast<std::uint32_t>(da[0]);
    target.rank = static_cast<std::uint32_t>(da[1]);
    target.out = static_cast<std::uint32_t>(db[1]);
    target.experts = experts ? static_cast<std::uint32_t>(da[2]) : 0;
    target.scale = alpha && *alpha != 0.0f
                       ? *alpha / static_cast<float>(target.rank)
                       : 1.0f;
    if (!ToFloat(*p.a, &target.a) || !ToFloat(*p.b, &target.b)) {
      Fail(error_msg, file + ": " + base + " halves must be F32, F16 or BF16");
      return std::nullopt;
    }
    for (const auto* v : {&target.a, &target.b})
      for (float x : *v)
        if (!std::isfinite(x)) {
          Fail(error_msg, file + ": " + base + " has non-finite values");
          return std::nullopt;
        }
    adapter.targets.push_back(std::move(target));
  }
  if (adapter.targets.empty()) {
    Fail(error_msg, file + " has no LoRA tensors");
    return std::nullopt;
  }
  try {
    adapter.identity = GgufIdentityHex(*reader);
  } catch (const std::exception& e) {
    Fail(error_msg, file + ": " + e.what());
    return std::nullopt;
  }
  return adapter;
}

}  // namespace gufo::core
