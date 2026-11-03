#include "src/core/direction_adapter.hpp"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstring>
#include <exception>
#include <string>
#include <tuple>

#include "src/core/gguf_identity.hpp"
#include "src/core/gguf_reader.hpp"

namespace gufo::core {
namespace {

bool Fail(std::string* error_msg, std::string message) {
  if (error_msg)
    *error_msg = std::move(message);
  return false;
}

// blk.{layer}.{site}.direction
bool ParseName(std::string_view name, std::uint32_t* layer,
               std::string_view* site) {
  constexpr std::string_view kPrefix = "blk.", kSuffix = ".direction";
  if (!name.starts_with(kPrefix) || !name.ends_with(kSuffix))
    return false;
  name.remove_prefix(kPrefix.size());
  name.remove_suffix(kSuffix.size());
  const auto dot = name.find('.');
  if (dot == 0 || dot == std::string_view::npos || dot + 1 == name.size())
    return false;
  const auto* end = name.data() + dot;
  const auto [ptr, ec] = std::from_chars(name.data(), end, *layer);
  if (ec != std::errc{} || ptr != end)
    return false;
  *site = name.substr(dot + 1);
  return true;
}

}  // namespace

std::string_view ToString(DirectionMode mode) noexcept {
  return mode == DirectionMode::kAdd ? "add" : "project";
}

std::optional<DirectionAdapter> LoadDirectionAdapter(
    const std::filesystem::path& path, const DirectionAdapterShape& shape,
    std::string* error_msg) {
  const std::string file = path.string();
  std::string open_error;
  const auto reader = GgufReader::OpenFile(path, &open_error);
  if (!reader) {
    Fail(error_msg, file + ": " + open_error);
    return std::nullopt;
  }
  if (reader->GetMetadataString("general.type") != "adapter" ||
      reader->GetMetadataString("adapter.type") != "direction") {
    Fail(error_msg, file +
                        " is not a direction adapter (general.type = "
                        "adapter, adapter.type = direction)");
    return std::nullopt;
  }
  const auto arch = reader->GetMetadataString("general.architecture");
  if (arch != shape.architecture) {
    Fail(error_msg, file + " is an adapter for " +
                        std::string(arch.value_or("an unnamed model")) +
                        ", not " + std::string(shape.architecture));
    return std::nullopt;
  }
  DirectionAdapter adapter;
  adapter.width = shape.width;
  const auto mode = reader->GetMetadataString("adapter.direction.mode");
  if (mode == "project") {
    adapter.mode = DirectionMode::kProject;
  } else if (mode == "add") {
    adapter.mode = DirectionMode::kAdd;
  } else {
    Fail(error_msg,
         file + ": adapter.direction.mode must be \"project\" or \"add\"");
    return std::nullopt;
  }
  if (reader->FindMetadata("adapter.direction.scale")) {
    const auto scale = reader->GetMetadataFloat32("adapter.direction.scale");
    if (!scale || !std::isfinite(*scale)) {
      Fail(error_msg, file +
                          ": adapter.direction.scale must be a finite "
                          "float32");
      return std::nullopt;
    }
    adapter.scale = *scale;
  }
  for (const auto& tensor : reader->GetTensors()) {
    std::uint32_t layer = 0;
    std::string_view site_name;
    if (!ParseName(tensor.name, &layer, &site_name)) {
      Fail(error_msg, file + ": unexpected tensor " + std::string(tensor.name));
      return std::nullopt;
    }
    const auto site = std::ranges::find(shape.sites, site_name);
    if (site == shape.sites.end()) {
      Fail(error_msg, file + ": " + std::string(tensor.name) + " names site '" +
                          std::string(site_name) +
                          "', which this model does not have");
      return std::nullopt;
    }
    if (layer >= shape.layers) {
      Fail(error_msg, file + ": " + std::string(tensor.name) +
                          " is past the model's " +
                          std::to_string(shape.layers) + " layers");
      return std::nullopt;
    }
    const auto& dims = tensor.dimensions;
    if (tensor.type != GgmlType::kF32 || dims.empty() || dims.size() > 2 ||
        dims[0] != shape.width || (dims.size() == 2 && dims[1] == 0)) {
      Fail(error_msg, file + ": " + std::string(tensor.name) +
                          " must be F32 with ne = [" +
                          std::to_string(shape.width) + ", k]");
      return std::nullopt;
    }
    DirectionSet set;
    set.layer = layer;
    set.site = static_cast<std::uint32_t>(site - shape.sites.begin());
    set.count = dims.size() == 2 ? static_cast<std::uint32_t>(dims[1]) : 1;
    set.vectors.resize(std::size_t{set.count} * shape.width);
    if (tensor.size_bytes != set.vectors.size() * sizeof(float)) {
      Fail(error_msg, file + ": " + std::string(tensor.name) + " is truncated");
      return std::nullopt;
    }
    std::memcpy(set.vectors.data(), tensor.data, tensor.size_bytes);
    if (!std::ranges::all_of(set.vectors,
                             [](float v) { return std::isfinite(v); })) {
      Fail(error_msg,
           file + ": " + std::string(tensor.name) + " has non-finite values");
      return std::nullopt;
    }
    if (std::ranges::all_of(set.vectors, [](float v) { return v == 0.0f; }))
      continue;
    adapter.sets.push_back(std::move(set));
  }
  std::ranges::sort(adapter.sets, {}, [](const DirectionSet& s) {
    return std::tuple(s.layer, s.site);
  });
  for (std::size_t i = 1; i < adapter.sets.size(); ++i) {
    if (adapter.sets[i].layer == adapter.sets[i - 1].layer &&
        adapter.sets[i].site == adapter.sets[i - 1].site) {
      Fail(error_msg, file + ": duplicate directions for layer " +
                          std::to_string(adapter.sets[i].layer));
      return std::nullopt;
    }
  }
  if (adapter.sets.empty()) {
    Fail(error_msg, file + " has no nonzero directions");
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
