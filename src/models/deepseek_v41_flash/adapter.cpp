#include "src/models/deepseek_v41_flash/adapter.hpp"

#include <array>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <string_view>

#include "src/core/gguf_identity.hpp"
#include "src/models/deepseek_v41_flash/runtime/model_data_internal.h"

namespace gufo::models::deepseek_v41_flash {
namespace {

constexpr std::string_view kArchitecture = "deepseek41";

bool Fail(std::string* error_msg, std::string message) {
  if (error_msg)
    *error_msg = std::move(message);
  return false;
}

std::string ScaleText(float scale) {
  char text[32];
  std::snprintf(text, sizeof(text), "%.9g", scale);
  return text;
}

// "blk.{layer}.{target}.weight" -> layer and DS41_LORA_* target.
bool ParseTarget(std::string_view name, uint32_t* layer, uint32_t* target) {
  constexpr std::string_view kPrefix = "blk.", kSuffix = ".weight";
  if (!name.starts_with(kPrefix) || !name.ends_with(kSuffix))
    return false;
  name.remove_prefix(kPrefix.size());
  name.remove_suffix(kSuffix.size());
  const auto dot = name.find('.');
  if (dot == 0 || dot == std::string_view::npos)
    return false;
  const auto [ptr, ec] =
      std::from_chars(name.data(), name.data() + dot, *layer);
  if (ec != std::errc{} || ptr != name.data() + dot || *layer >= DS41_N_LAYER)
    return false;
  const std::string_view matrix = name.substr(dot + 1);
  for (uint32_t t = 0; t < DS41_N_LORA_TARGETS; ++t)
    if (matrix == DS41_LORA_TARGETS[t]) {
      *target = t;
      return true;
    }
  return false;
}

}  // namespace

bool Adapter::Load(const std::string& path, float scale,
                   std::string* error_msg) {
  if (!std::isfinite(scale))
    return Fail(error_msg, "--adapter-scale must be finite");
  std::string error;
  const std::string type = core::AdapterType(path, &error);
  if (!error.empty())
    return Fail(error_msg, error);
  if (type == "direction") {
    static constexpr std::array<std::string_view, DS41_N_DIRECTION_SITES>
        kSites = {DS41_DIRECTION_SITES[0], DS41_DIRECTION_SITES[1],
                  DS41_DIRECTION_SITES[2]};
    directions_ = core::LoadDirectionAdapter(path,
                                             {.architecture = kArchitecture,
                                              .width = DS41_N_EMBD,
                                              .layers = DS41_N_LAYER,
                                              .sites = kSites},
                                             &error);
    if (!directions_)
      return Fail(error_msg, error);
    for (const auto& set : directions_->sets)
      sets_.push_back({.layer = set.layer,
                       .site = set.site,
                       .count = set.count,
                       .vectors = set.vectors.data()});
    direction_scale_ = directions_->scale * scale;
    identity_ = "direction:" + std::string(core::kGgufIdentityScheme) + ':' +
                directions_->identity +
                ":mode=" + std::string(core::ToString(directions_->mode)) +
                ":scale=" + ScaleText(direction_scale_);
    return true;
  }
  if (type == "lora") {
    lora_ = core::LoadLoraAdapter(path, kArchitecture, &error);
    if (!lora_)
      return Fail(error_msg, error);
    for (const auto& t : lora_->targets) {
      uint32_t layer = 0, target = 0;
      if (!ParseTarget(t.tensor, &layer, &target)) {
        std::string supported;
        for (const char* name : DS41_LORA_TARGETS)
          supported += std::string(supported.empty() ? "" : ", ") + name;
        return Fail(error_msg, path + ": LoRA on " + t.tensor +
                                   " is not supported; DeepSeek V4.1 Flash "
                                   "takes LoRA on " +
                                   supported);
      }
      matrices_.push_back({.layer = layer,
                           .target = target,
                           .rank = t.rank,
                           .in = t.in,
                           .out = t.out,
                           .experts = t.experts,
                           .scale = t.scale * scale,
                           .a = t.a.data(),
                           .b = t.b.data()});
    }
    identity_ = "lora:" + std::string(core::kGgufIdentityScheme) + ':' +
                lora_->identity + ":scale=" + ScaleText(scale);
    return true;
  }
  return Fail(error_msg, path + ": adapter.type '" + type +
                             "' is not supported (direction or lora)");
}

void Adapter::Apply(ds41_engine_options* options) const {
  options->directions = sets_.data();
  options->n_directions = static_cast<uint32_t>(sets_.size());
  options->direction_scale = direction_scale_;
  options->direction_add =
      directions_ && directions_->mode == core::DirectionMode::kAdd;
  options->loras = matrices_.data();
  options->n_loras = static_cast<uint32_t>(matrices_.size());
}

}  // namespace gufo::models::deepseek_v41_flash
