#ifndef GUFO_MODELS_DEEPSEEK_V41_FLASH_ADAPTER_HPP_
#define GUFO_MODELS_DEEPSEEK_V41_FLASH_ADAPTER_HPP_

#include <optional>
#include <string>
#include <vector>

#include "src/core/direction_adapter.hpp"
#include "src/core/lora_adapter.hpp"
#include "src/models/deepseek_v41_flash/runtime/model.h"

namespace gufo::models::deepseek_v41_flash {

/// A loaded --adapter file: a direction adapter or a LoRA adapter on the
/// matrices that write into the residual streams (DS41_LORA_TARGETS). It
/// owns the host data the engine options point into until the engine opens.
class Adapter {
public:
  /// Loads `path` (its adapter.type picks the kind) with the run-time scale.
  [[nodiscard]] bool Load(const std::string& path, float scale,
                          std::string* error_msg);
  /// Points the engine options at the loaded data (nothing when empty).
  void Apply(ds41_engine_options* options) const;
  /// "none", or the kind, file digest, mode and effective scale, for cache
  /// identities.
  [[nodiscard]] const std::string& Identity() const noexcept {
    return identity_;
  }

private:
  std::optional<core::DirectionAdapter> directions_;
  std::optional<core::LoraAdapter> lora_;
  std::vector<ds41_direction_set> sets_;
  std::vector<ds41_lora_matrix> matrices_;
  float direction_scale_ = 0.0f;
  std::string identity_ = "none";
};

}  // namespace gufo::models::deepseek_v41_flash

#endif  // GUFO_MODELS_DEEPSEEK_V41_FLASH_ADAPTER_HPP_
