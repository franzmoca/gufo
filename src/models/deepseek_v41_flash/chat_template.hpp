#ifndef GUFO_MODELS_DEEPSEEK_V41_FLASH_CHAT_TEMPLATE_HPP_
#define GUFO_MODELS_DEEPSEEK_V41_FLASH_CHAT_TEMPLATE_HPP_

#include <cstddef>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/core/reasoning.hpp"

namespace gufo::models::deepseek_v41_flash {

struct ChatMessage {
  struct ToolArgument {
    std::string name;
    std::string value;
    bool is_string{true};
  };

  struct ToolCall {
    std::string name;
    std::vector<ToolArgument> arguments;
    std::string id;
  };

  std::string role;
  std::string content;
  std::string reasoning_content;
  std::vector<ToolCall> tool_calls;
  std::string tool_call_id;
  /// Bytes of `content` before which an image goes, in order; equal offsets
  /// keep their order. Images render as kImagePlaceholder, joined with the
  /// text around them by blank lines as the official encoder joins content
  /// blocks. User and tool messages only.
  std::vector<std::size_t> image_offsets;
};

/// One image in a rendered prompt; prepare_vl_inputs expands it into the
/// image's span.
inline constexpr std::string_view kImagePlaceholder = "<｜deepseek_image｜>";

struct ChatTool {
  std::string name;
  std::string description;
  std::string parameters_json{"{}"};
  /// Complete function definition, including optional fields such as strict.
  std::string definition_json;
};

struct ChatTemplateOptions {
  // Literal encoder options; user-facing defaults are resolved below.
  bool enable_thinking{false};
  ReasoningEffort reasoning_effort{ReasoningEffort::kLow};
  bool preserve_thinking{false};
  bool tools_present{false};
};

/// User-facing defaults follow DeepSeek's API: thinking enabled, high effort.
/// Explicit options override them; the checkpoint encoder itself has no
/// default thinking mode.
[[nodiscard]] ChatTemplateOptions ResolveDeepSeekChatOptions(
    const ReasoningOptions& reasoning);

[[nodiscard]] std::string_view GenerationPrompt(bool enable_thinking);

/// Renders the pinned DeepSeek V4.1 Flash 1017 conversation format. Throws
/// std::invalid_argument for text containing kImagePlaceholder, which the
/// official encoder rejects, and for invalid image offsets.
[[nodiscard]] std::string RenderChat(std::span<const ChatMessage> messages,
                                     const ChatTemplateOptions& options = {});
[[nodiscard]] std::string RenderChat(std::span<const ChatMessage> messages,
                                     std::span<const ChatTool> tools,
                                     const ChatTemplateOptions& options = {});

/// Rejects DeepSeek artifacts whose embedded template is not the pinned
/// converter version implemented by the compiled formatter.
[[nodiscard]] bool ValidateGgufTemplate(const core::GgufReader& reader,
                                        std::string* error_msg = nullptr);

[[nodiscard]] constexpr std::string_view ChatTemplateId() noexcept {
  return "deepseek-v4.1-flash-1017-compiled-v1";
}

[[nodiscard]] constexpr std::string_view ArtifactTemplateSha256() noexcept {
  return "6f9bd1f87d9dad376b4c79683934b4254396d826ae514ab0244efa904ce1d1a4";
}

[[nodiscard]] constexpr std::string_view EncoderReferenceSha256() noexcept {
  return "13231c592e958fb39e2af6001dae69d90e7a36b265702b9b390236fb387599e2";
}

/// Native DeepSeek effort name after mapping provider-neutral effort levels.
[[nodiscard]] std::string_view DeepSeekReasoningEffortName(
    ReasoningEffort effort) noexcept;

}  // namespace gufo::models::deepseek_v41_flash

#endif  // GUFO_MODELS_DEEPSEEK_V41_FLASH_CHAT_TEMPLATE_HPP_
