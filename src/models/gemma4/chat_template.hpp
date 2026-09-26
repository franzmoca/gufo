#ifndef GUFO_MODELS_GEMMA4_CHAT_TEMPLATE_HPP_
#define GUFO_MODELS_GEMMA4_CHAT_TEMPLATE_HPP_

#include <cstddef>
#include <optional>
#include <span>
#include <string>
#include <string_view>

#include "src/core/gguf_reader.hpp"
#include "src/core/reasoning.hpp"
#include "src/models/qwen/chat_template.hpp"

namespace gufo::models::gemma4 {

struct ChatOptions {
  bool add_generation_prompt{true};
  /// Gemma 4 thinking is opt-in: the template default is off.
  bool enable_thinking{false};
  bool preserve_thinking{false};
};

/// Resolves provider-neutral controls against the Gemma 4 template defaults.
/// Effort levels have no Gemma equivalent; any level other than minimal
/// enables thinking when it was not set explicitly.
[[nodiscard]] ChatOptions ResolveChatOptions(const ReasoningOptions& reasoning);

struct RenderedPrompt {
  std::string text;
  /// Byte offset where the generation prompt starts; equals text.size()
  /// without one. Everything before it is the stable conversation prefix.
  std::size_t generation_prompt_offset{0};
};

/// Compiled renderer of Unsloth's Gemma 4 chat template. The GGUF template
/// is only accepted when its SHA-256 matches a qualified revision; the Jinja
/// source is kept in reference/ as data.
///
/// Known, documented approximations of the Jinja source: message content
/// parts arrive flattened by the HTTP layer, JSON numbers render as Python
/// ints when integral (Python would print 5.0 for a float literal), and
/// images are rejected because this runtime is text-only.
class ChatTemplate {
public:
  [[nodiscard]] static constexpr std::string_view
  UnslothTemplateSha256() noexcept {
    return "845f1ee48e39fc942fe190da9df6a1c5db229e17a96ea08966ad1c9274e73d1b";
  }
  [[nodiscard]] static constexpr std::string_view TemplateId() noexcept {
    return "gemma4-unsloth-845f1ee4";
  }

  [[nodiscard]] static bool ValidateGgufTemplate(
      const core::GgufReader& reader, std::string* error_msg = nullptr);

  [[nodiscard]] static std::optional<RenderedPrompt> Render(
      std::span<const tokenization::ChatMessage> messages,
      std::span<const tokenization::ChatTool> tools, const ChatOptions& options,
      std::string* error_msg = nullptr);
};

/// Reasoning channel markup produced by the model.
inline constexpr std::string_view kThoughtStart = "<|channel>thought\n";
inline constexpr std::string_view kThoughtEnd = "<channel|>";
inline constexpr std::string_view kToolCallStart = "<|tool_call>";
inline constexpr std::string_view kToolCallEnd = "<tool_call|>";
/// String delimiter of the tool-call argument syntax.
inline constexpr std::string_view kStringQuote = "<|\"|>";

}  // namespace gufo::models::gemma4

#endif  // GUFO_MODELS_GEMMA4_CHAT_TEMPLATE_HPP_
