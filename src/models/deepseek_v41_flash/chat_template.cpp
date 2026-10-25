#include "src/models/deepseek_v41_flash/chat_template.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "src/core/crypto/sha256.hpp"
#include "src/models/deepseek_v41_flash/tool_calls.hpp"

namespace gufo::models::deepseek_v41_flash {
namespace {

constexpr std::string_view kBos = "<｜begin▁of▁sentence｜>";
constexpr std::string_view kEos = "<｜end▁of▁sentence｜>";
constexpr std::string_view kUser = "<｜User｜>";
constexpr std::string_view kAssistant = "<｜Assistant｜>";
constexpr std::string_view kThinkStart = "<think>";
constexpr std::string_view kThinkEnd = "</think>";

constexpr std::string_view kSystem = "<｜System｜>";

/// V4.1 reasoning effort is a 1-100 budget; deepseek-recipe maps the API
/// levels to 50 (low), 75 (default, high, xhigh) and 100 (max).
int EffortBudget(ReasoningEffort effort) noexcept {
  switch (effort) {
    case ReasoningEffort::kMinimal:
    case ReasoningEffort::kLow:
      return 50;
    case ReasoningEffort::kMedium:
    case ReasoningEffort::kHigh:
    case ReasoningEffort::kXHigh:
      return 75;
    case ReasoningEffort::kMax:
      return 100;
  }
  return 75;
}

void AppendEffortPrompt(std::string& output, ReasoningEffort effort) {
  output.append("Reasoning Effort: ");
  output.append(std::to_string(EffortBudget(effort)));
  output.append(
      " (range 1-100, the higher the value, the more thorough the "
      "reasoning)\n\n");
}

std::string Sha256(std::string_view value) {
  const auto* begin = reinterpret_cast<const std::uint8_t*>(value.data());
  return crypto::Sha256Hex({begin, value.size()});
}

void AssignError(std::string* error_msg, std::string message) {
  if (error_msg != nullptr) {
    *error_msg = std::move(message);
  }
}

void AppendJsonString(std::string& output, std::string_view value) {
  output.push_back('"');
  for (const unsigned char character : value) {
    switch (character) {
      case '"':
        output.append("\\\"");
        break;
      case '\\':
        output.append("\\\\");
        break;
      case '\b':
        output.append("\\b");
        break;
      case '\f':
        output.append("\\f");
        break;
      case '\n':
        output.append("\\n");
        break;
      case '\r':
        output.append("\\r");
        break;
      case '\t':
        output.append("\\t");
        break;
      default:
        output.push_back(static_cast<char>(character));
        break;
    }
  }
  output.push_back('"');
}

std::string PythonJsonSpacing(std::string_view value) {
  std::string output;
  output.reserve(value.size() + value.size() / 8);
  bool in_string = false;
  bool escaped = false;
  for (const char character : value) {
    output.push_back(character);
    if (in_string) {
      if (escaped) {
        escaped = false;
      } else if (character == '\\') {
        escaped = true;
      } else if (character == '"') {
        in_string = false;
      }
      continue;
    }
    if (character == '"') {
      in_string = true;
    } else if (character == ':' || character == ',') {
      output.push_back(' ');
    }
  }
  return output;
}

void AppendToolsPrompt(std::string& output, std::span<const ChatTool> tools) {
  if (tools.empty()) {
    return;
  }
  output.append(
      "## Tools\n\n"
      "You have access to a set of tools to help answer the user's "
      "question. You can invoke tools by writing a \"");
  output.append(DeepSeekToolSyntax::kCallsOpen);
  output.append("\" block like the following:\n\n");
  output.append(DeepSeekToolSyntax::kCallsOpen);
  output.push_back('\n');
  output.append(DeepSeekToolSyntax::kInvokeOpens[0]);
  output.push_back(' ');
  output.append(DeepSeekToolSyntax::kNameAttribute);
  output.append("$TOOL_NAME\">\n");
  output.append(DeepSeekToolSyntax::kParameterOpens[0]);
  output.push_back(' ');
  output.append(DeepSeekToolSyntax::kNameAttribute);
  output.append("$PARAMETER_NAME\" ");
  output.append(DeepSeekToolSyntax::kStringAttribute);
  output.append("true|false\">$PARAMETER_VALUE");
  output.append(DeepSeekToolSyntax::kParameterCloses[0]);
  output.append("\n...\n");
  output.append(DeepSeekToolSyntax::kInvokeCloses[0]);
  output.push_back('\n');
  output.append(DeepSeekToolSyntax::kInvokeOpens[0]);
  output.push_back(' ');
  output.append(DeepSeekToolSyntax::kNameAttribute);
  output.append("$TOOL_NAME2\">\n...\n");
  output.append(DeepSeekToolSyntax::kInvokeCloses[0]);
  output.push_back('\n');
  output.append(DeepSeekToolSyntax::kCallsClose);
  output.append(
      "\n\n"
      "String parameters should be specified as is and set `string=\"true\"`. "
      "For all other types (numbers, booleans, arrays, objects), pass the "
      "value in JSON format and set `string=\"false\"`.\n\n"
      "If thinking_mode is enabled (triggered by <think>), you MUST output "
      "your complete reasoning inside <think>...</think> BEFORE any tool "
      "calls or final response.\n\n"
      "Otherwise, output directly after </think> with tool calls or final "
      "response.\n\n"
      "### Available Tool Schemas\n\n");
  for (std::size_t index = 0; index < tools.size(); ++index) {
    const auto& tool = tools[index];
    if (!tool.definition_json.empty()) {
      output.append(PythonJsonSpacing(tool.definition_json));
      if (index + 1 < tools.size())
        output.push_back('\n');
      continue;
    }
    output.append("{\"name\": ");
    AppendJsonString(output, tool.name);
    output.append(", \"description\": ");
    AppendJsonString(output, tool.description);
    output.append(", \"parameters\": ");
    output.append(tool.parameters_json.empty()
                      ? "{}"
                      : PythonJsonSpacing(tool.parameters_json));
    output.push_back('}');
    if (index + 1 < tools.size()) {
      output.push_back('\n');
    }
  }
  output.append(
      "\n\nYou MUST strictly follow the above defined tool name and parameter "
      "schemas to invoke tool calls.\n");
}

void AppendToolCalls(std::string& output,
                     std::span<const ChatMessage::ToolCall> calls) {
  if (calls.empty()) {
    return;
  }
  output.append("\n\n");
  output.append(DeepSeekToolSyntax::kCallsOpen);
  output.push_back('\n');
  for (std::size_t call_index = 0; call_index < calls.size(); ++call_index) {
    const auto& call = calls[call_index];
    output.append(DeepSeekToolSyntax::kInvokeOpens[0]);
    output.push_back(' ');
    output.append(DeepSeekToolSyntax::kNameAttribute);
    output.append(call.name);
    output.append("\">\n");
    for (std::size_t argument_index = 0; argument_index < call.arguments.size();
         ++argument_index) {
      const auto& argument = call.arguments[argument_index];
      output.append(DeepSeekToolSyntax::kParameterOpens[0]);
      output.push_back(' ');
      output.append(DeepSeekToolSyntax::kNameAttribute);
      output.append(argument.name);
      output.append("\" ");
      output.append(DeepSeekToolSyntax::kStringAttribute);
      output.append(argument.is_string ? "true" : "false");
      output.append("\">");
      output.append(argument.value);
      output.append(DeepSeekToolSyntax::kParameterCloses[0]);
      if (argument_index + 1 < call.arguments.size()) {
        output.push_back('\n');
      }
    }
    output.push_back('\n');
    output.append(DeepSeekToolSyntax::kInvokeCloses[0]);
    if (call_index + 1 < calls.size()) {
      output.push_back('\n');
    }
  }
  output.push_back('\n');
  output.append(DeepSeekToolSyntax::kCallsClose);
}

}  // namespace

ChatTemplateOptions ResolveDeepSeekChatOptions(
    const ReasoningOptions& reasoning) {
  return {
      .enable_thinking = reasoning.enabled.value_or(true),
      .reasoning_effort = reasoning.effort.value_or(ReasoningEffort::kHigh),
      .preserve_thinking = reasoning.preserve_thinking.value_or(false),
  };
}

std::string_view DeepSeekReasoningEffortName(ReasoningEffort effort) noexcept {
  switch (EffortBudget(effort)) {
    case 50:
      return "low";
    case 100:
      return "max";
    default:
      return "high";
  }
}

std::string RenderChat(std::span<const ChatMessage> messages,
                       const ChatTemplateOptions& options) {
  return RenderChat(messages, {}, options);
}

/// Renders the official V4.1 encoding with deepseek-recipe serving rules:
/// request tools join a leading system message (inserted empty when the
/// conversation has none), `developer` is a system message, and earlier
/// reasoning is rendered only with tools or when explicitly preserved.
std::string RenderChat(std::span<const ChatMessage> messages,
                       std::span<const ChatTool> tools,
                       const ChatTemplateOptions& options) {
  const auto is_system = [](std::string_view role) {
    return role == "system" || role == "developer";
  };
  const auto is_user_turn = [](std::string_view role) {
    return role == "user" || role == "tool" || role == "function";
  };
  const bool thinking = options.enable_thinking;
  const bool drop_thinking =
      !options.preserve_thinking && !options.tools_present && tools.empty();
  const bool insert_system =
      !tools.empty() && (messages.empty() || !is_system(messages[0].role));

  std::string output(kBos);
  if (thinking || insert_system ||
      (!messages.empty() && is_system(messages[0].role)))
    output.append(kSystem);
  if (thinking)
    AppendEffortPrompt(output, options.reasoning_effort);
  if (insert_system) {
    output.append("\n\n");
    AppendToolsPrompt(output, tools);
  }

  std::unordered_map<std::string_view, std::size_t> tool_call_order;
  for (std::size_t index = 0; index < messages.size(); ++index) {
    const auto& message = messages[index];
    const std::string_view role = message.role;
    const bool first = index == 0 && !insert_system;
    if (is_system(role)) {
      if (!first)
        output.append(kSystem);
      output.append(message.content);
      if (first && !tools.empty()) {
        output.append("\n\n");
        AppendToolsPrompt(output, tools);
      }
      continue;
    }
    if (is_user_turn(role)) {
      // One user turn: consecutive user text and tool results joined by blank
      // lines, with results sorted by the preceding call order as in
      // sort_tool_results_by_call_order. Unknown IDs rank first; ties keep
      // their arrival order.
      std::size_t end = index;
      std::vector<const ChatMessage*> tool_results;
      while (end < messages.size() && is_user_turn(messages[end].role)) {
        if (messages[end].role != "user")
          tool_results.push_back(&messages[end]);
        ++end;
      }
      const auto rank = [&](const ChatMessage* result) {
        const auto found = tool_call_order.find(result->tool_call_id);
        return found == tool_call_order.end() ? std::size_t{0} : found->second;
      };
      std::stable_sort(tool_results.begin(), tool_results.end(),
                       [&](const auto* left, const auto* right) {
                         return rank(left) < rank(right);
                       });
      std::size_t result_index = 0;
      output.append(kUser);
      for (std::size_t part = index; part < end; ++part) {
        if (part != index)
          output.append("\n\n");
        if (messages[part].role == "user") {
          output.append(messages[part].content);
        } else {
          output.append("<tool_result>");
          output.append(tool_results[result_index++]->content);
          output.append("</tool_result>");
        }
      }
      index = end - 1;
      continue;
    }
    if (role == "assistant") {
      if (!message.tool_calls.empty()) {
        tool_call_order.clear();
        for (std::size_t call = 0; call < message.tool_calls.size(); ++call) {
          if (!message.tool_calls[call].id.empty())
            tool_call_order[message.tool_calls[call].id] = call;
        }
      }
      output.append(kAssistant);
      if (thinking && index > 0 && !drop_thinking) {
        output.append(kThinkStart);
        output.append(message.reasoning_content);
      }
      output.append(kThinkEnd);
      output.append(message.content);
      AppendToolCalls(output, message.tool_calls);
      output.append(kEos);
    }
  }

  if (!messages.empty() && messages.back().role != "assistant")
    output.append(GenerationPrompt(thinking));
  return output;
}

std::string_view GenerationPrompt(bool enable_thinking) {
  return enable_thinking ? "<｜Assistant｜><think>" : "<｜Assistant｜></think>";
}

bool ValidateGgufTemplate(const core::GgufReader& reader,
                          std::string* error_msg) {
  if (reader.GetMetadataString("general.architecture") != "deepseek41") {
    AssignError(error_msg,
                "DeepSeek V4.1 Flash artifact has unexpected architecture");
    return false;
  }
  const auto raw_template = reader.GetMetadataString("tokenizer.chat_template");
  if (!raw_template.has_value() || raw_template->empty()) {
    AssignError(error_msg,
                "DeepSeek GGUF is missing tokenizer.chat_template metadata");
    return false;
  }
  const std::string actual_hash = Sha256(*raw_template);
  if (actual_hash != ArtifactTemplateSha256()) {
    AssignError(error_msg,
                "DeepSeek GGUF chat template SHA-256 is not the pinned 1017 "
                "artifact version: " +
                    actual_hash);
    return false;
  }
  return true;
}

}  // namespace gufo::models::deepseek_v41_flash
