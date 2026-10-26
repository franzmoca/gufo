#include "src/models/deepseek_v41_flash/chat_template.hpp"

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iterator>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/json.hpp"

namespace {

using gufo::ReasoningEffort;
using gufo::models::deepseek_v41_flash::ChatMessage;
using gufo::models::deepseek_v41_flash::ChatTemplateOptions;
using gufo::models::deepseek_v41_flash::ChatTool;

ReasoningEffort ParseEffort(const gufo::json::Value* effort) {
  if (effort == nullptr || !effort->is_string())
    return ReasoningEffort::kHigh;
  if (effort->str() == "low")
    return ReasoningEffort::kLow;
  if (effort->str() == "max")
    return ReasoningEffort::kMax;
  return ReasoningEffort::kHigh;
}

std::vector<ChatMessage> ParseMessages(const gufo::json::Value& messages) {
  std::vector<ChatMessage> parsed;
  for (const auto& message : messages.items()) {
    ChatMessage& out = parsed.emplace_back();
    out.role = message.member_str("role");
    out.content = message.member_str("content");
    out.reasoning_content = message.member_str("reasoning_content");
    out.tool_call_id = message.member_str("tool_call_id");
    if (const auto* images = message.find("images")) {
      for (const auto& offset : images->items())
        out.image_offsets.push_back(offset.as_size());
    }
    if (const auto* calls = message.find("tool_calls")) {
      for (const auto& call : calls->items()) {
        ChatMessage::ToolCall& tool_call = out.tool_calls.emplace_back();
        tool_call.id = call.member_str("id");
        tool_call.name = call.member_str("name");
        for (const auto& argument : call.find("arguments")->items()) {
          ChatMessage::ToolArgument& out_argument =
              tool_call.arguments.emplace_back();
          out_argument.name = argument.member_str("name");
          out_argument.value = argument.member_str("value");
          out_argument.is_string = argument.find("is_string")->as_bool();
        }
      }
    }
  }
  return parsed;
}

}  // namespace

int main() {
  std::ifstream file(GUFO_DS41_CHAT_GOLDENS, std::ios::binary);
  const std::string text{std::istreambuf_iterator<char>(file), {}};
  const auto fixture = gufo::json::parse(text);
  if (fixture.member_str("encoding_sha256") !=
      gufo::models::deepseek_v41_flash::EncoderReferenceSha256()) {
    std::cerr << "fixture was generated from a different encoder\n";
    return 1;
  }
  int failures = 0;
  std::size_t cases = 0;
  for (const auto& golden : fixture.find("cases")->items()) {
    const auto messages = ParseMessages(*golden.find("messages"));
    std::vector<ChatTool> tools;
    for (const auto& tool : golden.find("tools")->items()) {
      ChatTool& out = tools.emplace_back();
      out.name = tool.member_str("name");
      out.definition_json = tool.member_str("definition_json");
    }
    const ChatTemplateOptions options{
        .enable_thinking = golden.find("thinking")->as_bool(),
        .reasoning_effort = ParseEffort(golden.find("effort")),
        .tools_present = !tools.empty(),
    };
    const std::string rendered =
        gufo::models::deepseek_v41_flash::RenderChat(messages, tools, options);
    const std::string expected = golden.member_str("expected");
    ++cases;
    if (rendered == expected)
      continue;
    ++failures;
    std::size_t at = 0;
    while (at < rendered.size() && at < expected.size() &&
           rendered[at] == expected[at])
      ++at;
    const std::size_t from = at > 80 ? at - 80 : 0;
    std::cerr << golden.member_str("name") << ": mismatch at byte " << at
              << "\n  expected: " << expected.substr(from, 160)
              << "\n  rendered: " << rendered.substr(from, 160) << '\n';
  }
  if (cases == 0) {
    std::cerr << "no golden cases\n";
    return 1;
  }
  // The official encoder rejects the placeholder in text (and in reasoning).
  for (const bool reasoning : {false, true}) {
    ChatMessage message;
    message.role = "user";
    message.content = "hi";
    (reasoning ? message.reasoning_content : message.content)
        .append(gufo::models::deepseek_v41_flash::kImagePlaceholder);
    try {
      (void)gufo::models::deepseek_v41_flash::RenderChat(std::span(&message, 1),
                                                         ChatTemplateOptions{});
      std::cerr << "placeholder text was not rejected\n";
      ++failures;
    } catch (const std::invalid_argument&) {
    }
  }
  std::cout << cases - failures << "/" << cases << " cases match\n";
  return failures == 0 ? 0 : 1;
}
