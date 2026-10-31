#include "src/models/deepseek_v41_flash/chat_template.hpp"

#include <unistd.h>

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "src/core/gguf_reader.hpp"
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

// A tensorless GGUF v3 holding string metadata only.
std::unique_ptr<gufo::core::GgufReader> MetadataGguf(
    const std::vector<std::pair<std::string, std::string>>& strings) {
  std::string bytes = "GGUF";
  const auto put = [&](std::uint64_t value, int size) {
    for (int i = 0; i < size; ++i)
      bytes.push_back(static_cast<char>(value >> (8 * i)));
  };
  const auto put_string = [&](const std::string& text) {
    put(text.size(), 8);
    bytes += text;
  };
  put(3, 4);
  put(0, 8);
  put(strings.size(), 8);
  for (const auto& [key, value] : strings) {
    put_string(key);
    put(8, 4);  // string
    put_string(value);
  }
  bytes.resize((bytes.size() + 31) / 32 * 32, '\0');
  const auto path =
      std::filesystem::temp_directory_path() /
      ("ds41_template_test_" + std::to_string(::getpid()) + ".gguf");
  std::ofstream(path, std::ios::binary).write(bytes.data(), bytes.size());
  std::string error;
  auto reader = gufo::core::GgufReader::OpenFile(path, &error);
  std::filesystem::remove(path);
  if (!reader)
    std::cerr << "test GGUF: " << error << '\n';
  return reader;
}

// Prompts always render with the compiled formatter: another embedded
// template only warns, another architecture is rejected.
int CheckArtifactTemplates() {
  namespace ds41 = gufo::models::deepseek_v41_flash;
  int failures = 0;
  const auto check = [&](const char* name,
                         std::vector<std::pair<std::string, std::string>> kv,
                         bool valid, bool warns) {
    const auto reader = MetadataGguf(kv);
    std::string error, warning;
    if (!reader ||
        ds41::ValidateGgufTemplate(*reader, &error, &warning) != valid ||
        warning.empty() == warns) {
      std::cerr << name << ": error '" << error << "' warning '" << warning
                << "'\n";
      ++failures;
    }
  };
  check("other template",
        {{"general.architecture", "deepseek41"},
         {"tokenizer.chat_template", "{{ messages }}"}},
        true, true);
  check("no template", {{"general.architecture", "deepseek41"}}, true, true);
  check("other architecture",
        {{"general.architecture", "deepseek4"},
         {"tokenizer.chat_template", "{{ messages }}"}},
        false, false);
  return failures;
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
  failures += CheckArtifactTemplates();
  return failures == 0 ? 0 : 1;
}
