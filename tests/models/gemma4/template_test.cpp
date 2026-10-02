#include <cstdint>
#include <iostream>
#include <span>
#include <string>
#include <vector>

#include "src/core/crypto/sha256.hpp"
#include "src/models/gemma4/chat_template.hpp"
#include "tests/models/gemma4/check.hpp"
#include "tests/models/gemma4/template_cases.hpp"

namespace g4 = gufo::models::gemma4;
using gemma4_test::Require;

namespace {

std::string Sha256(const std::string& text) {
  return gufo::crypto::Sha256Hex(std::span<const std::uint8_t>(
      reinterpret_cast<const std::uint8_t*>(text.data()), text.size()));
}

/// Every case must render byte-identically to the Jinja source.
void CheckGoldens() {
  const auto goldens = gemma4_test::ReadJson(GUFO_CHAT_TEMPLATE_HF_GOLDENS);
  const auto* gemma = goldens.find("gemma4");
  Require(gemma != nullptr, "goldens lack a gemma4 section");
  Require(gemma->member_str("template_sha256") ==
              g4::ChatTemplate::UnslothTemplateSha256(),
          "goldens were rendered from another template");
  const auto* expected = gemma->find("cases");
  const auto cases = gemma4_test::LoadTemplateCases(
      std::string(GUFO_GEMMA4_FIXTURES) + "/template_cases.json");
  Require(cases.size() == expected->size(), "case/golden count mismatch");
  for (const auto& c : cases) {
    std::string error;
    std::vector<std::size_t> images;
    const auto rendered = g4::ChatTemplate::Render(c.messages, c.tools,
                                                   c.options, &error, &images);
    Require(rendered.has_value(), c.name + ": " + error);
    std::size_t expected_images = 0;
    for (const auto& message : c.messages) {
      expected_images += message.images.size();
    }
    Require(images.size() == expected_images, c.name + ": image count");
    for (const auto offset : images) {
      Require(rendered->text.compare(offset, g4::kImagePlaceholder.size(),
                                     g4::kImagePlaceholder) == 0,
              c.name + ": image offset is not a placeholder");
    }
    const auto* golden = expected->find(c.name);
    Require(golden != nullptr, c.name + ": no golden");
    if (Sha256(rendered->text) != golden->member_str("rendered_sha256")) {
      std::cerr << "---- " << c.name << " rendered ----\n"
                << rendered->text << "\n----\n";
      throw std::runtime_error(c.name + ": rendering differs from Jinja");
    }

    // The generation prompt is a pure suffix of the stable conversation.
    auto without = c.options;
    without.add_generation_prompt = false;
    const auto prefix = g4::ChatTemplate::Render(c.messages, c.tools, without,
                                                 nullptr, &images);
    Require(
        prefix && prefix->text == rendered->text.substr(
                                      0, rendered->generation_prompt_offset),
        c.name + ": generation prompt is not a suffix");
  }
}

void CheckOptions() {
  gufo::ReasoningOptions reasoning;
  Require(!g4::ResolveChatOptions(reasoning).enable_thinking,
          "thinking is off by default");
  reasoning.effort = gufo::ReasoningEffort::kHigh;
  Require(g4::ResolveChatOptions(reasoning).enable_thinking,
          "an effort level enables thinking");
  reasoning.effort = gufo::ReasoningEffort::kMinimal;
  Require(!g4::ResolveChatOptions(reasoning).enable_thinking,
          "minimal effort keeps thinking off");
  reasoning.enabled = true;
  Require(g4::ResolveChatOptions(reasoning).enable_thinking,
          "explicit enable wins");
}

void CheckRejections() {
  gufo::tokenization::ChatMessage image;
  image.images.push_back({});
  std::string error;
  Require(!g4::ChatTemplate::Render(std::span(&image, 1), {}, {}, &error) &&
              !error.empty(),
          "image input accepted without image offsets");
  gufo::tokenization::ChatTool bad;
  bad.definition_json = "[1]";
  Require(!g4::ChatTemplate::Render({}, std::span(&bad, 1), {}, &error),
          "non-object tool accepted");
  gufo::tokenization::ChatTool braced;
  braced.name = "a{b";
  Require(!g4::ChatTemplate::Render({}, std::span(&braced, 1), {}, &error),
          "braced tool name accepted");
  gufo::tokenization::ChatMessage call(gufo::tokenization::ChatRole::kAssistant,
                                       "");
  call.tool_calls.push_back({.id = "c", .name = "x{y", .arguments = {}});
  Require(!g4::ChatTemplate::Render(std::span(&call, 1), {}, {}, &error),
          "braced replayed call accepted");
  gufo::tokenization::ChatTool dotted;
  dotted.name = "github.create_issue";
  dotted.definition_json =
      R"({"type":"function","function":{"name":"github.create_issue"}})";
  const auto rendered =
      g4::ChatTemplate::Render({}, std::span(&dotted, 1), {}, &error);
  Require(rendered && rendered->text.find("declaration:github.create_issue{") !=
                          std::string::npos,
          "dotted tool name rendered: " + error);
}

void CheckShownConstants() {
  gufo::tokenization::ChatTool tool;
  tool.name = "record";
  tool.definition_json = R"({"type":"function","function":{"name":"record",
    "parameters":{"type":"object","properties":{
      "value":{"type":"string","const":"alpha"},
      "items":{"type":"array","items":{"type":"object","properties":{
        "kind":{"type":"string","const":"x"}}}},
      "n":{"type":"integer","const":3}},"required":["value"]}}})";
  std::string error;
  const auto rendered =
      g4::ChatTemplate::Render({}, std::span(&tool, 1), {}, &error);
  Require(rendered.has_value(), "const tool rendered: " + error);
  const auto& text = rendered->text;
  Require(text.find("value:{enum:[<|\"|>alpha<|\"|>]") != std::string::npos &&
              text.find("kind:{enum:[<|\"|>x<|\"|>]") != std::string::npos &&
              text.find("n:{type:") != std::string::npos,
          "a string const is shown as a one-value enum: " + text);
}

}  // namespace

int main() {
  return gemma4_test::Run([] {
    CheckGoldens();
    CheckOptions();
    CheckRejections();
    CheckShownConstants();
  });
}
