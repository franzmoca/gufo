#include "src/cli/prompt/gemma4_prompt.hpp"

#include <iostream>
#include <string>
#include <vector>

#include "src/models/gemma4/chat_template.hpp"
#include "src/models/gemma4/engine.hpp"

namespace gufo::cli {
namespace {

namespace g4 = models::gemma4;

constexpr std::uint32_t kDefaultContext = 4096;

double SecondsSince(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
      .count();
}

std::shared_ptr<g4::Model> LoadModel(
    const PromptOptions& opt, std::chrono::steady_clock::time_point start) {
  if (opt.force_cpu) {
    std::cerr << "Gemma 4 is supported only by the ROCm backend\n";
    return nullptr;
  }
  if (!opt.image_paths.empty() || !opt.vision_model_path.empty()) {
    std::cerr << "Gemma 4 runs text-only; image input is not supported\n";
    return nullptr;
  }
  if (!opt.speculative_backend.empty()) {
    std::cerr << "Gemma 4 --speculative is not available yet\n";
    return nullptr;
  }
  std::string error;
  auto model = g4::Model::Load(
      opt.model_path, g4::ModelOptions{.max_context = kDefaultContext}, &error);
  if (model == nullptr) {
    std::cerr << "Error loading Gemma 4 model: " << error << '\n';
    std::cerr << "[Model Load]: " << SecondsSince(start) << " s (failed)\n";
    return nullptr;
  }
  std::cout << "[Model Load]: " << SecondsSince(start) << " s\n";
  return model;
}

std::vector<g4::TokenId> RenderPrompt(
    const g4::Model& model, std::span<const tokenization::ChatMessage> history,
    const g4::ChatOptions& options, std::string* error) {
  const auto rendered = g4::ChatTemplate::Render(history, {}, options, error);
  if (!rendered) {
    return {};
  }
  return model.Tokenize(rendered->text);
}

/// Generates one reply after `prompt`; `reply` receives the decoded text.
int Generate(const PromptOptions& opt, g4::Model& model, g4::Session& session,
             std::span<const g4::TokenId> prompt, std::string* reply) {
  if (prompt.empty()) {
    std::cerr << "Gemma 4 prompt produced no tokens\n";
    return 1;
  }
  if (prompt.size() + opt.max_tokens > session.ContextSize()) {
    std::cerr << "Gemma 4 prompt and output exceed the "
              << session.ContextSize() << "-token CLI context\n";
    return 1;
  }
  std::string error;
  const auto prefill_start = std::chrono::steady_clock::now();
  if (!session.Sync(prompt, &error)) {
    std::cerr << "Gemma 4 prefill failed: " << error << '\n';
    return 1;
  }
  const double prefill_seconds = SecondsSince(prefill_start);
  if (opt.verbose) {
    std::cout << "[Engine]: Gemma 4 ROCm (gfx1151)\n"
              << "Model: " << model.ModelName() << '\n'
              << "Prompt tokens: " << prompt.size() << '\n'
              << "--- Generation Output ---\n";
  }
  std::vector<sampling::TokenId> history(prompt.begin(), prompt.end());
  sampling::SamplerState sampler(opt.sampling, history);
  const auto decode_start = std::chrono::steady_clock::now();
  std::size_t generated = 0;
  for (; generated < opt.max_tokens; ++generated) {
    const auto token =
        static_cast<g4::TokenId>(sampler.Sample(session.Logits()));
    if (model.IsStopToken(token)) {
      break;
    }
    sampler.Accept(static_cast<sampling::TokenId>(token));
    const std::string piece = model.TokenText(token);
    if (reply != nullptr) {
      reply->append(piece);
    }
    std::cout << piece << std::flush;
    if ((reply != nullptr || generated + 1 < opt.max_tokens) &&
        !session.Evaluate(token, &error)) {
      std::cerr << "\nGemma 4 decode failed: " << error << '\n';
      return 1;
    }
  }
  std::cout << '\n';
  if (opt.verbose && generated > 0) {
    const double seconds = SecondsSince(decode_start);
    std::cout << "Prefill " << prompt.size() << " tokens in " << prefill_seconds
              << " s (" << static_cast<double>(prompt.size()) / prefill_seconds
              << " tok/s)\nGenerated " << generated << " tokens in " << seconds
              << " s (" << static_cast<double>(generated) / seconds
              << " tok/s)\n";
  }
  return 0;
}

}  // namespace

bool IsGemma4(const core::GgufReader& reader) {
  return reader.GetMetadataString("general.architecture") == "gemma4";
}

int RunGemma4Prompt(const PromptOptions& opt,
                    std::chrono::steady_clock::time_point load_start) {
  auto model = LoadModel(opt, load_start);
  if (!model) {
    return 1;
  }
  std::string error;
  auto session = model->CreateSession(kDefaultContext, &error);
  if (!session) {
    std::cerr << "Gemma 4 session creation failed: " << error << '\n';
    return 1;
  }
  std::vector<g4::TokenId> tokens;
  if (opt.use_chat_template) {
    std::vector<tokenization::ChatMessage> messages;
    if (!opt.system_prompt.empty()) {
      messages.emplace_back(tokenization::ChatRole::kSystem, opt.system_prompt);
    }
    messages.emplace_back(tokenization::ChatRole::kUser, opt.prompt_text);
    tokens = RenderPrompt(*model, messages,
                          g4::ResolveChatOptions(PromptReasoningOptions(opt)),
                          &error);
    if (tokens.empty()) {
      std::cerr << "Gemma 4 chat template failed: " << error << '\n';
      return 1;
    }
  } else {
    tokens = model->tokenizer().Encode(opt.prompt_text, true, true);
  }
  return Generate(opt, *model, *session, tokens, nullptr);
}

int RunGemma4Chat(const PromptOptions& opt,
                  std::chrono::steady_clock::time_point load_start) {
  if (!opt.use_chat_template) {
    std::cerr << "Gemma 4 interactive chat requires chat framing; use prompt "
                 "--raw for raw text\n";
    return 1;
  }
  auto model = LoadModel(opt, load_start);
  if (!model) {
    return 1;
  }
  std::string error;
  auto session = model->CreateSession(kDefaultContext, &error);
  if (!session) {
    std::cerr << "Gemma 4 session creation failed: " << error << '\n';
    return 1;
  }
  const auto chat_options = g4::ResolveChatOptions(PromptReasoningOptions(opt));
  std::vector<tokenization::ChatMessage> history;
  if (!opt.system_prompt.empty()) {
    history.emplace_back(tokenization::ChatRole::kSystem, opt.system_prompt);
  }
  std::cout << "=== Gufo Interactive Chat (Gemma 4) ===\n"
            << "Type 'exit' or Ctrl+D to quit.\n\n";
  for (std::string input;;) {
    std::cout << ">>> User: " << std::flush;
    if (!std::getline(std::cin, input) || input == "exit" || input == "quit") {
      break;
    }
    if (input.empty()) {
      continue;
    }
    history.emplace_back(tokenization::ChatRole::kUser, input);
    const auto tokens = RenderPrompt(*model, history, chat_options, &error);
    if (tokens.empty()) {
      std::cerr << "Gemma 4 chat template failed: " << error << '\n';
      return 1;
    }
    std::cout << "<<< Assistant: ";
    std::string reply;
    if (Generate(opt, *model, *session, tokens, &reply) != 0) {
      return 1;
    }
    // The template drops thinking channels from model history itself.
    history.emplace_back(tokenization::ChatRole::kAssistant, std::move(reply));
    std::cout << '\n';
  }
  return 0;
}

}  // namespace gufo::cli
