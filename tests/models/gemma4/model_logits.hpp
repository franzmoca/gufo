#ifndef GUFO_TESTS_MODELS_GEMMA4_MODEL_LOGITS_HPP_
#define GUFO_TESTS_MODELS_GEMMA4_MODEL_LOGITS_HPP_

// Logit comparisons and fixture prompts shared by the model-backed Gemma 4
// checks.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <span>
#include <string>
#include <vector>

#include "src/models/gemma4/chat_template.hpp"
#include "src/models/gemma4/engine.hpp"
#include "tests/models/gemma4/check.hpp"

namespace gemma4_test {

namespace g4 = gufo::models::gemma4;

struct Stats {
  double mean_kl{0.0};
  double max_kl{0.0};
  std::size_t top1{0};
  std::size_t rows{0};
};

inline Stats Compare(const std::vector<float>& ref,
                     const std::vector<float>& got, std::size_t vocab) {
  Stats s;
  s.rows = ref.size() / vocab;
  for (std::size_t r = 0; r < s.rows; ++r) {
    const float* a = &ref[r * vocab];
    const float* b = &got[r * vocab];
    double ma = -INFINITY, mb = -INFINITY;
    std::size_t aa = 0, ab = 0;
    for (std::size_t i = 0; i < vocab; ++i) {
      Require(std::isfinite(b[i]), "non-finite GPU logit");
      if (a[i] > ma) {
        ma = a[i];
        aa = i;
      }
      if (b[i] > mb) {
        mb = b[i];
        ab = i;
      }
    }
    double za = 0.0, zb = 0.0;
    for (std::size_t i = 0; i < vocab; ++i) {
      za += std::exp(a[i] - ma);
      zb += std::exp(b[i] - mb);
    }
    const double la = std::log(za) + ma, lb = std::log(zb) + mb;
    double kl = 0.0;
    for (std::size_t i = 0; i < vocab; ++i) {
      const double pa = a[i] - la;
      kl += std::exp(pa) * (pa - (b[i] - lb));
    }
    s.mean_kl += kl / static_cast<double>(s.rows);
    s.max_kl = std::max(s.max_kl, kl);
    s.top1 += aa == ab ? 1 : 0;
  }
  return s;
}

inline std::vector<float> GpuLogits(g4::Model& model,
                                    const std::vector<g4::TokenId>& tokens,
                                    std::size_t chunk) {
  std::string error;
  auto session = model.CreateSession(0, &error);
  Require(session != nullptr, error);
  std::vector<float> all, part;
  for (std::size_t begin = 0; begin < tokens.size(); begin += chunk) {
    const std::size_t count = std::min(chunk, tokens.size() - begin);
    Require(session->EvaluateAll(std::span(tokens).subspan(begin, count), &part,
                                 &error),
            error);
    all.insert(all.end(), part.begin(), part.end());
  }
  return all;
}

/// The fixture conversation: turns separated by "@@ user" / "@@ model".
inline std::vector<g4::TokenId> ConversationTokens(const g4::Model& model) {
  std::ifstream in(std::string(GUFO_GEMMA4_FIXTURES) +
                   "/long_conversation.txt");
  Require(in.good(), "long_conversation.txt is missing");
  std::vector<gufo::tokenization::ChatMessage> messages;
  std::string line;
  std::string text;
  auto role = gufo::tokenization::ChatRole::kUser;
  bool open = false;
  const auto flush = [&] {
    if (open) {
      while (!text.empty() && text.back() == '\n') {
        text.pop_back();
      }
      messages.emplace_back(role, text);
    }
    text.clear();
  };
  while (std::getline(in, line)) {
    if (line == "@@ user" || line == "@@ model") {
      flush();
      role = line == "@@ user" ? gufo::tokenization::ChatRole::kUser
                               : gufo::tokenization::ChatRole::kAssistant;
      open = true;
    } else {
      text += line + "\n";
    }
  }
  flush();
  std::string error;
  const auto rendered = g4::ChatTemplate::Render(messages, {}, {}, &error);
  Require(rendered.has_value(), error);
  return model.Tokenize(rendered->text);
}

inline std::vector<g4::TokenId> PromptTokens(const g4::Model& model,
                                             const std::string& text) {
  gufo::tokenization::ChatMessage message(gufo::tokenization::ChatRole::kUser,
                                          text);
  std::string error;
  const auto rendered =
      g4::ChatTemplate::Render(std::span(&message, 1), {}, {}, &error);
  Require(rendered.has_value(), error);
  return model.Tokenize(rendered->text);
}

}  // namespace gemma4_test

#endif  // GUFO_TESTS_MODELS_GEMMA4_MODEL_LOGITS_HPP_
