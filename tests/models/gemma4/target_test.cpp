// Model-backed GPU qualification of the Gemma 4 target against the scalar
// reference on the same GGUF: decode arithmetic (FP32 activations) must be
// essentially exact, prefill (Q8_1 activations) must stay inside the
// envelope measured at introduction, and session prefix extension and
// snapshot restore past a sliding-ring wrap must be bitwise stable.
// Requires GUFO_GEMMA4_MODEL; exits 77 without it.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/core/sampling.hpp"
#include "src/models/gemma4/chat_template.hpp"
#include "src/models/gemma4/engine.hpp"
#include "src/models/gemma4/reference.hpp"
#include "tests/models/gemma4/check.hpp"

namespace g4 = gufo::models::gemma4;
using gemma4_test::Require;

namespace {

/// Limits recorded when the runtime was introduced (2026-09-26); they are
/// never loosened. Observed: decode mean KL 2.4e-7, prefill mean KL 0.0084.
constexpr double kDecodeMeanKl = 1e-5;
constexpr double kPrefillMeanKl = 0.015;
/// Bulk prefill vs exact rows over the 1353-token snapshot prompt;
/// observed 1.39 (2026-09-26).
constexpr double kLongPrefillMeanKl = 1.6;

struct Stats {
  double mean_kl{0.0};
  double max_kl{0.0};
  std::size_t top1{0};
  std::size_t rows{0};
};

Stats Compare(const std::vector<float>& ref, const std::vector<float>& got,
              std::size_t vocab) {
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

std::vector<float> GpuLogits(g4::Model& model,
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

/// Tokens from autoregressive evaluation and from DecodeStep (greedy by
/// default).
std::vector<g4::TokenId> Generate(
    g4::Model& model, std::span<const g4::TokenId> prompt, std::size_t count,
    bool speculative, gufo::sampling::SamplingConfig config = {},
    g4::Session::SpeculativeStats* stats_out = nullptr) {
  std::string error;
  auto session = model.CreateSession(0, &error);
  Require(session && session->Sync(prompt, &error), error);
  gufo::sampling::SamplerState sampler(config, {});
  std::vector<g4::TokenId> out;
  if (!speculative) {
    while (out.size() < count) {
      const auto token =
          static_cast<g4::TokenId>(sampler.Sample(session->Logits()));
      sampler.Accept(static_cast<gufo::sampling::TokenId>(token));
      out.push_back(token);
      Require(session->Evaluate(token, &error), error);
    }
    return out;
  }
  while (out.size() < count) {
    g4::Session::DecodeResult step;
    Require(session->DecodeStep(count - out.size(), sampler, &step, &error,
                                false) &&
                !step.tokens.empty(),
            error);
    out.insert(out.end(), step.tokens.begin(), step.tokens.end());
  }
  const auto& stats = session->Statistics();
  std::cout << "speculative: cycles " << stats.cycles << ", drafted "
            << stats.drafted << ", accepted " << stats.accepted << '\n';
  if (stats_out != nullptr) {
    *stats_out = stats;
  }
  return out;
}

std::vector<g4::TokenId> PromptTokens(const g4::Model& model,
                                      const std::string& text) {
  gufo::tokenization::ChatMessage message(gufo::tokenization::ChatRole::kUser,
                                          text);
  std::string error;
  const auto rendered =
      g4::ChatTemplate::Render(std::span(&message, 1), {}, {}, &error);
  Require(rendered.has_value(), error);
  return model.Tokenize(rendered->text);
}

}  // namespace

int main() {
  const char* path = std::getenv("GUFO_GEMMA4_MODEL");
  if (path == nullptr || *path == '\0') {
    std::cout << "SKIP: GUFO_GEMMA4_MODEL is not set\n";
    return 77;
  }
  const char* draft = std::getenv("GUFO_GEMMA4_MTP_MODEL");
  return gemma4_test::Run([&] {
    std::string error;
    g4::ModelOptions options;
    // A small prefill chunk keeps the sliding ring short (window + 256), so
    // the snapshot check below wraps it inside this context.
    options.max_context = 2048;
    options.prefill_chunk = 256;
    auto model = g4::Model::Load(path, options, &error);
    Require(model != nullptr, error);
    const std::size_t vocab = model->VocabSize();
    const auto tokens =
        PromptTokens(*model, "Write one sentence about the sea near Genoa.");
    Require(tokens.size() > 16, "prompt too short for the prefill path");

    // Independent scalar reference over the same weights.
    const auto weights = g4::ModelWeights::Bind(model->reader(), &error);
    Require(weights.has_value(), error);
    g4::Reference reference(*weights, g4::Reference::Storage::kHalfKv);
    std::vector<float> oracle;
    reference.Forward(tokens, &oracle, nullptr);

    const auto decode = GpuLogits(*model, tokens, 1);
    const auto chunked = GpuLogits(*model, tokens, 8);
    const auto prefill = GpuLogits(*model, tokens, tokens.size());
    const Stats d = Compare(oracle, decode, vocab);
    const Stats c = Compare(oracle, chunked, vocab);
    const Stats p = Compare(oracle, prefill, vocab);
    std::cout << "decode:  mean KL " << d.mean_kl << ", max " << d.max_kl
              << ", top-1 " << d.top1 << "/" << d.rows << '\n'
              << "8-row:   mean KL " << c.mean_kl << ", top-1 " << c.top1 << "/"
              << c.rows << '\n'
              << "prefill: mean KL " << p.mean_kl << ", max " << p.max_kl
              << ", top-1 " << p.top1 << "/" << p.rows << '\n';
    Require(d.mean_kl < kDecodeMeanKl && d.top1 == d.rows,
            "decode diverges from the reference");
    Require(c.mean_kl < kDecodeMeanKl && c.top1 == c.rows,
            "small-batch rows diverge from the reference");
    Require(p.mean_kl < kPrefillMeanKl, "prefill exceeds its KL envelope");

    // Extending a synced prompt matches evaluating the tokens one by one.
    const std::size_t split = tokens.size() - 4;
    const std::span<const g4::TokenId> head(tokens.data(), split);
    auto a = model->CreateSession(0, &error);
    auto b = model->CreateSession(0, &error);
    Require(a && b, error);
    Require(a->Sync(head, &error) && b->Sync(head, &error), error);
    for (std::size_t i = split; i < tokens.size(); ++i) {
      Require(a->Evaluate(tokens[i], &error), error);
    }
    Require(b->Sync(tokens, &error), error);
    const auto as_vector = [](std::span<const float> v) {
      return std::vector<float>(v.begin(), v.end());
    };
    const Stats ext =
        Compare(as_vector(a->Logits()), as_vector(b->Logits()), vocab);
    Require(ext.top1 == 1 && ext.mean_kl < kDecodeMeanKl,
            "prefix extension diverges from token-by-token evaluation");
    // Rewinding to a shorter prompt reuses the prefix and matches a fresh sync.
    Require(b->Sync(head, &error), error);
    auto fresh = model->CreateSession(0, &error);
    Require(fresh && fresh->Sync(head, &error), error);
    const Stats rw =
        Compare(as_vector(fresh->Logits()), as_vector(b->Logits()), vocab);
    Require(rw.top1 == 1 && rw.mean_kl < kPrefillMeanKl,
            "rewound session diverges from a fresh one");

    // A snapshot taken after the ring wrapped restores into a fresh session
    // (typed and serialized) and continues bitwise like the original.
    std::string text;
    for (int i = 0; i < 90; ++i) {
      text += "Ship " + std::to_string(i) +
              " left Genoa at dawn carrying salt, wine and letters. ";
    }
    const auto long_prompt = PromptTokens(*model, text);
    Require(long_prompt.size() > model->SessionRingSlots() + 64 &&
                long_prompt.size() + 16 < options.max_context,
            "snapshot prompt does not wrap the sliding ring");
    auto original = model->CreateSession(0, &error);
    Require(original && original->Sync(long_prompt, &error), error);
    const auto snapshot = original->SaveSnapshot(&error);
    Require(snapshot != nullptr &&
                snapshot->SizeBytes() == original->SnapshotBytes(),
            error);
    const std::vector<std::uint8_t> bytes(snapshot->bytes().begin(),
                                          snapshot->bytes().end());
    auto typed = model->CreateSession(0, &error);
    auto serialized = model->CreateSession(0, &error);
    Require(typed && typed->RestoreSnapshot(*snapshot, &error), error);
    Require(serialized && serialized->RestoreSnapshot(bytes, &error), error);
    for (int step = 0; step <= 8; ++step) {
      const auto logits = as_vector(original->Logits());
      Require(as_vector(typed->Logits()) == logits &&
                  as_vector(serialized->Logits()) == logits,
              "restored session diverges at step " + std::to_string(step));
      if (step == 8) {
        break;
      }
      const auto next = static_cast<g4::TokenId>(
          std::max_element(logits.begin(), logits.end()) - logits.begin());
      Require(original->Evaluate(next, &error) &&
                  typed->Evaluate(next, &error) &&
                  serialized->Evaluate(next, &error),
              error);
    }
    std::vector<std::uint8_t> corrupt = bytes;
    corrupt.pop_back();
    Require(!serialized->RestoreSnapshot(corrupt, &error),
            "truncated snapshot was accepted");
    original.reset();
    typed.reset();
    serialized.reset();

    // Bulk prefill over many attention tiles and a wrapped ring stays close
    // to the exact small-batch rows. The repeated text makes this prompt very
    // sensitive to prefill's binary16/Q8_1 rounding: llama.cpp's teacher-forced
    // logits sit at mean KL 2.93 from the same exact rows.
    {
      const auto exact = GpuLogits(*model, long_prompt, 8);
      const auto bulk = GpuLogits(*model, long_prompt, long_prompt.size());
      const Stats lp = Compare(exact, bulk, vocab);
      std::cout << "long prefill vs exact rows (" << lp.rows
                << " tokens): mean KL " << lp.mean_kl << ", max " << lp.max_kl
                << ", top-1 " << lp.top1 << "/" << lp.rows << '\n';
      Require(lp.mean_kl < kLongPrefillMeanKl,
              "long prefill exceeds its KL envelope");
    }

    if (draft == nullptr || *draft == '\0') {
      std::cout
          << "note: GUFO_GEMMA4_MTP_MODEL not set; speculation unchecked\n";
      return;
    }
    model.reset();
    options.mtp_model_path = draft;
    options.draft_tokens = 4;
    auto mtp = g4::Model::Load(path, options, &error);
    Require(mtp != nullptr, error);
    // With a drafter, verification rows round exactly like decode...
    Require(GpuLogits(*mtp, tokens, 1) == GpuLogits(*mtp, tokens, 5),
            "verification rows differ from single-token decode");
    // ...so greedy speculation reproduces autoregressive output.
    for (const char* text :
         {"Write one sentence about the sea near Genoa.",
          "List the first twelve prime numbers, separated by commas.",
          "Explain in three sentences why the sky is blue."}) {
      const auto prompt = PromptTokens(*mtp, text);
      const auto ar = Generate(*mtp, prompt, 64, false);
      const auto spec = Generate(*mtp, prompt, 64, true);
      Require(ar == spec, std::string("greedy MTP differs from AR: ") + text);
    }
    // Sampled MTP (a chat front end's sampler): drafts are sampled and
    // verified by p/q rejection; a seed replays the same tokens.
    gufo::sampling::SamplingConfig chat;
    chat.temperature = 1.0F;
    chat.top_k = 64;
    chat.top_p = 0.95F;
    chat.repeat_penalty = 1.05F;
    chat.seed = 7;
    const auto prompt =
        PromptTokens(*mtp, "Write a short story about a lighthouse keeper.");
    g4::Session::SpeculativeStats stats;
    const auto first = Generate(*mtp, prompt, 96, true, chat, &stats);
    const auto again = Generate(*mtp, prompt, 96, true, chat);
    Require(first.size() == 96 && first == again,
            "sampled MTP does not replay its seed");
    Require(stats.accepted > 0, "sampled MTP accepted no drafts");
  });
}
