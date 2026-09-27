# Gemma 4 31B

Dense Gemma 4 text model on gfx1151: 60 layers, five sliding-window (1024)
layers per global layer, tied embedding and a final logit softcap.
Supported target: `unsloth/gemma-4-31B-it-GGUF`, **UD-Q4_K_XL** (single file),
with the optional `gemma4-assistant` MTP drafter from the same repository.
Text only; vision, fine-tunes and QAT (Q4_0) checkpoints are rejected at load.

[Benchmarks](BENCHMARKS.md) · [Quality](QUALITY.md) · [Experiments](EXPERIMENTS.md)

## Load and run

```sh
nix develop -c hf download unsloth/gemma-4-31B-it-GGUF \
  --revision c1ac76e99d5513b141e8adde7288b85c3f9c32ec \
  --include gemma-4-31B-it-UD-Q4_K_XL.gguf MTP/mtp-gemma-4-31B-it-Q8_0.gguf \
  --local-dir models/gemma-4-31b
nix build
MODEL=models/gemma-4-31b/gemma-4-31B-it-UD-Q4_K_XL.gguf
MTP=models/gemma-4-31b/MTP/mtp-gemma-4-31B-it-Q8_0.gguf
./result/bin/gufo chat --model "$MODEL"
./result/bin/gufo serve llm --model "$MODEL" --speculative mtp \
  --mtp-model "$MTP" --context 131072
```

Omit the speculative options for autoregressive decoding. `-d`/`--draft-tokens`
caps the drafts per cycle (1–7, default 7); each cycle stops drafting once the
drafter's confidence that all its drafts will be accepted falls below a floor;
sampled chains use at most four. The drafter attends to the target's own KV cache,
so it adds no per-session cache.
Greedy speculative output equals single-token decoding of the same
configuration token for token; an AR-only server uses a faster one-row
projection kernel with a different FP32 summation order, so long greedy
completions of the two configurations can differ. Sampled requests verify
drafts by p/q rejection, so emitted tokens follow the target distribution.

The chat template is Unsloth's variant (SHA-256 `845f1ee4…73d1b`), compiled
into the engine and checked against Jinja renders. It defaults to thinking
off; the [reasoning controls](../../SERVER.md#reasoning-controls) turn the
thinking channel on, returned as `reasoning_content`. Gemma tool calls
(`<|tool_call>call:NAME{…}<tool_call|>`) are returned as OpenAI `tool_calls`.

Native context is 262144. Global layers keep every token, sliding layers a
fixed ring. A global layer's K and V come from one projection, so its cache
stores V and only the 128 rotated key dims of each head (50 KiB per token
across the ten global layers; the other key dims are V scaled by `k_norm`).
A 131072 context needs about 6.3 GiB of global KV plus 2.3 GiB of rings next
to the 17.5 GiB of weights.
Prompt snapshots feed the RAM and `--cache-disk` prompt caches; a trimmed or
edited history re-prefills from the latest retained checkpoint before the
change, so dropping the oldest messages costs a full prefill.

## Tools and artifacts

Tests live in `tests/models/gemma4` (`gemma4.*` in CTest; set
`GUFO_GEMMA4_MODEL` and `GUFO_GEMMA4_MTP_MODEL` for the model-backed ones).
`tools/gemma4` holds the tokenizer/template golden generators, the
teacher-forced llama.cpp logit dumper (`llama_logits.cpp`) and
`compare_logits.py`. The implementation plan and dated progress are in
[PLAN.md](PLAN.md).
