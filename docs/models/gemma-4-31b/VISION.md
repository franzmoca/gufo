# Gemma 4 31B vision: implementation plan

Image input for the native Gemma 4 engine (`src/models/gemma4`), served through
the same `--mmproj` sidecar option as Qwen. Text-only behavior, speed and
snapshots stay unchanged when no image is present.

## Sources

- Vision sidecar: `unsloth/gemma-4-31B-it-GGUF` `mmproj-BF16.gguf`, SHA-256
  `7a4601b12ec680f706a7e0c7e1f78f579b1db64d485a9e352ee87d4b9daa45e4`
  (1,200,726,496 bytes, downloaded 2026-09-28 next to the Unsloth target).
- Graph semantics: llama.cpp `tools/mtmd/models/gemma4v.cpp`, `clip.cpp`
  (`PROJECTOR_TYPE_GEMMA4V`), `mtmd.cpp` and `src/models/gemma4.cpp`, as read in
  the strix-llama.cpp checkout. The repo reference `.#llama-cpp-reference`
  (b11069) ships the same `clip_graph_gemma4v` and `llama-mtmd-cli`.
- Preprocessing and prompt: `google/gemma-4-31B-it` `processor_config.json`,
  `config.json`, `chat_template.jinja`, and transformers `models/gemma4`
  `processing_gemma4.py` / `image_processing_gemma4.py`.

## Architecture facts

**Sidecar** (`clip`, projector `gemma4v`, 356 tensors): 27 ViT blocks, width
1152, 16 heads of 72, FFN 4304, eps 1e-6, patch 16, `projection_dim` 5376.
Matrices are BF16; norms, `v.patch_embd` (16×16×3→1152, no bias),
`v.position_embd` ([2][10240][1152]) and `v.std_bias`/`v.std_scale` are F32.
No tensor has a bias, and this checkpoint has no clamp scalars
(`use_clipped_linears: false`).

**Encoder**, per image of `W×H` pixels (both multiples of 48):
1. `p = 2·(rgb/255) − 1`, then a 16×16 patch convolution. Patches are row-major:
   `x = i mod (W/16)`, `y = i div (W/16)`.
2. `h = patch + pos[0][x] + pos[1][y]` (learned tables, no interpolation).
3. For each block:
   - `a = rms(h)·ln1`.
   - `Q = rope2d(rms(Wq a)·q_norm)`, `K = rope2d(rms(Wk a)·k_norm)`, `V = rms_noweight(Wv a)`.
   - Full bidirectional attention with **scale 1.0**.
   - `h += rms(Wo·att)·attn_post_norm`.
   - `h += rms(Wd(gelu_tanh(Wg b)·(Wu b)))·ffn_post_norm`, with `b = rms(h)·ln2`.
4. `rope2d`: NEOX on each half of the head. Dims [0,36) rotate by `x` and
   [36,72) by `y`, each as pairs (i, i+18), θ = 100.
5. Pool: 3×3 average over the patch grid, giving `(W/48)·(H/48)` rows, then
   multiply by √1152.
6. Standardize: `(h − std_bias)·std_scale`.
7. Embed: `rms_noweight(h)`, then `mm.input_projection` 1152→5376.

**Text side** (`llama.cpp src/models/gemma4.cpp`, `llama-kv-cache.cpp`):
- Image rows take the encoder output **unscaled**; only token embeddings are
  multiplied by √5376. The first-layer `attn_norm` applies to both.
- Positions stay 1D and consecutive. There is no M-RoPE.
- The 31B uses `LLAMA_NON_CAUSAL_TYPE_SWA_ONLY` for image chunks (E2B/E4B are
  always causal). Every row of one image may attend to every key of that same
  image in **sliding** layers; the ordinary window still masks older keys.
  Global layers stay causal. mtmd decodes each image as one non-causal batch,
  so an image never splits across forwards. transformers agrees:
  `modeling_gemma4.py` builds sliding masks as `window AND (causal OR
  same-image block)` over image soft tokens only, and global masks as causal.

**Prompt:**
- The chat template emits `<|image|>` for each image part, in content order.
- The processor replaces it with `<|image>` (255999), N × `<|image|>` (258880)
  and `<image|>` (258882). No newlines are added.
- llama.cpp wraps the embeddings with the same `<|image>`/`<image|>` pair.

**Resize:**
- HF (`get_aspect_ratio_preserving_size`): scale by `sqrt(max_patches·256/(H·W))`,
  where `max_patches = budget·9`, then floor each side to a multiple of 48.
  A side that floors to 0 becomes 48 and the other side is capped. The image is
  then resized with antialiased bicubic.
- The processor default is 280 soft tokens (valid budgets: 70, 140, 280, 560,
  1120). llama.cpp instead clamps to [70, 1120] tokens.

## Decisions

- **Budget:** follow the HF processor, filling a 280-token budget up or down.
  llama.cpp can spend up to 4× more tokens on large images; a budget option is
  deferred until a request needs it.
- **Resampler:** move Qwen's PIL-exact fixed-point bicubic resampler to
  `src/core/image` so both models share it. Qwen output stays bit-identical.
- **Code placement:** the Gemma encoder lives in `src/models/gemma4/vision/`. It
  shares the BF16 hipBLASLt GEMM and image decoding, but not the Qwen ViT: that
  graph differs in norms, positions, attention and the merger.
- **Oracle:** the CPU FP32 encoder reference and llama.cpp. llama.cpp is fed
  images already resized to 48-multiples within its token range, so both see
  identical pixels. Text quality is checked with teacher-forced logits on image
  prompts.

## Milestones

**V1 — Prompt and preprocessing (CPU, hosted-CI testable).**
- Shared resampler.
- Gemma resize sizing.
- Template image parts.
- Token expansion with boi/eoi.
- Per-image prefix identities (SHA-256 of pixels, grid, offset and encoder
  identity), matching Qwen's `cache_prefixes` contract.
- *Exit:* template goldens with image parts match jinja2. Sizing matches the HF
  formula on edge cases (tiny, extreme aspect, exact fit).

**V2 — Encoder.**
- CPU FP32 reference, then the HIP BF16 encoder with stage observers.
- *Exit:* the GPU encoder matches the reference at each stage within declared
  tolerances. Final embeddings match llama.cpp's `gemma4v` output for the same
  pixels.

**V3 — Engine.**
- `Executor::Forward` takes optional image rows (unscaled embeddings) and image
  spans. Sliding attention extends each span's key limit to its end.
- Prefill chunks never split a span.
- `Session::Sync` limits prefix reuse to the longest prefix whose images also
  match.
- Snapshots record span identities.
- *Exit:* text-only outputs and snapshots are bitwise unchanged. Image prompts
  give top-1/KL against teacher-forced llama.cpp logits within the text-model
  limits.

**V4 — Serving and CLI.**
- `--mmproj` for Gemma in `gufo serve llm` and `gufo prompt --image`.
- `Gemma4TextRunner::PreparePrompt`/`SetPromptContext`.
- RAM/disk cache identities.
- *Exit:* OpenAI image_url requests work, including a multi-turn replay that
  reuses the image prefix. A changed image never reuses cached KV.

**V5 — Performance and docs.**
- Encoder and image-prefill timings against llama.cpp.
- README/QUALITY/BENCHMARKS rows, plus EXPERIMENTS entries for kernel work.

## Progress

**2026-09-28 — Plan.** Merged `origin/main` (JSON Schema output) into
`feat/gemma`. Structured output stays Qwen-only: Gemma does not build a
constraint vocabulary. Downloaded and verified the BF16 sidecar.
