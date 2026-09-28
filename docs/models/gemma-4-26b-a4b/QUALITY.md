# Gemma 4 26B-A4B quality

**Decode matches the scalar reference up to routing near-ties; greedy MTP
matches single-token decoding; prefill stays within 4e-5 of the reference,
where llama.cpp lands at 0.4.** Unsloth UD-Q4_K_XL and UD-Q6_K_XL targets with
the Unsloth Q8_0 `gemma4-assistant` drafter;
[identities](artifacts/model-identities.json). The arbiter is Gufo's scalar
CPU implementation of the llama.cpp `gemma4` graph, routed experts included,
over the same GGUF (FP32 activations, binary16 KV,
`src/models/gemma4/reference.cpp`). These are not unquantized-model or
GGUF-conversion checks. Measured September 28, 2026.

| Check | UD-Q4_K_XL | UD-Q6_K_XL |
| --- | --- | --- |
| GPU decode (FP32 activations) vs scalar reference, 22-token chat prompt | Mean KL 2.7e-5 (limit 1e-4), max 2.6e-4, top-1 22/22 | Mean KL 6.8e-7, max 9.1e-6, top-1 22/22 |
| Eight-row verification | Bit-identical to single-token decode | Bit-identical to single-token decode |
| GPU prefill (binary16 activations, binary16 WMMA attention) vs scalar reference, same prompt | Mean KL 3.7e-5 (limit 1e-3), top-1 21/22 (a near-tie) | Mean KL 3.7e-5, top-1 22/22 |
| Bulk prefill vs exact rows, 1542-token conversation past the window and ring | Mean KL 0.027 (limit 0.06), top-1 1490/1542 | Mean KL 0.030, top-1 1490/1542 |
| Bulk prefill vs exact rows, 1353-token repetitive prompt (reported, not a gate) | Mean KL 1.74, top-1 794/1353 | Mean KL 1.80, top-1 798/1353 |
| Greedy MTP vs single-token decode with the drafter loaded, three prompts, plus prompt-lookup copies; batched sessions vs their own decode | Identical token IDs (`gemma4.target`) | Identical token IDs |
| Sampled MTP | Seeded replay repeats the same tokens (`gemma4.target`) | Same |
| Session state | Prefix extension, rewind and snapshot restore after a ring wrap continue bit for bit (`gemma4.target`) | Same |

## Routing sensitivity

Top-8 expert selection is discrete. Two exact FP32 decodes that differ only
in the summation order of one RMSNorm (256- versus 1024-thread blocks) flip
near-tied expert choices. They land at mean KL 1.3e-6 and 2.7e-5 from the
reference, with single positions up to 3e-4. The expert-model limits are
therefore wider than the dense 31B's (decode 1e-4, prefill 1e-3, conversation
0.06, recorded at introduction).

## Activation precision

This checkpoint's activations carry large per-block outliers, so Q8_1
activations (32-value blocks with one scale) cost far more than on the 31B.
All rows below are the scalar reference with its projection inputs rounded
as listed, against unrounded FP32, on the 22-token prompt:

| Projection inputs rounded | Mean KL | Top-1 |
| --- | ---: | ---: |
| Every projection, Q8_1 | 0.74 | 19/22 |
| Attention output only, Q8_1 | 0.44 | 18/22 |
| K / V only, Q8_1 | 0.12 / 0.066 | 20/22 / 21/22 |
| Dense MLP gate / up / down only, Q8_1 | 0.064 / 0.066 / 0.085 | 20/22 each |
| Expert gate/up / down only, Q8_1 | 0.023 / 0.017 | 20/22 / 22/22 |
| Every projection, BF16 | 0.066 | 20/22 |
| Every projection, binary16 | 3.7e-5 | 21/22 |

Gufo therefore prefills with binary16 activations. llama.cpp b11069 (ROCm)
runs Q8_1 activations. Its teacher-forced logits on the same tokens sit at
mean KL 0.42 (max 4.4, top-1 18/22) from the FP32 reference.

Its layer-0 intermediates confirm the reference's semantics. Router logits
agree to 9.5e-7 relative, and the top-8 experts, their weights and the expert
input norm (1.1e-7) match exactly. Its dense projections match a Q8_1
emulation of the reference to 1.4e-4. Its expert projections differ from the
reference by 1–6%, beyond what Q8_1 rounding explains.

## Reproduce

```sh
export GUFO_GEMMA4_MODEL=/path/to/gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf
export GUFO_GEMMA4_MTP_MODEL=/path/to/mtp-gemma-4-26B-A4B-it.gguf
nix develop -c cmake --build --preset gpu-test --target gemma4_target_test \
  gemma4_moe_ops_test gemma4_projection_ops_test
nix develop -c ctest --preset gpu-full -R '^gemma4\.(target|moe_ops|projection_ops)$' \
  --output-on-failure
```

`gemma4.moe_ops` checks routing against FP64, every routed format (Q4_K,
Q5_K, Q6_K, Q8_0 gate/up; Q5_1, Q8_0 down) against FP64 dots and batch
invariance from one to sixteen rows, the routed Q6_K prefill GEMM and the
mixture epilogue. Reference and llama.cpp logits:
`gemma4_reference_probe --model M --chat "Write one sentence about the sea
near Genoa." --half-kv --tokens-out T.i32 --logits-out R.g4lg`, then
`llama_logits --model M --tokens T.i32 --output L.g4lg`
(`tools/gemma4/build_llama_logits.sh`) and
`tools/gemma4/compare_logits.py R.g4lg L.g4lg`.

## Benchmark method

September 28–29, 2026; one warmed sample per point (sampled MTP: three seeds),
greedy, thinking off, the same driver workloads, server flags and table grid
as the [31B](../gemma-4-31b/QUALITY.md#benchmark-method), per quant. Gufo
drafts up to seven tokens under the calibrated policy (`--draft-tokens 7`),
llama.cpp up to four. Single-user tables, multi-user AR, loading and memory
ran on revision 6a0af62; multi-user MTP and its C1 AR completion references
on 7133015, after the batch-aware draft policy (single-session decoding is
unchanged between the two). Loading drops the page cache
(`sync; echo 3 > /proc/sys/vm/drop_caches`) before each launch. Every Gufo
multi-user MTP completion (C1–C8, both workloads, both quants) matches its
AR C1 hash; commands, counts and server flags are recorded per row in the
artifacts.
