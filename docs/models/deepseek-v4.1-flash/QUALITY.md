# DeepSeek V4.1 Flash Quality

Correctness of the ROCm decoder against the CPU/torch oracle in `tools/ds41/`.
The oracle ports the official `inference/model.py` and `inference/engram.py`,
dequantizes the same GGUF and keeps the official KV fake quantization
(FP8 window, FP4 compressed and indexer KV). Parity therefore measures the
engine, not the quantization.

Artifact: ISTA-DASLab GSQ-RCO `IQ1_M` (4 shards) with
`DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf`. Hardware: one Strix Halo (gfx1151).

## Token parity

Greedy continuation of "The capital of France is Paris. The capital of Germany
is" (13 prompt tokens, 8 generated). Both engines produce
" Berlin. The capital of Italy is Rome".

| Step | Top-1 equal | KL(oracle ‖ GPU) | p(top), oracle / GPU |
|---:|:---:|---:|---|
| 0 | yes | 1.2e-4 | 0.986 / 0.985 |
| 1 | yes | 9.7e-5 | 0.876 / 0.874 |
| 2 | yes | 4.9e-4 | 0.850 / 0.848 |
| 3 | yes | 1.8e-4 | 0.992 / 0.992 |
| 4 | yes | 7.4e-5 | 0.997 / 0.997 |
| 5 | yes | 7.4e-4 | 0.647 / 0.644 |
| 6 | yes | 2.4e-5 | 0.998 / 0.998 |
| 7 | yes | 3.0e-5 | 0.998 / 0.997 |
| 8 | yes | 4.4e-5 | 0.862 / 0.860 |

## Teacher-forced 3K tokens

2,999 positions of `docs/KV-CACHE.md`, scored token by token. This range uses
the indexer's top-512 selection on every compressed-attention layer.

| Positions | Oracle NLL | GPU NLL | Oracle top-1 | GPU top-1 | Argmax agreement |
|---|---:|---:|---:|---:|---:|
| 0-511 | 2.479 | 2.479 | 0.494 | 0.484 | 0.951 |
| 512-1023 | 1.420 | 1.416 | 0.664 | 0.658 | 0.977 |
| 1024-2047 | 1.573 | 1.577 | 0.634 | 0.629 | 0.955 |
| 2048-2998 | 2.016 | 2.012 | 0.528 | 0.530 | 0.955 |
| all | | | | | 0.958 |

Perplexity: oracle 6.309, GPU 6.306; mean |ΔNLL| 0.078 per position.

## Teacher-forced 18.5K tokens and candidate blocks

18,499 positions of repository documentation (`docs/KV-CACHE.md`,
`docs/TESTING.md` and following), scored through 2048-token batches. Past
16,384 tokens layer 20's candidate blocks (top 2048 of the 8-row blocks) start
masking the rescoring layers.

| Positions | Oracle NLL | GPU NLL | Oracle top-1 | GPU top-1 | Argmax agreement | Mean \|ΔNLL\| |
|---|---:|---:|---:|---:|---:|---:|
| 0-2047 | 1.761 | 1.762 | 0.604 | 0.601 | 0.959 | 0.077 |
| 2048-8191 | 2.095 | 2.096 | 0.531 | 0.530 | 0.947 | 0.085 |
| 8192-14335 | 1.681 | 1.681 | 0.627 | 0.627 | 0.964 | 0.069 |
| 14336-16382 | 1.502 | 1.504 | 0.666 | 0.668 | 0.962 | 0.063 |
| 16383-17399 | 1.472 | 1.469 | 0.636 | 0.640 | 0.964 | 0.075 |
| 17400-18498 | 2.338 | 2.347 | 0.532 | 0.534 | 0.956 | 0.090 |
| all | | | | | 0.958 | 0.076 |

Perplexity: oracle 6.267, GPU 6.271. Long-context parity holds past the
threshold. The score alone does not prove the mask: with the GPU mask
disabled the same run also gives perplexity 6.271 and agreement 95.7%, because
at 18.5K tokens the dropped blocks are rarely in a top-512. The mask is
therefore checked directly: `ds41.candidate_mask` compares the GPU mask and the
masked scores with a host port of the reference selection, exactly, for
prefill batches below, at and across 16,384 rows and for decode tokens up to
40K rows (up to 70,872 dropped rows per case).

The same positions scored at the native 1M context (576-token batches,
indexer scores in 64-token groups, per-block mask): perplexity 6.273,
agreement 95.9%, mean |ΔNLL| 0.075; 97.2% agreement over 16,383-17,399.

## Chat template

`tests/models/deepseek_v41_flash/fixtures/encoding-1017.json` holds 12
conversations rendered by the official `encoding/encoding.py`
(SHA-256 `13231c59…`), with the deepseek-recipe serving conventions: request
tools join a leading system message, and `developer` is a system message.
The cases cover chat and thinking modes, every reasoning-effort level, history
with and without tools, parallel calls whose results arrive out of order, a
mid-conversation system message, Unicode, and escapes. `ds41.template` renders
all 12 byte-for-byte.

## Serving

`gufo serve` on IQ1_M, 16K context:

| Request | Result |
|---|---|
| Chat, no thinking: capital of France | "The capital of France is Paris." |
| Thinking, low effort: bat-and-ball puzzle | correct reasoning, answer $0.05 |
| Tool call: Rome weather for 3 days | `get_weather {"city":"Rome","days":3}`, `finish_reason=tool_calls` |
| Tool result returned | summarizes the three forecast days; 390 of 466 prompt tokens reused |

Measured at M1, before the M2 performance work: decode 12-14 tok/s with
short prompts, prefill one token at a time. Current speeds are in
[BENCHMARKS.md](BENCHMARKS.md).

## Batched prefill and M2 kernels

The same 3K positions scored through 2048-token batches (matrix-core
attention and indexer, bounded routed MoE tiles): perplexity 6.310 against
the oracle's 6.309, argmax agreement 95.8%, mean |ΔNLL| 0.080. Scored token by
token through the decode path (split-row attention, FP32-activation dense
GEMV): perplexity 6.303, agreement 96.1%, mean |ΔNLL| 0.077. Across the M2 changes the batched perplexity moved within
6.296-6.319: half-precision attention operands and the FP8 fake quantization
of the attention output turn rounding-level differences into small logit
changes. The smoke continuation keeps all 9 greedy tokens, KL ≤ 5.4e-3 per step.

Kernel checks: `ds41.candidate_mask` (above, plus the indexer top-k with tied scores); `ds41.attention` compares both attention paths with an fp64
reference (split decode 1.6e-7, batched 3e-4 relative error), the indexer scores (fp32
decode 2.4e-7, matrix-core batches 3.7e-4) and the fused q_a/KV norm against the separate launches;
`ds41.mmq_typed` compares batched, bounded-routed and grouped matvec launches
with the matvec path for every V4.1 weight type, and the single-token Q5_K/Q8_0
GEMV with an fp64 dot of the dequantized rows, and the IQ routed-expert GEMV
with the ggml matvec (shared and per-slot inputs, dropped slots write zeros);
the fused gate/up SwiGLU passes match the unfused ones.
With it the decode path scores perplexity 6.305 on the 3K positions,
agreement 96.5%, mean |ΔNLL| 0.075; the smoke continuation keeps all 9 greedy
tokens, KL ≤ 1.6e-3 per step. With the hc collapse fused into the following RMSNorm (a different
summation order) the decode path scores 6.291, agreement 96.0%, mean |ΔNLL|
0.073, and the batched path 6.324, agreement 95.9%, mean |ΔNLL| 0.078; the
smoke continuation keeps all 9 greedy tokens, KL ≤ 3.2e-3 per step.
