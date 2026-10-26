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

### Functional suites after M2

`tests/functional/run.py` with the `deepseek4` sampling preset (V4.1 is
served with the V4 defaults), 32K context, 2 sessions, against the M2 build:

| Suites | Result |
|---|---|
| discovery, responses, stops, conversation, tools, tool-edges, tool-streaming, structured, sampling-defaults, long-context | passed |
| cache (RAM and disk, restart, disk checkpoint spacing) | passed |
| tool-reasoning | 8 of 9 cases pass; the first case fails |

The failing case asks the model to quote a corrupted docstring containing a
literal `<tool_call>` opener and then call `edit`, with 1,024 output tokens at
low effort. V4.1 IQ1_M is still reasoning when it reaches the limit. With 4,096
tokens it calls `edit` with the exact arguments after 2,529 tokens at low
effort and 1,953 at high. The fixture budget is unchanged.

`tests/functional/pi_agent.py` (Pi 0.87.0, server sampling defaults): 13 of
15 tasks pass, covering simple, tools, edit, creation, bugfix,
literal-protocol and wide-tool. Two of three `tools` runs wrote `hello.txt`
as `bonjour` plus a newline where the task asks for exactly `bonjour`. The
model generates that newline inside the string parameter. The official
`encoding.py` and `deepseek-recipe` both keep string parameter values
verbatim, as Gufo does.

### Component cache (M4)

Sessions serve the component cache: compressed-KV and index-K rows of the
four KV source layers are lent rows, the window rings, compressor frontiers
and frontier logits private state, and the Engram history is rebuilt from
the tokens. Capture cost on a 55K-token conversation (65K context):

| Capture | Whole-state snapshot | Component cache |
|---|---:|---:|
| 55,014-token prompt | 284 MB, 274 ms | 119 MB, 9.1 ms |
| 10-token follow-up turn | 182 MB, 140 ms | 25 MB, 1.5 ms |

Functional suites (32K context, 2 sessions) against the whole-state
snapshot build: `cache` (RAM and disk, restart, disk checkpoint spacing)
passes qualification. cache-edits, cache-rotation, cache-shared-prefix and
cache-concurrency pass on both builds with the same outputs. Per request,
captures take a median 0.066 of the old time and restores 0.44; prefill
and decode medians are unchanged (0.998). cache-growth and cache-depth fail
on both builds with the same replies: V4.1 answers a trivial turn without
thinking, and the depth fixture is 16,210 tokens with this tokenizer, below
its 16,384 floor.

Flags that remain, from alternating runs:

- One rotation return reuses 2 more tokens (8,470 instead of 8,468) and
  prefills 216 ms instead of 202: its 7-token batch takes 136 ms where the
  old 9-token batch took 120, also without a restore (BACKLOG item 7).
- Restores queue 5-6 ms longer in cache-shared-prefix (`long_tasks_1-3`),
  outside the restore itself; a fresh server with few checkpoints shows no
  difference. Time to first token is still lower than before.
- The first token now comes before the capture completes, so single-token
  and concurrent decodes count up to one step more while the requests end
  sooner (cache-concurrency `identical`: 8,107 against 8,124-8,140 ms).

## DSpark drafter (M5)

`tools/ds41/dspark_reference.py` recomputes a draft on the CPU from the
features and anchor that `ds41_load_probe --dspark-dump` writes, following
the official DSpark block. On the 1,871-token roadmap prompt with the
unsloth Q8_0 drafter (MXFP4 experts) the five drafts match the GPU exactly;
head logits before the Markov bias agree to cosine 0.9995-0.9998 (max
|diff| 0.3-0.7 on logits up to 13-21).

Drafts never change the output: the target evaluates the anchor and the
drafts in one batch with decode numerics, the request's sampler draws every
row, and the first rejected draw becomes the next token. Greedy generation
with DSpark is token-identical to plain decoding in every probe run (128,
256 and 384 tokens, both drafters, every draft width) and over HTTP, also
when the request follows a sampled one and restores from the cache.

Served through the component cache (`cache` functional suite, 32K
context, 2 sessions, Q2_K/Q4_K drafter) every case passes, and 29 of 33
replies are byte-identical to the run without a drafter. The other four are
sampled cancellation cases interrupted after a fixed time: DSpark samples in
a different order and generates more tokens before the cut, so the partial
reply and its follow-up differ; each passes its exact-continuation check.

Prompt-lookup copies are verified the same way, in batches of at most
seven rows: from eight rows the batched attention kernel replaces the
decode one, and `ds41_load_probe --spec-check 16 --spec-width W` keeps
bit-identical logits against single-token decode for 6 and 7 rows but not
for 8 (max |diff| 11.6, same argmax). Greedy generation with lookup is
token-identical to plain decoding on the file-edit, roadmap and chat
prompts, and the `cache` suite with the drafter passes every case with all
30 completed replies byte-identical to the run without lookup. The lookup
state that the tokens cannot rebuild (resume point, request start) travels
in snapshots and in the component cache metadata.

Each step now emits the token after its accepted proposals before
evaluating it; the next step verifies it as its first row. Greedy output
stays token-identical to plain decoding with and without the drafter (file
edit, 512 tokens), and seeded sampled `gufo prompt` output (temperature 1.0,
160 tokens) is byte-identical to the previous build with DSpark and without
a drafter. A reply's last token is then evaluated by the next turn's
prefill, as in plain decoding, so DSpark follow-ups prefill one token more
than before (cached 187 → 186 tokens) and three cancellation follow-ups in
the `cache` suite changed their sampled text; every case passes. Without a
drafter all 30 completed `cache` replies are byte-identical to the previous
build.

`ds41.attention` checks the drafter's block attention (ring rows with
wrap-around plus every block row, with sinks) against an fp64 reference
(≤ 7e-7 relative), the hc mean exactly and the Markov argmax (first index
of equal maxima, non-finite values skipped).

## Decoder SWA bounded replay (M5)

Prompts run the decoder layers (20-39) over their last 140 positions only,
as DeepSeek's DeepInfer does with its V4.1 defaults (`compact` mode, 140
kept tokens: the 128-row window plus 12 rows of backtracking). Layer 20
still compresses every position, so the global KV and index keys are
complete; the replayed rows' windows start at the first replayed position.
In serve each prefill segment between cache stops is one such prompt, so a
state is only captured where its windows are complete, and a restored
prefix continues exactly as the cold request did.

`ds41_load_probe --replay-check 256` prefills the ids but the last 256
once exactly and once with bounded replay, then scores the last 256
teacher-forced after each:

| Prompt | NLL exact | NLL replay | Top-1 agreement | Mean KL (max) |
|---|---:|---:|---:|---:|
| `docs/KV-CACHE.md`, 8,863 tokens | 1.9495 | 1.9488 | 255 / 256 | 0.00097 (0.0135) |
| `v41_decode.cpp`, 8,192 tokens | 0.4029 | 0.4106 | 254 / 256 | 0.0023 (0.10) |
| `v41_decode.cpp`, 23,848 tokens | 0.2147 | 0.2159 | 254 / 256 | 0.00021 (0.0073) |

Top-1 hits on the reference token: 148/148, 229/230 and 240/240.
`ds41.attention` checks the window floor against the fp64 reference in the
batched, scalar and split kernels.

Served (32K context) against the previous build, the `long-context`,
`conversation` and `cache` suites pass every case. The needle is still
recovered, and its cancelled thinking replay and resume give the same
answer with shorter reasoning (27 → 17 tokens), still identical to each
other; three sampled thinking-tool cancellation replies changed text and
pass their exact-continuation checks. 876 of 878 per-request timings pass,
two decode max inter-token gaps are inconclusive (42.9 → 46.1 and
43.9 → 46.9 ms).

With the DSpark drafter (Q2_K/Q4_K), whose features now come from the
replayed rows only, greedy output after the 8,863- and 8,192-token prompts
stays token-identical to plain decoding (256 tokens each, bounded replay
and `--exact-prefill`): 50.77 and 42.46 tok/s against 52.72 and 32.75
with exact prefill, which continues with different text. Served with the
drafter against the previous build, `long-context` and `cache` pass every
case with the same output changes as above; over all requests the same 116
draft rounds accepted 75.7% of drafts (68.3% before) and decode took
8.22 s (9.02 s). 59 of 602 timings are inconclusive: short replies (5-8
tokens) that took one draft round more or fewer than before.

`--exact-prefill` (`serve`, `bench`, `prompt`) evaluates every decoder row:
against the build before bounded replay, all 70 request outputs of the
`long-context`, `conversation` and `cache` suites are byte-identical.

## Vision (M6)

**Encoder against the official forward.** `tools/ds41/vision_reference.py`
runs the official preprocessing, ViT and aligner on the CPU in the official
BF16 default dtype (and in FP32 with `--fp32`, as a precision yardstick);
`ds41_vision_probe` writes the GPU span. Every grid matches the reference
and the padded pixels are byte-identical to PIL:

| Image | Padded | Tokens | Relative RMS to BF16 reference | Median row cosine | Worst row cosine |
|---|---|---:|---:|---:|---:|
| carrots.jpeg (official example) | 1036 × 714 | 444 | 4.26% | 0.99994 | 0.780 |
| corn.jpeg (official example, enlarged) | 658 × 462 | 189 | 1.81% | 0.99995 | 0.992 |
| 2000 × 300 synthetic | 2002 × 308 | 394 | 2.92% | 0.99991 | 0.960 |
| 250 × 1900 synthetic | 252 × 1904 | 324 | 2.60% | 0.99992 | 0.972 |
| 97 × 61 synthetic (enlarged) | 686 × 434 | 200 | 1.49% | 0.99995 | 0.995 |
| 4000 × 3000 synthetic (token cap) | 1512 × 1134 | 1,001 | 2.42% | 0.99992 | 0.925 |

The differences are BF16 rounding sensitivity, not an error: on carrots the
official BF16 forward itself is 5.66% from the FP32 forward (worst row
0.842), and the GPU span is 5.55% from it (worst row 0.890). The few
divergent rows are scattered over the image.

**Image routing bias.** No published GGUF carries the gates' `bias_vl`, so
image positions would otherwise route with the text bias (layer 3: up to 4.9
apart, correlation −0.22). Eleven greedy questions on the two official
examples and four synthetic images (text transcription, counting, a bar
chart, digits; `--reasoning-effort none`): with the image bias, all nine
questions with a reference answer are right; with the text bias in its place
(a control sidecar), eight are, and the carrots are counted as 5 instead of
4. Three of the eleven answers change. One differing answer is too little to
size the effect.

**Serving.** `tests/functional/run.py` (`deepseek4` preset, 32K context, two
sessions, `--mmproj`): `discovery` (text and image inputs), `image-inputs`,
`tool-images`, `image-count` (with cancellation and disk replay) and
`messages-image-counts` pass. A follow-up turn after an image restores the
image prefix from the cache (219 of 232 tokens). The text suites
`conversation`, `tools` and `cache` (RAM, disk and restart) without
`--mmproj` pass their assertions. The report still fails one measurement
check: with `--speculative off` the harness expects zero drafts, and two
`tools` requests verified prompt-lookup copies (16 and 28 copied, 15 and 27
accepted). The build before M6 reports the same. Drafter-less prompt lookup
stays on in `off` mode by choice, so this check does not apply to this model.

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
smoke continuation keeps all 9 greedy tokens, KL ≤ 3.2e-3 per step. With batched routed gate/up stored
through the SwiGLU epilogue of one MMQ pair (one-ulp SwiGLU differences) the
batched path scores 6.300, agreement 96.1%, mean |ΔNLL| 0.081. With a routed tile width per expert (rounding-level
differences in the expert projections) the batched 3K run scores 6.307,
agreement 95.5%, and the batched 18.5K run 6.277, agreement 95.9% (oracle
6.267).
