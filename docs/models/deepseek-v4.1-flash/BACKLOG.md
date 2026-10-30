# DeepSeek V4.1 Flash backlog

Open work only. Everything already measured, retained or rejected, is in
[EXPERIMENTS.md](EXPERIMENTS.md); read the matching entries before retrying
an item. The streaming design is in [STREAMING.md](STREAMING.md). Impact
figures are estimates unless stated.

## Where the time goes

**All modes, 2026-10-29** ([trace sweep](artifacts/2026-10-29-trace-sweep/summary.txt);
measured ceilings 240 GB/s DRAM read, 55.4 TFLOP/s FP16/BF16 WMMA). A
decoded token reads 7.15 GiB of weights (`IQ1_M`) or 7.4 GiB (`IQ2_XXS`),
a ~31–32 tok/s ceiling.

| Mode (96-token prompt, 256 tokens) | `IQ1_M` | `IQ2_XXS` streamed | Position |
| --- | ---: | ---: | --- |
| pp2048 (tok/s) | 1,393 | 1,208 | 95% GPU busy; WMMA at ~55% of peak |
| AR greedy / sampled (tok/s) | 24.0 / 24.4 | 22.0 / 22.0 | 39.0 / 40.4 ms of kernels per token against ~31; 2.9 / 5.4 ms idle |
| DSpark greedy / sampled (tok/s) | 38.1 / 38.5 | 34.4 / 35.1 | Verification at the read ceiling (24–36 distinct experts per layer per step) |

Single-token decode bandwidth: output head 99%, dense Q5_K/Q8_0 GEMVs
86–97%, routed gate+up 85% (IQ1_S) and 71% (IQ2_XXS layers), routed down
71–72%, BF16 router 66%. 2.35 of AR's 2.9 ms idle per token are gaps under
5 µs between ~975 launches.

**Streamed `IQ2_XXS` decode** (rocprofv3, warm cache, tg128 after pp512,
medians per token, 2026-10-27): 50.1 ms per token, 40.3 ms of kernels and
9.8 ms of GPU idle. Since then the per-layer tables removed most of the
copies (48.1 ms traced).

| Part | ms/token |
| --- | ---: |
| Dense Q5_K GEMV | 22.2 |
| Routed expert GEMV | 11.5 |
| Other kernels (attention, hyper-connections, router, …) | 6.6 |
| Idle: miss reads between the hit and miss launches | 5.3 |
| Idle: small blocking copies (now 23 per token, from ~200) | ~3 |

The dense GEMV runs near the measured DRAM ceiling, so verification of
several rows per weight read (DSpark) is the main decode lever. A streamed
DSpark tg512 trace splits into routed experts 44% (36% the 8-row
verification groups, ~36 distinct experts per layer at the single-token
per-expert cost), dense GEMV 31%, other kernels 11%, idle 10% and the
drafter 4%.

**Resident `IQ1_M` decode** (2K context, 2026-10-25): 42.4 ms wall, ~2.7 ms of
it gaps between ~1,000 dependent launches. Dense projections and the output
head take ~23.9 ms near the 235 GB/s read peak.

**Prefill** (pp2048 at 32K with bounded replay, 2026-10-26): 1,773.84 ms of
GPU time; routed FP16 606 ms, dense FP16 518, attention 219, mHC 137,
indexer 83, top-k 45. After E19/E20 (2026-10-27) the indexer scores take
about a quarter and the top-k a twentieth of that per layer at 128K; native
pp2048 runs 1,182 tok/s at 32K, 1,155 at 64K and 1,098 at 128K.

**Verification rows** (`--spec-bench`, 1–8 rows, 2026-10-27, ms):

| Model | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `IQ2_XXS` streamed | 43.4 | 58.0 | 69.8 | 78.8 | 96.6 | 107.5 | 120.4 | 133.8 |
| `IQ1_M` resident | 41.1 | 52.8 | 65.5 | 75.7 | 87.0 | 98.3 | 113.5 | 123.6 |

## Open items

| # | Item | Area | Est. impact | Confidence |
| ---: | --- | --- | --- | --- |
| 3 | Expert reads for rejected verification rows | Streamed DSpark | Part of the 36% verification-expert time | Low |
| 4 | Launch gaps between decode kernels | Decode | ~2.35 ms/token AR (~6%), ~3 ms per DSpark step | Medium |
| 5 | Compact selection for `--exact-prefill` rescoring | Long context | ~85 ms per masked layer per 2K chunk at 128K, `--exact-prefill` only | Medium |
| 6 | Routed FP16 kernel: weight decode per slot | Prefill | ~4.4 ms of ~21 per layer | Low |
| 7 | Remaining prefill glue | Prefill | ~25–50 ms per 2K batch | Medium |
| 9 | Stable-boundary prefill pass per chat request | Serving | ~65 ms per request | Medium |
| 10 | Concurrent drafter forwards | Serving decode | ~9 ms per drafting session per step; DSpark 5% under AR at C8 | Medium |
| 13 | Route bias toward cached experts (changes outputs) | Streamed decode | Opt-in approximation | Low |
| 14 | Higher-precision Engram tables | Quality | Unmeasured | — |
| 15 | Smaller items | Various | Small | — |
| 16 | Multi-user AR decode below the 2026-10-26 card | Serving decode | Distinct-prompt C8 63.0 tok/s against a ~93 tok/s read ceiling | Medium |
| 17 | Single-row routed expert GEMV bandwidth | Decode | ~1.7 ms/token (~4% AR) | Medium |
| 18 | Small decode kernels: hyper-connections, router, routing | Decode | ~1–1.5 ms of ~4 ms/token | Medium |
| 19 | DSpark host round trips | Speculative decode | ~0.7 ms per step | Low |
| 20 | `markov_select` | Drafter | Part of 1.6 ms per step | Medium |
| 21 | Two-layer-ahead decode prediction with a usefulness gate | Streamed decode | Small warm; part of the read stall cold and after long prompts | Low |
| 22 | Decode misses after a long prompt under scan-resistant admission | Streamed decode | Unmeasured | — |
| 23 | Streamed `IQ2_XS` with cache-aware routing | Quality option | Unmeasured; lower KL than `IQ2_XXS` at a residency cost | Low |
| 27 | Streamed kernels for the `IQ2_XS` build (IQ3_XXS down, IQ2_S gate/up) | Model support | Prerequisite for item 23 | High |
| 24 | Selection-only miss penalty, mako's form of item 13 (changes outputs) | Streamed decode, short prefills | Up to the ~5 ms/token of miss-read idle; more cold, after long prompts and in verification | Medium |
| 25 | Direction projection and steering vectors | Feature | Per-request behavior edits; ~0.1–0.5 ms/token if unfused | Medium |
| 26 | LoRA adapters (GGUF `adapter.type = lora`) | Feature | Adapter compatibility; rank-1 abliteration adapters reduce to item 25 | Medium |

### 3. Expert reads for rejected verification rows

On streamed DSpark every verification row pays for its own experts, and
verification batches predict no next-layer experts. The verified prefix
already weighs the head's per-row acceptance against measured cycle time
([E18](EXPERIMENTS.md#e18-cost-aware-draft-prefix-retained)), which prices
misses only on average. Verification batches now predict next-layer
experts when a layer misses (E31: +3.5–6.7% on a cold cache, neutral warm).
Open: charge a row the miss bytes of its own routes. Route bias is a separate,
output-changing item (13).

### 4. Launch gaps between decode kernels

Re-traced 2026-10-29 (IQ1_M, tg32 after pp512): 974 launches and 2.9 ms
idle per 41 ms token, ~0.2 ms of it at the step start (output head →
next step's first copy) and ~2.7 ms as 2–3 µs gaps between dependent
launches. Graph replay keeps those gaps (E44: ~0.13 µs saved per launch,
~0.3% per token); only fewer launches recover them, through the fusions
below (~0.2–0.5% each) or by batching the per-session launches of a joint
step (~2,000 per C8 step, mostly attention).

~1,000 dependent launches per token leave ~2.7 ms idle (resident profile).
Left to fuse: the q rope after q_b (40 per token; would need a GEMV epilogue
pairing rows 8 lanes apart), route after the router matvec (40; 7 µs each),
the window copy (43; cannot run before attention, which reads the slot it
overwrites). hc mixes read 2 MB of F32 weights in ~14 µs (80 per token)
regardless of blocks per row. ~90 µs of host work starts each step
(embedding dequant and two synchronous uploads ~70 µs); a device-side
embedding lookup removes most of it.

### 5. Compact selection for `--exact-prefill` rescoring

Verification already rescores in groups of four rows over layer 20's
candidate blocks (E1). Prefill rescoring (WMMA) of the masked decoder index
layers 24, 28, 32 and 36 still scores the full history and masks afterwards,
but with bounded replay those layers see only the last 140 prompt rows:
5.6 ms of scores and 0.7 ms of top-k per layer at 128K, ~25 ms once per
prompt (E34). Compact rescoring pays only under `--exact-prefill`, where the
same four layers score every 2,048-row chunk (~85 ms each at 128K). The
encoder index layers 2, 8 and 14 (~40 ms each per 2K batch at 128K) have no
candidate mask. Keep absolute position tie order, the pinned newest block,
causal partial blocks and the emitted row order.

### 6. Routed FP16 kernel: weight decode per slot

`kernels/rocm/ds41_routed_f16.hip.cpp`, after E25: IQ1_M gate+up takes about
21 ms per 2,048-token layer, of which the weight decode and its permutes are
~4.4 ms and the matrix-core skeleton (block fields, fragment reads,
products) ~14.7 ms, at 39 of the measured 55 TFLOP/s. A decoded fragment
feeds at most four 16-slot tiles, on average 2.3. 128-slot tiles cut the
recorded prefill layers' weight decodes by 19% but ran no faster (E35).
IQ2_XXS's 16-byte random codebook reads still
cost ~2 ms; 8-byte state entries would need ~12 more VALU per decode.
IQ2_S stays on MMQ.

### 7. Remaining prefill glue

Per 2,048-token chunk at 128K of depth (2026-10-28 profile, 1,342 ms of GPU
time): the Q8_0 projections still go through MMQ (30 ms) and its Q8_1 pass
(9.5 ms); the BF16 matmuls take 49 ms (the 384-row router at ~5 TFLOP/s);
the routed rows' closing expand before Engram layers and replay boundaries
8 ms; the shared-expert SwiGLU 5 ms and the routed input's FP16 conversion
4 ms. Each chunk still starts with ~4.5 ms of embedding staging: ~1.2 ms
of host dequantization on four threads, ~1 ms of pinned upload and ~2 ms of
copies into the four residual streams (E33). The Engram
projection (25,600 × 6,144, 17.8 ms per call, two per chunk) still reads
FP32 rows; FP16 rows from the host reader would save 0.8 ms per call and
half the upload, and the shared down projection's SwiGLU input 0.04 ms.

### 9. Stable-boundary prefill pass on every chat request

The prompt cache captures a checkpoint only at a prefill pass end, and every
chat request captures its stable boundary three tokens before the prompt
end: a 2,035-token prompt runs as 2,032 + 3 rows. The 3-row pass costs about
65 ms (`gufo bench -p 3`: 64.6 ms), the only serving-side prefill overhead
found. The checkpoint serves turns whose rendering diverges after the user
message (thinking mode, regenerated or edited replies). Capture it only when
the next turn can need it, or snapshot row N−3 from inside the final pass
(compressor accumulators, SWA windows and bounded replay make this
invasive). Gemma's generation-prompt lookahead is the local precedent. The
policy is shared (`src/cache/retention.cpp`), so other models' reuse must
hold.

### 10. Batch-level draft width

Concurrent DSpark sessions share one drafter pass (E42) and one 16-row
verification (E27), but with different prompts per user DSpark still
trails AR at C4 (53.6 against 55.5 tok/s) and C8 (59.3 against 63.0).
E27's v3, a batch plan that narrows or skips drafting when the joint step
does not pay, is the next lever.

### 13. Route bias toward cached experts (quality-affecting, optional)

Xolotl multiplies streamed experts' router logits by ~0.5 for 13.7 →
30.3 tok/s on a 4090 at ~1 point of top-1 agreement; ×0.25 keeps 85%
agreement. This changes outputs: opt-in only, with its own quality gates.

### 14. Higher-precision Engram tables

Measure speed and quality of Engram tables above IQ4_NL (144 B rows).
[unsloth](https://huggingface.co/unsloth/DeepSeek-V4.1-Flash-1017-GGUF)
`engram/` offers Q5_K (125.89 GiB), Q6_K (150.21 GiB) and MXFP8
(188.83 GiB) in its own split format (`*.gguf_file`, `set_split.py`); BF16
tables (512 B rows, ~366 GiB) would come from the official checkpoint.
Quality: teacher-forced perplexity and agreement against the oracle at 3K
and 18.5K tokens. Speed: prefill and decode at the card depths plus Engram
read and wait times, since rows grow up to 3.6× (E28 measured the current reads at under 1% of warm streamed prefill).

### 15. Smaller items

- **Routed-expert GEMV, wide loads.** The retuned decode kernels reach
  162–205 GB/s; IQ1 down projections stay at ~165 GB/s with a fixed
  per-launch floor (85–96 µs across 13.8–22.7 MB). A plain read of the same
  six experts streams at 230–238 GB/s.
- **Draft vocabulary.** A request-vocabulary draft head, as Qwen and Gemma
  use, with request tokens and a multilingual fallback; the verifier keeps
  the full vocabulary.
- **Tokenizer merges.** E3 memoizes pieces per span; the merge loop still
  rescans adjacent pairs (`bpe_rank`/`bpe_emit_piece`). Qwen's heap merge
  with token-ID pairs is the local precedent.
- **Engram worker setup.** `engram_read` starts its workers per evaluation;
  test a persistent pool if setup becomes visible.
- **Encoder SWA replay on a frontier miss.** Approximate; needs its own
  cache and quality contract.
- **`gufo diagnose`** does not report the streaming statistics.
- **Decode expert GEMV spills.** Several `expert_group_kernel` variants
  use 256 VGPRs and still spill 80–540 bytes per lane to scratch; check
  whether fewer columns per lane or `ext_vector_type` arrays remove them.

### 16. Multi-user AR decode below the card

The card's multi-user tables give every user the same prompt, which keeps
sessions in lockstep and lets a joint pass share experts (E27); with
distinct prompts (2026-10-29, after E41) IQ1_M AR decodes C1 / C4 / C8 at
24.5 / 55.5 / 63.0 tok/s. A C8 step (8 rows, ~42 distinct experts per
layer) took ~115 ms of kernels and ~10 ms of host time against ~86 ms of
reads before E38; the grouped expert GEMV now reads at ~190-210 GB/s for
IQ1_S and IQ2_XXS gate+up and ~175-200 GB/s for the down projections
(standalone), IQ1_M's down at ~180. Refresh the card's multi-user tables
with distinct prompts.

### 17. Single-row routed expert GEMV bandwidth

`expert_gemv_kernel` at one token (2026-10-29 trace, `IQ1_M` resident):
gate+up 27.6 MB in 135.8 µs (203 GB/s, IQ1_S), down 13.8 MB in 80.6 µs
(171 GB/s) and 15.5 MB in 89.0 µs (IQ1_M, 174 GB/s); the IQ2_XXS layers'
gate+up 31 MB in 180.8 µs (171 GB/s). The dense Q5_K GEMVs reach 206–238
GB/s. Each launch covers six experts of 2,304 (gate+up) or 5,120 (down)
rows; check the decode's VALU share (IQ1 and IQ2 codebook decode) against
the loads with `tools/prof/isa_mix.py` before changing the layout. Paired IQ1
down rows (E46) and single-slot IQ1 verification experts on this kernel (E51)
are retained; more tiles per block were slower.

### 18. Small decode kernels

Per AR token: hyper-connections 1.78 ms (`hc_mix_small` 80 launches of
14 µs over 160 MB of FP32 mixing weights, ~0.67 ms at the read ceiling),
the BF16 router 1.0 ms (3.93 MB per layer in 24.9 µs, 158 GB/s), attention
~1 ms, routing and norms ~0.6 ms. Fusing launches also shortens item 4.
Fusing the mix with the shifted collapse was slower in the model (E49).

### 19. DSpark host round trips

Verification argmax now runs on the device (E50). The 2026-10-29 trace of
greedy DSpark C1 (`IQ1_M`) leaves ~0.7 ms of gaps over 20 µs per ~89 ms step,
mostly around the logits copies after the output head (`copyBuffer` →
`copyBufferRectAligned`, ~0.28 ms) and the following copies; downloading full
logit rows only when sampling needs them would remove most of it.

### 20. `markov_select`

`markov_select_kernel<1u>` takes 301 µs per call, 5.3 calls and 1.6 ms per
step, more than the drafter's attention (0.66 ms), at ~92% of the read
ceiling for its 66 MB FP16 W2. An exact norm bound would still score most
rows (E47).

### 21. Two-layer-ahead decode prediction with a usefulness gate

Streamed decode predicts one layer ahead and only after a layer waited on
reads ([design](STREAMING.md#prefetch-from-router-predictions); +0.8% at
82–83% hits). On Maya-S, mako found layer l+1's router on layer l's FFN
input catches 34% of misses at top 4 and layer l+2's 30%, but
one-ahead reads often had not finished (0.4–1.5 s of waits over 200 steps,
~0.1 s two ahead). Its default reads the top 4 non-resident experts two
layers ahead and stops predicting while under 35% of recent reads ahead were
routed (one layer in 16 keeps measuring): +3.5% on a story, −0.3% instead of
−2.7% after a 16K code prompt. Read time per expert is ~2–3× theirs here
(~4.9 GB/s against 12–16 GB/s), which favors the longer lead. Measure where
misses remain: cold cache, reduced cap, after long prompts, and with items
13 and 24.

### 22. Decode misses after a long prompt under scan-resistant admission

Once the cache is full, a large prefill admits only its final 16 rows and
its frequent experts. mako found the opposite trade-off on its decayed
counts: leaving prefill out of the eviction scores raised steady decode
misses by 20% (19–21 against 15–17 per step), because a prompt's experts
are the ones its decode uses. Measure route hits and tok/s over the first
64–256 decode steps after 4K and 16K prompts with the current rule against
admitting every prompt expert, and against a looser frequency threshold.

### 23. Streamed `IQ2_XS` with cache-aware routing

ISTA's `IQ2_XS` backbone (~155 GB) is closer to the reference than
`IQ2_XXS` in mako's measurements (top-1 92.5% / KL 0.068 against 91% /
0.103 on Maya-S logits; prose / code perplexity 1.873 / 2.903 against
1.935 / 3.025) and, with item 24 at 0.1, decoded 34.0 against 33.1 tok/s
there. Here the ~104 GiB cache would hold about three quarters of its
expert bytes, against 83% of `IQ2_XXS`'s experts (12,688 seeded), on a
slower SSD, so expect it slower; measure whether item 13 or 24 makes it a
usable higher-quality streamed option. It does not run streamed yet: item 27.

### 24. Selection-only miss penalty (quality-affecting, optional)

[mako](https://github.com/jhanssen/mako/blob/8567de3/src/kernels/moe_route.h)
(`--miss-penalty F`, Apple Silicon) subtracts a constant from the selection
score (sqrt(softplus) plus the router bias) of each expert the cache lacks.
The mixing weights stay the unpenalized scores, so the penalty only breaks
near-ties: a missing expert is replaced only by a resident one within F of
it. It applies to decode and to prefill groups under 512 tokens, such as
short chat turns (16–32 tokens after 8K of context missed ~30 experts per
token against ~10 in steady decode). On Maya-S, 1,300 positions against the
reference logits:

| F | Misses/step | Top-1 | KL | Step |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 13.4 | 98.2% | 0.0056 | 44.4 ms |
| 0.05 | 8.6 | 98.2% | 0.0067 | 40.2 ms |
| 0.1 | 6.6 | 98.0% | 0.0088 | 37.9 ms |
| 0.2 | 4.4 | 97.5% | 0.0100 | 35.4 ms |
| 0.5 | 2.1 | 96.5% | 0.0185 | 33.4 ms |

Prose and code perplexity stayed within noise up to 0.2, and on ISTA
`IQ2_XS` 0.1 halved the misses (9.5 → 4.8 per step, 31.4 → 34.0 tok/s).
With its read-ahead (item 21) the gains add up: 24.6 → 29.1 tok/s. Unlike
Xolotl's logit scaling (item 13), it leaves the mixing weights alone and
swaps only near-ties; test both on the same quality gates and speed runs.

At ~99.3% warm route hits (~1.7 misses per token) the gain is bounded by
the ~5 ms/token of miss-read idle; measure it cold, at reduced
`--expert-cache-bytes`, for the first steps after a long prompt, for short
continuation prefills and in DSpark verification, where every row pays for
its own experts (item 3). The penalty goes into the router's top-k with the
cache's residency as a device bitmap per layer. Outputs then depend on the
cache contents: no longer bit-identical across cache sizes, not
reproducible across sessions, and a replayed prefix may route differently
from its first pass. Opt-in only; gate both with teacher-forced KL and top-1
against the oracle plus perplexity at 3K and 18.5K tokens.

### 25. Direction projection and steering vectors

Remove (or add) per-layer directions from what a sublayer writes into the
hyper-connection streams, at run time, with the weights unchanged. mako's
`--ablate FILE` (`src/mtl/ablation.hpp`, format `MAKOABL1`: n_embd,
n_layer, n_site, k, 16-byte site names, then F32 vectors
`[layer][site][k][n_embd]`) applies `x -= scale (x · v) v` per direction to
each row of the attention output (`attn_out`) and of the shared expert's
output (`shexp_out`) before `hc_post`; the routed experts are added after
the projection and keep the direction. The published DeepSeek file
(`skafte/mako-ablations`, `refusal.abl`) holds two directions per layer at
both sites: 40 × 2 × 2 × 5,120 F32 = 3.28 MB.

Projection is linear, so projecting a sublayer's output equals projecting
each of its writers (attention, shared expert, every routed expert); the
site choice decides which writers keep the direction. Plan:

- One fused kernel per site (dot, then axpy over 5,120 floats per row, k
  directions), or better an epilogue of the existing attention-output and
  MoE-combine launches, so decode adds no launches (item 4: ~2–3 µs each,
  ~80 sites per token).
- Sites: attention output, shared-expert output, and the full FFN output
  (shared plus routed), so both mako's files and full-writer projections
  load.
- Additive steering (`x += α v`, llama.cpp's control vectors) on the same
  sites.
- `gufo serve` option for the file and a global scale; per-request scale
  only once the prompt cache keys on it (projected prefixes produce
  different KV and snapshots).
- Gates: outputs bit-identical with no file or zero vectors; teacher-forced
  KL and perplexity at 3K and 18.5K tokens with a file; DSpark acceptance
  (the drafter reads the target's hidden states).

### 26. LoRA adapters

`BoldingBuilds/DeepSeek-V4.1-Flash-1017-Abliteration-LoRA-GGUF` is a
standard llama.cpp LoRA (`general.type = adapter`, `adapter.lora.alpha =
1`) for the same purpose built another way: one direction r taken at
layer 22 and removed from every residual writer of every layer as
`W' = W − r (rᵀ W)`, written as a rank-1 pair per matrix with `lora_b = r`
and `lora_a = Wᵀ r`. Its 240 F32 tensors (459 MB):

| Tensor | Shapes per layer | Bytes |
| --- | --- | ---: |
| `ffn_down_exps` | a 2,304 × 1 × 384, b 1 × 5,120 × 384 | 456.1 MB |
| `attn_output_b` | a 8,192 × 1, b 1 × 5,120 | 2.1 MB |
| `ffn_down_shexp` | a 2,304 × 1, b 1 × 5,120 | 1.2 MB |

It is ~140× mako's file although it carries less: one direction instead of
160 per-layer vectors. The LoRA form must express the edit per weight
matrix, so it stores `Wᵀ r` for each of the 15,360 routed down projections
(a product of the weights the engine already has) and repeats the same r
as `lora_b` 15,440 times (~316 MB). `lora_a` was computed from Maya-S's
dequantized weights, so on another quant it only approximately removes r;
a runtime projection is exact for any quant. It reports 98.3% → 4.2%
harmful-prompt refusals and HumanEval 95.1% → 95.7% on Maya-S.

Plan:

- Load GGUF LoRA adapters for the dense and shared-expert matrices as
  low-rank epilogues of their GEMVs and GEMMs (`y += scale · B (A x)`).
- Routed experts: per-expert A and B indexed by the route, resident even
  while experts stream (~29 KB per expert at rank 1); grouped expert
  kernels gain a rank-r tail.
- Detect rank-1 adapters whose `lora_b` is one vector r on every writer of
  a layer and `lora_a ≈ Wᵀ r`, and load them as item 25's full-writer
  projection instead: exact on our quants, no 456 MB, no extra reads.
- Gates as item 25; adapters change the outputs, so the prompt cache must
  key on the adapter set and scales.

Both items apply beyond DeepSeek V4.1; place shared loaders and kernels so
other models can adopt them.

### 27. Streamed kernels for the `IQ2_XS` build

The ISTA `IQ2_XS` backbone (4 shards, ~145 GiB) only runs with
`--stream-experts`. Its [allocation](https://huggingface.co/ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF/blob/main/tensor-allocation/DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XS.rco-allocation.txt)
uses the dense types, IQ2_XXS and IQ2_XS experts and IQ2_S down projections
already exercised by the `IQ1_M` and `IQ2_XXS` builds, plus two expert
placements no streamed path handles (traced in `runtime/v41_decode.cpp`, not
run):

- **IQ3_XXS down projection (`blk.23.ffn_down_exps`).** The loader accepts
  it and the resident MMQ/MMVQ kernels have it, but the table-indexed expert
  GEMV does not (`expert_block_bytes` returns 0) and there is no FP16 tile,
  so single-token, verification and prefill rows of layer 23 all fail when
  streamed. Add IQ3_XXS traits (98-byte blocks, `mmq/ggml-common.h`) to
  `kernels/rocm/ds41_expert_gemv.hip.cpp`, at least the per-slot down
  variant.
- **IQ2_S gate/up (layers 17, 25, 30).** The expert GEMV kernels take IQ2_S
  (fused SwiGLU and grouped), so verification batches work, but
  `fused_pair` leaves IQ2_S out and the streamed single-token miss path
  then fails; adding it there should suffice. Streamed prefill fails at "no
  table-indexed kernel for this shape" because `routed_f16_type` excludes
  IQ2_S: add an IQ2_S FP16 tile or the per-row expert GEMV fallback the
  IQ2_S down projections use.

Check each against the resident MMQ/MMVQ outputs (`ds41.mmq_typed`,
`ds41_gemv_bench`) and keep the existing builds' outputs bit-identical.

## Memory

Resident weights take 112.27 GiB GTT. A session at the native 1M context
adds 0.9 GiB of FP4 state; the batch scratch (1.4 GiB at 1M, 256 MiB of it
indexer scores) is shared by all sessions, and a 2048-token prefill peaks
0.3 GiB higher. Further reductions need smaller prefill batches, which cost
speed (EXPERIMENTS.md).
