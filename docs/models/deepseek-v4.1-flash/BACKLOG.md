# DeepSeek V4.1 Flash backlog

Open work only. Everything already measured, retained or rejected, is in
[EXPERIMENTS.md](EXPERIMENTS.md); read the matching entries before retrying
an item. The streaming design is in [STREAMING.md](STREAMING.md). Impact
figures are estimates unless stated.

## Where the time goes

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
| 2 | Fused SwiGLU and Q8_0 pair launches through the split GEMV | Verification | sh_gate costs 103 µs at 8 tokens against 75 µs for one | Medium |
| 3 | Expert reads for rejected verification rows | Streamed DSpark | Part of the 36% verification-expert time | Low |
| 4 | Launch gaps between decode kernels | Decode | Up to ~2.7 ms/token | Medium |
| 5 | Compact selection and WMMA prefill rescoring | Long context | Prefill and verification beyond 16K | Medium |
| 6 | Routed FP16 kernel: weight decode per slot | Prefill | ~5–20% of routed time | Low |
| 7 | Remaining prefill glue | Prefill | ~30–60 ms per 2K batch | Medium |
| 9 | Stable-boundary prefill pass per chat request | Serving | ~65 ms per request | Medium |
| 10 | Concurrent drafter forwards | Serving decode | ~9 ms per drafting session per step; DSpark 5% under AR at C8 | Medium |
| 13 | Route bias toward cached experts (changes outputs) | Streamed decode | Opt-in approximation | Low |
| 14 | Higher-precision Engram tables | Quality | Unmeasured | — |
| 15 | Smaller items | Various | Small | — |
| 16 | Multi-user AR decode below the 2026-10-26 card | Serving decode | C8 51.8 against 78.5 tok/s | Medium |

### 2. Fused SwiGLU and Q8_0 pair launches through the split GEMV

The task-major split GEMV covers single Q5_K projections only. The fused
shared gate/up + SwiGLU launch (sh_gate 75 → 103 µs from one to eight
tokens) and the Q8_0 q_a/KV pair still use the lane layout whose
multi-token cost is activation loads. Port them to the split kernel with the
same xor-tree and slice order.

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
launches. Fusions and the device-side embedding lookup below are each
worth ~0.2–0.5%; only graph capture addresses the bulk.

~1,000 dependent launches per token leave ~2.7 ms idle (resident profile).
Left to fuse: the q rope after q_b (40 per token; would need a GEMV epilogue
pairing rows 8 lanes apart), route after the router matvec (40; 7 µs each),
the window copy (43; cannot run before attention, which reads the slot it
overwrites). hc mixes read 2 MB of F32 weights in ~14 µs (80 per token)
regardless of blocks per row. ~90 µs of host work starts each step
(embedding dequant and two synchronous uploads ~70 µs); a device-side
embedding lookup removes most of it. Graph capture would need the mixed-MMQ
layers' route plans on the device (E7 covers the FP16 layers) and the Engram
reads as capture boundaries.

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
feeds at most four 16-slot tiles, on average 2.3. 128-slot tiles for large
experts would cut decode work by 13–24% but need 128 accumulator VGPRs and
twice the activation LDS. IQ2_XXS's 16-byte random codebook reads still
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

### 10. Concurrent drafter forwards

Concurrent sessions now share one 16-row verification (E27), but each
drafting session still runs its own drafter forward, one after another
(~9 ms each, ~34 ms per step at C4). With different prompts per user DSpark
matches AR at C4 (47.6 against 46.0 tok/s) and trails it at C8 (51.5
against 54.6). One drafter forward over all drafting sessions, as Gemma's
`DraftChains` does, removes most of that time. E27's v3 (a batch plan that
skips drafting when it does not pay) is the fallback if batching is not
possible.

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

Today's build decodes C4 / C6 / C8 AR at 62.9 / 60.0 / 51.8 tok/s against
the card's 60.4 / 71.3 / 78.5 (2026-10-26, 4-shard layout, which can no
longer be loaded for a direct A/B). The joint evaluation itself is not
slower (`--joint-check 8 --joint-rows 8`: 129.5 ms on a 1,096-token prompt),
so the time is in the serving path around it. The card's multi-user tables
also give every user the same prompt, which keeps sessions in lockstep and
lets a joint pass share experts (E27); refresh them with distinct prompts.

## Memory

Resident weights take 112.27 GiB GTT. A session at the native 1M context
adds 0.9 GiB of FP4 state; the batch scratch (1.4 GiB at 1M, 256 MiB of it
indexer scores) is shared by all sessions, and a 2048-token prefill peaks
0.3 GiB higher. Further reductions need smaller prefill batches, which cost
speed (EXPERIMENTS.md).
