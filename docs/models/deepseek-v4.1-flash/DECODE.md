# DeepSeek V4.1 Flash decode: where the time goes

A starting point for decode work on the streamed `UD-IQ2_XXS` build
(146 GiB of routed experts, ~95-100 GiB expert cache, tails sidecar on a
second NVMe): AR, sampled and DSpark decode through `gufo serve`, with the
measured budgets, each kernel's position against the gfx1151 read ceiling,
how a step runs, and the levers by measured size. Open work is ranked in
[BACKLOG.md](BACKLOG.md). Prefill is in [PREFILL.md](PREFILL.md), the
expert store in [STREAMING.md](STREAMING.md), every `E<n>` in
[EXPERIMENTS.md](EXPERIMENTS.md).

The profile was measured 2026-11-02 on `feat/ds41-ud-iq2xxs-prefill` at
`c9faa470` ([artifacts](artifacts/2026-11-02-decode/)); E87-E92 used the
same workloads afterwards and are folded into the levers. The 2026-10-29
all-mode traces of the resident `IQ1_M` and streamed `IQ2_XXS` builds are
in the [trace sweep](artifacts/2026-10-29-trace-sweep/summary.txt).

## Summary

One fresh server per mode, three requests in order (a 41-token chat
prompt, a 3,477-token code prompt, a 23,974-token document), 512 tokens
each. Unprofiled `decode_tps` at the profiled build, and after E87 (greedy
width policy) and E88 (top-p totals on the device):

| Mode | Chat | Code | After the 24K document | After E87/E88 (chat / code / long) |
|---|---:|---:|---:|---|
| AR greedy | 16.5 | 18.7 | 17.2 | unchanged |
| AR sampled (T 0.7, top-p 0.95) | 16.9 | 18.3 | 17.4 | 17.0-17.1 / 18.4-18.5 / 17.6 (E88) |
| DSpark greedy | 16.9 | 27.3 | 25.8 | 17.6-18.9 / 28.3-28.7 / 26.4-26.7 (E87) |
| DSpark sampled (T 0.7, top-p 0.95) | 19.0 | 24.8 | 26.9 | 19.3 / 25.7 / 27.9-28.0 (E88) |

AR means no drafter; the server still verifies prompt-lookup copies in that
mode, a few times per request. A decoded token reads ~10 GB (6.3 GB dense
projections, 2.5 GB routed experts, 1.2 GB head, Engram, router, mix
weights and attention state): ~42 ms at the 240 GB/s ceiling, a **~24
tok/s AR ceiling**. Steady AR runs 53-58 ms per token. The gap, largest
first: expert read stalls (lever 2), kernel bandwidth (lever 5), launch
gaps (lever 6). DSpark adds the verification row cost (lever 3) and the
drafter pass (lever 7); levers 1 and 4 (greedy width policy, top-p host
sampling) are fixed.

## Workloads and how they were measured

| Item | Value |
|---|---|
| Server | `gufo serve --sessions 1 llm --model UD-IQ2_XXS --context 65536 --stream-experts [--speculative dspark --dspark-model Q2K-Q4K]` |
| Client | [client.py](artifacts/2026-11-02-decode/client.py): chat (story, 41 prompt tokens), code (rewrite a function of `tools/ds41/replay_sim.py`, 3,477), long (explain the HTTP server documentation, 23,974); `max_tokens` 512, `reasoning_effort: none`; sampled requests T 0.7, top-p 0.95, seed 1234 |
| Runs | [run.sh](artifacts/2026-11-02-decode/run.sh): a fresh server per mode, once under `rocprofv3 --kernel-trace --memory-copy-trace` and once without |
| Caches | AR 99.55 GiB, DSpark ~95 GiB (the drafter takes 4.6 GiB); seeded from the built-in ranking |

Profiling costs ~2% (profiled AR greedy 16.5/18.2/16.9 against
16.5/18.7/17.2 tok/s), and the tracer adds ~4.5 µs to a launch after a long
kernel (E60). It also shifted the pre-E87 greedy DSpark policy (7 rounds
with drafts on the profiled chat, 52 unprofiled). Profile rates are single
runs: compare against an alternating control; greedy DSpark chat varies by
itself (17.6-19.2 tok/s, E87).

**Analysis scripts** (artifacts directory; Python and sqlite3 only):

| Script | What it does |
|---|---|
| `steps.py` | Splits a trace into requests at idle gaps over 2.5 s, and into decode steps at each logits download (a copy of k × 517,120 bytes, one vocabulary row per verified row). |
| `budget.py` | Per request: kernel time by class per step, and every idle gap in the step attributed by what follows it. |
| `roles.py` | Per-role kernel duration and bandwidth from the known weight shapes (AR). |
| `rowcost.py` | Cycle cost by verified rows, split into the pre-verification part (drafter) and verification. |
| `hostgap.py` | The host gap after each logits download. |
| `drafter.py` | The drafter pass by kernel. |

**Reading idle gaps:** before an upload of ≥1 MiB or an `expert_*`
kernel, a demand read; before a 24,576-byte upload, the Engram rows; right
after the logits download, host work (sampling, verification, commit,
setup); in DSpark cycles, up to the drafter's token download, drafter time.
Route traces (`DS41_EXPERT_TRACE`, prediction off) of these workloads feed
`tools/ds41/prefetch_sim.py` and `tools/ds41/evict_sim.py` (E89, E92).

## Bytes per decoded token (UD-IQ2_XXS)

From the GGUF headers ([gguf_bytes.py](artifacts/2026-11-02-decode/gguf_bytes.py),
[output](artifacts/2026-11-02-decode/gguf-bytes.txt)):

| Part | Bytes/token | At 240 GB/s |
|---|---:|---:|
| `attn_q_b`, Q8_0 (27 layers) / Q6_K (13) | 1.65 GB | 6.9 ms |
| `attn_output_b`, 5,120 × 8,192 | 1.65 GB | 6.9 ms |
| `attn_output_a`, 8 groups 1,024 × 4,096 | 1.32 GB | 5.5 ms |
| Shared expert gate+up+down, Q8_0 / Q5_K | 1.33 GB | 5.5 ms |
| `attn_q_a` + `attn_kv` | 0.36 GB | 1.5 ms |
| Routed experts, 6 × 40 (IQ1_S…IQ3_XXS) | 2.45 GB | 10.2 ms |
| Output head, Q6_K 129,280 × 5,120 | 0.54 GB | 2.3 ms |
| Engram projection, 2 × Q6_K 25,600 × 6,144 | 0.26 GB | 1.1 ms |
| BF16 router, 40 × 384 × 5,120 (+ prediction calls) | 0.16 GB | 0.7 ms |
| Hyper-connection mix weights, FP32, 80 × 2 MB | 0.16 GB | 0.7 ms |
| Indexer, compressor, attention state | ~0.1 GB | ~0.5 ms |
| **Total** | **~10 GB** | **~42 ms (24 tok/s)** |

The dense part is identical for every row of a verification; routed experts
are not. Each row adds its routes' distinct experts, about 2.4 MiB × new
experts per layer, which is where verification's per-row cost comes from.

## How a decode step runs

Every DeepSeek V4.1 decode goes through the speculative step (`DecodeStep`
→ `ds41_sessions_dspark_step_batch`), with or without a drafter
(`kPromptLookup` is on). One cycle (`runtime/v41_session.cpp`):

1. **prepare**: draws the anchor if needed, looks for a prompt-lookup copy,
   otherwise sets the draft width. Greedy with the confidence head
   (`c.learned`): full width while the learned saving of a draft
   (`gain_ms`) exceeds the drafter pass (`draft_ms`), else none, with
   probes (E87). Sampled: full width. Neither: `draft_width`. Drafted widths
   are cut after drafting.
2. **draft** (`ds41_dspark_draft_batch`): embeds and uploads the anchor and
   noise rows, runs three drafter stages, the target's output head over the
   drafted rows, five sequential Markov selections (66 MB FP16 W2 each) and
   the confidence head, then downloads the tokens, its only host round trip.
3. **evaluate**: `ds41_decoder_eval` (one row),
   `ds41_decoder_eval_speculative` (several rows, single-token arithmetic
   per row) or `ds41_decoder_eval_joint` (several sessions).
4. **finish**: verifies each row (device argmax for greedy; device sampler,
   or host sampler with device top-p totals, for sampled), commits the
   accepted prefix as queued device copies, injects the drafter's features,
   updates the width policy.

Plain AR steps take the same path with one row, then draw on the host
(greedy: host argmax over the logits row; sampled: `device_draw` for plain
temperature, else the host sampler).

**One evaluation** (`eval_rows`, `runtime/v41_decode.cpp`):

1. **Host setup**: hashes the Engram rows and starts their reads
   (`std::async` plus `engram_read` threads); dequantizes and uploads the
   embedding row.
2. **Per layer** (40), on the null stream: hc mix on a side stream (E57);
   collapse and norm; `attn_q_a`+`attn_kv` paired GEMV, q/kv norm,
   `attn_q_b`, rope; on compressed layers the compressor and indexer;
   split attention and combine (E65); `attn_output_a`, `attn_output_b`, hc
   expand; hc mix, collapse and norm for the FFN; BF16 router, route; routes
   event and the shared expert queued; then `ds41_expert_store_prepare`
   (the host copies the 6 routes on `routes_stream` and waits, counts hits
   and misses, resolves the prefetch job, uploads changed table rows and
   hit/miss arrays, and runs the read phase, whose overlap callback queues
   the hit experts' fused gate+up+SwiGLU and, after a miss, the next
   layer's prediction, then waits for the reads); routed gate+up over the
   miss rows, routed down over all six slots; hc expand-sum.
3. **Close**: final collapse and norm, output head (2.3 ms), device argmax
   rows for verification, logits download (517 KB per row),
   `hipDeviceSynchronize`.

The per-layer host round trip is hidden while all routes hit: the route
copy and planning overlap the shared expert (113-177 µs), and the routed
gate+up starts 2-3 µs after the shared down in every traced layer.

## Measured budgets

All from the profiled build (before E87/E88).

### AR greedy, per token (profiled, ms)

[budget-ar-greedy.txt](artifacts/2026-11-02-decode/budget-ar-greedy.txt)

| | Chat | Code | Long |
|---|---:|---:|---:|
| Step wall, mean / median | 60.8 / 57.4 | 61.4 / 57.3 | 60.3 / 57.5 |
| Dense GEMV | 29.3 | 29.4 | 29.6 |
| Routed experts | 11.7 | 13.1 | 12.4 |
| Hyper-connections (mix on its side stream) | 3.5 | 3.6 | 3.5 |
| Output head | 2.3 | 2.3 | 2.3 |
| Attention, indexer, misc | 1.4 | 2.2 | 2.7 |
| Router and BF16 | 1.7 | 1.7 | 1.7 |
| Engram projection and gate | 1.1 | 1.1 | 1.1 |
| Idle: expert reads | 6.6 | 5.1 | 4.3 |
| Idle: launch gaps under 20 µs | 2.9 | 2.9 | 2.9 |
| Idle: host after the logits download | 0.4 | 0.4 | 0.4 |
| Idle: Engram rows | 0.1 | 0.1 | 0.1 |

The kernel rows overlap: the mix and the upload copies run beside the main
stream. GPU busy, as the union of all streams, is 49.6-52.8 ms. The
code and long requests include a few 16-row lookup verifications.

### AR sampled

As greedy, plus 1.29-1.51 ms of host time after each logits download
against 0.43-0.49 ms greedy
([hostgap.txt](artifacts/2026-11-02-decode/hostgap.txt)): lever 4.

### DSpark, per cycle by verified rows (profiled)

[rowcost.txt](artifacts/2026-11-02-decode/rowcost.txt). "Pre-verify" covers
the host work after the previous download plus the drafter pass. Verify is
the target evaluation. The greedy rows show the pre-E87 policy, which
picked only 0 or 4-5 drafts.

| Greedy | Cycles | Cycle | Pre-verify | Verify | GPU busy in verify | Routed experts |
|---:|---:|---:|---:|---:|---:|---:|
| 1 row | 830 | 72.0 ms | 9.3 | 62.6 | 51.0 | 12.1 |
| 5 rows | 124 | 131.8 | 9.3 | 122.4 | 99.5 | 47.7 |
| 6 rows | 22 | 155.0 | 9.4 | 145.6 | 111.3 | 56.7 |
| 16 rows (lookup) | 7 | 300.0 | 0 | 300.0 | 226.0 | 130.5 |

| Sampled | Cycles | Cycle | Pre-verify | Verify | GPU busy in verify | Routed experts |
|---:|---:|---:|---:|---:|---:|---:|
| 1 row | 76 | 79.7 | 11.5 | 68.1 | 53.6 | 12.5 |
| 2 rows | 106 | 86.0 | 12.0 | 74.0 | 60.8 | 20.2 |
| 3 rows | 95 | 107.1 | 12.5 | 94.5 | 75.7 | 28.3 |
| 4 rows | 76 | 114.2 | 11.9 | 102.4 | 81.9 | 35.3 |
| 5 rows | 64 | 136.7 | 12.2 | 124.5 | 96.9 | 46.4 |
| 6 rows | 174 | 147.0 | 12.9 | 134.1 | 107.6 | 54.0 |

Each row past the first adds ~9 ms of routed expert GEMVs, ~3 ms of other
GPU work (attention, hc, router) and ~4.5 ms of read stall (verify idle
11.6 ms at one row, 34 ms at six). Before E88, sampled cycles also paid
~0.85 ms of host verification per row (4.9-5.0 ms after a 6-row download,
0.24-0.71 ms after a 1-row one, [hostgap.txt](artifacts/2026-11-02-decode/hostgap.txt)).

**The drafter pass** ([drafter.txt](artifacts/2026-11-02-decode/drafter.txt)),
greedy code request, 250 drafting cycles, 9.18 ms from the first kernel to
the token download:

| Part | Per cycle |
|---|---:|
| Target output head over the drafted rows (Q6_K) | 2.54 ms |
| `markov_select` × 5, sequential (66 MB FP16 W2 each) | 1.48 ms |
| Stage GEMVs, Q4_K (q, kv, output, shared) | ~1.9 ms |
| Routed drafter experts, Q2_K (`expert_group_kernel<10>`) | ~1.5 ms |
| Drafter attention | 0.61 ms |
| hc, norms, confidence head, gathers | ~0.4 ms |

## Per-kernel position (AR, single token)

[roles-ar-greedy.txt](artifacts/2026-11-02-decode/roles-ar-greedy.txt),
chat request, 511 steps. Medians per call; ms/token includes the extra
calls of split hit/miss launches.

| Kernel | Bytes | Median | Rate | ms/token | At 240 GB/s |
|---|---:|---:|---:|---:|---:|
| `attn_output_b` Q8_0 (27) | 44.6 MB | 208.9 µs | 213 GB/s | 5.68 | 5.01 |
| `attn_q_b` Q8_0 (27) | 44.6 MB | 202.5 µs | 220 | 5.50 | 5.01 |
| `attn_output_a` Q8_0 (27) | 35.7 MB | 163.6 µs | 218 | 4.46 | 4.01 |
| Routed gate+up IQ2_S (15 layers + splits) | 45.3 MB | 207.8 µs | 218 | 3.19 | 3.29 |
| Shared gate+up Q8_0 (27) | 25.1 MB | 114.9 µs | 218 | 3.12 | 2.82 |
| Routed gate+up IQ2_XXS (15) | 36.5 MB | 171.0 µs | 213 | 2.61 | 2.52 |
| Output head Q6_K | 543 MB | 2,334 µs | 233 | 2.34 | 2.26 |
| hc mix, FP32 weights (80, side stream) | 2.0 MB | 28.6 µs | 69 | 2.28 | 0.66 |
| `attn_output_b` Q6_K (13) | 34.4 MB | 164.7 µs | 209 | 2.17 | 1.86 |
| `attn_q_b` Q6_K (13) | 34.4 MB | 148.2 µs | 232 | 1.96 | 1.86 |
| Routed down IQ3_XXS (15) | 27.1 MB | 123.4 µs | 220 | 1.91 | 1.69 |
| Routed down IQ2_S (15) | 22.7 MB | 108.4 µs | 209 | 1.67 | 1.42 |
| Shared down Q8_0 (27) | 12.5 MB | 60.0 µs | 209 | 1.63 | 1.41 |
| `attn_output_a` Q6_K (13) | 27.5 MB | 119.0 µs | 231 | 1.57 | 1.49 |
| Routed gate+up IQ1_S (10) | 27.7 MB | 133.0 µs | 208 | 1.35 | 1.28 |
| `attn_q_a`+`attn_kv` Q8_0 (27) | 9.8 MB | 46.2 µs | 211 | 1.26 | 1.10 |
| Engram projection Q6_K (2) | 129 MB | 546.8 µs | 236 | 1.11 | 1.08 |
| Shared gate+up Q5_K (13) | 16.2 MB | 74.0 µs | 219 | 0.98 | 0.88 |
| BF16 router (40 + predictions) | 3.9 MB | 18.6 µs | 211 | 0.84 | 0.74 |
| Routed down IQ2_XS (6) | 20.5 MB | 100.0 µs | 205 | 0.61 | 0.51 |
| Shared down Q5_K (13) | 8.1 MB | 39.0 µs | 208 | 0.51 | 0.44 |
| `attn_q_a`+`attn_kv` Q6_K (13) | 7.5 MB | 38.2 µs | 197 | 0.51 | 0.41 |
| Routed down IQ2_XXS (3), Q2_K (1) | 18-23 MB | 89-108 µs | 205-214 | 0.38 | 0.33 |
| **Total** | | | | **47.6** | **42.1** |

Without the hc mix, which runs beside the projections and costs only the
bandwidth it takes from them, the critical-path kernels are ~3.9 ms/token
(~9%) above the read ceiling: 45.3 against 41.4 ms.
- The Q8_0 `attn_output_b` (−0.67 ms) and `attn_q_b` (−0.49 ms) are the
  largest single gaps.
- The Q8_0 down-shaped GEMVs (shared down, `attn_output_b`) sit at
  209-213 GB/s against 218-220 for the up-shaped ones.
- The Q6_K `attn_q_b` and `attn_output_a` reach 231-232 in the model, but
  not in a DRAM-cold bench (lever 5).

## Levers, by measured size

### 1. DSpark greedy width policy (bug; fixed in E87)

The greedy policy ran the drafter (~9.3 ms) every cycle before picking a
width, and never learned the plain-step cost: it kept its 42.4 ms resident
prior while a streamed one-row cycle cost 72 ms with the discarded draft,
so it drafted only above ~3.1 expected tokens at width 5 instead of ~1.8
(131.8 / 72.0). On the profiled chat 492 of 499 cycles drafted and
verified one row. E87 learns the plain step and the draft cost and drafts
only while the measured saving exceeds it (rates in the Summary). Chat
still varies, because there a draft saves about what it costs.

### 2. Expert read stalls (4.2-6.5 ms/token AR, 8-13 ms per DSpark cycle)

Route hits per 64-token interval
([server-logs.txt](artifacts/2026-11-02-decode/server-logs.txt)):

| Phase | Hits | Read/token | Read wait/token |
|---|---:|---:|---:|
| Fresh server, first 64 tokens of the first reply | 74-78% | 266-304 MiB | 23.6-24.5 ms |
| Same reply, tokens 128-512 | 97.5-98.8% | 29-68 MiB | 3.3-6.6 ms |
| First 64 tokens after the 3.5K / 24K-token prompts | 95.6-97.2% | 82-128 MiB | 7.5-10 ms |
| Steady | 98.4-99.2% | 21-50 MiB | 2.3-5 ms |

At 98.5% hits a token still misses ~3.6 experts (~36 MiB), 1-2 per missing
layer, each ~10 MiB or ~1 ms on two drives, only partly hidden behind the
hit experts' gate+up. Verification multiplies the exposure (lever 3) and
the drafter's 4.6 GiB come out of the cache. Sampled DSpark's per-token
host wait is route wait 18.3-18.5 ms, read wait 4.8-5.5 ms and 0.9-1.0 ms
of everything else, admission included (E92).

- **Prediction** runs only after a missed layer
  ([STREAMING.md](STREAMING.md#prefetch-from-router-predictions));
  prefetched shares are 12-50%. Every-layer prediction is +2.3-2.8% in
  `prefetch_sim` but measured slower from per-prepare job resolution
  (~0.1 ms per layer); two layers ahead predicts fewer misses (E89).
  [BACKLOG item 37](BACKLOG.md#37-cheap-prefetch-adoption-then-every-layer-decode-prediction).
- **Eviction**: victim lists replaced the scan (E91, E92; the scan cost
  ≤ ~0.15 ms per admission). Belady's optimum would remove about half of
  the decode misses (−4.87 s of 10.24 s modeled stall), past-use rules
  gain 3.6-7.5% in replay (E92); the gap needs reading ahead (item 37).
  [BACKLOG item 31](BACKLOG.md#31-eviction-scoring-and-prompt-admission).
- **Capacity** ([item 30](BACKLOG.md#30-more-memory-for-the-expert-cache)),
  **after long prompts** ([item 22](BACKLOG.md#22-decode-misses-after-a-long-prompt)),
  **fresh-server seed** ([item 39](BACKLOG.md#39-expert-cache-seed-from-served-text)).

### 3. Verification row cost on the streamed build

Each row past the first costs ~12 ms of GPU time plus ~4.5 ms of read
stall; a 6-row cycle takes 155 ms with its draft against 59.5 ms for an AR
token, so it must emit more than 2.6 tokens to win. Resident `IQ1_M` also
pays ~12 ms per row (E18). The routed part is genuinely distinct reads: a
6-row layer routes 36 rows over 22.6 distinct experts, the dedup minimum,
at 205-220 GB/s (E89). Open: per-row miss pricing
([BACKLOG item 3](BACKLOG.md#3-expert-reads-for-rejected-verification-rows));
output-changing cache-aware routing is opt-in (items 13, 24).

### 4. Host sampling (0.85 ms per verified row; top-p fixed in E88)

The device sampler (`DsparkSamplerBridge::DeviceTemperature`) takes only
plain temperature; other requests build the full 129,280-entry
distribution on the host (`sampler.Sample`, `working_.Distribution`) while
the GPU idles: ~0.85 ms per AR token (1.5%) and per verified row, 2.5-5 ms
per 4-6-row cycle. E88 computes the top-p softmax totals on the device
(min-p allowed; no top-k, penalties or constraints), seeded outputs
identical. Top-k remains:
[BACKLOG item 15](BACKLOG.md#15-smaller-items).

### 5. Kernel bandwidth (~3.9 ms/token on the critical path, ~7%)

The Q8_0 GEMVs are the largest block (27 layers × `q_b`, `output_a`,
`output_b`, shared at 208-220 GB/s). It is not a format gap: DRAM-cold,
every format runs 199-224 GB/s at one token, so the served Q6_K 231-232 is
not reproducible; better slice choices are worth 0.1-0.4%, under the
rounding bar (E90). Partial writebacks after split attention cost ~0.6 ms
per token (E65, [item 33](BACKLOG.md#33-attention-partials-written-back-during-the-dense-gemvs));
routed GEMVs are [item 17](BACKLOG.md#17-single-row-routed-expert-gemv-bandwidth).
Retained already: E46, E51, E69; nontemporal loads did not help (E59, E65).

### 6. Launch gaps (~2.9 ms/token traced)

~975 launches per token, ~25 per layer; traced gaps average ~3 µs, ~6.5-8
µs after a long GEMV, of which ~4.5 µs is the tracer (E60). Graph replay
keeps them (E44); only fewer launches help. Estimate the untraced share
from event timing first:
[BACKLOG item 4](BACKLOG.md#4-launch-gaps-between-decode-kernels).

### 7. Drafter pass (9.2 ms per drafting cycle, ~7% of a 5-row cycle)

The target head over the drafted rows (2.5 ms, 213 GB/s) and five
dependent `markov_select` passes (1.5 ms at ~92% of the read ceiling, E47)
would shrink with a request-vocabulary head (E64: +1.1% greedy in replay);
the drafter stages (~4.5 ms) have no bandwidth table yet:
[BACKLOG item 38](BACKLOG.md#38-drafter-pass).

### 8. Host work per step (0.3-0.45 ms greedy)

Host argmax or verification, the Engram hash, per-evaluation
`std::async`/`engram_read` threads and the embedding dequant and uploads
sit between the logits download and the first kernel; under 1% of a token
([BACKLOG item 19](BACKLOG.md#19-host-work-and-round-trips-per-step)).

## Things that look like levers but are not

- **The per-layer route round trip**: hidden by the shared expert while
  routes hit.
- **The hc mix's 69 GB/s**: on a side stream (E57); fusing it with the
  collapse was slower (E49).
- **Verification grouped-GEMV redundancy** (E89) and **a Q8_0-specific
  GEMV gap** (E90): neither exists.
- **GPU clocks during read waits** (BACKLOG item 32, closed), **graph
  capture** (E44, ~0.13 µs per launch), **Engram reads in decode**
  (~0.1 ms/token).

## Reproducing

```sh
# Release build
cmake --build --preset release --parallel 4

# One mode, profiled (1) or not (0); D, CODE and DOC as in run.sh
D=/models/ds41 CODE=tools/ds41/replay_sim.py DOC=docs/SERVER.md OUT=prof \
  docs/models/deepseek-v4.1-flash/artifacts/2026-11-02-decode/run.sh ar-greedy ar 0 1
#   modes: "ar 0", "ar 1", "dspark 0", "dspark 1"

# Analysis (rocpd sqlite from rocprofv3)
A=docs/models/deepseek-v4.1-flash/artifacts/2026-11-02-decode
python3 $A/budget.py prof/ar-greedy        # per-token classes and idle
python3 $A/roles.py prof/ar-greedy -3      # per-kernel bandwidth, request -3/-2/-1
python3 $A/rowcost.py prof/ds-greedy       # DSpark cycle cost by rows
python3 $A/hostgap.py prof/ar-sampled      # host time after each logits download
python3 $A/drafter.py prof/ds-greedy -2    # drafter pass by kernel

# DSpark per-cycle log (multi-row and learned cycles)
GUFO_DEEPSEEK41_DSPARK_TRACE=1 build/release/gufo serve ...
```

Correctness for any decode change: greedy speculative output equals AR
output on the same prompt (the card's rule). Also run the 3K decode scores
(`ds41_load_probe --stream-experts --decode IDS --score`) and the
`tests/functional` DeepSeek suites for the affected modes. For sampled
changes, seeded outputs must not change, unless the change is to sampling
itself.
