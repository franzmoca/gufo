# DeepSeek V4.1 Flash decode: where the time goes

A starting point for decode work on the streamed `UD-IQ2_XXS` build
(146 GiB of routed experts, ~95-100 GiB expert cache, tails sidecar on a
second NVMe). It covers autoregressive (AR), sampled and DSpark decode
through `gufo serve`:
- the measured per-token and per-cycle budgets;
- each kernel's position against the gfx1151 read ceiling;
- how a decode step runs, end to end;
- what has already been tried;
- the open levers, ranked by measured size.

Measured 2026-11-02 on `feat/ds41-ud-iq2xxs-prefill` at `c9faa470`.
Scripts and outputs are in [artifacts/2026-11-02-decode](artifacts/2026-11-02-decode/).
Read [STREAMING.md](STREAMING.md) for the expert store, [PREFILL.md](PREFILL.md)
for prompt processing, and [EXPERIMENTS.md](EXPERIMENTS.md) for every
experiment cited as `E<n>`.

## Summary

Decode rates, one fresh server per mode, three requests in order:
- a 41-token chat prompt;
- a 3,477-token code prompt;
- a 23,974-token document.

Each request generates 512 tokens. Unprofiled `decode_tps` from the server log:

| Mode | Chat (first request) | Code | After the 24K document |
|---|---:|---:|---:|
| AR greedy | 16.5 | 18.7 | 17.2 |
| AR sampled (T 0.7, top-p 0.95) | 16.9 | 18.3 | 17.4 |
| DSpark greedy | 16.9 | 27.3 | 25.8 |
| DSpark sampled (T 0.7, top-p 0.95) | 19.0 | 24.8 | 26.9 |

AR means no drafter. The server still verifies prompt-lookup copies in that
mode, but only a few times per request.

A decoded token reads about 10 GB of weights:
- 6.3 GB of dense projections, every token;
- 2.5 GB of routed experts (6 per layer × 40 layers);
- 1.2 GB of output head, Engram projection, router, mix weights and
  attention state.

At the measured 240 GB/s read ceiling that is ~42 ms, a **~24 tok/s AR
ceiling** for this build. Steady AR runs at 53-58 ms per token. The gap of
11-16 ms per token, largest first:

1. **Expert read stalls (4.2-6.5 ms/token AR, 8-13 ms per DSpark cycle;
   7-10%).** Steady decode hits 98-99% of routes, which still leaves 30-50
   MiB of reads per token on the critical path. The first tokens of a fresh
   server hit 74-78%, and the first 64 after a 24K-token prompt hit 96-97%.
2. **Kernel bandwidth (~3.9 ms/token on the critical path, ~7%).** The
   weight-streaming kernels run at 197-236 GB/s. Their bytes would take
   42.1 ms at 240 GB/s; the profile shows 47.6 ms, including the hc mix,
   which runs on a side stream.
3. **Launch gaps (~2.9 ms/token traced, ~975 launches).** These are partly
   added by the tracer (E60).

DSpark has three problems of its own, all found while writing this document:

- **The greedy width policy drafts every cycle, then mostly verifies only
  the anchor (bug, lever 1).** The plain-step cost it compares against is
  never learned in this mode. It stays at its 42.4 ms prior, from resident
  `IQ1_M`, while a streamed plain cycle costs ~72 ms including the wasted
  draft.
  - In the chat profile, 492 of 499 cycles ran the 9.3 ms drafter and then
    verified one row.
  - Greedy DSpark ties AR on chat (16.9 against 16.5 tok/s) and trails
    sampled DSpark (19.0), which uses a different width rule.
- **Sampled requests with top-p, top-k or min-p sample on the host (lever 4).**
  The device sampler takes only plain temperature sampling. Every verified
  row then waits ~0.85 ms on the host for a full-vocabulary distribution:
  2-5 ms per DSpark cycle, ~0.9 ms per AR token.
- **Each verification row costs ~12 ms of GPU time plus more read stall
  (lever 3).** A 6-row cycle takes 155 ms including its draft, against
  59.5 ms for an AR token, so it must emit more than 2.6 tokens to beat
  AR.

## Workloads and how they were measured

| Item | Value |
|---|---|
| Server | `gufo serve --sessions 1 llm --model UD-IQ2_XXS --context 65536 --stream-experts [--speculative dspark --dspark-model Q2K-Q4K]` |
| Client | [client.py](artifacts/2026-11-02-decode/client.py): chat (story, 41 prompt tokens), code (rewrite a function of `tools/ds41/replay_sim.py`, 3,477), long (explain the HTTP server documentation, 23,974); `max_tokens` 512, `reasoning_effort: none`; sampled requests T 0.7, top-p 0.95, seed 1234 |
| Runs | [run.sh](artifacts/2026-11-02-decode/run.sh): a fresh server per mode, once under `rocprofv3 --kernel-trace --memory-copy-trace` and once without |
| Caches | AR 99.55 GiB, DSpark ~95 GiB (the drafter takes 4.6 GiB); seeded from the built-in ranking |

Profiling costs ~2% here: profiled AR greedy decodes at 16.5/18.2/16.9
tok/s against 16.5/18.7/17.2 unprofiled. The tracer adds ~4.5 µs to a
launch after a long kernel (E60), so traced launch gaps overstate the real
ones. It also shifts the greedy DSpark policy, which reads cycle times: 7
rounds with drafts on the profiled chat, 52 unprofiled. Every rate in this
document is a single run. Compare changes against an alternating control,
because fresh servers differ by a few percent.

**Analysis scripts** (artifacts directory; Python and sqlite3 only):

| Script | What it does |
|---|---|
| `steps.py` | Splits a trace into requests at idle gaps over 2.5 s, and into decode steps at each logits download (a copy of k × 517,120 bytes, one vocabulary row per verified row). |
| `budget.py` | Per request: kernel time by class per step, and every idle gap in the step attributed by what follows it. |
| `roles.py` | Per-role kernel duration and bandwidth from the known weight shapes (AR). |
| `rowcost.py` | Cycle cost by verified rows, split into the pre-verification part (drafter) and verification. |
| `hostgap.py` | The host gap after each logits download. |
| `drafter.py` | The drafter pass by kernel. |

**Reading idle gaps:**
- A gap before an upload of ≥1 MiB, or before an `expert_*` kernel, is a
  demand read. Reads arrive as pinned-buffer uploads before the miss launch.
- A gap before a 24,576-byte upload is the Engram rows.
- A gap right after the logits download is host work: sampling,
  verification, commit and the next step's setup.
- In DSpark cycles, gaps up to the drafter's token download count as
  drafter time.

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

The server drives every DeepSeek V4.1 decode through the speculative step
(`DecodeStep` → `ds41_sessions_dspark_step_batch`), with or without a
drafter: `kPromptLookup` is on for this model. One cycle
(`runtime/v41_session.cpp`):

1. **prepare**
   - Draws the anchor if the request has none.
   - Looks for a prompt-lookup copy.
   - Otherwise sets the draft width:
     - greedy with the drafter's confidence head (`c.learned`): always the
       full width, cut after drafting;
     - sampled: the full width, cut after drafting to the confident prefix;
     - neither: `draft_width`, which can skip the drafter.
2. **draft** (`ds41_dspark_draft_batch`)
   - Embeds the anchor and the noise rows on the host and uploads them.
   - Runs three drafter stages.
   - Runs the target's output head over the drafted rows (Q6_K, 2.5 ms for
     5 rows).
   - Runs five sequential Markov selections, each reading the 66 MB FP16 W2,
     plus the confidence head.
   - Downloads the tokens. This is the drafter's only host round trip.
3. **evaluate**
   - One row: `ds41_decoder_eval`.
   - Several rows: `ds41_decoder_eval_speculative`, which keeps single-token
     arithmetic per row.
   - Several sessions: `ds41_decoder_eval_joint`.
4. **finish**
   - Verifies each row: the device argmax for greedy; for sampled, the
     device sampler or the host sampler.
   - Commits the accepted prefix: queued device copies of the window and
     compressor rows.
   - Injects the drafter's features.
   - Updates the width policy.

**Plain AR steps** take the same path with `m = 1`: `eval` with one row,
then `draw` on the host. Greedy is the host argmax over the downloaded
logits row. Sampled uses `device_draw` when the request is plain
temperature sampling, otherwise the request's host sampler.

**One evaluation** (`eval_rows`, `runtime/v41_decode.cpp`), for one token
or a verification block:

1. **Host setup**
   - Hashes the Engram rows, then starts their reads: `std::async`, plus
     `engram_read` threads.
   - Dequantizes the embedding row on the host and uploads it.
2. **Per layer** (40), on the null stream:
   - hc mix: the side stream, overlapped with the next projections (E57).
   - Collapse and norm.
   - `attn_q_a`+`attn_kv` paired GEMV, q/kv norm, `attn_q_b`, rope.
   - On compressed layers: the compressor and indexer (BF16 matmuls, scores,
     top-k).
   - Attention: split and combine (E65).
   - `attn_output_a`, `attn_output_b`, hc expand.
   - hc mix, collapse and norm for the FFN.
   - BF16 router, route.
   - **Routes event; shared expert queued** (gate+up paired, down).
   - **`ds41_expert_store_prepare`**:
     - the host copies the 6 routes on `routes_stream` and waits;
     - counts hits and misses, scores, resolves the prefetch job;
     - uploads changed table rows and the hit/miss route arrays;
     - runs the read phase. Its overlap callback queues the hit experts'
       fused gate+up+SwiGLU and, after a miss, the next layer's prediction.
       The host then waits for the reads.
   - Routed gate+up over the miss rows, then routed down over all six slots.
   - hc expand-sum.
3. **Close**
   - Final collapse and norm.
   - Output head (2.3 ms).
   - Device argmax rows for verification.
   - Logits row download (517 KB per row).
   - `hipDeviceSynchronize`.

**The per-layer host round trip is hidden** while all routes hit. The route
copy and the host's planning overlap the shared expert, 113-177 µs, and
the routed gate+up starts 2-3 µs after the shared down ends in every traced
layer. Only misses expose it.

## Measured budgets

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

The same per token, plus **~0.9 ms of host time after each logits download**:
1.29-1.51 ms against 0.43-0.49 ms greedy
([hostgap.txt](artifacts/2026-11-02-decode/hostgap.txt)). Top-p 0.95 keeps the
request off the device sampler, so `sampler.Sample` builds the distribution
over all 129,280 logits on the host while the GPU waits.

### DSpark, per cycle by verified rows (profiled)

[rowcost.txt](artifacts/2026-11-02-decode/rowcost.txt). "Pre-verify" covers
the host work after the previous download plus the drafter pass. Verify is
the target evaluation.

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

- **Greedy picks only 0 or 4-5 drafts.** That follows from the policy bug
  below.
- **Each row past the first adds:**
  - ~9 ms of routed expert GEMVs;
  - ~3 ms of other GPU work: attention, hc, the router;
  - ~4.5 ms of read stall. Verify idle grows from 11.6 ms at one row to
    34 ms at six.
- **Sampled cycles also pay ~0.85 ms of host verification per row**
  ([hostgap.txt](artifacts/2026-11-02-decode/hostgap.txt): 4.9-5.0 ms
  after a 6-row download, 0.24-0.71 ms after a 1-row one).

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
- The Q6_K `attn_q_b` and `attn_output_a` already reach 231-232.

## Levers, by measured size

### 1. DSpark greedy width policy (bug; up to AR-equal on chat)

With a drafter that has the confidence head, greedy cycles are `learned`:
- `prepare` sets `c.tail = limit`, so every cycle runs the drafter
  (~9.3 ms).
- `finish_draft` then picks `best_width(confidence)`, which may be 0.

`best_width` compares `1 / cycle_ms[0]` with `expected / cycle_ms[w]`.
But `finish` updates `cycle_ms[0]` only for `c.plain` cycles, and a learned
cycle is never `c.plain`. So `cycle_ms[0]` stays at its prior, 42.4 ms,
measured on resident `IQ1_M` without a draft. The drafting widths learn
streamed costs, ~130 ms at 5 rows. Measured:

- **A plain learned cycle really costs 72 ms**: a 62.6 ms evaluation plus
  the 9.3 ms draft it discarded. The policy believes 42.4 ms, so it drafts
  only when the expected emitted tokens exceed ~3.1 at width 5. The correct
  bar is ~1.8: 131.8 / 72.0.
- **On chat, 492 of 499 profiled cycles drafted and verified one row.**
  Unprofiled, chat verified drafts in 52 of ~424 cycles, and every one of
  those cycles ran the drafter.
- **Greedy DSpark chat decodes 16.9 tok/s**, the same as AR (16.5) and below
  sampled DSpark (19.0). Sampled uses the confident-prefix rule instead.

Fix:
- Learn `cycle_ms[0]` from every one-row cycle.
- Account for the draft separately: a draft cost plus a verify cost per
  width.
- Decide before drafting whether to draft at all, from the recent
  confidence and acceptance history, with the probing `draft_width`
  already uses.

Measure chat, code and long greedy, plus `tests/functional` greedy-equality
checks; outputs must not change. The policy also runs on resident `IQ1_M`,
where its prior is roughly right, but the discarded draft is still unpriced.

### 2. Expert read stalls (4.2-6.5 ms/token AR, 8-13 ms per DSpark cycle)

Route hits per 64-token interval
([server-logs.txt](artifacts/2026-11-02-decode/server-logs.txt)):

| Phase | Hits | Read/token | Read wait/token |
|---|---:|---:|---:|
| Fresh server, first 64 tokens of the first reply | 74-78% | 266-304 MiB | 23.6-24.5 ms |
| Same reply, tokens 128-512 | 97.5-98.8% | 29-68 MiB | 3.3-6.6 ms |
| First 64 tokens after the 3.5K / 24K-token prompts | 95.6-97.2% | 82-128 MiB | 7.5-10 ms |
| Steady | 98.4-99.2% | 21-50 MiB | 2.3-5 ms |

- At 98.5% hits a token still misses ~3.6 experts (~36 MiB), mostly 1-2 per
  layer in the layers that miss. Each miss is ~10 MiB at ~10 GB/s on two
  drives, ~1 ms, and only part of it hides behind the hit experts' gate+up.
- Prediction runs only after a layer missed (STREAMING.md); prefetched
  shares are 12-50%.
- Open items:
  - **Two-layer-ahead prediction with a usefulness gate** (`BACKLOG.md`
    item 21). Mako's form gained +3.5% on a story.
  - **Decode after long prompts** (item 22): the first 64 tokens after the
    24K prompt read 2-3× the steady rate.
  - **More cache memory** (item 30): replays put +3.7 GiB at −28% decode
    misses.
  - **The eviction scan** (item 31): ~48 µs per eviction on the host,
    0.2-0.7 ms per token at an 88 GiB cap, sitting between the routes and
    the miss launch.
- **A fresh server's start** costs ~24 ms per token for its first 64
  tokens. The built-in ranking is a calibration trace and does not match
  chat text. A chat-and-code histogram (`MODEL.gguf.expert-histogram`), or
  a seed refreshed from served traces, would move that. It matters for
  benchmarks and short sessions, not for long-running servers.
- DSpark verification multiplies the exposure: 6 rows read ~4.5 ms more
  per extra row (lever 3), and the drafter's 4.6 GiB come out of the cache.

### 3. Verification row cost on the streamed build

Each row past the first costs ~12 ms of GPU time (9 ms of routed experts)
plus ~4.5 ms of read stall. The 2026-10-27 `--spec-bench` table in
`BACKLOG.md` shows ~12 ms per row on resident `IQ1_M` too. The GPU part is
the rows' routed reads and attention; the stall part is specific to
streaming.
- **Item 3, rejected rows' expert reads:** charge a row its own miss bytes
  when choosing the width. Today the policy prices misses only through
  average cycle times.
- **Item 13 / 24, cache-aware routing:** changes outputs; opt-in only.
- **Grouped expert GEMV for 2-6 rows** (`expert_group_kernel`): it already
  reads each distinct expert once per layer. The verify routed time of
  ~9 ms per row is ~2.4 MiB × new experts per row, at the same 205-220 GB/s
  as AR. Count distinct experts per verification layer from a route trace
  before working on the kernel.

### 4. Host sampling with top-p, top-k or min-p (0.9 ms/token AR, 2-5 ms per DSpark cycle)

`DsparkSamplerBridge::DeviceTemperature` allows the device sampler
(`ds41_v41_sample_row`) only for plain temperature sampling. With top-p
0.95, a common client setting, every row goes to the host:
- AR: `sampler.Sample` after `CopyLogits`;
- DSpark: `Verify` → `working_.Distribution`.

Each row builds the full 129,280-entry distribution while the GPU idles:
- AR sampled: +0.85 ms/token (1.5%);
- sampled DSpark: +0.85 ms per verified row, 2.5-5 ms per 4-6-row cycle
  (2-3.5%).

A device top-k/top-p/min-p filter over the downloaded row is a contained
change, using the same uniforms and the same tie rules as the host sampler.
That keeps seeded outputs identical. Penalties and constraint masks can
stay on the host.

### 5. Kernel bandwidth (~3.9 ms/token on the critical path, ~7%)

See the per-kernel table. The Q8_0 GEMVs (`gemv_kernel<8, …>`) are the
largest block: 27 layers × (`q_b`, `output_a`, `output_b`, shared) at
208-220 GB/s, ~2.3 ms per token short of 240. The Q6_K ones reach 231-232
on the same shapes.
- Already measured:
  - E46: paired IQ1 down rows (retained);
  - E51: single-slot experts (retained);
  - wide loads (`BACKLOG.md` item 15);
  - nontemporal loads in the model (item 33, no gain).
- Attention partials written back during the dense GEMVs: E65, item 33,
  ~0.6 ms/token, with changed rounding.
- Work from `tools/bench/gfx1151_peak.hip` against `ds41_mmq_bench --dense`
  per shape. Then check in the model, where the expert uploads and the hc
  mix's side stream contend for the same DRAM.

### 6. Launch gaps (~2.9 ms/token traced)

There are ~975 launches per token, about 25 per layer. Traced gaps average
~3 µs, and after a long GEMV ~6.5-8 µs, of which ~4.5 µs is the tracer
(E60).
- Graph replay does not remove them (E44).
- Fewer launches might: the fusions listed in `BACKLOG.md` items 4 and 18
  (q rope into `q_b`, route into the router, the window copy).
- A per-layer persistent kernel is the larger step.

Estimate the untraced share from event timing before starting.

### 7. Drafter pass (9.2 ms per drafting cycle, ~7% of a 5-row cycle)

- **Target head over the drafted rows (2.5 ms).** It reads the 543 MB head
  once per cycle, at 213 GB/s. A request-vocabulary head (`BACKLOG.md`
  item 15, E64: +1.1% greedy in replay) would shrink it.
- **`markov_select` (1.5 ms; item 20).** Five dependent passes over the
  66 MB W2. Only an exact bound that skips rows would cut it (E47).
- **Drafter stages (~4.5 ms).** Q4_K GEMVs and Q2_K experts, at bandwidths
  not yet tabulated. Do that before any drafter kernel work.

### 8. Host work per step (0.3-0.45 ms greedy)

Between the logits download and the first kernel:
- the host argmax over 129,280 floats (greedy AR) or the verification;
- the Engram hash;
- an `std::async` chain plus `engram_read` threads, created per evaluation;
- the embedding dequantization and its uploads.

Under 1% of a token. A persistent Engram reader pool (`BACKLOG.md`
item 15) and the device argmax for one-row steps would take most of it.
Low priority.

## Things that look like levers but are not

- **The per-layer route round trip.** The shared expert covers it while
  routes hit. Every traced hit layer starts its routed kernel 2-3 µs after
  the shared down.
- **The hc mix's 69 GB/s.** It runs on a side stream beside the
  projections (E57). Fusing it with the collapse was slower (E49).
- **GPU clocks during read waits** (`BACKLOG.md` item 32): they stay at
  2.2-2.85 GHz.
- **Graph capture** (E44): ~0.13 µs per launch.
- **Engram row reads in decode:** ~0.1 ms/token of wait.

## Reproducing

```sh
# Release build
cmake --build --preset release --parallel 4

# One mode, profiled (1) or not (0); D, CODE and DOC as in run.sh
D=/models/ds41 CODE=tools/ds41/replay_sim.py DOC=docs/server.md OUT=prof \
  docs/models/deepseek-v4.1-flash/artifacts/2026-11-02-decode/run.sh ar-greedy ar 0 1
#   modes: "ar 0", "ar 1", "dspark 0", "dspark 1"

# Analysis (rocpd sqlite from rocprofv3)
A=docs/models/deepseek-v4.1-flash/artifacts/2026-11-02-decode
python3 $A/budget.py prof/ar-greedy        # per-token classes and idle
python3 $A/roles.py prof/ar-greedy -3      # per-kernel bandwidth, request -3/-2/-1
python3 $A/rowcost.py prof/ds-greedy       # DSpark cycle cost by rows
python3 $A/hostgap.py prof/ar-sampled      # host time after each logits download
python3 $A/drafter.py prof/ds-greedy -2    # drafter pass by kernel

# DSpark per-cycle log (multi-row cycles only)
GUFO_DEEPSEEK41_DSPARK_TRACE=1 build/release/gufo serve ...
```

Correctness for any decode change: greedy speculative output equals AR
output on the same prompt (the card's rule). Also run the 3K decode scores
(`ds41_load_probe --stream-experts --decode IDS --score`) and the
`tests/functional` DeepSeek suites for the affected modes. For sampled
changes, seeded outputs must not change, unless the change is to sampling
itself.
