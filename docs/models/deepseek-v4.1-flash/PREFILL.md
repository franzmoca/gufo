# DeepSeek V4.1 Flash prefill: where the time goes

A starting point for prefill work on the streamed `UD-IQ2_XXS` build
(146 GiB of routed experts, ~95 GiB expert cache, tails sidecar on a second
NVMe). It records the measured budget, each kernel's position against the
gfx1151 ceilings, what has already been tried, and the open levers ranked by
measured size. Measured 2026-11-02 on `feat/ds41-ud-iq2xxs` at `c8220c2b`.
Profile tables are in [artifacts/2026-11-02-prefill](artifacts/2026-11-02-prefill/).

**Status:** the profile below predates the fix for lever 1. Server prefill
steps no longer replay the decoder layers at cache checkpoints
([E73](EXPERIMENTS.md#e73--one-decoder-replay-per-served-prompt-retained)):
fresh 14-24K-token documents now take 2.4-2.7 s less to the first token.
The other levers are unchanged.

Read [STREAMING.md](STREAMING.md) for the expert store and
[EXPERIMENTS.md](EXPERIMENTS.md) for every experiment cited as `E<n>`.

## Summary

A fresh 19,927-token document prompt served over HTTP takes 24.06 s
(828 tok/s). `gufo bench` pp2048 at 8K depth reports 1,100-1,160 tok/s for
the same build. The served number is the one users see, and it differs from
the bench in three ways:

1. **Five decoder replays instead of one (~4.4 s, 18%; fixed, E73).** Every
   prefill step that ended at a cache checkpoint ran the 140-row decoder
   replay (layers 20-39), and those layers' experts are mostly cold: 29.5 ms
   of read stall per layer, against 10.5 ms in the encoder batches.
2. **Expert read stalls in the encoder batches (2.1 s, 9%).** About 10 ms per
   layer of 2,048 rows, even with wide next-layer prediction.
3. **Real text routes to more experts than the bench prompt.** The routed
   FP16 kernels take 24-25 ms per 2,048-row gate+up on the document, against
   18.5 ms on the bench prompt (23.5 against 31.4 TFLOP/s).

On the GPU itself, routed experts take 47% of busy time, the dense FP16
projections ~27%, attention 11%, the hyper-connection state passes 7.5% and
the BF16 matmuls (mainly the router) 5%.

## Workloads and how they were measured

| Workload | Command | Result |
|---|---|---|
| Bench | `gufo bench -m MODEL --stream-experts -p 2048 -n 0 -d 8192 -r 3` | 1,098.6 ± 50.1 tok/s profiled; 1,150-1,160 unprofiled |
| Served document | `gufo serve ... llm --model MODEL --context 65536 --stream-experts --speculative dspark --dspark-model DRAFT --sessions 1`, then two fresh documents with `max_tokens: 1` ([two_docs.py](artifacts/2026-11-02-prefill/two_docs.py)); the second is analysed | 23,954 tokens in 30.00 s, then 19,927 tokens in 24.07 s |

The documents are long Gufo sources: the HTTP server documentation,
`expert_store.cpp` and the KV cache documentation, 14-24K tokens each. Any
real document of that size works. Do not judge routed kernels on the bench
prompt alone (see "Bench against served").

Profiles: `nix develop -c rocprofv3 --kernel-trace --memory-copy-trace
--output-format rocpd -d DIR -o t -- <command>`, then
`tools/prof/prof.py show DIR/t_results.db --passes 1000 --pass N --stages
ds41`. The kernel trace adds no measurable time here: the served request ran
at 828 tok/s profiled and unprofiled. Attribute idle time by the kernel that
follows each gap: a gap before `routed_counts_kernel` is the expert store's
prepare (route copy, plan, reads), and a gap before the Q6_K
`dense_f16_kernel` with 100 row blocks is the Engram projection waiting on
its row reads.

## Model shape (UD-IQ2_XXS)

- 40 layers, `n_embd` 5,120, 384 routed experts of 5,120 × 2,304 per
  projection, 6 routed per token, one shared expert (Q8_0, Q5_K in
  layers 35-39).
- Routed types by layer:

| Layers | gate / up | down |
|---|---|---|
| 0 | IQ1_S | Q2_K |
| 1, 5-9 | IQ1_S | IQ2_XS |
| 2-4 | IQ1_S | IQ2_XXS |
| 10-19 | IQ2_XXS | IQ2_S |
| 20-24 | IQ2_XXS | IQ3_XXS |
| 25-34 | IQ2_S | IQ3_XXS |
| 35-39 | IQ2_S | IQ2_S |

- Dense attention: `attn_q_a` 5,120 → 1,280, `attn_q_b` 1,280 → 32,768,
  `attn_kv` 5,120 → 512, `attn_output_a` 8 groups of 4,096 → 1,024,
  `attn_output_b` 8,192 → 5,120. Q8_0 in most layers, Q6_K at the end.
- BF16: the router (5,120 → 384, FP32 weights converted at load), the
  compressor KV (5,120 → 512) and the indexer `q_b` (1,280 → 4,096).
- Engram: a 25,600 × 6,144 Q6_K projection on two layers, rows read from
  disk.
- **Bounded replay** (`DS41_DECODER_REPLAY_ROWS`, `model.h`): a prompt runs
  encoder layers 0-19 over every row, and decoder layers 20-39 over only the
  last 140 rows before the evaluation's `prompt_end`. Layer 20 still
  compresses every position.

## Ceilings

| Resource | Measured ceiling | Source |
|---|---:|---|
| FP16/BF16 WMMA | 55.4 TFLOP/s | `tools/bench/gfx1151_peak.hip` |
| DRAM read | 240 GB/s | same |
| Expert reads from two NVMe drives | ~10.4 GB/s (0.38 ms + 0.10 ms/MiB fitted) | `tools/ds41/prefetch_sim.py` read-cost fit on the served traces |

## Served budget: one 19,927-token fresh prompt

| Part | Time | Share |
|---|---:|---:|
| Wall span | 24.06 s | 100% |
| GPU busy | 18.41 s | 76.5% |
| Idle: expert prepare in 2,048-row encoder batches (200 layer calls) | 2.10 s | 8.7% |
| Idle: expert prepare in the 140-row decoder replays (100 layer calls) | 2.95 s | 12.3% |
| Idle: Engram row reads before the projection | 0.27 s | 1.1% |
| Idle: other (copies, host) | ~0.3 s | ~1% |

GPU busy by stage (`prof.py --stages ds41`):

| Stage | ms | % of busy |
|---|---:|---:|
| Routed FP16 experts | 8,617 | 46.8 |
| Dense FP16 projections and the rest ("other") | 4,842 | 26.3 |
| Attention | 1,991 | 10.8 |
| Hyper-connection state passes | 1,377 | 7.5 |
| BF16 matmuls (router, compressor, indexer) | 931 | 5.1 |
| Indexer scores and top-k | 203 | 1.1 |
| FP16 conversion, Engram, routing, SwiGLU, copies, rope, norms | ~460 | 2.5 |

### The decoder replays

Captures for this request (server log): 4,096, 8,192, 12,288, 18,432
(grid checkpoints), 19,924 (stable boundary) and 19,927 (prompt end). Each
capture ends a prefill step. The server passed each step the prompt cut at
that step's end (`text_model_runner.cpp` completes the model frontier at
every cache boundary), so DS41 took it as `prompt_end` and ran the decoder
layers over the step's last 140 rows. Five steps ended in a 140-row replay.
The final 3-row step runs as a decode-size batch.

Fixed in E73: the runner state now carries the whole prompt's length
(`TextRunnerState::prefill_prompt_end`), so only the rows within 140 of the
real prompt end run the decoder layers, here split between the stable step
and the last one.

- **Who plans them:** `cache::PlanCaptures` (`src/cache/retention.cpp`):
  - The required boundaries: reused, stable and prompt.
  - Up to four grid checkpoints at 2,048 multiples, capped by `GridLimit`
    (`min(4, budget / 12 / private bytes)`).
  - Called from `ComponentTextCache::Lease::PlannedCaptures`
    (`src/cli/serve/component_text_cache.cpp`).
- **Cost per replay:** ~0.3 s of GPU work, plus ~0.6 s of read stall for
  20 layers of mostly cold decoder experts. That's ~0.9 s each, ~4.4 s for
  this prompt.
- **The bench doesn't see this:** it has no cache checkpoints, so it runs one
  replay per pass.

### Bench against served

| Kernel (2,048-row batch) | Bench prompt | Served document |
|---|---:|---:|
| IQ2_XXS gate+up, layers 10-19 | 18.46 ms (31.4 TFLOP/s) | 24.67 ms (23.5) |
| IQ1_S gate+up, layers 0-9 | 17.75 ms (32.7) | 23.72 ms (24.4) |
| IQ2_S down, layers 10-19 | 10.28 ms (28.2) | 14.50 ms (20.0) |
| IQ2_XS down | 9.39 ms (30.9) | 13.72 ms (21.1) |
| Expert prepare stall per encoder layer | 7.8 ms | 10.5 ms |
| Expert prepare stall per replay layer | 7.3 ms (repeated prompt, warm decoder experts) | 29.5 ms |

Both run 12,288 routes per layer. The likely cause is the spread of routes
over experts: more distinct experts mean more partly filled 64-slot tiles,
and each tile decodes its expert's weights again. This hasn't been
confirmed. Check it with `ds41_mmq_bench --f16 1` on routes recorded from
the served document (E25 and E35 used recorded counts this way).

## Per-kernel position (served document)

FLOPs are counted on useful rows: routes × 2 × 5,120 × 2,304 per
projection for the routed kernels, rows × K × tokens × 2 for the dense ones.
For BF16, the rate counts one pass of the matrix, although the kernel runs
three (hi/mid/lo splits of the FP32 activations).

| Kernel | Shape | Per call | Rate | Calls per 20K prompt | Total |
|---|---|---:|---:|---:|---:|
| Routed gate+up IQ2_XXS | 2,048 rows | 24.7 ms | 23.5 TFLOP/s | 90 | 2.22 s |
| Routed gate+up IQ1_S | 2,048 rows | 23.7 ms | 24.4 | 90 | 2.13 s |
| Routed down IQ2_S | 2,048 rows | 14.5 ms | 20.0 | 90 | 1.30 s |
| Routed down IQ2_XS | 2,048 rows | 13.7 ms | 21.1 | 54 | 0.74 s |
| Routed down IQ2_XXS / Q2_K | 2,048 rows | 13.1 / 13.0 ms | 22.1 / 22.3 | 27 / 9 | 0.47 s |
| Routed gate+up IQ2_S (replay) | 140 rows | 7.75 ms | 5.1 | 75 | 0.58 s |
| Routed gate+up IQ2_XXS (replay) | 140 rows | 5.2 ms | 7.6 | 25 | 0.13 s |
| Routed down IQ3_XXS / IQ2_S (replay) | 140 rows | 3.9 / 4.3 ms | 5.1 / 4.6 | 75 / 25 | 0.40 s |
| Attention (`attention_f16_kernel`) | 2,048 tokens × 64 heads, up to 20K deep | 9.8 ms | — | 180 | 1.76 s |
| Dense `attn_q_b` Q8_0 | 32,768 × 1,280 | 6.17 ms | 27.8 | 180 | 1.11 s |
| Dense `attn_output_a` Q8_0 | 8 × 1,024 × 4,096 | 4.87 ms | 28.2 | 180 | 0.88 s |
| Dense `attn_output_b` Q8_0 | 5,120 × 8,192 | 4.84 ms | 35.5 | 180 | 0.87 s |
| Dense shared gate+up Q8_0 (paired) | 4,608 × 5,120 | 2.69 ms | 35.9 | 180 | 0.48 s |
| Dense shared down Q8_0 | 5,120 × 2,304 | 1.38 ms | 35.1 | 180 | 0.25 s |
| Dense `attn_q_a` + `attn_kv` Q8_0 (paired) | 1,792 × 5,120 | 1.17 ms | 32.2 | 180 | 0.21 s |
| Dense Engram projection Q6_K | 25,600 × 6,144 | 20.7 ms | 31.2 | 18 | 0.37 s |
| `hc_state_pass_kernel` | 2,048 × 4 streams × 5,120 | 3.28 ms | bandwidth-bound | 333 | 1.09 s |
| BF16 router (+ next-layer prediction) | 384 × 5,120 | 1.58 ms | 5.1 (×3 terms) | 360 | 0.57 s |
| BF16 compressor KV | 512 × 5,120 | 1.95 ms | 5.5 (×3) | 59 | 0.12 s |
| BF16 indexer `q_b` | 4,096 × 1,280 | 2.64 ms | 8.1 (×3) | 27 | 0.07 s |
| BF16 router (replay) | 384 × 5,120, 140 tokens | 0.38 ms | 1.4 (×3) | 195 | 0.07 s |

### Per token at the ceilings (encoder layers, 2,048-row batches)

| Part | Now | At 55 TFLOP/s | Realistic target |
|---|---:|---:|---|
| Routed experts (8.5 GFLOP/token) | 0.37 ms | 0.155 ms | ~0.25 ms at ~35 TFLOP/s |
| Dense projections (6.5 GFLOP/token) | 0.21 ms | 0.118 ms | ~0.17 ms at ~40 TFLOP/s |
| Attention (20 layers, 8-20K deep) | 0.10 ms | — | Depth-dependent |
| Hyper-connection passes | 0.06 ms | Near the DRAM limit | — |
| BF16 matmuls | 0.04 ms | ~0.01 ms | ~0.015 ms |
| Everything on the GPU | 0.85 ms | — | — |

The encoder batches of the served prompt take 16.9 s of GPU time,
0.85 ms/token: 1,180 tok/s if the read stalls and the four extra replays
were gone (the replays now are, E73). The bench reaches 1,100-1,160 tok/s because its routed kernels
run faster (see "Bench against served"), and it has one replay per pass and
warm decoder experts.

## Levers, by measured size

### 1. Decoder replays at grid checkpoints (fixed, E73)

The extra replays came from the server cutting the prompt at each capture,
not from the captures themselves. Each step now replays against the whole
prompt's end, so a fresh prompt runs the decoder layers once over its last
140 rows. The stable boundary just before the end shares that window with
the final step instead of costing a replay of its own.

A mid-prompt checkpoint now holds no decoder rows near its end. Its
metadata records where its decoder rows start, and a restore accepts it
only for prompts at least 140 tokens past it; shorter ones fall back to the
previous checkpoint. Restores that resume are bit-identical to uncached
prompts. Serving those short continuations without the fallback is
`BACKLOG.md` item 36.

Still open: the one remaining replay's read stalls. Decoder-layer experts
are cold during a fresh prompt, so each replay layer waits ~29.5 ms. The
ordinary next-layer prediction runs for these 140-row batches (E68) but
doesn't hide it.

### 2. Expert read stalls in encoder batches (2.1 s, ~9%)

About 10.5 ms per 2,048-row layer, with wide next-layer prediction already
on (E68).
- **More cache memory** (`BACKLOG.md` item 30): served-trace replays put
  +3.7 GiB at −17% prefill stall and +6 GiB at −26%.
- **Two-wave layers:** compute the cached experts while the misses are read.
  ds4-argodrive measured +1.1%. Needs the routed kernels to run over a
  subset of routes; the decode hit/miss split already does that.
- **Prompt admission rule** (`BACKLOG.md` item 31): admitting every expert
  routed twice cut follow-up misses 32%, but slowed prefill through
  admission copies.
- **Layer-major prefill** (`wip/ds41-layer-major-prefill`, shelved):
  - Reads fell 32%, but GPU time rose 4-6 s.
  - It has a known race: reusing adopted job rows across a layer's batches.
  - Reconsider only with a simpler design.

### 3. Routed FP16 kernels (8.6 s busy, 47%)

`kernels/rocm/ds41_routed_f16.hip.cpp`, `routed_f16_kernel<type, pair,
half_out>`. Structure:
- 64-slot tiles per expert, planned on the GPU (E7).
- 8 waves.
- Weights are the B operand, decoded to FP16 with scales folded in.
- Activations are double-buffered in LDS.

Already measured:

| Change | Result | Ref |
|---|---|---|
| 32-slot tiles | No gain; code generation trap with an ownership early return | E8 |
| 128-slot tiles | 19% fewer weight decodes, no faster | E35 |
| 4,096-token batches | +2-3% natively, memory cost; kept as an option | E16 |
| Staging, LDS lookups, conflict-free codebook | Retained (IQ2_XXS 24.5 → 20.4 ms in bench) | E25 |
| Rolled weight loop (I-cache) | Retained | `EXPERIMENTS.md` table |
| Nontemporal weight loads | Worse (refetches) | `EXPERIMENTS.md` table |
| IQ2_S, IQ3_XXS, Q2_K formats | Retained, correctness first | E67 |

E25's ablation of a 2,048-token layer (IQ1_M gate+up, 21.8 ms):
- 14.7 ms skeleton: block fields, fragment reads, products, at ~39 TFLOP/s.
- +1.0 ms weight loads.
- +1.7 ms activation staging with two barriers per stage (+4.9 for IQ2_XXS).
- +4.4 ms weight decode with its permutes.

Untried:
- **Measure on recorded served routes first.** The 25-30% gap between the
  bench and the document is the largest unexplained number here.
- **Count the tiles a real 2,048-row batch produces,** and the live 16-slot
  groups per tile. If partly filled tiles dominate, try grouping two small
  experts into one tile's activation stage. That shares no weights, so the
  win would come only from more products per barrier; measure before
  building.
- **IQ2_S and IQ2_XS downs** run 7-10% slower than IQ2_XXS down per call. If
  their decode is the gap, a 2% prefill gain.
- **The 140-row replay calls** run at 5-8 TFLOP/s. With 840 routes over 384
  experts most tiles hold 1-4 routes, so each call reads nearly every
  routed expert's weights once (an estimate; count the live tiles to
  confirm), which makes it a weight-read kernel. One call per layer remains
  per prompt since E73.

### 4. Dense FP16 projections (~3.9 s, ~16%)

`kernels/rocm/ds41_dense_f16.hip.cpp`. `attn_q_b` (27.8 TFLOP/s) and
`attn_output_a` (28.2) are the weakest:
- E23: the grouped `attn_output_a` input is re-read by each 256-row block,
  and dropping the activation loads took it from 5.50 to 3.96 ms.
- `attn_q_b` measured 4.19 ms in `ds41_mmq_bench --dense` but 6.17 ms in
  the served model. Earlier sessions put the model at 5.6-5.8 ms; E68
  records a 41 ms layer-7 outlier.

Already measured:
- E52: Q6_K occupancy, Q8_0 path, paired launches.
- E53: rejected fragment prefetch, LDS fragments, paired loads.
- E23: FP16 inputs.

Tools: `ds41_mmq_bench --dense [--x-half 1]` and
`tools/bench/ds41_dense_f16_tune.hip`.

### 5. BF16 router (0.57 s, ~2.4%, plus 0.07 s in replays)

`matmul_bf16_wmma_kernel<128>` on 384 rows launches only 3 × 32 = 96
workgroups for 2,048 tokens and reaches 5.1 TFLOP/s per pass of the matrix
(15 across its three terms). It runs twice per encoder layer: routing, plus
the wide next-layer prediction.
- A grid with more workgroups (64-row tiles, or split K with an FP32
  reduction in the same order) could reach the rate of the larger BF16
  shapes.
- The result must stay bit-identical. E71's narrow kernel shows how: keep
  the 32-value K steps, the part order and the FP32 sums per step.

Worth ~0.3-0.4 s per 20K prompt.

### 6. Attention (1.76 s, ~7%)

`attention_f16_kernel`, 9.8 ms per encoder layer per 2,048 rows on the
document (8.8 ms at 8-14K in the bench); it grows with depth.
- E21: staging, retained.
- E53: fixed costs, rejected.
- E19 / E20: indexer.

Check `ds41_attention_test --bench` at 16K and 32K before working on it.

### 7. Hyper-connection state passes (1.09 s, ~5%)

`hc_state_pass_kernel`, one read of the 4 × 5,120 FP32 state per token.
E22 fused the FFN's closing expand into it, and it is near the DRAM limit.
The only lever left is fewer passes or a smaller state type, which changes
numerics.

### 8. Engram row reads (0.27 s, ~1%)

Two Engram layers per 2,048-row batch wait ~13.7 ms each for host row reads
before the projection. Prefetching the rows one layer early would hide this.
E26 already moved the rows into pinned staging.

## Things that look like levers but are not

- **Prediction lead time** for short passes: one-ahead predictions already
  land 7.7 ms before the target layer, more than its 3.2 ms of reads. Two or
  three layers ahead catch fewer misses (see `prefetch_sim.py` recall and
  the 2026-11-02 trace).
- **Undrained prefetch adoption and always-on prediction** for short passes
  (E72): within noise.
- **Expert repacking on disk** (E30; E70 replay): runs of missed neighbours
  are rare in agent sessions. Prompt-heavy sessions are still unreplayed
  (`BACKLOG.md` item 35).
- **GPU clocks during read waits** (`BACKLOG.md` item 32): they stay at
  2.2-2.85 GHz.

## Reproducing

```sh
# Release build
cmake --build --preset release --parallel 4
M=/path/DeepSeek-V4.1-Flash-1017-UD-IQ2_XXS-00001-of-00006.gguf

# Bench profile
nix develop -c rocprofv3 --kernel-trace --output-format rocpd -d prof -o t -- \
  build/release/gufo bench -m $M --stream-experts -p 2048 -n 0 -d 8192 -r 3
python3 tools/prof/prof.py show prof/t_results.db --passes 200 --stages ds41

# Served profile: start the server under rocprofv3, send two fresh
# documents (artifacts/2026-11-02-prefill/two_docs.py), stop the server,
# then take the second request's pass:
python3 tools/prof/prof.py show prof/t_results.db --passes 1000 --pass 1 --stages ds41 --gaps 15

# Route traces for replays (prediction is off while tracing)
DS41_EXPERT_TRACE=routes.trace build/release/gufo serve ...
python3 tools/ds41/prefetch_sim.py routes.trace --cache --limit-gib 98.8
```

Kernel microbenchmarks: `ds41_mmq_bench --f16 1` (routed, recorded counts),
`ds41_mmq_bench --dense`, `ds41_attention_test --bench`,
`tools/bench/ds41_dense_f16_tune.hip`, `tools/bench/gfx1151_peak.hip`.
Correctness: `ds41.mmq_typed`, `ds41.attention`, and the 3K batched and
decode scores (`ds41_load_probe --stream-experts --decode IDS --score
[--batch]`, compared against the oracle). These are bit-identical
requirements for kernel changes.
