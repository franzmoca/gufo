# DeepSeek V4.1 Flash prefill: where the time goes

A starting point for prefill work on the streamed `UD-IQ2_XXS` build
(146 GiB of routed experts, ~95 GiB expert cache, tails sidecar on a second
NVMe). It records the measured budget, each kernel's position against the
gfx1151 ceilings, what has already been tried, and the open levers ranked by
measured size. Measured 2026-11-02 on `feat/ds41-ud-iq2xxs` at `984ad880`.
Profile tables are in [artifacts/2026-11-02-prefill](artifacts/2026-11-02-prefill/).

Read [STREAMING.md](STREAMING.md) for the expert store and
[EXPERIMENTS.md](EXPERIMENTS.md) for every experiment cited as `E<n>`.

## Summary

A fresh 19,927-token document prompt served over HTTP takes 20.59 s
(968 tok/s). `gufo bench` pp2048 at 8K depth reports 1,100-1,160 tok/s for
the same build. The served number is the one users see. The gap to the
bench:

1. **Expert read stalls in the encoder batches (2.1 s, 10%).** About 10.5 ms
   per layer of 2,048 rows, even with wide next-layer prediction.
2. **The decoder replay waits on cold experts (0.9 s, 4.4%).** The prompt's
   last 140 rows run layers 20-39 once, and each of those layers waits
   45 ms for its experts, against 7.3 ms in the bench.
3. **Real text routes to more experts than the bench prompt.** The routed
   FP16 kernels take 24-26 ms per 2,048-row gate+up on the document, against
   17.8-18.5 ms on the bench prompt (22.5-23.9 against 31.4-32.7 TFLOP/s).

On the GPU itself (17.0 s), routed experts take 46% of busy time, the dense
FP16 projections ~27%, attention 11%, the hyper-connection state passes 8%
and the BF16 matmuls (mainly the router) 5%.

Until E73 every cache checkpoint of the prompt also ran a decoder replay:
five replays instead of one, 24.06 s for this prompt. Fixing that saved
2.4-2.7 s per fresh 14-24K-token document.

## Workloads and how they were measured

| Workload | Command | Result |
|---|---|---|
| Bench | `gufo bench -m MODEL --stream-experts -p 2048 -n 0 -d 8192 -r 3` | 1,098.6 ± 50.1 tok/s profiled; 1,150-1,160 unprofiled |
| Served document | `gufo serve ... llm --model MODEL --context 65536 --stream-experts --speculative dspark --dspark-model DRAFT --sessions 1`, then two fresh documents with `max_tokens: 1` ([two_docs.py](artifacts/2026-11-02-prefill/two_docs.py)); the second is analysed | 23,954 tokens in 26.54 s, then 19,927 tokens in 20.60 s |

The documents are long Gufo sources: the HTTP server documentation,
`expert_store.cpp` and the KV cache documentation, 14-24K tokens each. Any
real document of that size works. Do not judge routed kernels on the bench
prompt alone (see "Bench against served").

Profiles: `nix develop -c rocprofv3 --kernel-trace --memory-copy-trace
--output-format rocpd -d DIR -o t -- <command>`, then
`tools/prof/prof.py show DIR/t_results.db --passes 1000 --pass N --stages
ds41`. The kernel trace adds no measurable time here (828 tok/s profiled and
unprofiled before E73). Attribute idle time by the kernel that follows each
gap: a gap before `routed_counts_kernel` is the expert store's prepare
(route copy, plan, reads), and a gap before the Q6_K `dense_f16_kernel`
with 100 row blocks is the Engram projection waiting on its row reads. The
routed kernel's grid tells the batch apart: `grid_y` 576 for 2,048 rows,
523 for the last encoder batch, 396 for the 140-row decoder replay.

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
  encoder layers 0-19 over every row, and decoder layers 20-39 over only its
  last 140 rows. Layer 20 still compresses every position. The server passes
  the whole prompt's length to every prefill step
  (`TextRunnerState::prefill_prompt_end`), so steps that end at cache
  checkpoints skip the decoder layers (E73).

## Ceilings

| Resource | Measured ceiling | Source |
|---|---:|---|
| FP16/BF16 WMMA | 55.4 TFLOP/s | `tools/bench/gfx1151_peak.hip` |
| DRAM read | 240 GB/s | same |
| Expert reads from two NVMe drives | ~10.4 GB/s (0.38 ms + 0.10 ms/MiB fitted) | `tools/ds41/prefetch_sim.py` read-cost fit on the served traces |

## Served budget: one 19,927-token fresh prompt

| Part | Time | Share |
|---|---:|---:|
| Wall span | 20.59 s | 100% |
| GPU busy | 17.00 s | 82.6% |
| Idle: expert prepare in encoder batches (180 + 20 layer calls) | 2.11 s | 10.2% |
| Idle: expert prepare in the 140-row decoder replay (20 layer calls) | 0.91 s | 4.4% |
| Idle: Engram row reads before the projection | 0.37 s | 1.8% |
| Idle: other (copies, planning, host) | ~0.2 s | ~1% |

GPU busy by stage (`prof.py --stages ds41`):

| Stage | ms | % of busy |
|---|---:|---:|
| Routed FP16 experts | 7,827 | 46.0 |
| Dense FP16 projections and the rest ("other") | 4,566 | 26.8 |
| Attention | 1,811 | 10.6 |
| Hyper-connection state passes | 1,346 | 7.9 |
| BF16 matmuls (router, compressor, indexer) | 860 | 5.1 |
| Indexer scores and top-k | 187 | 1.1 |
| FP16 conversion, Engram, routing, SwiGLU, copies, rope, norms | ~400 | 2.4 |

### The decoder replay

Captures for this request (server log): 4,096, 8,192, 12,288, 18,432
(grid checkpoints), 19,924 (stable boundary) and 19,927 (prompt end). Each
capture ends a prefill step. Only the rows from 19,787 on run layers 20-39:
137 of them at the end of the stable step, the last 3 in the final
decode-size step. The grid checkpoints hold no decoder rows; a restore uses
one only for prompts at least 140 tokens past it (E73, `BACKLOG.md`
item 36).

The replay costs ~0.3 s of GPU work and 0.91 s of read stall: 45.4 ms per
layer, against 10.5 ms per 2,048-row encoder layer. Its 840 routes per layer
land on decoder-layer experts, which the prompt's encoder batches never
touch or prefetch; the stall says most of them miss. Before E73 the earlier
replays had warmed them: the five replays averaged 29.5 ms per layer.

### Bench against served

| Kernel (2,048-row batch) | Bench prompt | Served document |
|---|---:|---:|
| IQ2_XXS gate+up, layers 10-19 | 18.46 ms (31.4 TFLOP/s) | 24.28-24.67 ms (23.5-23.9) |
| IQ1_S gate+up, layers 0-9 | 17.75 ms (32.7) | 23.72-25.81 ms (22.5-24.4) |
| IQ2_S down, layers 10-19 | 10.28 ms (28.2) | 14.28-14.50 ms (20.0-20.3) |
| IQ2_XS down | 9.39 ms (30.9) | 13.55-13.72 ms (21.1-21.4) |
| Expert prepare stall per encoder layer | 7.8 ms | 10.5 ms |
| Expert prepare stall per replay layer | 7.3 ms (repeated prompt, warm decoder experts) | 45.4 ms |

Served ranges are two profiles of the same request (`c8220c2b` and
`984ad880`; the kernels are unchanged between them). Both prompts run 12,288
routes per layer; the difference is in how they spread over experts:
recorded served routes reproduce the served kernel times in
`ds41_mmq_bench`, and a served batch fills its 64-slot tiles only 46% (see
lever 3).

## Per-kernel position (served document)

FLOPs are counted on useful rows: routes × 2 × 5,120 × 2,304 per
projection for the routed kernels, rows × K × tokens × 2 for the dense ones.
For BF16, the rate counts one pass of the matrix, although the kernel runs
three (hi/mid/lo splits of the FP32 activations).

| Kernel | Shape | Per call | Rate | Calls per 20K prompt | Total |
|---|---|---:|---:|---:|---:|
| Routed gate+up IQ1_S | 2,048 rows | 25.8 ms | 22.5 TFLOP/s | 90 | 2.32 s |
| Routed gate+up IQ2_XXS | 2,048 rows | 24.3 ms | 23.9 | 90 | 2.19 s |
| Routed down IQ2_S | 2,048 rows | 14.3 ms | 20.3 | 90 | 1.29 s |
| Routed down IQ2_XS | 2,048 rows | 13.6 ms | 21.4 | 54 | 0.73 s |
| Routed down IQ2_XXS / Q2_K | 2,048 rows | 12.9 / 12.8 ms | 22.4 / 22.7 | 27 / 9 | 0.46 s |
| Routed gate+up IQ2_S (replay) | 140 rows | 8.4 ms | 4.7 | 15 | 0.13 s |
| Routed gate+up IQ2_XXS (replay) | 140 rows | 6.1 ms | 6.5 | 5 | 0.03 s |
| Routed down IQ3_XXS / IQ2_S (replay) | 140 rows | 4.3 / 4.5 ms | 4.6 / 4.4 | 15 / 5 | 0.09 s |
| Attention (`attention_f16_kernel`) | 2,048 tokens × 64 heads, up to 20K deep | 9.1 ms | — | 180 | 1.64 s |
| Dense `attn_q_b` Q8_0 | 32,768 × 1,280 | 6.19 ms | 27.8 | 180 | 1.11 s |
| Dense `attn_output_b` Q8_0 | 5,120 × 8,192 | 4.80 ms | 35.8 | 180 | 0.86 s |
| Dense `attn_output_a` Q8_0 | 8 × 1,024 × 4,096 | 4.78 ms | 28.7 | 180 | 0.86 s |
| Dense shared gate+up Q8_0 (paired) | 4,608 × 5,120 | 2.68 ms | 36.1 | 180 | 0.48 s |
| Dense shared down Q8_0 | 5,120 × 2,304 | 1.36 ms | 35.5 | 180 | 0.24 s |
| Dense `attn_q_a` + `attn_kv` Q8_0 (paired) | 1,792 × 5,120 | 1.16 ms | 32.4 | 180 | 0.21 s |
| Dense Engram projection Q6_K | 25,600 × 6,144 | 20.5 ms | 31.4 | 18 | 0.37 s |
| `hc_state_pass_kernel` | 2,048 × 4 streams × 5,120 | 3.29 ms | bandwidth-bound | 333 | 1.09 s |
| BF16 router (+ next-layer prediction) | 384 × 5,120 | 1.57 ms | 5.1 (×3 terms) | 360 | 0.57 s |
| BF16 compressor KV | 512 × 5,120 | 1.92 ms | 5.6 (×3) | 63 | 0.12 s |
| BF16 indexer `q_b` | 4,096 × 1,280 | 2.64 ms | 8.1 (×3) | 27 | 0.07 s |

### Per token at the ceilings (encoder layers, 2,048-row batches)

| Part | Now | At 55 TFLOP/s | Realistic target |
|---|---:|---:|---|
| Routed experts (8.5 GFLOP/token) | 0.37 ms | 0.155 ms | ~0.25 ms at ~35 TFLOP/s |
| Dense projections (6.5 GFLOP/token) | 0.21 ms | 0.118 ms | ~0.17 ms at ~40 TFLOP/s |
| Attention (20 layers, 8-20K deep) | 0.09 ms | — | Depth-dependent |
| Hyper-connection passes | 0.06 ms | Near the DRAM limit | — |
| BF16 matmuls | 0.04 ms | ~0.01 ms | ~0.015 ms |
| Everything on the GPU | 0.84 ms | — | — |

The encoder batches take ~16.7 s of GPU time, 0.84 ms/token: 1,190 tok/s
if every read stall were gone. The bench reaches 1,100-1,160 tok/s because
its routed kernels run faster (see "Bench against served") and its decoder
experts are warm.

## Levers, by measured size

### 1. Expert read stalls in encoder batches (2.1 s, ~10%)

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

### 2. Cold experts in the decoder replay (0.91 s, 4.4%)

Each of the 20 replay layers waits 45.4 ms. The next-layer prediction runs
for these 140-row batches (E68) but hides little: the 840 routes of a layer
spread over most of its 384 experts, and the misses are many small reads.
- **Check first** with a route trace (`DS41_EXPERT_TRACE`) of a fresh
  document after a different one: how many replay misses per layer, and
  whether those experts were cached before the prompt's encoder batches
  evicted them. If they were, the first decode tokens pay for the same
  misses.
- **Read the replay layers' experts early.** The replay's rows are known
  once the last encoder batch starts; their layer-20 routes are known after
  layer 19 of that batch. Predicting further ahead than one layer helped
  little for short passes (see "Things that look like levers"), so measure
  the recall at these rows before building anything.
- **Protect decoder-layer experts during long prompts,** if the trace shows
  they are evicted. Combines with item 31's admission rule.

### 3. Routed FP16 kernels (7.8 s busy, 46%)

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
| 128- or 32-value activation stages (on served routes) | Both slower: gate+up +0.2-0.3 ms, down +0.5 ms | E76, E77 |
| IQ2_S sign masks by bit spreading instead of the LDS table | Flat to +4% on the down | E78 |
| `iterative-ilp` scheduler for the routed kernels | Flat (23.07 / 12.89 against 23.18 / 12.91 ms) | E79 |
| IQ2_S grid table as packed bytes in LDS (16 → 8 KiB), converted at decode | Down +7% (13.82-13.87 against 12.87-12.91 ms) | E80 |

E25's ablation of a 2,048-token layer (IQ1_M gate+up, 21.8 ms):
- 14.7 ms skeleton: block fields, fragment reads, products, at ~39 TFLOP/s.
- +1.0 ms weight loads.
- +1.7 ms activation staging with two barriers per stage (+4.9 for IQ2_XXS).
- +4.4 ms weight decode with its permutes.

The bench-against-served gap comes from the routes themselves.
`tools/ds41/route_occupancy.py` turns a route trace into `--counts` lines
for `ds41_mmq_bench`; on a 2,048-row batch recorded from the served
document, the microbenchmark reproduces the served times (IQ2_XXS gate+up
23.2 ms, IQ2_S down 12.9 ms; `ds41_mmq_bench --f16 1 --type 16
--down-type 22 --counts FILE --line 10`). Over the served session's 520
prompt batches, a batch fills 419 64-slot tiles on average, with 46% of
their slots holding a route: more than half of each tile's products are
padding. Use these recorded counts, not the bench prompt, for routed
kernel work.

Untried:
- **Fuller tiles.** With 46% of slots used, grouping two small experts
  into one tile's activation stage, or tiles of 16/32 slots for small
  experts alongside 64 for large ones, attacks the padding directly. Both
  share no weights across experts, so the gain comes from fewer padded
  products per barrier. E8 (32-slot tiles) found no gain; repeat it on the
  recorded counts before building anything new.
- **IQ2_S and IQ2_XS downs** run 5-10% slower than IQ2_XXS down per call. If
  their decode is the gap, a 2% prefill gain.
- **The 140-row replay calls** run at 4-7 TFLOP/s (0.25 s per prompt). With
  840 routes over 384 experts most tiles hold 1-4 routes, so each call reads
  nearly every routed expert's weights once (an estimate; count the live
  tiles to confirm), which makes it a weight-read kernel. Small now that a
  prompt runs one replay.

### 4. Dense FP16 projections (~3.8 s, ~18%)

`kernels/rocm/ds41_dense_f16.hip.cpp`. `attn_q_b` (27.8 TFLOP/s) and
`attn_output_a` (28.7) are the weakest:
- E23: the grouped `attn_output_a` input is re-read by each 256-row block,
  and dropping the activation loads took it from 5.50 to 3.96 ms.
- `attn_q_b` measured 4.19 ms in `ds41_mmq_bench --dense` but 6.19 ms in
  the served model. Earlier sessions put the model at 5.6-5.8 ms; E68
  records a 41 ms layer-7 outlier.

Already measured:
- E52: Q6_K occupancy, Q8_0 path, paired launches.
- E53: rejected fragment prefetch, LDS fragments, paired loads.
- E23: FP16 inputs.

Tools: `ds41_mmq_bench --dense [--x-half 1]` and
`tools/bench/ds41_dense_f16_tune.hip`.

### 5. BF16 router (0.57 s, ~2.8%, before E75)

In the profile above, `matmul_bf16_wmma_kernel<128>` on 384 rows launches
only 3 × 32 = 96 workgroups for 2,048 tokens and reaches 5.1 TFLOP/s per
pass of the matrix (15 across its three terms). It runs twice per encoder
layer: routing, plus the wide next-layer prediction.
- E75 moved the router and compressor (≤ 512 rows) to E71's narrow kernel
  with 32-token tiles above 64 tokens: 1.27 → 1.16 ms and 1.64 → 1.43 ms
  per call in a warm microbenchmark, bit-identical. About 0.05 s per 20K
  prompt; reprofile the served document to see the per-call time there.
- 64-row tiles of the wide kernel were slower for all three shapes (E74).
- Still open: split K with an FP32 reduction in the same order, which
  keeps the sums bit-identical (E71: the 32-value K steps, the part order
  and the FP32 sums per step must stay).

### 6. Attention (1.64 s, ~8%)

`attention_f16_kernel`, 9.1 ms per encoder layer per 2,048 rows on the
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

### 8. Engram row reads (0.37 s, ~1.8%)

Two Engram layers per 2,048-row batch wait ~20 ms each for host row
reads before the projection. Prefetching the rows one layer early would
hide this. E26 already moved the rows into pinned staging.

## Things that look like levers but are not

- **Decoder replays at cache checkpoints:** fixed in E73 (one replay per
  prompt, −2.4 to −2.7 s per fresh 14-24K-token document).
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
