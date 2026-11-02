# DeepSeek V4.1 Flash prefill: where the time goes

A starting point for prefill work on the streamed `UD-IQ2_XXS` build
(146 GiB of routed experts, ~95 GiB expert cache, tails sidecar on a second
NVMe): the measured budget, each kernel's position against the gfx1151
ceilings, and the levers by measured size. Open work is ranked in
[BACKLOG.md](BACKLOG.md). Decode is in [DECODE.md](DECODE.md), the expert
store in [STREAMING.md](STREAMING.md), every `E<n>` in
[EXPERIMENTS.md](EXPERIMENTS.md).

The profile was measured 2026-11-02 on `feat/ds41-ud-iq2xxs` at `984ad880`
([artifacts](artifacts/2026-11-02-prefill/)). E82-E85 since changed the
read stalls and Engram waits (levers 1, 8); the kernels are unchanged.

## Summary

A fresh 19,927-token document served over HTTP took 20.59 s (968 tok/s)
in the profile and 19.2 s after E82; `gufo bench` pp2048 at 8K depth gave
1,100-1,160 tok/s, 1,160-1,162 after E82. The served gap to the bench has
three causes, each a lever below: encoder expert read stalls (lever 1),
the decoder replay's cold experts (lever 2), and real text spreading its
routes over more experts: 24-26 ms per 2,048-row routed gate+up against
17.8-18.5 ms on the bench prompt (lever 3).

## Workloads and how they were measured

| Workload | Command | Result |
|---|---|---|
| Bench | `gufo bench -m MODEL --stream-experts -p 2048 -n 0 -d 8192 -r 3` | 1,098.6 ± 50.1 tok/s profiled; 1,150-1,160 unprofiled |
| Served document | `gufo serve ... llm --model MODEL --context 65536 --stream-experts --speculative dspark --dspark-model DRAFT --sessions 1`, then two fresh documents with `max_tokens: 1` ([two_docs.py](artifacts/2026-11-02-prefill/two_docs.py)); the second is analysed | 23,954 tokens in 26.54 s, then 19,927 tokens in 20.60 s |

The documents are long Gufo sources (HTTP server and KV cache
documentation, `expert_store.cpp`), 14-24K tokens each; any real document
of that size works. Profile with `rocprofv3 --kernel-trace
--memory-copy-trace --output-format rocpd` and `tools/prof/prof.py show
... --stages ds41` (commands under "Reproducing"); the trace adds no
measurable time. Attribute each idle gap to the kernel after it: before
`routed_counts_kernel` it is the expert store's prepare, before the Q6_K
`dense_f16_kernel` with 100 row blocks the Engram row reads. The routed
kernel's `grid_y` tells batches apart: 576 for 2,048 rows, 523 for the
last encoder batch, 396 for the 140-row replay. Served runs drift (~250-320
ms per fresh-server run on the follow-up), so alternate builds and fit run
order with the build (E83).

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
- **Bounded replay** (`DS41_DECODER_REPLAY_ROWS`, `model.h`): encoder layers
  0-19 run every row, decoder layers 20-39 only the prompt's last 140 rows;
  layer 20 still compresses every position. The server passes the whole
  prompt's length to every step (`TextRunnerState::prefill_prompt_end`),
  so steps ending at cache checkpoints skip the decoder layers (E73).

## Ceilings

| Resource | Measured ceiling | Source |
|---|---:|---|
| FP16/BF16 WMMA | 55.4 TFLOP/s | `tools/bench/gfx1151_peak.hip` |
| DRAM read | 240 GB/s | same |
| Expert reads from two NVMe drives | ~10.4 GB/s (0.38 ms + 0.10 ms/MiB fitted) | `tools/ds41/prefetch_sim.py` read-cost fit on the served traces |

## Served budget: one 19,927-token fresh prompt

Profiled at `984ad880`, before E82-E85.

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

**The decoder replay.** Captures for this request: 4,096, 8,192, 12,288,
18,432 (grid checkpoints), 19,924 (stable boundary), 19,927 (prompt end),
each ending a prefill step. Only rows from 19,787 on run layers 20-39: 137
at the end of the stable step, 3 in the final step. The grid checkpoints
hold no decoder rows (E73, [BACKLOG item 36](BACKLOG.md#36-decoder-rows-for-short-continuations-of-mid-prompt-checkpoints)).
The replay costs ~0.3 s of GPU work and 0.91 s of read stall (45.4 ms per
layer): its 840 routes per layer land on decoder-layer experts the encoder
batches never touch. Before E73 the earlier replays at each checkpoint had
warmed them: the five replays then averaged 29.5 ms per layer.

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
`984ad880`, same kernels). Both prompts run 12,288 routes per layer; the
difference is how they spread over experts (lever 3).

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
its routed kernels run faster and its decoder experts are warm.

## Levers, by measured size

### 1. Expert read stalls in encoder batches (2.1 s, ~10%)

~10.5 ms per 2,048-row layer in the profile, with wide prediction on
(E68). Since then two-wave layers run the resident experts' gate+up and
down during the reads (E82, E84: pp2048 1,109 → 1,160 tok/s, served −1.7 /
−1.2 s, then −1.2%), and layer-0 prediction cut encoder read idle 1.53 →
0.96 s over two documents (E83). After E84, 819 ms remain over the two
documents: ~0.7 ms of host work per layer after the reads and 11 of 400
layers whose reads outlast the resident work. See BACKLOG items
[30](BACKLOG.md#30-more-memory-for-the-expert-cache) and
[31](BACKLOG.md#31-eviction-scoring-and-prompt-admission).

### 2. Cold experts in the decoder replay (0.91 s, 4.4%)

45.4 ms per replay layer (~42 ms after E85); the 840 routes spread over
most of the 384 experts as many small misses. Before E73, 5.1 GiB of a
replay's 9.3 GiB of simulated misses were experts evicted earlier in the
session, mostly by the prompt's encoder admissions, and 2.9 GiB never
resident. Read-ahead from layer 19 recovers only 14-15%, because a replay's
~8.5 GiB is bound by drives the encoder reads already keep ~70% busy, and
protecting decoder experts costs ~4 GiB of prompt reads per GiB (E86).
What remains is capacity:
[BACKLOG item 30](BACKLOG.md#30-more-memory-for-the-expert-cache).

### 3. Routed FP16 kernels (7.8 s busy, 46%)

`kernels/rocm/ds41_routed_f16.hip.cpp`, `routed_f16_kernel<type, pair,
half_out>`: 64-slot tiles per expert planned on the GPU (E7), 8 waves,
weights as the B operand decoded to FP16 with scales folded in,
activations double-buffered in LDS. E25's ablation of a 2,048-token IQ1_M
gate+up (21.8 ms): 14.7 ms skeleton (block fields, fragment reads,
products, ~39 TFLOP/s), +1.0 ms weight loads, +1.7 ms activation staging
(+4.9 for IQ2_XXS), +4.4 ms weight decode with its permutes.

**Use served routes.** `tools/ds41/route_occupancy.py` turns a route trace
into `--counts` lines; `ds41_mmq_bench --f16 1 --type 16 --down-type 22
--counts FILE --line 10` reproduces the served times (IQ2_XXS gate+up
23.2 ms, IQ2_S down 12.9 ms). Over the session's 520 prompt batches a
batch fills 419 64-slot tiles with 46% of slots holding a route (E74).
Fuller tiles are closed by arithmetic: the live 16-slot group count does
not change with tile width, and cross-expert groups would raise decode work
2.26× and re-read weights (E81).

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
| Activation-fragment sharing off for the downs | IQ2_S down 12.91 → 13.25 ms, IQ2_XS 12.07 → 12.35 | E81 |
| Cross-expert groups; wave64 | Ruled out by arithmetic and by the builtin signature | E81 |

What remains is the per-type weight decode (IQ2_S and IQ2_XS downs run
5-10% slower than IQ2_XXS) and the 140-row replay calls at 4-7 TFLOP/s
(0.25 s per prompt; most tiles hold 1-4 routes, likely weight-read bound,
unconfirmed): [BACKLOG item 6](BACKLOG.md#6-routed-fp16-kernel-weight-decode).

### 4. Dense FP16 projections (~3.8 s, ~18%)

`kernels/rocm/ds41_dense_f16.hip.cpp`; `attn_q_b` (27.8 TFLOP/s) and
`attn_output_a` (28.7) are the weakest. `attn_q_b` takes 5.61 ms in
`ds41_mmq_bench --dense` against 6.19 ms served; four rotated weight copies
change nothing (5.60 vs 5.62 ms, `tools/bench/ds41_qb_rotate.hip`), so the
gap is contention with expert streaming and outlier layers (E68's 41 ms
layer 7). E23, E52 and E53 exhausted the tile and load knobs; no open
item. Tools: `ds41_mmq_bench --dense [--x-half 1]`,
`tools/bench/ds41_dense_f16_tune.hip`.

### 5. BF16 router (0.57 s, ~2.8%, before E75)

`matmul_bf16_wmma_kernel<128>` on 384 rows launches only 96 workgroups for
2,048 tokens (5.1 TFLOP/s per pass), twice per encoder layer (routing and
prediction). E75's narrow kernel takes 1.27 → 1.16 ms per call warm, about
0.05 s per 20K prompt, not yet reprofiled served; 64-row wide tiles were
slower (E74). Split K: [BACKLOG item 7](BACKLOG.md#7-remaining-prefill-glue).

### 6. Attention (1.64 s, ~8%)

`attention_f16_kernel`, 9.1 ms per encoder layer per 2,048 rows on the
document (8.8 ms at 8-14K in the bench), growing with depth. E21 staging is
retained, E53 found no fixed cost, E19/E20 sped up the indexer. FP16 Q:
[BACKLOG item 40](BACKLOG.md#40-fp16-q-for-prefill-attention).

### 7. Hyper-connection state passes (1.09 s, ~5%)

`hc_state_pass_kernel` reads the 4 × 5,120 FP32 state once per token, with
the FFN's closing expand fused in (E22), near the DRAM limit. Only fewer
passes or a smaller state type, which changes numerics, remain; no open
item.

### 8. Engram row reads (0.37 s, ~1.8%, before E83/E85)

The two Engram layers per batch waited for host row reads (pinned staging
since E26). E83 reads layer 1's rows before layer 14's (1.55 → 1.07 s over
two documents); E85 reads each batch's rows during the previous batch
(1.21 → 0.50 s; −2.1% on a 58K-token prompt). Only a prompt's first batch
still waits.

## Things that look like levers but are not

- **Decoder replays at cache checkpoints:** fixed in E73.
- **Prediction lead time** for short passes: one-ahead predictions land
  7.7 ms before the layer, more than its 3.2 ms of reads; two or three
  ahead catch fewer misses (`prefetch_sim.py`).
- **Undrained adoption and always-on prediction** for short passes (E72).
- **Decoder-replay read-ahead and protection** (E86, lever 2).
- **Expert repacking on disk** (E30, E70;
  [BACKLOG item 35](BACKLOG.md#35-rank-ordered-expert-layout)).
- **GPU clocks during read waits** (BACKLOG item 32, closed).

## Reproducing

```sh
# Release build
cmake --build --preset release --parallel 4
M=/path/DeepSeek-V4.1-Flash-1017-UD-IQ2_XXS-00001-of-00006.gguf

# Bench profile
nix develop -c rocprofv3 --kernel-trace --output-format rocpd -d prof -o t -- \
  build/release/gufo bench -m $M --stream-experts -p 2048 -n 0 -d 8192 -r 3
python3 tools/prof/prof.py show prof/t_results.db --passes 200 --stages ds41

# Served profile: start the server under rocprofv3 (add --memory-copy-trace),
# send two fresh documents (artifacts/2026-11-02-prefill/two_docs.py), stop
# the server, then take the second request's pass:
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
