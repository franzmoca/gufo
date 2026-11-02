# DeepSeek V4.1 Flash experiments

Everything measured on the DeepSeek V4.1 Flash engine, retained or rejected,
with its evidence. Open work is in [BACKLOG.md](BACKLOG.md); the streaming
design is in [STREAMING.md](STREAMING.md); quality checks are in
[QUALITY.md](QUALITY.md) and the benchmark card in
[BENCHMARKS.md](BENCHMARKS.md).

Target: DeepSeek-V4.1-Flash-1017 with the
[ISTA-DASLab GSQ-RCO GGUFs](https://huggingface.co/ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF)
on one Strix Halo (gfx1151, 124 GiB GTT). The engine is
`src/models/deepseek_v41_flash/`, a fork of the V4 Flash engine (unchanged);
`tools/ds41/` holds an independent CPU/torch oracle that ports the official
`inference/model.py` and `inference/engram.py` and reads the same GGUF.

## Index

Every numbered experiment, in order. E1 is a row of the
[post-M6 table](#post-m6-resident-optimization-pass-2026-10-26) rather than a
heading (its raw record is
[artifacts/2026-10-26-rescore](artifacts/2026-10-26-rescore/README.md)); E2–E50
are subsections of the sections they were run under, E51 onward top-level
sections. No other file holds experiment entries; QUALITY.md has a quality
section for E75. Unnumbered work is in
[the resident engine table](#resident-engine-m1m5) (M1–M5), the SSD streaming
subsections from [storage reads](#storage-reads) to
[DSpark on streamed IQ2_XXS](#dspark-on-streamed-iq2_xxs-2026-10-27), and
[prefill at depth](#prefill-at-depth-2026-10-27). Outcomes: retained (in
production), rejected (built or measured, then dropped), measured (a
measurement or probe with no code change kept), simulated (trace replay
only).

Measurements first recorded in the backlog, without an E-number, are in
[backlog measurements without an E-number](#backlog-measurements-without-an-e-number).

| Entry | Area | Outcome | Result |
| --- | --- | --- | --- |
| [E1](#post-m6-resident-optimization-pass-2026-10-26) | decode kernels | retained | Compact candidate blocks for FP32 indexer rescoring past 16K: tg128 at 128K 21.10 → 21.89 tok/s (+3.7%), at 32K +1.2%; selected indices and order unchanged. |
| [E2](#e2-stochastic-dspark-proposals-retained-with-confidence-gated-verification) | DSpark/speculation, sampling | retained | Sampled proposals from the post-Markov top eight, verified while q's best ≥ 0.6: +3.85% geomean over 12 sampled pairs (a single-seed first trial was rejected, then re-measured). |
| [E3](#e3-request-local-bpe-piece-memoization-retained) | tokenizer | retained | Request-local BPE piece memoization: identical token IDs, probe 172.53 → 149.19 ms on the documentation corpus; no model-level gain claimed. |
| [E4](#e4-stop-unused-dspark-markov-suffixes-retained) | DSpark/speculation | retained | Markov chain stopped at the used proposal count: one-proposal draft ~11.7 → 8.9 ms; aggregate throughput flat. |
| [E5](#e5-score-only-requested-draft-head-rows-retained) | DSpark/speculation | retained | Draft head scores only the requested rows: +0.33% geomean over 12 pairs, identical tokens. |
| [E6](#e6-explicit-fp32-index-scoring-for-verification-retained-for-numerics) | DSpark/speculation, quality | retained | Verification widths 5–7 keep serial FP32 index scoring: every row matches serial decode; no throughput gain (32K decode ~1% slower in one pair). |
| [E7](#e7-stable-device-route-plans-for-fp16-layers-retained-groundwork) | prefill pipeline | retained | Route plans built on the device for 38 of 40 FP16 layers: 38 host round trips removed, throughput flat. |
| [E8](#e8-32-slot-routed-fp16-tiles-rejected) | prefill kernels | rejected | 32-slot routed FP16 tiles: pp2048 1243.8/1247.4 → 1232.9/1244.3 tok/s, no gain. |
| [E9](#e9-fuse-the-markov-projection-into-its-selection-retained) | DSpark/speculation | retained | Markov W2 projection fused into its selection: draft 12.6 → 10.0 ms; +2.09% over E2, +5.26% over the baseline (12 sampled pairs). |
| [E10](#e10-fp32-activation-gemv-for-the-drafters-q4_k-projections-retained) | DSpark/speculation | retained | Drafter Q4_K projections on the FP32-activation GEMV: draft 10.0 → 9.3 ms; end to end +0.41%, within noise. |
| [E11](#e11-learned-dspark-acceptance-head-for-greedy-cycles-retained-for-greedy) | DSpark/speculation | retained | Learned acceptance head with a ≥ 0.5 cut: greedy +13.07% geomean (12/12); sampled −1.0%, not used there. The cut was later replaced by E18. |
| [E12](#e12-device-target-draws-for-sampled-dspark-verification-retained) | DSpark/speculation, sampling | retained | Sampled verification draws on the device: +1.99% geomean (12/12), identical outputs. |
| [E13](#e13-one-launch-set-for-all-verification-rows-rejected) | DSpark/speculation, sampling | rejected | One launch set for all verification rows: −0.50% geomean; per-row draws kept. |
| [E14](#e14-joint-evaluation-of-concurrent-dspark-sessions-retained) | concurrency | retained | Joint evaluation of concurrent DSpark sessions: per-user greedy decode +40% at C2, +60% at C4, bit-identical rows. |
| [E15](#e15-16-row-single-session-verification-retained) | DSpark/speculation | retained | Single-session verification up to 16 rows (15 lookup copies): repeated-source rename 58.5 → 67.6 tok/s (+15.6%). |
| [E16](#e16-4096-token-prefill-batches-after-bounded-replay-rejected-as-default) | prefill pipeline | rejected | 4096-token batches as default after bounded replay: +2.0–3.1% natively, but served the prompt cache loses its backing (970.3 vs 968.5 tok/s); stays an option. |
| [E17](#e17-streamed-iq2_xxs-reader-and-mixed-quant-kernels-retained) | streaming/reads, prefill kernels | retained | Four direct readers, IQ2_XS FP16 routed kernel and shared-expert overlap: cold pp2048/tg128 144.5/5.77 → 219.6/9.02 tok/s. |
| [E18](#e18-cost-aware-draft-prefix-retained) | DSpark/speculation | retained | Cost-aware draft prefix: streamed IQ2_XXS +3.76% (10/10), resident IQ1_M +0.65% (8/10). |
| [E19](#e19-two-pass-indexer-top-k-retained) | prefill kernels | retained | Two-pass indexer top-k: 102.3 → 4.7 ms per index layer at 128K, byte-identical. |
| [E20](#e20-register-resident-keys-for-the-wmma-indexer-scores-retained) | prefill kernels | retained | Register-resident keys for the WMMA indexer scores: 84.5 → 39.9 ms per layer at 128K. |
| [E21](#e21-prefill-attention-staging-retained) | prefill kernels | retained | Prefill attention staging: 10.52 → 8.48 ms per layer at 128K, byte-identical. |
| [E22](#e22-the-ffns-closing-expand-fused-with-the-next-state-pass-retained) | prefill kernels | retained | FFN closing expand fused with the next state pass: 3.40 → 2.71 ms per layer; pp2048 at d16384 1,225.6 → 1,229.1 tok/s. |
| [E23](#e23-fp16-inputs-for-the-dense-fp16-projections-retained) | prefill kernels | retained | FP16 inputs for the dense FP16 projections: pp2048 at d16384 1,234.4 → 1,281.4 tok/s (+3.8%), bit-identical. |
| [E24](#e24-half-wave-permutes-without-register-copies-retained) | prefill kernels | retained | Half-wave permutes without register copies: pp2048 at d16384 1,284.9 → 1,307.6 tok/s (+1.8%). |
| [E25](#e25-routed-fp16-staging-and-lds-lookups-retained) | prefill kernels | retained | Routed FP16 double-buffered stages, IQ2 sign bytes and IQ2 fragment permutes, each byte-identical: IQ2_XXS gate+up 24.46 → 22.72 ms from the sign bytes alone. |
| [E26](#e26-engram-staging-fp16-q_b-rows-and-128-row-bf16-tiles-retained) | prefill pipeline | retained | Full-batch pinned Engram staging (pp2048 at d16384 +1.7%), FP16 `q_b` rows, 128-row BF16 tiles (router 1.25 → 0.98 ms). |
| [E27](#e27-one-joint-verification-for-concurrent-dspark-sessions-retained) | concurrency | retained | One 16-row joint verification per step: card mixed text C4/C8 42.6/44.7 → 51.4/57.8 tok/s. |
| [E28](#e28-engram-row-reads-under-a-full-expert-cache-rejected) | streaming/reads | rejected | Engram reads under a full expert cache: skipping them all gains only +0.8% warm, +3.6% on the first batch; not pursued. |
| [E29](#e29-6-column-dense-and-grouped-expert-gemvs-retained) | decode kernels | retained | 6-column dense and grouped expert GEMVs: 5–6-row verifications 0.5–0.8% faster. |
| [E30](#e30-expert-contiguous-repack-read-side-rejected) | streaming/reads | rejected | Expert-contiguous repack: ~1% per expert span; only long runs of neighbouring experts gain (+36%), not worth a conversion. |
| [E31](#e31-next-layer-prediction-for-verification-batches-retained) | streaming/reads | retained | Next-layer prediction for verification batches: first streamed DSpark tg128 +3.5%/+6.7%, warm tests unchanged. |
| [E32](#e32-batched-paths-for-7--and-8-token-prompt-batches-retained) | prefill pipeline | retained | 7- and 8-token prompts on the batched paths: pp8 121.8 → ~99 ms. |
| [E33](#e33-pinned-embedding-staging-and-threaded-dequantization-retained) | prefill pipeline | retained | Pinned embedding staging and threaded dequantization: embedding stage ~6.0 → ~4.5 ms per batch; throughput within noise. |
| [E34](#e34-dense-rescoring-left-on-the-bounded-replay-measured-not-changed) | prefill kernels | measured | Dense rescoring on the bounded replay costs ~25 ms per 128K prompt (~0.03%); left unchanged. |
| [E35](#e35-128-slot-routed-fp16-tiles-rejected) | prefill kernels | rejected | 128-slot routed FP16 tiles: slower on both types (IQ2_XXS gate+up 22.33 → 23.83 ms on line 0). |
| [E36](#e36-fused-swiglu-and-q8_0-pairs-through-the-split-gemv-retained) | decode kernels | retained | Fused SwiGLU and Q8_0 pairs through the split GEMV: SwiGLU at 16 tokens 210 → 155 µs; 16-row joint pass 177.7 → 174.4 ms. |
| [E37](#e37-grouped-expert-launches-split-by-group-size-retained) | decode kernels | retained | Grouped expert launches split by group size: server C8 AR 57.04 → 60.71 tok/s (+6.4%). |
| [E38](#e38-whole-row-prefetch-for-grouped-down-projections-retained) | decode kernels | retained | Whole-row prefetch for grouped down projections: server AR C4 +5.9%, C8 +2.0%. |
| [E39](#e39-joint-step-boundary-without-per-session-syncs-retained) | concurrency | retained | Joint-step boundary without per-session syncs: 4.75 → 3.33 ms per boundary; server rates within noise. |
| [E40](#e40-one-token-per-wave-in-the-bf16-decode-matmul-retained) | decode kernels | retained | One token per wave in the BF16 decode matmul: router at 8 tokens 83.6 → 38.3 µs; server C8 +1.5%. |
| [E41](#e41-split-gemv-scratch-for-5-8-row-output-heads-retained) | decode kernels | retained | 32 MiB split-GEMV scratch for 5–8-row output heads: 8 rows 2,918 → 2,190 µs. |
| [E42](#e42-one-drafter-pass-for-concurrent-dspark-sessions-retained) | concurrency | retained | One drafter pass for concurrent DSpark sessions: C4 +4.3%, C8 +7.3%. |
| [E43](#e43-joint-rows-by-draft-confidence-across-sessions-rejected) | concurrency | rejected | Joint rows by draft confidence across sessions: C8 −1.1% at the ≥ 0.5 rule. |
| [E44](#e44-graph-replay-of-dependent-decode-launches-measured-not-pursued) | decode pipeline | measured | Graph replay of dependent launches saves 0.12–0.15 µs per launch (~0.3% of a token); not pursued. |
| [E45](#e45-device-copies-batched-into-one-launch-retained) | decode pipeline | retained | Device copies batched into one launch: server AR C8 61.69/60.15 → 63.67/61.86, DSpark C8 54.53/53.82 → 58.14/55.43 tok/s. |
| [E46](#e46-paired-rows-for-single-token-iq1-down-projections-retained) | decode kernels | retained | Paired rows for single-token IQ1 downs: IQ1_S 86.0 → 78.0 µs; AR +0.5%. |
| [E47](#e47-exact-markov-candidate-pruning-measured-not-pursued) | DSpark/speculation | measured | Exact Markov candidate pruning would still score 99.2% of tokens; not pursued. |
| [E48](#e48-grouped-routed-experts-in-the-drafter-retained) | DSpark/speculation | retained | Grouped routed experts in the drafter: drafter expert kernels 2.06 → 1.70 ms per step. |
| [E49](#e49-collapse-norm-folded-into-the-hyper-connection-mix-rejected) | decode kernels | rejected | Collapse-norm folded into the hyper-connection mix: 22.1 µs against 13.7 + 4.3 in the model; AR 24.35 → 24.22 tok/s. |
| [E50](#e50-verification-argmax-on-the-device-retained) | DSpark/speculation | retained | Verification argmax on the device: greedy DSpark 39.77/39.74 → 40.00/40.04 tok/s. |
| [E51](#e51--iq1-single-slot-experts-leave-the-grouped-kernel-retained) | decode kernels | retained | IQ1 single-slot experts on the ungrouped kernel: IQ1_M gate/up 717 → 609 µs (synthetic); server C8 greedy DSpark 60.71/60.33 → 61.79/60.75 tok/s. |
| [E52](#e52--dense-fp16-prefill-q6_k-occupancy-q8_0-paired-launches-retained) | prefill kernels | retained | Dense FP16 kernel: Q6_K occupancy, Q8_0 support, paired launches: UD-IQ1_M pp2048 1,135 → 1,278 tok/s. |
| [E53](#e53--prefill-fragment-loads-and-attention-fixed-costs-rejected) | prefill kernels | rejected | Dense fragment-load variants all slower (`q_b` 4.19 → 5.02–30 ms); FP16 Q for attention estimated at ~1% of prefill, not attempted. |
| [E54](#e54--decode-global-expert-loads-fp8-output-in-the-split-combine-retained) | decode kernels | retained | Global instead of flat expert weight loads (streamed UD-IQ1_M tg128 20.15/20.32 → 20.59/20.61 tok/s) and FP8 output in the split combine. |
| [E55](#e55--decode-hc-mix-kernel-rejected) | decode kernels | rejected | HC mix with 2–6 rows per block: 14.2–15.6 µs against 13.98; at most ~3 µs per call above the read. |
| [E56](#e56--decode-shared-expert-during-the-streamed-route-read-retained) | decode pipeline | retained | Shared expert queued during the streamed route-id copy: tg128 about +2%. |
| [E57](#e57--decode-hc-mixes-on-a-side-stream-retained) | decode pipeline | retained | HC mixes on a side stream: tg128 about +0.6% at the median. |
| [E58](#e58--multi-token-q6_k-and-q8_0-dense-gemvs-on-the-task-major-kernel-retained) | decode kernels | retained | Multi-token Q6_K and Q8_0 dense GEMVs on the task-major kernel: streamed UD-IQ1_M C8 63.81 → 66.83 tok/s (+4.6%). |
| [E59](#e59--decode-router-loads-in-flight-routing-without-argmax-rounds-gemv-probes-retained-gemv-probes-measured) | decode kernels | retained | Router loads in flight, routing without argmax rounds: ~0.24 ms per token; GEMV probes not pursued (`wo_b` 140 vs 125 µs standalone, cause open). |
| [E60](#e60--launch-gaps-after-long-kernels-are-a-tracing-artifact-measured) | tooling/profiling | measured | 6–11 µs gaps after long kernels are added by kernel tracing (2.0–2.1 µs untraced); every-layer prediction read 30% more with no decode gain. |
| [E61](#e61--expert-reads-split-across-two-drives-retained) | streaming/reads | retained | Expert reads split across two drives: one cold miss 1.32–1.47 → 0.87 ms; streamed IQ2_XXS tg +7.6%, pp512 +16%. |
| [E62](#e62--dspark-drafter-size-and-quantization-measured-current-drafter-kept) | DSpark/speculation | measured | Other drafter sizes and quantizations do not beat Q2_K/Q4_K (37.18 replayed tok/s); the Unsloth confidence head is now loaded. |
| [E63](#e63--dspark-drafter-no-host-sync-after-injection-f32-router-per-token-retained) | DSpark/speculation | retained | No host sync after feature injection, per-token F32 drafter router: drafting 8.5 → 8.3 ms per cycle; multi-token HC mix rejected. |
| [E64](#e64--dspark-cycles-cut-to-zero-drafts-measured-not-pursued) | DSpark/speculation | measured | Zero-draft cycles: perfect foresight would gain 1.5% greedy, 3.9% sampled; no gate known before drafting finds them. |
| [E65](#e65--attention-partials-written-back-during-the-dense-gemvs-measured) | decode kernels | measured | Split-attention partials (2.64 MB per row and layer) delay the next dense GEMV by ~16–76 µs; 10 splits gain ~0.5% AR but change rounding; not retained. |
| [E66](#e66--multi-token-bf16-matmuls-for-verification-measured-not-retained) | decode kernels | measured | Multi-token BF16 matmuls for verification save 0.2–0.35 ms per cycle but change rounding; not retained. |
| [E67](#e67--iq3_xxs-and-q2_k-routed-downs-iq2_s-fused-gateup-retained) | decode kernels, prefill kernels | retained | IQ3_XXS and Q2_K routed downs and IQ2_S fused gate/up for UD-IQ2_XXS: IQ3_XXS down 126.8 µs (214 GB/s), IQ2_S pair 295 → 236 µs. |
| [E68](#e68--streamed-prefill-next-layer-prediction-for-mid-size-batches-retained) | streaming/reads | retained | Mid-size prompt batches predict once a layer missed: UD-IQ2_XXS warm pp2048 about +3.5%, first pp512 +11–15%. |
| [E69](#e69--decode-q8_0-q_b-gemv-in-four-slices-retained) | decode kernels | retained | Q8_0 `q_b` GEMV in four slices: about 0.17 ms per decode token. |
| [E70](#e70--served-coding-agent-on-streamed-ud-iq2_xxs-expert-layout-from-its-trace-measured) | serving, streaming/reads | measured | Pi agent on streamed UD-IQ2_XXS: 36.2 MiB read per evaluated token; a reordered expert file would save ~0.5 s of 38.6 s read wait. |
| [E71](#e71--narrow-bf16-matrix-core-tiles-for-short-prompt-passes-retained) | prefill kernels | retained | Narrow BF16 matrix-core tiles for 16–64-token passes: router 350 → 89 µs, pp20 +7%. |
| [E72](#e72--short-prompt-passes-earlier-prediction-and-undrained-adoption-measured-not-retained) | streaming/reads | measured | Earlier prediction and undrained adoption for short passes: 2–10 ms of a ~210 ms pp20 pass; not retained. |
| [E73](#e73--one-decoder-replay-per-served-prompt-retained) | serving | retained | One decoder replay per served prompt: fresh 24K-token document TTFT 28.87/28.71 → 26.11/26.13 s. |
| [E74](#e74--bf16-64-row-wide-tiles-rejected) | prefill kernels | rejected | BF16 64-row wide tiles: indexer 2.60 → 3.01 ms at 2,048 tokens. |
| [E75](#e75--grouped-bf16-routercompressor-prefill-retained) | prefill kernels | retained | Narrow BF16 kernel for router/compressor prefill: router 1.274 → 1.159 ms, compressor 1.640 → 1.425 ms; end to end inconclusive. |
| [E76](#e76--routed-fp16-128-value-activation-stages-rejected) | prefill kernels | rejected | Routed FP16 128-value activation stages: gate/up 23.18 → 23.45 ms, down 12.91 → 13.44. |
| [E77](#e77--routed-fp16-32-value-activation-stages-rejected) | prefill kernels | rejected | Routed FP16 32-value activation stages: gate/up 23.18 → 23.38 ms, down 12.91 → 13.45. |
| [E78](#e78--iq2_s-sign-masks-without-lds-lookup-rejected) | prefill kernels | rejected | IQ2_S sign masks without the LDS lookup: down 12.96–13.46 ms against 12.91. |
| [E79](#e79--routed-iterative-ilp-scheduler-rejected) | prefill kernels | rejected | Iterative-ILP scheduler for routed FP16: 23.07/12.89 against 23.18/12.91 ms, no meaningful gain. |
| [E80](#e80--iq2_s-grid-table-as-packed-bytes-in-lds-rejected) | prefill kernels | rejected | IQ2_S grid table as packed bytes in LDS: down 12.87–12.91 → 13.82–13.87 ms (+7%). |
| [E81](#e81--routed-fp16-down-fragment-sharing-off-and-why-tile-surgery-is-closed-rejected) | prefill kernels | rejected | Down-fragment sharing off: IQ2_S down 12.91 → 13.25 ms; narrower tiles and wave64 ruled out on arithmetic. |
| [E82](#e82--two-wave-prefill-the-resident-experts-gateup-during-the-read-stall-retained) | prefill pipeline | retained | Resident experts' gate+up during the read stall: pp2048 at d8192 1,109 → 1,160 tok/s; 18.5K probe −5.9%. |
| [E83](#e83--layer-0-prediction-encoder-batches-stop-at-layer-19-engram-rows-in-layer-order-retained) | prefill pipeline | retained | Layer-0 prediction, no prediction past layer 19, Engram rows in layer order: encoder read idle 1.53 → 0.96 s; fresh document −1.4%. |
| [E84](#e84--two-wave-prefill-the-resident-experts-down-in-the-read-stall-too-retained) | prefill pipeline | retained | Resident experts' down in the read stall too: encoder idle 963 → 819 ms; fresh document −1.2%. |
| [E85](#e85--engram-lookahead-the-next-prompt-batchs-rows-during-the-current-one-retained) | prefill pipeline | retained | Engram lookahead for the next prompt batch: Engram waits 1,213 → 500 ms; 58K-token prompt −2.1%. |
| [E86](#e86--decoder-replay-read-ahead-and-cache-protection-simulated-not-built) | expert cache | simulated | Decoder-replay read-ahead recovers only 14–15% of the replay stall; cache protection trades ~4 GiB of prompt reads per GiB of replay reads; not built. |
| [E87](#e87--greedy-dspark-learn-the-plain-step-price-the-draft-decide-before-drafting-retained) | DSpark/speculation | retained | Greedy policy learns the plain step and draft cost and decides before drafting: streamed code 26.6–27.4 → 28.3–28.7 tok/s. |
| [E88](#e88--top-p-softmax-totals-on-the-device-retained) | sampling | retained | Top-p softmax totals on the device: sampled DSpark long request 26.9 → 27.9–28.0 tok/s. |
| [E89](#e89--decode-prediction-every-layer-and-two-layers-ahead-rejected) | streaming/reads | rejected | Decode prediction every layer while misses persist: simulated +2.3–2.8%, measured 58.9 → 63.1 ms/token (+7.1%). |
| [E90](#e90--dense-gemv-slice-sweep-over-the-decode-shapes-measured-not-retained) | decode kernels | measured | Dense GEMV slice sweep: no format gap; two beatable splits worth ~0.06 ms per AR token, not worth a rounding change. |
| [E91](#e91--eviction-victim-lists-retained-pop-check-corrected-in-e92) | expert cache | retained, corrected by E92 | Eviction victim lists: 3K probe 58.85 → 56.15 ms/token; the served gains did not reproduce (E92). |
| [E92](#e92--exact-victim-list-pops-eviction-scoring-against-the-offline-optimum-retained-scoring-change-not-retained) | expert cache | retained; scoring change not retained | Exact victim-list pops restore baseline reads (63.1–64.4 MiB per token); Belady's optimum halves decode misses; decay 3,200/bonus 8 inconclusive, not retained. |

## Milestones

| Milestone | Result |
| --- | --- |
| M0 Oracle and loader | `tools/ds41/reference.py` (official KV quantization: FP8 SWA, FP4 compressed and indexer; chunked sparse attention with sinks, candidate blocks and Engram; dumps logits, routes and per-layer states; `verify-engram` matches the GGUF's hashing tables exactly) and `tools/ds41/quants.py` (IQ1_S, IQ1_M, IQ2_XXS, IQ2_S and IQ4_NL dequant, bit-exact with llama.cpp). Oracle on IQ1_M: 3K-token teacher-forced perplexity 6.31, top-1 58%. The loader maps the split GGUF as one range and validates all `deepseek41.*` metadata and 1,008 tensors; the tokenizer matches the official `tokenizer.json` on ~19k tokens. |
| M1 Correct AR decode on IQ1_M | Single-token decoder over plain kernels with typed MMVQ entries for every expert and dense type; sessions with prefix reuse and snapshots; the official V4.1 chat template pinned by SHA-256 (a GGUF with another template loads with a warning); `deepseek41` in `gufo serve` with DSML tools and the official sampling defaults (temperature 1.0, top-p 1.0). 3K perplexity 6.306 against the oracle's 6.309, 12/12 template goldens, decode 12–14 tok/s. |
| M2 Resident performance | `gufo prompt`/`bench` support, candidate-block parity beyond 16K, batched layer-major prefill on the FP16 matrix cores, overlapped Engram reads (0.04 ms wait per prefill batch, 0.33 ms per decode token): pp2048 703 and tg64 23.9 tok/s at depth 0, 372 and 21.0 at 131K. Steps in [the resident engine table](#resident-engine-m1m5). |
| M4 Component cache and serving | Compressed KV and index K as append rows (ratio-2 layers add a row every 2 tokens); SWA rings, compressor frontiers and logits as private state; the Engram n-gram history rebuilt from the tokens on restore. Disk cache and prefix reuse as for Gemma 4. |
| M5 DSpark, prompt lookup, bounded replay | `--dspark-model` loads the unsloth (`dflash`) or Lucebox (`deepseek41-dflash-draft`) layouts; drafts of up to 5 tokens verified with decode numerics, so output is unchanged: greedy +21% on a document prompt, +61–67% on a coding request. Prompt lookup copies up to six (later 15) tokens: file edit 23.6 → 54.6 tok/s without a drafter, 43.9 → 56.4 with DSpark. Decoder SWA bounded replay: pp2048 703 → 1,177 tok/s, 372 → 855 at 131K, continuation NLL within 0.008 of exact prefill (`--exact-prefill` keeps every row). Not used: unsloth BF16 (7.92 GiB), Lucebox `-MXFP4-Q8.gguf` (7.42 GiB), the peasantsmith `.bin` drafters (not GGUF). |
| M6 Vision | Opt-in `--mmproj` (0.90 GiB); the sidecar adds each MoE gate's image routing bias (`bias_vl`) from the official checkpoint (`tools/ds41/vision_sidecar.py`). Official preprocessing (padded pixels byte-identical to PIL), the 32-layer ViT and 3 × 3 aligner in the official BF16 rounding (`tools/ds41/vision_reference.py` is the CPU teacher); image positions take no Engram term and route with the image bias. [Quality](QUALITY.md#vision-m6), [speed](BENCHMARKS.md#vision-encoder). |
| M3 SSD expert streaming | `IQ2_XXS` streaming qualified for evaluation; [below](#ssd-expert-streaming-m3). |
| M7 Higher-precision Engram tables | Open ([backlog](BACKLOG.md)). |

The inherited V4 `runtime/session.cpp`, `runtime/rocm_graph.cpp`,
`dspark_internal.h` and `dspark_policy.h`, which the V4.1 target never
built, were removed on 2026-10-27; the active V4.1 runtime is
`v41_decode.cpp`, `v41_session.cpp` and `v41_dspark.cpp`.

## Measured limits

- Measured DRAM read peak (`tools/bench/gfx1151_peak.hip`): 235 GB/s. The
  dense decode projections run at 207-221 GB/s on the large shapes; decode
  spends 42.6 ms of GPU time per token at 2K (47.2 ms wall).
- Routed expert matvecs: a plain read of the same six DRAM-cold 2.3 MB
  experts streams at 230-238 GB/s (58 µs); the pre-October-27 expert GEMV
  reached 160-175 GB/s on IQ1 and 121-134 GB/s on IQ2, and mmvq 145-170
  GB/s, bound by the texture addresser (`TA_BUSY` ~92% of active cycles):
  every 32-value sub-block cost four 2-byte weight loads and eight
  activation loads per lane. The retuned kernels
  ([expert GEMV decode retune](#expert-gemv-decode-retune-2026-10-27))
  decode at 162-205 GB/s, the dense GEMV's level for these shapes.
- Streamed `IQ2_XXS` expert reads: 5.7 GiB/s effective, 8.2 GB/s for purely
  sequential reads on the 990 PRO ([storage reads](#storage-reads)).

## Resident engine (M1–M5)

IQ1_M resident on one Strix Halo (gfx1151). Speeds come from
`ds41_load_probe --batch` on the `gpu-test` build; prefill is a 2K-3K token
prompt, decode is greedy at 2K context. Quality gates: the 13-token smoke
continuation (same greedy tokens, KL against the oracle) and the 3K-token
teacher-forced score against the oracle's perplexity 6.309
([QUALITY.md](QUALITY.md)).

| Experiment | Decision / evidence |
| --- | --- |
| Layer-major batched prefill (512-token batches, MMQ for every quantized projection, IQ1_M added to the MMQ tiles) | Retained. 3K perplexity 6.300, argmax agreement 96.2%. Prefill 104.7 tok/s at depth 0 falling to 60.2 tok/s at 2.5K. Token-at-a-time prefill was ~13 tok/s. |
| WMMA prefill attention: 32 heads per block, half Q/K tiles, fp32 online softmax over window and indexed compressed rows | Retained. Attention 9.0 s → 0.55 s per 2K prefill + 16 decode steps; prefill 104.7 → 140.9 tok/s at depth 0, 60.2 → 89.5 at 2.5K. Perplexity 6.308. |
| Routed MoE launch bounds from the real expert buckets (largest bucket bounds the column-tile grid, tile width from the bucket distribution) | Retained. IQ1_S expert GEMM 6.1 → 2.9 s; prefill 140.9 → 198.9 tok/s at depth 0. Bounded and unbounded launches agree with the matvec path to the same 2e-4 rounding (`ds41.mmq_typed`). |
| WMMA indexer scores: 16 tokens × 128 compressed rows per block, half operands, fp32 relu-weighted head sum | Retained. Prefill no longer slows with depth: 200 tok/s flat to 3K (was 198.9 → 115.8). Perplexity 6.296. |
| Split-row decode attention: 20 row splits × 16 heads per block, rows staged once in LDS, fp32 online softmax, merge with the sink | Retained. Matches an fp64 reference to 1.6e-7 (`ds41.attention`). Decode at 2K 89 → 69.8 ms/token; decode-path 3K perplexity 6.307. |
| Sinkhorn with the hc = 4 matrix as a template constant | Retained. The runtime-sized matrix lived in scratch memory: 90 → 2.5 µs per call, decode 69.8 → 62.0 ms/token. |
| Non-finite guard folded into the matvec store, dropping the per-call clear and sanitize launches | Retained. Same values; 723 → 443 small launches per token; decode 62.0 → 58.7 ms/token. |
| Eight output-LoRA groups in one channel-batched matvec | Retained. Bit-identical at one token (7e-8 at three, from a different row tiling). Decode 58.7 → 56.3 ms/token. |
| Engram rows read on worker threads from the start of each evaluation, overlapping the layers before 1 and 14 | Retained. Bit-identical; decode 56.3 → 53.7 ms/token. |
| BF16 matmul with 16-byte weight loads (eight weights per lane per step) | Retained. Decode 53.7 → 50.5 ms/token, prefill 200 → 233 tok/s. Smoke KL unchanged in range. |
| Prefill batches of 1024 / 2048 tokens (from 512) | 2048 retained: 233 → 261 → 292 tok/s; experts average 32 rows per batch instead of 8. Activations take ~1 MiB per token; 6.8 GiB stay free after the decoder allocates. The context budget still lowers the cap at long contexts. |
| Batched hc mixes: two tokens and all 24 rows per block (state read once) | Retained for batches of 16+ tokens. Prefill 292 → 313 tok/s; perplexity 6.310. Decode keeps the per-row kernel (one block per token would idle the GPU). |
| Single-token Q5_K/Q8_0 projections through an FP32-activation GEMV forked from the Gemma 4 K-quant GEMV (no Q8_1 activation pass), eight waves split into 8 / S row groups of S slices | Retained. DRAM-cold bandwidth, mmvq → GEMV: q_b 190 → 221, wo_b 191 → 219, grouped wo_a 200 → 219, shared gate 179 → 210, shared down 172 → 207, wq_a 154 → 198, wkv 102 → 166 GB/s (measured DRAM read peak 235 GB/s). S = 2 for K ≤ 2304, 4 for K ≤ 5120 with M ≤ 4096, else 8. Matches an fp64 dot to 2e-7 (mmvq: ~5e-3 from Q8_1 activations). Decode 50.5 → 47.2 ms/token; decode-path 3K perplexity 6.303, agreement 96.1%, mean |ΔNLL| 0.077. |
| Single-token routed experts through an FP32-activation expert GEMV (IQ1_S, IQ1_M, IQ2_XXS, IQ2_S): IQ1 groups as d s (sum n x + (delta - 1) sum x), IQ1 codebook in LDS, IQ1_S loads two blocks ahead | Retained for IQ1 projections and IQ2 down projections; IQ2_XXS gate/up stay on mmvq (141 → 154 µs). DRAM-cold six-expert launches, mmvq → GEMV: IQ1_S gate/up 87.5 → 79.3 µs, down 94.1 → 85.2; IQ1_M 91.3 → 89.9, 106.1 → 95.9; IQ2_XXS down 208.5 → 147.2; IQ2_S down 224.5 → 158.7. Matches mmvq to its Q8_1 rounding (5e-3 to 9e-3, `ds41.mmq_typed`). Decode at 2K 47.1 / 47.8 → 46.5 / 46.8 ms/token (alternating runs); decode-path 3K perplexity 6.305, agreement 96.5%, mean \|ΔNLL\| 0.075. |
| Expert GEMV variants (same DRAM-cold launches) | Rejected, each slower than the retained kernel: mmvq with 2, 4 or 8 waves per row (IQ1_S gate/up 91-118 µs); activations held in registers across 2-8 rows per lane (110-150 µs); activations staged in LDS, with or without bank padding (100-300 µs); non-temporal weight loads (102 µs); 32-row tiles staged into LDS with dword loads and one row per lane (126-245 µs, register spills); per-weight decode to floats before the FMA (95-100 µs). |
| MMQ scratch pool keeps freed buffers for reuse on the freeing stream (was `hipMallocAsync`/`hipFreeAsync` per call: ~0.45 ms of host time per free, ~0.9 ms for the first allocation after each synchronize) | Retained. Same kernels and values (`ds41.mmq_typed`). `gufo bench` -p 512 -n 64 -r 3: tg64 21.83 → 22.17 t/s, tg64 @ d2048 20.96 ± 0.65 → 21.88 ± 0.01, pp512 303.6 → 308.3. |
| Engram rows of batches up to 16 tokens staged in pinned memory and uploaded asynchronously (the pageable upload waited for layer 0 to drain), four rows per read thread (was 24, so one thread per decode token) | Retained. Same generated tokens. Traced 2K decode: the ~0.5 ms stall before layer 1 is gone, step 48 → 46.7 ms with the pool change. `gufo bench` tg64 22.16 → 22.20, tg64 @ d2048 21.88 → 21.95 t/s. |
| hc collapse fused with the following RMSNorm (block of 1,024 threads per token, values kept in registers; the separate 256-thread norm was latency-bound at 10 µs per 5,120-value row), and the rope tail fused with the FP8 fake quantization of the KV latent and the attention output | Retained. 160 fewer launches per token. Rope + FP8 is bit-identical; the fused norm changes the sum order: decode-path 3K perplexity 6.291, agreement 96.0%, mean \|ΔNLL\| 0.073; batched 6.324, 95.9%, 0.078; smoke continuation keeps all 9 greedy tokens, KL ≤ 3.2e-3 per step. `gufo bench` -p 512 -n 64 -r 3: tg64 22.29 → 22.81, tg64 @ d2048 21.94 → 22.43 t/s, pp512 308.0 → 316.7, pp512 @ d2048 323.6 → 331.2. |
| hc mixes and Sinkhorn as one kernel below 16 tokens: each mix row split over 8 blocks with 16-byte loads, the last block of a token reduces the partial sums and runs its Sinkhorn | Retained. 80 fewer launches per token; the mix itself stays at ~14 µs (2 MB of F32 weights; 1, 4 and 8 blocks per row measured 14.1, 14.1 and 13.7 µs). Decode-path 3K perplexity 6.310, agreement 96.0%, mean \|ΔNLL\| 0.073; smoke continuation keeps all 9 greedy tokens, KL ≤ 5.3e-3. `gufo bench` tg64 22.75 → 22.95, tg64 @ d2048 22.43 → 22.54 t/s. Batched prompts keep the separate Sinkhorn kernel in its original form: routing it through the shared device function changed fast-math rounding and moved the batched 3K scores (6.324 → 6.321, agreement 95.9% → 95.4%, mean \|ΔNLL\| 0.078 → 0.082); with the original form they are bit-identical again. |
| Indexer top-k threshold by 8-bit radix select (four LDS histogram passes, the digit found in one wave) instead of a 32-step binary search with a block scan per step | Retained. Same threshold, so bit-identical (decode-path 3K scores unchanged); `ds41.candidate_mask` adds top-k cases with tied scores. Per call at 2K decode 45.3 → 6.9 µs; largest prefill call 2.5 → 0.34 ms. `gufo bench` tg64 @ d2048 22.55 → 22.74, tg64 @ d32768 20.91 → 21.55, pp512 @ d32768 257.0 → 298.0 t/s. |
| Indexer scores for up to 4 tokens in fp32 without matrix cores (the 16-token WMMA tile ran one live token and a block barrier per head): 64 keys per block staged by coalesced loads into padded shared memory, a thread per (key, 8 heads), groups summed in shared memory | Retained. Per call at 18.5K decode 195 → 74 µs (a thread per key with strided key loads: 112 µs). Matches fp64 to 2.4e-7 (matrix-core path 3.7e-4; `ds41.attention`). Decode-path 3K perplexity 6.311, agreement 96.0%, mean \|ΔNLL\| 0.073. `gufo bench` tg64 @ d2048 22.73 → 22.84, tg64 @ d32768 21.56 → 22.17 t/s. |
| Single-token gate and up projections with SwiGLU in one launch: a wave takes the same rows of both matrices (IQ1 routed experts, with the route weight; Q5_K shared expert) | Retained. Bit-identical (decode-path 3K scores unchanged; `ds41.mmq_typed` compares fused and unfused within 4.3e-8). 120 fewer launches per token; at 2K the routed pass is 144 µs against 153.5 µs for gate, up and SwiGLU, the shared pass 73.0 against 75.3 µs. `gufo bench` tg64 23.31 → 23.65, tg64 @ d2048 22.84 → 23.17 t/s. |
| Single-token q_a and KV projections (both Q8_0 over the same input) as one 1,792-row GEMV launch, and both RMSNorms plus the KV rope + FP8 quantization in one kernel | Retained. Bit-identical on the decode and batched paths once the norm keeps rms_norm's loop form (a register-array form rounded differently under fast-math: max 9.5e-7, batched agreement 95.4%); `ds41.attention` compares it with the separate launches. 120 fewer launches per token. `gufo bench` tg64 23.67 → 23.82, tg64 @ d2048 23.18 → 23.43 t/s. |
| Routed-expert sum (six down rows plus the shared expert) folded into the FFN hc expand | Retained. Bit-identical; 40 fewer launches. `gufo bench` tg64 23.81 → 23.88, tg64 @ d2048 23.43 → 23.47 t/s. |
| Attention output rope + FP8 quantization fused into the split-attention merge (head row staged in shared memory) | Rejected. 40 fewer launches but no measured gain (`gufo bench` tg64 @ d2048 23.47 → 23.48 ± 0.01 t/s), and the merge rounds differently in rare cases (3K decode path: 115 argmax changes, perplexity 6.311 → 6.309). |
| IQ2_XXS routed gate/up through the fused FP32 pair instead of two mmvq launches and SwiGLU | Retained. 252 µs against 251 µs for the three launches at 2K, so the gain is the launches (`gufo bench` tg64 23.84 → 23.94, tg64 @ d2048 23.46 → 23.57 t/s). Decode-path 3K perplexity 6.313, agreement 96.0%, mean \|ΔNLL\| 0.073. |
| Split decode attention with 10 or 40 row splits (20 retained) | Rejected. At 2K, split + merge per token: 20 splits 1,107 + 208 µs; 10 splits 1,312 + 94 µs (half the partial traffic, half the blocks); 40 splits 1,213 + 438 µs. |
| Routing kernel with one wave or four waves per token (eight retained) | Rejected. Per call at 2K decode: 8 waves 7.0 µs, 4 waves 7.4 µs, 1 wave 13.4 µs (12 serial softplus evaluations per lane). Same selections. |
| Prefill batch sized from free memory instead of context: indexer scores, candidate mask and top-k run in groups of up to 256 MiB of score rows, the candidate mask keeps one byte per 8-row block (was one per row, ~2 GiB at 1M context for 2048 tokens) | Retained. The batch had shrunk with context to stay under 256 MiB of score rows: 1,538 tokens at 35K, 409 at 131K, 51 at the native 1M. The same 2K prompt prefills at 344 tok/s with a prompt-sized context; at 131K 245 → 338 tok/s (2048-token batch), at 1M ~120 → ~255 tok/s (576-token batch, 1.7 GiB RAM left). `gufo bench` pp2048 @ d32768 330.5 → 349.3 t/s. Bit-identical batched 3K scores at a prompt-sized context; 18.5K scored at 1M context: perplexity 6.273, agreement 95.9% (97.2% in the masked 16,383-17,399 span). |
| Prefill glue: every MMQ type but Q8_0 zeroes non-finite outputs at the tiled store (was a whole-buffer sanitize pass after 642 of 722 launches per 2K prefill, 150 ms), and batched routed gate/up of one type run as one pair that quantizes the input once and stores SwiGLU × route weight from the up pass (was two routed launches, each quantizing and gathering the input, then a SwiGLU pass) | Retained. The store-side guard is bit-identical (batched 3K scores); the pair gives the same gate rows and SwiGLU within one ulp (4-5e-8, `ds41.mmq_typed` for all six types), which shifts the batched 3K run to perplexity 6.300 (was 6.324), agreement 96.1%, mean \|ΔNLL\| 0.081. `gufo bench` pp2048 380.2 → 395.5 t/s; tg unchanged. |
| Prefill batches of 4096 tokens (experts average 64 rows) | Rejected for now. 16K-token prompt 345.4 → 361.5 tok/s (+4.7%) for 2 GiB more resident activations (4.2 GiB RAM left at 16K context). Doubling the rows per expert cut the routed MMQ cost per token only 10-20% (IQ1_S 10.2 → 9.3 µs per token and launch). |
| Routed MMQ column-tile width forced to 16, 32, 48 or 64 (the bucket model picks 32 at 2048 tokens) | Rejected; 32 retained. Per 2048-token batch of a 4K prompt: 16 → 6.24 s, 32 → 5.80 s, 48 → 6.03 s, 64 → 6.20 s. Tile fill (real / issued columns) 82%, 67%, 56%, 48%: narrow tiles pad less but repeat the weight-panel unpack, about 40% of a 32-column tile's cost. The largest bucket (up to 1,537 of 2,048 tokens) sizes the column grid for every expert, but launch time is flat in it (11 → 50 column tiles: 20.5 → 21-23 ms), so empty blocks cost little. |
| Compressed KV and index-K rows stored as bfloat16 (FP4 values times an E4M3 or power-of-two scale have at most 6 significant bits, so the conversion is exact); the FP4 quantization writes the state directly instead of quantizing in place and copying | Retained. Bit-identical on the batched and decode 3K paths. State at the native 1M context ~6.3 → ~3.1 GiB: the 2K prompt there runs as one 2048-token batch at 342 tok/s (was 576-token batches, ~255), 3.1 GiB RAM left. Saved states change format (magic `S4STav2`). |
| Batched output LoRA as one channel-batched MMQ (8 groups as channels: one input quantization, one launch; was per group a row copy in, a quantization, a launch and a row copy out: 640 copies, 128 ms per 2K prefill), and q_a/KV and shared gate/up each quantizing their common input once (dense pair) | Retained. Bit-identical batched 3K scores; `ds41.mmq_typed` checks the grouped and pair launches against separate launches exactly for all six types. `gufo bench` pp2048 385.6 → 409.8 t/s against the binary before the bfloat16 state in the same session (395.5 in the previous session). |
| Routed MMQ tile width per expert: 16, 32 or 64 columns from each bucket's size (cost ceil(c / w) × (w + 21) column-equivalents, the panel term fitted from the fixed-width sweep), one launch per width over its experts only (the kernel skips other experts' blocks) | Retained. Prefill 2048-token batch 5.84 → 5.42 s; 18.5K-token prefill 56.9 → 50.3 s (with the channel-batched output LoRA). Tile widths change the matrix-core accumulation split: class-split launches match one launch to ≤1.3e-7 (`ds41.mmq_typed`, all six types, every class used); batched 3K perplexity 6.307 (oracle 6.309), agreement 95.5%; 18.5K perplexity 6.277, agreement 95.9% (was 6.272, 95.9%). Adding 128-column tiles or moving the panel term within 14-28: within noise. `gufo bench` pp2048 409.8 → 421.8, pp2048 @ d32768 → 378.7 t/s. |
| Token embedding left in the mapped file instead of device memory (it is dequantized on the host one row per token) | Retained. Resident weights 112.69 → 112.27 GiB GTT. |
| One batch-scratch set shared by every decoder of an engine and context (the engine serializes evaluations and no scratch value outlives one), and the host embedding rows uploaded once per token then copied on the device into the four hc streams (was four host copies) | Retained. Bit-identical on the batched and decode 3K paths. A second `gufo serve` session at the native 1M context costs its state only (0.90 GiB, was 5.6 GiB with its own scratch); host staging 160 → 40 MiB. |
| Compressed KV and index-K rows packed as FP4: E2M1 codes two per byte plus one scale byte per block (E4M3 for KV rows, block 16; the exponent of the power-of-two scale for index-K rows, block 32); the readers decode on load, eight values per code word in the batched attention and indexer staging | Retained. Bit-identical on the batched 3K and 18.5K and decode 3K paths (decoding multiplies the same E2M1 value and scale as the quantization; `ds41.attention` checks the packed bytes against the fake-quantized floats bit for bit). State at 1M ~3.1 → 0.90 GiB; saved states and cache snapshots shrink by the same factor (magic `S4STav3`). Four-value loads cost 7% in the batched attention kernel at 32K; the eight-value staging brings it to within 1% of bfloat16 (10.96 against 10.85 s of a 32K-depth bench trace), the indexer to parity. |
| Batch scratch overlaid by phase: the attention, FFN and Engram buffers, each dead outside its phase, share one region | Retained. Bit-identical on the batched and decode 3K paths. Scratch per 2048-token batch 2.1 → 1.2 GiB. |
| Candidate-block mask as one bit per block (was one byte) | Retained. Bit-identical 18.5K scores (the mask acts above 16K). 256 → 32 MiB at 1M. |
| Batched BF16 matmuls (router, compressor, indexer projections; 16 tokens and up) on the matrix cores: FP32 activations split into three exact bfloat16 terms, accumulator restarted every 32 values and summed in FP32 registers | Retained. 2K prefill BF16 time 292 → 103 ms, `gufo bench` pp2048 414.6 → 433.0 t/s (one run each). The matrix-core accumulator loses precision over long chains (2e-5 relative over k = 5120 with one accumulator; restarts every 16, 64, 128 or 256 values measured 1.1e-6, 7.5e-7, 1.1e-6 and 2.1e-6), so the result is within 8e-7 of the largest output against 2.5e-7 for the FP32 kernel (`ds41.attention`). Decode keeps the FP32 kernel (bit-identical). Batched 3K perplexity 6.307 → 6.326, agreement 95.5 → 95.1% (exact-arithmetic reorderings earlier measured 6.300-6.324, 95.5-96.1%); 18.5K 6.277 → 6.279, agreement 95.9 → 95.8% (oracle 6.267). |
| Component cache rows in chunks of 2048 rows (from 256; rows are 4-256 bytes) | Retained. Lending ~400 small chunks per 10K-token slot spilled at ~1.7 GB/s; with 2048-row chunks 8 MB spill in 0.8 ms. Restore queue time in cache-shared-prefix 16.7-17.3 → 13.7-16.0 ms. |
| One private component for all 40 window rings and one for the 4 compressor frontiers (from one per layer: 63 → 21 components) | Retained. Against the whole-state snapshot, per-request capture time median 0.107 → 0.066, restore 0.71 → 0.44. |
| No decoder reset when the cache invalidates a slot (`Sync` resets an invalid session, a restore replaces every part) | Rejected. Queue times unchanged (cache-shared-prefix `long_tasks_1` 16.7 → 16.8 ms). |
| DSpark draft width fixed at 1-5 (Lucebox Q2_K/Q4_K drafter; roadmap prompt, 1,871 tokens, 256 greedy tokens; plain 23.46 tok/s) | Basis for the cost model. Width 1: 25.28 tok/s, 77.1% accepted, 1.78 tokens/cycle; 2: 28.16, 68.8%, 2.37; 3: 27.88, 56.1%, 2.69; 4: 25.90, 45.8%, 2.84; 5: 23.12, 37.4%, 2.88. A draft costs 12.5 ms; verifying 2-6 rows costs most of a cycle, so long blocks lose once acceptance drops. Conditional acceptance per position: about 0.77, 0.78, 0.53, 0.47, 0.27. |
| DSpark draft width chosen per cycle: running estimates of each position's conditional acceptance and of each width's cycle time, width maximizing expected tokens per millisecond, plain steps when no width beats one | Retained. Same run 28.49 tok/s (+21%), 104 cycles, 2.46 tokens/cycle. Variants: a 32-step plain lockout after a losing estimate 26.0 tok/s; one-draft probes with backoff 4 → 64 plain steps 27.45; also extending the width after a fully accepted block over-explored (rejected); relaxing undrafted positions toward the prior 28.49 (retained). Acceptance estimates restart with each request (a greedy request after a sampled one otherwise drafted less); cycle times carry over. |
| DSpark drafters: unsloth Q8_0 (Q8_0 dense, MXFP4 experts, 7.42 GiB) against Lucebox Q2_K/Q4_K (4.71 GiB) | Q2_K/Q4_K retained as the recommended drafter. Roadmap prompt, 128 tokens: Q8_0 26.49 tok/s, 51.7% accepted; Q2_K/Q4_K 27.57, 52.6%. Chat coding prompt (45 tokens, 384 tokens): Q8_0 39.17 tok/s, 76.3%, 4.09 tokens/cycle; Q2_K/Q4_K 37.85, 76.3%, 3.84 (plain 23.5). Host memory after load: 2.1 GiB with Q8_0, 4.7 GiB with Q2_K/Q4_K. |
| Multi-token decode GEMVs (2-8 verification rows) load the next Q5_K block's raw words before decoding the current one, keeping each column's dot order | Retained. Bit-identical (`ds41.mmq_typed`). `--spec-bench`, 1-8 rows at 137-186 tokens: 41.8 / 55.8 / 67.3 / 77.7 / 88.5 / 99.0 / 112.0 / 113.9 → 41.8 / 54.9 / 66.0 / 76.4 / 88.1 / 97.1 / 109.6 / 110.8 ms. |
| Multi-token dense Q5_K GEMVs (2-16 tokens, every slice count; not the fused pair or SwiGLU launches): one wave per Q5_K task over 32 rows, one row per lane, so a wave's activations are uniform and arrive through scalar loads; workgroup (tile, slice) runs the slice's blocks in order and reduces the tasks in the single-token kernel's xor-tree order through LDS, and the tile's last workgroup (atomic counter, 16 MiB scratch on the null stream) adds the slice results in wave order (`split_gemv_kernel`) | Retained. At 8 tokens the lane layout spent its time on activation vector loads, not weights: feeding all tokens from one token's activations (same FMAs, wrong results) took wo_b 222 → 138 µs against 131 µs for one token. Bit-identical to single-token rows (`ds41.mmq_typed`, now with 2 slices and a partial last tile; `tools/bench/ds41_dense_gemv_tune.hip`). DRAM-cold, 8 tokens: wo_b 221 → 146 µs (130 → 198 GB/s), wo_a 166 → 123, q_b 211 → 139, wq_a 40 → 30, shared down 70 → 49; one token keeps the lane layout. `--spec-bench` on resident IQ1_M, 1-8 rows, alternating pairs: 40.9/52.6/67.7/77.1/92.8/102.3/117.5/129.7 → 41.0/51.9/65.5/73.7/87.4/96.7/107.9/118.8 ms (8 rows −8%). `gufo bench` tg512 with DSpark, three alternating pairs: resident IQ1_M 29.17/29.42/29.38 → 29.71/29.98/29.95 tok/s (+1.9%); streamed IQ2_XXS 33.17/34.14/33.28 → 33.15/33.93/34.73, inside its cache-warming spread. Rejected on the way: two or four rows per lane sharing activation loads in the lane layout (wo_b 8 tokens 222 → 238 / 294 µs); task-major with every slice in one workgroup (wo_b 219 → 289 µs, wo_a 167 → 192 µs: one wave per task walks the whole row, too few waves to hide latency; a round of 8 block loads per lane helped one token, 211 → 169 µs, not eight). |
| The expert-group plan built in O(slots²) (each slot finds its group's first slot, leaders count earlier leaders) instead of O(slots³) | Retained. Same plan (`ds41.mmq_typed`); 40 plan launches per 6-row pass 1.18 → 0.11 ms; `--spec-bench` 6 / 7 / 8 rows 97.1 → 96.1, 109.6 → 107.7, 110.8 → 109.3 ms. |
| Prompt lookup in DSpark sessions: a cycle whose context repeats earlier tokens (PromptLookup gates: 12 matching tokens, 24 and prompt-only for sampled requests, copies only as far as competing occurrences agree) verifies up to six copied tokens instead of drafting | Retained. File edit, 1,302 tokens, 512 generated: 43.88 → 56.43 tok/s, 71 of 78 cycles copied, 414 of 420 copies accepted; roadmap 28.95 and chat 37.81 tok/s (0 and 3 lookup cycles). Seven copies (eight rows) changed the output: the batched attention kernel starts at eight rows. |
| Decoder SWA bounded replay: prompts run layers 20-39 over their last 140 positions (DeepInfer's V4.1 default), layer 20 still compresses every position, and the replayed windows start at the first replayed position | Retained. Prefill 1.77-1.97× faster (8,863 tokens 14.70 → 7.75 s; 23,848 tokens 40.37 → 22.84 s); continuation NLL within 0.008 of exact prefill and top-1 agreement 254-255 of 256 ([QUALITY.md](QUALITY.md)). Serve: 11,327-token prompt 16.95 → 9.52 s. |
| Prompt lookup without a drafter, through the same speculative step | Retained. File edit 23.59 → 54.61 tok/s, chat 23.81 → 24.30 tok/s, greedy output identical. The first version evaluated each emitted token before returning it: serve's `cache` suite showed decode 37-39 → 42 ms per token and the second token 43 → 86 ms after the first, one evaluation more per request than plain decoding (which emits a token, then evaluates it). Steps now emit the token after the accepted copies unevaluated and verify it as the next step's first row: against the previous build, all 479 per-request timings pass or stay inconclusive (single flags that did not repeat over three alternating pairs), and the 30 completed replies are identical. With the DSpark drafter, five alternating pairs left only flags that changed direction between pairs, except one disk restore after a restart that waited 22-88 ms for a transfer stream in three candidate runs and 39 ms in one unchanged-build run (about 2 ms otherwise): inconclusive, and present without this change. |
| Grouped routed-expert kernel for 2-8 verification rows (IQ1_S gate/up is 17.9 ms of a 6-row pass for 21.2 distinct experts per layer; its weights alone would take about 13.4 ms at the single-row kernel's 203 GB/s) | Not adopted, all slower: the next weight block loaded across loop iterations (routed total 46.8 → 48.2 ms); per-group activation sums precomputed once per column (no gain); 10 waves per SIMD (177 → 144 VGPRs) 17.9 → 20.5 ms; two rows per lane sharing each activation load 17.9 → 45.9 ms (register pressure); 2-column tiles in a runtime loop (24 → 10 KB of code, 177 → 161 VGPRs) 17.9 → 19.5 ms. Counters: the 6-row kernel issues 4.3× the VALU instructions of the single-row kernel for 3.5× the experts, at about 55% VALU issue. |
| Vision encoder linears: hipBLAS `GemmEx` (BF16, small-tile kernels), Flash-Next's dense BF16 WMMA GEMM, hipBLASLt | hipBLASLt retained. Carrots encode: hipBLAS 1,460 ms, WMMA GEMM 1,030 ms; with the fused attention, WMMA GEMM 302 ms and hipBLASLt 249 ms. 4000 × 3000 with the fused attention: WMMA GEMM 1,252 ms, hipBLASLt 853 ms. Linears at 29 TFLOPS on the capped image. Spans within BF16 noise of the official forward. |
| Vision attention fused (online softmax, BF16 WMMA, 16 queries per wave) instead of tiled score GEMMs, softmax and value GEMMs | Retained. Carrots 1,030 → 302 ms. |
| Fused vision attention computing S^T = K Q^T and O^T = V^T P^T, so each lane keeps one query's softmax state and the half-waves swap probabilities (`permlanex16`) instead of shuffles and an LDS round trip; branchless BF16 packing; K/V tiles staged in LDS for eight waves | Retained. 4000 × 3000: 853 → 742 ms; carrots 249 → 236 ms. Bit-identical between the transposed variants. Variants: the hardware exponential (same output, 2%, within noise); LDS staging before the transposition (slower, the kernel was ALU-bound with 256 VGPRs); 16 waves (783 ms); register prefetch of the next tile (763 ms). |
| Batched IQ1_S routed experts on the FP16 matrix cores (`ds41_routed_f16.hip.cpp`): FP16 activations (one conversion per layer; the gate/up pass stores the SwiGLU rows in FP16 for an IQ1_S down pass), each weight decoded to FP16 with its sub-block scale folded in from an LDS table of the codebook's FP16 high bytes, host-built 64-slot tiles per expert, weight rows as the B operand. Each half-wave decodes one of the two 8-value groups of a K16 step and takes the other with `permlanex16`; half 1 holds the step with its groups swapped and reads its activation fragment swapped to match (each half of the wave uses its own operand copy) | Retained. `ds41_mmq_bench --type 19` on recorded 2K-prefill expert counts: gate/up+SwiGLU 37.5 → 22.8 ms, down 21.4 → 12.2 ms per layer (MMQ). Max error against the FP32-activation expert GEMV 0.18% of the largest SwiGLU row value and 0.05% for down, against 3.5% and 1.2% for MMQ's Q8_1 inputs. 2K batch 5220 → 4680 ms, 952-token batch 2714 → 2348 ms; batched 3K perplexity 6.326 → 6.311 (oracle 6.309), agreement 95.1 → 95.9%. Steps that measured: ordinary instead of nontemporal weight loads 30.5 → 23.8 ms (nontemporal loads refetched the partly used lines; L2 misses 39 → 31 M); FP16 instead of FP32 activation staging 32.2 → 30.5 ms; the group split 23.8 → 22.8 ms. No gain: 16-byte instead of 4-byte weight loads, weights two blocks ahead (32.7 → 32.2 ms), occupancy hints. The 4-byte nibble codebook with the 1024-magic decode instead of the FP16 byte table: 23.8 → 25.7 ms. |
| The FP16 routed kernel for IQ1_M and IQ2_XXS too (a format trait per type: IQ1_M takes its 3-bit scale per 16 values and its delta per group, from the block's spread f16 scale; IQ2_XXS reads each codebook entry as eight FP16 magnitudes and each sign index as eight FP16 sign bits from LDS, then one XOR and one multiply per pair) | Retained. `ds41_mmq_bench` against MMQ, gate/up+SwiGLU and down per layer: IQ1_M 49.8 → 24.2 and 28.0 → 12.7 ms, IQ2_XXS 39.3 → 27.2 and 22.8 → 14.1 ms (IQ2_S stays on MMQ). 2K batch 4680 → 4337 ms (472 tok/s), 952-token batch 2348 → 2154 ms; batched 3K perplexity 6.316, agreement 96.2%; 18.5K prefill 48.5 → 39.6 s, perplexity 6.279 (unchanged, oracle 6.267), agreement 95.8 → 96.0%. |
| Batched dense Q5_K projections (q_b, the grouped output LoRA, o_b, shared expert) on the FP16 matrix cores (`ds41_dense_f16.hip.cpp`): 128 rows x 128 tokens per workgroup, 16 rows per wave against eight 16-token tiles, each lane loading only the half-blocks its half-wave decodes (no LDS for weights), FP32 activations prefetched in registers one 64-value stage ahead and rounded to FP16 into LDS, each half-wave reading eight values of an activation fragment and taking the other eight with `permlanex16` | Retained; replaces MMQ and its Q8_1 passes for 9 tokens and up. `ds41_mmq_bench --dense` at 2048 tokens, MMQ (with its quantization) → FP16: q_b 3.52 → 2.51 ms, output LoRA (8 groups) 4.04 → 3.90 ms, o_b 6.95 → 5.30 ms, shared gate 2.26 → 1.84 ms, shared down 2.00 → 1.44 ms; faster from 16 tokens up. Relative error 3.5e-4 against the FP32-activation GEMV. 2K batch 4337 → 3977 ms (515 tok/s), 952-token batch 2154 → 2102 ms, 18.5K prefill 39.6 → 37.2 s; batched 3K perplexity 6.285 (oracle 6.309), agreement 95.9%; 18.5K 6.269 (oracle 6.267), 96.0%. Steps: FP16 activations from a separate conversion pass were slower than staging FP32 in the kernel; staging loads issued one at a time (no register prefetch) and a branch-free full-tile variant (the compiler hoisted it into 256 VGPRs and 760 spills) both lost; the fragment exchange instead of two 16-byte LDS reads per lane took o_b 7.01 → 6.29 ms (it measured slower in the routed kernel, whose fragments feed two products). |
| Batched attention rewritten (`attention_f16_kernel`): 256 threads per (token, 32 heads), window and compressed rows in separate 16-row tiles prefetched in registers one tile ahead, each thread's two staged columns kept in registers as the B operands of the output products (each half-wave multiplies by its own operand copy, so its own columns and the opposite half's via `permlanex16` cover them), Q fragments in registers, FP4 rows decoded through LDS tables with one FP16 multiply (exact), DPP reductions; the q rotation and the output's inverse rotation and FP8 rounding run inside the kernel from a per-batch (cos, sin) table, replacing the two separate passes | Retained. `ds41_attention_test --bench` at 2048 tokens with the rotations: window-only layers 3.1 ms, ratio-1 layers 9.4 ms, ratio-2 8.5 ms (old kernel plus passes 13.5 ms per layer on average, plus 4.5 ms of rope passes). In the 2K trace attention 540 → 364 ms and the rope passes are gone. 2K batch 3977 → 3562 ms (575 tok/s), 952-token batch 2102 → 1881 ms, 18.5K prefill 37.2 → 33.3 s; batched 3K perplexity 6.294 (oracle 6.309), agreement 96.4%; 18.5K 6.267 (oracle 6.267), 96.2%. The kernel's code size decided its speed: the first fused version (44 KB of code, above the 32 KB instruction cache) ran slower than the old kernel; IEEE divisions in the FP8 rounding replaced by exact exponent shifts and a rolled epilogue (36 → 21 KB) took the ratio-1 layer 17.6 → 13.0 ms, homogeneous tiles with table decoding (14 KB) → 9.4 ms; row ids read from global memory per row (26 KB) lost. No gain: the partial scores sharing the KV row buffer (three workgroups per WGP, one more barrier). |
| The FP16 routed kernel's weight-block loop rolled into one copy of the block body (the two-block ring shifts by register moves instead of two unrolled steps plus a tail copy): 40 → 16 KB of code per variant, below the 32 KB instruction cache | Retained. `ds41_mmq_bench --f16 1`, gate/up+SwiGLU and down per layer: IQ1_S 22.8 → 22.2 and 12.2 → 11.4 ms, IQ1_M 24.2 → 22.8 and 12.7 → 11.8 ms, IQ2_XXS 27.2 → 26.1 and 14.1 → 13.1 ms; routed kernels in the 2K trace 1436 → 1377 ms. Same values (`ds41.mmq_typed`); batched 3K perplexity 6.294, agreement 96.4%. |
| FP16 activations into and out of the dense FP16 kernel (the producer stores FP16, the kernel stages them without conversion), which would also halve those activation buffers | Not adopted. `ds41_mmq_bench --dense` at 2048 tokens, FP32 → FP16 input: q_b 5.23 → 5.66 ms, output LoRA 6.36 → 6.87 ms, o_b 5.43 → 5.88 ms, shared down 1.38 → 1.56 ms; only the shared gate/up gained (3.27 → 3.11 ms). The kernel is not bound by its activation reads. |
| Larger prefill batches for the routed experts (`ds41_mmq_bench --f16 1 --tokens 4096` on the recorded counts scaled ×2) | Available as `--prefill-batch 4096` (`serve`, `bench`, `prompt`); the default stays 2048. The skewed expert counts leave fewer padded 16-slot tiles at 4096 tokens: routed throughput 26 → 31.5 TOPS. End to end, with the FP16 matrix-core kernels: `gufo bench -p 4096` 694 → 720 tok/s (+3.7%, alternating runs), the first 4096-token prompt in a fresh session 618 → 682 tok/s, 18.5K prefill 31.5 → 29.7 s with perplexity 6.270 (oracle 6.267), agreement 96.2%; activations +1.12-1.21 GiB. 8192-token batches were slower (18.5K prefill 37.2 s, +3.4 GiB), so the option stops at 4096. |
| The dense FP16 kernel at 256 weight rows per workgroup (16 waves sharing one staged activation tile, which halves the activation reads) and its activations prefetched two stages ahead | Retained. Removing the activation loads (timing only) had taken the output LoRA 6.38 → 3.76 ms, while removing the weight decode or the fragment exchange saved 3-6%. `ds41_mmq_bench --dense` at 2048 tokens: q_b 5.23 → 5.21 ms, output LoRA 6.36 → 5.55 ms, o_b 5.43 → 5.14 ms, shared gate/up 3.27 → 3.11 ms, shared down 1.38 → 1.41 ms. 2K batch 3443-3608 → 3427 ms, 952-token batch 1845 → 1820 ms; batched 3K perplexity 6.294, agreement 96.4% (unchanged). No gain: 512 rows per workgroup (one workgroup per WGP), the stage loop rolled into one body (23 → 8 KB of code, 1-3% slower), a barrier that skips the wait for global loads. Without its two barriers per stage (timing only) the kernel ran the output LoRA in 4.20 ms and o_b in 4.83 ms, but double-buffered activation tiles with one barrier per stage were slower (o_b 5.14 → 6.34 ms with 64-value stages, one workgroup per WGP; 6.49 ms with 32-value stages, two per WGP). |
| Batched mHC state passes fused (`hc_state_pass_kernel`): the mixes and the collapse with its RMS norm read the state once (the collapse uses the pre-mix that was known before the mix, from the shifted mHC), and before the FFN the attention expand runs in the same pass, so the expanded state is not read back twice. The mix loop is rolled (34 → 7 KB of code) | Retained for 16 tokens and up. `ds41.hc_state_pass`: mixes and normalized rows within 3e-7 of the separate kernels, the state bit-identical. 2K batch 3427 → 3359 ms, 952-token batch 1820 → 1790 ms, 18.5K prefill 33.3 → 31.5 s (with the 256-row dense kernel); 18.5K perplexity 6.273 (oracle 6.267), agreement 96.2%; batched 3K 6.320, 96.0% (rounding drift through the FP8/FP4 fake quantization; earlier rounding-only changes spanned 6.285-6.326). |

## Post-M6 resident optimization pass (2026-10-26)

Baseline source: `ee49a0fd` (sampling defaults from `edce52dd`); production
`release` builds with the same Nix compiler/dependencies. The pass followed a
source review against DeepInfer (`5537415`, ROCm PR #201 `45ccedf`),
deepseek-recipe, ds4 and its Strix Halo PR, Kyojin, the Spark EXL3 recipe,
Project Maya, Xolotl and Gufo's Qwen/Gemma paths; its proposals that remain
open are in the backlog. Each experiment records its disposition, including
rejected attempts. Raw observations and provenance
for E1 are in [artifacts/2026-10-26-rescore](artifacts/2026-10-26-rescore/README.md).

| Experiment | Decision / evidence |
| --- | --- |
| E1: compact candidate blocks for FP32 indexer rescoring, 1–4 rows after 16K; keep dense score storage and the original final top-k | Retained for the measured AR gain: tg128 at 32K 22.78 → 23.05 tok/s (+1.2%), at 128K 21.10 → 21.89 (+3.7%); depth-zero control 23.97 → 24.03. Selected scores and all 512 output indices retain their original bytes/order; independent FP64 score error ≤2.2e-7 in the correctness cases. Uses the existing block-select scratch, with no extra persistent allocation. Full-history source indexing and WMMA prefill rescoring remain future work. |

E1 compacts layer 20's mask into increasing physical block IDs for each
reindex call and clears unselected dense scores to −inf. The existing FP32
score body gathers only those blocks. Keeping the original final selector
preserves both the selected set and its emitted order, including ties and a
partial newest block. A separate template specialization leaves unrestricted
source scoring on its original arithmetic path. The eight-decode 32K trace
confirms 32 compact calls, 32 unrestricted calls and zero old apply-mask calls;
GPU time falls 327.27 → 325.91 ms and span 362.24 → 360.86 ms. Each scorer
specialization uses 248 VGPRs, 128 SGPRs, 51,328 B LDS and no private scratch.

Qualification: `ds41.attention` and `ds41.candidate_mask` pass, including
varying FP4 scales, tied scores, different query masks, causal partial blocks
and exact-size allocations. Teacher-forced controls remain at perplexity
6.320 / 6.273 (3K / 18.5K). At 18.5K, an eight-cycle width-four speculative
probe retains 20 rows: every full logit vector is bit-identical both to serial
decode and after changing rejected suffixes. Details are in [QUALITY.md](QUALITY.md).

Long-context API correctness passes with and without the Q2_K/Q4_K drafter,
at 22,591 prompt tokens: sampled replay, prefix reuse, cancellation/resume
and output stop handling. AR passes all 124 per-request/phase timing checks.
DSpark performance is **inconclusive and unqualified**: three alternating
pairs and an unchanged-baseline control yield 86 passing and 38 inconclusive
metrics, zero confirmed slowdowns, and no quality/coverage changes. The
flagged cancellation queues and short replies accompany varying draft widths;
their observed timing ranges overlap. Every observation is retained at the
unchanged 5% / 3 ms margins. No DSpark speedup is claimed.

The C2 32K control (one repetition) measures 11.50 → 11.55 tok/s per user,
with aggregate prefill 1127.04 → 1122.41 tok/s. It is a concurrency control,
not evidence of a concurrency speedup.

The same C1 runs' prefill controls were 1187.67 → 1219.95 tok/s at depth zero,
1109.71 → 1123.29 at 32K and 856.10 → 829.19 at 128K. These paths are unchanged
by E1; their variance is retained, rather than counted as prefill improvements.

P0 profiling refresh: the final pp2048 at 32K, with decoder bounded replay,
uses 1,773.84 ms GPU time over a 1,805.14 ms span. Stage sums: routed FP16
606.39 ms, dense FP16 518.14, attention 219.06, mHC 137.38, indexer 82.95 and
top-k 45.34. Loading and depth preparation are excluded. The older backlog's
3.30-second prefill breakdown predates bounded replay and is not a current
end-to-end cost estimate.

### E2: stochastic DSpark proposals (retained, with confidence-gated verification)

Connected the existing bounded-q proposal and exact p/q residual callbacks to
post-Markov top-eight candidates, enabled unfiltered sampled requests, and
froze sampled cycle costs so wall-time variation could not change seeded
proposal draws. Each next Markov bias used the actual sampled proposal.

One focused check passed in 0.17 s: biased GPU top-eight with ties/non-finites,
independent target-frequency correction (including tokens outside q), and
seeded proposal/residual replay. A real 256-token candidate replay was
identical with tracing enabled; 20 probabilistic verification rows ran.

Matched IQ1_M + Lucebox Q2_K/Q4_K, temperature/top-p 1.0, seed 73, 256 tokens:
coding 36.335 → 37.527 tok/s (+3.3%), prose 23.469 → 23.045 (−1.8%). Coding
acceptance was 73.9 → 76.9%; prose 46.1 → 22.2%, with the candidate mostly
backing off to plain steps. Proposal draws change the continuation, so these
single-seed observations are not a general speed qualification. The prototype
is rejected for insufficient consistent gain; production retains point-mass
sampling. No broad functional matrix was run. The patch, settings, focused
check and every measured run are retained in
[the E2 record](artifacts/2026-10-26-sampled-proposals/summary.json).

**Re-measurement.** The single-seed trial was too small: the prose candidate
drafted 18 tokens in total, so its 22% acceptance is noise. Against
`8d51885a`, four prompts (coding, prose, explanation, Rust) × seeds
73/11/29, 256 sampled tokens and temperature/top-p 1.0, the original
prototype gains +1.8% geometric mean with its fixed costs and +2.5% with
adaptive costs. Pooled acceptance per drafted token rises from 0.656 to
0.675/0.686. Moving the draw to the GPU (below) adds nothing on its own
(+0.7% fixed, +2.4% adaptive). Per-pair SD is 5–10%, because each variant
samples a different continuation.

Forced five-token drafts (8 runs, 680 cycles) show where the cost goes.
Proposals with q's best probability ≥ 0.7 (point masses) are accepted 85% of
the time; diffuse ones only 19–51%, by confidence bin. A verified row costs
11–13 ms (spec-bench: 41.8 ms for one row, 99.1 ms for six), about 30% of
a plain step. That is routed-expert traffic: about 73% of the 1 → 4 row
increase, with `expert_group` near the measured read ceiling. Verifying
diffuse rows therefore costs more than it emits. An offline replay of the
traces with the measured costs ranks "draft the limit, verify up to the first
proposal with q's best < 0.6" above every fixed width (32.1 vs 27.9 tok/s
simulated over all cycles).

Retained design:

- Sampled requests (any temperature > 0, now including top-p 1.0) draft
  from the post-Markov top eight. The draw happens on the device: the host
  draws one 24-bit uniform per row up front from the request RNG, and one
  kernel per row finds the biased top eight, forms q (the temperature
  softmax in 2^-24 units, or a point mass when its best is ≥ 0.7) and writes
  the proposal where the next Markov row reads it. q returns once per cycle,
  and the bridge verifies against exactly that q with p/q acceptance and
  residual correction.
- Sampled cycles draft the request's limit and verify the prefix of
  proposals whose q best is ≥ 0.6; a cut at zero becomes a plain step.
  Widths depend only on q, never on timing, so seeded replay is
  deterministic. Repeated runs and the gated/fused builds give identical
  hashes. Greedy requests keep the cost-based controller.

Same 12-pair protocol: +3.85% geometric mean against `8d51885a` (9/12
faster; coding +4.7%, explanation +4.2%, prose +4.6%, Rust +1.9%). Pooled
29.76 → 30.93 tok/s; accepted per verification 1.80 → 2.07. The focused
check covers the biased top eight with ties/non-finites, device q (softmax,
mass sum, draw boundaries and the point-mass rule) and exact recovery of the
target distribution through Retain/Verify over 30,000 seeds, including
tokens outside q. Each configuration's runs, the traces, the replay
simulator and the variant patches are in
[the re-measurement](artifacts/2026-10-26-sampled-proposals/remeasure/summary.txt).

### E3: request-local BPE piece memoization (retained)

Cache previously tokenized pretokenizer pieces within each immutable text span,
following the existing Qwen/Gemma memoization approach. The cache holds at most
512 pieces of 2–64 bytes and activates for spans of at least 1,024 bytes. Borrowed
keys never outlive the input; cached IDs never cross a call or vocabulary reload.
Pretokenization, merges, Unicode handling and special-token handling are unchanged.

Every emitted token-ID line matches the baseline for eight copies each of a
repository documentation/source corpus with multilingual and combining-character
text, and a corpus of 12,000 unique identifiers. The metadata-only probe workload
(including process setup and ID printing) takes 172.53 → 149.19 ms for the first
corpus and 113.53 → 108.22 ms for the second. These are single-run host component
measurements; no model throughput or TTFT gain is claimed. Production and probe
builds and the C++ format check pass. No GPU model sweep was needed for unchanged
token IDs. Settings, corpus descriptions and output hashes are retained in
[the E3 record](artifacts/2026-10-26-tokenizer-memo/summary.json); raw inputs and
outputs are under `/tmp/gufo-ds41-tokenizer` on the measurement host.

### E4: stop unused DSpark Markov suffixes (retained)

Pass the controller's selected proposal count into the drafter. Keep the full
five-row transformer, normalization and output head, then stop the dependent
Markov gather/projection/argmax chain at that count and copy only those IDs.
The used proposal prefix retains its arithmetic; no target or rollback work changes.

The existing dump probe now checks widths 1–5 against the complete draft on a
128-row history, checks untouched caller output tails, and rejects widths zero
and six. It passes. Production and probe builds and the format gate pass.

C1 production CLI, 256 sampled tokens, temperature/top-p 1.0 and seed 73:
coding baseline 37.455 / clean control 37.873 tok/s, candidate 37.656; prose
23.219 / clean control 24.133, candidate 24.188. Each corpus has identical
output-token hashes in every run. Trace medians confirm short-width draft
latency falls: one proposal about 11.7 → 8.9 ms, two 11.6 → 9.7 ms, four
11.7 → 11.0 ms; five stays about 11.8 ms. Adaptive widths and plain-step
backoff vary between runs. **Aggregate throughput is approximately flat and
no general speedup is qualified.** Retained for removing measured unused work
with unchanged proposal prefixes; no larger quality sweep was run. All runs,
settings, trace distributions and focused checks are in
[the E4 record](artifacts/2026-10-26-markov-tail/summary.json).

### E5: score only requested draft-head rows (retained)

After E4, shortened the shared target vocabulary head to the requested draft
count, keeping the transformer and collapse normalization at five rows.
An extended dump check found every used vocabulary-logit row bit-identical
at widths 1–5 on a 128-row target history; proposal prefixes, untouched output
tails and invalid-width handling also passed. Production/probe builds and
formatting passed.

Matched to E4's candidate runs (same binary retained as this baseline), C1,
256 sampled tokens, temperature/top-p 1.0 and seed 73: coding 37.656 → 38.214
tok/s (+1.5%), prose 24.188 → 23.851 (−1.4%). Output-token hashes match for
both prompts. Short-width drafting is cheaper, but adaptive-controller
changes prevent a consistent total-request gain in this single-seed trial.
That single pair was first rejected; the trial was too small for a change
with identical outputs, since adaptive draft widths alone vary a run by
several percent.

Re-measured against `8d51885a` with the same production toolchain: four
prompts (coding, prose, explanation, Rust) × seeds 73/11/29, 256 sampled
tokens, temperature/top-p 1.0, alternating order without tracing. All 12
pairs emit identical tokens. Candidate throughput is +0.33% geometric mean
(9/12 pairs faster, per-pair SD 2.4%); pooled 29.78 → 29.87 tok/s. The
draft-prefix/logit probe passes again on the rebased change. Retained as a
small measured saving with unchanged output: the head now evaluates only the
requested rows, and the transformer body keeps all five. The paired runs
and scripts are in [the re-measurement](artifacts/2026-10-26-draft-head/remeasure-summary.txt);
the first trial remains in [the E5 record](artifacts/2026-10-26-draft-head/summary.json).

### E6: explicit FP32 index scoring for verification (retained for numerics)

Use the existing speculative-evaluation purpose flag to group index scoring
into at most four rows. This keeps the serial FP32 score/reduction at widths
five through seven and enables E1's compact rescoring there; ordinary prefill
keeps its WMMA dispatch. No kernel arithmetic or persistent-state format changes.
The probe now rejects non-finite logits and serial-logit mismatches, rather
than merely printing the serial comparison while gating only rejected-tail
independence. Correct the session comment that treated all sub-eight-row
kernels as inherently scalar-equivalent.

At 18.5K, baseline width six already matches serial on all 21 retained rows:
the dispatch gap did not demonstrate a model bug on this history. Candidate
width six also matches all 21 rows, and width seven at 3K matches all 28.
Each test exercises every commit length and corrupts the rejected suffix;
maximum logit differences from serial and between clean/corrupt passes are zero.
E1's independent FP64 scoring/selection evidence applies to the unchanged
FP32 kernel. Production/probe builds and the format gate pass.

One matched production C1 IQ1_M + Q2_K/Q4_K, greedy pp2048/tg128 pair:

| Phase | Baseline tok/s | Candidate tok/s |
| --- | ---: | ---: |
| pp2048 | 1108.95 | 1071.44 |
| tg128 | 23.69 | 26.12 |
| pp2048 at 32K | 1117.50 | 1118.19 |
| tg128 at 32K | 54.54 | 53.97 |

Both depth-zero and 32K generated-token hashes match. The 32K trace exercises
six-copy lookup plus anchor (seven verification rows), so the changed path
actually runs. Short decoding has adaptive-policy variation; prefill's code
is unchanged. No phase exceeds both slowdown gates (5% and 3 ms), but this
single pair does **not** qualify a speedup: the affected long-context decode
is about 1% slower. Retained as an explicit numerical contract for verification
and stronger qualification, not as a throughput win. Eight-row attention and
wider projection equivalence remain open; no verification limit was widened.
All checks, rates, hashes, commands and raw benchmark traces are in
[the E6 record](artifacts/2026-10-26-verification-scores/summary.json).

### E7: stable device route plans for FP16 layers (retained groundwork)

Large batches whose gate/up and down both use FP16 now build their expert
counts, bounds, stable slot lists and 64-slot tiles on the GPU. Reuse the
existing plan allocation and launch a bounded tile grid: unused entries are
−1 and return before reading expert metadata or weights. No added model or
scratch allocation. The IQ1_M artifact takes this path in 38 of 40 layers;
the two mixed-MMQ layers retain their CPU counts/classes and plans. Small
batches retain their existing GEMV plans.

The focused planner/operator check passes: exact rows, bounds and tile order
against the original host plan (with a standard-library stable-sort check),
63/64/65-slot tails, 1/7/384/512 experts, skewed routes, invalid and all-invalid
IDs, plus fused gate/up and down for IQ1_S/IQ1_M/IQ2_XXS. The 3K matched model
check retains all 2,999 argmax/NLL records at the probe's six-decimal NLL
precision, perplexity 6.319756. Generated-token hashes match. Production and
probe builds and formatting pass.

One production C1 pair (no drafter):

| Phase | Baseline tok/s | Candidate tok/s |
| --- | ---: | ---: |
| pp16 | 71.31 | 72.41 |
| pp128 | 362.46 | 362.58 |
| pp2048 | 1205.51 | 1205.80 |
| tg128 | 23.99 | 24.01 |

No phase exceeds both slowdown gates (5% and 3 ms), but these single pairs
qualify no throughput gain. A separate pp2048 trace confirms 38 device plans;
the three planner kernels total 1.18 ms. Excluding load/setup, GPU work is
1624.49 → 1621.79 ms, GPU span 1651.56 → 1645.07 ms; removing host plan/class
uploads offsets the new planning kernels. Retained for eliminating the 38
host round trips and making those layers' route planning device-resident,
with effectively flat unprofiled throughput. Graph capture and the mixed-MMQ
plan remain open. All observations and checks are in
[the E7 record](artifacts/2026-10-26-device-route-plan/summary.json).

### E8: 32-slot routed FP16 tiles (rejected)

Template the routed FP16 kernel on its tile width and launch a 32-slot grid
for device-planned tiles with at most 32 live slots, beside the existing
64-slot grid. Both consume the same 64-slot plan, so each live tile has one
owner. The 32-slot body needs 135–146 VGPRs and 15–25 KB LDS, against 169–180
VGPRs and 24–34 KB for 64 slots.

The first prototype measured pp2048 1210 → 562 tok/s. The cause was code
generation, not the tile width: its ownership early return lets the compiler
prove at least three live 16-slot groups in the 64-slot body, which then
needs 256 VGPRs, 538–671 spills and about 2 KB of private scratch per lane.
Hiding the count's range after the check restores the original 64-slot
resources exactly, with no spills in any variant.

Spill-free variant, two alternating production `bench` runs (no drafter),
against `8d51885a`: pp512 768.3/769.0 → 758.8/772.5 tok/s, pp2048
1243.8/1247.4 → 1232.9/1244.3, tg16 unchanged. Narrow tiles therefore give
no measurable prefill gain. Rejected; the 64-slot kernel is unchanged. Both
patches, per-kernel resource usage and the runs are in
[the E8 record](artifacts/2026-10-26-routed-32-slot/resource-usage.txt).

### E9: fuse the Markov projection into its selection (retained)

After E2, a five-row draft spent 4.25 ms in its Markov tail, more than the
2.48 ms vocabulary head. Each row ran the rank-256 FP16 W2 projection
through the generic matmul (686 µs for 66 MB, about 96 GB/s), wrote a
129K-entry bias, then scanned logits + bias in a single workgroup
(278 µs for the top eight).

One pass now computes each W2 row (one 16-byte load per lane), adds the
head logit and keeps the best one or eight per wave. Each of 256 blocks owns
a contiguous token range, and emits its candidates by descending value then
lowest token. A one-block merge (argmax, or top eight plus the E2 device q
draw) therefore keeps the lowest-token tie rule. Other W2 formats keep the
previous path.

The focused check compares the fused argmax/top eight with an FP64
reference over the full 129,280-token vocabulary. It covers NaN/−inf rows and
an exact tie split across blocks, which must resolve to the lower token.
Greedy outputs and acceptance statistics are identical to `8d51885a` for
coding and prose (the drafted tokens match); median sampled draft time falls
12.6 → 10.0 ms with an identical token hash.

12-pair sampled protocol (as E2): +2.09% over E2 (11/12 faster, per-pair SD
1.2%; the continuations are identical), +5.26% over `8d51885a` (11/12;
coding +4.2%, explanation +3.7%, prose +8.2%, Rust +5.0%). Pooled
30.10 → 31.73 tok/s. Runs and checks are in
[the E9 record](artifacts/2026-10-26-fused-markov/summary.txt).

### E10: FP32-activation GEMV for the drafter's Q4_K projections (retained)

The drafter's dense Q4_K projections (q_a/q_b, KV, output a/b and the shared
expert, about 87 MiB per stage) ran through Q8_1 MMVQ for five rows. They
now take the target's K-quant GEMV with a Q4_K format: FP32 activations,
each weight decoded once for every row. The target model's own dispatch is
unchanged.

The operator test adds Q4_K against the reference dequantizer: relative error
≤ 5.2e-7 for 1,280/5,120-wide rows at every slice count. Two to eight
tokens, pair and fused SwiGLU are bit-identical to single-token passes. Median
sampled draft time falls 10.0 → 9.3 ms. Draft arithmetic changes, so sampled
continuations differ; greedy outputs remain identical.

12-pair sampled protocol against E9: +0.41% geometric mean (7/12 faster,
per-pair SD 7.4%), pooled 31.89 → 32.14 tok/s. Acceptance per draft
0.682 → 0.693, positional 0.804 → 0.800; no acceptance loss is visible. The
end-to-end gain is within noise. Retained for the measured draft-stage
saving. Runs and checks are in
[the E10 record](artifacts/2026-10-26-draft-q4k-gemv/summary.txt).

### E11: learned DSpark acceptance head for greedy cycles (retained for greedy)

The head is retained; its fixed ≥ 0.5 cut was replaced by the cost-aware
prefix of [E18](#e18-cost-aware-draft-prefix-retained).

The drafter GGUF ships a learned acceptance head
(`dflash.dspark.confidence.weight`, 5,376 F16, zero bias): the sigmoid of
one dot product over the pre-norm head collapse of each draft row and that
row's Markov W1 input. It is now bound when present and evaluated per row by a
one-block kernel after the Markov gather. With the head, greedy cycles draft
the request's limit and verify the prefix whose rows each predict acceptance
≥ 0.5, replacing the timing policy's one width per cycle. Sampled cycles keep
the q-top ≥ 0.6 cut and do not run the head.

Forced width-5 traces (two seeds sampled plus greedy, four prompts) predict
acceptance far better than q-top: AUC 0.856 vs 0.735 sampled, 0.877 greedy
(q-top carries no greedy information), and the head is close to calibrated
(the 0.5 bin is accepted 64%, the 0.9 bin 99%). Simulated with measured verify
costs, the cut-at-first-row-below-τ policy is flat for τ 0.45–0.55 and matches
cumulative-survival and expected-rate rules within 1%.

The operator test compares the kernel with an FP64 reference, with and
without bias. Greedy outputs are identical to E10. Greedy protocol (four
prompts × three repetitions): +13.07% geometric mean, 12/12 faster
(coding +9.9%, explanation +22.9%, prose +9.3%, Rust +10.7%); pooled
29.87 → 33.94 tok/s, acceptance 0.757 → 0.798. The same cut on sampled cycles
measured −1.0% (3/11 faster, per-pair SD 4.9%): 13% fewer verified rows,
but accepted tokens per step fell 2.00 → 1.92. It is not used there, and
sampled outputs remain byte-identical to E10. Functional sampling, batch,
stop and tool suites pass with DSpark. Runs, traces and the simulator are in
[the E11 record](artifacts/2026-10-26-learned-confidence/summary.txt).

### E12: device target draws for sampled DSpark verification (retained)

A cycle split showed that copying verification logits to the host costs
0.07 ms, and greedy host verification about 1.3 ms per six-row cycle (not
worth moving). Sampled host verification cost about 5 ms per cycle: each row
built a 129,280-entry FP64 softmax (0.6 ms), and a rejection added a hashed
residual pass.

Sampled rows are now drawn on the device while the request's selection is
plain softmax sampling: positive temperature, no top-k, top-p 1, no min-p,
penalties, active grammar, forced reasoning end or deferred draw. Three
kernels compute per-span maxima, per-span FP64 softmax sums and the draw: p/q
acceptance of a proposal with two or more tokens, otherwise a plain draw, and
a residual draw after a rejection. The draw walks its span in token order
with the host sampler's arithmetic. The session draws the host's uniforms in
the host's order and restores the RNG to the consumed count. Rows the host
would draw without a uniform (fewer than two nonzero probabilities) and every
other configuration use the host sampler. The bonus draw after a full block
(or a plain step) uses the same path. The device argmax supplies the
positional statistic.

The operator test compares 240 device draws (97 accepted and 23 rejected
proposals, 120 plain draws, rows with NaN/−inf, temperatures 0.6 and 1) with
the shared host sampler: tokens, acceptance, uniform consumption and argmax
are identical, and a single-token row is reported degenerate. Sampled and
greedy outputs and speculative statistics are identical to E11. Full-width
verification falls 106.0 → 101.1 ms.

12-pair sampled protocol against E11: +1.99% geometric mean, 12/12 faster
(per-pair SD 0.6%; coding +2.6%, explanation +2.0%, prose +1.4%, Rust
+2.0%), pooled 32.58 → 33.21 tok/s, with identical outputs. Each device row
still costs three launches and a synchronizing copy, so the realized gain is
below the removed host time. Functional sampling, batch, stop and tool suites
pass with DSpark. Runs are in
[the E12 record](artifacts/2026-10-26-device-target-draws/summary.txt).

### E13: one launch set for all verification rows (rejected)

The E12 draw kernels gained a row dimension: every verification row and the
bonus row drawn in one launch set with one copy, the uniforms drawn up front
in the host's order with the RNG state after each, and rows handed out in
order with per-row eligibility rechecks. A five-row batch matches the
single-row draws exactly and outputs are identical to E12. 12-pair sampled
protocol against E12: −0.50% geometric mean (3/12 faster, per-pair SD 0.8%).
Drawing the rows after the first rejection costs about what the saved
launches and copies gain, so E12's per-row draws remain.

Plain DeepSeek V4.1 decode needs no separate device draw: with prompt lookup
always on, its plain steps run through the lookup/DSpark step and already take
E12's bonus draw (31 of 32 tokens in a traced run; the request's first token
is drawn on the host). A `SampleNext` path for the generic decode loops was
therefore dropped. Patches and runs are in
[the E13 record](artifacts/2026-10-26-device-draw-rows/summary.txt).

### E14: joint evaluation of concurrent DSpark sessions (retained)

Concurrent DSpark requests stepped one session after another, so every
session's verification read all weights again. A batch step now prepares
every session's block (anchor, prompt-lookup copy or draft), evaluates the
blocks of up to 16 rows as one joint speculative evaluation, then verifies,
commits and emits per session. Projections, routed and shared experts,
Engram, mixes and the output head run once over all rows; RoPE, the
compressor, the indexer and attention run per session against its own
state through activation views that start at its rows. Each drafter state
keeps its own copy of its device proposals, which the next session's draft
would otherwise overwrite. Batches with image tokens or another activation
set fall back to per-session evaluation.

Every row keeps single-token arithmetic. The dense FP32 GEMV and the
grouped expert GEMV take up to 16 rows (expert plans of 128 slots); a
decoder's `decode_rows` keeps ordinary decode and prefill on their 8-row
switch points and lets joint evaluations use 16. Two kernels changed
arithmetic at exactly 16 rows: BF16 projections switch to WMMA tiles, so
decode batches issue them in groups of eight; the per-token hc mix kernel
now covers 16 rows, which previously always took the state pass.

The kernel test matches 12- and 16-row launches against one launch per
row for every dense and expert type. `ds41_load_probe --joint-check`
evaluates 2–4 sessions of 1–6 rows each, jointly and each on its own: all
328 rows of the four-session run are bit-identical, and the joint
evaluation takes 201 ms against 292 ms for 13.7 rows. In `gufo bench`
(tg128 at depth 512, alternating builds), per-user decode rises 17.4 →
24.3 tok/s at C2 (+40%) and 8.6 → 13.8 at C4 (+60%) greedy, 13.6 → 19.1
and 6.8 → 11.0 sampled, with C1 unchanged and every request's output and
speculative statistics identical. Functional sampling, batch, stop and tool
suites pass with four sessions; their C4 timing rows stay inconclusive, as
an unchanged-build control flags the same arrival-order-dependent cases.
Runs are in [the E14 record](artifacts/2026-10-26-joint-sessions/summary.txt).

### E15: 16-row single-session verification (retained)

Prompt-lookup verification stopped at seven rows (the anchor and six
copies): from eight rows the matrix-core attention replaced the decode
kernel and changed logits. On a code-edit reply, every lookup cycle copied
and accepted six tokens, so the cap set the speed. Split attention, which
evaluates each row on its own, now runs up to 16 rows whenever the caller
supplies its partial buffer, and a speculative evaluation uses the decode
switch points up to 16 rows, as E14's joint evaluations do. A session now
verifies the anchor with up to 15 copies; DSpark blocks are unchanged.
Ordinary evaluations of exactly eight rows, which already used the per-row
GEMVs, now also take split attention.

The attention test adds 12- and 16-row split cases against the f32
reference. `ds41_load_probe --spec-check` keeps every row bit-identical to
single-token decode at widths 8, 12 and 16 on the 1,096-token prompt, and at
width 16 on the 3,000- and 18,500-token histories (136 rows each). A
verification costs 113.9 ms for 7 rows and 223.1 ms for 16, so fully
accepted lookup commits 71.7 instead of 61.5 tokens per second. Repeating a
3,000-character source file with one rename (greedy, 900 tokens) rises from
58.5 to 67.6 tok/s (+15.6%) with 62 instead of 131 verifications, every
copy accepted and identical output. The four standard prompts, which copy
nothing, are unchanged and identical in sampled and greedy runs, and so is
prefill. The `long-context` and `cache-edits` suites pass without quality
changes; one first-run decode flag did not recur on two reruns or two
unchanged-build controls, and the remaining prefill flags match the
controls' variance and stay inconclusive.
Runs are in [the E15 record](artifacts/2026-10-26-wide-verification/summary.txt).

### E16: 4096-token prefill batches after bounded replay (rejected as default)

Bounded replay changed what a prefill batch costs, so the 4096-token batch
was measured again. Natively it gains 2.0–2.2% at depth 0 and 2.8–3.1% at
32K for prompts of 2,560 to 8,192 tokens (before bounded replay: +3.7%).
Served in the default resident setup (native 1M context, one session,
Q2_K/Q4_K drafter), its extra activation memory leaves the prompt cache no
backing: every capture is refused, and a 2K continued prefill gains nothing
(970.3 against 968.5 tok/s). 2048 stays the default; 4096 remains an option where
memory allows a cache beside it (not measured).
Runs are in [the E16 record](artifacts/2026-10-26-prefill-batch/summary.txt).

## SSD expert streaming (M3)

All on the 128.75 GiB `IQ2_XXS` build with a Samsung 990 PRO unless stated;
`gufo bench` pp2048/tg128 with synthetic tokens, alternating binaries and
targeted page-cache eviction of the model shards between runs. The design
as it stands is in [STREAMING.md](STREAMING.md).

### E17: streamed IQ2_XXS reader and mixed-quant kernels (retained)

The first streaming path serialized expert reads and staged all selected
weights. Four parallel readers improved a matched cold pp2048/tg128 from
144.5–144.8 / 5.77–6.01 to 172.6–172.9 / 6.80–7.16 tok/s. Aligned direct
reads improved the four-reader result to 213.8–214.1 / 8.97–8.98 tok/s;
unaligned end-of-file reads and the buffered fallback are covered by the
expert-store test. A repeated warm prefill/decode gives the same rates with
buffered and direct I/O. Four readers beat one, but eight and twelve did not
improve the per-layer direct-read test. `io_uring` and polling offered no
measured benefit over threads in a sustained raw-read control.

The GGUF mixes IQ2_XS with IQ1_M and IQ2_XXS. Its new IQ2_XS FP16 routed
matrix-core kernel halves the measured 140- and 2048-row expert projection
time; on cold pp2048, it gives 217.5–218.1 against 213.9–214.5 tok/s and
preserves the independent 3K scoring quality. Starting selected-expert reads
while the independent shared expert computes raises that result to 219.6–219.8
tok/s and decode from 8.84 to 9.02 tok/s. Both overlap controls have identical
generated-token hashes; a 16-row full-logit and route comparison is byte-exact.
The prior generic and FP16 prompt paths have different rounding, so their
score files are compared to the oracle rather than claimed byte-identical.
[Storage reads](#storage-reads) ·
[quality](QUALITY.md#streamed-iq2_xxs-m3-preliminary).

### Storage reads

*Corrected by E61: the KIOXIA's 2.3 GB/s below was measured through btrfs compression; with compression off for the file it reads 5.15 GB/s with direct I/O.*

October 26, 2026: Samsung 990 PRO, XFS mounted `noatime`, kernel 7.2.11,
NVMe scheduler `none`, 4096 KiB read-ahead; no global kernel or filesystem
setting changed. `tools/ds41/stream_io_bench.py` reads actual IQ2_XXS expert
offsets in 40 layer-sized batches of 1, 2 or 6 missing experts, 1.02 GB of
payload per arm after dropping the shards' clean pages, two rounds in
opposite orders:

| Read mode | Readers | Payload GiB/s, two runs | Median layer ms |
| --- | ---: | ---: | ---: |
| buffered | 1 | 3.04, 3.11 | 5.22, 5.14 |
| buffered | 4 | 5.33, 5.31 | 3.05, 3.08 |
| buffered, `POSIX_FADV_RANDOM` | 4 | 5.27, 5.41 | 2.97, 3.02 |
| direct | 4 | 5.68, 5.89 | 2.65, 2.66 |
| direct | 8 | 5.81, 5.75 | 2.74, 2.73 |
| direct | 12 | 5.63, 5.68 | 2.69, 2.73 |

Four readers reach the useful queue depth; random advice was inconclusive.
The end-to-end effect of direct reads is in E17 above; on a repeated
identical prompt the second prefill (938 tok/s) and decode (13.44 tok/s)
were the same with either mode. Kyojin's
[`uring_bench.c`](https://github.com/Yamz-Labs/kyojin/blob/0c5b91a0686acc8129190f753e032715f0b0dcc2/tools/tier/uring_bench.c)
(3 MiB random direct reads, queue depth four, 16 GiB of one shard, two
seconds) reached about 4.9 GB/s with `pread` threads, `io_uring` and polling
`io_uring` alike (p99 about 3 ms), so `io_uring` was not adopted. The
global scheduler and read-ahead were not tested (not writable by this user).
A `rocprofv3` pp140/tg16 trace with buffered reads showed kernels occupying
1.36 s of the 9.76 s from the first router to the last kernel, the largest
gaps around expert staging: kernel throughput alone does not predict
streamed speed.

**Read-path profile (2026-10-27,** `522c8653`**).** `/proc/<pid>/io` sampling
around the bench phases plus `tools/ds41/engram_row_bench.py`:

| Read path | Pattern | Mechanism | Measured |
| --- | --- | --- | --- |
| Expert weights (demand + prefetch jobs) | per layer, three ~1 GiB blocks (gate, down, up), expert-strided within each | O_DIRECT `pread`, 4 workers, ≤32 MiB chunks | 5.7 GiB/s effective; 8.2 GB/s pure-sequential ceiling |
| Engram rows | 144 B at hashed-random offsets, 2 layers × 24 rows per token | buffered `pread`, ≤32 threads, page cache | 278 K rows/s cold (~1.06 GiB/s at 4 KiB granularity) |

A pp2048 batch moves ~4 GiB of expert reads as ~140 large IOs and 98,304
Engram rows as 4 KiB IOs: the Engram is under 10% of the bytes but ~99% of
the operations. `rchar == read_bytes` for whole runs: once the expert cache
has grown (its chunk growth reclaims exactly the clean file pages), the
Engram's page cache scores no hits, so every prefill batch spends ~0.35 s of
device time on random Engram reads that take queue depth from the expert
reads. This fits the fuller-cache-slower-prefill pattern of the compact
stage arms below, the 98 GiB-cache build's faster prefill and the DSpark
runs (4.7 GiB more free memory).

Two disks, via symlinks to the second NVMe (KIOXIA EXCERIA 1.8T, 2.3 GB/s
sequential against the 990 PRO's 8.2):

- **Engram and shards 3–4 on the KIOXIA: pp2048 113–449 tok/s.** Shards are
  layer-major, so those layers read at the slow disk's pace and gate every
  phase that touches them.
- **Engram alone on the KIOXIA: pp2048 724.9–727.3** (739.9–747.9 on one
  disk), decode unchanged: its random-read latency costs more than the
  contention it removes.

[I/O bench](artifacts/m3-streaming/io-bench.json) ·
[profile](artifacts/m3-streaming/profile-iq2-stream.log) ·
[io_uring control](artifacts/m3-streaming/io-uring-bench.log) ·
[direct-I/O candidate](artifacts/m3-streaming/bench-iq2-direct-io-candidate1.log) ·
[buffered control](artifacts/m3-streaming/bench-iq2-buffered-io-control1.log)

### Overlapped staging (2026-10-27)

Against the October 26 tree (four direct readers, stage copies of resident
experts, shared-expert overlap), same machine, model, histogram sidecar and
`gpu-test` binary family:

| Change | Decision / evidence |
| --- | --- |
| Read resident experts through per-expert pointer tables instead of staging copies | Retained: removes up to ~57 MB/layer of device-to-device copies per decode token. `ds41_gemv_bench` puts the table indirection at parity with the packed layout, bit-identical; `ds41.expert_store` checks table contents, eviction and admission on synthetic shards. |
| Split gate+up launches so resident experts compute while misses stream | Retained: the overlapped launch zeroes the read rows and the post-read launch completes them; each row keeps its single-launch arithmetic (Kyojin skips this lever because its separate launches change the FP32 sum order), so tg128 hashes `04842be7…` on both builds, AR and DSpark (110 drafted / 71 accepted in both). |
| Per-reader upload streams instead of blocking legacy-stream copies | Retained: uploads overlap the layers' kernels. |
| Runtime table-pointer choice inside the expert GEMV kernels | Rejected: rescheduled the IQ2 loops with 249 inserted `s_delay_alu` (914 → 1,141 instructions for IQ2_XXS), −4.2% resident tg128 (24.06 → 23.06 over six alternating pairs). Specializing the indirection at compile time (`kTable`) restores the resident address arithmetic: every `ds41_gemv_bench` case matches (81.1/85.7/88.0/93.7/151.0/146.3/156.2 µs) and resident pairs return to 23.9–24.1 tok/s. |

Matched `gufo bench` (seed 19 s, 82.8% route hits):

| Build | Prefill tok/s | Decode tok/s | DSpark decode tok/s |
| --- | ---: | ---: | ---: |
| October 26 (stage copies, shared overlap) | 592.1 | 11.65 | 13.50 |
| Tables, split overlap, upload streams | 664.7–669.6 | 14.99–15.08 | 17.92–17.96 |

Decode host wait 6.51 → 5.25 ms/token at unchanged route hits, so the gain
is overlap, not cache policy. Resident `IQ1_M` pp2048/tg64 (1070/23.0)
stayed inside the previous build's spread (945–1129 / 22.5–23.7). IQ2_XS
kernel checks: routed FP16 22.7 TOPS at pp2048 shapes (98% of IQ2_XXS),
decode GEMV 112–116 GB/s with the table mode bit-identical. Qualification:
`ds41.expert_store`, `ds41.mmq_typed`, the format check, the native-context
probe (106.4 GiB RSS peak, 8.7 GiB available at 1M context) and the tools,
conversation, sampling-defaults and batch HTTP suites with the Lucebox
drafter. [Logs](artifacts/m3-streaming/bench-iq2-tables-ar1.log)

### Expert GEMV decode retune (2026-10-27)

The IQ2 expert decode kernels ran at 121-134 GB/s, half the dense GEMV's
207-221. Two limiters, found with `tools/prof/isa_mix.py`: the IQ2 decode
cast each `uint64_t` grid entry's extracted byte straight to float, which
lowered to a quarter-rate `v_ldexp_f32` (129 per loop) plus 64-bit shifts,
and the sign ternary emitted a compare/select pair per element; the loads
were also fragmented (five 2-byte weight loads and 16 activation loads per
32-value task in the SwiGLU pass). External references agree on the failure
mode: the sanssushi and MBufo Gufo forks measure 102-118 GB/s on their
per-lane grid-LUT IQ2 decode against a 214 GB/s Q2_0 kernel at the same
216 GB/s peak.

| Experiment | Decision / evidence |
| --- | --- |
| Retuned all five formats in `ds41_expert_gemv.hip.cpp`: task payloads as whole-word loads (memcpy from the 2-byte-aligned blocks), IQ2 grids and signs staged in LDS as u32 word pairs, the IQ2 decode on the entry's u32 halves with signs applied by flipping the float sign bit, and the fused gate/up pass loading x once per block into registers (AccR); U=2 for IQ1_S and the IQ2 formats | Retained, bit-exact by construction and by measurement. Standalone `tools/bench/ds41_expert_gemv_tune.hip` (production TU included, rotating DRAM-cold experts): IQ2_XXS gate/up 143.0 -> 194.9 GB/s, down 123.9 -> 189.4; IQ2_XS 148.9 -> 205.1 / 133.5 -> 189.9; IQ2_S down 143.5 -> 200.7; IQ1_M gate/up 171.5 -> 194.5; IQ1_S down 153.3 -> 162.1 (its gate/up already streamed at 200.3). `ds41_gemv_bench`: fp32 162-201 GB/s, table mode bit-identical; `ds41.mmq_typed` and `ds41.expert_store` pass; 0 VGPR spills (grouped kCols=16 peaks at 218 VGPRs). |
| Fast-math contraction context in the IQ1_M tail | A pragma-free re-expression of `acc + fmaf(s1, d01, s2*(d23))` reproduces the production bits; freezing it with `#pragma clang fp reassociate(off)` does not (production's own emission reassociates). The port keeps the source expression verbatim; the tune bench's replica variant guards it. |
| Alternating matched `gufo bench` pairs (gpu-test binaries, seed 19, two controls and two candidates interleaved) | Streamed IQ2_XXS AR tg128 15.90/16.06 -> 16.57/16.94 tok/s and DSpark 18.54 -> 19.34 (71 accepted and token hash 04842be7… unchanged; 110 -> 107 drafted under the timing-adaptive drafter); decode host wait 5.4-6.5 -> 5.2-5.6 ms/token; resident IQ1_M tg128 23.55/23.36 -> 23.86/24.38 (hash 70d5a48a… unchanged). Prefill untouched (routed_f16 path). |

### Post-retune kernel survey (2026-10-27)

After the expert GEMV retune, a rocprof pass over a streamed DSpark
pp2048/tg64 run (4.15 s GPU busy) ranked every remaining kernel family and
the survey measured each candidate's headroom. The model tensor mixes were
read with llama.cpp's gguf-py: the streamed build's routed experts are
gate/up IQ1_M x16, IQ2_XXS x17, IQ2_XS x7 and down IQ1_M x28, IQ2_XXS x8,
IQ2_XS x4; the resident build gate/up IQ1_S x28, IQ2_XXS x9, IQ1_M x3 and
down IQ1_S x27, IQ1_M x9, IQ2_XXS x2, IQ2_S x2; the Lucebox drafter's
experts are Q4_K x3 and Q2_K x3 of six layers.

| Candidate | Decision / evidence |
| --- | --- |
| Dense Q5_K FP16 prefill (`ds41_dense_f16.hip.cpp`, 493 ms / 11.9% of the run) | Kept. 32.2-33.5 TFLOP/s across q_b/o_b/sh_gate/wo_a (`tools/bench/ds41_dense_f16_tune.hip`), WMMA-issue bound (128 WMMAs against ~2.1k VALU per body, 615 s_delay_alu); the iterative-ILP scheduler flag measured 32.1-33.4 with identical hashes (rejected), and the tile/stage knobs are the ones the Gemma 4 audit already exhausted on the same kernel class at 37-40 TFLOP/s. The remaining ~15% is scheduling overlap, not instructions. |
| LDS-staged activations for the IQ1 down GEMV (162-167 GB/s, the weakest decode kernels; 28 of 40 streamed down layers are IQ1_M) | Rejected: the per-lane 128-byte activation stride makes every read an 8-way LDS bank conflict -- 43-49 GB/s, 4x slower than the global path. The down launches also show a fixed per-launch floor (85-96 us across 13.8-22.7 MB and all five formats), so activation traffic is not what holds them at ~165 GB/s. |
| DSpark drafter expert projections through the vendored mmvq (`mul_mat_vec_q_moe<10>` 49.8 ms + Q4_K paths ~12 ms + q8_1/fp4 passes ~19 ms of the run) | Not attempted: ~80 ms of a 4.15 s run. The drafter is not the DSpark bottleneck; the target's verification and head dominate. |
| Dense Q5_K FP16 prefill: double-buffered activation LDS (`ds41_dense_f16.hip.cpp`) -- stage g stores into buffer g&1 while stage g-1 still reads the other, one `lds_barrier` per K stage instead of two | Retained, bit-exact (output hashes equal at n=2048 and the ragged n=1904 tail, `tools/bench/ds41_dense_f16_variants.hip.cpp`). `ds41_mmq_bench --dense`: q_b 32.9 -> 33.6, o_b 33.5 -> 33.9, sh_gate 27.5 -> 28.1, wo_a 32.4 -> 32.5 TFLOP/s. A 64-token tile was 2.4x slower (matching the Gemma audit's tile sweep) and the iterative-ILP scheduler flag measured 32.1-33.4 with identical hashes (rejected). |
| DSpark drafter routed experts through the FP32 expert GEMV: new Q4_K and Q2_K traits in `ds41_expert_gemv.hip.cpp` (one 16-byte header + one 16-byte payload load per Q4_K task; Q2_K reads its 16 scale bytes at 0, the 32 bit-planed payload bytes of its plane group, and d/dmin at 80/82), and `moe()` fused into one gate/up/SwiGLU plus one down launch | Retained. Kernel level: Q4_K 216-217 GB/s, Q2_K 204-212 GB/s against the FP64 host reference (rel 3e-7, `ds41_expert_gemv_tune`), vs the mmvq path's ~100-150; two launches and the Q8_1 quantization passes drop. E2E (alternating matched pairs): streamed DSpark tg128 19.08/19.15 vs 19.08 (one control pair was cold-cache at 12.3), resident 26.26-26.43 vs 26.34-26.38, AR regression 16.74 vs 16.83 -- parity within noise, as the drafter is ~1.5% of the run. Outputs bit-identical (04842be7... / 70d5a48a...), acceptance identical (71/51); sampled sessions will draw slightly different drafts (FP32 vs Q8_1 activations), which verification keeps correct. |
| Q2_K trait first cut | Rejected and fixed: it assumed {d, dmin, scales, qs} with consecutive payload packing; the real block (`block_q2_K`) has the scales at 0, bit-planed qs at 16 (value v's bits in byte 16 + v%32 + 32*(v/128), plane 2*((v/32)&3)) and d/dmin at 80/82. The standalone FP64 check had passed because its reference and synthetic generator repeated the same wrong layout; kernel references are now checked against the vendored struct or gguf-py. |
| Persistent row-tiles / wider workgroups for the IQ1 down GEMV | Rejected / not ported: row-tiles (2 or 4 per workgroup) serialize work behind one table fill and lose 8-12% to reduced parallelism; 512-thread workgroups gain only the IQ1 downs 1.3-3.5% (the measured empty-launch floor is ~3 ns/block, 17.8 us for the 5760-block down launch). An LDS-staged activation variant was 4x slower: the per-lane 128-byte stride is an 8-way LDS bank conflict. |
| Output head and dense decode GEMV, attention prefill/decode, routed F16 prefill | Kept. The head streams 227-237 GB/s (near the 239.7 measured peak); the routed F16 decodes through LDS FP16-packed codebooks with XOR signs at fork-grade (22.8-26.5 TOPS, no slow-opcode pathologies in isa_mix); attention carries its own retained tuning history. |

### Prefill read pipelining with router predictions (2026-10-27)

Reads of a prefill layer's missing experts start during the previous
layer's compute: the previous layer's read overlap runs this layer's router
on its own FFN input (one layer stale), and the predicted experts' reads run
as a background job that the layer's prepare adopts for the experts it
routes ([design](STREAMING.md#prefetch-from-router-predictions)). This first
version gave background jobs a second whole-layer stage set in layer parity
(~3.6 GiB, bought after the seed), so the runtime cache grew to 98.0 instead
of 101.6 GiB.

A router-free sweep of the next layer's uncached experts, as ds4's Strix
Halo path does, was measured first and rejected on this model: V4.1 Flash routing is skewed enough that a
2048-token prompt touches only a fraction of each layer's experts, and with
the ranked seed resident the sweep read 3.6x the demand volume of the first
pass (22.9 vs 6.4 GiB), dropping pp2048 to 389.61/386.69 tok/s against
664.03/661.93 for the control. The predicted job reads ~1.6x the demand
volume, and the overlap absorbs it.

Matched `gufo bench` arms (pp2048/tg128, alternating binaries, targeted
shard page eviction between runs, two rounds each). The tg128 token hash
04842be7… and DSpark's 71 accepted of 107 drafted are identical on every
arm; the streamed pipeline changes only when bytes arrive, never which
bytes the kernels see.

| Build | AR pp2048 | AR tg128 | DSpark pp2048 | DSpark tg128 |
| --- | ---: | ---: | ---: | ---: |
| control (028086ab) | 652.6–663.6 | 16.53–16.59 | 616.7–620.5 | 19.05–19.11 |
| prefill prediction | 779.3–785.1 | 15.75–15.94 | 737.2–748.0 | 17.78–17.91 |

Prefill gains 19–21% on both paths. Decode keeps its kernels and its host
wait (4.88–4.93 ms/token on every arm) but loses 4–6% to the smaller cache:
the second stage set is the price of the pipelining, which the compact
stage below recovers.

[Router-free sweep, candidate](artifacts/m3-streaming/bench-iq2-routersweep-cand1.log) ·
[Router-free sweep, control](artifacts/m3-streaming/bench-iq2-routersweep-ctrl1.log) ·
[Prefill prediction, AR](artifacts/m3-streaming/bench-iq2-prefetch-ar-cand1.log) ·
[Prefill prediction, AR control](artifacts/m3-streaming/bench-iq2-prefetch-ar-ctrl1.log) ·
[Prefill prediction, DSpark](artifacts/m3-streaming/bench-iq2-prefetch-ds-cand1.log) ·
[Prefill prediction, DSpark control](artifacts/m3-streaming/bench-iq2-prefetch-ds-ctrl1.log)

### Streamed decode prefetch (2026-10-27)

Single-token steps use the same pipeline: when a layer's
read overlap actually waited on reads, it enqueues the next layer's router
on the current FFN input, and the store thread submits the predicted
experts' reads, which the next layer's prepare adopts for the experts it
routes. Layers without misses predict nothing — a prediction without a
read stall to hide its matvec costs more than the read it saves — and
adoption skips unrouted experts, so wrong predictions cost bandwidth only.

Matched arms against the prefill-pipelining build (same protocol, two rounds; the
token hash 04842be7… and DSpark's 71/107 are identical on every arm):

| Build | AR pp2048 | AR tg128 | DSpark pp2048 | DSpark tg128 |
| --- | ---: | ---: | ---: | ---: |
| prefill prediction only | 789.1–792.4 | 15.78–15.79 | 746.7–748.2 | 17.83–17.86 |
| with decode prediction | 778.7–788.0 | 15.91–15.92 | 748.4–750.2 | 17.84–17.89 |

Decode gains 0.8% on AR (every round separated well outside the ±0.01
run-to-run spread) and is neutral on DSpark, whose verification batches of
2–16 rows predict nothing by design. The gain is small because at 82–83%
route hits only ~5 ms/token of read stall exists and one-layer-stale recall
covers part of it; the larger decode lever was the cache capacity the second
stage set consumed (next section). Decode reads rise from 11.99 to
13.2 MiB/token (10.9 vs 4.1 GiB prefetched overall), the price of the
predictions that miss.

[Decode prediction, AR](artifacts/m3-streaming/bench-iq2-decode-ar-cand1.log) ·
[Decode prediction, AR control](artifacts/m3-streaming/bench-iq2-decode-ar-ctrl1.log) ·
[Decode prediction, DSpark](artifacts/m3-streaming/bench-iq2-decode-ds-cand1.log) ·
[Decode prediction, DSpark control](artifacts/m3-streaming/bench-iq2-decode-ds-ctrl1.log)

### Compact prefetch stage and cache reserve (2026-10-27)

The decode dip of the prefill pipelining came from the second stage set
taking ~3.6 GiB of expert-cache headroom; two changes give the memory back.

- **The prefetch stage no longer mirrors the demand stage.** Demand reads
  were always safe in one whole-layer buffer — the blocking routes copy
  orders them behind the previous layer's compute — so only the background
  jobs needed a second buffer, and jobs touch just the predicted experts.
  The prefetch stage is now a 192-row compact buffer with two 96-row parity
  halves (~1.5 GiB): a job's experts take rows in its layer's half, a job
  beyond 96 experts truncates (the excess stays a demand read), and a failed
  job is simply not adopted. Resolving the job moved ahead of the phase's
  table build so the table can point adopted experts at their rows.
- **The cache reserve drops from 8 GiB to 4 GiB.** The resident path runs
  with a 1 GiB dispatch reserve under the same loads; the streamed reserve
  now covers the shared batch scratch (1.4 GiB at 1M context), one session's
  FP4 state and transient peaks. 3 GiB measured one 572 tok/s prefill round
  riding the reclaim floor; 4 GiB is tight across every round.

Matched arms against the decode-prefetch build (same protocol). The cache reaches
103.8 GiB (AR) / 99.1 GiB (DSpark, which loads the drafter); hashes and
DSpark's 71/107 identical on every arm:

| Build | AR pp2048 | AR tg128 | DSpark pp2048 | DSpark tg128 |
| --- | ---: | ---: | ---: | ---: |
| decode prefetch (8 GiB reserve, whole-layer set) | 787.5–788.0 | 15.86–15.92 | 744.5–749.3 | 17.87–17.91 |
| compact set, 4 GiB reserve | 739.9–747.9 | 17.34–17.38 | 763.9–765.1 | 19.03–19.07 |
| original control (028086ab) | 652.6–663.6 | 16.53–16.59 | 616.7–620.5 | 19.05–19.11 |

Decode recovers past the original control (+4.8% AR, parity on DSpark, from
4–6% below it), and DSpark prefill improves. AR prefill sits 5% under the
decode-prefetch build while DSpark prefill gains: across the pipelined builds the
AR prefill rate anti-correlates with the route-hit rate (88.3% hits → 787,
95.3% → 742 tok/s), so the residual is a residency effect (the Engram reads of the
[read-path profile](#storage-reads)), not a pipeline cost — reads fall from 13.2 to 9.3 MiB/token on the same
arms.

[Compact stage, AR](artifacts/m3-streaming/bench-iq2-compact-ar-cand1.log) ·
[Compact stage, DSpark](artifacts/m3-streaming/bench-iq2-compact-ds-cand1.log)

### Opt-in streaming, IQ1_M and the compact stage stride (2026-10-27)

Streaming became opt-in (`--stream-experts`, `--expert-cache-bytes`); a
model that does not fit now fails to load instead of streaming silently.
While streaming, the 5.31 GiB of dense weights go through the device heap
instead of managed memory.

**Compact stage stride.** The 3K teacher-forced score of the streamed
IQ2_XXS build had fallen from 6.0041 (`4b5ee8ea`) to 35.83 (`10ad16a4`).
Bisection put it on the compact prefetch stage (`522c8653`): disabling
prefetch or wide jobs restored 6.0041, and so did a whole-layer stage, also
with jobs still truncated at 96 experts. The stage is sized by the largest
expert of any layer, but rows were addressed with the current layer's
expert size; the build mixes 56-, 66- and 74-byte block types, so a layer
with large experts ran past the start of the other parity half, and the next
layer's job overwrote rows the current layer was still reading. Rows now use
the allocation stride. The streamed score is 6.0041 again, identical to
`4b5ee8ea` on all 2,999 positions. `ds41.expert_store` gained a mixed-size
fixture (odd layers half the size) that fails without the fix. `gufo bench`
token hashes did not catch it: their synthetic prompts score no text.

**IQ1_M streamed.** The IQ1_M build has 17 layers of IQ1_S experts and two
IQ2_S down projections, which the streamed batched path had never run.
IQ2_S has no FP16 tile kernel, so those rows go through the table-indexed
expert GEMV; the store also skips dropped routes (-1) as the resident
kernels do, instead of failing the batch. Teacher-forced 3K, 2048- and
952-token exact batches:

| IQ1_M path | Perplexity | Target top-1 | Agreement with resident |
| --- | ---: | ---: | ---: |
| Resident | 6.320 | 57.7% | — |
| Streamed, 32 GiB expert cache | 6.303 | 57.8% | 96.1% |

Single-token decode scores are bit-identical to resident on 256 positions;
the batched difference is the FP16 tile and GEMV rounding seen between the
resident batched and decode paths.

`gufo bench` pp2048/tg128 on IQ1_M, with the memory free once `gufo serve`
is ready at 262,144 tokens (no drafter):

| Expert cache | Free memory | pp2048 | tg128 |
| --- | ---: | ---: | ---: |
| Resident | 5.6 GiB | 1,212.3 ± 33.9 | 24.44 ± 0.13 |
| 64 GiB | 48.6 GiB | 616.2 ± 29.4 | 16.45 ± 3.19 |
| 32 GiB | 78.8 GiB | 437.2 ± 5.8 | 12.17 ± 2.34 |
| 16 GiB | 93.3 GiB | 374.4 ± 1.3 | 9.13 ± 1.23 |

Streamed IQ2_XXS against the pre-merge build with the same stride fix,
alternated twice: tg128 18.81/19.13 against 19.13/18.96 tok/s, no
measurable change from the heap placement. Prefill was inconclusive
(pp2048 853/927 against 735/953, each ± 300-460): the startup seed caches
the 10,240 ranked experts (82 GiB) of 15,360, so the first repetition reads
the rest (81% route hits) and the second hits on every route.

### Slots per expert size and the full built-in ranking (2026-10-27)

Two changes let a capped cache hold more experts; neither touches the
arithmetic.

- **One slot size per expert size.** Every slot was sized for the largest
  expert of any layer, but the builds mix block types: IQ1_M has six expert
  sizes (6.59-8.26 MiB, mean 7.11) and IQ2_XXS eight (7.38-9.76 MiB, mean
  8.20), so 14% and 16% of the cache went unused. Slots now come in one
  class per distinct size, each growing by 128 MiB chunks (512 MiB before,
  to bound what a class strands in a part-used chunk). Once the cache stops
  growing, an expert may take the smallest larger free slot or evict from
  its own or a larger class, never a smaller one.
- **The built-in ranking orders all 15,360 experts.** The import kept the
  first 10,240 of the source permutation, so a cache larger than 82 GiB
  (IQ2_XXS) started part empty and the first request read the rest. The
  table now keeps the whole ranking; the first 10,240 entries are
  unchanged. Experts a custom histogram omits follow its ranked ones,
  interleaved across layers.

`ds41.expert_store` gained cases for both (seven half-size experts in the
cache of four full-size ones, a large expert that streams rather than take
a small slot, and the seed past a one-expert histogram); they fail on the
previous store.

`gufo bench` pp2048/tg128 (`-r 2`), against the opt-in build above:

| Model, expert cache | Experts seeded | pp2048 | tg128 | Route hits |
| --- | ---: | ---: | ---: | ---: |
| IQ1_M, 64 GiB | 7,436 → 8,658 | 616.2 → 751.6 ± 98.1 | 16.45 → 17.90 | 70.8% → 85.7% |
| IQ1_M, 32 GiB | 3,718 → 4,325 | 437.2 → 462.5 ± 1.8 | 12.17 → 12.94 | 45.5% → 48.6% |
| IQ1_M, 16 GiB | 1,859 → 2,155 | 374.4 → 383.6 ± 2.0 | 9.13 → 9.74 | 30.2% → 32.7% |
| IQ2_XXS, automatic (105 GiB) | 10,240 → 12,688 | 853/927 → 918.9 ± 403.5 | 18.81/19.13 → 19.96 | 91.4% → 97.0% |

Free memory is unchanged: the cap bounds the allocation either way. The
seed takes longer in proportion (13.1 → 14.6 s at 64 GiB, 18.7 → 20.2 s
for IQ2_XXS). The IQ2_XXS prefill spread no longer comes from expert reads
(1.57 MiB/token in the first repetition, 93.6% hits); the Engram row reads
([read-path profile](#storage-reads)), whose page cache the full expert
cache displaces, are the likely remainder.

### Decode profile and per-layer expert tables (2026-10-27)

A rocprofv3 trace of warm IQ2_XXS decode (119 tokens of tg128 after
pp512, automatic cache, medians per token) splits the 50.1 ms token into
40.3 ms of kernels and 9.8 ms of GPU idle:

| Part | ms/token |
| --- | ---: |
| Dense Q5_K GEMV (`gemv_kernel<13,…>`) | 22.2 |
| Routed expert GEMV | 11.5 |
| Other kernels (attention, hyper-connections, router, …) | 6.6 |
| Idle: miss reads between the hit and miss launches (~4 per token) | 5.3 |
| Idle: ~200 small blocking copies | ~3 |

The dense GEMV runs near the measured bandwidth ceiling, so speculative
verification, which reads it once for several tokens, is the larger decode
lever. Timers around the Engram futures and the read phases put the Engram
wait at ~0.1 ms per pp2048 batch and the first prefill's expert read wait at
0.35 s (0 once warm): the Engram is not what slows a new prompt.

The copies were each layer's three expert tables and two hit/miss row
arrays, uploaded with blocking `hipMemcpy`. The store now keeps one device
table per layer and uploads it asynchronously only when one of its entries
changed; a warm cache almost never changes one. The hit/miss rows are
uploaded, in one asynchronous copy, only for a batch that both hits and
misses — the only case whose launches read them. The routes copy that
starts the next prepare orders every upload before its pinned mirror is
rewritten. Copies per token fell from ~200 to 23 (the slot fills after
reads), GPU idle from 9.8 to 8.1 ms and the traced token from 50.1 to
48.1 ms.

Alternated `gufo bench` (pp2048/tg128, `-r 2`): tg128 20.23/20.40 →
20.93/20.95 tok/s (+3.2%); pp2048 985.9/1061.2 → 1074.3/1077.2, inside its
first-repetition spread. Streamed IQ2_XXS decode scores are bit-identical
on 256 positions with the automatic cache and with a 16 GiB cache (71%
route hits, so most layers split).

### DSpark on streamed IQ2_XXS (2026-10-27)

`gufo bench` pp512/tg512 with the Lucebox Q2_K/Q4_K drafter: 33.5 tok/s
against 21.3 plain (+58%). Route hits settle at ~99.3% either way, but
verification reads are never predicted (0% prefetched), so ~13 MiB/token
of misses remain. A rocprofv3 trace of the second half of a DSpark tg512
splits the span into routed experts 44% (36% the 8-row verification
groups: ~36 distinct experts per layer at ~32 µs each, the single-token
per-expert cost, so bandwidth-bound), dense GEMV 31% (8-row launches at
1.4-1.8× the single-token time), other kernels 11%, idle 10% (about half
of it miss reads) and the drafter 4%.

The task-major multi-token dense GEMV ([resident engine
table](#resident-engine-m1m5)) took 8-row verification steps 129.7 →
118.8 ms on resident IQ1_M; streamed DSpark tg512 stayed inside its
cache-warming spread (33.17–34.14 → 33.15–34.73 tok/s over three
alternating pairs).

### E18: cost-aware draft prefix (retained)

Greedy cycles with the learned head now verify the draft prefix that
maximizes expected emitted tokens per cycle time, using the head's
predictions as acceptance probabilities and the session's measured cycle
times by width (the timing policy's estimates). Every emitted token still
takes exactly one draw at its position, so outputs do not depend on the
width. A cycle's time error also moves the other widths' estimates by 5%;
without that, widths the cut seldom picks kept their priors.

Per-cycle traces (all five drafts, the head's predictions and the decoded
sequence) give each draft's uncensored acceptance. The head is close to
calibrated (0.9+ rows accepted ≥ 99%, 0.5 rows about 55%). Verification
times by rows (`--spec-bench`) are 43.4, 58.0, 69.8, 78.8, 96.6, 107.5,
120.4 and 133.8 ms on streamed IQ2_XXS for 1-8 rows; 41.1, 52.8, 65.5,
75.7, 87.0, 98.3, 113.5 and 123.6 ms on resident IQ1_M. Replayed over five
prompts against the ≥ 0.5 cut, expected tokens per time gains +3.1%
streamed and +0.9% resident; a cumulative-confidence cut gains less
(+2.4%), any fixed width loses.

Matched A/B, five prompts × two alternating repetitions of 512 greedy
tokens against `d258afc1`:

| Model | Geomean | Faster | Base → candidate tok/s range |
| --- | --- | --- | --- |
| Streamed IQ2_XXS | +3.76% | 10/10 | 29.37-38.87 → 30.37-40.03 |
| Resident IQ1_M | +0.65% | 8/10 | 31.15-43.74 → 31.39-43.65 |

All 40 runs match plain decoding token for token. Fewer drafts are verified
(streamed 404-440 instead of 468-492 per 512 tokens) at higher acceptance.
The coding prompt, already at 84-88% acceptance with full blocks, measured
−1.1% and −0.2% resident. Rejected: the same cut with per-width estimates
only (+2.91% streamed, −1.87% resident, 0/10 faster): the 1- and 3-draft
estimates stayed at their 69.6 and 95.5 ms priors against about 62 and
86 ms measured, so those widths were almost never chosen. Runs, traces and
the replay are in [the E18 record](artifacts/2026-10-27-cost-aware-draft-prefix/summary.txt).

## Prefill at depth (2026-10-27)

Native pp2048 on resident IQ1_M fell from 1,126 tok/s at depth 0 to 1,017
at 64K and 838 at 128K. A rocprofv3 trace of 128K of depth preparation put
the growth in two kernels on encoder layers 2, 8 and 14, which score every
prompt row against depth / 2 compressed index keys: per 2,048-token chunk the
indexer scores grew 4 → 227 ms and the top-k 1 → 270 ms by ~115K, while
attention, projections and mHC stayed flat. Both kernels serve the streamed
`IQ2_XXS` model unchanged. [Record](artifacts/2026-10-27-indexer-depth/summary.txt).

| Depth | Base | E19 | E19 + E20 | E19–E21 |
| ---: | ---: | ---: | ---: | ---: |
| 32K | 1,133 | 1,161 | 1,182 | 1,217 |
| 64K | 1,017 | 1,109 | 1,155 | 1,185 |
| 128K | 838 | 1,003 | 1,098 | 1,126 |

Depth 0 is the first test of each process and reads low in this harness
(1,126, 1,161, 980 and 1,122 across the four builds); a matched d0 A/B of
E19 + E20 against the base sat inside the runs' own spread.

### E19: two-pass indexer top-k (retained)

The radix select read each token's score row six times with per-thread
contiguous slices (uncoalesced) and histogrammed the top byte, where nearly
every score shares one of a few bins of LDS atomics. Now one coalesced pass
builds a 12-bit histogram (narrowed only while the k-th key's bin holds more
than 2,048 keys), one ordered pass appends the higher bins' indices and the
bin's keys to shared lists, the exact key and its ties are resolved in
shared memory, and two index-ordered lists are merged. The output keeps the
old order: keys above the k-th in index order, then the first ties (not one
sorted list; a first draft that sorted everything differed). Per index layer
at 128K: 102.3 → 4.7 ms; at depth 0, 0.29 → 0.48 ms. Byte-identical on the
128K/64K dumps and on 3,400 adversarial rows against the old kernel.

### E20: register-resident keys for the WMMA indexer scores (retained)

Each block covered 16 tokens and 128 compressed rows and converted every
head's FP32 queries again, about 16 GB of query reads per layer at 128K.
Now each of 16 waves keeps its 16 rows' keys as binary16 A fragments in
registers, a block covers 256 rows, the next head's queries are loaded
before the current head's products and stored after them, and the scores
leave through shared memory as contiguous rows. rocwmma's fragments had
held each 16-dim step's two halves swapped in the upper half-wave, which
changes the odd rows' summation order; the raw-builtin kernel mirrors that
(selecting in registers, so both half-waves keep reading the same LDS
addresses) and stays byte-identical. Per layer at 128K: 84.5 → 39.9 ms
(13.1 → 27.8 TFLOP/s); 2× at every depth. Rejected variants: 512-row blocks
(45.7 ms), two 16-token tiles per block (50.4), interleaved head pairs
(56.2, register pressure), head weights from LDS (42.5), ReLU as a
compare-select (no change), a second pre-swapped query copy in LDS for the
upper half-wave (40.9: twice the LDS reads), two 16-row tiles per wave with
8 or 16 waves (47.7, 44.3).

### E21: prefill attention staging (retained)

`attention_f16_kernel` (20 layers, ~207 ms per 2K batch at any depth past a
few thousand tokens) read each compressed row's index from global memory
right before that row's bytes, a chain of dependent loads per 16-row tile;
branched on window versus compressed rows once per row; and addressed rows
with 64-bit per-lane arithmetic. The token's indices now sit in shared
memory, the branch is taken once per tile, and rows are 32-bit offsets from
uniform bases (compressed rows stay below 2^24 at any supported context).
Per layer (2,048 tokens at 128K, `ds41_attention_test --bench`): 10.52 →
10.03 → 9.20 → 8.48 ms, and 12–13% at depth 0, ratio 1 and the 140-row
decoder replay. Outputs byte-identical to the base kernel on four shapes
(ratios 2, 1 and 0). Rejected: the two head groups of a token adjacent in
the grid (no change), the rows' scale bytes staged in shared memory
(10.3 ms), skipping unit softmax rescales (9.8).

Also measured on the shared FP16 projection kernels and rejected: dense
Q5_K activation fragments read twice from LDS instead of exchanged with
`permlanex16` (33 → 28 TFLOP/s, LDS-bound); the dense kernel without its
per-tile guard on full tiles (spills, as its comment warns); routed
activations prefetched a stage ahead in registers (IQ1_S and IQ2_XXS both
~5% slower).

### E22: the FFN's closing expand fused with the next state pass (retained)

Each batched block ended with `hc_expand_sum` (the six routed rows, the
shared output and the residual into the next state) and the next block began with a state pass that read that state
again for its attention mixes and collapse. `hc_state_pass` now also takes
the rows to sum, so a block whose successor runs on the same rows (no Engram
add or replay boundary between them) closes with the successor's attention
state pass. Per layer at 2,048 tokens (4,096 channels in the
microbenchmark): 3.40 → 2.71 ms, byte-identical state,
mixes and collapse (also 1,904 and 17 rows). With bounded replay most
batches run only the 20 encoder layers: pp2048 at d16384 1,225.6 → 1,229.1
tok/s (two alternating runs each), greedy text identical with and without
`--exact-prefill`.

Rejected on the indexer scores (E20's kernel, 39.9 ms per layer at 128K):
two or four heads' queries staged per barrier (48.7 / 44.5 ms), the head
weights loaded one head ahead (42.7), and, as an upper bound, the upper
half-wave's B operand without its swap (41.1, and not byte-identical). The
time is linear in heads: about 6 ms fixed (writing the scores, decoding the
keys) plus 1.05 ms per head (33 TFLOP/s in the head loop).

### E23: FP16 inputs for the dense FP16 projections (retained)

Measured per launch at 2,048 tokens (`tools/bench/ds41_dense_f16_tune.hip`,
whose `wo_a` shape was 8× too large before: it is 1,024 rows × 4,096 per
group, 8 groups): `wo_a` 5.50 ms (25 TFLOP/s) and the shared expert's
2,304-row gate and up 1.75 ms each (28) against 33–34 for `q_b`, `o_b`
and the shared down. A flat 8,192-row matrix over the same 4,096-wide input
takes 4.27 ms: the grouped input (268 MB of FP32, re-read by each 256-row
block) is what costs. Dropping the activation loads takes `wo_a` to 3.96 ms
and the per-superblock weight loads to 4.39.

The kernel already rounds its activations to FP16 as it stages them, so FP16
inputs give the same products. A first FP16-input build was 15–20% slower on
every shape: its two-entry `uint4` prefetch arrays went to scratch memory (80
bytes per lane, 38 scratch accesses per kernel). As `ext_vector_type` arrays
they stay in registers, and FP16 rows (half the registers) run four 64-value
stages ahead instead of two:

| Shape | FP32 input | FP16 input |
| --- | ---: | ---: |
| `q_b` 32,768 × 1,280 | 5.13 ms | 4.95 ms |
| `o_b` 5,120 × 8,192 | 5.04 | 4.66 |
| shared gate 2,304 × 5,120 | 1.72 | 1.44 |
| shared down 5,120 × 2,304 | 1.41 | 1.37 |
| `wo_a` 8 × 1,024 × 4,096 | 5.32 | 4.25 |

The producers now write FP16 where the dense kernel is the only reader: the
matrix-core attention stores its FP8-rounded output as FP16 rows for `wo_a`,
`wo_a` stores FP16 for `o_b`, and the shared expert's gate/up read the FP16
FFN input the routed FP16 kernels already convert (now converted before the
streamed store's overlap, which runs the shared expert). Every path is
bit-identical: `ds41.mmq_typed` checks the three FP16 modes against the FP32
launch and `ds41.attention` the FP16 attention output against the rounded
FP32 one; the 3,000-position batched score (`ds41_load_probe --batch
--score`) gives the same argmax and NLL at every position. pp2048 at d16384:
1,234.4 → 1,281.4 tok/s (+3.8%, two alternating runs each). Streamed
IQ2_XXS gives the same 3,000-position scores before and after (perplexity
6.004 against the oracle's 6.028, argmax agreement 95.6%) and batches 2,048
tokens in 4,252 instead of 4,901 ms.

Rejected on the way: 128- and 512-row dense blocks (every shape slower,
`wo_a` 6.23 / 5.88 ms), neighbouring lanes on neighbouring 16-byte chunks of
a token's activation row (no change), and Gemma 4's binary16 K-quant GEMM,
which runs these shapes 10–30% faster with FP16 inputs but sums the upper
half-wave's products in a different order.

Profile of the 128K chunk afterwards: GPU time per 2,048-token chunk 1,481
→ 1,380 ms; routed experts 488 ms (35%), dense FP16 projections 393 (28%),
attention 163, the state passes 106, indexer scores 61, MMQ (Q8_0 `q_a`/kv)
50, the BF16 matmuls 49.

Also measured and rejected: the routed FP16 kernel's activations prefetched
a stage ahead in vector registers (IQ1_S gate+up 21.6 → 22.3 ms, IQ2_XXS
25.0 → 24.9; its arrays never went to scratch). Ablations of that kernel
(`ds41_mmq_bench --f16 1`, 2,048 tokens): without the matrix-core products
gate+up takes 11.5 ms for both types, without the weight decode 20.5
(IQ1_S) and 18.9 (IQ2_XXS): the products and the rest barely overlap, and
IQ2_XXS's two table lookups per eight weights cost about a quarter. The BF16
matmul (`matmul_bf16_wmma_kernel`, three BF16 terms per activation, 19–24
TFLOP/s counting all three) with the next K step's operands loaded during
the current one: bit-identical, router 1.27 → 1.67 ms, and 1.46 with
LDS-only barriers.

### E24: half-wave permutes without register copies (retained)

The FP16 kernels' `v_permlanex16` swaps between half-waves passed
`fetch_inactive` and `bound_ctrl` false. The instruction's `old` operand is
then tied to the destination, so the compiler copied (or zeroed) a register
before every permute. With both flags set, the instruction reads its source
lanes directly and the copies go away. Every lane is active in these
kernels, so the results do not change. Dense FP16 projections got 4–5%
faster per shape, routed FP16 gate/up/down 0.5–3%, and the matrix-core
attention 8.47 → 8.34 ms at 2,048 tokens. The 3,000-position batched scores
are byte-identical for IQ1_M and streamed IQ2_XXS. pp2048 at d16384:
1,284.9 → 1,307.6 tok/s (+1.8%, two alternating runs each).

### E25: routed FP16 staging and LDS lookups (retained)

Ablations of the routed FP16 gate+up (`ds41_mmq_bench --f16 1`, 2,048
tokens, 32 or 48 slots per expert) split IQ1_M's 21.8 ms into the
matrix-core skeleton (14.7 ms: block fields, fragment reads, products), the
weight loads (+1.0), the activation staging and its two barriers per stage
(+1.7; +4.9 for IQ2_XXS) and the weight decode with its permutes (+4.4).
Making the codebook lookups conflict-free (every lane reading the same entry)
took IQ2_XXS from 24.5 to 20.4 ms and IQ1_M from 21.2 to 20.2: random
16-byte LDS rows cost about 2 ms per IQ2_XXS table.

Retained, each byte-identical on the 3,000-position batched scores of IQ1_M
and streamed IQ2_XXS:

- Double-buffered 64-value activation stages, the next one loaded before the
  current stage's products and stored after them, one barrier per stage:
  gate+up 21.84 → 21.09 ms (IQ1_M), 25.07 → 24.42 (IQ2_XXS), 21.50 → 20.79
  (IQ1_S), down 4–5% faster. 128-value buffers were neutral (51 KB of LDS
  took IQ1 from 6 to 4 waves per SIMD); 32-value stages gained 1.5% on IQ1
  and lost 2.5% on IQ2_XXS. IQ1_M pp2048 at d16384: 1,310.0 → 1,317.2 tok/s.
- IQ2 sign indices as eight bytes (0x80 per negative weight) placed by one
  `v_perm` per word instead of 16-byte FP16 masks: IQ2_XXS gate+up 24.46 →
  22.72 ms, IQ2_XS 24.82 → 23.21. IQ1 keeps the raw 4-byte codebook entries
  and maps their nibbles to FP16 high bytes with a `v_perm` lookup (half the
  table): gate+up unchanged, down 1–5% faster. Streamed IQ2_XXS pp2048 at
  d16384: 1,268.3 → 1,281.9 tok/s.
- For the IQ2 formats, each half-wave reads its own 16 bytes of an
  activation fragment and permutes the other half across lanes (the dense
  kernel's scheme): IQ2_XXS 22.82 → 21.93 ms, IQ2_XS 23.23 → 22.45. It costs
  the VALU-bound IQ1 formats 3–7%, which keep two LDS reads. Streamed
  IQ2_XXS pp2048 at d16384: 1,284.4 → 1,292.7 tok/s.

Rejected: the fragments of all four tiles read before the decode
(IQ1_M 21.2 → 25.8 ms: dead tiles' reads cost LDS bandwidth), and only live
tiles' fragments read before the decode (22.6). In the dense FP16 kernel,
two LDS reads in place of the fragment permutes were 25% slower on every
shape (it is bound by fragment reads), and waves of 32 rows × 64 tokens
(two B operands, half the fragment reads per product) were 5–10% slower
(238 registers, 6 instead of 8 waves per SIMD, twice the decode).

### E26: Engram staging, FP16 `q_b` rows and 128-row BF16 tiles (retained)

A 128K-depth profile after E25 (1,342 ms of GPU time per 2,048-token chunk)
left 18.8% of the wall span idle. Timing `eval_rows` showed 16 ms per chunk
before the first launch, spent allocating and zero-filling three host
vectors for the Engram rows (`n × 6,144` floats per Engram layer) that then
went up with synchronous pageable copies, draining the queue at each Engram
layer. The pinned Engram staging now holds a full batch (capacity rows), so
every batch's rows are read straight into it and uploaded asynchronously:
pp2048 at d16384 1,325.4 → 1,347.9 tok/s (+1.7%), byte-identical scores.
Between chunks only the embedding step remains (host dequantization and a
synchronous upload, 5–8 ms).

`q_kv_norm` also stores `qr` as FP16 rows (in the attention output buffer,
idle until the attention writes it) for the dense `q_b` projection, which
rounded them the same way when staging; the BF16 indexer projection keeps
the FP32 rows. `ds41.attention` checks the FP16 copy bit for bit. pp2048 at
d16384: 1,352.6 → 1,357.1 tok/s.

The batched BF16 matmul takes 128 weight rows per workgroup when the row
count allows (8 waves of 16 rows × 64 tokens), so each activation step is
split into its three BF16 terms once per 128 rows instead of 64. Every
output keeps the same products and sums (identical output hashes):

| Shape (2,048 tokens) | 64-row tiles | 128-row tiles |
| --- | ---: | ---: |
| router 384 × 5,120 | 1.25 ms | 0.98 ms |
| compressor 512 × 5,120 | 1.52 | 1.51 |
| compressor 1,024 × 5,120 | 2.70 | 2.54 |
| indexer `q_b` 4,096 × 1,280 | 2.56 | 2.21 |

In the model the BF16 matmuls of three d4096 chunks took 163 → 139 ms
(total GPU time −0.65%), below the end-to-end noise of pp2048.

Rejected: 32-token BF16 tiles (twice the workgroups, every shape 17–30%
slower), and skipping the attention output's softmax rescale when every
factor of a wave is exactly 1 (a uniform branch around the multiplies:
8.36 → 9.2 ms; interleaved with the products they cost nothing). Also
measured: the Engram projection is 25,600 × 6,144 (17.8 ms per call, 36
TFLOP/s; the tune bench had a 1,280-wide input), and FP16 rows would save
0.8 ms of it.

### E27: one joint verification for concurrent DSpark sessions (retained)

Concurrent sessions verified jointly in groups of up to 16 rows (E14), but
each still drafted a full block, so from four users a step needed two or
more joint evaluations and mixed text decoded slower than AR. A batch now
gives each session at most its even share of one 16-row evaluation
(16 / users − 1 copies or greedy drafts), plus the rows earlier sessions
left, and verifies anchors only past eight users. Cycles in a batch no
longer feed the batch's time into a session's own width timings. Sampled
drafts keep their own width: a shorter block consumes different draws, and
seeded outputs must not depend on the batch (the `batch` suite's
cross-endpoint replay caught this).

On the card workload (the same prompt for every user, greedy) mixed text
rises 42.6 / 43.7 / 44.7 → 51.4 / 55.0 / 57.8 tok/s at C4 / C6 / C8 and
repetitive text 75.4 / 74.9 / 74.8 → 89.2 / 97.4 / 92.3, every completion
identical to the isolated AR reference. With a different prompt per user
(11-prompt corpus) it rises 40.6 / 42.1 → 47.6 / 51.5 at C4 / C8, against
46.0 / 54.6 for AR.

Two policies on top were measured and not kept. Trimming drafts after
drafting by an expected-tokens-per-ms model (v2) stayed within noise. A
batch plan before drafting (v3), from measured drafter (8–10 ms per session,
run one after another) and joint pass costs, skips drafting where it does
not pay: +6–13% on the card's mixed text, −4–7% on its repetitive text, and
a tie on distinct prompts (47.3 / 52.0). The card's identical prompts keep
v1's sessions in lockstep, so a 16-row pass holds four copies of the same
tokens and shares their experts (166 against 185 ms once sessions drift);
that lockstep, not the policy, explains v1's repetitive lead. The remaining
cost is the serial drafter forwards ([backlog](BACKLOG.md) item 10).
Runs are in [the E27 record](artifacts/2026-10-29-joint-draft-sizing/summary.txt).

### E28: Engram row reads under a full expert cache (rejected)

BACKLOG item 11 blamed the Engram's ~98,000 random 4 KiB reads per pp2048
batch for streamed prefill slowing as the expert cache fills. On real text
a batch reads 49,152 rows per Engram layer, 39–43K of them unique (12–20%
repeats); unique 4 KiB pages are slightly more than unique rows (144 B rows
straddle pages), so sorting by offset shares nothing and deduplication
would remove at most ~15% of the reads. As an upper bound, a timing-only
build skipped every Engram read (`IQ2_XXS`, `--stream-experts`, synthetic
tokens, eight pp2048 tests per process, alternating twice): warm batches
1,365 → 1,376 tok/s (+0.8%), the first batch 1,027–1,063 → 1,064–1,102
(+3.6%). Streamed prefill now runs ~1,370 tok/s, against the 742–787 the
item was based on, and the Engram reads no longer limit it; a hot-row cache
would also take memory from the expert cache. Not pursued.

### E29: 6-column dense and grouped expert GEMVs (retained)

BACKLOG item 1 expected 4–5 ms from 5- and 6-row verifications running the
8-column kernels. The dense split GEMV and the grouped expert GEMV now
instantiate 6 columns for 5–6 rows (expert groups of 5–6 tokens). Every
row stays bit-identical: the kernel test matches 5-, 6- and 7-row launches
against one launch per row for every type, and `--spec-check` keeps widths
5 and 6 identical to single-token decode. `--spec-bench` (IQ1_M, 1,096-token
prompt, alternating builds, ms): 5 rows 89.5 / 89.4 → 89.1 / 88.9, 6 rows
100.5 / 100.6 → 99.6 / 99.9, other widths unchanged. The column jump the
item measured is gone on the current kernels (4 → 5 rows adds 12.4 ms, on
the per-row trend), so the gain is 0.5–0.8%.

### E30: expert-contiguous repack, read side (rejected)

BACKLOG item 12 proposed repacking the streamed build so each expert's
gate, up and down rows are one read instead of three about 1 GiB apart. A
read benchmark over the real `IQ2_XXS` expert spans (O_DIRECT, four
readers, per-layer batches, alternating arms; [script](artifacts/2026-10-29-expert-repack-io/repack_io.py))
compared the three projection spans with one contiguous span of the same
size per expert, and with one long run per layer:

| Batch | Three spans per expert | One span per expert | One run per layer |
| --- | ---: | ---: | ---: |
| 6 experts per layer (decode) | 5.53–5.83 GiB/s | 5.70–5.83 GiB/s | — |
| 64 experts per layer (prefill) | 4.66–4.68 GiB/s | 4.70–4.74 GiB/s | 6.35 GiB/s |

A ~2.7 MiB projection span already reads at the drive's large-IO rate, so
contiguity per expert is worth ~1%. Only long runs of neighbouring experts
gain (+36%); a rank-ordered file would give those only when a batch misses
many rank-adjacent experts, mainly a cold first prefill, while decode
misses a few random experts and warm prefill waits ~0 on reads. Not worth
a second model conversion for users.

### E31: next-layer prediction for verification batches (retained)

BACKLOG item 3: verification and joint batches (2–16 rows) predicted no
next-layer experts, so a streamed DSpark step read every missed expert on
demand. They now predict, like single tokens, when the current layer waited
on reads. Reads only; outputs are unchanged. `gufo bench` (`IQ2_XXS`,
`--stream-experts`, DSpark Q2K/Q4K, pp512 then six tg128 tests per process,
alternating, tok/s): the first test 45.21 / 43.71 → 46.77 / 46.66
(+3.5% / +6.7%), the second 50.52 / 52.26 → 49.73 / 50.82, warm tests 3–6
53.9 / 53.2 → 53.6 / 53.0 (within noise). The gain is while the cache still
misses; charging rows their own miss bytes stays open.

### E32: batched paths for 7- and 8-token prompt batches (retained)

BACKLOG item 8: small prompt batches were not monotonic in cost, because up
to 8 rows took the per-row decode paths (`gufo bench`, IQ1_M, ms): pp6 91.3,
pp7 107.7, pp8 121.8, pp9 ~98 once warm. Prompt evaluations need no
single-token arithmetic, so they now switch to the batched paths above 6
rows; decode, verification and joint evaluations keep theirs. pp7 107.7 →
~99, pp8 121.8 → ~99 ms (five repetitions, two alternations), pp2–pp6 and
pp9+ unchanged. The first batch of a new batched shape in a process costs
~20% more once, in both builds. `cache-edits` and `long-context` pass; the
DS4.1 GPU tests pass.

### E33: pinned embedding staging and threaded dequantization (retained)

BACKLOG item 7: each prefill batch dequantized its embedding rows on one
host thread and uploaded them, with the layer-0 one-hot pre-mix, through
synchronous pageable copies while the GPU waited. The rows and pre-mix now
stage in a pinned buffer beside the Engram staging and upload
asynchronously, and batches of 256 rows or more dequantize on four threads.
Embedding stage per 2,048-row batch (IQ1_M, pp8192, host timer up to a
device synchronize): ~6.0 → ~4.5 ms (dequantization 2.0 → 1.2 ms, upload
~2.5 → ~1 ms); the remaining ~2 ms are the device copies into the residual
streams. pp2048 / pp8192 throughput moves within noise (~0.1%). Greedy
output is identical, and `--joint-check 4` keeps all 164 rows bit-identical.


### E34: dense rescoring left on the bounded replay (measured, not changed)

BACKLOG item 5: prefill batches rescore the masked decoder index layers
(24, 28, 32, 36; compression ratio 1) over the full history before masking.
With bounded replay those layers run only on a prompt's last 140 rows.
`ds41_indexer_bench` with 144 rows at ratio 1 (best of 5): 0.73 / 0.08 ms
(scores / top-k) at 16K, 2.82 / 0.36 at 64K and 5.59 / 0.70 at 128K, about
25 ms per 128K prompt over the four layers (~0.03% of its prefill).
Compact rescoring would matter only under `--exact-prefill`.

### E35: 128-slot routed FP16 tiles (rejected)

BACKLOG item 6: on 80 recorded 2,048-token prefill layers (384 experts,
up to 1,116 slots per expert, 50 above 64 in the first), 128-slot tiles
would decode each expert's weights 19% fewer times than 64-slot tiles. The
kernel with 128-slot tiles (eight 16-slot accumulator tiles per projection,
twice the activation LDS, two staged quarters per thread), `ds41_mmq_bench
--f16 1` on recorded lines 0/20/40/60, best of 5, ms:

| Type | 64-slot gate+up | 128-slot gate+up | 64-slot down | 128-slot down |
| --- | --- | --- | --- | --- |
| IQ1_M | 21.51 / 20.68 / 22.16 / 20.19 | 22.04 / 20.52 / 22.43 / 20.02 | 10.92 / 10.47 / 11.23 / 10.24 | 11.46 / 10.84 / 11.78 / 10.51 |
| IQ2_XXS | 22.33 / 21.28 / 23.06 / 20.71 | 23.83 / 22.30 / 24.48 / 21.64 | 11.29 / 10.78 / 11.71 / 10.50 | 12.24 / 11.45 / 12.61 / 11.09 |

Even the most skewed layer gains under 1% on IQ1_M gate+up, so 128-slot
tiles only for the large experts have no room either.

### E36: fused SwiGLU and Q8_0 pairs through the split GEMV (retained)

BACKLOG item 2: the task-major split GEMV now also takes the shared
expert's fused gate/up + SwiGLU (Q5_K, both matrices in one pass, their
slice results side by side) and the Q8_0 `q_a`/KV pair (rows from 1,280 on
come from the second matrix; the split needs it at a multiple of 32 rows).
Below 6 (SwiGLU) and 4 (pair) tokens the lane layout stays, as fast or
faster there. Per launch at the model's shapes (µs, 1 / 2 / 4 / 6 / 8 / 12
/ 16 tokens): SwiGLU 79 / 84 / 93 / 106 / 125 / 179 / 210 →
79 / 84 / 93 / 100 / 106 / 134 / 155; pair 47 / 48 / 53 / 59 / 73 / 92 /
105 → 47 / 48 / 50 / 53 / 59 / 74 / 83. `ds41_mmq_typed_test` keeps every
token bit-identical to single-token launches (a 320-row pair case now
covers the split). `--joint-check 4` (16 rows): joint pass 177.7 / 177.4 →
174.4 / 175.5 ms, 164 rows bit-identical. `gufo bench` DSpark tg128
(about 8 verification rows) moved within noise.

### E37: grouped expert launches split by group size (retained)

A C8 AR server trace with distinct prompts (2026-10-29) put 70.7 of a
step's 114.9 ms of kernels in the grouped expert GEMV: 8 rows route to ~42
distinct experts per layer, so almost every group holds one or two slots,
yet the launch took the 8-column kernel, whose fused SwiGLU spilled (540
bytes of scratch for IQ1_S, 444 for IQ2_XXS, 256 VGPRs) and read at 172 and
148 GB/s. Launches for more than 4 columns now run in two passes over the
same plan: groups of one or two slots take the 2-column kernel, larger
groups the 16-column one (no spills) for the fused SwiGLU, or the 6/8-column
kernel for the down projection and IQ1_M's SwiGLU, which do not spill.
Every slot keeps the ungrouped kernel's arithmetic (`ds41_mmq_typed_test`
and a standalone check against the ungrouped launch are bit-identical).

Standalone, 384-expert matrices in managed memory, best of 3 (µs), distinct
routing, rows 6 / 8 / 16:

| Type | gate+up before | after | down before | after |
| --- | --- | --- | --- | --- |
| IQ1_S | 987 / 1286 / 2622 | 847 / 1095 / 2085 | 563 / 675 / 1564 | 562 / 750 / 1432 |
| IQ2_XXS | 1200 / 1826 / 2983 | 1027 / 1343 / 2560 | 571 / 804 / 1846 | 575 / 759 / 1454 |
| IQ1_M | 1073 / 1401 / 3263 | 1077 / 1401 / 2630 | 717 / 1068 / 2025 | 579 / 773 / 1492 |

With every row on the same experts (lockstep prompts) the IQ1_S and
IQ2_XXS gate+up at 8 rows take 2,209 → 598 and 2,108 → 568 µs; the second
launch costs the other shapes 2-7%.

Server, IQ1_M, distinct prompts (`serving_bench.py --corpus-layout
distinct`, AR, 128 tokens, two alternations): C8 57.04 / 56.97 → 60.71 /
60.58 tok/s (+6.4%), C4 51.92 / 51.97 → 51.88 / 52.22 (unchanged: up to 4
columns keep one launch). All 22 completions identical between builds in
both alternations. Single-user DSpark greedy (256 tokens): 38.13 / 37.92 →
38.41 / 38.08 tok/s, token trace identical.

### E38: whole-row prefetch for grouped down projections (retained)

The routed down projection's 2,304-value rows are 9 blocks; the grouped
kernel loaded 2 blocks ahead (1 for IQ1_M) and read IQ1_S at ~142 GB/s for
every group width. Its grouped launches now load a whole row at once
(`UDown = 9` for IQ1_S, IQ1_M and IQ2_XXS). The single-slot kernel kept its
depth: 9 blocks ahead slowed it from 85 to 242 µs (IQ1_S). IQ1_S's fused
SwiGLU also splits 3- and 4-column launches by group size (E37's two passes),
which loses 1-5% for the other types and projections and stays with them on
one launch. Slots keep their arithmetic (`ds41.mmq_typed` and the standalone
check against the ungrouped launch are bit-identical).

Standalone with production flags, 384-expert matrices in managed memory,
best of 3 (µs), distinct routing, rows 2 / 4 / 8:

| Type | down before | after | gate+up before | after |
| --- | --- | --- | --- | --- |
| IQ1_S | 192 / 383 / 746 | 152 / 297 / 582 | 288 / 686 / 1079 | 288 / 561 / 1080 |
| IQ1_M | 174 / 334 / 680 | 172 / 327 / 655 | - | - |
| IQ2_XXS | 186 / 365 / 713 | 186 / 354 / 702 | - | - |

With every row on the same experts the down projection improves too
(IQ1_S 4 rows 212 → 170 µs); IQ1_S's split SwiGLU costs 2% there.

Server, IQ1_M, distinct prompts (AR, 128 tokens, two alternations): C4
51.84 / 51.91 → 54.97 / 54.87 tok/s (+5.9%), C8 60.60 / 60.56 → 61.83 /
61.82 (+2.0%). All 22 completions identical between builds in both
alternations. Single-user DSpark greedy (256 tokens): 38.29 / 38.56 →
38.41 / 39.69 tok/s, token trace identical.

### E39: joint-step boundary without per-session syncs (retained)

A C8 AR server trace (distinct prompts, after E38) spent 4.75 ms between a
step's output head and the next step's first layer: ~405 us downloading 8
sessions' logits into pageable vectors, then each session's commit queued
its ~38 window-row copies and waited for them (`hipDeviceSynchronize`)
before the next session's host work could start. Commits no longer
synchronize: the copies stay queued ahead of the next evaluation, which
every later reader follows on the null stream (snapshots read with blocking
copies), and sessions' `spec_logits` are registered as pinned memory. The
same trace takes 3.33 ms at the boundary (logits download 153 us), and the
mean step 125.2 -> 124.2 ms. Server C8 60.75 / 61.96 against 61.80 / 61.78
tok/s and C4 54.82 / 55.05 against 54.87 / 54.35: inside the noise. All 22
completions and the single-user DSpark token trace identical.

### E40: one token per wave in the BF16 decode matmul (retained)

The router (384 x 5,120 BF16) and the indexer's BF16 projections took
`matmul_bf16_vec8_kernel`, whose wave walked one row for up to 8 tokens:
at 8 rows the router took 55.7 us per layer against 25.2 at one. A wave now
takes one row of one token (the grid's second dimension); every output keeps
its lane-strided sums and xor reduction, bit-identical. Standalone with
production flags (DRAM-cold weights, us), 1 / 8 tokens:

| Rows x 5,120 | before | after |
| --- | --- | --- |
| 384 (router) | 29.0 / 83.6 | 22.2 / 38.3 |
| 512 | 34.8 / 109.4 | 29.0 / 50.7 |
| 128 | 14.3 / 31.2 | 11.6 / 15.9 |
| 32 | 12.0 / 27.8 | 8.6 / 8.6 |

Server, IQ1_M, distinct prompts (AR, two alternations): C8 61.92 / 61.90 ->
62.85 / 62.72 tok/s (+1.5%), C4 55.40 / 55.17 -> 55.10 / 55.81. Single-user
(256 tokens) AR 24.07 / 24.08 -> 24.12 / 24.22, DSpark 38.68 / 39.17 ->
38.98 / 39.84. All completions and token traces identical.

### E41: split GEMV scratch for 5-8-row output heads (retained)

The split GEMV's slice results (`kSplitPartFloats`, 16 MiB) held no
129,280-row output head beyond 4 tokens at 8 slices, so 5-8-row heads
(DSpark verification, C6-C8 joint steps) took the lane kernel. The scratch
is now 32 MiB. Standalone (DRAM-cold, bit-identical to single-token
launches): 6 rows 2,553 -> 2,147 us, 8 rows 2,918 -> 2,190 (1 row 1,920).
Server, IQ1_M, distinct prompts: C8 62.69 / 62.70 -> 63.08 / 63.00 tok/s,
C4 unchanged; single-user DSpark (256 tokens) 39.25 / 39.04 -> 40.36 /
39.32, AR unchanged. All completions and token traces identical.

### E42: one drafter pass for concurrent DSpark sessions (retained)

Each drafting session in a joint step ran its own drafter forward. The
drafter now runs once over all drafting sessions' blocks (up to 8 x 5
rows): projections, router, routed and shared experts and the output head
take every row together, while attention, ring state, Markov selection
and sampling stay per session. Q4_K projections take the multi-token GEMV
in chunks of at most 16 rows, the hyperconnection mix in chunks of 16 (its
small kernel ends at 16 rows; wider launches change rounding), and each
row keeps its single-session arithmetic. A load probe compares a session
drafted first and last in a two-session batch against its solo draft:
head logits and drafts are identical. Seeded sampled traces at C1/C4/C8
and the greedy single-user trace are unchanged.

Server, IQ1_M, distinct prompts, DSpark (two alternations): C4 51.66 /
51.14 -> 54.13 / 53.13 tok/s (+4.3%), C8 54.67 / 55.84 -> 60.06 / 58.50
(+7.3%); C1 unchanged. AR remains ahead at C4/C8 (55.5 / 63.0).
Functional `batch` and `sampling-defaults` (DSpark, four sessions) pass
with outputs matching the previous build. C4 `batch` timings stay
inconclusive: the request that waits behind its peer's prefill swaps
between runs, and two unchanged-build controls flag as many timings as
the candidate (no request is flagged in every candidate run but no control).

### E43: joint rows by draft confidence across sessions (rejected)

With one drafter pass (E42), a batch's learned drafts could be drafted at
full width and the 16 joint rows handed to the drafts most likely to be
emitted (product of the chain's acceptance estimates) across sessions,
instead of each session's even share trimmed by its own width policy.
Server, IQ1_M, distinct prompts, DSpark, tok/s (rows drafted / accepted):

| Rows to drafts with reach | C4 | C8 |
| --- | --- | --- |
| even share (current) | 57.5 / 57.5 (~880 / 735) | 57.3 / 55.7 (~880 / 780) |
| every row | 50.4 / 50.9 (1,541 / 842) | 54.4 / 49.0 (1,339 / 938) |
| >= 0.25 | 57.0 / 56.7 (1,060 / 806) | 58.3 / 54.5 (1,080 / 924) |
| >= 0.5 | 57.0 / 56.9 (~885 / 745) | 58.6 / 53.1 (976 / 888) |

C8 alone, three alternations: >= 0.5 61.41 / 61.13 / 61.11 against
62.07 / 61.95 / 61.88 (-1.1%). Reallocation accepts more drafts per step,
but the wider blocks cost more than they return.

### E44: graph replay of dependent decode launches (measured, not pursued)

A chain of 1,000 dependent kernels (`tools/bench/graph_gap_bench.hip`,
ROCm 7.2, gfx1151) takes 1.96-2.42 us per launch for empty 4-256-block
kernels launched eagerly (null or nonblocking stream alike) and 1.84-2.26
us replayed as a graph; with 64-512 iterations of work per thread the saving
stays 0.12-0.15 us per launch. The host already enqueues ahead of the GPU:
the 2-3 us gaps between decode kernels are the device's dispatch latency
between dependent packets, which a graph keeps. At ~1,000 launches per
token that is ~0.13 ms of the ~2.7 ms idle (~0.3%), against an explicit
stream, device-side positions and capacity-sized indexer grids through
the whole decoder. Only fewer launches recover the gaps.

### E45: device copies batched into one launch (retained)

A C8 AR step issued ~627 small device-to-device copies (window rows into
the pending buffer and the ring, compressor pool rows), each a dependent
blit with ~6 us of gap before it (3.95 ms per step). `ds41_v41_copy_many`
takes up to 96 copies as its kernel argument, so it needs no staging
buffer: a joint layer queues every session's window and pool rows into
one launch, a commit its ~40-80 copies into one or two. Values move as
bits; seeded sampled DSpark completions at C1/C4/C8 and greedy AR
completions are identical.

Server, IQ1_M, distinct prompts, tok/s (two alternations):

| Mode | before | after |
| --- | --- | --- |
| AR C4 | 55.87 / 55.75 | 56.10 / 56.11 |
| AR C8 | 61.69 / 60.15 | 63.67 / 61.86 |
| DSpark C4 | 57.14 / 57.22 | 58.06 / 57.71 |
| DSpark C8 | 54.53 / 53.82 | 58.14 / 55.43 |
| AR C1 | 31.54 / 31.55 | 31.56 / 31.56 |
| DSpark C1 | 43.59 / 43.79 | 43.58 / 43.53 |

### E46: paired rows for single-token IQ1 down projections (retained)

The single-token routed down projection (5,120 rows of 2,304) read IQ1_S at
160 GB/s against 198 for the fused gate+up pass, which loads each x block
once into registers for its two matrices; down lanes reloaded 128 B of x
per 50 B weight block. A lane group of the down pass now takes rows r and
r + 4 through that second stream (64 rows per block) and stores both sums.
Each row keeps its FMA order (standalone outputs bit-identical). Standalone
with production flags, 384 managed-memory experts, six distinct slots (µs):

| Down projection | before | paired |
| --- | --- | --- |
| IQ1_S | 86.0 | 78.0 |
| IQ1_M | 91.9 | 86.6 |
| IQ2_S | 112.4 | 113.0 (not used) |
| IQ2_XXS | 94.7 | 102.6 (not used) |

More rows per block with one codebook staging (2-8 tiles of 32 rows) was
slower for every type but IQ1_S down (86 -> 82 µs) and is not used.
Single-user AR greedy (`gufo prompt`, 256 tokens, three alternations):
24.25 / 24.24 / 24.24 -> 24.37 / 24.36 / 24.36 tok/s (+0.5%); DSpark
39.44 / 39.64 -> 39.66 / 39.58. Generated text identical in every run.

### E47: exact Markov candidate pruning (measured, not pursued)

`markov_select` reads the drafter's 129,280 x 256 FP16 W2 (66 MB) once per
draft row in ~300 µs (~220 GB/s, 92% of the read ceiling), 1.66 ms per
single-user DSpark step; the rows are sequential (each row's x is the
previous draft token's W1 row). A Cauchy-Schwarz bound (logit + |W2[t]|
|x|, against the exact score of the 64 best logits) would let a kernel
skip rows that cannot win. On 379 captured greedy calls it still had to
score 99.2% of tokens (median): |W2 . x| reaches 24-40 against a logit
spread of ~15-20, and the winner was outside the top-64 logits in some
calls. The selection stays at its read ceiling.

### E48: grouped routed experts in the drafter (retained)

The drafter's routed experts ran one slot per launch row (5 rows x 3
experts) through the single-token GEMV, reading a shared expert once per
slot. Blocks of 2-16 rows now plan their slots by expert and take the
grouped GEMV (bit-identical per slot); wider batched blocks keep the
per-slot launch. Single-user DSpark greedy trace: drafter expert kernels
2.06 -> 1.70 ms per step (plus 0.11 ms of plan launches). `gufo prompt`,
256 tokens, three alternations: greedy 39.82 / 39.62 / 39.58 -> 41.60 /
39.93 / 39.76 tok/s (the first run's timing-based widths drafted 211
instead of 220 rows), seeded sampled 40.26 / 40.19 -> 40.24 / 40.32 with
identical text, drafts and acceptance.

### E49: collapse-norm folded into the hyper-connection mix (rejected)

With shifted mHC the collapse before attention and the FFN reads the
previous pre-mix, so it does not wait for the mix that precedes it: an
extra 256-thread block of `hc_mix_small_kernel` can run it, reproducing the
1,024-thread kernel's reduction (thread x holds threads x + 256 q, which
share its lane; bit-identical at 1-16 tokens). Standalone with warm caches
the pair took 12.9 -> 8.4 us at one token. In the model it lost: the
collapse block (128 loads per thread) outlasted the mix with cold weights,
the fused kernel took 22.1 us against 13.7 + 4.3, and VGPRs rose 48 -> 72.
Single-user AR 24.35 -> 24.22 tok/s (collapse block first; 24.12 with it
last).

### E50: verification argmax on the device (retained)

A single-user DSpark step left the GPU idle ~680 µs between the target's
output head and the commit: the host scanned each verified 129,280-logit
row with `ds41_sample_argmax` (67 µs per scan, scalar), twice for accepted
rows (positional statistics and the greedy draw), plus the bonus row,
8-9 scans per step. The speculative and joint evaluations now launch
`ds41_v41_argmax_rows` (the drafter's argmax kernel, one block per row;
same first-largest-finite rule, -1 without a finite value) before their
final sync and download the ids with the logits; verification and greedy
draws use them. Sampled requests keep their device draws.

`gufo prompt`, 256 tokens: greedy DSpark 39.77 / 39.74 -> 40.00 / 40.04
tok/s for runs with the same drafted rows (220), text identical; seeded
sampled identical text, drafts and acceptance (40.31 -> 40.32). Server,
distinct prompts: seeded sampled completions at C1/C4/C8 and greedy
DSpark completions at C1/C4/C8 identical to the previous build.

## E51 — IQ1 single-slot experts leave the grouped kernel (retained)

The expert plan also lists, per slot, the expert id when no other slot
shares it; for IQ1_S and IQ1_M those slots (and dropped ones) take the
ungrouped kernel, which loads its id directly and pairs down rows, and the
grouped launches start at two slots. A one-column grouped kernel instead
was no faster than the two-column one (rejected); Q2_K, Q4_K and IQ2_XXS
stay grouped, where the ungrouped kernel was equal or slower. Synthetic
24 slots of 4 routed-apart rows: IQ1_M gate/up 717 -> 609 us, IQ1_S 562 ->
523 us, down 2-4% faster; fully shared rows pay one launch (+2-4%), all
bit-exact. `gufo prompt`, 256 tokens, C1: greedy DSpark +0.15-0.5% (text
identical), seeded sampled 40.16 / 40.00 -> 40.25 / 40.33 tok/s with
identical text, drafts and acceptance; AR unchanged. Server, distinct
prompts, greedy DSpark: C4 47.01 / 46.86 -> 47.14 / 47.39, C8 60.71 /
60.33 -> 61.79 / 60.75 tok/s, completions identical.

## E52 — dense FP16 prefill: Q6_K occupancy, Q8_0, paired launches (retained)

Unsloth UD-IQ1_M keeps its attention projections in Q6_K and its shared
experts in Q8_0. Q6_K took 1.3-1.45x the Q5_K time for 1.19x the bytes,
with the same instruction mix and DRAM traffic: at 200 VGPRs it held one
16-wave workgroup per WGP instead of two (the limit is 192). A shorter
activation prefetch ring for the wide formats (2 stages for FP16 inputs,
1 for FP32) brings it to 184 VGPRs. Q8_0 joins the kernel as eight 34-byte
blocks per 256-value step (sign-flipped bytes through the FP16 magic
constant, exact), so the UD shared experts and the ISTA `attn_q_a`/`attn_kv`
leave MMQ. One launch now takes two matrices on the same rows: `attn_q_a` +
`attn_kv` (with `xn` rounded to FP16 once into the idle `o` scratch) and
shared gate + up. Rejected: padding Q6_K to 32-byte-aligned blocks (0-7%),
an 8-wave occupancy cap (16 spilled VGPRs, no faster), full activation
fragments from LDS instead of `permlanex16` (LDS-bound, 15-30% slower),
token tiles as the fastest grid index (activation re-reads, 20-80% slower),
and router BF16 staging of 64 values per barrier with register prefetch
(bit-identical, 3-24% slower at 2,048 tokens).

`ds41_mmq_bench --dense --x-half 1`, Q6_K: `attn_q_b` 5.65 -> 5.32 ms,
`attn_output_a` (8 groups) 5.82 -> 4.74, `attn_output_b` 5.06 -> 4.50; Q8_0
shared gate 2.18 (MMQ) -> 1.34 ms. Last warm pp2048 prefill of a profiled
`gufo bench` (streamed): UD-IQ1_M 1,804 -> 1,603 ms (1,135 -> 1,278 tok/s),
GSQ IQ2_XXS 1,722 -> 1,675 ms. The Q6_K ring and the paired launches are
bit-identical: the dense kernel matches the previous one on every shape
with FP32 and FP16 inputs, and the UD-IQ1_M 3K-token batch score with Q8_0
still on MMQ matches the previous build on all 2,999 positions. Only Q8_0
changes the arithmetic (FP16 activations instead of Q8_1): UD-IQ1_M PPL
5.772 -> 5.779 (mean dNLL +0.0012 +- 0.0035, argmax agreement 94.1%),
GSQ IQ2_XXS 6.022 -> 6.041 (+0.0031 +- 0.0028, 95.3%), both within noise,
for about 28 ms (1.7%) of each UD-IQ1_M pp2048.

Remaining prefill (UD-IQ1_M, 1,549 ms of kernels): routed FP16 tiles 42%
(about 6 ms per gate/up call is weight decode, independent of the tokens;
the per-token part runs near 39 TFLOPS), dense FP16 30% (32-40 TFLOPS),
attention 9.6%, `hc_state_pass` 7.7% (at the DRAM ceiling), router 4.9%
(exact three-term BF16 split, about 25 effective TFLOPS).

## E53 — prefill fragment loads and attention fixed costs (rejected)

Dense FP16 kernel, Q6_K/Q8_0 at 2,048 tokens (`ds41_mmq_bench --dense
--x-half 1`; `q_b` 4.19 ms). Each 16-token product waits on its own LDS
fragment load behind the per-tile guard. All three variants were
bit-identical to the committed kernel on every dense shape, with FP32 and
FP16 inputs:

- Next tile's fragment loaded during the current product (full tiles,
  scheduling barrier between tiles): the compiler hoists every load anyway,
  256 VGPRs with 815 spills, `q_b` 30 ms.
- Both fragment halves read from LDS instead of four `permlanex16` per
  product (already rejected in E21 on Q5_K): 185 VGPRs, `q_b` 5.02 ms, every
  shape 20–37% slower.
- Fragments loaded in pairs under the runtime guard: 196–202 VGPRs, one
  workgroup per WGP, `q_b` 5.85 ms, `wo_a` 6.73.

The second resident workgroup already hides the waits.

Prefill attention (`ds41_attention_test --bench`, 2,048 tokens, FP16
output as in production; ablations are timing-only):

| Case | ratio 128, depth 0 | ratio 4, depth 0 | ratio 4, depth 16K |
| --- | ---: | ---: | ---: |
| Kernel | 2.59 ms | 5.31 | 8.26 |
| No Q loads | 2.06 | 4.95 | 8.08 |

With FP32 output, against 3.14 / 5.48 / 8.31 ms for the full kernel:

| Ablation | ratio 128, depth 0 | ratio 4, depth 0 | ratio 4, depth 16K |
| --- | ---: | ---: | ---: |
| No tile loop | 2.90 | 2.94 | — |
| No score products | 3.12 | 4.09 | 5.46 |
| No PV products | 3.02 | 4.42 | 6.25 |
| No window-row loads | 2.96 | 5.28 | 8.12 |
| Without rope/FP8 | 2.91 | 5.21 | — |

Reading FP32 Q costs 0.2–0.5 ms per call. FP16 Q written by `q_b` (rope
and rounding moved into its epilogue) would save about half of that, about
1% of prefill. Not attempted.

## E54 — decode: global expert loads, FP8 output in the split combine (retained)

Profile of one warm UD-IQ1_M decode layer (streamed experts): 23 launches,
990 us busy, 78 us of gaps. The dense GEMVs read at 87-95% of the
240 GB/s ceiling (`ds41_gemv_bench`); the table-indexed expert GEMVs that
streaming uses ran at 77-81%.

**Retained, global expert loads.** The streamed store passes each expert's
rows through a pointer table. A pointer loaded from memory is generic, so
the expert GEMV's weight loads compiled to `flat_load`, which count against
the same wait counter as the LDS codebook reads: every codebook wait also
waited for the weight loads in flight. The table entries are now read as
global pointers. Neither an address-space cast of the loaded pointer nor
`__builtin_assume(!is_shared && !is_private)` changed the code; a
global-typed table element did. `ds41_gemv_bench`, table path, x6 slots:

| Format | before | after |
| --- | ---: | ---: |
| IQ1_S up / down | 90.3 / 91.0 us | 75.6-76.1 / 77.5-78.2 |
| IQ2_XXS up / down | 103.5 / 104.5 | 97.5 / 99.3 |
| IQ2_XS up / down | 110.5 / 110.7 | 105.8 / 104.3 |

The table path now matches the plain path bit for bit and in time.
`gufo bench` tg128, streamed UD-IQ1_M, alternating: 20.15 / 20.32 ->
20.59 / 20.61 tok/s.

**Retained, FP8 output in the split combine.** The decode split
attention's combine applies `rope_fp8`'s inverse rotation and FP8 rounding
(4 dimensions per thread, block maximum over 8 lanes, the rotation's FMAs
written as `rope_fp8_kernel` contracts them), dropping one launch per
layer. `ds41.attention` checks the result bit for bit against the separate
`rope_tail`, unrotated attention and `rope_fp8` launches at 1, 3 and 16
tokens. One-token attention call (rope, split path): 0.042 -> 0.038 ms
(ratio 4), 0.023 -> 0.020 (ratio 128), 0.023 -> 0.019 (window only).

Rejected:

- Rotating q in the split kernel's registers instead of `rope_tail`: the
  rotation compiled identically, but the compiler then reassociated the
  score dot products, changing about half of the FP32 outputs by an ulp.
- The expert GEMV's first blocks loaded before its codebook copy and
  barrier: every format slower (IQ1_S 76 -> 81 us up, 79 -> 88 down).
- Global instead of flat weight loads in the prefill routed FP16 tiles
  (the aligned-word pointer was formed through an integer mask): 20.53 ->
  20.42 ms IQ1_S gate+up, 21.39 -> 21.27 IQ2_XXS, within noise; those tiles
  are bound by the matrix cores.

A synthetic dependent chain (`hipcc`, gfx1151) gives a tiny kernel 1.9 us
after a 2,048-block working kernel and 5-11 us after 64-1,024-block ones,
so the profile's ~7 us gaps after the large-grid GEMVs are not explained
by grid size.

## E55 — decode: HC mix kernel (rejected)

`hc_mix_small` (24 rows of `fn`, 2 MB FP32, plus the Sinkhorn tail) runs
twice per layer. A standalone loop over 64 rotated `fn` buffers (beyond
the MALL) times one-token calls at 13.98 us; a 2 MB streaming read in the
same loop takes 10.3 us at any grid from 80 to 960 blocks, and an empty
launch 1.9 us. Without the last-block reduction and Sinkhorn the kernel
takes 11.8 us; with one Sinkhorn iteration instead of 20 it takes 13.27 us.
At most ~3 us per call (~0.25 ms per token) is above the read.

Rejected:

- 2, 3, 4 or 6 mix rows per block, loading `h` once: 14.2, 14.6, 15.0 and
  15.6 us at one token (faster only at four tokens), and the compiler
  contracted the row dot products differently.

A four-lane Sinkhorn could save part of the 2.2 us tail; at about 0.1% of
a token it was not attempted.

## E56 — decode: shared expert during the streamed route read (retained)

With streamed experts, each layer's prepare copied the route ids with a
blocking `hipMemcpy`, planned the layer and only then launched the shared
expert from the read overlap, leaving the GPU idle about 11 us per layer.
Decode batches (one token or the grouped expert path) now record an event
after the router, queue the shared expert, and let the store copy the ids
on its own non-blocking stream after that event. The event still follows
every earlier launch, so the stage, slot and table uploads keep their
ordering. Logits and per-layer states of a 40-token streamed UD-IQ1_M
greedy decode are unchanged bit for bit.

`gufo bench` streamed UD-IQ1_M tg128, alternating with the previous
commit: 20.11 / 20.00 -> 20.48 / 20.44, 20.11 / 20.08 -> 20.48 / 20.51,
19.70 / 20.11 -> 20.37 / 20.50 tok/s (about +2%); host wait 42.8 -> 41.3
ms per token.

Prefill keeps the shared expert in the read overlap. The pp2048 means
(cold expert reads, about +-400 t/s per run) cannot resolve a change of
this size: with the launch moved for every batch they read 1.7-3.8%
lower, but the decode-only build, whose prefill path matches the previous
commit, also read 0.7-2.4% lower.

## E57 — decode: HC mixes on a side stream (retained)

A layer's mixes (pre, post and comb) feed only the expand closing their
half, and its collapse uses the previous half's pre-mix, so decode batches
of up to 16 tokens now run `hc_mix` on a non-blocking stream forked after
the state is written and joined before that expand. The kernels and their
inputs are unchanged; logits and per-layer states of a 40-token streamed
UD-IQ1_M greedy decode match bit for bit.

The mix still reads its 2 MB while the projections it overlaps are
bandwidth-bound, so only its latency is hidden. The profile shows the
single-block collapse slowed from 4.2 to 10 us beside the mix, and the
attention half's critical path shortened from about 28 to 19 us.
`gufo bench` streamed UD-IQ1_M tg128, alternating with the previous
commit: 20.19 -> 20.61, 20.48 -> 20.60 (pp2048 runs), 20.14 -> 20.22,
20.06 -> 20.18 tok/s (pp512 runs); about +0.6% at the median.

Rejected: launching each collapse before forking its mix. The collapse
kept its 4.2 us, but the fork's host calls then delayed the first
projection; tg128 20.49 / 20.49 -> 20.50 / 20.56, and 20.12 / 19.86 in the
three-way pp512 runs, no gain.

## E58 — multi-token Q6_K and Q8_0 dense GEMVs on the task-major kernel (retained)

Joint decode and verification batches run the dense projections through
`ds41_v41_gemv` with one column per token. Q5_K switched to the
task-major `split_gemv_kernel` from 2 tokens (scalar activation loads, one
wave per task); Unsloth's Q6_K projections and output head never did, and
Q8_0 only for gate/up pairs, so at 8 tokens their per-wave activation loads
bounded them: Q6_K q_b 223 -> 153 GB/s from 1 to 8 tokens, against 218 ->
205 for Q5_K (standalone, weights rotated beyond the caches).

Q6_K gained the split kernel's load and decode (`LoadQ6K32` and the
unchanged `DecodeQ6K32` arithmetic), and plain or paired Q6_K/Q8_0 rows now
take the task-major kernel from 4 tokens, their fused SwiGLU from 6. The
slice scratch grew from 32 to 64 MiB so the 129,280-row head fits at 16
tokens. Every output matches the previous kernels bit for bit (`ds41.mmq_typed`
batched against single-token launches; a dump of every type, shape,
SwiGLU and grouped case at 1-16 tokens compared with `cmp`).

| Kernel time | 4 tokens | 8 tokens | 16 tokens |
| --- | ---: | ---: | ---: |
| Q6_K q_b, wo_b, shared expert | -3 to -10% | -17 to -25% | -23 to -41% |
| Q8_0 shared expert gate+up / down | -0 / -14% | -15 / -17% | -34 / -29% |
| Q6_K output head (ms) | 2.72 -> 2.54 | 3.12 -> 2.64 | 5.43 -> 2.90 |
| Q5_K output head (ms, ISTA builds) | unchanged | unchanged | 4.57 -> 2.61 |

At 2 tokens the lane layout stays as fast or faster (+0-8% for the split
kernel), as does the SwiGLU at 4. Streamed UD-IQ1_M over HTTP (distinct
prompts, 4,096 context, alternating): C8 63.81 / 63.63 -> 66.83 / 66.52
tok/s (+4.6%), C4 48.23 / 47.93 -> 47.80 / 48.69 (no change).

## E59 — decode: router loads in flight, routing without argmax rounds; GEMV probes (retained; GEMV probes measured)

Resident `IQ1_M`, ROCm 10.1, single-token decode (rocprofv3 kernel times,
medians over 33 tokens).

**Retained.** The BF16 matvec gave each wave one 10 KiB router row with one
load in flight per iteration; four loads in flight (`#pragma unroll 4`,
same sums in the same order) take the 384-row router 21.9 -> 19.2 us, the
4-block matvec 8.3 -> 3.6 and the 64-block one 28.2 -> 24.2. `route_kernel`
ran k block-wide argmax rounds with two barriers each; each wave now takes
its own k best with shuffles and wave 0 merges the candidates by the same
order (larger choice, then lower index): 7.2 -> 5.6 us, ids and weights
identical on 2,000 tokens with ties, unchanged at 2,048 rows (419 against
427 us). About 0.24 ms per token; greedy, DSpark and seeded DSpark text
identical on `IQ1_M`, streamed `IQ2_XXS` and streamed UD-IQ1_M. Ranks
computed per expert (one pass, 384 compares per thread) were no faster
(6.2 us).

**Measured, not pursued.**

- Expert GEMVs (`ds41_gemv_bench`, fused SwiGLU added): U = 3 or 4 blocks
  ahead are slower for IQ1_S and IQ1_M (IQ1_S gate+up 202 -> 172 / 167
  GB/s); U = 2 instead of 1 gives IQ1_M gate+up +2%. Two to six row tiles
  per block (amortizing the LDS codebook) were never faster.
- Dense Q5_K at one token: the multi-token path's one-block-ahead register
  prefetch moved shapes by -3 to +3% with no consistent gain; nontemporal
  weight loads (`slc dlc`) were 3-20% slower.
- `wo_b` (Q5_K 5,120 x 8,192, 28.8 MB) takes 140 us in the model against
  125 us standalone (230 GB/s), every layer. A cold `wo_b` launched after
  another `wo_b` takes 128 us; after `wo_a` 140, whatever it reads. Not the
  activations' cache state (a disjoint copy keeps 128), placement (a fresh
  `hipMalloc` copy keeps 140), slices (S4 141), or a memset in between.
  Dispatches use agent-scope fences (CLR 10.1 default). ~0.5 ms per token,
  cause open. A second read of the same weights right after takes 45 us
  (MALL).
- The standalone benches run the shader clock at ~0.95 GHz and ~38 W; the
  model's decode at 2.3-2.9 GHz and ~68 W.

## E60 — launch gaps after long kernels are a tracing artifact (measured)

Streamed decode traces (UD-IQ1_M, `IQ2_XXS`) show ~6-8 us before the
launch after each 2,048-block dense GEMV, against ~2.5 us elsewhere
(backlog item 4). A standalone chain of a streaming read (2,048 blocks,
cold 1 GiB pool) and a one-block kernel reproduces it under rocprofv3:
the gap is 1.9 us after reads of up to 14 MB (60 us) and 6.4 us from
16 MB (69 us) on; warm reads of the same sizes, which finish in under
26 us, keep 1.9 us. Fixed-length spin kernels give the same steps by
duration alone: 1.9 us after 54 us, 6.4 us after 100-500 us, 10.7 us
after 750 us or more.

Without the profiler (hipEvent over 100 iterations, medians of 21
trials) the one-block kernel adds 2.0-2.1 us after cold reads of 8, 12,
14, 16, 20 and 28 MB alike, so the step is added by kernel tracing. Spin
kernels timed with events vary by 5-39 us per iteration and are not
usable for this.

Streamed decode work: the miss-read idle (~5.8 ms per `IQ2_XXS` token at
98.5% route hits, 32.7 MiB per token) is the remaining streamed-specific
cost; the cache holds 12,696 of 15,360 experts within unified memory.
Predicting the next layer on every layer instead of only after a stall
read 30% more bytes (14.7 -> 19.2 MiB per token over pp512 + tg128) with
no decode gain in alternating runs (19.33 / 19.56 / 17.27 against 18.71 /
17.20 / 17.47 tok/s). Precomputed IQ1 group sums of x staged in LDS were
slower (IQ1_M up 100 -> 102 us, IQ1_S up 76 -> 89 us in `ds41_gemv_bench`).


## E61 — expert reads split across two drives (retained)

Each miss reads an expert's three projections (~7.7 MB for `IQ2_XXS`)
from one drive, and a single direct read stream on the Samsung 990 PRO does
not get past ~5.5 GB/s (1.32-1.47 ms per cold miss; ~6.5 GB/s for six
misses at once). More readers (3-32) or smaller chunks did not raise it.
The second NVMe, a KIOXIA EXCERIA BASIC on the root btrfs, reads
5.15 GB/s with direct I/O once compression is off for the file (NOCOW);
the 2.3 GB/s in the earlier [storage reads](#storage-reads) note was
measured through btrfs compression.

`tools/ds41/pack_expert_tails.py` copies the second half of every routed
expert projection into one file on the second drive (61.6 GiB for
`IQ2_XXS`, 61.7 GiB for UD-`IQ1_M`, instead of a full copy). With
`MODEL.gguf.expert-tails` present, each projection read becomes a head
read from the GGUF and a tail read from that file, issued together, so
both drives read every miss. Read workers went from 4 to 8 so each drive
keeps its queue depth. The bytes are unchanged: greedy outputs of `IQ2_XXS`
and UD-`IQ1_M` were identical with and without the file in AR, DSpark and
DSpark-sampled modes.

Cold reads of real expert spans (2026-11-01):

| Case | One drive | Split, 50-55% from the GGUF |
| --- | ---: | ---: |
| 1 miss | 1.32-1.47 ms (5.3-5.9 GB/s) | 0.87 ms (8.9 GB/s) |
| 2 misses | — | 1.61 ms |
| 6 misses | ~7.1 ms (~6.5 GB/s) | 4.35 ms (10.7 GB/s) |

Head shares of 0.50-0.60 and one merged tail read per expert, instead of
three, measured the same.

`gufo bench` on streamed `IQ2_XXS` (pp512 + tg128, random prompt, no
drafter, alternating runs): tg 17.06-17.68 → 17.87-19.16 tok/s (median
+7.6%), pp512 510-538 → 613-621 tok/s (+16%), cache seed 19.7 → 14.9 s.

**Served workloads.** A fresh `gufo serve` per run (DSpark, temperature 0,
alternating without and with the file, two rounds each) played an 8-turn
chat that switches topic and language, an 18.4K-token document with two
follow-up questions, and a Pi 0.87.0 coding-agent session (creation,
bugfix and edit tasks, all passing). Chat and document outputs were
identical across runs; the agent's turn count varies by one. Prefill and
decode are the server's per-request rates, summed over each workload.
Rounds 1 / 2:

| Model, expert cache | Workload | Prefill tok/s, without → with | Decode tok/s, without → with |
| --- | --- | --- | --- |
| `IQ2_XXS`, ~100 GiB (81% of experts) | chat | 112.8 / 111.3 → 121.1 / 113.2 | 30.46 / 30.63 → 31.20 / 30.84 |
| | document | 879.1 / 878.2 → 925.8 / 912.7 | 32.48 / 32.19 → 32.68 / 32.36 |
| | agent | 322.5 / 311.8 → 355.0 / 352.4 | 41.94 / 40.72 → 42.83 / 43.30 |
| UD-`IQ1_M`, ~100 GiB | chat | 192.3 / 182.4 → 211.9 / 212.6 | 28.34 / 28.64 → 29.24 / 29.22 |
| | document | 827.0 / 827.7 → 878.7 / 905.8 | 30.84 / 30.92 → 31.44 / 31.18 |
| | agent | 312.8 / 296.9 → 339.8 / 338.6 | 40.40 / 42.43 → 44.00 / 43.42 |
| `IQ2_XXS`, 88 GiB cap (71%) | chat | 88.6 / 88.0 → 109.2 / 109.1 | 28.62 / 28.57 → 29.96 / 29.69 |
| | document | 706.8 / 710.3 → 841.1 / 838.6 | 30.49 / 30.62 → 31.39 / 31.30 |
| | agent | 233.6 / 240.7 → 296.0 / 296.4 | 38.91 / 37.14 → 40.19 / 40.16 |
| `IQ2_XXS`, 64 GiB cap (52%) | chat | 44.4 / 44.5 → 65.6 / 65.6 | 20.74 / 20.70 → 25.41 / 25.08 |
| | document | 402.5 / 413.4 → 663.5 / 664.2 | 23.30 / 23.27 → 26.61 / 26.55 |
| | agent | 125.3 / 125.4 → 189.5 / 190.2 | 27.78 / 28.35 → 31.21 / 33.75 |

With the full cache, `IQ2_XXS` hit 98.0% of routes and read 6.4 MiB per
evaluated token over the session (chat 13.8, agent 10, document 4.2 MiB),
under one expert per token, so decode gains 1-3% while prefill of short
requests gains 5-17% and the agent's summed time to first token falls from
7.25 / 8.52 s to 6.39 / 7.33 s. The caps stand in for larger
quantizations at the full cache. At 88 GiB, about the share of experts a
~150 GB backbone such as `IQ2_XS` would keep cached, routes hit 96.3% and
the session read 12.5-12.7 MiB per token (chat 28, agent 21 MiB): the
split gains 18-27% in prefill and 2-5% in chat and document decode. At
64 GiB, 91.5% route hits and 35 MiB per token (chat 90, agent 67 MiB), it
gains 47-65% in prefill and 12-23% in decode; the Pi tasks took 59-67 s
instead of 76-87 s.

The per-token host wait now separates the wait for the layer's routes from
the wait on demand reads after the overlapped work. At 88 GiB the split
cut the session's read wait from 1.40 / 1.42 to 0.95 / 0.95 ms per token,
with the route wait unchanged at ~5.1 ms. At 64 GiB it
cut the session's read wait from 3.88 to 2.45 ms per token and the route
wait from 5.73 to 5.26 ms; in chat, read wait fell 9.8 → 6.8 ms and route
wait 19.8 → 17.4 ms. The route wait follows the reads too (it includes the
previous layers' queued miss work), so read wait alone understates what
reads cost a token.

A served session on 2026-10-31 (UD-`IQ1_M`, 275 coding-agent requests,
prompts up to 73K tokens, no tails) ran at a median 99.7% route hits and
7.2 MiB per token (p90 13.1).

## E62 — DSpark drafter size and quantization (measured, current drafter kept)

Streamed `IQ2_XXS` frees memory for the drafter, so bigger and smaller
drafters were compared against the Q2_K/Q4_K drafter (4.59 GiB resident).
Ten prompts, greedy, 512 tokens; every cycle drafted five tokens and was
scored against the plain greedy sequence, so acceptance is not censored by
the width rule. Replayed rate: the learned-head width rule over those
cycles with fixed measured costs.

| Drafter | Resident | Accepted of 5 | Drafting | Replayed tok/s |
| --- | ---: | ---: | ---: | ---: |
| Q2_K experts, Q4_K dense (current) | 4.59 GiB | 2.26 | 8.5 ms | 37.18 |
| Same, fine-tuned dense/head tensors | 4.59 GiB | 2.22 | 8.5 ms | 37.00 |
| `IQ2_XXS` experts (imatrix), Q4_K dense | 3.70 GiB | 2.22 | 8.5 ms | 36.89 |
| `IQ1_S` experts (imatrix), Q4_K dense | 2.91 GiB | 2.11 | 8.4 ms | 36.21 |
| Q8_0 dense, MXFP4 experts | 7.42 GiB | 2.30 | 10.3 ms | 36.88 |
| Same, fine-tuned dense/head tensors | 7.42 GiB | 2.29 | 10.3 ms | 36.64 |

Drafting time is bound by fixed costs (output head, Markov selection,
dense layers), not by expert bytes, so smaller experts do not draft
faster. Replaying the expert-cache traces with the memory each drafter
frees or takes: `IQ2_XXS` experts gain 0.89 GiB of cache (decode stall
-0.2% at the full cache, -0.9% at 64 GiB), `IQ1_S` 1.70 GiB (-0.4% to
-1.7%), Q8_0 loses 2.83 GiB (+1.1% to +3.2%). Net: `IQ2_XXS` experts are
even, `IQ1_S` 1-2% slower, Q8_0 2-4% slower. The drafter's expert use is
skewed (layer 2 uses 62 of 128 experts; the top 32 take 98% of routes), but
experts unseen in calibration quantize to 52-60% error at `IQ2_XXS`.

The Unsloth drafter names its confidence head `confidence_head.proj.weight`;
it is now loaded (it was ignored, so the drafter ran without learned
widths).

## E63 — DSpark drafter: no host sync after injection, F32 router per token (retained)

The feature injection ended with `hipDeviceSynchronize` (312 µs of host
block per cycle), and the drafter's block rows, draws and tokens went up
with blocking pageable copies. The drafter and decoder share a stream, so
the injection now only checks the launch, and the drafter's inputs are
staged in pinned memory and uploaded with `hipMemcpyAsync` behind the
queued work. The drafter's 128-expert F32 router ran on `matmul_kernel`
(16 blocks for 6 rows, 131 µs per layer); up to 16 tokens now take a wave
per row of one token (`matmul_f32_row_token_kernel`, the same per-lane
order, bit-identical), 36 µs.

Drafting 8.5 → 8.3 ms per cycle in per-cycle traces. Drafts matched the
previous build at 1,518 of 1,518 positions; greedy, seeded sampled and AR
outputs are identical to the previous build.

**HC mixes of several tokens in one launch (rejected).** A mix kernel that
reads each weight once for all verification rows took 14.1 instead of
17.1 µs at 6 tokens and 25.3 instead of 36.5 at 16, but per-cycle
verification at 2-6 rows did not move beyond noise (±1 ms). Matching
single-token decode bit for bit needed explicit FMA chains, which changed
single-token rounding against earlier builds (~1e-7, a near-tie flips
greedy text after ~30 tokens), so it was dropped.

## E64 — DSpark cycles cut to zero drafts (measured, not pursued)

When the width rule keeps no drafts, the cycle has paid for drafting and
verifies only its anchor. Five prompts, 256 tokens: 67 of 493 greedy
cycles and 185 of 563 sampled cycles (`top < kConfidentDraft`), 9% and 23%
of the time. The one-row verification costs the same as a plain step
(42.1-42.3 ms decoder, 0.28 ms feature injection), so only the 8.3 ms of
drafting is lost: skipping every cut with perfect foresight would gain
1.5% greedy and 3.9% sampled. Gates known before drafting do not find
them: the previous cycle's cut (P(cut) 0.42-0.50 after a cut, 0.24-0.41
otherwise) or the target's probability at the anchor (median 0.48-0.51 for
cut cycles, 0.66 for drafted ones). Skipping when the target's top
probability is below 0.3/0.5/0.7 gives -0.4/-2.9/-7.9% greedy and
+0.1/+0.1/-3.0% sampled. Stopping the drafter after the first position
decides a cut would save only the remaining Markov selections (~1.2 ms).

**Draft vocabulary estimate.** DS4.1 numbers frequent tokens first less
strongly than Qwen: of 10,220 generated tokens, ids below 32,768 / 65,536 /
98,304 cover 90.4 / 95.8 / 98.9% (95.7 / 98.6 / 99.5% with the specials
and the request's earlier tokens). Scaling the head and Markov selection
(3.66 ms of drafting) to the subset and truncating acceptance at each
uncovered token, the greedy replay gains -0.7 / +1.1 / +0.7%.

## E65 — attention partials written back during the dense GEMVs (measured)

Per-width verification profiles put the dense GEMVs ~3.8 ms per 4-row
cycle above their read time; most of it is one GEMV. Timed with events
inside the model, `attn_output_b` (Q5_K, 5,120 x 8,192, 28.8 MB) took
149 / 176 / 214 µs at 1 / 3 / 6 rows (193 / 164 / 135 GB/s), while
`gemv_bench` ran the same call at 134-146 µs (198-216 GB/s), also with the
real weights, misaligned bases, and the `attn_output_a` launch before it.
Fresh allocations for the weights, the input or the output, and the
resident model without SSD reads, measured the same in the model. A second
`attn_output_b` launch after the first ran at full speed, and an extra
launch placed before `attn_output_a` moved the delay to it and to
`attn_output_a`: a fixed cost of ~16 µs at 1 row and ~76 µs at 6 rows lands
on whatever follows attention.

Split decode attention writes 20 splits x 64 heads x 516 floats, 2.64 MB
per row and layer, which the combine kernel reads once. Writing that many
bytes with a memset before the pair in `gemv_bench` reproduces the delay
(1 row, 2.64 MB: `attn_output_b` 134 → 145 µs; 6 rows, 15.8 MB: 128 / 146 →
158 / 182 µs). With 10 splits (half the bytes), `attn_output_b` in the
model took 143 / 160 / 174 µs at 1 / 3 / 5 rows; AR on resident `IQ1_M` went
24.95 / 24.99 / 25.05 → 25.08 / 25.12 / 25.16 tok/s (alternating, different
text from the changed rounding), and per-width verification 0.4-1.7 ms
faster. Non-temporal Q5_K weight loads removed the delay in the bench at
one row but not in the model (AR 24.94 → 25.03 tok/s, `attn_output_b`
unchanged), and made the multi-row split kernel twice as slow, since its
eight waves re-read each block's header. Not retained; backlog item 33.

## E66 — multi-token BF16 matmuls for verification (measured, not retained)

Verification runs the BF16 indexer `q_b` (4,096 x 1,280), compressor
(512 x 5,120) and router (384 x 5,120) with one wave per row and token, so
each weight row is read from DRAM once per token: the indexer took 46.7 /
57 / 82 / 102 / 122 µs at 1 / 2 / 4 / 6 / 8 tokens
(`tools/bench/ds41_bf16_rows.hip`, production flags `-O3 -ffast-math
-fno-finite-math-only`). A wave taking two rows and up to eight tokens
read each weight once: indexer 50-53 µs at 2-6 tokens, compressor 33.5 /
34.9 µs at 4 / 6 (from 33.3 / 40.0); the router stays faster with a wave
per token. About 0.2-0.35 ms per verification cycle.

Under `-ffast-math` the compiler reassociates the two kernels' sums
differently (the per-token kernel adds four unrolled iterations' tree sums
into the accumulator), so results differ at rounding level (up to ~2e-3
relative) from the one-token launch used by AR. Explicit FMA chains in
either order, and an isolated 8-term sum behind a register barrier, did
not reproduce the reference bits. Keeping the reference loop unchanged
and looping tokens outside it is bit-identical, with the row re-read from
cache, but only gains 7-10% on the indexer and is slower on the compressor
and router. Not retained: the faster kernel needs a rounding change in
the verification path.

## E67 — IQ3_XXS and Q2_K routed downs, IQ2_S fused gate/up (retained)

Unsloth `UD-IQ2_XXS` stores routed downs as IQ3_XXS (layers 20-34) and
Q2_K (layer 0), and gate/up as IQ2_S in layers 25-39. The streamed path had
no kernel for the two down types and ran IQ2_S gate/up unfused. Added: an
IQ3_XXS format in the per-expert GEMV and the grouped 2-16-token GEMV
(grid staged as 32-bit words through a shared `StageIq2` helper), IQ3_XXS
and Q2_K formats in the FP16 routed tiles (down only), IQ2_S in the fused
SwiGLU gate/up pair, and the 98 / 84-byte block sizes in the expert store.
`ds41.mmq_typed` checks each against the ggml matvec.

`ds41_gemv_bench`, six expert slots, 5,120 × 2,304 down:

| Kernel | Time | Read rate |
|---|---:|---:|
| IQ3_XXS down, 2 blocks loaded before decoding (retained) | 126.8 µs | 214 GB/s |
| IQ3_XXS down, 1 / 3 / 4 blocks | 142 / 128.5 / 130.7 µs | — |
| IQ3_XXS down, two rows per lane group (`kPair`) | 130.1 µs | — |
| Q2_K down (existing format) | 112.6 µs | 207 GB/s |
| IQ2_S gate/up fused pair | 236 µs | against 295 µs for two mmvq launches |

FP16 routed down tiles at a 2,048-token batch (`ds41_mmq_bench
--down-type`): IQ2_XXS 11.95 ms, IQ2_S 12.93 ms, IQ3_XXS 13.24 ms, Q2_K
11.3 ms. The first Q2_K tile took 18.2 ms: selecting its 32-bit code word
with a lane-dependent index (`s.q[2*l]`) put the array in 96 B of scratch;
explicit select chains keep it in registers. Every new kernel is at most
23.8 KB of code (under the 32 KB instruction cache) with no scratch. The
ISTA `IQ1_M` 3K scores are identical before and after, decode and batched.
Quality and speed of the build are in
[QUALITY.md](QUALITY.md#streamed-ud-iq2_xxs).

## E68 — streamed prefill: next-layer prediction for mid-size batches (retained)

A warm streamed `UD-IQ2_XXS` pp2048 (146 GiB of experts, 102 GiB cache)
still read 5.4 GiB per pass at 99% route hits, and its GPU idled 235 ms of
a 1.91 s pass. 148 ms of the 189 ms of stalls before `routed_counts` fell
in layers 20-39, the 140-row decoder replay: batches of 17-1,023 rows sent
no next-layer prediction, so every miss there was read on demand. They now
always send an ordinary (not wide) prediction, as 1,024-row batches send a
wide one; batches up to 16 rows keep predicting only after a missed layer.

`gufo bench -p 2048,512 -n 0 -r 4 -v`, alternating, tok/s per repetition:

| Arm | pp2048 passes 2-4 | pp512 pass 1 | pp512 passes 2-4 |
|---|---|---:|---|
| before (2 rounds) | 988 1046 1079 / 979 1089 1093 | 520 / 548 | 565 629 636 / 594 617 620 |
| after (2 rounds) | 1026 1099 1122 / 1019 1102 1124 | 597 / 611 | 586 642 642 / 585 625 645 |

About +3.5% on warm pp2048 and +11-15% on a first pp512, at more reads:
pp512 reads 10-12 MiB per token instead of 4-6, pp2048 3.3-4.2 instead of
2.5-3.5. Reads only: the 3K batched score is identical on all 2,999
positions.

Always predicting cost GSQ `IQ2_XXS`, whose warm passes hit every route:
pp512 865-884 → 803-834 tok/s (the extra router pass per layer and
mispredicted reads, 0 → 2 MiB per token), pp2048 about −1%. A mid-size
batch now predicts only once a layer of the same evaluation waited on
reads. The store counted miss rows only for batches of up to 96 route
rows, so prompt batches always reported none (a first gated attempt
therefore stopped predicting entirely and lost the `UD-IQ2_XXS` gain); it
now counts them for every batch. Alternating, tok/s:

| Build | Arm | pp2048 passes 2-4 | pp512 passes 1-4 | Warm reads, MiB per token |
|---|---|---|---|---|
| `UD-IQ2_XXS` | always | 1070 1140 1149 / 1024 1110 1098 | 631 652 678 675 / 611 618 650 616 | pp512 9.9-12.1 |
| | gated | 1062 1137 1146 / 1033 1110 1126 | 638 654 668 675 / 625 643 669 669 | pp512 8.3-10.4 |
| GSQ `IQ2_XXS` | before | 1307 1317 1257 / 1273 1284 1289 | 757 755 648 795 / 626 764 909 793 | pp512 0, pp2048 0.43 |
| | gated | 1275 1297 1285 / 1281 1297 1290 | 735 791 813 822 / 568 832 865 869 | pp512 0, pp2048 0.43 |

Warm GSQ passes never miss, so they never predict and run the previous
code; its first pass predicts (pp512 reads 1.2 instead of 0.6 MiB per
token).

The same trace showed one 41-42 ms `q_b` call (layer 7, Q8_0, otherwise
5.6-5.8 ms, the `ds41_mmq_bench` time) in two of four passes, with no
other kernel or copy running; not explained, not reproduced in decode.

## E69 — decode: Q8_0 `q_b` GEMV in four slices (retained)

`gemv_slices` came from a Q5_K sweep. `UD-IQ2_XXS` keeps 27 `q_b`
projections in Q8_0 (32,768 × 1,280), which ran at ~211 GB/s with two
slices. Q8_0 `q_b` by slices and rows (`ds41_v41_gemv`, DRAM-cold, all
widths bit-identical to single-token launches):

| Slices | 1 row | 2 | 4 | 8 | 16 |
|---|---:|---:|---:|---:|---:|
| 2 (before) | 214.7 µs | 221.3 | 228.1 | 238.5 | 275.1 |
| 4 (retained) | 208.3 | 216.0 | 215.0 | 223.9 | 266.6 |
| 8 | 202.1 | 200.7 | 217.0 | 245.6 | 311.4 |

Four slices win at every verification width; eight lose from 8 rows. The
shared down projection (k 2,304) measured within ±2 µs at one or two
slices and stays at two. About 0.17 ms per decode token. The split changes
the sum order: 3K decode score 5.788 (oracle 5.818, ΔNLL against the
previous build +0.0018 ± 0.0040), and verification at widths 8 and 16
stays bit-identical to serial decode on 36 of 36 rows.

Decode budget after both (`UD-IQ2_XXS` tg128, 55 ms per token): dense
GEMVs 32.7 ms (6.9 GB, floor 28.9 ms at the measured 238.6 GB/s), routed
experts 11.5 ms (floor 10.2 ms), hc mixes and norms 3.6 ms, BF16 router,
compressor and indexer 1.6 ms, attention 0.7 ms; the GPU is busy between
~2 µs launch gaps.

## E70 — served coding agent on streamed `UD-IQ2_XXS`; expert layout from its trace (measured)

Pi 0.87.0 through `tests/functional/pi_agent.py` (creation, bugfix and edit,
five rounds in one retained session, thinking off) against a fresh
`gufo serve` (`--stream-experts`, Lucebox DSpark drafter, 64K context,
expert tails on a second drive). All 15 tasks pass in both runs.

| Run | Requests | Decode | Prefill (summed) | Route hits | Read per evaluated token | Read wait per evaluated token |
|---|---:|---|---|---:|---:|---:|
| production | 52 | 5,756 tokens, 38.5 tok/s | 7,476 tokens, 223 tok/s | 97.6% | 36.2 MiB | 2.05 ms |
| traced (no prediction) | 51 | 6,349 tokens, 36.2 tok/s | 7,706 tokens, 199 tok/s | 97.9% | 21.6 MiB | 2.59 ms |

Prompts are ~8.2K tokens (median) of which ~37 are new per request. The
bench's repeated pp2048 read 2.5-3.5 MiB per token; the agent reads ten
times that, as E61's 64 GiB stand-in did. Prediction reads 68% more to cut
the read wait by a fifth.

The trace (55,760 layer prepares, 315 GiB of missed experts, 38.6 s of
read wait in a 214 s session) was replayed against expert orders per
layer, keeping its recorded misses; a run joins misses at most one
unneeded expert apart:

| Batches | Missed | Read wait | In runs of ≥ 4 experts: index order | usage order, this trace | usage order, built-in rank |
|---|---:|---:|---:|---:|---:|
| 1-16 rows (decode, verification) | 204 GiB | 26.2 s | 0.0% | 0.0% | 0.0% |
| 17+ rows (prompts) | 112 GiB | 12.4 s | 4.4% | 11.3% | 14.5% |

Decode and verification miss about one rarely routed expert per layer
step, which no file order makes adjacent; prompt misses form long runs for
at most 15% of their bytes, worth ~0.5 s of the session at E30's +36% for
long runs. A rank-ordered or replicated expert file does not pay for this
workload; single-expert decode misses (~12% of the session) are the cost.

## E71 — narrow BF16 matrix-core tiles for short prompt passes (retained)

A served follow-up turn is one 20-40 row prefill pass. At 20 rows its BF16
router (384 × 5120) took 350 µs per layer in the 128-row WMMA kernel: three
blocks for the whole matrix. `matmul_bf16_wmma_narrow_kernel` handles
16-64 tokens with 64 weight rows × 16 tokens per block (four waves sharing
one split of the activations), four 32-value K steps per barrier, and the
next group's loads in flight during the matrix-core work. Products, K
steps, hi/mid/lo part order and FP32 sums match the wide kernel; the test
compares both bit for bit at 384 × 5120, 512 × 5120 and 4096 × 1280 for
16-64 tokens.

| Router variant (pp20 @ d8192) | Mean per call |
|---|---:|
| 128-row wide kernel | 350 µs |
| one wave, 16 rows × 32 tokens | 146 µs |
| four waves sharing activations | 149 µs |
| + four K steps per barrier | 115 µs |
| + 16-token tiles (retained) | 89 µs |

Unprofiled `UD-IQ2_XXS` with streamed experts, depth 8192, two alternated
rounds of three repetitions each:

| Test | Before | After |
|---|---:|---:|
| pp20 | 91.3 / 92.2 tok/s | 98.5 / 98.4 tok/s (+7%) |
| pp32 | 122.1 / 121.6 tok/s | 129.3 / 129.8 tok/s (+6%) |
| pp64 | 198.0 / 204.0 tok/s | 214.5 / 213.7 tok/s (+6%) |
| pp2048 | 1162.0 / 1156.7 tok/s | 1161.6 / 1156.7 tok/s |

pp16, the first test after warmup, varies by ±25 tok/s in both builds.

## E72 — short prompt passes: earlier prediction and undrained adoption (measured, not retained)

A served follow-up turn on streamed `UD-IQ2_XXS` is one 15-20 row prompt
pass; at 24K context its time to first token is 340-500 ms, of which a
warm pp20 pass is ~210 ms.

**Predicting from the first layer.** Batches of 17-64 rows predicted the
next layer only after a layer of the same pass missed (E68). Predicting
from every layer did not change the long-document Q&A follow-ups (two
alternated rounds, all answers passing, time to first token per follow-up
within ±40 ms; summed 5.97 / 6.44 s before, 6.09 / 6.64 s after): about 3%
of a follow-up's ~4,800 routes miss, spread over all 40 layers, so the
first or second layer already opens the gate.

**GPU idle in pp20 passes.** A warm pp20 pass at depth 8192 left the GPU
idle ~32 ms in gaps of 50 µs-1 ms before each layer's routed experts. The
host spends them in the store, not in HIP calls: per pass ~15 ms waiting
on demand reads and 14-21 ms in `Resolve`, which drains the adopted
prediction's in-flight reads before planning. Adopting without draining
(the phase's final wait still covers those reads) moved the time into the
read wait (27 ms per pass) and saved 2-10 ms of a 209-219 ms pass; the
overlapped work at 20 rows (shared expert, next-layer router) is ~1 ms,
too little to hide reads behind. pp20/pp64/pp2048 benches were within
noise. The remaining idle is SSD reads, which only fewer or earlier reads
reduce.

## E73 — one decoder replay per served prompt (retained)

Served prefill steps end at cache captures (grid checkpoints every 2,048
tokens, the stable boundary, the prompt end). The server passed each step
the prompt cut at that capture, so every step treated its end as the prompt
end and ran the decoder layers (20-39) over its last 140 rows: five
replays for a 20K-token document, ~0.9 s each on streamed `UD-IQ2_XXS`
(cold decoder experts), see [PREFILL.md](PREFILL.md). The runner state now
carries the whole prompt's length, and the decoder layers run only the rows
within 140 of the real prompt end.

A mid-prompt checkpoint then holds no decoder rows near its end. The
session tracks where its decoder rows start; checkpoints and snapshots
record it, and a restore (or live frontier) serves only prompts whose
replay starts there or later. Shorter continuations fall back to the
previous checkpoint (`BACKLOG.md` item 36); none of the 50 short resumes in
the recorded agent and document sessions came from such a checkpoint. A
step that skips all decoder rows empties the DSpark ring at its end, so
these checkpoints restore with the drafter on.

Prompts now run the decoder layers exactly as an uncached single pass does.
Restores that resume (from the 22,528 checkpoint of a 24K-token document on
`UD-IQ2_XXS` with DSpark; six prefixes of a 24K-token document on resident
`IQ1_M`, including one that fell back) gave the same tokens and logprobs as
uncached requests, bit for bit. The previous build's uncached requests
differ slightly (first logprob -0.0892 against -0.0907): their extra
replays sent rows through layer 20's full block instead of its
compression-only path.

Long-document Q&A on streamed `UD-IQ2_XXS` with DSpark, two alternated
rounds, time to first token:

| Request | Before | After |
|---|---:|---:|
| Fresh 24.0K-token document | 28.87 / 28.71 s | 26.11 / 26.13 s (-2.7 s) |
| Fresh 20.0K-token document | 24.05 / 26.01 s | 21.22 / 23.98 s (-2.4 s) |
| Fresh 14.2K-token document | 17.81 / 17.91 s | 14.55 / 15.73 s (-2.7 s) |
| Reopened 24K document, 1,459 tokens past the 22,528 checkpoint | 2.74 / 2.80 s | 3.35 / 3.09 s |

All 24 answers passed in every round. The reopened document reads more
expert data per token in the new build (16.7 against 15.7-16.1 MiB); the
earlier build's extra replays had left more decoder-layer experts cached.
On resident `IQ1_M` a fresh 23,936-token prompt went from 19.5 to 17.8 s.
The `cache`, `cache-edits` and `long-context` functional suites pass on
`IQ1_M` with DSpark; `cache-depth` stops at its 16,384-token prompt
minimum, which DS41 tokenizes to 16,210.

## E74 — BF16 64-row wide tiles (rejected)

On `c49e1fec`, compare the existing 128- and 64-row wide specializations
on the same inputs, 30 event-timed calls after four warmups, three repeats.
At 2,048 tokens, 64-row tiles regress the router (384 × 5,120) from
~1.27 to ~1.30 ms, compressor (512 × 5,120) from ~1.65 to ~1.80 ms,
and indexer (4,096 × 1,280) from ~2.60 to ~3.01 ms. Every output is
bit-identical, also at 65 and 140 tokens. These are warm microbenchmarks,
not request timings; reject the long-prefill proposal before model runs.
Built with HIP `-O3 -ffast-math -fno-finite-math-only --offload-arch=gfx1151`.
[Source and raw results](artifacts/2026-11-02-bf16-prefill/).

The new `tools/ds41/route_occupancy.py` reports per-prepare occupancy and
exports 2,048-token `COUNTS` lines for `ds41_mmq_bench`. A route trace
(`DS41_EXPERT_TRACE`) of the served long-document Q&A session contains 520
such batches: mean 419.19 64-slot tiles, mean useful-slot occupancy 45.86%.
This covers the complete session, not an isolated request. Older count files lack matched
provenance; these figures do not establish the cause of the served/bench gap.

## E75 — grouped BF16 router/compressor prefill (retained)

Reuse the existing narrow kernel at 32 tokens × 64 weight rows, four waves,
two 32-value K steps per barrier, for batches above 64 tokens with at most
512 weight rows and K divisible by 64. Smaller batches and the 4,096-row
indexer keep their existing dispatch. No new arithmetic or runtime switch.

Warm HIP-event microbenchmarks, four warmups, 30 calls, three repetitions,
`-O2 -ffast-math -fno-finite-math-only`, 2,048 tokens:

| Shape | Existing ms | Candidate ms |
|---|---:|---:|
| Router 384 × 5,120 | 1.274 | 1.159 |
| Compressor 512 × 5,120 | 1.640 | 1.425 |
| Indexer 4,096 × 1,280 (excluded) | 2.605 | 2.772 |

All outputs match bit for bit, also at 65 and 140 tokens. The candidate
passes `ds41.attention`, including added double-precision cases at 65 and
140 tokens; the shared formatting check passes. Full model 3K teacher
scores and DSpark qualification were not rerun for this experiment.

Production AR pp2048/tg128 at depth 8192, UD-IQ2_XXS streamed, fixed 96 GiB
expert budget, three repetitions per process in ABBA order:

| Run | pp tok/s | tg tok/s |
|---|---:|---:|
| Baseline A1 | 1091.14 ± 59.85 | 19.04 ± 0.48 |
| Candidate B1 | 1095.30 ± 65.38 | 19.05 ± 0.47 |
| Candidate B2 | 1094.55 ± 62.76 | 19.04 ± 0.46 |
| Baseline A2 | 1089.00 ± 59.41 | 19.04 ± 0.48 |

HTTP AR, context 65536, one session, fixed 90 GiB expert budget, two fresh
documents per process, one generated token (no draft/accepted tokens):

| Request | A1 prompt ms | B1 | B2 | A2 |
|---|---:|---:|---:|---:|
| 23,954 tokens | 25965.90 | 25071.84 | 25152.41 | 25247.44 |
| 19,927 tokens | 19784.48 | 19505.21 | 20307.85 | 21185.86 |

All responses produced the same single token; this is limited output
agreement, not a full quality evaluation. Request timing remains inconclusive:
streaming/cache variance is larger than the expected kernel saving (~0.05 s
per 20K-token prompt from the kernel times above). Retained for the measured,
bit-identical kernel gain; it is not a qualified end-to-end speedup. Raw runs
are in the [artifacts](artifacts/2026-11-02-bf16-prefill/).

The excluded indexer already launches 1,024 wide blocks at 2,048 tokens,
versus 96 for the router. The candidate makes four times as many blocks,
with half the rows and half the tokens per tile: nominal weight loads and
activation splitting are each duplicated across twice as many tiles.
Reduced synchronization and more blocks benefit the small-row shapes; for
the already-wide indexer the extra work likely outweighs that benefit.
This explanation follows the tile geometry; no hardware-counter ablation
has isolated the dominant cause.

## E76 — routed FP16 128-value activation stages (rejected)

Double `kStageK` from 64 to 128, leaving slot tiles, decode and WMMA order
unchanged. Recorded served trace exported by `route_occupancy.py`, counts
line 10, 2,048 tokens, IQ2_XXS gate/up and IQ2_S down, seven repetitions:
gate/up 23.18 → 23.45 ms, down 12.91 → 13.44 ms. Both benchmark numerical
checks have identical errors against their independent reference. Halving
the stage barriers does not pay for the larger shared activation allocation
on this shape. Reverted before full-model timing.
[Raw results](artifacts/2026-11-02-routed-stages/).

## E77 — routed FP16 32-value activation stages (rejected)

Same E76 served route counts and seven-repeat benchmark, but halve the
64-value stage. Gate/up 23.18 → 23.38 ms; down 12.91 → 13.45 ms.
Benchmark reference errors remain identical. Reverted: neither direction
improves this recorded IQ2_XXS/IQ2_S layer. E76's full `ds41.mmq_typed`
check also passed (25.82 seconds); its log is retained with these artifacts.

## E78 — IQ2_S sign masks without LDS lookup (rejected)

On the E76 recorded route workload, replace the sign-byte LDS lookup with
integer bit spreading. Shift/mask expansion: down 13.46 ms versus 12.91 ms
baseline. Two nibble multiplies by `0x10204080`, masked by `0x80808080`:
12.96 ms. Removing the now-unused 2 KiB sign table and its initialization:
12.99 ms. All three give identical independent benchmark-reference errors;
all 16 nibble inputs were exhaustively checked for exact bit spreading.
Gate/up (unchanged IQ2_XXS control) is 23.05–23.11 ms versus 23.18 ms.
No demonstrated benefit, so all variants are reverted. No full model run
or full operator suite is claimed for these rejected variants.

## E79 — routed iterative-ILP scheduler (rejected)

Apply `-mllvm -amdgpu-sched-strategy=iterative-ilp` only to routed FP16.
E76's recorded-route benchmark: IQ2_XXS gate/up 23.07 ms, IQ2_S down
12.89 ms against 23.18 / 12.91 ms. Reference errors unchanged. This is
not a meaningful demonstrated gain, so restore the compiler settings.
Baseline resource analysis shows 186 VGPRs/23,808 B LDS for IQ2_XXS pair
and 208 VGPRs/37,120 B LDS for IQ2_S down, with no spills. The gate/up
static ISA has 401 half-wave exchanges and 530 delay instructions versus
128 WMMA instructions; these static counts are not dynamic cycle costs.
[Resources and timing](artifacts/2026-11-02-routed-layout/).

## E80 — IQ2_S grid table as packed bytes in LDS (rejected)

The IQ2_S down keeps 37,120 B of LDS per block (E79), 16 KiB of it the
grid table decoded to FP16. Storing the grid as its packed bytes (8 KiB)
and converting to FP16 at each decode leaves the values and outputs
unchanged (identical benchmark reference errors), but costs more ALU work
than the smaller table saves. On E76's recorded served routes, three
alternated rounds of seven repetitions: down 12.87-12.91 ms before,
13.82-13.87 ms after (+7%); the IQ2_XXS gate+up control is unchanged
(23.18-23.29 ms). Reverted.

## E81 — routed FP16: down-fragment sharing off, and why tile surgery is closed (rejected)

Three checks on `ds41_mmq_bench --f16 1` over E76's recorded served routes
(counts regenerated from the surviving served trace; line 10, seven
repetitions, three rounds; gate+up 23.06-23.13 / down 12.88-12.91 ms
baseline, reproducing E76):

- Activation-fragment sharing off for the down kernels (each down fragment
  feeds one product, where a pair fragment feeds two, so the exchange
  permutes buy half as much): IQ2_S down 12.91 → 13.25 ms, IQ2_XS down
  12.07 → 12.35, IQ1_S control unchanged. The two 16-byte LDS reads cost
  more than the permutes even for single-product fragments, as in E25's
  dense kernel. Reverted.
- Tile-width arithmetic on the served traces: 2,048-row encoder batches
  hold 493,128 live 16-slot groups against 399,360 ideal (23.5% padding),
  and the live count is identical for 64- and 32-slot tiles (per-expert
  `ceil(count/16)` is invariant), so E8's direction cannot reduce WMMA
  work. Cutting the remaining padding needs cross-expert groups, which
  share no B fragment: one decoded fragment per group raises decode work
  948/419 = 2.26x and re-reads each expert's weights per group (an extra
  ~2.5 GiB of DRAM per gate+up call). Dead on arithmetic, not worth
  building.
- A wave64 rewrite would halve the issue count only if the w64 fragments
  held half the data per lane; the gfx11 builtin signatures
  (`wmma_f32_16x16x16_f16_w64` "V4fV16xV16xV4f" against w32's
  "V8fV16xV16xV8f") keep 16 halves per lane in both, doubling operand
  redundancy. Ruled out from the ISA definition, no build.

## E82 — two-wave prefill: the resident experts' gate+up during the read stall (retained)

The store's read overlap already split decode batches (`hit_ids`/`miss_ids`
kept route positions); prefill batches never got the split (`kSplitRows`
covered only speculative rows), so every 2,048-row layer idled ~10.5 ms
waiting for its misses' reads before the routed FP16 kernel could start.
`kSplitRows` now covers `DS41_MAX_PREFILL_BATCH` rows, and the overlap
callback plans `hit_ids` on the device and launches the routed FP16
gate+up over the resident experts while the readers stream; after the
prepare the same kernel runs over `miss_ids` (the plan buffer is
stream-ordered per use), and the down pass replans the full routes. Each
slot is written once by the same arithmetic, so outputs are unchanged;
batches without a hit/miss mix keep one full pass, and decode-sized
batches keep their grouped-GEMV split (non-FP16 prefill builds skip the
split).

Correctness: `ds41.mmq_typed`, `ds41.attention`, `ds41.expert_store` pass;
the 3,000-position decode and batched scores and an 18,499-position
4096-token-batch score are bit-identical to the pre-change build over
every position, argmax and NLL (`ds41_load_probe --stream-experts`, fresh
cache per run, tails attached).

Speed, production binaries, alternating runs:

| Test | Before | After |
| --- | ---: | ---: |
| `gufo bench -p 2048 -n 0 -d 8192 -r 3` | 1109.35 / 1108.03 tok/s | 1160.17 / 1161.78 |
| 18.5K probe, 4,096-token batches (logits per position) | 33.3 s | 31.4 s (−5.9%) |
| Served doc 1 (23,954 tokens, fresh server) | 26.30 / 26.28 s | 24.73 / 24.47 s |
| Served doc 2 (19,927 tokens, same server) | 20.21 / 20.75 s | 19.19 / 19.31 s |

The store's host wait per token barely moves (1.66 → 1.53 ms at 4,096
tokens): the gain is the resident experts' ~20 ms of gate+up filling the
read stall, not shorter reads. Full-model 3K teacher scores and DSpark
qualification were not rerun; the bit-identical teacher-forced scores
above cover the changed path.

## E83 — layer-0 prediction, encoder batches stop at layer 19, Engram rows in layer order (retained)

A served profile after E82 (two fresh documents, `PREFILL.md` method,
attributing each idle gap to the routed kernel that follows it) left
1.53 s of encoder read idle across 400 layer calls, median 0.9 ms. Layer
0 of every 2,048-row batch carried most of it, 15-73 ms per batch (228 ms
on a fresh server): it has no earlier layer to predict its experts.
Meanwhile layer 19 of each encoder-only batch predicted layer 20, which
the batch never runs, and the next batch's layer-0 prepare dropped that
job (9-10 GiB of dropped prefetch per served run). The Engram projection
of layer 1 also waited 50-117 ms per batch: layer 14's 49K row reads
started at the same time and shared the drive.

- `predict_first_layer`: prompt batches of 1,024+ rows route layer 0 on
  the embedding, normed with layer 0's FFN norm, before its attention;
  `ds41_expert_store_expect` reopens the prediction window at layer 0 (the
  previous evaluation's kernels are ordered before the prediction event
  the store waits on, so the parity half is free).
- `pass_end`: a batch whose rows all stop at the replay boundary predicts
  nothing past layer 19.
- Engram rows are read on one thread chain in layer order, layer 1 first.

Correctness: predictions and reads only move data earlier, so the 3K and
18.5K (4,096-token batches) batched scores match the previous build at
every position; `ds41.expert_store` passes.

Profile of the same two documents:

| | E82 | E83 |
| --- | ---: | ---: |
| Encoder read idle (400 calls) | 1.53 s | 0.96 s |
| Largest encoder idle | 228 ms | 40 ms |
| Engram row waits | 1.55 s | 1.07 s |
| Decoder replay idle (40 calls) | 1.71 s | 1.67 s |
| Dropped prefetch per run | 9-10 GiB | 0.7-1.0 GiB |

Served, fresh server per round, three alternated rounds (ms):

| Request | E82 | E83 |
| --- | --- | --- |
| 23,954-token document | 24,460 / 24,275 / 24,526 | 24,043 / 24,115 / 24,125 |
| 19,927-token document, same server | 19,945 / 20,746 / 21,095 | 20,064 / 21,146 / 20,926 |

The follow-up document slows by ~250-320 ms with every fresh server
run, on either build, so an A-then-B round order biases it against the
candidate. Fitting run order and build together over the six runs: the
fresh document gains 345 ± 89 ms (−1.4%), the follow-up 135 ± 286 ms
(within noise). Layer-0 prediction alone (without the Engram order) fits
−162 ± 198 and −68 ± 140 ms. `gufo bench -p 2048 -n 0 -d 8192` warm
repetitions are unchanged (UD-IQ2_XXS 1,169-1,193 against 1,169-1,190;
GSQ IQ2_XXS 1,310-1,317 against 1,312-1,318).

## E84 — two-wave prefill: the resident experts' down in the read stall too (retained)

E82 ran only the resident experts' gate+up during the reads; the down
pass waited for the reads and replanned every route. The overlap
callback now runs the resident down on the same device plan right after
their gate+up, and after the reads the read rows run gate+up and down on
the miss plan. The full-route replan is gone. Every slot is still
written once by the same arithmetic.

Correctness: the 3K and 18.5K (4,096-token batches) batched scores match
the previous build at every position.

The split down costs no GPU time: over the two served documents the
2,048-row down kernels take 5,444 ms in 800 launches against 5,611 ms in
400. Encoder idle after the resident work falls from 963 to 819 ms; what
remains is ~0.7 ms of host work per layer after the reads (admission
copies, planning) and 11 of 400 layers whose reads outlast the resident
work.

Served, fresh server per round, three alternated rounds (ms), E83 then
E84:

| Request | E83 | E84 |
| --- | --- | --- |
| 23,954-token document | 23,606 / 23,960 / 23,766 | 23,409 / 23,550 / 23,702 |
| 19,927-token document, same server | 18,643 / 19,413 / 19,943 | 19,014 / 19,904 / 20,082 |

Fitted with the run-order drift (see E83): −280 ± 115 ms (−1.2%) on the
fresh document, +38 ± 162 ms on the follow-up. `gufo bench` warm
repetitions are unchanged (1,157-1,181 against 1,158-1,189 tok/s).

## E85 — Engram lookahead: the next prompt batch's rows during the current one (retained)

After E83, layer 1's Engram projection still waited 40-85 ms in almost
every 2,048-row batch of long prompts (1.53 s over a 58,028-token prompt,
2.8%): its 49K random 144-byte row reads start with the batch, and the
110 GB table does not stay in the page cache. Dropping duplicate rows
would save at most 16% of them (75% of 2-gram, 86% of 3-gram and 90% of
4-gram rows are distinct within a batch of the 18.5K scoring document).

`ds41_session_sync` now passes each batch the next batch's tokens
(`ds41_decoder_set_lookahead`). Once the batch's own rows are read, it
hashes the next tokens on a copy of the n-gram history and reads their
rows into a second set of pinned buffers (`SharedActivations`, ~100 MB at
2,048-row batches). The next batch uses them when its position, tokens,
history length and last three history entries match; otherwise, and for
image prompts, it reads as before. The pending read is joined by the
chain thread, so only the first Engram layer can wait for it, and by the
joint path, decoder destruction and the shared buffers' destructor.

Correctness: on a 58,028-token prompt, 16 greedy tokens and their top-5
logprobs are identical to the previous build. The probe does not use the
session, so its scores are unaffected.

Engram waits over the two served documents fall from 1,213 to 500 ms;
what remains is each prompt's first batch.

Long runs heat-soak the machine: four back-to-back 58K runs took 63-68 s
on either build against 52-54 s with a 90 s pause before each run. With
the pause, ABBA order:

| Request | E84 | E85 |
| --- | --- | --- |
| 58,028-token prompt | 53.28 / 53.47 / 53.13 s | 52.12 / 52.10 / 52.29 s (−2.1%) |

Two served documents, ABBA order over four rounds (ms):

| Request | E84 | E85 |
| --- | --- | --- |
| 23,954-token document | 23,522 / 23,802 / 23,735 / 24,292 | 23,129 / 23,463 / 23,552 / 24,048 |
| 19,927-token document | 19,491 / 20,219 / 20,472 / 20,437 | 19,826 / 20,025 / 20,398 / 20,713 |

The fresh document gains 183-393 ms in every round; the follow-up, whose
rows are partly cached by the first document, is unchanged.

Pi agent sessions (`tests/functional/pi_agent.py`, creation, bugfix and
edit, five passes in one conversation, server-default sampling) against
a5b5ea09, before E82-E85, two ABBA rounds with 90 s pauses: all 60 tasks
pass on both builds. Trajectories differ, so requests are compared by
prefill size:

| Prefill tokens | a5b5ea09 | E85 |
| --- | --- | --- |
| 0-64 | 72 requests, 79 tok/s, mean TTFT 417 ms | 66 requests, 83 tok/s, 368 ms |
| 64-512 | 35 requests, 219 tok/s, 932 ms | 33 requests, 241 tok/s, 793 ms |
| 512-2,048 | 3 requests, 556 tok/s | 6 requests, 517 tok/s |

Decode is unchanged (33.9 against 33.5 tok/s).

## E86 — decoder replay: read-ahead and cache protection (simulated, not built)

Each prompt longer than 140 rows ends with layers 20-39 on its last 140
rows, and each of those layers waited ~42 ms for reads (0.85 s per
20-24K-token prompt after E85). Trace mode now records predictions for
passes of up to 140 rows and every decoder layer ahead, and, for the last
batch, the replayed rows' predictions of every decoder layer from the last
encoder layer's FFN input. `tools/ds41/replay_sim.py` replays those traces
on `prefetch_sim.py`'s cache model (99.2% and 99.6% of route hits
reproduced). Two traces: the two served documents (2 replays, 2.03 s of
recorded replay wait) and a Pi agent session (creation, bugfix and edit,
five passes; 10 replays, 2.79 s).

| | Documents | Agent session |
| --- | --- | --- |
| Replay misses caught by layer 19's top-6 / 8 / 12 | 65 / 73 / 83% | 54 / 61 / 71% |
| Extra uncached experts those predictions read | 90 / 114 / 151% of the misses | 249 / 316 / 424% |
| Replay stall, read-ahead from the end of layer 19 (top-6 to 12) | 2.03 → 1.72 s | 2.79 → 2.39-2.42 s |

The read-ahead estimate charges nothing for the extra reads and still
recovers only 14-15%: a replay needs ~8.5 GiB and runs ~100 ms of
compute, so it is bound by the drives' ~10.4 GB/s, which the prompt's own
encoder reads already hold at ~7.5 GB/s on average (7.9 MiB per token).

Cache policies, documents trace (estimated stalls; trace mode runs
without prediction, so prompt stalls are overstated against production,
where prediction hides most of them):

| Policy | Replay reads, stall | Prompt reads |
| --- | --- | --- |
| Current | 17.1 GiB, 1.75 s | 153.3 GiB |
| Prompt batches never evict decoder-layer experts | 12.2 GiB, 1.25 s | 172.6 GiB |
| Protect decoder experts a replay or decode used within 500-10,000 prepares | unchanged | unchanged |
| Prompt admits experts routed ≥ 16 times (n/128) | 19.1 GiB, 1.95 s | 134.6 GiB |
| Prompt admits every expert it reads | 32.5 GiB, 3.33 s | 67.0 GiB |

Protection trades ~4 GiB of prompt reads per GiB of replay reads, as the
earlier simulation found; the finer rule protects nothing, because a
fresh server's replay misses were seeded experts the prompt evicted, not
experts a replay had used. Admitting more halves the prompt's reads (a
batch's misses recur in the next batch) but evicts the replay's experts,
and item 31 measured admission's per-eviction scan and copies costing
more than the reads they saved. In the agent trace none of the policies
moves anything by more than 0.15 s; its reads are dominated by decode
(177 GiB against 24 GiB for replays). Not built.


## E87 — greedy DSpark: learn the plain step, price the draft, decide before drafting (retained)

DECODE.md lever 1. With the confidence head, every greedy cycle ran the
drafter (~9 ms) and only then picked a width, and `cycle_ms[0]` was never
learned in that mode: it stayed at its 42.4 ms resident prior while a
streamed plain step costs ~57 ms. The policy now:
- learns `cycle_ms[0]` from every one-row cycle, after the draft when one
  was discarded;
- learns the drafter pass alone (`draft_ms`) and, once a cycle has
  drafted, compares widths on their cost after the draft;
- drafts only while `gain_ms` exceeds `draft_ms`. `gain_ms` follows the
  time each drafting cycle actually saved: its emitted tokens at the
  plain-step cost minus its time after the draft, zero when it verified
  none. Joint cycles, whose time is shared, count the saving the width
  predicted. While drafting does not pay, single probes re-measure it
  after 4 plain steps, doubling up to 64.

`GUFO_DEEPSEEK41_DSPARK_TRACE` now also prints each learned cycle.
Served on UD-IQ2_XXS streamed (chat, code, 24K document; 512 tokens
each), alternating builds; outputs identical across builds and to AR
greedy:

| Greedy DSpark tok/s | Chat | Code | Long |
| --- | --- | --- | --- |
| Before (3 runs) | 16.5-16.9 | 26.6-27.4 | 24.7-26.2 |
| After (3 runs) | 17.6-18.9 | 28.3-28.7 | 26.4-26.7 |

A first version that ranked drafting by the saving its width *predicted*
(from the confidence head) gave 17.0 and 19.2 on chat: the confidence
underestimates chat acceptance, and the estimate could settle below the
draft cost. Chat still varies with the realized saving, because there a
draft saves about what it costs; that variance is left open.

## E88 — top-p softmax totals on the device (retained)

DECODE.md lever 4. With top-p, `SamplerState` made two passes over the
129,280-entry row (maximum, then an FP64 exp-sum) before selecting
candidates, ~0.85 ms per verified row while the GPU idled. The session
now runs the device sampler's max and sum kernels on the row
(`ds41_v41_softmax_totals`), reduces the 128 block partials on the host in
the same order as the device pick, and hands the totals to the sampler,
which skips its own passes. `SamplerState::UsesSoftmaxTotals` limits this
to temperature with top-p (min-p allowed) and no top-k, penalties, forced
tokens or constraints; everything else keeps the host path.
`sampling_test` checks bit-identical distributions and draws with and
without the totals.

Served as in E87, T 0.7, top-p 0.95, seeded; outputs and draft counts
identical across builds (two rounds, alternated):

| tok/s | Chat | Code | Long |
| --- | --- | --- | --- |
| Sampled DSpark, before | 18.9-19.0 | 24.8 | 26.9 |
| Sampled DSpark, after | 19.3 | 25.7 | 27.9-28.0 |
| AR sampled, before | 16.9 | 18.3 | 17.4-17.5 |
| AR sampled, after | 17.0-17.1 | 18.4-18.5 | 17.6 |

## E89 — decode prediction every layer and two layers ahead (rejected)

DECODE.md lever 2 and BACKLOG item 21. A sampled-DSpark trace
(`DS41_EXPERT_TRACE`, the DECODE.md workloads, prediction off: 24,160
prepares, 95.52% route hits, 8.99 MiB read/token) says the router predicts
79.3% of the decode misses one layer ahead at top-6 but only 74.0% two
layers ahead, and every lead-2 policy loses to lead-1 at the same cap. The
policies the simulator (`tools/ds41/prefetch_sim.py`, router pass 30 µs,
0-0.004 ms/MiB prefetch tax) prefers:

| Policy | Decode estimate | Prefetch |
| --- | ---: | ---: |
| production: lead 1 top-6 after a miss | 72.6-72.9 s | 64 GiB (39.5% used) |
| lead 2 top-6 | 72.2-72.4 s | 201 GiB |
| lead 1 top-6, every layer while the previous evaluation missed | 70.6-71.2 s (+2.3-2.8%) | 148 GiB (30.2% used) |

That window rule (`predict_hot`) was implemented — the previous
evaluation's misses keep a decode batch predicting every layer — and the
3K decode scores stayed bit-identical (argmax and NLL over 2,999
positions). Direct measurement rejected it:

- 3K single-token probe: 58.9 → 63.1 ms/token (+7.1%); demand reads
  3.54 → 2.79/token, prefetch 21.5 → 87.7 GiB, dropped 0.1 → 21.5 GiB,
  read wait 52.2 → 56.6 ms/token.
- `--spec-bench`: 6-row cycles 119.3 → 123.3 ms, 8-row 133.5 → 138.0.
  Gating the rule to multi-row batches (single rows keep the after-miss
  rule) measured the same: 123.3/138.0.

The simulator misses the per-prepare cost of resolving a running prefetch
job — the stop handshake, the in-flight read's remainder and the job
bookkeeping land on every layer's prepare (~0.1 ms/layer, ~4 ms/cycle at
40 layers), which the +2.8% estimate does not price. Misses spread over
all layers (13%/27%/61% of the n>1 miss experts by layers 10/20/30), so
no narrower window captures the value. Reverted; the after-miss rule
stands, and cheap adoption in the store (a hit layer's prepare not
stopping a running job) is the prerequisite for any every-layer decode
prediction.

The same trace closes DECODE.md lever 3's count: a 6-row verification
layer routes 36 rows over 22.6 distinct experts (62.8%), and each row
past the first adds 2.7-3.9 new ones. `expert_group_kernel`'s per-layer
dedup is already at that arithmetic minimum, so the ~9 ms routed cost per
verification row is genuinely distinct reads, not kernel redundancy.

## E90 — dense GEMV slice sweep over the decode shapes (measured, not retained)

DECODE.md lever 5. `tools/bench/ds41_dense_gemv_types.hip` times the
single-token GEMV over the five decode shapes for Q8_0, Q6_K and Q5_K at
S 1-8 and 1-8 tokens, DRAM-cold (weights rotating through 768 MiB of
copies, 50 reps), checking every multi-token result bit for bit against
single-token launches. Three findings:

- **No format gap.** At one token every format runs 199-224 GB/s on every
  shape (`wo_b`: Q8_0 217.7, Q6_K 222.2, Q5_K 220.1 GB/s). The served
  trace's Q6_K `attn_q_b` at 232 GB/s is not reproducible cold, so there
  is no Q8_0-specific deficiency to fix; the remaining 8-12% under the
  240 GB/s ceiling is common to all these GEMVs and, in the model, partly
  the E65-class writeback and side-stream contention.
- **Multi-token sag.** From 1 to 8 tokens the rates drop 3-20%
  (`wq_akv` Q6_K 204 → 163 GB/s): each token adds its FMAs per weight
  byte, and more slices hide that VALU work (`wq_akv` Q8_0 at 6 tokens:
  135 GB/s at S1, 188 at S8).
- **Two production splits are beatable.** The shared down rows (k 2304,
  production S2) prefer S8 for Q8_0 and Q6_K at every width: 6 tokens
  184.7 → 205.2 and 170.8 → 187.5 GB/s, one token a tie or +0.2-2.1%.
  The q_a+kv rows (1792 x 5120, production S4) prefer S8 in every format
  (6 tokens +4.8-14%, one token -0.2% to +5.7%). Q5_K's shared down
  keeps S2, which wins one token by 3 µs.

Not retained: the corrections are worth ~0.06 ms per AR token and
0.4-0.6 ms per 6-8-row cycle (0.1-0.4%), under the E65/E66 bar for a
rounding change. The S paths are pure reorderings (same-weight launches
differ by at most 8e-3 per call, mean relative 1e-5), but a 3K decode
score run drifts through expert-selection flips (5.8% argmax changes,
mean |dNLL| 0.12 from position 0), so every output moves for a
sub-noise gain. The sweep numbers and the tool stay for the next time a
decode GEMV changes.

## E91 — eviction victim lists (retained; pop check corrected in E92)

*Corrected by E92: pops now skip every slot used since its list was built; the long-request and prefill gains below did not reproduce, and the old scan cost at most ~0.15 ms.*

BACKLOG item 31. Once the cache fills, every admitted miss ran `acquire`,
which scans all occupied slots (~13,000 at a 96 GiB cache, ~99 GiB with
the drafter resident) for the lowest decayed score — random host access
over the slot and entry arrays, 1-2 ms per scan at that size, not the
~48 µs the item estimated from a smaller cache. The scan sits in the
admission loop, between the reads and the miss kernels' launch.

Each size class now keeps a victim list: the 32 lowest decayed scores at
build time among occupied slots of that class or larger, rebuilt by one
scan when the list runs dry or ages past 512 prepares. Pops recompute
each score and skip freed, reoccupied or newly hot slots (a touch adds
at least 1 and the recency bonus), so nothing needs invalidation; the
admission comparison uses the recomputed score, and a list that yields
nothing rebuilds once before giving up. The lists make the eviction
choice near-argmin instead of exact: route hits moved 98.8 → 98.7% on
the 3K probe.

Gates and results:

- `ds41.expert_store` ctest passes (eviction, aging and batch-hit
  protection unchanged); 3K decode scores bit-identical to the build
  before (argmax and NLL over 2,999 positions).
- 3K single-token probe: 58.85 → 56.15 ms/token (−4.6%), read wait
  52.21 → 49.81 ms/token.
- Served, DSpark sampled T 0.7/top-p 0.95, two alternated rounds
  (decode_tps, chat/code/after-24K-document):

| Build | Chat | Code | Long |
| --- | --- | --- | --- |
| Before | 19.1 / 19.3 | 25.6 / 24.9 | 24.9 / 23.3 |
| After | 18.6 / 18.9 | 25.6 / 25.6 | 27.7 / 24.5 |

  Chat sits inside its known run-to-run spread (E87: 17.6-19.2); the
  long request, whose decode follows the 24K prompt in the phase where
  misses and admissions cluster, gains 5-11% in both rounds, and its
  prefill 876/813 → 1038/898 tok/s (+10-15%). AR sampled and greedy were
  within noise. The functional cache suite passes with identical
  outputs.

**Corrected by E92.** A pop skipped a slot only when its score had risen
by more than 8.0 since the build, but one route adds 1-5, so slots used
since the build were still evicted. On served sampled DSpark that read
4-6 MiB more per decoded token and cost chat 2.6% (19.4 → 18.9 tok/s in
both rounds against five baseline runs); sampled DSpark does not have
E87's greedy spread. The long-request and long-prefill gains above did
not reproduce against a baseline that decoded 27.9-28.1 tok/s there, and
served host time outside route and read waits puts the old scan at
≤ ~0.15 ms, not 1-2 ms (E92).

## E92 — exact victim-list pops; eviction scoring against the offline optimum (retained; scoring change not retained)

E91's pop test (`score > score_built + 8.0`) missed most uses since the
list was built. Each list entry now records the slot's occupant (layer,
expert) and its last-use tick at build; a pop skips the slot when either
changed, so freed, reoccupied and used slots are all skipped and the
order among the remaining ones is the scan's. `ds41.expert_store` passes.

Served sampled DSpark (T 0.7, top-p 0.95; DECODE.md chat, code and 24K
document, 512 tokens each), fresh server per run, alternated; outputs and
draft counts identical in every run:

| Build | Runs | Decode chat / code / long (tok/s) | Prefill code / long (tok/s) | Decode route hits | MiB read per decoded token |
| --- | --- | --- | --- | --- | --- |
| Before E91 | 5 | 19.4-19.5 / 25.5-25.8 / 27.9-28.1 | 600-613 / 1,033-1,042 | 97.28-97.35% | 63.6-65.3 |
| E91 as committed | 2 | 18.9, 18.9 / 25.7, 25.7 / 28.0, 27.8 | 630, 634 / 1,043, 1,046 | 97.08-97.12% | 69.0-69.8 |
| Exact pops (retained) | 2 | 19.2, 19.4 / 25.6, 25.8 / 28.1, 28.1 | 614, 614 / 1,038, 1,040 | 97.32-97.36% | 63.1-64.4 |

E91's +4% code prefill came from the looser victims and goes with them.
Neither version changes decode: the server's per-token host wait splits
into route wait 18.3-18.5 ms, read wait 4.8-5.5 ms and 0.9-1.0 ms of
everything else, admission included, in every build. At ~6 admitted
experts per decoded token the old scan cost at most ~0.15 ms each, in
line with item 31's original ~48 µs estimate.

**Eviction scoring.** `tools/ds41/evict_sim.py` replays a trace on
`prefetch_sim.py`'s cache model (without prefetch) under other rules,
including Belady's offline optimum. Two traces: the workload above
(`DS41_EXPERT_TRACE`, 24,160 prepares) and E86's Pi agent session (52,280
prepares). Decode misses relative to the current rule (decay 1,600
prepares, recency bonus 4), and modeled stall changes:

| Rule | DECODE.md misses | Decode / prefill stall | Agent misses | Decode / prefill stall |
| --- | --- | --- | --- | --- |
| Current | 1.67% | 10.24 s / 14.27 s | 1.07% | 22.68 s / 10.24 s |
| Decay 3,200, bonus 8 | −3.6% | −0.26 / +0.13 s | −7.5% | −1.32 / −0.26 s |
| Decay 800, bonus 8 | −8.4% | −0.76 / +0.47 s | +7.5% | +1.55 / +0.77 s |
| Decay 6,400, bonus 16 | 0% | +0.06 / +0.10 s | −9.3% | −1.49 / −0.50 s |
| LRU | −15.3% | −1.60 / +1.04 s | +10.1% | +2.10 / +0.64 s |
| Next use from each expert's reuse interval (3 forms) | +83 to +173% | | +31 to +95% | |
| Belady, always admitting | −47.9% | −4.87 / −6.68 s | −53.5% | −11.74 / −4.96 s |
| Belady with its admission bypass | −47.9% | −4.87 / −6.67 s | −53.8% | −11.82 / −4.98 s |

Unrelated requests reward recency, a long conversation rewards frequency,
and the current rule sits between them; decay 3,200 with bonus 8 (or 16,
slightly behind) is the only setting tried that helps both. Belady's gain is all victim choice (the
bypass adds nothing), and about half of all decode misses: what it knows
is which experts come back soon, which no rule built from past uses
recovers here. Reading ahead (DECODE.md lever 2) is the way toward it.

Decay 3,200 / bonus 8 was measured served (exact pops in both builds):
- DECODE.md workload, two runs: chat 19.9 / 20.2 tok/s against the five
  baseline runs' 19.4-19.5; code 25.4 / 25.7, long 27.7 / 28.0, prefill
  595 / 619 and 1,032 / 1,039 tok/s, all within the baseline; outputs
  identical.
- Pi agent session, two ABBA rounds, 15/15 tasks in every session:

| Scoring | Route hits | MiB read per evaluated token | Read wait (ms per token) | Decode tokens | Decode tok/s |
| --- | --- | --- | --- | --- | --- |
| Current | 97.19%, 97.18% | 36.7, 34.6 | 2.52, 2.36 | 5,085, 4,849 | 34.0, 34.9 |
| 3,200 / 8 | 97.32%, 97.51% | 35.3, 36.4 | 2.49, 2.62 | 5,636, 6,766 | 33.0, 33.4 |

  Hits rose in both rounds, but reads and read wait did not fall, and the
  unseeded sessions diverged (11-40% more decode tokens), so the lower
  decode rate cannot be attributed: inconclusive. Not retained; the change
  is kept on `wip/ds41-eviction-score` until a fixed-conversation replay
  can measure it.

## Backlog measurements without an E-number

Measurements and reference data first recorded in [BACKLOG.md](BACKLOG.md)
items, moved here verbatim; the items cite these subsections.

### Memory (BACKLOG)

Resident weights take 112.27 GiB GTT. A session at the native 1M context
adds 0.9 GiB of FP4 state; the batch scratch (1.4 GiB at 1M, 256 MiB of it
indexer scores) is shared by all sessions, and a 2048-token prefill peaks
0.3 GiB higher. Further reductions need smaller prefill batches, which cost
speed.

### Resident IQ1_M decode profile (2026-10-25)

Resident `IQ1_M` decode (2K context, 2026-10-25): 42.4 ms wall, ~2.7 ms of
it gaps between ~1,000 dependent launches. Dense projections and the output
head take ~23.9 ms near the 235 GB/s read peak.

### Item 4: decode launch gaps (2026-10-29, 2026-10-31)

Re-traced 2026-10-29 (IQ1_M, tg32 after pp512): 974 launches and 2.9 ms
idle per 41 ms token, ~0.2 ms of it at the step start (output head →
next step's first copy) and ~2.7 ms as 2–3 µs gaps between dependent
launches. ~90 µs of host work starts each step (embedding dequant and two
synchronous uploads ~70 µs). hc mixes read 2 MB of F32 weights in ~14 µs
(80 per token) regardless of blocks per row. A C8 joint step issues ~2,000
per-session launches, mostly attention.

Streamed UD-IQ1_M (2026-10-31, tg128 after pp2048, after E56/E57): a
layer takes ~1,046 µs over 22 launches, with ~7 µs gaps after the
2,048-block GEMVs (before `rope_tail`, the wo_b GEMV and `hc_expand`),
mostly added by rocprofv3 tracing (E60), and ~2.5 µs before every launch:
~55 µs per layer, ~2.4 ms per token.

### Item 7: prefill glue at 128K (2026-10-28)

Per 2,048-token chunk at 128K of depth (2026-10-28 profile, 1,342 ms of GPU
time): the Q8_0 projections still go through MMQ (30 ms) and its Q8_1 pass
(9.5 ms); the BF16 matmuls take 49 ms (the 384-row router at ~5 TFLOP/s);
the routed rows' closing expand before Engram layers and replay boundaries
8 ms; the shared-expert SwiGLU 5 ms and the routed input's FP16 conversion
4 ms. Each chunk still starts with ~4.5 ms of embedding staging: ~1.2 ms
of host dequantization on four threads, ~1 ms of pinned upload and ~2 ms of
copies into the four residual streams (E33). The Engram projection
(25,600 × 6,144, 17.8 ms per call, two per chunk) still reads FP32 rows;
FP16 rows from the host reader would save 0.8 ms per call and half the
upload, and the shared down projection's SwiGLU input 0.04 ms.

### Item 9: stable-boundary pass

Every chat request captures its stable boundary three tokens before the
prompt end: a 2,035-token prompt runs as 2,032 + 3 rows. The 3-row pass
costs about 65 ms (`gufo bench -p 3`: 64.6 ms).

### Item 19: DSpark host round trips (2026-10-29)

The 2026-10-29 trace of greedy DSpark C1 (`IQ1_M`) leaves ~0.7 ms of gaps
over 20 µs per ~89 ms step, mostly around the logits copies after the
output head (`copyBuffer` → `copyBufferRectAligned`, ~0.28 ms) and the
following copies.

### Item 29: zero-copy expert reads, first version

On gfx1151 the SSD can write straight into managed memory, which kernels
read at the full 240 GB/s (`tools/bench/ds41_zero_copy.hip`: O_DIRECT into
managed memory 6.46 GB/s, GPU checksums correct). Kernel bandwidth lost per
GB/s of expert reads fell from 3.0% with the copy chain to 1.2% with the
DMA alone.

A first version (tag `ds41-zero-copy-wip`, reverted after it) read every
projection in place: slots and stage rows got a page of slack before and
after each projection, a miss took its slot before its reads (never
evicting an expert the batch routes), and the tails packer now cuts at
4 KiB multiples, since btrfs needs page-aligned buffers and silently turns
a misaligned direct read into a buffered one through the page cache.
Outputs matched on IQ2_XXS (also capped at 24 GiB), UD-IQ1_M and IQ1_M, and
every read landed in place, but the served workloads at an 88 GiB cap (two
alternating rounds) got slower:

| 88 GiB cap | Misses/row | Read wait | Decode |
|---|---|---|---|
| With tails | +2-4% | -2 to -13% | chat/agent -2 to -4%, document noise |
| Without tails | +1 to +14% | about equal | -3 to -7% |

The slack takes ~30 KiB per expert, 0.4% of the cache, which replays of
the 88 GiB trace turn into 1.45% more misses; taking slots before the
reads changes misses by -0.16%. Without tails, per-expert reads also lose
the old merging of consecutive experts into 32 MiB reads. The copy savings
are small next to either: at 88 GiB decode moves ~0.8 GB/s of experts, so
less contention is worth at most ~1.5% of kernel time. The replayed
prefetch policies all land within ±1% with today's per-byte cost; a 64 GiB
cache reads ~27 MiB/token.

### Item 30: expert cache memory, replays

The cache gets what is left of unified memory after the dense weights
(5.3 GiB for `IQ2_XXS`), the DSpark drafter (4.6 GiB), a fixed 4 GiB
reserve and the store's own buffers. Replays of the served traces put
+1.8 GiB at -14% decode and -10% prefill misses, and +3.7 GiB at -28% and
-20% (uncapped, ~99.5 GiB cache; -20% and -17% at an 88 GiB cap). The
2026-11-02 prefill-trace replay (PREFILL.md) put +3.7 GiB at −17% prefill
stall and +6 GiB at −26%; available headroom was 5.7 GiB in the traced
session. Serving at 131k with one session, free memory never fell below
3.36 GiB, so ~3 GiB of the reserve sat idle. The prefetch stage is 192
rows (~1.8 GiB), allocated once decode prediction starts; decode jobs use
6-12 rows, the rows exist for prompt-scale (wide) predictions of ~70
experts. The demand stage (~3.7 GiB) holds one row per expert id per
projection; a stage of ~64 rows (~0.6 GiB) would serve any decode batch.
ds4-argodrive measured a two-wave prefill at +1.1% on top of its
read-ahead.

### Item 31: prompt admission, served runs

Admitting every expert a prefill batch routes at least twice (instead of
its final rows and experts routed at least max(4, n/32) times) cut misses
on the 18k-token document's follow-ups by 32% in two served rounds at
88 GiB, but prefill got slower: 797 -> 774 tok/s with tails, 700 -> 673
without; agent prefill -7% without tails, agent misses +16-25%. Fewer bytes
were read while read wait rose, so the cost is in admitting: one
stage-to-slot copy per admitted expert. The trace replay
(`tools/ds41/prefetch_sim.py`) predicted -16-19% prefill stall at full,
88 GiB and no-tails caches and about even at 64 GiB.

### Item 32: GPU clocks during read waits

Measured on streamed `IQ2_XXS` DSpark decode, reading the gfx1151 clock
and power sensors without changing clock settings: the shader clock stayed
at 2.24-2.85 GHz (mostly ~2.82) and the memory clock at 1,000 MHz
throughout, with no drops during read waits. (ds4-argodrive had found the
Apple GPU dropping into low-power states during expert-read waits; a tiny
kernel on a second queue gained 3-4% decode there.)

### Item 36: fallback for short continuations of mid-prompt checkpoints

A prompt ending within 140 tokens past a grid checkpoint falls back to the
previous checkpoint and prefills from there: 1.9 s instead of 0.18 s for a
10,261-token prompt past the 10,240 checkpoint on resident `IQ1_M`, more
when streamed. In the recorded agent (52 requests) and document Q&A (12)
sessions all 50 short resumes continued a live reply frontier, so none
would fall back. Keeping the layer-19 output (hc stream and FFN pre-mix)
of the 140 rows before each grid checkpoint takes ~11.5 MB.

### Layer-major prefill (shelved)

`wip/ds41-layer-major-prefill`: reads fell 32%, but GPU time rose 4-6 s.
It has a known race: reusing adopted job rows across a layer's batches.

### Items 13, 22-26: external measurements and reference data

- **Xolotl route bias (item 13).** Multiplying streamed experts' router
  logits by ~0.5 gave 13.7 → 30.3 tok/s on a 4090 at ~1 point of top-1
  agreement; ×0.25 keeps 85% agreement.
- **mako, prefill out of the eviction scores (item 22).** Raised steady
  decode misses by 20% (19-21 against 15-17 per step), because a prompt's
  experts are the ones its decode uses.
- **mako miss penalty (item 24,
  [source](https://github.com/jhanssen/mako/blob/8567de3/src/kernels/moe_route.h)).**
  `--miss-penalty F` subtracts a constant from the selection score
  (sqrt(softplus) plus the router bias) of each expert the cache lacks; the
  mixing weights stay the unpenalized scores. Short chat turns (16-32
  tokens after 8K of context) missed ~30 experts per token against ~10 in
  steady decode. On Maya-S, 1,300 positions against the reference logits:

| F | Misses/step | Top-1 | KL | Step |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 13.4 | 98.2% | 0.0056 | 44.4 ms |
| 0.05 | 8.6 | 98.2% | 0.0067 | 40.2 ms |
| 0.1 | 6.6 | 98.0% | 0.0088 | 37.9 ms |
| 0.2 | 4.4 | 97.5% | 0.0100 | 35.4 ms |
| 0.5 | 2.1 | 96.5% | 0.0185 | 33.4 ms |

  Prose and code perplexity stayed within noise up to 0.2, and on ISTA
  `IQ2_XS` 0.1 halved the misses (9.5 → 4.8 per step, 31.4 → 34.0 tok/s).
  With its read-ahead the gains add up: 24.6 → 29.1 tok/s. Its two-layer
  read-ahead found layer l+1's router on layer l's FFN input catches 34%
  of misses at top 4 and layer l+2's 30%; reading the top 4 non-resident
  experts two layers ahead, gated off under 35% useful reads, gave +3.5%
  on a story and −0.3% instead of −2.7% after a 16K code prompt.
- **ISTA `IQ2_XS` (item 23).** ~155 GB backbone, closer to the reference
  than `IQ2_XXS` in mako's measurements (top-1 92.5% / KL 0.068 against
  91% / 0.103 on Maya-S logits; prose / code perplexity 1.873 / 2.903
  against 1.935 / 3.025); with the miss penalty at 0.1 it decoded 34.0
  against 33.1 tok/s there. Here the ~104 GiB cache would hold about three
  quarters of its expert bytes, against 83% of `IQ2_XXS`'s experts (12,688
  seeded).
- **mako direction files (item 25).** `--ablate FILE`
  (`src/mtl/ablation.hpp`, format `MAKOABL1`: n_embd, n_layer, n_site, k,
  16-byte site names, then F32 vectors `[layer][site][k][n_embd]`) applies
  `x -= scale (x · v) v` per direction to each row of the attention output
  (`attn_out`) and of the shared expert's output (`shexp_out`) before
  `hc_post`; the routed experts are added after the projection and keep the
  direction. The published DeepSeek file (`skafte/mako-ablations`,
  `refusal.abl`) holds two directions per layer at both sites:
  40 × 2 × 2 × 5,120 F32 = 3.28 MB.
- **A rank-1 abliteration LoRA adapter (item 26).** A llama.cpp LoRA
  (`general.type = adapter`, `adapter.lora.alpha = 1`): one direction r
  taken at layer 22 and removed from every residual writer of every layer
  as `W' = W − r (rᵀ W)`, written as a rank-1 pair per matrix with
  `lora_b = r` and `lora_a = Wᵀ r`. Its 240 F32 tensors (459 MB):

| Tensor | Shapes per layer | Bytes |
| --- | --- | ---: |
| `ffn_down_exps` | a 2,304 × 1 × 384, b 1 × 5,120 × 384 | 456.1 MB |
| `attn_output_b` | a 8,192 × 1, b 1 × 5,120 | 2.1 MB |
| `ffn_down_shexp` | a 2,304 × 1, b 1 × 5,120 | 1.2 MB |

  It is ~140× mako's file although it carries less: it stores `Wᵀ r` for
  each of the 15,360 routed down projections and repeats the same r as
  `lora_b` 15,440 times (~316 MB). `lora_a` was computed from Maya-S's
  dequantized weights, so on another quant it only approximately removes r.
  It reports 98.3% → 4.2% harmful-prompt refusals and HumanEval 95.1% →
  95.7% on Maya-S.

### Item 27: `IQ2_XS` streamed-path survey (code trace, not run)

The ISTA `IQ2_XS` backbone (4 shards, ~145 GiB;
[allocation](https://huggingface.co/ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF/blob/main/tensor-allocation/DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XS.rco-allocation.txt))
only runs with `--stream-experts`. Traced in `runtime/v41_decode.cpp`:

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

### Item 28: ds4-argodrive split reads (external)

[ds4-argodrive](https://github.com/argonautlabsai/ds4-argodrive) (Metal)
splits reads across up to three drives holding complete model replicas,
in 256 KiB blocks by per-drive weight: decode and prefill want different
splits (10:6:6 against 10:5:5 there: the lower-latency internal drive
landed last on two thirds of decode reads at the prefill split), and a
per-drive count of how often it landed last and by how much tells which
way to move a split.

### Item 16: distinct-prompt C8 step (2026-10-29)

With distinct prompts (2026-10-29, after E41) IQ1_M AR decodes C1 / C4 / C8
at 24.5 / 55.5 / 63.0 tok/s, against a ~93 tok/s read ceiling at C8. A C8
step (8 rows, ~42 distinct experts per layer) took ~115 ms of kernels and
~10 ms of host time against ~86 ms of reads before E38; the grouped expert
GEMV now reads at ~190-210 GB/s for IQ1_S and IQ2_XXS gate+up and
~175-200 GB/s for the down projections (standalone), IQ1_M's down at ~180.
With different prompts per user DSpark trails AR at C4 (53.6 against
55.5 tok/s) and C8 (59.3 against 63.0).

### Item 17: resident single-row routed GEMVs (2026-10-29)

`expert_gemv_kernel` at one token (2026-10-29 trace, `IQ1_M` resident):
gate+up 27.6 MB in 135.8 µs (203 GB/s, IQ1_S), down 13.8 MB in 80.6 µs
(171 GB/s) and 15.5 MB in 89.0 µs (IQ1_M, 174 GB/s); the IQ2_XXS layers'
gate+up 31 MB in 180.8 µs (171 GB/s). The retuned decode kernels reach
162–205 GB/s; IQ1 down projections stay at ~165 GB/s with a fixed
per-launch floor (85–96 µs across 13.8–22.7 MB). A plain read of the same
six experts streams at 230–238 GB/s. Several `expert_group_kernel`
variants used 256 VGPRs and spilled 80–540 bytes per lane to scratch.
