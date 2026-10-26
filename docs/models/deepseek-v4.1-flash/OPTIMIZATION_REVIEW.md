# Resident IQ1_M optimization review

Reviewed 2026-10-26 at Gufo `4c298c2111129e2edcc1da4a28867e066e05af37`,
after M6 and before SSD expert streaming. This is a source and architecture
review, not a new performance measurement. Existing timings below come from
[BENCHMARKS.md](BENCHMARKS.md), [BACKLOG.md](BACKLOG.md) and
[EXPERIMENTS.md](EXPERIMENTS.md). Proposed gains remain unmeasured.

The resident implementation contains the main architectural features needed
for this artifact. The most useful next pass is compact candidate rescoring,
sampled speculation, verification kernels and real request batching. Dense
single-token projections are already close to the measured DRAM ceiling.

Follow-up: [E1](EXPERIMENTS.md#post-m6-resident-optimization-pass-2026-10-26)
retains compact FP32 rescoring for 1–4 rows, with measured AR gains and exact
operator/rollback results. WMMA rescoring and compact final selection remain
open; DSpark per-request timing remains inconclusive and unqualified.
[E3](EXPERIMENTS.md#e3-request-local-bpe-piece-memoization-retained) retains
bounded per-span BPE memoization, and
[E4](EXPERIMENTS.md#e4-stop-unused-dspark-markov-suffixes-retained) removes
unrequested Markov proposals without changing the five-row transformer/head.
E4's aggregate request throughput is roughly flat; its short-width draft stage
is faster. Stochastic proposals (E2) and shortened head rows (E5) were rejected
for mixed single-seed request results; neither is active production behavior.
E6 now keeps speculative index scoring on the serial FP32 reduction, with
six-row/18.5K and seven-row/3K exact-logit checks and a stricter probe. Its
32K throughput is roughly flat (about 1% lower in one pair); no speedup is
qualified. Eight-row attention, wider dense verification, actual joint request
batching remain open architectural priorities. E7 now builds stable plans on
device for large batches in the 38/40 fully FP16-routed IQ1_M layers, with
unchanged numerical controls and flat production throughput. Mixed-MMQ plan
metadata and actual graph capture remain open.

## Scope and source pins

The production CMake target builds `v41_decode.cpp`, `v41_session.cpp` and
`v41_dspark.cpp`. It does **not** build the inherited `runtime/session.cpp`
or `runtime/rocm_graph.cpp`. Those files contain capabilities that must not be
mistaken for active V4.1 features, including stochastic proposal callbacks,
confidence policies and batched execution. The same distinction applies to
the V4.1 sampler bridge: an exposed callback is not evidence it is invoked.

All repositories were read at the following revisions. PRs are proposals,
not merged guarantees; their descriptions' performance claims were not
reproduced. External rates are not matched comparisons with Gufo.

| Source | Revision / inspected material |
| --- | --- |
| [Official report](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/DeepSeek_V41_Tech_Report.pdf) | Sections 2.2–2.4 and 3.2; downloaded on review date |
| [DeepInfer](https://github.com/deepseek-ai/DeepInfer/tree/553741557a57df44a46fb3ab72b6aafeaf945337) | `5537415`; CSA2 indexer, heads, bounded replay, sampling and execution structure |
| [DeepInfer ROCm PR #201](https://github.com/deepseek-ai/DeepInfer/pull/201) | `45ccedf`; fused mHC and ROCm kernel adapters; replaces closed PRs #202–206 |
| [deepseek-recipe](https://github.com/deepseek-ai/deepseek-recipe/tree/935817c66f5d6e64838f419692545f06787e485f) | `935817c`; tokenizer documentation, request options and encoding |
| [ds4 main](https://github.com/antirez/ds4/tree/ee7e8405232c8971000cc7ba2b945b00bf6d20d3) | `ee7e840`; reference engine structure |
| [ds4 Strix Halo PR #1036](https://github.com/antirez/ds4/pull/1036) | `815e8fb`; ROCm indexer, matrix paths, verification and published qualification |
| [Kyojin](https://github.com/Yamz-Labs/kyojin/tree/a22e1e398adfc2a2bb4530264eb2313d0ab35f10) | `a22e1e3`; DS41 tiering, gfx1151 kernel notes and memory budget |
| [Spark EXL3 recipe](https://github.com/vcruz305/DSV4.1-Flash-1017-EXL3-Spark-recipe/tree/63220db0a7cc62dd8b783f71c9cf28f909c9b09b) | `63220db`; measured flags, tile-K, graph and Markov choices |
| [Project Maya](https://github.com/mw00/project-maya/tree/6afa525bbabc3a8f5fd165f89e9bc72155d9bd39) | `6afa525`; `src/ds4/` draft, confidence, sampling and overlap paths |
| [Xolotl](https://github.com/Indras-Mirror/Xolotl/tree/03a138f8c4f314ec7196bcef48ed620cf5842e9a) | `03a138f`; `tools/v41/engine/` batch, dense, MoE and DSpark interfaces |
| Gufo Qwen / Gemma | Same Gufo revision; tokenizer, GPU token operations, batched verification, draft policy, vocabulary selection and continuation paths |

Downloaded report SHA-256:
`ba68e2e40408125ae6d2f63a9a241b61c73910691c74ec1a2a7023c851eac08d`.

## What is already present

Gufo implements CED/YoCo sharing, CSA2 own/reindex/reuse behavior, candidate
selection semantics, shifted mHC, disk-backed Engram with overlapped reads,
FP8 SWA numerics and packed FP4 compressed/index state. It also has layer-major
prefill, native FP16 WMMA paths for the major resident quant types, decoder
bounded replay, DSpark, prompt lookup, component caching and vision.

The report's compact kernel flow motivates further fusion: reuse layers use
11 decode kernels in the official system. That count is a direction for
dependency analysis, not a target attainable by copying NVIDIA kernels into
GGUF/HIP. Its separate encoder SWA replay/cache policy is an optional,
approximate extension beyond Gufo's existing decoder replay.

## Priorities

Confidence means confidence in the gap, not in a speedup. Effort is relative.
P1 items should be selected by workload; they do not all need to precede M3.

| Order | Work | Primary benefit | Effort | Evidence |
| --- | --- | --- | --- | --- |
| P0 | Refresh profiles after bounded replay; add active-path counters | Reliable optimization choices | Small | Existing prefill breakdown predates bounded replay |
| P1a | Score compact candidates at decoder reindex layers | Long-context prefill/decode/verification | Medium–large | Confirmed full-history work before masking |
| P1b | Connect stochastic DSpark q and exact p/q verification | Sampled decode | Medium–large | Confirmed unused callbacks |
| P1c | Preserve decode numerics through wider verification | Prompt lookup and high-acceptance drafting | Large | Width boundary already changes output |
| P1d | Batch independent sessions in the transformer body | C2/C4 throughput | Large | Current batch APIs execute serially |
| P2 | Device plans, selected fusions and graphable segments | Prefill and launch latency | Medium | Host route round trips and eager launches |
| P2 | Optimize draft head/Markov path; use confidence | Draft cycle latency/acceptance | Medium | Full five-row draft and full vocabulary head |
| P2 | Heap BPE and bounded piece memoization | Cached-turn latency | Small–medium | Quadratic merge scan and repeated allocations |
| P2 | GPU argmax / exact candidate sampling | Greedy and sampled host overhead | Medium | Full logits always downloaded |
| P3 | IQ wide-load GEMV, routed tiles, dense barriers | Remaining kernel cost | Large | Profile-backed, but several failed variants |
| P3 | Generation-prompt lookahead and cache policy refinements | Agent TTFT and memory | Medium–large | Useful precedents; measure current cost first |

### 1. Compact candidate rescoring is the strongest long-context gap

**Code:** `runtime/v41_decode.cpp:attention`, and
`kernels/rocm/ds41_v41.hip.cpp:ds41_v41_indexer_score`.

At layers 24/28/32/36 the runtime calls the same full-history score function
as an unrestricted indexer. Only afterwards does `ds41_v41_apply_mask` discard
positions outside layer 20's selected blocks. Thus the candidate pool bounds
the answer but not the work. The score kernels' grids are sized by
`max_comp_len`, not the number of candidates.

[DeepInfer's indexer](https://github.com/deepseek-ai/DeepInfer/blob/553741557a57df44a46fb3ab72b6aafeaf945337/deepinfer/model_executor/models/v4_modules/index_attn.py)
constructs compact lengths, sparse scoring metadata and a mapping back to
absolute indices. Apply that design to Gufo's FP4 keys: retain layer 20's
selected block list, score at most 2048 × 8 positions per query in later
reindex layers, and perform top-512 on that compact domain.

At 131072 positions, these four score domains can shrink by roughly 8×;
near 1M by roughly 64×. These are score-domain ratios, **not** model speedups.
Layers 2/8/14/20 retain their required unrestricted search. At or below 16K,
keep the simple dense route unless compact dispatch measures faster.

Preserve absolute-position tie order, the newest pinned block, causal tails,
and FP32 ranking. In particular, compact-list order must not become tie order.
Preserve the selected rows' emitted order too: attention accumulation can
change even when the selected set is identical. The current selector emits
above-threshold rows in position order, then the selected threshold ties.
Keep the small-batch reduction arithmetic unchanged. This can also reduce
rescoring scratch/traffic; it does not eliminate full-score scratch required
by unrestricted source layers. Next, assess tiled exact top-k/block-max to
avoid repeatedly scanning full scores where selection dominates.

### 2. Sampled DSpark uses point-mass proposals, despite a richer bridge

**Code:** `runtime/v41_session.cpp:dspark_step`,
`runtime/v41_dspark.cpp:ds41_dspark_draft`, `dspark_sampler.hpp`.

The active drafter returns greedy argmax IDs after its Markov bias. The
session loop calls `draw()` on target rows and accepts a draft only when
that target draw agrees. It never calls `sampler->propose` or
`sampler->verify`. The bridge's bounded q and residual correction are therefore
unused by this implementation. This is exact point-mass speculation, not a
target-distribution correctness bug. It explains why the existing callback
implementation alone cannot improve sampled acceptance.

Implement q generation **after adding the Markov bias**, sample proposals,
retain the actual normalized q and call exact p/q acceptance with residual
correction. Each next Markov row must use the preceding sampled proposal,
not the old argmax chain. `ds41_dspark_copy_logits` exposes pre-Markov logits
and is insufficient on its own. A bounded q remains exact when the full
target p, penalties and grammar constraints are handled correctly.

Local precedents are Qwen's `hip/executor.cpp:VerifySampledToken` and
`hip/ops/token.hpp`, and Gemma's `engine.cpp:FinishCycle` plus
`sibling_verification.hpp`. Maya also retains proposal distributions in its
[DSpark implementation](https://github.com/mw00/project-maya/blob/6afa525bbabc3a8f5fd165f89e9bc72155d9bd39/src/ds4/ds4_model.cu).
Use those as algorithm references; keep DS41 numerical code model-private.

The existing point-mass path consumes target draws in sequence, so timing
changes to its draft width need not change its sampled tokens when logits
agree. Once p/q is active, changing proposal width can change RNG consumption.
Use a deterministic sampled controller/calibration scope, as Gemma does;
retain timing adaptation for greedy requests. Do not assume the existing
seeded replay evidence qualifies the new sampler.

### 3. Verification needs an explicit numerical mode

**Code:** `v41_session.cpp:kVerifyRows`, `v41_decode.cpp:linear_raw`,
`ds41_v41.hip.cpp:ds41_v41_attention` and `ds41_v41_indexer_score`.

Prompt lookup is capped at six copies plus an anchor because attention
switches at eight rows. Experiments already record changed output with seven
copies. Dense Q5_K switches above eight rows; the indexer independently
switches from the small FP32 kernel to WMMA above four rows. A comment saying
all sub-eight-row kernels retain scalar numerics is consequently too broad:
the indexer boundary also needs direct qualification.

Separate the purpose of a call (prefill versus decode-equivalent verification)
from row count. Extend verification with per-row scalar-equivalent attention,
dense projections and index scores; preserve each reduction/FMA order while
sharing weights. Test widths 1–16, especially 4/5, 7/8 and 8/9, at contexts
where top-512 selection is active. Compare complete logits, selected indices,
state and every rollback prefix against serial evaluation of the same history.

Then widen lookup where it pays. DSpark's learned five-row block is a separate
limit: a wider verifier helps lookup immediately but does not make its
drafter predict additional positions. Do not import Gemma's longer-chain or
sibling defaults: DS41 rows touch many distinct experts and can lose more
weight bandwidth than they save. The current six-row verifier is about
96 ms versus 42 ms for one row in the retained short-context measurements.

### 4. The request batch APIs do not share weight reads

**Code:** `runtime/v41_session.cpp:ds41_sessions_eval_batch` and
`ds41_sessions_dspark_step_batch`; `ds41_session_prepare_batch_execution` is
empty. Both batch functions iterate over complete single-session executions.
Shared scratch saves memory, but does not implement a joint forward.

Flatten active rows from independent sessions into a common layer-major
forward. Share dense weights, group equal routed experts across the cohort,
and scatter results to each session's rings, compressed state and logits.
The drafter body must also batch; batching only its vocabulary head misses
most of the opportunity. Qwen's `hip/batched_decode.cpp` and
`hip/dflash/batch.cpp`, and Gemma's batched cycles are useful local designs.
Xolotl's [batch interface](https://github.com/Indras-Mirror/Xolotl/blob/03a138f8c4f314ec7196bcef48ed620cf5842e9a/tools/v41/engine/ds41_batch.hpp)
also distinguishes wanted logit rows from prompt rows.

This is primarily a C2/C4 throughput improvement. Each row needs its own
position, candidate pool, SWA ring/frontier, Engram history, image routing,
sampler/RNG and speculative commit length. Cancellation must remove rows
before state mutation; rollback in one request must not affect another.
Small cohorts must use verification numerics even when aggregate rows cross
the prefill thresholds. Memory budgets may restrict C4 with DSpark/vision:
report admission and actual residency rather than forcing a headline count.

### 5. Remove route planning round trips before attempting whole graphs

**Code:** `runtime/v41_decode.cpp:moe` downloads route IDs for larger batches,
builds per-expert classes/tiles on the CPU and synchronously uploads the plan.
This occurs per layer. The small-batch expert plan is already device-side.

Build histograms, prefix offsets, stable grouped slot lists and tile classes
on the GPU. A bounded grid over device metadata avoids downloading the tile
count. Preserve slot ownership and summation order. Measure under skewed,
uniform and disjoint routing; do not replace small verification plans with
a large histogram blindly. This removes synchronization and enables capture.

The active V4.1 runtime has eager launches; the inherited graph filename is
not an active captured execution path. Start with stable layer/segment graphs
and widths, with positions and masks supplied as device metadata. Engram
host reads and route-plan round trips are capture boundaries. Preallocate
scratch and test that Engram uploads, state saves and every operation replay.

Existing decode launch gaps total about 2.7 ms/token in the older profile;
that bounds savings from eliminating those gaps alone. The Spark recipe's
graph gains are motivation, not a transferable percentage. Its
[measured configuration](https://github.com/vcruz305/DSV4.1-Flash-1017-EXL3-Spark-recipe/blob/63220db0a7cc62dd8b783f71c9cf28f909c9b09b/README.md)
also uses another GPU, quantization and sampled workload.

### 6. Fuse along actual dependencies, especially mHC boundaries

**Code:** `v41_decode.cpp:block`, `ds41_v41.hip.cpp:hc_state_pass_kernel`,
and the router/RoPE/window launch sites.

Gufo already fuses attention expand into the FFN state pass for larger
batches. Remaining candidates are FFN routed-sum/expand into the next layer's
state pass (with Engram boundaries handled separately), router projection
plus selection, Q projection plus RoPE, shared up projection plus SwiGLU,
and producer stores that can own the SWA update after its last read.

[DeepInfer's ROCm mHC path](https://github.com/deepseek-ai/DeepInfer/blob/45ccedf06de93325e2eadc7b15b744a94aee9c22/deepinfer/rocm_ops/mhc.py)
uses a split partial pass and a reduction/norm pass across sublayer
boundaries. Study that dependency decomposition, not its gfx950 launch
parameters. Its BF16/FP8 contracts differ from Gufo's FP32 residual path.

The existing backlog already identifies most local fusions. Prioritize from
a fresh post-replay trace: decoder layers now run only 140 prompt rows, so
old 2048-row totals overstate their current cost. Preserve intermediate
rounding and do not overwrite a ring slot still needed by attention.

### 7. Optimize the drafter as a separate approximate predictor

**Code:** `v41_dspark.cpp:moe`, `ds41_dspark_draft`,
`v41_dspark.h:ds41_dspark_model`, `model_data.cpp` draft bindings.

Every draft executes all five transformer rows, the full target vocabulary
head, and five dependent Markov projections/argmax reductions, even if only
one or two drafts will be verified. A shorter accepted width therefore does
not save most drafting compute. The model binding also has no confidence
head; the learned confidence in
[DeepInfer's heads](https://github.com/deepseek-ai/DeepInfer/blob/553741557a57df44a46fb3ab72b6aafeaf945337/deepinfer/model_executor/models/v4_modules/heads.py)
is not used. Check supported GGUFs for the tensor before implementing it;
some artifacts may require a converter/sidecar extension.

First profile transformer body, output head, Markov path and feature
injection separately. Potential improvements:

- Score only requested output-head rows and stop the sequential Markov tail
  at the selected width. Keep all five transformer rows unless independently
  qualified: block attention reads the other noise rows.
- Reuse Qwen/Gemma's request-vocabulary technique for the **draft** head,
  with request tokens and multilingual fallback. Gather matching Markov
  output rows too. Keep the target verifier's vocabulary complete.
- Fuse draft gate/up/SwiGLU and group the draft's quantized expert work;
  its current `moe` launches separate MMVQ gate and up passes.
- Use calibrated confidence to trim verification, accounting for conditional
  prefix survival and real cycle cost. Keep the existing acceptance fallback
  when confidence weights are unavailable or uncalibrated for the quant.

Restricting q or changing draft arithmetic can change acceptance without
changing the target distribution when verification is exact. It can still
regress speed or seeded behavior. The Spark recipe's NVFP4 Markov result is
hardware-specific and lossy; do not requantize Gufo's Markov tables by default.

### 8. Tokenization has straightforward local precedents

**Code:** `runtime/tokenizer.cpp:bpe_rank` and `bpe_emit_piece` repeatedly
allocate/concatenate strings, scan all adjacent pairs for the best merge,
and shift the remaining symbols. Long pieces incur quadratic merge work.

Qwen's `tokenizer.cpp:BpeMergeChunk` uses token-ID pairs, precomputed merged
IDs, a linked symbol array and a heap with stale-entry rejection. It also
memoizes repeated pieces within a call. Gemma's `tokenizer.cpp` adds bounded
cross-request caching; its recorded long cached-turn gains concern host
latency, not model tokens/s.

Port those mechanisms while retaining DS41's exact piece boundaries, byte
mapping, special-token handling and leftmost rank tie rule. Cache complete
pieces with a byte budget; do not inherit Gemma's long-line threshold without
measuring DS41's different pretokenizer. Do not copy Qwen's NFC normalization
or framing. Compare cold and repeated 2K/32K/128K conversations, code,
multilingual text, combining characters, CRLF, long identifiers and specials
against the pinned official tokenizer. A cached model prefix still pays for
tokenizing the newly rendered conversation.

### 9. Keep logits on device when the request permits it

**Code:** `v41_decode.cpp:eval_rows` always copies requested full-vocabulary
logit rows to the host; `v41_session.cpp:draw` also scans argmax on the CPU.

For unconstrained greedy requests reduce argmax on device and transfer IDs.
For eligible sampled requests use exact candidate distributions with full
normalization information and a completeness/fallback test. Qwen's
`hip/ops/token.hpp` and Gemma's `Executor::ReadArgmax` / `ReadCandidates`
show both patterns. Grammar masks, logprobs and penalties may require extra
logits or a full fallback. Never substitute an arbitrary top-k target for p.

Preserve lazy full-logit access for probes, public APIs and snapshots, or
explicitly retain enough hidden state to reconstruct it. GPU reduction
alone does not remove the dominant vocabulary matrix read; its benefit is
host scan, transfer and synchronization cost. Embeddings are deliberately
host-resident to save 0.42 GiB: prefer batching/asynchronous staging and a
device one-hot/replication initializer before restoring the whole embedding
allocation merely to save the recorded ~90 microseconds of host setup.

### 10. Retain focused kernel work, with the rejected variants visible

The following remain valid experiments, not confirmed speedups:

| Area | Next experiment | Existing constraint / rejected work |
| --- | --- | --- |
| IQ1/IQ2 decode GEMV | Cooperative wide packed-weight loads with existing lane mapping, bounded LDS staging | Texture-address pressure is measured; one-row-per-lane staging spilled and lost badly |
| Grouped verification experts | Reuse decoded fragments across genuinely shared routes; specialize ragged group sizes | Prefetch, lower registers, two rows/lane and runtime column loops already lost; distinct experts impose a bandwidth floor |
| Routed prefill | Bucket-specific slot widths, lower weight-decode repetition, possibly an IQ2_S FP16 path | Large tiles waste slots or exceed register/LDS budgets; IQ2_S contributes a small old-profile fraction |
| Dense prefill | Explain activation/barrier stalls with counters before redesign | FP16 activation buffers and double-buffered LDS stages were slower; dense AR is already near bandwidth |
| Small follow-up turns | Dispatch tuned per projection and purpose at 3–20 rows | Global row thresholds create non-monotonic costs; WMMA changes verification arithmetic |
| Attention / indexer | Tune selected head pairing, tile size and bounded prefetch after compact rescoring | Existing code-size, spill and rounding failures rule out indiscriminate fusion |

Measure DRAM-cold production allocation patterns, VGPRs, LDS, private scratch,
instruction-cache size and full-request time. Spark's tile-K=16 result and
Kyojin's gfx1151 trellis/SAD work illustrate shape-specific tuning, but EXL3
trellis decoding is not IQ1 codebook decoding. DeepInfer ROCm PR #201 targets
gfx950 matrix instructions and much larger LDS, not RDNA3.5 WMMA.

### 11. Improve cached turns before changing the cache's numerical contract

Gemma's generation-prompt lookahead joins the 3–7 generation-prompt tokens
to the final prefill pass, preserving the separately captured stable
boundary. DS41's engine currently exposes no analogous lookahead hook.
Profile a real agent turn to determine whether a separate tiny suffix pass
is significant, then adapt the approach with DS41 ring/frontier rollback and
bounded replay. Merely appending tokens past the snapshot boundary is unsafe.

Gufo already has shared append components and small private frontiers;
reimplementing Gemma's old whole-snapshot optimization is unnecessary.
Further optional work is separating short-lived SWA frontiers from durable
global state and using encoder bounded replay on a frontier miss. This is
approximate, changes cache-hit-position numerics, and needs a new explicit
quality/cache contract. It is lower priority while exact frontier captures
are already cheap. Do not present global-only restore as an exact replacement.

### 12. Engram and streaming preparation: keep resident evidence in scope

Engram waits in the existing trace are about 0.04 ms/prefill batch and
0.33 ms/decode token, overlapped with GPU work. Sorted reads, deduplication,
row caches and O_DIRECT remain secondary for the resident path. If small
read/thread setup becomes visible, a persistent bounded worker pool is worth
testing; `eval_rows` currently starts async work and `engram_read` creates
read workers per evaluation.

[Kyojin's DS41 notes](https://github.com/Yamz-Labs/kyojin/blob/a22e1e398adfc2a2bb4530264eb2313d0ab35f10/doc/DS41.md)
are more useful for M3: scan-resistant admission, decayed use, prompt-tail
promotion, coalesced reads and large logical expert batches with blocked
activation buffers. Its 172 GiB EXL3 target streams experts and reports much
lower real decode; this is not a regression comparison against resident IQ1_M.
Keep future logical I/O batches separate from GPU compute tile size so
streaming does not require huge activation allocations. Re-profile Engram
under the actual shared-SSD workload before changing its reader.

## Other source conclusions and documentation gaps

- **ds4 #1036:** useful gfx1151 indexer, guarded Q8 WMMA, small HC/router and
  qualification references. Its Q2/192 GiB or two-machine numbers do not rank
  against IQ1_M on 128 GiB. Some prefill attention work explicitly derives
  from Gufo, so it is not independent evidence of a missing implementation.
- **Spark recipe:** interesting verification economics, unified mixed-format
  dispatch, TP1 shape tuning and graph coverage. Cold-expert ATS aliasing is
  an NVIDIA placement mechanism, not an immediate resident gfx1151 feature.
  Its reported rates include DSpark and depend strongly on acceptance.
- **Maya / Xolotl:** useful draft confidence, exact sampling, selected logit
  rows, batch and overlap ideas. Their discrete-GPU expert transfers, CPU
  fallback, routing biases and quant choices are different experiments.
  Do not import routing changes or publish their token rates as kernel gains.
- **deepseek-recipe:** primarily request/template/tokenizer semantics rather
  than a kernel source. The pinned
  [request options](https://github.com/deepseek-ai/deepseek-recipe/blob/935817c66f5d6e64838f419692545f06787e485f/deepseek-recipe/src/request/options.rs)
  resolved V4.1 top-p differently from Gufo's shared DeepSeek 0.95 default at
  the reviewed revision:
  non-thinking resolves to 1.0; thinking uses the requested value (default
  1.0) with a 0.95 floor. Subsequently fixed at the user's request: V4.1
  defaults to temperature 1.0 and top-p 1.0 in both modes, preserving explicit
  overrides. Changing a sampling default is a behavior change, not an
  optimization. Pin explicit
  sampling values in comparisons; do not silently alter user settings.
- **Backlog/card maintenance:** the backlog's prefill breakdown and 4096-row
  tradeoff predate bounded replay, and ROADMAP performance expectations
  predate the current ~24 tok/s result. Refresh them only with matched new
  evidence. Distinguish active stochastic support from inherited bridge code.

## Proposed execution and qualification

1. Retain the current production binary, revision and toolchain. Capture
   unprofiled pp2048/tg128 and separate stage profiles at d2K/d32K. Add d128K
   for compact rescoring, then d1M only if needed to qualify scaling. Record
   layer, row count, dispatch, source/reindex work, routes, draft length,
   actual accepted prefix, output tokens and wall time. Separate model load,
   vision encoding, Engram waits, prefill and decode.
2. Implement compact rescoring first for long-context users, or stochastic
   p/q first for sampled agent use. Keep them separate changes. Establish
   decode-equivalent verification before widening speculation or batching
   cohorts across dispatch boundaries. Add tokenizer work independently.
3. Use the affected `ds41` operator tests and independent DS41 oracle with
   matched token histories. For index work check exact selections, ties,
   causal partial blocks and >16K candidate activation. For numerical changes
   retain the existing 3K/18.5K quality checks and first-divergence analysis.
   A small perplexity delta alone does not qualify speculative exactness.
4. For speculation cover greedy, p/q acceptance and residual draws, bounded-q
   support, grammar/penalties, multiple seeds, EOS, every commit prefix,
   cancellation, image positions, continuation and snapshot/disk restores.
   Include counter assertions showing the new callbacks/kernels actually ran.
5. Run only affected DS41 functional suites: chat/tools for sampled changes,
   long-context/cache for state or selection changes, affected image cases
   for batching/continuation, and replay the reported coding workflow with
   `tests/functional/pi_agent.py`. Retain actual tool results and sessions.
6. Compare C1 and affected C2/C4 cohorts, including shared and disjoint expert
   routes, short follow-ups, ordinary prose and near-perfect lookup. Preserve
   every per-request/per-phase timing at the repository's 5% and 3 ms gates;
   investigate flags with alternating matched controls. Report memory with
   target alone, drafter and vision. Re-run the standard speed benchmark;
   qualify with the format check before committing any C++ changes.

No model execution, new kernel benchmark or functional suite was run for
this review. Only this review and its backlog link were added. The next
implementation should turn one proposal into measured evidence rather than
combining the entire list into an untraceable optimization branch.
