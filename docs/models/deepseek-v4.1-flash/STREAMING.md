# DeepSeek V4.1 Flash SSD streaming research

This note records the implementations inspected and the measurements on the
local Strix Halo machine. Their published rates are workload references, not
matched comparisons: quantization, hardware, context, prompt history and
speculative decoding differ.

## Implementations inspected

- [ds4](https://github.com/antirez/ds4) and its [Strix Halo PR](https://github.com/antirez/ds4/pull/1036),
  inspected at [`815e8fb`](https://github.com/kyuz0/ds4/tree/815e8fb6194ffa3160f1bd5f80df0e55efd190fe):
  the ROCm path uses aligned direct reads into staging buffers, falls back to
  buffered reads and can stage whole subsequent layers during prefill. The PR
  reports 10.18 tok/s after 2K prefill with a 76 GiB cache for its Q2 build;
  it does not support DSpark with SSD streaming.
- [Kyojin](https://github.com/Yamz-Labs/kyojin/blob/0c5b91a0686acc8129190f753e032715f0b0dcc2/doc/DS41.md)
  streams a larger 172 GiB EXL3 pack on 128 GB Strix Halo. It uses direct I/O,
  12 readers and 4 MiB chunks; prefill issues later-layer reads during current
  work. Its cache protects frequently used and recent-prompt experts, and its
  row-blocked 32K prefill reduces repeated scans. It reports roughly 5–11
  tok/s, with substantial workload and machine-sharing variation.
- The [Spark recipe](https://github.com/vcruz305/DSV4.1-Flash-1017-EXL3-Spark-recipe/blob/63220db0a7cc62dd8b783f71c9cf28f909c9b09b/README.md)
  uses a smaller 1.62 bpw EXL3 quant on NVIDIA GB10. Its 31–34 tok/s result
  uses DSpark; the cold-expert file is read through an NVIDIA ATS mapping,
  while hot experts stay resident. Its no-draft control is 13.4 tok/s. This
  is a different memory and kernel path from this AMD GGUF stream.
- [Project Maya](https://github.com/mw00/project-maya/tree/8b4b39f84e35393833b31b80e82bee92e69bb646)
  tiers experts across GPU, RAM and SSD, and uses threaded direct reads and
  CPU expert execution where appropriate. Its multi-GPU/CPU measurements
  cannot be attributed to a single Strix Halo.
- [Xolotl](https://github.com/Indras-Mirror/Xolotl/tree/961d390fea6d848431822f7c6e94a6e5c0775133)
  combines VRAM, pinned host RAM and NVMe, with ranking and layer read-ahead.
  Its reported 4090 results use different hardware and its own quant; the
  project also reports loading the ISTA GGUFs. Predicted read-ahead may waste
  bandwidth, so it is not assumed to help this path.

The prefill read-ahead and larger, row-blocked prefill are promising follow-up
experiments. They need independent memory, output and end-to-end measurements;
the current store reads selected experts after the router has chosen them.

## Local storage measurements

October 26, 2026: Samsung 990 PRO, XFS mounted `noatime`, kernel 7.2.11,
NVMe scheduler `none`, 4096 KiB kernel read-ahead, 1023 requests. No global
kernel or filesystem setting was changed. `tools/ds41/stream_io_bench.py`
reads actual IQ2_XXS expert offsets in 40 layer-sized batches of 1, 2 or 6
missing experts. Each arm reads 1.02 GB of payload, drops only clean pages of
the model shards first, and compares aligned direct-read payloads with buffered
reads. Two rounds are run in opposite orders. Device counters are whole-drive
counters and may include other activity.

| Read mode | Readers | Payload GiB/s, two runs | Median layer ms |
| --- | ---: | ---: | ---: |
| buffered | 1 | 3.04, 3.11 | 5.22, 5.14 |
| buffered | 4 | 5.33, 5.31 | 3.05, 3.08 |
| buffered, `POSIX_FADV_RANDOM` | 4 | 5.27, 5.41 | 2.97, 3.02 |
| direct | 4 | 5.68, 5.89 | 2.65, 2.66 |
| direct | 8 | 5.81, 5.75 | 2.74, 2.73 |
| direct | 12 | 5.63, 5.68 | 2.69, 2.73 |

On this workload four readers reached the useful queue depth; more did not
improve layer latency. Random advice was inconclusive. The matched cold
production run using four direct readers improved prefill from 172.6–172.9 to
213.8–214.1 tok/s and decode from 6.80–7.16 to 8.97–8.98 tok/s. Each arm
used pp2048/tg128, identical arguments and targeted page-cache eviction. The
whole-model gain includes page-cache and staging effects, not just raw NVMe
throughput. In a separate identical repeated-prompt history, the second prefill
was 938 tok/s with either mode and the second decode was 13.44 tok/s with
either mode; the first decode was 7.05 buffered and 8.99 direct.

As a separate sustained raw-read control, Kyojin's
[`uring_bench.c`](https://github.com/Yamz-Labs/kyojin/blob/0c5b91a0686acc8129190f753e032715f0b0dcc2/tools/tier/uring_bench.c)
read 3 MiB random direct-I/O requests at queue depth four for two seconds over
a 16 GiB span of the same shard. Pread threads, `io_uring` and polling
`io_uring` each reached about 4.9 GB/s; p99 was about 3 ms. The short burst
test above and this sustained test are different workloads. Replacing the
reader pool with `io_uring` has no measured benefit here. Changing the global
scheduler or read-ahead was not tested: those sysfs settings are not writable
by this user.

A separate `rocprofv3` pp140/tg16 trace with buffered reads showed GPU kernel
dispatches occupying 1.36 s of the 9.76 s from the first router to the last
kernel; this excludes model loading and does not count dedicated DMA activity.
The largest gaps surround streamed expert staging. This is why raw matrix
kernel throughput alone does not predict streamed decode speed.

## Overlapped staging (2026-10-27)

Three changes removed the host-side serialization around streamed experts;
none of them touches the arithmetic, and the tg128 token hash of the matched
bench below is identical to the previous build's:

- **Per-expert weight tables.** Each projection now consumes a device table
  that maps an expert index to its weights (its cache slot when resident, the
  stage row a read just filled), so resident experts are read in place. The
  previous design copied every routed resident expert slot-to-stage before
  each launch, up to ~57 MB per layer of pure device-to-device traffic per
  decode token. `ds41_gemv_bench` shows the table indirection at parity with
  the packed layout (within noise, bit-identical outputs).
- **Split hit/miss launches.** While the reads for the missing experts
  stream, the shared expert and the gate+up of the resident experts compute.
  The miss rows are zeroed by the overlapped launch and completed by a second
  launch over the read rows once the reads join. Kyojin lists this as a lever
  but skips it because separate launches change the fp32 sum order; here each
  row keeps its single-launch arithmetic, so the outputs are bit-identical.
- **Per-reader upload streams.** The four read workers upload through their
  own streams and events instead of blocking copies on the legacy stream, so
  staging no longer queues behind the layers' kernels.

Matched `gufo bench` pp2048/tg128 on the same model, histogram sidecar and
machine as the October 26 runs (gpu-test binary, seed 19 s, 82.8% route hits):

| Build | Prefill tok/s | Decode tok/s | DSpark decode tok/s |
| --- | ---: | ---: | ---: |
| October 26 (stage copies, shared overlap) | 592.1 | 11.65 | 13.50 |
| October 27 (tables, split overlap, upload streams) | 664.7–669.6 | 14.99–15.08 | 17.92–17.96 |
| October 27 (retuned expert GEMV) | 521.6–529.6* | 16.57–16.94 | 19.34 |

*Prefill keeps its kernels; the page-cache spread spans 494–669 tok/s across
these runs. The retuned expert GEMV (see EXPERIMENTS.md) leaves every token
hash unchanged: 04842be7… on AR and DSpark (71 accepted in both, 110 vs 107
drafted under the timing-adaptive drafter), resident `IQ1_M` at 70d5a48a…
(23.86–24.38 vs 23.36–23.55 tok/s).

The DSpark run drafted and accepted the same 110/71 tokens with the same
output hash. Decode host wait fell from 6.51 to 5.25 ms/token; the remaining
stall is the per-layer read latency itself, which the next two backlog items
address from different sides. Resident `IQ1_M` pp2048/tg64 (1070/23.0) stayed
inside the previous build's own run-to-run spread (945–1129 / 22.5–23.7).

[Raw results and logs](artifacts/m3-streaming/io-bench.json) ·
[Profile](artifacts/m3-streaming/profile-iq2-stream.log) ·
[io_uring control](artifacts/m3-streaming/io-uring-bench.log) ·
[Direct-I/O candidate](artifacts/m3-streaming/bench-iq2-direct-io-candidate1.log) ·
[Buffered control](artifacts/m3-streaming/bench-iq2-buffered-io-control1.log) ·
[Overlapped staging](artifacts/m3-streaming/bench-iq2-tables-ar1.log) ·
[Retuned GEMV pairs](artifacts/m3-streaming/bench-iq2-fast-ctrl2.log)

## Prefill read pipelining with router predictions (2026-10-27)

Backlog item 11. A prefill layer's missing experts are no longer read after
its router has chosen them: the reads start during the previous layer's
compute.

- **Background read jobs.** The four read workers also serve one background
  job between demand phases, and a demand phase always preempts it. The
  next prepare of the job's layer adopts its un-run tasks — grouped per
  expert, and only for the experts the batch actually routes, so a wrong
  prediction costs bandwidth but never phase time. A prepare for the layer
  after parks the job across its phase; anything overtaken is dropped once
  its in-flight tasks drain.
- **Ping-pong stage sets.** A background job writes the stage set of its
  layer's parity, disjoint from the rows the current layer's compute reads.
  The second set (~3.6 GiB) is bought lazily after the startup seed, so
  seeding keeps its memory budget, but the runtime cache then grows to 98.0
  instead of 101.6 GiB and decode pays for the difference in extra reads;
  a compact stage set sized to the predicted expert count is the follow-up
  that recovers it.
- **Router prediction.** The read overlap of a prefill layer L enqueues
  layer L+1's router matmul on this layer's FFN input (one layer stale),
  where the extra matvec hides in the read stall. A store thread submits
  the read job when the ids land in pinned memory, so layer L+1's reads
  overlap layer L's compute.

A router-free sweep of the next layer's uncached experts — the original
item-11 design, as ds4's Strix Halo path does — was measured first and
rejected on this model: V4.1 Flash routing is skewed enough that a
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
the second stage set is the price of the pipelining, and the compact-set
follow-up targets exactly that. Decode-side predictions are the next
section.

[Router-free sweep, candidate](artifacts/m3-streaming/bench-iq2-routersweep-cand1.log) ·
[Router-free sweep, control](artifacts/m3-streaming/bench-iq2-routersweep-ctrl1.log) ·
[Prefill prediction, AR](artifacts/m3-streaming/bench-iq2-prefetch-ar-cand1.log) ·
[Prefill prediction, AR control](artifacts/m3-streaming/bench-iq2-prefetch-ar-ctrl1.log) ·
[Prefill prediction, DSpark](artifacts/m3-streaming/bench-iq2-prefetch-ds-cand1.log) ·
[Prefill prediction, DSpark control](artifacts/m3-streaming/bench-iq2-prefetch-ds-ctrl1.log)

## Streamed decode prefetch (2026-10-27)

Backlog item 12. Single-token steps use the same pipeline: when a layer's
read overlap actually waited on reads, it enqueues the next layer's router
on the current FFN input, and the store thread submits the predicted
experts' reads, which the next layer's prepare adopts for the experts it
routes. Layers without misses predict nothing — a prediction without a
read stall to hide its matvec costs more than the read it saves — and
adoption skips unrouted experts, so wrong predictions cost bandwidth only.

Matched arms against the item-11 build (same protocol, two rounds; the
token hash 04842be7… and DSpark's 71/107 are identical on every arm):

| Build | AR pp2048 | AR tg128 | DSpark pp2048 | DSpark tg128 |
| --- | ---: | ---: | ---: | ---: |
| prefill prediction only | 789.1–792.4 | 15.78–15.79 | 746.7–748.2 | 17.83–17.86 |
| with decode prediction | 778.7–788.0 | 15.91–15.92 | 748.4–750.2 | 17.84–17.89 |

Decode gains 0.8% on AR (every round separated well outside the ±0.01
run-to-run spread) and is neutral on DSpark, whose verification batches of
2–16 rows predict nothing by design. The gain is small because at 82–83%
route hits only ~5 ms/token of read stall exists and one-layer-stale recall
covers part of it; the larger decode lever is the cache capacity the second
stage set consumes, which item 11a targets. Decode reads rise from 11.99 to
13.2 MiB/token (10.9 vs 4.1 GiB prefetched overall), the price of the
predictions that miss.

[Decode prediction, AR](artifacts/m3-streaming/bench-iq2-decode-ar-cand1.log) ·
[Decode prediction, AR control](artifacts/m3-streaming/bench-iq2-decode-ar-ctrl1.log) ·
[Decode prediction, DSpark](artifacts/m3-streaming/bench-iq2-decode-ds-cand1.log) ·
[Decode prediction, DSpark control](artifacts/m3-streaming/bench-iq2-decode-ds-ctrl1.log)

## Compact prefetch stage and cache reserve (2026-10-27)

Backlog item 11a. The item-11 decode dip came from the second stage set
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

Matched arms against the item-12 build (same protocol). The cache reaches
103.8 GiB (AR) / 99.1 GiB (DSpark, which loads the drafter); hashes and
DSpark's 71/107 identical on every arm:

| Build | AR pp2048 | AR tg128 | DSpark pp2048 | DSpark tg128 |
| --- | ---: | ---: | ---: | ---: |
| item 12 (8 GiB reserve, whole-layer set) | 787.5–788.0 | 15.86–15.92 | 744.5–749.3 | 17.87–17.91 |
| 11a (4 GiB reserve, compact set) | 739.9–747.9 | 17.34–17.38 | 763.9–765.1 | 19.03–19.07 |
| original control (028086ab) | 652.6–663.6 | 16.53–16.59 | 616.7–620.5 | 19.05–19.11 |

Decode recovers past the original control (+4.8% AR, parity on DSpark, from
4–6% below it), and DSpark prefill improves. AR prefill sits 5% under the
item-12 build while DSpark prefill gains: across the pipelined builds the
AR prefill rate anti-correlates with the route-hit rate (88.3% hits → 787,
95.3% → 742 tok/s), so the residual is a residency effect worth a profile,
not a pipeline cost — reads fall from 13.2 to 9.3 MiB/token on the same
arms.

[Compact stage, AR](artifacts/m3-streaming/bench-iq2-compact-ar-cand1.log) ·
[Compact stage, DSpark](artifacts/m3-streaming/bench-iq2-compact-ds-cand1.log)

## Storage read-path profile (2026-10-27)

Where every byte of a streamed run comes from, measured on the r522c8653
build with `/proc/<pid>/io` sampling around the bench phases plus targeted
micro-benchmarks (`tools/ds41/engram_row_bench.py`).

| Read path | Pattern | Mechanism | Measured |
| --- | --- | --- | --- |
| Expert weights (demand + prefetch jobs) | per layer, three ~1 GiB blocks (gate, down, up), expert-strided within each | O_DIRECT `pread`, 4 workers, ≤32 MiB chunks | 5.7 GiB/s effective; 8.2 GB/s pure-sequential ceiling on the 990 PRO |
| Engram rows | 144 B at hashed-random offsets, 2 layers × 24 rows per token | buffered `pread`, ≤32 threads, page cache | 278 K rows/s cold (~1.06 GiB/s at 4 KiB granularity) |
| Resident experts | slots in managed chunks, read by the kernels | device memory | not an I/O path |

A pp2048 batch moves ~4 GiB of expert reads as ~140 large IOs and 98,304
Engram rows (2048 tokens × 24 × 2 layers) as 4 KiB-granule IOs — the Engram
is under 10% of the bytes but ~99% of the operations. The profile also
explains the item-11a oddity where a fuller cache slowed AR prefill:
`rchar == read_bytes` for entire runs, i.e. the Engram's page cache scores
zero hits once the expert cache has grown (clean file pages are exactly what
its chunk growth reclaims), so every prefill batch spends ~0.35 s of device
time on random Engram reads that steal queue depth from the expert streams.
The 98 GiB-cache build (fastest prefill) and the DSpark runs (4.7 GiB more
free memory) fit the same mechanism.

Two-disk experiments on the second NVMe (KIOXIA EXCERIA 1.8T, `/`,
2.3 GB/s sequential vs the 990 PRO's 8.2), via symlinks so no code changed:

- **Engram + shards 3-4 on the KIOXIA: pp2048 collapsed to 113-449 tok/s.**
  Shards are layer-major (layers 0-12 in shard 1, …), so moving whole shards
  makes those layers' sweeps read at the slow disk's pace and gate every
  phase that touches them.
- **Engram alone on the KIOXIA: pp2048 724.9-727.3** (vs 739.9-747.9
  same-disk), decode unchanged — the KIOXIA's random-read latency costs more
  than the contention it removes. A split only pays with a second fast disk,
  or after a repack that stripes each layer's experts proportionally across
  both.

Follow-ups are backlog items 14 and 15: Engram batch dedup/sort with a
bounded hot-row cache, and a rank-major expert-contiguous repack whose jobs
become single sequential runs (the 8.2 vs 5.7 GiB/s gap is its headroom).

## Opt-in streaming, IQ1_M and the compact stage stride (2026-10-27)

Streaming is now explicit: `--stream-experts` keeps dense weights resident
and streams routed experts, `--expert-cache-bytes` bounds the expert cache,
and without the flag a model that does not fit available memory (free CMA
excluded) fails to load with what it needs. While streaming, the weight
placement is decided from the resident tensors only, so the 5.31 GiB of
dense weights go through the device heap instead of managed memory.

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

## Slots per expert size and the full built-in ranking (2026-10-27)

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
of backlog item 14, whose page cache the full expert cache displaces, are
the likely remainder.


## Decode profile and per-layer expert tables (2026-10-27)

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
