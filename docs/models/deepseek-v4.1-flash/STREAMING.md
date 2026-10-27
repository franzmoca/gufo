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
follow-up targets exactly that. Decode-side predictions measured +0.8%
once adoption skips unrouted experts (15.89–15.94 vs 15.75–15.78 tok/s)
and land as backlog item 12.

[Router-free sweep, candidate](artifacts/m3-streaming/bench-iq2-routersweep-cand1.log) ·
[Router-free sweep, control](artifacts/m3-streaming/bench-iq2-routersweep-ctrl1.log) ·
[Prefill prediction, AR](artifacts/m3-streaming/bench-iq2-prefetch-ar-cand1.log) ·
[Prefill prediction, AR control](artifacts/m3-streaming/bench-iq2-prefetch-ar-ctrl1.log) ·
[Prefill prediction, DSpark](artifacts/m3-streaming/bench-iq2-prefetch-ds-cand1.log) ·
[Prefill prediction, DSpark control](artifacts/m3-streaming/bench-iq2-prefetch-ds-ctrl1.log)
