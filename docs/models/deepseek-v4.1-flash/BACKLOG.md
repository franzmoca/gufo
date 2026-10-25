# DeepSeek V4.1 Flash performance backlog

Open performance work for the DeepSeek V4.1 Flash engine (IQ1_M resident),
as of 2026-10-25 on branch `feat/ds41`. Measured results, including every
rejected attempt, are in [EXPERIMENTS.md](EXPERIMENTS.md); read the matching
rows before retrying anything listed here. Impact figures are estimates on
decode at 2K context unless stated.

## Where the time goes

Decode, one token at 2K context: 42.4 ms wall, ~39.9 ms of GPU time and
~2.7 ms of gaps between ~1,000 dependent launches. The dense Q5_K/Q8_0
projections and the output head take ~23.9 ms for ~4.8 GB, near the
measured 235 GB/s DRAM read peak; the routed expert matvecs take ~10 ms at
160-175 GB/s (IQ1) and 125-145 GB/s (IQ2), against 230-238 GB/s for a plain
read of the same six DRAM-cold experts.

Prefill, 2048-token batches: 703 tok/s at depth 0, 585 tok/s at 32K. A
`rocprofv3` trace of a 2K batch (3.30 s): routed experts on the FP16 matrix
cores 1.38 s (22-26 TOPS on issued slots; IQ2_S stays on MMQ, 57 ms), dense
Q5_K projections 0.88 s (~33 TOPS against 55 measured FP16 WMMA),
attention 0.36 s, mHC state passes and the routed sum 0.26 s, BF16 matmuls
95 ms, Q8_0 MMQ 59 ms.

## Open items

| # | Item | Area | Effort | Est. impact | Confidence |
| ---: | --- | --- | --- | --- | --- |
| 1 | Routed-expert GEMV with wide-load weight tiles | Decode | Hard | Up to ~3 ms/token (~6%) if the experts reach ~210 GB/s | Low |
| 2 | Launch gaps between decode kernels | Decode | Medium | Up to ~2.7 ms/token | Medium |
| 3 | Routed-expert FP16 kernel: weight decode per slot | Prefill | Hard | ~5-20% of the routed time | Low |
| 4 | Dense FP16 kernel barriers | Prefill | Hard | Up to ~0.15 s per 2K batch | Low |
| 5 | Remaining prefill glue | Prefill | Medium | ~30-60 ms per 2K batch | Medium |
| 7 | Small-batch evaluation cost | Prefill | Medium | ~15 ms per short follow-up turn | Low |

### 1. Routed-expert GEMV with wide-load weight tiles

`kernels/rocm/ds41_expert_gemv.hip.cpp`. The retained kernel and mmvq are
both bound by the texture addresser (`TA_BUSY` ~92% of active cycles,
`rocprofv3 --pmc TA_BUSY_avr`), not by DRAM or ALU: per 32-value sub-block a
lane issues four 2-byte weight loads and eight float4 activation loads for
about 6 bytes of weights. The design that addresses it stages contiguous
weight tiles into LDS with wide (dword or wider) loads and decodes from
LDS, so the texture path sees a few wide loads per row instead of many
narrow ones.

The first attempt (one row per lane, 32-row tiles of G blocks staged by
dword loads, activations as wave-uniform LDS broadcasts, register double
buffering) measured 126-245 µs against 79 µs for IQ1_S gate/up: the eight
unrolled sub-blocks and the prefetch registers pushed it to 185-256 VGPRs
with scratch spills, and the version with scalar activation loads exhausted
the SGPRs. Starting points for a retry:

- Keep the retained kernel's lane mapping (8 lanes per row, one sub-block
  each) and replace only its weight loads with a cooperative wide-load
  stage into LDS; activations stay in global memory, which measured faster
  than LDS staging.
- Size tiles so double buffering fits without spills; check
  `.vgpr_count` and `.private_segment_fixed_size` in the assembly
  (`tools/prof/isa_mix.py`) before timing.
- Measure with `ds41_gemv_bench` (DRAM-cold six-expert launches) against
  the ceiling probe of a plain read of the same pattern (230-238 GB/s).

### 2. Launch gaps between decode kernels

~1,000 dependent launches per token leave ~2.7 ms of the GPU idle (~2.5 µs
each; the host stays ahead of the queue). Done: the MMQ scratch pool reuses
buffers instead of allocating and freeing per call; Engram rows upload
asynchronously from pinned memory; hc collapse + RMSNorm, rope + FP8
quantization, (below 16 tokens) hc mixes + Sinkhorn, (single tokens) gate +
up + SwiGLU and q_a + KV projections, the q_a/KV norms and the routed sum +
hc expand run as single kernels; the top-k threshold and decode indexer
scores have their own faster kernels. Remaining, from a `rocprofv3` trace
of 2K decode:

- Launches left to fuse, per token: the q rope after q_b (40, would need a
  GEMV epilogue pairing rows 8 lanes apart), route after the router matvec
  (40; 7 µs each, the block size does not help), the window copy (43; it
  cannot run before attention, which still reads the slot it overwrites).
  Fusing the output rope + FP8 into the attention merge measured no gain.
- hc mixes read 2 MB of F32 weights in ~14 µs (80 per token) regardless of
  how many blocks share a row.
- ~90 µs of host work at the start of each step (embedding dequant and two
  synchronous uploads take ~70 µs); a device-side embedding lookup would
  remove most of it.

### 3. Routed-expert FP16 kernel: weight decode per slot

`kernels/rocm/ds41_routed_f16.hip.cpp`. Without the weight decode (timing
only) IQ1_S gate/up drops 22.6 → 17.0 ms per layer and IQ2_XXS 26.0 →
17.5 ms; the activation and weight loads cost ~2 ms each, the codebook
lookup ~1 ms. A decoded fragment feeds at most four 16-slot tiles, and on
average 2.3 since most experts get under 64 slots. 128-slot tiles for the
large experts would cut the decode work by 13-24% (recorded counts) but
need 128 accumulator VGPRs and twice the activation LDS. 4096-token
batches (`--prefill-batch 4096`) give +21% routed throughput, +3.7%
end to end, for ~1.2 GiB more activations.

### 4. Dense FP16 kernel barriers

`kernels/rocm/ds41_dense_f16.hip.cpp`. Without its two barriers per stage
(timing only) the output LoRA runs 5.55 → 4.20 ms and o_b 5.14 → 4.83 ms;
double-buffered stages with one barrier were slower (EXPERIMENTS.md), so
the cost is not understood yet.

### 5. Remaining prefill glue

The FFN-end routed sum and expand (`hc_expand_sum`, 115 ms per 2K batch)
could feed the next layer's state pass directly except before the Engram
layers (~30 ms); Q8_0 projections still go through MMQ and its Q8_1 pass
(77 ms); the shared-expert SwiGLU (9 ms) could use the dense up pass's
store.

### 6. Engram reads (not on the critical path)

Rows are read on up to 32 `pread` threads, started when the batch starts.
Measured with `gufo bench -p 2048 -n 64`:

- Prefill: a 2,048-token batch reads layer 1's rows in 12 ms and layer 14's
  in 21 ms. The host reaches those layers after 49 ms and 971 ms and waits
  0.04 ms.
- Decode: each layer's reads take 0.31 ms (p95 0.6 ms, rare 10 ms outliers).
  The host waits 0.33 ms per token on average at layer 1, while block 0
  still runs on the GPU.

Sorted, deduplicated reads, `O_DIRECT` and a pinned row cache stay unbuilt
until a measurement shows reads stalling the GPU, for example under SSD
contention once M3 streams experts.

### 7. Small-batch evaluation cost

Short follow-up turns evaluate 3-20 tokens per batch, and the cost is not
monotonic in the batch size: 7 tokens from position 8,469 take 136 ms and
9 tokens from 8,467 take 120 ms (both on a live slot); at position 0,
10 tokens take 160 ms and 12 take 143. Find which kernels switch paths
between these sizes (`tools/prof/prof.py` on the two shapes).

### 8. DSpark verification cost

A DSpark cycle at 1.9K context costs 12.5 ms of drafting plus the
verification batch: one-draft cycles take 69.6 ms, five-draft cycles
117 ms, against 42.4 ms for a plain step, so each row beyond the second adds
about 12 ms. Batches of 2-8 tokens use the decode kernels for identical
numerics; a multi-column GEMV that reads each weight row once for all
columns while keeping the per-column sum order would cut the extra rows'
cost and let the width policy draft longer blocks.

## Memory

Resident weights take 112.27 GiB GTT. A session at the native 1M context
adds 0.9 GiB of FP4 state; the batch scratch (1.4 GiB at 1M, of it 256 MiB
of indexer scores) is shared by all sessions, and a 2048-token prefill peaks
0.3 GiB higher. Further reductions would need smaller prefill batches, which
cost speed (see EXPERIMENTS.md).
