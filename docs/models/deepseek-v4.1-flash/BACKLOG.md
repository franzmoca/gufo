# DeepSeek V4.1 Flash performance backlog

Open performance work for the DeepSeek V4.1 Flash engine (IQ1_M resident),
as of 2026-10-25 on branch `feat/ds41`. Measured results, including every
rejected attempt, are in [EXPERIMENTS.md](EXPERIMENTS.md); read the matching
rows before retrying anything listed here. Impact figures are estimates on
decode at 2K context unless stated.

## Where the time goes

Decode, one token at 2K context: 46.5 ms wall, ~42 ms of GPU time and
~4.6 ms of host time between launches (~1,660 dispatches per token). The
routed expert matvecs take ~10 ms of the GPU time at 160-175 GB/s (IQ1) and
125-145 GB/s (IQ2), against 230-238 GB/s for a plain read of the same six
DRAM-cold experts. The dense Q5_K/Q8_0 projections run at 207-237 GB/s.

Prefill, 2048-token batches: ~350 tok/s at depth 0, 268 tok/s at 32K. The
IQ1_S routed-expert MMQ is the largest prefill cost.

## Open items

| # | Item | Area | Effort | Est. impact | Confidence |
| ---: | --- | --- | --- | --- | --- |
| 1 | Routed-expert GEMV with wide-load weight tiles | Decode | Hard | Up to ~3 ms/token (~6%) if the experts reach ~210 GB/s | Low |
| 2 | Host overhead between decode launches | Decode | Medium | Up to ~4.6 ms/token | Medium |
| 3 | IQ1_S routed-expert MMQ | Prefill | Hard | Unmeasured | Low |

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

### 2. Host overhead between decode launches

~1,660 dispatches per token; ~4.6 ms per token is not covered by GPU work.
In progress. Done: the MMQ scratch pool reuses buffers instead of
allocating and freeing per call; Engram rows upload asynchronously from
pinned memory; hc collapse + RMSNorm, rope + FP8 quantization, (below 16
tokens) hc mixes + Sinkhorn, (single tokens) gate + up + SwiGLU and q_a + KV
projections, the q_a/KV norms and the routed sum + hc expand run as single
kernels; the top-k threshold and decode indexer scores have their
own faster kernels. Remaining, from a `rocprofv3` trace of 2K decode:

- ~1,550 dependent-launch gaps of ~2.5 µs per token (~4 ms). Fusion
  candidates by count per token: rope_tail 53 (40 on q after q_b),
  attention_combine 40 followed by the output rope + FP8, route 40, the
  window copyBuffer 43 (it cannot run before attention, which still reads
  the slot it overwrites).
- hc mixes read 2 MB of F32 weights in ~14 µs (80 per token) regardless of
  how many blocks share a row.
- ~90 µs of host work at the start of each step (embedding dequant and two
  synchronous uploads take ~70 µs); a device-side embedding lookup would
  remove most of it.

### 3. IQ1_S routed-expert MMQ

The vendored MMQ tiles with routed launch bounds from the real expert
buckets. Not profiled in detail since the bounds change.
