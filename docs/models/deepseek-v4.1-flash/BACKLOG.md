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

Prefill, 2048-token batches: ~394 tok/s at depth 0, 362 tok/s at 32K. A
`rocprofv3` trace of a 2K prefill (6.0 s GPU time, 94% busy): MMQ 64%
(routed experts 2.72 s at 10-15 TOPS: IQ1_S 1.73 s, IQ2_XXS 0.48 s, IQ1_M
0.45 s; dense Q5_K/Q8_0 1.10 s near 27 TOPS, against 57 TOPS measured int8
WMMA), attention 8.8%, Q8_1 activation quantization 5.0%, BF16 matmuls
4.7%, the MMQ non-finite sanitize pass 2.5%, row gathers 2.1%.

## Open items

| # | Item | Area | Effort | Est. impact | Confidence |
| ---: | --- | --- | --- | --- | --- |
| 1 | Routed-expert GEMV with wide-load weight tiles | Decode | Hard | Up to ~3 ms/token (~6%) if the experts reach ~210 GB/s | Low |
| 2 | Launch gaps between decode kernels | Decode | Medium | Up to ~2.7 ms/token | Medium |
| 3 | Routed-expert MMQ tiles | Prefill | Hard | Up to ~1.4 s per 2K prefill (~20%) at the dense rate | Low |
| 4 | Q8_1 quantization in the producing kernels | Prefill | Medium | ~0.2 s per 2K prefill (~3%) | Low |
| 5 | FP4-packed KV state | Memory | Medium | ~2.2 GiB more free at 1M context | Medium |

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

### 3. Routed-expert MMQ tiles

The vendored MMQ (int8 WMMA) with routed launch bounds from the real expert
buckets. 2048 tokens give each expert ~32 rows, so a 64-row weight tile is
unpacked from its IQ1/IQ2 codebook for only 32 columns: IQ1_S gate/up runs
at 14.6 TOPS (19.9 ms per launch), IQ1_M at 9.8, IQ2_XXS at 12, while the
dense Q5_K tiles reach ~27. Measured since: a 32-column tile is 67% full
at 2048 tokens and its fixed weight-panel unpack costs about 40% of the
tile; neither wider batches (4096 tokens: −10-20% per token) nor a fixed
16/48/64 width help (see EXPERIMENTS.md). Directions: a tile width per
expert within one launch (16 columns for small buckets, 64 for the
largest), or a cheaper IQ1 unpack (int8 codebook rows in LDS); see the Qwen
3.8 Flash Next routed GEMM rows in
`docs/models/qwen3.8-flash-next/EXPERIMENTS.md` for a routed WMMA kernel
reaching 22-32 TFLOPS on this GPU.

### 4. Prefill glue passes

Done: non-finite guard at the MMQ store, routed gate/up as one pair with the
SwiGLU store, the output LoRA groups as MMQ channels (no row copies), and
q_a/KV and shared gate/up quantizing their input once. Left per 2K prefill:
one Q8_1 quantization per remaining projection input (~0.4 ms each); folding
it into the producing norm kernels would remove those passes, and the
shared-expert SwiGLU (9 ms) could use the dense up pass's store.

### 5. FP4-packed KV state

Done: compressed KV and index-K rows are stored as bfloat16 (exact), halving
the state to ~3.1 GiB at 1M and restoring 2048-token prefill batches there.
Packing them as FP4 codes plus block scales would shrink them ~3.5x further
(the attention and indexer kernels would decode on load); not needed for the
batch size now.
