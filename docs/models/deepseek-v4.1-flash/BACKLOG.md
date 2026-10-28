# DeepSeek V4.1 Flash backlog

Open work only. Everything already measured, retained or rejected, is in
[EXPERIMENTS.md](EXPERIMENTS.md); read the matching entries before retrying
an item. The streaming design is in [STREAMING.md](STREAMING.md). Impact
figures are estimates unless stated.

## Where the time goes

**Streamed `IQ2_XXS` decode** (rocprofv3, warm cache, tg128 after pp512,
medians per token, 2026-10-27): 50.1 ms per token, 40.3 ms of kernels and
9.8 ms of GPU idle. Since then the per-layer tables removed most of the
copies (48.1 ms traced).

| Part | ms/token |
| --- | ---: |
| Dense Q5_K GEMV | 22.2 |
| Routed expert GEMV | 11.5 |
| Other kernels (attention, hyper-connections, router, …) | 6.6 |
| Idle: miss reads between the hit and miss launches | 5.3 |
| Idle: small blocking copies (now 23 per token, from ~200) | ~3 |

The dense GEMV runs near the measured DRAM ceiling, so verification of
several rows per weight read (DSpark) is the main decode lever. A streamed
DSpark tg512 trace splits into routed experts 44% (36% the 8-row
verification groups, ~36 distinct experts per layer at the single-token
per-expert cost), dense GEMV 31%, other kernels 11%, idle 10% and the
drafter 4%.

**Resident `IQ1_M` decode** (2K context, 2026-10-25): 42.4 ms wall, ~2.7 ms of
it gaps between ~1,000 dependent launches. Dense projections and the output
head take ~23.9 ms near the 235 GB/s read peak.

**Prefill** (pp2048 at 32K with bounded replay, 2026-10-26): 1,773.84 ms of
GPU time; routed FP16 606 ms, dense FP16 518, attention 219, mHC 137,
indexer 83, top-k 45. After E19/E20 (2026-10-27) the indexer scores take
about a quarter and the top-k a twentieth of that per layer at 128K; native
pp2048 runs 1,182 tok/s at 32K, 1,155 at 64K and 1,098 at 128K.

**Verification rows** (`--spec-bench`, 1–8 rows, 2026-10-27, ms):

| Model | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `IQ2_XXS` streamed | 43.4 | 58.0 | 69.8 | 78.8 | 96.6 | 107.5 | 120.4 | 133.8 |
| `IQ1_M` resident | 41.1 | 52.8 | 65.5 | 75.7 | 87.0 | 98.3 | 113.5 | 123.6 |

## Open items

| # | Item | Area | Est. impact | Confidence |
| ---: | --- | --- | --- | --- |
| 1 | 5–7 token GEMV instantiations | Verification | ~5 ms per 5-row verification | Medium |
| 2 | Fused SwiGLU and Q8_0 pair launches through the split GEMV | Verification | sh_gate costs 103 µs at 8 tokens against 75 µs for one | Medium |
| 3 | Expert reads for rejected verification rows | Streamed DSpark | Part of the 36% verification-expert time | Low |
| 4 | Launch gaps between decode kernels | Decode | Up to ~2.7 ms/token | Medium |
| 5 | Compact selection and WMMA prefill rescoring | Long context | Prefill and verification beyond 16K | Medium |
| 6 | Routed FP16 kernel: weight decode per slot | Prefill | ~5–20% of routed time | Low |
| 7 | Remaining prefill glue | Prefill | ~30–60 ms per 2K batch | Medium |
| 8 | Small-batch evaluation cost | Short turns | ~15 ms per follow-up turn | Low |
| 9 | Stable-boundary prefill pass per chat request | Serving | ~65 ms per request | Medium |
| 10 | Multi-user DSpark draft sizing | Serving decode | C4–C8 mixed text from 51–56 toward AR's 60–78 tok/s | Medium |
| 11 | Engram row reads under cache pressure | Streamed prefill | ~0.35 s of device reads per pp2048 batch | Medium |
| 12 | Rank-major expert-contiguous repack | Streaming | Reads from 5.7 GiB/s toward 8.2 GB/s | Medium |
| 13 | Route bias toward cached experts (changes outputs) | Streamed decode | Opt-in approximation | Low |
| 14 | Higher-precision Engram tables | Quality | Unmeasured | — |
| 15 | Smaller items | Various | Small | — |

### 1. 5–7 token GEMV instantiations

The dense and routed decode GEMVs instantiate 1, 2, 4, 8 and 16 columns, so
5–7 rows run the 8-column kernels. In the table above, row 5 adds 17.8 ms
streamed and 11.3 ms resident, against a trend of ~13 ms (streamed) per row:
about 4–5 ms of the 5-row step is the column jump. DSpark verifies up to six
rows, so 5- and 6-row steps are common. Instantiate the exact counts (or 6
and 8) for the dense split GEMV and the expert-group kernel and measure code
size and `--spec-bench`; rows must stay bit-identical to single-token decode.

### 2. Fused SwiGLU and Q8_0 pair launches through the split GEMV

The task-major split GEMV covers single Q5_K projections only. The fused
shared gate/up + SwiGLU launch (sh_gate 75 → 103 µs from one to eight
tokens) and the Q8_0 q_a/KV pair still use the lane layout whose
multi-token cost is activation loads. Port them to the split kernel with the
same xor-tree and slice order.

### 3. Expert reads for rejected verification rows

On streamed DSpark every verification row pays for its own experts, and
verification batches predict no next-layer experts. The verified prefix
already weighs the head's per-row acceptance against measured cycle time
([E18](EXPERIMENTS.md#e18-cost-aware-draft-prefix-retained)), which prices
misses only on average. Open: predict the anchor row's next-layer experts,
or charge a row the miss bytes of its own routes. Route bias is a separate,
output-changing item (13).

### 4. Launch gaps between decode kernels

~1,000 dependent launches per token leave ~2.7 ms idle (resident profile).
Left to fuse: the q rope after q_b (40 per token; would need a GEMV epilogue
pairing rows 8 lanes apart), route after the router matvec (40; 7 µs each),
the window copy (43; cannot run before attention, which reads the slot it
overwrites). hc mixes read 2 MB of F32 weights in ~14 µs (80 per token)
regardless of blocks per row. ~90 µs of host work starts each step
(embedding dequant and two synchronous uploads ~70 µs); a device-side
embedding lookup removes most of it. Graph capture would need the mixed-MMQ
layers' route plans on the device (E7 covers the FP16 layers) and the Engram
reads as capture boundaries.

### 5. Compact selection and WMMA prefill rescoring

E1 restricts FP32 rescoring of 1–4 rows to layer 20's candidate blocks
after 16K. Prefill rescoring (WMMA) still scores the full history and masks
afterwards, and the final top-k still scans dense scores (both about 4×
cheaper per row after E19/E20; the encoder index layers 2, 8 and 14 have no
candidate mask and remain the main depth cost, ~40 ms each per 2K batch at
128K). Keep absolute
position tie order, the pinned newest block, causal partial blocks and the
emitted row order; at or below 16K keep the dense route unless compact
dispatch measures faster.

### 6. Routed FP16 kernel: weight decode per slot

`kernels/rocm/ds41_routed_f16.hip.cpp`, after E25: IQ1_M gate+up takes about
21 ms per 2,048-token layer, of which the weight decode and its permutes are
~4.4 ms and the matrix-core skeleton (block fields, fragment reads,
products) ~14.7 ms, at 39 of the measured 55 TFLOP/s. A decoded fragment
feeds at most four 16-slot tiles, on average 2.3. 128-slot tiles for large
experts would cut decode work by 13–24% but need 128 accumulator VGPRs and
twice the activation LDS. IQ2_XXS's 16-byte random codebook reads still
cost ~2 ms; 8-byte state entries would need ~12 more VALU per decode.
IQ2_S stays on MMQ.

### 7. Remaining prefill glue

Per 2,048-token chunk at 128K of depth (2026-10-28 profile, 1,342 ms of GPU
time): the Q8_0 projections still go through MMQ (30 ms) and its Q8_1 pass
(9.5 ms); the BF16 matmuls take 49 ms (the 384-row router at ~5 TFLOP/s);
the routed rows' closing expand before Engram layers and replay boundaries
8 ms; the shared-expert SwiGLU 5 ms and the routed input's FP16 conversion
4 ms. Between chunks the host still dequantizes the 2,048 embedding rows and
uploads them with a synchronous pageable copy (5–8 ms with the GPU idle);
a pinned buffer or a device gather would hide most of it. `q_b` and the
Engram projection still read FP32 inputs (E23: ~4% per launch with FP16).

### 8. Small-batch evaluation cost

Short follow-up turns evaluate 3–20 tokens per batch and the cost is not
monotonic: 7 tokens from position 8,469 take 136 ms, 9 tokens from 8,467
take 120 ms; at position 0, 10 tokens take 160 ms and 12 take 143. Find
which kernels switch paths between these sizes (`tools/prof/prof.py`), then
tune dispatch per projection and purpose; WMMA changes verification
arithmetic, so verification keeps its own switch points.

### 9. Stable-boundary prefill pass on every chat request

The prompt cache captures a checkpoint only at a prefill pass end, and every
chat request captures its stable boundary three tokens before the prompt
end: a 2,035-token prompt runs as 2,032 + 3 rows. The 3-row pass costs about
65 ms (`gufo bench -p 3`: 64.6 ms), the only serving-side prefill overhead
found. The checkpoint serves turns whose rendering diverges after the user
message (thinking mode, regenerated or edited replies). Capture it only when
the next turn can need it, or snapshot row N−3 from inside the final pass
(compressor accumulators, SWA windows and bounded replay make this
invasive). Gemma's generation-prompt lookahead is the local precedent. The
policy is shared (`src/cache/retention.cpp`), so other models' reuse must
hold.

### 10. Multi-user DSpark draft sizing

Concurrent sessions verify jointly in batches of up to 16 rows (E14), but
each still drafts a full block (anchor + 5). From four users a step needs two
or more joint evaluations, and mixed text decodes slower than AR: 51.22 /
52.42 / 55.93 tok/s at C4 / C6 / C8 against 60.35 / 71.33 / 78.46 (card
sweep). Fit the drafts into one 16-row pass, about 16 / users − 1 rows per
session as Gemma does, and fall back to AR rows when that leaves none.

### 11. Engram row reads under cache pressure

`engram_read` issues one buffered 144 B `pread` per row; once the expert
cache has grown the page cache holds none of them (`rchar == read_bytes` for
whole runs), so each row is a 4 KiB IO: ~98,000 per pp2048 batch, ~0.35 s of
device time taking queue depth from the expert reads. This is the likely
cause of streamed prefill getting slower as the cache fills (88.3% hits →
787, 95.3% → 742 tok/s) and of the remaining streamed prefill spread. In
order: deduplicate repeated rows within a batch, sort by file offset so
same-page rows share a read, then evaluate a bounded hot-row cache (2–4 GiB
pinned) after recording a real row-id trace to size its hit rate. A second
fast NVMe would also work; the slower KIOXIA did not.

### 12. Rank-major expert-contiguous repack

The GGUF stores each layer's experts as three projection-major ~1 GiB
blocks, so one expert is three reads a block apart. Repacking to
expert-contiguous rows ordered by the built-in rank (gate, down, up adjacent)
makes a decode job one sequential read and a prefill job one contiguous run,
toward the 8.2 GB/s sequential ceiling from today's 5.7 GiB/s. Needs a
repack tool beside `tools/ds41/quants.py`, a loader mapping expert id → rank
row, and byte-identical streamed outputs.

### 13. Route bias toward cached experts (quality-affecting, optional)

Xolotl multiplies streamed experts' router logits by ~0.5 for 13.7 →
30.3 tok/s on a 4090 at ~1 point of top-1 agreement; ×0.25 keeps 85%
agreement. This changes outputs: opt-in only, with its own quality gates.

### 14. Higher-precision Engram tables

Measure speed and quality of Engram tables above IQ4_NL (144 B rows).
[unsloth](https://huggingface.co/unsloth/DeepSeek-V4.1-Flash-1017-GGUF)
`engram/` offers Q5_K (125.89 GiB), Q6_K (150.21 GiB) and MXFP8
(188.83 GiB) in its own split format (`*.gguf_file`, `set_split.py`); BF16
tables (512 B rows, ~366 GiB) would come from the official checkpoint.
Quality: teacher-forced perplexity and agreement against the oracle at 3K
and 18.5K tokens. Speed: prefill and decode at the card depths plus Engram
read and wait times, since rows grow up to 3.6×, which interacts with item 11.

### 15. Smaller items

- **Routed-expert GEMV, wide loads.** The retuned decode kernels reach
  162–205 GB/s; IQ1 down projections stay at ~165 GB/s with a fixed
  per-launch floor (85–96 µs across 13.8–22.7 MB). A plain read of the same
  six experts streams at 230–238 GB/s.
- **Draft vocabulary.** A request-vocabulary draft head, as Qwen and Gemma
  use, with request tokens and a multilingual fallback; the verifier keeps
  the full vocabulary.
- **Tokenizer merges.** E3 memoizes pieces per span; the merge loop still
  rescans adjacent pairs (`bpe_rank`/`bpe_emit_piece`). Qwen's heap merge
  with token-ID pairs is the local precedent.
- **Engram worker setup.** `engram_read` starts its workers per evaluation;
  test a persistent pool if setup becomes visible.
- **Encoder SWA replay on a frontier miss.** Approximate; needs its own
  cache and quality contract.
- **`gufo diagnose`** does not report the streaming statistics.
- **Decode expert GEMV spills.** Several `expert_group_kernel` variants
  use 256 VGPRs and still spill 80–540 bytes per lane to scratch; check
  whether fewer columns per lane or `ext_vector_type` arrays remove them.

## Memory

Resident weights take 112.27 GiB GTT. A session at the native 1M context
adds 0.9 GiB of FP4 state; the batch scratch (1.4 GiB at 1M, 256 MiB of it
indexer scores) is shared by all sessions, and a 2048-token prefill peaks
0.3 GiB higher. Further reductions need smaller prefill batches, which cost
speed (EXPERIMENTS.md).
