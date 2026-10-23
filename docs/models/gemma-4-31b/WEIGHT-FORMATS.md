# Gemma 4 31B weight formats for faster prefill

Study of alternatives to GGUF K-quants for Gemma 4 31B on gfx1151, aimed at
prefill throughput (backlog L7). Measured 2026-10-22. Status: research; no
format is integrated.

## Summary

- Prefill spends ~91% of a 2048-token chunk at depth 0 in the projections,
  which run Q4_K on binary16 WMMA at ~39 TFLOP/s (ceiling 55.4). The only
  instruction with a higher rate is `v_wmma_i32_16x16x16_iu4` (~111 TOPS
  measured), which needs int4 weights *and* int4 activations (W4A4). int8
  WMMA runs at the binary16 rate, so W4A8 gains nothing.
- A standalone W4A4 GEMM (128×128 tiles, per-128 scales) runs Gemma's
  projection shapes at 62–73 TOPS: **1.72 s** for a 2048-token chunk's
  projections against **3.06 s** for the production Q4_K kernel (1.77×).
  The activation rotate+quantize step adds ~0.1 s per chunk unfused.
- Quality is the obstacle. Rotated symmetric int4 weights with per-128
  scales and GPTQ reach KL 0.061 against BF16 on the model's own replies,
  ~3× UD-Q4_K_XL (0.021); int4 activations add ~0.05 (W4A4 0.109). Used as
  the only copy (prefill and decode) this is not acceptable.
- As a second, prefill-only copy beside UD-Q4_K_XL (+~16 GB), W4A4 moves
  the reply KL from 0.021 to 0.027–0.030 (asymmetric or per-64 weights at
  the low end); the prompt-side damage alone, with BF16 decode, is 0.012.
- EXL3 at 4.00 bpw matches UD-Q4_K_XL at 4.87 bpw (KL 0.024 vs 0.021):
  ~18% fewer bytes per decoded token for the same quality, if a trellis
  decode GEMV can keep Gufo's ~94% of bandwidth.
- A separate int4 copy used only for prompt processing (decode keeps its own
  weights) limits the damage to the prompt's KV and hidden states: W4A4
  prefill with UD-Q4_K_XL decode scores KL 0.027–0.030 against 0.021 for
  UD-Q4_K_XL alone (+30–40%); W4A4 prefill with BF16 decode 0.012.
- Trellis formats (QTIP / EXL3 / Halogen HT) give the best quality per bit:
  EXL3 at 4.00 bpw scores KL 0.024, close to UD-Q4_K_XL at 4.87 bpw. They
  cannot feed integer WMMA; decoding them inside a binary16 WMMA GEMM cost
  27–30% (4-bit) and 48–54% (3-bit) over the same kernel with binary16
  weights, 1.5–1.7× the production Q4_K kernel.

## Formats in other gfx1151 engines

| Engine | Format | Prefill path | Notes |
| --- | --- | --- | --- |
| Halogen (HT3/HT4, "ht43" profile) | QTIP bitshift trellis (L=16, K=3/4, V=1), EXL3 "mul1" computed codebook, 128-point Hadamard on both sides, one f16 vector per row/column, 16×16 tiles in an RDNA-friendly order | closed source | ht43 vs its 4-bit rotated format: decode no faster, prefill −2…−10%. Its dense 27B checkpoint stores a rotated int4 copy for prefill and an FP8/NVFP4 copy for decode instead of trellis weights |
| Kyojin (EXL3 on HIP) | same trellis family, Hessian LDLQ, mixed bits per tensor | ≤144 rows fused trellis decode; above, full fp16 reconstruction + rocBLAS/hipBLASLt (16–34 TF) | ~1 bpw better than GGUF at equal KL; decode kernels 55–65% of DRAM bandwidth |
| Strixite (STRIXW) | plain asymmetric RTN, Q4 g128 / Q8 g64, BF16 scale and min planes, no rotation or calibration | binary16 WMMA, ~2-op dequant (`v_perm` + 0x6400 + packed FMA) | Q8 for every dense (always-read) tensor dominated quality (KLD 0.124 → 0.021 on its MoE) |
| Hipfire (MQ4) | FWHT-rotated int4, f16 scale per 128 (group 256), GPTQ/AWQ calibration | W4A4 on iu4 WMMA, 256×256 tiles, activations rotated and quantized per 128 | Qwen3.8-27B dense pp8192 1203 tok/s, GEMMs at 75.6 TOPS; A4 costs +0.02–0.04 KLD |

## Method

`tools/gemma4/quant_sim/` streams the BF16 checkpoint layer by layer and
runs every variant's hidden states through each layer (projections on
binary16 inputs with FP32 sums, as Gufo's prefill; norms, attention and the
residual in FP32). Its forward matches exllamav3's Gemma 4 implementation
token for token. Variants replace the projection weights (and optionally
quantize their inputs); embeddings and the tied LM head stay BF16.

Evaluation: 24 chat prompts (agentic tool use from recorded opencode/curl
sessions at 10–16K tokens, coding, long-context prose, reasoning and
multilingual) rendered with the Gemma 4 template; the BF16 model wrote the
replies itself (temperature 1, top-k 64, top-p 0.95, ≤1024 tokens, stopping
at end of turn or a tool call). Only the 16.6K reply tokens are scored:
KL(BF16 ‖ variant), its 99th percentile, top-1 agreement. Prompt and raw-text
positions are out of distribution for the instruction-tuned model (BF16
perplexity ~700 on raw prose) and inflate KL 10–30×.

Calibration (GPTQ, EXL3): 32 × 2048 tokens of the exllamav3 standard mix
(web, wiki, code, technical, multilingual), disjoint from the evaluation.
GPTQ: static per-group scales, damping 1% (10× on two down-projection
Hessians that were not positive definite). Rotation: random signs, then an
orthonormal 128-point Walsh-Hadamard transform per 128-value block of K.

## Quality

Reply tokens, against BF16 (perplexity 1.126 on its own replies).

| Variant | Projection bpw | KL mean | KL p99 | Top-1 |
| --- | ---: | ---: | ---: | ---: |
| UD-Q8_K_XL | 9.15 | 0.0004 | 0.006 | 0.995 |
| UD-Q4_K_XL | 4.87 | 0.021 | 0.42 | 0.970 |
| int4 rotated, RTN, f16 scale/128 | 4.12 | 0.086 | 1.52 | 0.939 |
| int4 rotated, GPTQ, scale/128 | 4.12 | 0.061 | 1.11 | 0.949 |
| int4 rotated, GPTQ, scale/64 | 4.25 | 0.050 | 0.92 | 0.954 |
| W4A8 (GPTQ/128, int8 activations/128) | 4.12 | 0.061 | 1.08 | 0.949 |
| W4A4 (RTN/128) | 4.12 | 0.137 | 2.39 | 0.923 |
| W4A4 (GPTQ/128) | 4.12 | 0.109 | 1.89 | 0.931 |
| W4A4 (GPTQ/64) | 4.25 | 0.097 | 1.74 | 0.934 |
| int4 rotated, asymmetric GPTQ, scale and min/128 | 4.25 | 0.050 | 0.90 | 0.954 |
| W4A4 (asymmetric GPTQ/128) | 4.25 | 0.095 | 1.71 | 0.935 |
| EXL3 4-bit trellis (Hessian LDLQ) | 4.00 | 0.024 | 0.42 | 0.965 |

Prefill-only W4A4: prompt tokens through the W4A4 weights, the scored reply
through the decode weights (a second weight copy for prefill).

| Prompt weights | Reply weights | KL mean | KL p99 | Top-1 |
| --- | --- | ---: | ---: | ---: |
| W4A4 GPTQ/128 | BF16 | 0.012 | 0.22 | 0.979 |
| W4A4 GPTQ/128 | UD-Q4_K_XL | 0.030 | 0.55 | 0.963 |
| W4A4 GPTQ/64 | UD-Q4_K_XL | 0.028 | 0.51 | 0.965 |
| W4A4 asymmetric GPTQ/128 | UD-Q4_K_XL | 0.027 | 0.53 | 0.966 |

By category (KL mean; agentic replies are short tool calls, ~300 tokens):

| Variant | Agentic | Code | Prose | Reasoning |
| --- | ---: | ---: | ---: | ---: |
| UD-Q4_K_XL | 0.008 | 0.011 | 0.036 | 0.021 |
| EXL3 4-bit | 0.019 | 0.015 | 0.032 | 0.027 |
| W4A4 asym prefill + UD-Q4_K_XL | 0.029 | 0.015 | 0.049 | 0.023 |
| W4A4 GPTQ/128 (single copy) | 0.060 | 0.071 | 0.159 | 0.115 |
| int4 rotated, asymmetric GPTQ (f16 scale and min/128) | 4.25 | 0.050 | 0.90 | 0.954 |
| W4A4, asymmetric GPTQ/128 | 4.25 | 0.095 | 1.71 | 0.935 |
| EXL3 4-bit trellis (Hessian LDLQ) | 4.00 | 0.024 | 0.42 | 0.965 |

Prompt tokens through one format, the scored reply through another (a
prefill-only weight copy; bpw is the prefill copy's):

| Prefill → decode | Prefill bpw | KL mean | KL p99 | Top-1 |
| --- | ---: | ---: | ---: | ---: |
| W4A4 GPTQ/128 → BF16 | 4.12 | 0.012 | 0.22 | 0.979 |
| W4A4 GPTQ/128 → UD-Q4_K_XL | 4.12 | 0.030 | 0.55 | 0.963 |
| W4A4 GPTQ/64 → UD-Q4_K_XL | 4.25 | 0.028 | 0.51 | 0.965 |
| W4A4 asymmetric GPTQ/128 → UD-Q4_K_XL | 4.25 | 0.027 | 0.53 | 0.966 |

Prefill-copy recipe sweep (W4A4 GPTQ prefill, UD-Q4_K_XL decode; binary16
attention products, the two longest agentic prompts left out; UD-Q4_K_XL
alone 0.0213 on this set):

| Prefill recipe | KL mean | KL p99 | Top-1 |
| --- | ---: | ---: | ---: |
| base: rotation 128, weight and activation scales /128, standard calibration | 0.0284 | 0.53 | 0.966 |
| chat-format calibration | 0.0279 | 0.56 | 0.966 |
| SmoothQuant scales (alpha 0.5) | 0.0291 | 0.56 | 0.966 |
| rotation blocks up to 512 | 0.0282 | 0.55 | 0.966 |
| rotation blocks up to 1024 | 0.0286 | 0.52 | 0.965 |
| activation scales /64 | 0.0273 | 0.50 | 0.965 |
| chat calibration + smoothing + rotation 1024 | 0.0283 | 0.55 | 0.965 |
| the same + activation scales /64 | 0.0273 | 0.51 | 0.966 |

None of these moves the prefill copy's cost beyond about ±4% of the base;
only per-64 activation scales are consistently (slightly) better.

Calibration corpus and layers kept on the decode format for the prompt too
(same setup; differences paired per sequence, 95% intervals from resampling
the 22 sequences; activation scales /64 throughout):

| Prefill recipe | KL mean | − UD-Q4_K_XL alone | − base |
| --- | ---: | ---: | ---: |
| base (standard calibration) | 0.0273 | +0.0061 [+0.0033, +0.0099] | — |
| calibration-v6 (bartowski, 252K tokens) | 0.0278 | +0.0066 [+0.0028, +0.0126] | +0.0005 [−0.0021, +0.0036] |
| layers 0–2 and 57–59 on UD-Q4_K_XL | 0.0275 | +0.0062 [+0.0037, +0.0098] | +0.0002 [−0.0012, +0.0017] |
| the 10 global-attention layers on UD-Q4_K_XL | 0.0272 | +0.0059 [+0.0039, +0.0084] | −0.0002 [−0.0021, +0.0014] |
| both (15 layers) | 0.0268 | +0.0055 [+0.0031, +0.0088] | −0.0005 [−0.0020, +0.0008] |
| calibration-v6 + both | 0.0258 | +0.0045 [+0.0021, +0.0080] | −0.0015 [−0.0034, +0.0004] |

The prefill copy's cost over UD-Q4_K_XL is real (+0.005 to +0.006 KL,
about +25%) and no recipe tested changes it measurably. Single-variant
intervals are wide (UD-Q4_K_XL alone: 0.0213, [0.0152, 0.0281]): the 22
replies differ a lot, so only paired differences are informative.

By category (KL mean) the prefill copy costs most on prose, whose prompts
are long documents: UD-Q4_K_XL 0.036 → 0.049–0.053 with a W4A4 prefill;
code 0.011 → 0.015–0.018; reasoning (short prompts) 0.021 → 0.023–0.025.

W4A4 GPTQ/128 on one projection class, the rest BF16 (the costs add up to
the full variant's):

| Class | KL mean |
| --- | ---: |
| q/k/v | 0.017 |
| o | 0.018 |
| gate/up | 0.043 |
| down | 0.030 |

## Kernels

Standalone, 2048 tokens, random weights rotated beyond the 32 MiB MALL,
medians of 10 (`tools/bench/gemma4_w4a4_gemm_bench.hip`,
`gemma4_trellis_gemm_bench.hip`, `gemma4_prefill_bench.hip`).

| Shape | Q4_K (production) | W4A4 iu4 | Trellis K=4 (binary16 WMMA) | Trellis K=3 |
| --- | ---: | ---: | ---: | ---: |
| gate\|up 43008×5376 | 23.8 ms | 13.5 ms | 35.7 ms | 40.8 ms |
| down 5376×21504 | 11.8 ms | 6.7 ms | 20.0 ms | 24.3 ms |
| q, k, v sliding (16384×5376 in total) | 9.3 ms (q\|k Q4_K + v Q6_K) | 5.4 ms | — | — |
| q\|k sliding 12288×5376 | 6.8 ms | — | 10.4 ms | 11.8 ms |
| o global 5376×16384 | 9.3 ms | 5.1 ms | 18.5 ms | 26.1 ms |
| Chunk projections | 3062 ms | 1724 ms | — | — |

The trellis kernel decodes each weight once per 128 tokens into LDS; the
same kernel with binary16 weights takes 27.5 / 15.8 ms on gate|up / down,
so the trellis decode itself costs 27–30% (K=4) and 48–54% (K=3). Both
trellis and W4A4 kernels are first versions; Gufo's Q4_K kernel is tuned.

## Integration design: an int4 prefill sidecar

Decode keeps reading the GGUF; a second weight file serves prefill only.

### Sidecar file

A GGUF of plain tensors (no new container code), built from the BF16
checkpoint, never from the K-quants (that would round twice), by
`tools/gemma4/quant_sim/sidecar.py`. Per layer:

| Tensor | Type | Shape | Content |
| --- | --- | --- | --- |
| `blk.N.<input>.w4a4.vec` | F32 | K | per-input-channel factor D (random signs) |
| `blk.N.<proj>.w4a4.codes` | I32 | K/128 × M × 16 | signed int4 codes (two's complement), eight per little-endian word, even k in the low nibble, K-block-major |
| `blk.N.<proj>.w4a4.scales` | F16 | K/128 × M | per-128 weight scales |

`<input>` is `attn_qkv`, `attn_output`, `ffn_gate_up` or `ffn_down`;
`<proj>` uses the model GGUF's names (`attn_q`, `attn_k`, `attn_v` on
sliding layers, `attn_output`, `ffn_gate`, `ffn_up`, `ffn_down`), so gate
and up fuse at load as they do from the model GGUF. The rotated input is
x' = H(vec · x) with H the orthonormal Walsh-Hadamard transform over blocks
of 128; each weight is exactly code × scale.

Metadata (`gufo.w4a4.*`): format version, rotation block, weight and
activation group sizes, method, calibration, block count, embedding length,
and the model identity (`google/gemma-4-31B-it` and the SHA-256 of its
`config.json`). The sidecar pairs with the model, not with one GGUF: any
quantization of the same checkpoint uses the same sidecar; the loader checks
the identity and every tensor shape. 14 GiB for the 31B.

### Converter

`sidecar.py` reuses the simulator's quantizer: the per-layer calibration
Hessians (from the cache, or computed by a calibration forward with the
WMMA Hessian kernel), the signs and block transform, and GPTQ (1% damping,
q|k|v and gate|up stacked). The weights it writes are bit-identical to the
`w4a4gptq-c6` simulation variant. With cached calibration-v6 Hessians a
build takes about 20 minutes on this machine (19 s per layer, 26 s on
global layers); computing the Hessians first adds about 30 minutes.

### Runtime

- Loading: the sidecar uploads into the existing per-tensor prefill-copy
  slot (`DeviceTensor::prefill`), checked against the loaded GGUF's shapes
  and identity. The option name and form are open: `--prefill-weights`
  promises bit-identical prefill today, which a lossy copy breaks.
- Prefill (`Executor::Project`, rows > 16): when the tensor has an int4
  copy, quantize the FP32 input rows once per input (q|k|v and gate|up share
  theirs): multiply by the input's vector, block Walsh-Hadamard, per-token
  per-128 scale, int4 codes; then the iu4 GEMM writes FP32 rows as today.
  Norms, RoPE, attention, residuals and the KV cache are unchanged. A first
  version quantizes from the FP32 rows in a separate kernel (~0.1 s per
  2048-token chunk); fusing it into the producers (attention norm, attention
  output, pre-FFN norm, GeGLU) follows.
- Decode and verification (≤16 rows) read the GGUF as today, and attend to
  the KV that prefill wrote: the configuration the prefill-only rows above
  measure.

### Numerics and tests

- Converter, simulation and kernels must share one activation scale rule
  (the simulation's four-way clip search and the benchmark's {7, 5} pair
  differ today) and one rounding of codes and scales.
- An operator test against FP64 on production shapes (codes, scales and
  activations generated from the reference transform).
- Prefill KL against the scalar reference on the existing fixtures, the
  `long-context` and `cache` functional suites, and pp2048 at depth 0, 16K
  and 32K against main.

## Prototype in Gufo (2026-10-23)

`--prefill-sidecar PATH` (bench, prompt, serve) loads the sidecar beside the
GGUF (14 GiB more device memory, +5 s load). Prefill chunks (more than 16
rows) of a dense binary16-prefill model project through the int4 copies:
`LaunchW4A4Quantize` rotates and quantizes each binary16 input once (q/k/v
and gate/up share theirs) with the simulator's four-way clip rule, then
`LaunchW4A4Gemm` (the 128×128 benchmark kernel, ragged rows padded to the
tile) writes FP32 rows; an int4 gate|up writes FP32 rows and GeGLU follows
as in the unfused route. Decode, verification and the vocabulary head keep
the GGUF.

`gufo bench`, UD-Q4_K_XL, 3 repetitions (`--prefill-weights decoded`, the
fastest exact copy: 616 tok/s at depth 0):

| Test | GGUF | With sidecar |
| --- | ---: | ---: |
| pp2048 | 575 tok/s | 881 tok/s (+53%) |
| pp2048 @ d16384 | 456 | 648 (+42%) |
| pp2048 @ d32768 | 393 | 531 (+35%) |
| tg32 | 11.93 | 11.94 |

Quality (QUALITY.md, `gemma4.prefill_sidecar`): decode is unchanged bit for
bit; the int4 rows reproduce the simulation (conversation KL 1.15 in Gufo
vs 1.18 simulated, top-1 0.68 vs 0.67); decoding after an int4 prefill
stays at KL 0.10 from exact rows on that fixture.

## Open questions

- Fusing rotate+quantize into the norm, attention and GeGLU producers, and
  a tuned GEMM (the prototype runs the first benchmark kernel).
- Task-level evaluation (agentic sessions, long-context retrieval) of the
  int4 prompt copy against UD-Q4_K_XL alone.
- Prefill-copy quality: larger rotations, per-64 activation groups, or
  keeping gate/up (the most A4-sensitive class) on the binary16 path.
- Decode: a trellis GEMV for EXL3-style weights at 1 and 3–8 rows (MTP
  verification), against Gufo's Q4_K GEMV at ~94% of bandwidth.
- EXL3 at 3 bits on the same evaluation (not run: ~4 min of encoding per
  layer and bit width).
