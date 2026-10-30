# DeepSeek V4.1 Flash: Engram table precision

Does a higher-precision Engram table improve the model? The ISTA GSQ-RCO
builds share one IQ4_NL table (144 B rows, 110.6 GB). The
[unsloth repository](https://huggingface.co/unsloth/DeepSeek-V4.1-Flash-1017-GGUF)
(`engram/`) publishes the same table at Q5_K, Q6_K and MXFP8; MXFP8
(264 B rows, 202.8 GB) is the checkpoint's own format, so it is lossless.
This page compares IQ4_NL with MXFP8 on the same backbone.

Hardware: one Strix Halo (gfx1151), 128 GB. Engine: `gufo` on
`feat/bigger-engrams`.

## Summary

On `IQ1_M` and `IQ2_XXS`, MXFP8 gives no measurable quality gain over
IQ4_NL: teacher-forced perplexity, MMLU-Pro, HumanEval+ and MBPP+ all agree
within noise, and a blind pairwise review (on `IQ1_M`) prefers neither. The
one consistent difference is style: with MXFP8 the model writes longer
coding answers and adds a prose explanation around its code more often.
The speed comparison is pending.

| | `IQ1_M` IQ4_NL | `IQ1_M` MXFP8 | `IQ2_XXS` IQ4_NL | `IQ2_XXS` MXFP8 |
|---|---:|---:|---:|---:|
| PPL, 18.5K tokens | 6.2665 | 6.2682 | 6.0029 | 6.0080 |
| MMLU-Pro, 1,001 questions | 74.3% | 74.4% | 75.4% | 75.4% |
| HumanEval+ (plus tests) | 90.2% | 90.2% | 87.8% | 88.4% |
| MBPP+ (plus tests) | 77.8% | 78.0% | 79.1% | 78.6% |

## Using the MXFP8 table

The unsloth `.gguf_file` is a plain single-tensor GGUF. Gufo reads it as
the Engram shard of an ISTA build:

1. Download `engram/DeepSeek-V4.1-Flash-1017-engram-MXFP8.gguf_file` and
   `engram/set_split.py`.
2. Stamp it as the last shard of the six-shard ISTA layout:
   `python set_split.py <file> 6 6`.
3. Point `…-00006-of-00006.gguf` of the build at it (a separate directory
   of symlinks keeps the IQ4_NL setup intact).

Format: GGUF type 43, 33-byte blocks of one E8M0 scale and 32 E4M3FN values
(value = 2^(scale−127) × code). Upstream ggml uses type 43 for unscaled
F8_E4M3 (32-byte rows per 32 values); such a file fails the loader's bounds
check instead of being misread. Rows are dequantized on the host like the
other Engram types; no GPU kernel is involved. The decode was checked
against the IQ4_NL rows (cosine 0.997, matching RMS).

## Teacher-forced perplexity

Batched path, same token files as [QUALITY.md](QUALITY.md). ΔNLL is
MXFP8 − IQ4_NL with a 95% interval.

| Backbone, tokens | PPL IQ4_NL | PPL MXFP8 | Top-1 IQ4_NL / MXFP8 | Argmax agreement | ΔNLL |
|---|---:|---:|---|---:|---:|
| `IQ1_M`, 3K | 6.2958 | 6.3024 | 0.5799 / 0.5772 | 95.4% | +0.0011 ± 0.0050 |
| `IQ1_M`, 18.5K | 6.2665 | 6.2682 | 0.5928 / 0.5919 | 95.9% | +0.0003 ± 0.0020 |
| `IQ2_XXS` streamed, 3K | 6.0220 | 6.0292 | 0.5852 / 0.5829 | 95.8% | +0.0012 ± 0.0055 |
| `IQ2_XXS` streamed, 18.5K | 6.0029 | 6.0080 | 0.5981 / 0.5975 | 95.9% | +0.0009 ± 0.0023 |

The 95.4–95.9% agreement equals the GPU-vs-oracle agreement of one engine
on one table, so swapping the table moves predictions about as much as
floating-point differences do.

## Task evaluation

`gufo serve` (release build), greedy (`temperature 0`, `seed 1`),
thinking off, context 16,384, four concurrent requests on `IQ1_M` and
eight on `IQ2_XXS` (streamed experts). Each item is
compared per question across the two tables (McNemar, exact).

- **HumanEval+ / MBPP+** (evalplus 0.1.10 / 0.2.0, 164 / 378 problems):
  prompt "Please provide a self-contained Python script that solves the
  following problem in a markdown code block:", 1,024 tokens. Scored by
  evalplus in a networkless container; "plus" requires the original and
  the extended tests.
- **MMLU-Pro**: 1,001 questions sampled per category in proportion
  (seed 20261030) from the test split at revision `b189ec76`. The prompt
  allows brief working and asks for a final `Answer: X`; 512 tokens.
  Unparsed or truncated answers count as wrong.

| `IQ1_M` | IQ4_NL | MXFP8 | Only IQ4_NL | Only MXFP8 | p |
|---|---:|---:|---:|---:|---:|
| HumanEval+ base | 95.7% | 95.1% | 2 | 1 | 1.00 |
| HumanEval+ plus | 90.2% | 90.2% | 1 | 1 | 1.00 |
| MBPP+ base | 91.3% | 91.3% | 5 | 5 | 1.00 |
| MBPP+ plus | 77.8% | 78.0% | 6 | 7 | 1.00 |
| MMLU-Pro | 74.3% | 74.4% | 30 | 31 | 1.00 |

| `IQ2_XXS` streamed | IQ4_NL | MXFP8 | Only IQ4_NL | Only MXFP8 | p |
|---|---:|---:|---:|---:|---:|
| HumanEval+ base | 95.1% | 95.7% | 0 | 1 | 1.00 |
| HumanEval+ plus | 87.8% | 88.4% | 0 | 1 | 1.00 |
| MBPP+ base | 91.3% | 91.3% | 4 | 4 | 1.00 |
| MBPP+ plus | 79.1% | 78.6% | 7 | 5 | 0.77 |
| MMLU-Pro | 75.4% | 75.4% | 27 | 27 | 1.00 |

On `IQ1_M`, MMLU-Pro picks the same answer on 916 of 1,001 questions
(922 on `IQ2_XXS`); per-category
differences go both ways and stay within small-sample noise (computer
science −11.7 points on 34 questions, history +6.2 on 32). About 5-8% of
answers hit the 512-token limit with either table (`IQ1_M`: 76 and 73,
58 of them the same questions); leaving out every question truncated on
either side gives 731 vs 733 correct of 910 on `IQ1_M`.

Swapping the backbone moves more results than swapping the table:
`IQ1_M` and `IQ2_XXS` (both IQ4_NL) differ on 6 HumanEval+ plus results
(5 favour `IQ1_M`, p = 0.22) and 21 MBPP+ plus results (13 favour
`IQ2_XXS`). The discordant HumanEval+ problems are ordinary greedy-path
flips: misread specifications and edge cases such as `fib(0)` and
booleans counted as integers, in both directions.

These results measure direct answers. DeepSeek reports V4 Flash in
Non-Think, Think High and Think Max modes, and the gap between modes is
large on coding (LiveCodeBench 55.2 / 88.4 / 91.6); thinking-mode
evaluation of the tables is not covered here.

## Writing style

On `IQ1_M` coding answers whose text differs between tables (111 HumanEval+,
258 MBPP+; the rest are identical):

| | IQ4_NL | MXFP8 |
|---|---:|---:|
| Median length, HumanEval+ / MBPP+ (chars) | 855 / 586 | 924 / 728 |
| Prose explanation only on this side, HumanEval+ | 4 | 15 (p = 0.02) |
| Prose explanation only on this side, MBPP+ | 7 | 55 (p < 0.0001) |
| Comment lines in code, MBPP+ | 261 | 300 |

Docstrings and test blocks are as frequent on both sides. On `IQ2_XXS`
the length difference repeats: median 734 vs 778 characters on the 84
differing HumanEval+ answers, 566 vs 620 on the 191 differing MBPP+ answers.

A blind pairwise review of 70 random differing pairs (random A/B order,
judged on code clarity, explanation quality, apparent correctness and fit
to the request; length not rewarded) preferred MXFP8 in 14, IQ4_NL in 11
and called 45 ties (sign test p = 0.69; summed strength 20 vs 20). Ten
pairs shown again with A and B swapped received the same verdict all ten
times. The decisive pairs turned on occasional slips present with both
tables: wrong code, a self-contradicting explanation, or an answer that
ends mid-sentence.

## Related measurements

Unsloth varies the table across IQ4_NL, Q5_K, Q6_K and MXFP8 in its
[DeepSeek V4.1 Flash guide](https://unsloth.ai/docs/models/deepseek-v4.1-flash)
and reports the KLD change as a chart; a third-party card summarises it
as barely moving top-1. Its backbone table (with IQ4_NL) puts `UD-IQ1_M`
at KL 0.314 / 87.1% top-1, an error far larger than the table swap.

## Pending

- Speed: pp2048 / tg128 at the card depths for both tables on both
  backbones, same binary.
