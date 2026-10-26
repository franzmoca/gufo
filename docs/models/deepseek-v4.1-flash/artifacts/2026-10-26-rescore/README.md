# E1: compact FP32 candidate rescoring

Disposition: retained for measured AR decode improvement. DSpark timing
qualification remains **inconclusive and unqualified**. All observations are
retained at the original 5% / 3 ms margins.

Production baseline is `ee49a0fd` (same code as the preceding sampling-default
commit); candidate changes only the small-batch indexer/rescore path. See
[provenance.json](provenance.json) for binary hashes, compiler paths,
flake lock, artifact identity, source-patch hash and profile database hashes.
The raw workspace `/tmp/gufo-ds41-opt/` retains the binaries, complete build
and server logs, SQLite traces, numerical score rows and reproduction scripts.
Model/teacher captures and binary profiles are kept outside source control.

## Measurements

`gufo bench -m MODEL -p 2048 -n 128 -d 0,32768 -r 2`, then a separate
`-d 131072 -r 2` invocation. Baseline/candidate binaries use identical
Nix dependencies, IQ1_M target, IQ4_NL Engram, greedy sampling, 2048-row
prefill and 140-row decoder bounded replay. Startup is excluded by `bench`.
The C2 control uses `-d 32768 -c 2 -r 1`: pp is aggregate, tg per user.

All benchmark stdout is retained, including the slower unchanged-prefill
control at 128K. C1 AR decode: 23.97 → 24.03 tok/s at depth zero,
22.78 → 23.05 at 32K, 21.10 → 21.89 at 128K. C2 32K decode:
11.50 → 11.55 tok/s/user, one repetition; no concurrency speedup claim.

The candidate-mask test's `--score-bench` measures dense score + mask versus
compaction + dense-score clear + compact score, with identical inputs and
alternating device-event intervals. Full score/top-512 equality and an
independent FP64 formula are checked before timing. The initial file uses
fixed FP4 exponents; the final file uses varying exponents. Both are retained.
These are operator timings, not model throughput.

## Correctness

- [operator-details.txt](operator-details.txt): `ds41.attention` and
  `ds41.candidate_mask`, including tied top-512 order and FP64 score checks.
- [numerical-quality.json](numerical-quality.json): 3K and 18.5K numerical
  controls. The reused independent 18.5K oracle's source hash and input IDs
  were checked against the current oracle and exact scored histories. Its
  first 2,999 positions are also reported as a current-reference prefix
  control; the historical standalone 3K oracle is not replaced.
- [spec-check4.txt](spec-check4.txt): eight speculative cycles at 18.5K,
  each commit length 1–4 twice; 20 full-logit rows bit-identical to serial
  decode and to a pass with corrupted rejected rows.

## Request qualification

The `long-context` suite uses `--sampling-preset deepseek41`,
`--request-timeout 240`, `--context 65536 --sessions 1`, and either
`--speculative off` or `--speculative dspark --dspark-model DRAFTER`.
The Lucebox Q2_K/Q4_K drafter is the same in each DSpark run. Actual cold
prompt depth is 22,591 tokens, above the candidate threshold. Reports retain
loaded mode, real draft counters, reply checks and per-request/phase metrics.

AR: `functional-base-ar` → `functional-candidate-ar` passes all 124 timing
measurements and reply/coverage comparisons.

DSpark: chronological order is base/candidate, base2/candidate2,
base3/candidate3, then one unchanged-base control. Every run passes
correctness. [dspark-qualification.json](dspark-qualification.json) aggregates
all three pairs plus the control with `tests/functional/compare.py`, using
the original 5% / 3 ms margins: 86 passed, 38 inconclusive, zero confirmed
slowdowns and no quality/coverage changes. The cancellation queue flag in
the first two pairs disappears in the third (candidate 80.4 ms versus
baseline 80.9–81.9 ms); other short-request timings and draft widths vary
on the baseline as well. These observations do not qualify DSpark performance.

## Profiling

Both SQLite traces were captured separately from throughput measurements:

```sh
nix develop -c python3 tools/prof/prof.py run -o PROFILE -- \
  BINARY bench -m MODEL -p 2048 -n 8 -d 32768 -r 1
```

The profiler's default stage map does not classify DS41 kernels. The retained
JSON summaries group by actual kernel symbols. Decode starts at the first
`hc_mix` after the final prefill FP16 dense kernel; it contains eight full
decode evaluations and excludes load/depth preparation. Compact scoring
launches exactly 32 times (four reindex layers × eight tokens). The final
prefill summary selects the preceding timed pp2048 region and excludes all
earlier preparation chunks. Whole-process GPU utilization is not used.

`SHA256.json` inventories the retained observations; the SQLite source traces
remain at the paths/hashes recorded in `provenance.json`.
