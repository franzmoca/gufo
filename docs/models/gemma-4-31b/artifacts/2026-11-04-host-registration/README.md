# Whole-block host cache backing: upstream Qwen investigation

2026-11-04, Strix Halo gfx1151, Linux 7.2.11, pinned ROCm 10.1.
This investigation tests P6 in [EXTENSION-PLAN](../../EXTENSION-PLAN.md).

## Builds and scope

Baseline: local `upstream-main` `d2b01f24a9a70104170bfe69bd4752fb7ce8e4e1`.
Candidate: that exact tree plus [candidate.patch](candidate.patch), extracted
from `8ee46f24`, excluding its independent `FD_CLOEXEC` change. Both use the
same release preset (RelWithDebInfo production optimization), pinned toolchain
and explicit `GUFO_EXTENSIONS=qwen;qwen38_flash_next` selection. The candidate
is isolated in `/tmp/gufo-2m-upstream`; no production change was applied to the
Gemma worktree. Binary and flake hashes are in
[build-manifest.json](build-manifest.json). The functional harness ran from the
Gemma worktree; its reported build-input hash describes that checkout, not the
binary's source. The manifest above records the actual binary provenance.

Targets: Qwen3.8-27B UD-Q4_K_XL and Flash-Next UD-Q4_K_XL. All model runs use
AR, two sessions and context 16384. DFlash2/MTP share the backing implementation
but were not measured. These focused runs do not replace full model quality or
standard speed qualification.

## Matched HTTP histories

[run_http.py](run_http.py) records commands and preserves every run. Each model
uses ABBA order: original A1, candidate B1, candidate B2, original A2. The
`cache-growth` suite runs through `cache_growth_drop_reasoning_cold_control_3`:
four growing turns, an unchanged retry, and four uncached controls. RAM cache
budget: 2 GiB. The runner owns a fresh server/cache for each invocation.

All 72 requests passed the selected functional checks. Output hashes, token
counts, prefill chunks, cached tokens, snapshot bytes and restore bytes match
A1 in every run. [http-summary.json](http-summary.json) retains the checks and
flags; full request/phase observations remain in each run directory.

Cache restore milliseconds, shown per request rather than averaged:

| Model / request | A1 | B1 | B2 | A2 |
| --- | ---: | ---: | ---: | ---: |
| Qwen 27B turn 1 | 5.307 | 5.781 | 5.631 | 5.557 |
| Qwen 27B turn 2 | 5.533 | 6.122 | 5.804 | 6.071 |
| Qwen 27B turn 3 | 6.078 | 6.303 | 6.078 | 6.408 |
| Qwen 27B unchanged retry | 6.149 | 6.309 | 6.420 | 6.181 |
| Flash-Next turn 1 | 2.726 | 2.769 | 2.856 | 2.818 |
| Flash-Next turn 2 | 3.025 | 3.023 | 3.178 | 3.127 |
| Flash-Next turn 3 | 3.151 | 3.195 | 3.172 | 3.130 |
| Flash-Next unchanged retry | 3.136 | 3.059 | 3.105 | 3.189 |

All restore checks pass the unchanged 5% and 10 ms gates. There is no demonstrated
restore gain. Whole-run timing is **inconclusive**: Qwen B2 has cold-prefill
flags of +6.2% and +8.6%, while the unchanged A2 control also slows, reaching
+20.8% on one cold prefill. Flash-Next B1's first prefill is +15.1%; B2 and A2
pass. No flags are on restore timing. Retain these flags; do not interpret the
passing cache metrics as a complete end-to-end performance qualification.

## Controlled neighbour unmap

Source: [host_registration_blocks.hip](../../../../../tools/bench/host_registration_blocks.hip).
Build with `nix develop -c tools/bench/build.sh host_registration_blocks`.
Run inside the pinned development environment, not directly:
`nix develop -c /tmp/host_registration_blocks 6`.

The final [reproducer-pinned.csv](reproducer-pinned.csv) has 96 observations:
128 MiB + 4 KiB and 1 GiB + 4 KiB registered buffers, an 8 MiB neighbour above
or below, shared-block versus isolated layouts, ABBA order and six repeats.
Each sample uses a fresh registration; allocation, population and registration
are outside the timed unmap/next-launch scopes. GPU writes at both ends of the
registered buffer are checked after every sample; all pass.

No multi-millisecond stall reproduced. The largest post-unmap launch plus stream
synchronisation was 0.263 ms. [reproducer-summary.json](reproducer-summary.json)
contains each shape/layout's median and maximum; raw samples stay separate.
This controlled layout does not reproduce every allocation/thread history of
the original Gemma report.

The earlier `reproducer-fork-toolchain`, `reproducer-upstream-toolchain` and
`reproducer-both-sides` files are retained preliminary diagnostics. Direct
execution resolved ROCr `/nix/store/ym4x29...` through the HIP library, whereas
production uses `/nix/store/88jjsz...` via its pinned environment. Only the
`reproducer-pinned` run is used for the conclusion above.

## Verified post-readiness backing growth

[probe_growth.py](probe_growth.py) uses an 8 GiB RAM budget,
distinct 5409-token prompts, then replays older requests. The diagnostic
[registration_trace.c](registration_trace.c) interposes actual `hipHostRegister`
calls, recording size, alignment and call duration without changing arguments.
Compile with `cc -shared -fPIC -O2 registration_trace.c -ldl -o
/tmp/gufo-registration-trace.so`; the probe supplies `LD_PRELOAD` only to its
owned servers. These instrumented runs establish growth/layout and provide
single observations, not repeated end-to-end timing qualification.


Each model/variant completed eight distinct captures and three older-prompt
replays (44 requests total). Actual registrations establish backing growth:

| Model | Registered segments: ready → final | Bytes per segment | Original alignment | Candidate alignment |
| --- | ---: | ---: | --- | --- |
| Qwen 27B | 17 → 57 | 133,496,832 | Not all aligned | Every base aligned to 2 MiB |
| Flash-Next | 17 → 55 | 133,496,832 | Not all aligned | Every base aligned to 2 MiB |

All registrations succeeded. Outputs, token/cache counts, snapshot bytes and
restore bytes match between original and candidate. Every replay prefills zero
tokens. Per-replay restore times:

| Model / replay | Restored bytes | Original ms | Candidate ms |
| --- | ---: | ---: | ---: |
| Qwen 27B 5 | 515,330,208 | 6.826 | 7.254 |
| Qwen 27B 6 | 515,330,208 | 7.023 | 7.361 |
| Qwen 27B 7 | 515,330,208 | 7.082 | 7.149 |
| Flash-Next 5 | 268,721,444 | 3.524 | 3.625 |
| Flash-Next 6 | 268,721,444 | 3.504 | 3.533 |
| Flash-Next 7 | 268,721,444 | 3.624 | 3.374 |

[growth-summary.json](growth-summary.json) and the four `*-growth-*` directories
retain the raw evidence. Memlock was 8 MiB, so this does not qualify the
high-memlock/long-term-pinned configuration.

## Decision

**Defer the upstream performance PR.** The shared Qwen allocation path and its
unaligned mappings are confirmed by execution, but the historical
multi-millisecond neighbour-unmap stall did not reproduce on the current host.
The alignment change has no demonstrated Qwen AR restore or end-to-end gain in
these measurements. Keep P6 in the plan with this limitation. Do not apply the
candidate to the active Gemma tree or publish it as a measured Qwen speedup.
The older Gemma evidence remains valid as historical evidence; these results
do not establish why the two investigations differ.
