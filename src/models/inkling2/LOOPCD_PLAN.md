# LoopCD for Inkling 2 Mini: working plan

Branch `feat/inkling2-loopcd` (from upstream `main` 4060f397). Working notes, not
for upstream. Discussion: https://github.com/gufo-org/gufo/discussions/1407

## Goal

Opt-in LoopCD-Logits ([arXiv 2610.02185](https://arxiv.org/abs/2610.02185)) for
Inkling 2 Mini, then measure whether it improves quality enough to justify its
cost. Off by default; with it off nothing is allocated and output is
bit-identical to `main`.

Recipe (the author's tweet plus the implementations surveyed):

- Comparison readout z_1: the state after pass 1 of the core (core output x and
  pass 1's block states) runs the coda (layers 18-23) again in **its own cache
  slots** (5 sliding-window visits + 1 global, own conv state), then the depth
  mix, final norm, head and soft cap. z_1 then equals the model with
  `n_passes = 1` bit for bit, which is our main correctness oracle.
- `z' = z + w (z - z_1)`, `w = w_max (1 - (p1 - p2))` from softmax(z), after
  the soft cap, in FP32. `w_max = 0.5`.
- Modes: `off`, `on` (every row guided), `skip-thinking` (rows inside a thinking
  block unguided; on guided rows control-token probabilities stay exactly those
  of z and ordinary tokens are renormalized to the same total mass, as omlx).

Non-goals for now: LoopCD-Hidden, the read-only coda variant (no own cache),
fewer passes, per-request strength.

## 1. Options

- `--loopcd off|on|skip-thinking` and `--loopcd-strength W` (w_max, default
  0.5, range (0, 4]) in `extension.json` `options` (serve, bench, prompt),
  read in `provider.cpp` through `ext::FindOption`. Names open; one option per
  behavior. No ext-api change needed.
- Per-request strength would need a request field the extension API does not
  have: out of scope, note it in the discussion if asked.
- Test-only: a way to run the engine with `n_passes = 1` (ParseConfig refuses
  it today) for the oracle. Keep it out of the CLI.

## 2. Engine state (the largest refactor)

The engine and the cache layer hard-code 48 visits in many places
(`state_layout.hpp` `kVisits`, slot table `fs::SlotEntry[48]`, `d_tbl + b*48`,
conv state `48 * kConvChannels * 3`, `d_vcap` `[48][6656][rows]`, `Tap`,
prefill visit units, cache components).

- Introduce a visit count = 48 + 6 reference visits when LoopCD is on, 48 when
  off. Replace the literal 48s with it (off keeps every size and bit).
- Drafter rows already sit at visits `kVisits + l` (48-50) in the state layout:
  decide where reference visits go (after the drafter, or renumber) and keep
  the layout self-describing.
- Memory per session: global reference slot 4 KiB/token (K+V f16), five window
  rings 5 x 1024 x 4 KiB = 20 MiB, conv +6 x 80 KB. Rows mapped as the session
  grows, like the existing global slots.

## 3. Forward

Common to decode (`BuildDecode`, graph), batched decode (`DecodeN`), verify
(`VerifyN`, up to 64 rows / 16 batched) and prefill (`Ubatch`):

1. **Stash pass 1.** At the end of pass 0 copy `cur` and the pass's block
   states (`d_states[prelude_n+1 .. d_ns)`): those buffers are reused by
   passes 2-3, so they need copies (or separate buffers for pass 0).
2. **Main coda** as today; DSpark `CaptureFeature` stays on the main path.
3. **Reference coda**: layers 18-23 on the stash, reference slots 48-53, own
   conv offsets. Runs on every row, prefill included (its cache must be
   complete); prefill needs no reference head except on the last row.
4. **Reference head**: `DMix(output_res_query)` over the reference states,
   `output_norm`, head GEMV with soft cap, into a second logits buffer.
5. **LoopCD kernel**, in place on `logits_d` before argmax / linear sampling /
   readback: per row one sweep for top-2 + softmax denominator, one sweep to
   combine; per-row guidance flag from device memory (skip-thinking) so graphs
   stay valid; control-token renormalization (logsumexp over ordinary ids for
   z and z'). Every sampling fast path (device argmax, GPU linear sampling,
   coupled proposals, verify) then reads guided logits with no runner change.

Implementation order: first sequential (steps 3-4 after the main coda, simple
and easy to check), then the **stacked** version: the reference rows join the
coda layers as extra rows with their own slot-table entries (like extra
sessions in `DecodeN`), and both head rows go through one GEMV call, so the
coda's dense weights and the head are read once. Requires the batched kernels
to be row-exact (check) and a row count that grows from nb to 2nb at layer 18.

Rollback and checkpoints: rejected drafts must roll back the reference window
rings and conv tails like the main ones (`d_vcap` for 54 visits,
`CommitVerify`); the in-pass prefill checkpoints from #1403 must include the
reference slots.

## 4. Runner

- skip-thinking flag per row from the runner's `Block` state (`kThinking`).
  For verify rows, advance the block tracker over the draft tokens so each
  row gets the flag of its own position (drafts can cross `<|end_message|>`).
- Control-token id set from the tokenizer's `is_special`; check whether it is
  the contiguous range >= 199998 the other engines assume.
- Constraints (tool grammar) apply after LoopCD; nothing changes there.

## 5. Cache

- New components for the reference slots (append + window + conv) in
  `state_layout`; bump `StateMeta::kVersion`.
- LoopCD on/off must be part of the cache identity (memory and disk): a prefix
  saved with LoopCD off has no reference cache. `w_max` and the mode do not
  change cached state (only `on` vs `off` allocation), so they stay out of it.

## 6. Correctness tests (CPU / GPU, focused targets)

- Kernel vs host FP64 oracle: fixed, adaptive, skip mask, control renorm; a row
  alone equals the same row in a batch, bit for bit.
- LoopCD off: existing `inkling2_*` tests unchanged, plus greedy text hashes
  equal to `main`.
- On with w = 0: logits equal off, bit for bit.
- Reference logits equal the engine with `n_passes = 1`, bit for bit
  (sequential, then stacked).
- DSpark verify == AR under LoopCD (greedy; sampled with a fixed seed), single
  and batched, including rejected-draft rollback.
- Graph == eager; concurrent sessions == alone.
- Cache: save/restore and disk round trip with LoopCD on; an off-cache is not
  restored into an on-server.
- Options: parsing, refusals.

## 7. Quality (needs the GPU; arms off / on / skip-thinking, w_max 0.5)

- `tests/functional/pi_agent.py` agent replays, sessions retained, tool results
  inspected.
- HumanEval and MMLU-Pro (high effort, cap above the thinking budget so budget
  forcing and LoopCD's loop fix are separated). No in-tree harness: pick or
  write a small external one (never `gufo eval`). Same sets across arms.
- Inkling 2 functional suites (`src/models/inkling2/functional/devlab.json`).
- Report wins/losses per item with a sign test; retain every run, including
  negative results. Optional second strength (0.3) only if 0.5 looks promising.

## 8. Speed

Decode AR and DSpark (prose, code), prefill, DSpark acceptance, memory, on
`result/bin/gufo` or `build/release/gufo`. Check tuned profile
(accelerator-performance) and fan control first; alternate base/candidate
(ABBA), no builds during timing runs. Compare sequential vs stacked.

## Milestones

1. M1: options, state refactor (off bit-identical), sequential reference coda +
   kernel, `on` mode, correctness tests 6.
2. M2: quick quality signal (pi_agent replays + one benchmark, off vs on).
   Decision gate: stop if nothing moves.
3. M3: skip-thinking (runner flags, verify rows, control renorm).
4. M4: stacked coda/head; speed measurements.
5. M5: full quality comparison of the three arms; write up results in the
   discussion.
6. M6: docs (`docs/models/inkling-2-mini.md`) and PR, or extension route,
   depending on the maintainers' answer.

## Open questions

- Upstream vs extension (discussion #1407).
- Does skip-thinking keep the gains that come from ending reasoning loops?
- Exact cost of the stacked version on gfx1151 (expert overlap between the two
  coda rows).
- Placement of the reference visits relative to the drafter rows.
