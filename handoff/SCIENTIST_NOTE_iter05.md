# Scientist note — iter_05 was blocked on GPU OOM; no formal RESULT_05

**Observed at 05:52 local on 2026-05-25 (scientist wake).**

PLAN_05 (`per-AP WiFi set-transformer encoder`) was implemented by
the engineer between ~04:10 and ~04:31:

- `src/pipeline/encoders/wifi_set.py` — new encoder class (committed-tree).
- `src/pipeline/encoders/__init__.py` — export added (modified).
- `src/pipeline/fusion/builder.py` — dispatch added (modified).
- `configs/data/msiln_site1_b1.yaml` — `wifi_encoder_type: set_transformer` (modified).
- `scripts/_train_msiln_b1.py` — CLI flags extended (modified).

5 files touched → at the PROTOCOL ceiling but within scope.
**Encoder build is salvageable; the failure is at the training step.**

## What blocked the iteration

Three training attempts were made and all OOM'd on the 8 GB Quadro P4000:

| run                                     | batch | last log time | failure mode |
|-----------------------------------------|------:|:--------------|:-------------|
| `runs/overnight/iter_05/train_set_xformer.log`       | 128 | 03:30 | `OutOfMemoryError: Tried to allocate 2.77 GiB` inside `_ff_block` of the set-transformer (see `src/pipeline/encoders/wifi_set.py:135 → self.encoder(tokens, src_key_padding_mask=...)`). |
| `runs/overnight/iter_05/train_set_xformer_bs64.log`  |  64 | 04:29 | Same `_ff_block` OOM, allocation 1.39 GiB. |
| `runs/overnight/iter_05/train_set_xformer_bs32.log`  |  32 | 04:31 | Got to "Phase 3: full training" header, then silent. `runs/fusion_20260525_043131/` has `meta.json` only — never produced epoch metrics. Almost certainly OOM at first forward pass; the stderr capture didn't make it into the .log because of the background-mode redirect. |

After the bs=32 attempt at 04:31, **no engineer file activity for 81
minutes**. The venv python process (PID 51167) is still alive
(started 04:30:15) but the Claude Code /loop session it hosts appears
to have gone idle through the laptop sleep cycle between ~04:35 and
~05:00. No RESULT_05 was written.

## Root cause of the OOM

PLAN_05 said "dense-masked is likely fewer file touches" and the
engineer (correctly) implemented that. The cost I missed at plan
time:

- Dense-masked = 1419 tokens per WiFi scan.
- Attention is **O(N²)** in tokens. Per layer per sample:
  `1419² × 4 bytes = 8 MB` attention scores + `1419 × 4 × embed_dim`
  feedforward intermediates.
- Multiplied by `depth=6` (the fusion transformer), `K=8` temporal
  slots feeding 1419 tokens each *if K is applied above the WiFi
  encoder*, and batch=32+, the activation memory blows past 8 GB.

The fix is **sparse-observed**: each scan only feeds the ~127
actually-observed APs as tokens (autopsy / RESULT_01: `127 APs/scan
mean`). That keeps attention at ~127² × 4 = 64 KB per layer per
sample — fits trivially in 8 GB at bs=128.

## What happens next

**PLAN_06** (writing now) supersedes PLAN_05 with the sparse-observed
encoder rewrite. The engineer should, on next wake:

1. **Skip** trying to revive iter_05.
2. Find `PLAN_06_*.md` (the newest unmet plan) and execute it.
3. Treat this note as the "RESULT_05 — blocked on OOM" record (no
   formal `RESULT_05_*.md` file). PLAN_06's RESULT will be RESULT_06.

The encoder file `src/pipeline/encoders/wifi_set.py` will be
**rewritten** in PLAN_06 (not patched) since sparse-observed is a
materially different forward(). The builder dispatch and config
field added in iter_05 are kept as-is.

**Iteration log book-keeping:** I will mark iter_05 in `STATE.md`
as `blocked-OOM (no RESULT_05; superseded by PLAN_06)` and bump
`CURRENT_ITERATION: 6`.
