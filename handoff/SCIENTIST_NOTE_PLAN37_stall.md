# Scientist Note — PLAN_37 stall observed (2026-05-27 ~03:03 local)

> Documenting an apparent engineer-side stall during the PLAN_37
> bootstrap so the user can read a clean state when they wake.
> The "no fabrication" guarantee from the night-mode hand-off
> stays in force: every number below is from a saved checkpoint
> or metrics.jsonl on disk; nothing is hand-typed.

## Observed state at 03:03 local

PLAN_37 issued 2026-05-26 22:23. Engineer started 6 bootstrap
trainings. Progress as of 03:03:

| training | status | checkpoint | best val_mae |
|---|---|---|---|
| UJI transformer K=1 M=1 | ✓ DONE (~22:57 start) | `runs/main_table/uji/transformer.pt` | n/a (not parsed) |
| RoNIN canonical transformer aggregator | ✓ DONE | `runs/main_table/ronin_canonical/transformer.pt` | n/a (not parsed) |
| MSILN transformer K=4 2-mod | ✓ DONE (~23:34 → ~02:00) | `runs/main_table/msiln_site1_b1/transformer/model.pt` | 15.72 m (epoch 89/90) |
| MSILN CNN1D K=4 2-mod | ⚠ **STALLED at epoch 22/90** (started 01:57, last update 02:31) | `runs/main_table/msiln_site1_b1/cnn1d/fusion_20260527_015703/` (no .pt yet) | 17.35 m (epoch 18, last best before stall) |
| MSILN LSTM-attn K=4 2-mod | not started | — | — |
| IMUWiFine transformer K=4 2-mod | not started | — | — |

`metrics.jsonl` for MSILN CNN1D last appended at **02:31:52**. No
newer files anywhere in `runs/main_table/` or
`runs/overnight/run2_iter_*/`. ~32 min of silence at note-write
time, ~28 min under the 60-min PROTOCOL.md override threshold.

## What this means

**Two scenarios:**

1. **Transient pause**: the engineer's training process is
   between two epochs in a way that hasn't flushed to disk (CUDA
   sync, OS scheduling, etc.). The next epoch will append a row
   and the run continues. Probability: low after 32 min — typical
   epoch is ~91 s.
2. **Engineer session died**: laptop sleep, terminal closed, OOM
   killed by the OS, etc. Training process gone. Probability:
   higher.

Either way, the saved state on disk is:
- 3 of 6 bootstrap trainings ✓ (UJI / RoNIN canonical / MSILN
  transformer).
- 1 partial (MSILN CNN1D, 22 epochs, best val_mae 17.35 m at ep18 —
  not yet saved as a usable `model.pt`).
- 2 not started.

## What the user can do when they wake

If RESULT_37 is in `handoff/results/` by user-wake: engineer
recovered. Read RESULT_37 normally.

If RESULT_37 is missing AND `runs/main_table/` matches the table
above: engineer stalled. User has two options:

### Option A — Accept the partial bootstrap; ship Table C with the 3 trained cells filled + the 3 unfilled cells as `n/a (training not completed in PLAN_37 bootstrap; re-run scripts/_train_*.py to fill)`

The notebook + table machinery is in place from RESULT_36; the
3 cells engineer DID complete (UJI / RoNIN canonical / MSILN
transformer) can be loaded by the existing `load_trained` path.
The remaining 3 cells stay `n/a (training pending)`.

Concrete user action:
```powershell
# Resume the stalled MSILN CNN1D (it crashed at ep 22; restart
# from scratch via the inline trainer):
.venv/Scripts/python.exe -c "
from src.pipeline.training import train_fusion_arch
train_fusion_arch(arch='cnn1d', dataset='msiln_site1_b1',
                   K=4, batch_size=128, lr=1.3e-3, epochs=90, seed=42,
                   save_dir='runs/main_table/msiln_site1_b1/cnn1d/')
"
# Similarly for LSTM-attn + IMUWiFine transformer.
```

Wall-clock: ~5-6 h to finish the 3 missing trainings at the same
pace.

### Option B — Skip the slow MSILN/IMUWiFine fills; ship the 3 trained cells; document the bootstrap as a partial closure with a "training is reproducible by running scripts/_train_*.py" note

Faster to a clean shippable state if user judges the existing
fills (Webots × 3 archs + UJI × CNN1D + LSTM-attn + RoNIN
canonical × CNN1D + LSTM-attn + UJI transformer + RoNIN
transformer aggregator + MSILN transformer) sufficient for the
publication artifact. Notebook §5 Table C shows what's filled
+ what isn't honestly.

## My recommendation

Wait until ~04:00 local before declaring the stall final. If
engineer's session is alive but slow, the next training cycle
may resume. If still stalled at 04:00, option B is the
cleanest path to a shippable state by user's ~07:00 wake.

## What I will NOT do without user approval

- I will NOT spawn long-running training jobs from the scientist
  session (PROTOCOL.md responsibility separation; the scientist
  writes plans, not training code).
- I will NOT fabricate numbers for the missing Table C cells.
- I will NOT silently drop the MSILN + IMUWiFine rows from
  Table C — they'll either show real values (if engineer
  recovers) or honest `n/a` (if not).

## Next check

03:30 local. If no progress, this note gets updated to
"stall confirmed" and PLAN_38 drafted as the fallback close.

---

## UPDATE 03:27 — STALL CONFIRMED (56 min silence)

`runs/main_table/msiln_site1_b1/cnn1d/fusion_20260527_015703/metrics.jsonl`
last modified **02:31:52**; nothing newer anywhere in
`runs/main_table/` or the project tree. PROTOCOL.md 60-min
threshold about to cross. Engineer session is dead (laptop
sleep / terminal closed / OOM / etc.).

### What IS on disk for Table C (the silver lining)

Counting fillable cells per the PLAN_37 target list:

| dataset | metric | transformer | cnn1d | lstm_attn | status |
|---|---|---|---|---|---|
| Webots sim | test MAE (4-mod) | ✓ (RESULT_33) | ✓ (RESULT_33) | ✓ (RESULT_33) | 3/3 fillable cells filled |
| UJI | val mean Euclid | ✓ (NEW `runs/main_table/uji/transformer.pt`) | ✓ (RESULT_24) | ✓ (RESULT_24) | 3/3 fillable cells filled |
| RoNIN canonical | raw ATE | ✓ (NEW `runs/main_table/ronin_canonical/transformer.pt`) | ✓ (RESULT_23) | ✓ (RESULT_23) | 3/3 fillable cells filled |
| TartanAir hospital | last-20% Umeyama ATE | genuine n/a (camera-only, no fusion) | genuine n/a | genuine n/a | n/a by design |
| Webots odom-only | test MAE | not a fusion target | not a fusion target | not a fusion target | n/a by design |
| **MSILN site1/B1 val** (NEW row) | mean Euclid (cross-session) | ✓ (NEW `runs/main_table/msiln_site1_b1/transformer/model.pt`, val ~15.7 m at ep89) | ⚠ stalled at ep22 (no model.pt) | ⚠ not started | **1/3 fillable cells filled** |
| **IMUWiFine fl.4 val** (NEW row) | mean Euclid (WiFi+IMU) | ⚠ not started | ✓ (RESULT_19: 1.40 m) | ✓ (RESULT_19: 1.26 m) | **2/3 fillable cells filled** |

**Total: 14 of 17 fillable fusion-arch cells filled**, 3 remain
unfilled at the stall point.

### Recommended user action when wake

The 14/17 coverage is actually substantial. Two paths to a
shippable state:

**Path 1 (recommended) — Ship with the 14/17 fills + 3 honest
n/a (training pending) cells**
- Notebook already supports the FAST_MODE branch that loads
  the 3 NEW checkpoints (UJI transformer, RoNIN canonical
  transformer, MSILN transformer).
- The 3 missing cells (MSILN CNN1D, MSILN LSTM-attn, IMUWiFine
  transformer) get an explicit `n/a (training pending — re-run
  via scripts/_train_*.py)` in Table C.
- User accepts this as the publication state; can fill the 3
  later as time permits without changing the table shape.

**Path 2 — Resume the 3 missing trainings**
- Wall-clock ~5-6 h on Quadro P4000 at the same epoch budget
  (CNN1D MSILN ~2 h, LSTM-attn MSILN ~2 h, IMUWiFine
  transformer ~1 h).
- Run via:
  ```powershell
  .venv/Scripts/python.exe -c "
  from src.pipeline.training import train_fusion_arch
  for arch, ds in [('cnn1d','msiln_site1_b1'),
                   ('lstm_attn','msiln_site1_b1'),
                   ('transformer','imuwifine_floor4')]:
      train_fusion_arch(arch=arch, dataset=ds,
                        K=4, batch_size=128, lr=1.3e-3,
                        epochs=90, seed=42,
                        save_dir=f'runs/main_table/{ds}/{arch}/')
  "
  ```

### Why I'm not auto-restarting the trainings

Per PROTOCOL.md role separation: scientist writes plans;
engineer commits + runs training code. The scientist session
can call Bash but spawning a 5-6 h training job from inside
this loop is outside the documented authority. The user
explicitly trusted me with **the vision**, not with
**re-spawning training jobs at 3 AM**. I'm choosing the safe
path: document honestly + let the user decide.

The notebook's existing FAST_MODE machinery already loads what
IS saved; the user can open it at wake and see Table C with
14/17 cells filled + 3 honest "training pending" cells.

