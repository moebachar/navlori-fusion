# Plan 07 — C2 closure on canonical RoNIN unseen-subjects (FRDR archive in `data/`)

> **Unblock — 2026-05-25 evening, user-side.** RESULT_05 left C2 as
> a MANUAL Phase-C item because the FRDR archive was Globus-gated.
> User has now placed the archive in `data/` as
> `FRDR_dataset_538_download_606_202605251142.zip` (~14.9 GB).
> This iteration discharges or refuses C2 per the locked plan.

## Hypothesis

C2 ("IMUCNN within 20 % of RoNIN ResNet1D on canonical RoNIN
unseen-subjects, raw-weighted") was last labelled in RESULT_05 as
`keep (in-domain only)` because we only had the a000 intra-session
proxy. Run-1's pre-history numbers (in restored
`docs/SOTA_BASELINES.md` IMU section) are **IMUCNN 14.41 m raw /
8.41 m SVD-aligned** vs **ResNet1D 5.93 m raw** (paper 5.14 m). The
raw-ATE gap is ~2.4×. Under the amended rubric (raw weighted ≥
aligned), this **does not** discharge C2 at the 20 % bar — we expect
the audit verdict to land at one of:

- **`keep (in-domain only)`** (current label) — RoNIN canonical
  numbers reproduce within ±10 % of run-1 references AND show the
  ~2.4× gap. The label sticks; Phase B uses IMUCNN with the gap
  noted in the PerCom paper.
- **`replace`** — gap > 50 % AND the in-domain proxy under-reported
  it. Phase B uses ResNet1D unmodified (Demand #3) as the IMU
  encoder.

In any case: this is the **only** iteration in run-2 where we can
get a paper-strength C2 measurement; produce it cleanly.

## Steps

### Step 0 — Unzip + feasibility probe (5–15 min depending on disk)

**Step 0a.** Extract the FRDR archive:

```powershell
$src = 'data\FRDR_dataset_538_download_606_202605251142.zip'
$dst = 'data\ronin_frdr'
New-Item -ItemType Directory -Force -Path $dst | Out-Null
Expand-Archive -LiteralPath $src -DestinationPath $dst -Force
```

If `Expand-Archive` rejects the 14.9 GB archive (PS5 has a 2 GB
limit on some builds), fall back to:

```powershell
.venv\Scripts\python.exe -c "import zipfile; \
  zipfile.ZipFile(r'data\FRDR_dataset_538_download_606_202605251142.zip').extractall(r'data\ronin_frdr')"
```

Expected size on disk: ~30 GB unpacked (HDF5 files compress well).
If `data\ronin_frdr` won't fit on `X:\`, extract under the largest
free drive and symlink:

```powershell
mklink /D data\ronin_frdr <target>
```

**Step 0b. Internal-layout probe.** RoNIN's `data_glob_speed.py`
loader opens `<root>/<seq>/data.hdf5` with fields:

| key | content |
|---|---|
| `synced/gyro_uncalib` | 3-axis uncalibrated gyro (not `gyro`!) |
| `synced/acce` | 3-axis accelerometer with gravity |
| `synced/time` | per-sample timestamps |
| `pose/tango_pos` | GT Tango position |
| `pose/tango_ori` | GT Tango orientation quaternion |

After extraction, pick one sequence from the unseen list
(`a006_2`) and verify all five keys exist:

```powershell
.venv\Scripts\python.exe -c "import h5py; \
  f=h5py.File(r'data\ronin_frdr\<path-to>\a006_2\data.hdf5','r'); \
  print(list(f['synced'].keys()), list(f['pose'].keys()))"
```

**Step 0c. Coverage probe.** Sachini/ronin ships canonical lists at
`C:\Users\FabLab\AppData\Local\Temp\ronin\lists\`:
- `list_train.txt` — 87 sequences
- `list_val.txt` — 16 sequences
- `list_test_unseen.txt` — 32 sequences (the C2 target)
- `list_test_seen.txt` — 33 sequences

After extraction, report how many of each list is actually present
on disk. If `list_train.txt` is < 80 sequences, training from
scratch is risky → prefer pretrained checkpoint (Step 1a). If
`list_test_unseen.txt` is < 32 sequences, **stop and document** —
C2 can't be closed cleanly with a partial test set.

**Acceptance for Step 0:** extracted size + per-list-coverage table
written into RESULT_07; a000 sequences (already in
`data/ronin_a000/`) should NOT be re-extracted (verify the FRDR
archive doesn't redundantly include subject a000 — if it does,
either skip or overwrite; document the choice).

### Step 1 — Day-1 SOTA reproduction: RoNIN ResNet1D on canonical unseen

Prefer (1a) if the FRDR archive includes pretrained weights; else (1b).

**(1a) Pretrained checkpoint, eval-only.** Look for
`<ronin_frdr>/pretrained_models/` or similar (RoNIN README says
pretrained models are distributed via the same FRDR DOI). If a
`.pt` checkpoint is present, run **unmodified vendored
`ronin_resnet.py --mode test`** with that checkpoint:

```powershell
.venv\Scripts\python.exe `
  C:\Users\FabLab\AppData\Local\Temp\ronin\source\ronin_resnet.py `
  --mode test `
  --test_list C:\Users\FabLab\AppData\Local\Temp\ronin\lists\list_test_unseen.txt `
  --root_dir data\ronin_frdr `
  --out_dir runs\overnight\run2_iter_07\resnet1d_eval `
  --model_path <pretrained-checkpoint-path>
```

**(1b) Train from scratch.** Same script, `--mode train` first,
then `--mode test`. ~30–60 min training. Demand #3: NO edits to
their source; runtime shims (`np.int = int`, etc.) live in OUR
wrapper if needed.

**Pre-test gate** (only if training): 5-epoch run on first 20
sequences of `list_train.txt` should show loss dropping monotonically.

**Memory budget check:** ResNet18-on-IMU is tiny (~5 MB params);
peak well under 1 GB on the P4000. Just confirm.

**Acceptance:** ResNet1D reproduction emits `metrics.csv` /
`results.json`. Compare to references:
- Run-1 ref: **5.93 m raw ATE** (within ±10 % = 5.34–6.52 m).
- Paper: **5.14 m** (within ±20 % = 4.11–6.17 m).
Report per-sequence ATE distribution (min, p25, median, p75, p90,
max) + RTE per RoNIN's `metric.compute_ate_rte`.

Source: `https://raw.githubusercontent.com/Sachini/ronin/master/source/ronin_resnet.py`,
`https://raw.githubusercontent.com/Sachini/ronin/master/source/metric.py`.

### Step 2 — Reproduce IMUCNN on canonical unseen-subjects

Use the restored `scripts/eval_ronin_ate_fixed.py` (already on this
branch from PLAN_02 Step 0a). One **mandatory** patch:

The script's current `_ate_aligned` is a hand-rolled SVD Procrustes
fit — fine as legacy column but per the amended rubric (correction
#3) the **canonical aligned metric is Umeyama** AND we additionally
report RoNIN's own metric for apples-to-apples with Step 1.

Add a small helper to OUR wrapper (Demand #3 untouched):

```python
# scripts/eval_ronin_ate_fixed.py — within OUR wrapper, not vendored source
# Import RoNIN's metric and Umeyama from scipy:
import sys; sys.path.insert(0, r'C:\Users\FabLab\AppData\Local\Temp\ronin\source')
from metric import compute_ate_rte  # RoNIN's own (raw RMSE, GT-start-anchored)
from scipy.spatial import procrustes  # for Umeyama-style aligned metric
```

Eval flow:
- For each unseen sequence, reconstruct trajectory by integrating
  per-step velocity (already done by the script).
- Report THREE numbers:
  - **Raw ATE (RoNIN's own)**: `compute_ate_rte(pred_pos, gt_pos)`
  - **Umeyama-aligned ATE**: standard `scipy.spatial.procrustes` or
    `evo.core.metrics.APE` with `align=True, correct_scale=True`.
  - **RTE (RoNIN's own)**: from the same call.

**Pre-test gate** (if any retraining): same as Step 1.

**Acceptance:** report per-sequence + aggregate (min, p25, median,
p75, p90, max, mean) for raw ATE, Umeyama ATE, RTE. Compare against
run-1 references:
- IMUCNN raw run-1: 14.41 m (±10 % = 12.97–15.85 m).
- IMUCNN SVD-aligned run-1: 8.41 m.

### Step 3 — C2 audit decision (the focused experiment)

Compare Step 2 against Step 1, **raw-weighted** under correction #3:

| metric | IMUCNN | ResNet1D | gap | 20 % gate |
|---|---|---|---|---|
| Raw ATE (RoNIN's own) | … | … | …% | PASS / FAIL |
| Umeyama ATE | … | … | …% | (secondary) |
| RTE | … | … | …% | (secondary) |

**Verdict logic:**
- Raw gap ≤ 20 % → **C2 DISCHARGED**. IMUCNN label upgrades to
  `keep`. Phase B continues with IMUCNN.
- Raw gap > 20 % AND Umeyama gap > 20 % → **C2 NOT discharged**.
  IMUCNN label stays `keep (in-domain only)`. Phase B uses
  IMUCNN by default with the gap noted; **add explicit Phase B
  contingency**: if 4-modality fusion shows IMU-leg saturation,
  swap to ResNet1D unmodified.
- Raw gap > 20 % but Umeyama gap ≤ 20 % → **C2 NOT discharged**
  per correction #3 (raw wins). Same as above.

### Step 4 — Update Phase A close-out + write RESULT_07

Update the Phase A summary table (RESULT_04 / RESULT_05) with the
canonical C2 column populated. If C2 discharged, RESULT_02's
addendum gets a "C2 CLOSED 2026-05-25 PLAN_07" line + IMUCNN label
restored to plain `keep`. If not discharged, the existing `keep
(in-domain only)` stays + a "C2 measured, gap N % > 20 %" line is
added.

Write `handoff/results/RESULT_07_c2-closure-ronin-canonical-v2.md`
with:

1. Step 0 outcomes — extract size, list-coverage table, layout
   probe results.
2. Step 1 — ResNet1D numbers + comparison to run-1 ref + paper.
3. Step 2 — IMUCNN numbers (raw + Umeyama + RTE).
4. Step 3 — gap table + verdict.
5. Step 4 — Phase A summary table updated.
6. Open question for scientist.

## Sources

- RoNIN paper: Herath, Yan, Furukawa, ICRA 2020. arXiv:1905.12853.
- Sachini/ronin repo: https://github.com/Sachini/ronin.
- Canonical lists: `lists/list_*.txt` in repo;
  `https://raw.githubusercontent.com/Sachini/ronin/master/lists/list_test_unseen.txt`
  (32 sequences confirmed via WebFetch 2026-05-25).
- RoNIN's metric: `source/metric.py` → `compute_ate_rte(est, gt,
  pred_per_min=12000)` = raw RMSE on (x, y) anchored at GT[0], NO
  rotation/scale alignment, RTE = sliding-window same metric.
- HDF5 layout: `source/data_glob_speed.py` →
  `synced/{gyro_uncalib, acce, time}` + `pose/{tango_pos,
  tango_ori}`.
- FRDR archive: `data/FRDR_dataset_538_download_606_202605251142.zip`
  (~14.9 GB, present 2026-05-25 ~19:46 local).
- Vendored RoNIN repo: `C:\Users\FabLab\AppData\Local\Temp\ronin\`.
- Existing eval script (already on this branch): `scripts/eval_ronin_ate_fixed.py`.
- Amended rubric: STATE.md "Amended audit rubric (locked 2026-05-25
  ~12:55 local)".
- Pre-condition: PLAN_05 RESULT_05 reframed C2 as `keep (in-domain
  only)` pending this iteration.

## What to report back

(See Step 4 above.)

## Reversibility

- Step 0 (extract): permanent on disk but gitignored.
  `data/ronin_frdr/` should be added to `.gitignore` if not already
  (engineer confirms).
- Step 1 (vendored repo run): throwaway; no edits.
- Step 2 (IMUCNN eval): existing script; only OUR wrapper gains
  ~10 lines of metric imports.
- Steps 3–4: documentation.

Files committed: RESULT_07, addenda to RESULT_02 + Phase A
close-out, small wrapper patch to `eval_ronin_ate_fixed.py`.

**Demand #3** — `C:\Users\FabLab\AppData\Local\Temp\ronin\source\`
unmodified. RoNIN's `compute_ate_rte` imported pure.

**Compute budget:** total iteration ≤ 90 min.
- Step 0: 10–15 min (15 GB extract on local disk).
- Step 1a (pretrained): 10 min eval-only / Step 1b: 45 min train+eval.
- Step 2: 20 min (IMUCNN train ~15 min + eval ~5 min).
- Step 3: 5 min.
- Step 4: 10 min writeup.

If overrun: skip Step 1b (RoNIN ResNet train-from-scratch) and rely
on Step 1a (pretrained checkpoint). If both Step 1a and 1b fail,
write a partial RESULT documenting the obstacle and ship Step 2's
IMUCNN-canonical-only number as a one-sided result — at least the
IMU-side number on canonical data is in evidence.

If FRDR archive turns out to be incomplete (missing the unseen
subjects), Branch-Y reaffirmation stays and we explicitly document
that C2 closure was attempted but data is incomplete.
