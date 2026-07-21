# Plan 08 — Camera external-SOTA validation on TartanAir hospital

> **Unblock — 2026-05-25 evening, user-side.** RESULT_03 left
> Camera's per-leg validation as a MANUAL Phase-C item because DPVO
> needed `lietorch`/`altcorr` CUDA ops that didn't install on this
> Windows machine in iter 03 (Branch Q). User has now placed a
> TartanAir hospital sample in `data/` as
> `hospital_sample_P000.tar.gz`. This iteration produces the
> paper-strength per-leg evidence the RESULT_03 review note called
> for.

## Hypothesis

`DPVOMotionEncoder`'s per-leg validation rests on Webots-only
numbers in RESULT_03 — that's project-internal evidence, not
defensible vs published VO baselines. The hospital sample lets us
compare **our motion encoder (trunk + correlation head)** against
**one published VO pipeline** (DPVO if its CUDA ops install on
this machine; else TartanVO MIT pure-PyTorch baseline) on the **same
sequence**, using the **same standard metric** (`evo`-computed ATE
RMSE with Sim(3) alignment).

Two outcomes:
- **Camera per-leg validation = paper-strength** — our encoder
  within ~30 % of the chosen public-VO pipeline's ATE on the
  hospital sequence (looser than the 20 % bar that applies to
  WiFi/IMU because DPVOMotionEncoder is a motion descriptor and
  the comparison VO is a full pipeline; the ~30 % bar is judged
  case-by-case in the audit).
- **Camera per-leg validation = paper-soft** — gap > 30 % to the
  chosen public-VO pipeline. The RESULT_03 `keep with smoothness
  debt` label is preserved with an explicit "in-sim only" caveat
  in the PerCom paper; Phase B can still consume the encoder for
  the C3 fusion claim.

## Steps

### Step 0 — Extract + feasibility probe (5–10 min)

**Step 0a.** Extract the TartanAir sample:

```powershell
$src = 'data\hospital_sample_P000.tar.gz'
$dst = 'data\tartanair_hospital'
New-Item -ItemType Directory -Force -Path $dst | Out-Null
tar -xzf $src -C $dst
```

`tar` ships with modern Windows; if not, use Python:

```powershell
.venv\Scripts\python.exe -c "import tarfile; \
  tarfile.open(r'data\hospital_sample_P000.tar.gz').extractall(r'data\tartanair_hospital')"
```

**Step 0b. Layout probe.** TartanAir canonical layout (per
`castacks/tartanair_tools/data_type.md`, verified via WebFetch
2026-05-25):

```
<root>/hospital/hospital/<Difficulty>/<Pxxx>/
  image_left/   (PNGs)
  depth_left/   (numpy or PNGs)
  pose_left.txt (one row per frame; 7 floats: tx ty tz qx qy qz qw, NED frame)
  image_right/  (optional)
  depth_right/  (optional)
  seg_left/     (optional)
  flow/         (optional)
```

The archive name `hospital_sample_P000.tar.gz` suggests a single
P-folder. After extraction, find the actual root path with:

```powershell
Get-ChildItem -Recurse -Filter 'pose_left.txt' -Path data\tartanair_hospital | Select-Object FullName, Length
```

Confirm one `pose_left.txt` exists; report the row count
(= frame count) and the parent directory path. Pick that as
`<seq_root>`.

**Step 0c. Frame count + modality probe.** Confirm:
- `image_left/` has the same count as `pose_left.txt` rows.
- No `imu/` subdir — TartanAir v1 is image-only (DPVO/TartanVO/DROID
  all applicable; no IMU-aware method needed).
- Pose frame: per the DPVO eval script's `PERM = [1,2,0,4,5,3,6]`
  convention, the file is NED + scalar-last quaternion. Confirm
  by reading row 0.

**Acceptance:** report `<seq_root>`, frame count, modality check
(image-only, confirmed).

### Step 1 — Day-1 SOTA method selection + setup (10–20 min)

Per the WebFetch research (2026-05-25): all three candidate pipelines
have a TartanAir entry point. Preference order is **DPVO →
TartanVO → DROID-SLAM**, but selection depends on which actually
installs on this Windows machine.

**Step 1a — Try DPVO first (Branch P from PLAN_03).** RESULT_03
already cloned `external/dpvo/` but only `extractor.py` was usable;
`lietorch` and `altcorr` weren't built. One more attempt:

```powershell
cd external\dpvo
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python setup.py build_ext --inplace
```

`lietorch` on Windows often needs MSVC + CUDA-toolkit alignment. If
it builds → **use DPVO**. If it fails → drop to Step 1b.

**Step 1b — TartanVO fallback** (pure PyTorch, MIT, no custom CUDA
ops). Clone if not already vendored:

```powershell
cd external; git clone https://github.com/castacks/tartanvo
cd tartanvo
.venv\Scripts\pip install -r requirements.txt
```

Download the pretrained TartanVO checkpoint (the repo README
links it). Pure PyTorch — should install without compilation.

**Step 1c — DROID-SLAM** (last resort, heavier setup). Skip unless
1a and 1b both fail.

**Acceptance:** ONE method installed; pretrained weights downloaded;
a one-line sanity command works
(`python <method>/eval.py --help` returns 0).

### Step 2 — Day-1 SOTA reproduction on the hospital sample

Run the **chosen unmodified pipeline** on `<seq_root>`.

**DPVO path** (if Step 1a succeeded):

```powershell
cd external\dpvo
.venv\Scripts\python.exe evaluate_tartan.py `
  --trials 1 `
  --datapath <seq_root> `
  --plot --save_trajectory
```

Their script outputs `<seq>_pred.txt` and `<seq>_gt.txt`; both
trajectory files for `evo` consumption.

**TartanVO path** (if Step 1b succeeded):

```powershell
cd external\tartanvo
.venv\Scripts\python.exe vo_trajectory_from_folder.py `
  --batch-size 1 --worker-num 1 `
  --test-dir <seq_root>\image_left `
  --pose-file <seq_root>\pose_left.txt `
  --model-name tartanvo_1914.pkl
```

(TartanVO's standard runner; see castacks/tartanvo README.)

**Metric computation** — use `evo` (already installed in venv per
PLAN_03 Step 0c retro):

```powershell
.venv\Scripts\python.exe -c "from evo.tools import file_interface; \
  from evo.core.metrics import APE, PoseRelation; \
  from evo.core.sync import associate_trajectories; \
  est = file_interface.read_tum_trajectory_file('pred.txt'); \
  gt  = file_interface.read_tum_trajectory_file('gt.txt'); \
  est, gt = associate_trajectories(gt, est); \
  est.align(gt, correct_scale=True); \
  ape = APE(PoseRelation.translation_part); ape.process_data((gt, est)); \
  print(ape.get_all_statistics())"
```

(Engineer wraps this in `scripts/_eval_tartanair_evo.py` —
underscore-prefix scoped script.)

**Acceptance:** SOTA pipeline's ATE on this sequence reported.
DPVO paper Table 1 reports ~0.21 m on TartanAir validation (Easy
+ Hard) — but those are the MH/ME competition seqs, not hospital
P000 specifically. Use the published number as a **rough sanity
ceiling** (the pipeline should be within an order of magnitude of
its own paper number on the same dataset); don't gate on it
strictly.

### Step 3 — Run `DPVOMotionEncoder` trunk on the same sample

Wrap the existing `src/pipeline/encoders/dpvo_motion.py` for
inference on a folder of images. This is the apples-to-apples
comparison.

The encoder takes pairs of frames `(B, 2, 3, H, W)` and emits
per-pair motion tokens; for end-to-end position MAE we need to
chain pairs into a trajectory. **Two consumption modes**:

- **Mode α — linear-probe trajectory** (cheap, ~5 min): use the
  encoder's existing trained head (from RESULT_03 P-A weights at
  `runs/_weights/dpvo.pth` + the head from
  `runs/overnight/run2_iter_03/`) to emit per-pair absolute
  position predictions; chain to a trajectory; compute ATE with
  same `evo` invocation. **Note this head was trained on Webots, not
  TartanAir** — the result is an *out-of-domain* number, which is
  the right comparison to make (does our encoder transfer?).
- **Mode β — motion-only chain** (more honest): use the encoder's
  per-pair `(dx, dy, dψ)` outputs (if extractable from the
  intermediate head — engineer judges) integrated forward from
  pose[0]. This is closer to a true VO comparison.

Pick Mode α first. If Mode α produces an obviously broken
trajectory (e.g. ATE > 50 m on a < 50-m sequence), document the
domain-shift collapse and skip Mode β.

**Acceptance:** ATE for our encoder on the sample sequence,
computed via the same `evo` invocation.

### Step 4 — Camera per-leg audit upgrade decision

Compare Step 2 (chosen public VO pipeline) vs Step 3 (our encoder):

| pipeline | ATE (m) | params | latency / pair (ms) | source |
|---|---|---|---|---|
| <chosen SOTA> | … | (theirs) | … | public benchmark |
| DPVOMotionEncoder (ours, Mode α) | … | 0.18 M trunk + 0.15 M head | (from RESULT_03: ~11 ms) | RESULT_03 + this iter |
| DPVO paper Table 1 ref (rough) | ~0.21 m on TartanAir avg | — | — | DPVO paper |

**Verdict logic** (paper-strength threshold 30 %):
- Gap ≤ 30 % → **Camera per-leg validation = paper-strength.**
  RESULT_03's `keep with smoothness debt` upgrades to
  `keep with smoothness debt, paper-strength per-leg`.
- Gap > 30 % but ≤ 100 % → **Camera per-leg validation = paper-
  qualifier**. Paper text reads "DPVOMotionEncoder is competitive
  with <SOTA> on Webots; on out-of-domain TartanAir hospital the
  gap is N % (likely domain shift)."
- Gap > 100 % → **paper-soft.** RESULT_03 verdict stays "in-sim
  only"; cite the gap honestly.

**Note**: the Mode α comparison is fundamentally a transfer test
(Webots-trained head on TartanAir input). The fair comparison
would be re-train the head on TartanAir; but doing that defeats the
"is our encoder useful out of the box?" question. Be explicit in
RESULT_08 about this framing.

### Step 5 — Update RESULT_03 + Phase A summary table

Write a small **RESULT_03 addendum** with the upgrade. Update the
Phase A close-out table (RESULT_05's version) with the new C2
column (Camera per-leg validated → yes/no).

Recommend PLAN_09 (Phase B add Camera as 3rd modality) as the
next iteration — both ext-SOTA items (C2 IMU + Camera) are now
addressed, Phase B can proceed unencumbered.

## Sources

- TartanAir: Wang et al., IROS 2020. `https://theairlab.org/tartanair-dataset/`.
- TartanAir tools + data type spec:
  `https://github.com/castacks/tartanair_tools` +
  `https://github.com/castacks/tartanair_tools/blob/master/data_type.md`.
- DPVO: Teed, Lipson, Deng, NeurIPS 2023. arXiv:2208.04726.
  - Repo: https://github.com/princeton-vl/DPVO
  - Eval: `evaluate_tartan.py` (hospital is in `validation` split per
    `dpvo/data_readers/tartan.py`).
- TartanVO: Wang et al., CoRL 2020. arXiv:2011.00359.
  - Repo: https://github.com/castacks/tartanvo (MIT).
  - Runner: `vo_trajectory_from_folder.py`.
- DROID-SLAM: Teed, Deng, NeurIPS 2021. arXiv:2108.10869.
  - Repo: https://github.com/princeton-vl/DROID-SLAM.
- Standard metric: `evo` toolkit — `evo.core.metrics.APE` with
  `pose_relation=translation_part, align=True, correct_scale=True`
  (Sim(3) alignment), matches DPVO Table 1's reported "ATE with
  scale alignment".
- `evo` install: already in venv from PLAN_03 Step 0c retro.
- TartanAir frame convention: NED + scalar-last quaternion;
  DPVO's `PERM = [1,2,0,4,5,3,6]` rearranges to xyz + scalar-first.
- Hospital subset: image-only (NO IMU in TartanAir v1; IMU was
  added in TartanAir V2 / TartanGround only).
- Sample archive: `data/hospital_sample_P000.tar.gz` (present
  2026-05-25 ~19:46 local).
- Vendored DPVO (partial, only extractor): `external/dpvo/`.
- RESULT_03 review note (2026-05-25 ~15:20 local): scientist note
  flagging the need for external-SOTA validation.

## What to report back

In `handoff/results/RESULT_08_camera-ext-sota-tartanair-hospital.md`:

1. **Step 0** — extracted root path, frame count, image-only
   confirmation.
2. **Step 1** — which method installed (DPVO / TartanVO / DROID);
   what failed for the others.
3. **Step 2** — SOTA pipeline ATE on the sequence + per-frame plot.
4. **Step 3** — our DPVOMotionEncoder ATE on the same sequence
   (Mode α; optionally Mode β).
5. **Step 4** — gap table + audit-upgrade label.
6. **Step 5** — RESULT_03 addendum link + updated Phase A summary
   row.
7. **One open question** for scientist.

## Reversibility

- Step 0 (extract): permanent on disk, gitignored.
- Step 1 (clone TartanVO or build DPVO ops): under `external/`,
  gitignored.
- Step 2 (vendored pipeline run): no edits.
- Step 3 (our encoder on TartanAir): throwaway; loads existing
  weights.
- Steps 4–5: documentation.

Files committed: RESULT_08, RESULT_03 addendum, Phase A summary
update, small `scripts/_eval_tartanair_evo.py` wrapper.

**Demand #3** — `external/dpvo` and any other vendored VO repo
NOT edited. Build-system invocations (`pip install -r
requirements.txt`, `setup.py build_ext --inplace`) only.

**Compute budget:** total iteration ≤ 90 min.
- Step 0: 10 min.
- Step 1: 10–30 min (DPVO build attempt + fallback to TartanVO).
- Step 2: 15 min (eval-only on one sequence).
- Step 3: 15 min (our encoder, image folder consumption).
- Step 4: 5 min.
- Step 5: 10 min writeup.

If Step 1 fails for ALL three pipelines: write a partial RESULT
documenting the install obstacles; ship Step 3 as a one-sided
"our encoder on TartanAir hospital, no SOTA reference" number. This
is still progress — establishes our encoder's TartanAir number
which Phase B can refer to.

If the hospital sample turns out to be only depth/seg/flow (no
RGB), document and stop — DPVOMotionEncoder needs RGB.
