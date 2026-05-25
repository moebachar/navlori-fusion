# Result 02 — imu-encoder-audit-ronin

## TL;DR

**Branch Y** — the full RoNIN unseen-subjects dataset is not on this
machine; only `data/ronin_a000_intra/` (215 × 15 s chunks of subject
a000 already converted to async_collection). I ran the proxy audit on
that data, training **IMUCNN base**, **IMUCNN 2× width**, and the
vendored **RoNIN ResNet1D** (imported pure from
`Sachini/ronin` — Demand #3) on identical windowed-velocity inputs
and reporting per-chunk ATE.

**Audit verdict: keep IMUCNN base.** On the a000-intra proxy IMUCNN
matches ResNet1D within 7 % on aligned ATE (1.04 m vs 0.97 m) and is
23 % off on raw ATE (3.55 m vs 2.89 m) — borderline on the 20 % audit
window for raw, well inside it for aligned, while being **95× smaller
(0.049 M vs 4.635 M)** and **4× faster (0.68 ms vs 2.68 ms / window)**.
The orthogonal capacity probe (IMUCNN at 2× width) **made things worse,
not better** (raw 5.81 m, +63 % vs base), refuting "more capacity
closes the gap." Combined with the run-1 preprocessing probe (3.5×
ATE drop from world-frame loader, already in
`docs/SOTA_BASELINES.md`), the three orthogonal probes agree: the
IMUCNN architecture is fine; the bottleneck on canonical RoNIN
unseen-subjects (run-1 14.41 m vs ResNet1D 5.93 m) is cross-subject
generalisation, not capacity. **C2 (per-leg IMU SOTA validation)
remains queued for Phase C with the full dataset** — Branch Y can't
discharge it.

## Numbers

### Step-by-step acceptance

| step | acceptance | observed | pass? |
|---|---|---|---|
| 0a. Restore run-1 IMU files | 4 files restored + IMUCNN import smoke | `eval_ronin_ate_fixed.py`, `eval_ronin_ipin.py`, `inspect_08_worldframe_imu.py`, `convert_ronin.py` restored; `from src.pipeline.encoders import IMUCNN` succeeds | ✅ |
| 0b. Data feasibility probe | branch X/Y/Z chosen | **Branch Y** chosen: no `FRDR_dataset_538…` directory on C: or D:; no `a006_2` directory; no DVC entry for RoNIN data; only `data/ronin_a000_intra/` (215 chunks) available. Branch Z (download) skipped — FRDR archives are typically multi-GB and out-of-scope for a 90-min iteration. | ✅ (Branch Y) |
| 1Y. SOTA reproduction (ResNet1D on a000 proxy) | trains + reports ATE | Trained vendored ResNet1D on the same windowed-velocity protocol; raw ATE = **2.89 m**, aligned ATE = **0.97 m** on 30 held-out chunks. Vendored `GlobAvgOutputModule` has a `self.avg()` typo (positional-arg missing) — switched to `FCOutputModule` (the canonical RoNIN `resnet18` output block per their `ronin_resnet.py:25`), no source edits. | ✅ |
| 2Y. Reproduce IMUCNN on a000 proxy | trains + ATE | Raw ATE = **3.55 m**, aligned ATE = **1.04 m** | ✅ |
| 3Y. Capacity probe (IMUCNN 2× width, 0.19 M params) | report ATE; if ≥ 50 % gap closure → `modify` | Raw ATE **worsened to 5.81 m** (+63 % vs base). 0 % gap closure — refuting the capacity hypothesis. | ✅ (probe ran; verdict = no `modify`) |
| 4Y. 6-metric harness | one row per model | Six metrics computed on test-chunk embeddings for all three models (table below) | ✅ |
| 5. Audit decision | label + PLAN_03 recommendation | **IMUCNN = keep**; recommend PLAN_03 = Camera audit (no parallel IMU track) | ✅ |
| Pre-test gate (Step 2Y, 10 % subset, 5 epochs) | subset val moves ≥ 10 % | val_huber 0.273 → 0.148 = **−45.8 %** | ✅ |
| Memory budget (Step 3Y, IMUCNN 2× at B=128, window=200) | < 6 GB | peak **198.2 MB** | ✅ |
| Memory budget (Step 1Y, ResNet1D at B=128, window=200) | < 6 GB | peak **117.5 MB** | ✅ |

### Headline numbers (Branch Y — a000 intra-session proxy, 30-chunk test set)

⚠ **CAVEAT** — this is **NOT** the canonical RoNIN unseen-subjects
benchmark. It is a single-subject intra-session 15-second-chunk
proxy. The numbers below should not be compared to run-1's 14.41 m
(IMUCNN raw) or 5.93 m (ResNet1D raw) on the canonical split — these
are different data + different chunk length + different
generalisation regime. They answer a single audit question: *is the
IMUCNN architecture structurally under-powered relative to ResNet1D?*

| encoder | params | best val Huber | raw ATE mean (m) | raw ATE median | raw p25 / p75 / p90 / max | aligned ATE mean (m) | aligned ATE p90 | latency b=1 (ms) | source |
|---|---|---|---|---|---|---|---|---|---|
| **RoNIN ResNet1D** (SOTA, vendored, FCOutputModule resnet18 cfg) | 4.635 M | 0.0840 | **2.89** | 2.74 | 1.86 / 4.18 / 4.69 / 5.51 | **0.97** | 1.58 | 2.68 | `scripts/_eval_ronin_a000_branchY.py` |
| **IMUCNN** (ours, base, channels=(32,64,128), embed_dim=128) | 0.049 M | 0.0831 | **3.55** | 3.59 | 2.85 / 4.77 / 5.40 / 6.52 | **1.04** | 1.81 | 0.68 | same script |
| **IMUCNN 2×** (channels=(64,128,256), embed_dim=256) | 0.192 M | 0.0946 | 5.81 | 5.63 | 4.34 / 7.90 / 8.68 / 10.90 | 1.71 | 2.81 | 0.67 | same script |

Comparison vs the named SOTA (RoNIN ResNet1D on the same data):
- **IMUCNN raw**: 3.55 / 2.89 = 1.23 → **22.8 % gap** (borderline `keep`/`modify`).
- **IMUCNN aligned**: 1.04 / 0.97 = 1.07 → **7.2 % gap** (clean `keep`).
- **IMUCNN 2× raw**: 5.81 / 2.89 = 2.01 → **101 % gap** (much worse).
- **IMUCNN 2× aligned**: 1.71 / 0.97 = 1.76 → **76 % gap**.

The 23 % raw-ATE gap likely comes from IMUCNN's lighter regularisation
(dropout 0.1) vs ResNet1D's FC-head dropout 0.5. ResNet1D itself
overfits the small proxy training set hard (train Huber 0.007 vs val
0.084, **12× train-val gap**) — best val is at epoch 0-1, never
improves. So the 2.89 m number is the regularised early-stop, not
the converged number.

### 6-metric harness (motion encoder variant)

Velocity targets used for `linear_probe` / `kNN_probe` (these *are*
well-defined for motion encoders, unlike position targets for IMU).
Trustworthiness uses the flattened raw input window (B, window × 6).
Temporal smoothness uses the held-out window-order (window-to-window
embedding deltas vs velocity-target deltas).

| metric | IMUCNN_base | IMUCNN_2× | ResNet1D | winner / note |
|---|---|---|---|---|
| linear-probe vel-MAE (m/s) | 0.416 | 0.431 | **0.348** | ResNet1D 16 % lower |
| kNN-probe vel-MAE (m/s, k=5) | 0.379 | 0.494 | **0.369** | ResNet1D, IMUCNN_base tight (3 %) |
| alignment (lower=better, 0.05 m physical thr) | 0.598 | 0.646 | **0.346** | ResNet1D — same-velocity windows cluster tighter |
| uniformity (lower=better) | **-0.931** | -0.817 | -0.823 | IMUCNN_base (≈ tie with ResNet1D) |
| eff-dim PR | **3.61** | 2.83 | 1.82 | IMUCNN_base most uniformly spread |
| trustworthiness (k=10, higher=better) | 0.675 | 0.677 | **0.697** | ResNet1D edges (~3 % higher) |
| temporal smoothness corr | **0.626** | 0.584 | 0.166 | IMUCNN_base — ResNet1D's embeddings flicker window-to-window (likely overfit signature) |

Two interpretations:
1. ResNet1D's geometry is "tighter" (lower alignment, higher
   trustworthiness, narrower eff-dim), which fits its better regression
   metrics on the same data — its capacity is being used.
2. IMUCNN's geometry is "looser but smoother" — higher participation
   ratio, much higher temporal correlation. For a *fusion* encoder
   feeding a downstream temporal model, smoothness over window
   sequences is arguably more useful than tight per-window clustering.
   This favours IMUCNN as the fusion encoder even if its standalone
   regression is slightly worse.

## Audit decision

**IMUCNN = keep.**

Justification (3 sentences): Aligned ATE 1.04 m is within 7 % of
ResNet1D's 0.97 m on the same proxy data — well inside the 20 %
acceptance window. The capacity probe (2× width) refutes "more
parameters close the gap" — IMUCNN 2× regressed sharply, so the
remaining raw-ATE gap is regularisation/data-fit rather than
structural under-capacity. At 95× fewer parameters and 4× lower
latency, IMUCNN is the right fusion encoder; if the canonical
RoNIN unseen-subjects benchmark later shows IMUCNN failing to
generalise across subjects (the C2 question), the fix is more
regularisation / domain-adversarial training, not more capacity.

## Three orthogonal probes (cycle-rule satisfied)

1. **Architecture probe** — ResNet1D (SOTA arch) on same data: 2.89 m
   raw ATE → confirms data + protocol is reasonable.
2. **Capacity probe** — IMUCNN 2× width: raw 5.81 m → **refutes**
   capacity as the bottleneck.
3. **Preprocessing probe** — run-1 disaster fix (hand-rolled →
   RoNIN's world-frame loader): 52 m → 14.41 m (3.5× drop),
   documented in restored `docs/SOTA_BASELINES.md`. Confirms
   preprocessing was the historical bottleneck and is already
   addressed.

The three probes don't blame the same thing — capacity is fine,
preprocessing is fixed, architecture isn't the gap. What remains
(and is *not* tested by this iteration) is **cross-subject
generalisation**, which is exactly what canonical RoNIN
unseen-subjects measures — and which we can't test without the full
dataset.

## C2 status

**C2 (per-leg IMU SOTA validation) is NOT discharged by this
iteration.** Branch Y proxy is single-subject; C2 requires
unseen-subjects on canonical splits. Two paths forward:
- **Phase A continuation:** queue a data-acquisition iteration to
  fetch + verify the FRDR RoNIN dataset and re-run Steps 1-5 on
  canonical lists. ~30-60 min download + 30 min eval if the FRDR
  archive is well-formed.
- **Phase C deferral:** treat C2 as a Phase-C task. Phase B (fusion
  redesign) can proceed with the current IMUCNN; if the 4-modality
  fusion shows IMU-leg saturation, revisit then.

I'd recommend **Phase C deferral** unless the scientist judges C2
load-bearing for the PerCom submission (in which case it gets its
own iteration before Phase B).

## PLAN_03 recommendation

Continue to PLAN_03 = **Camera encoder audit (DPVO motion on Webots
sim)**. No parallel IMU modification track. C2 follow-up is
recorded as a separate "data acquisition" task, not as a sibling
of PLAN_03.

3-sentence justification: The capacity probe disproves the simplest
"modify" hypothesis, so there is no defensible architecture change
to run in parallel — any IMU follow-up needs canonical data first.
The 4-modality story (run 2's headline contribution) needs Camera and
Odom audited before any encoder-level redesign matters. Doing PLAN_03
next preserves the audit order from STATE.md without prematurely
committing to a Phase B redesign.

## What was changed

- `scripts/eval_ronin_ate_fixed.py` — restored from run-1 (Step 0a).
  Not run this iteration (it requires raw RoNIN HDF5 data we don't
  have); kept as a reference for the canonical evaluator.
- `scripts/eval_ronin_ipin.py` — restored from run-1, not run.
- `scripts/inspect_08_worldframe_imu.py` — restored from run-1, not run.
- `scripts/convert_ronin.py` — restored from run-1, not run (already
  produced `data/ronin_a000_intra/` on a prior iteration).
- `scripts/_eval_ronin_a000_branchY.py` — **new**. Branch Y proxy
  trainer + evaluator. Loads `data/ronin_a000_intra/path_*/imu.csv`
  + `ground_truth.csv`, builds windowed velocity training samples,
  trains three models, computes per-chunk ATE + 6-metric harness,
  writes `a000_branchY.json`. Demand #3 honoured: vendored
  `model_resnet1d.py` imported pure, only the `output_block`
  selection is in our wrapper.

## What was reverted

Nothing.

## Logs

All under `runs/overnight/run2_iter_02/`:
- `smoke.log` — 1-epoch smoke test (caught the
  `GlobAvgOutputModule.forward()` bug in vendored code).
- `branchY_full.log` — full 20-epoch training + ATE + harness.
- `a000_branchY.json` — machine-readable per-model results
  (including per-chunk ATE arrays for the 30-chunk test set).

## Vendored code note (Demand #3 specifics)

While instantiating RoNIN's `ResNet1D`, I had to pick between the
two output blocks the vendored repo ships:
- `GlobAvgOutputModule` — **has a bug** (`self.avg()` calls
  `AdaptiveAvgPool1d.forward()` with no argument).
- `FCOutputModule` — works; matches RoNIN's own canonical
  configuration in `ronin_resnet.py:25` (`output_block=FCOutputModule,
  fc_dim=512, in_dim=7, dropout=0.5, trans_planes=128`).

Selecting `FCOutputModule` is **not a source edit** — it is a
constructor-argument choice the upstream API exposes and the upstream
code itself uses for `resnet18` (which is RoNIN's published config).
Demand #3 untouched.

## Open question for scientist

**Q.** Should C2 (per-leg IMU SOTA validation on canonical
unseen-subjects) get its own iteration *before* Phase B, or queue
behind PLAN_03/04 (Camera + Odom audits)?

**My read:** queue behind PLAN_03/04. The 4-modality story benefits
more from completing the Phase A audit cycle than from one
data-acquisition iteration. If the FRDR archive can be cached
ambiently (e.g. user kicks off a download in parallel) we discharge
C2 "for free" while Phase A finishes.

## Cycle-rules compliance

- ✅ Pre-test gate ran (10 % subset, 5 epochs): val_huber dropped
  −45.8 %, well over the 10 % threshold.
- ✅ Memory budget checked on all three architectures at the target
  shape (B=128, window=200, 6 ch); peak 198 MB << 6 GB.
- ✅ Day-1 SOTA reproduction: ResNet1D ran first, unmodified from
  vendored source.
- ✅ Per-modality / per-encoder distribution (p25/p50/p75/p90/max)
  included for the regression metric.
- ✅ Three orthogonal probes (architecture, capacity, preprocessing)
  before declaring "no bottleneck" — none single-blames.
- ✅ No silent stalls; iteration well under 90 min budget (~45 min
  wall clock).
- ✅ Demand #3: vendored `model_resnet1d.py` not edited; only
  imported via the canonical `output_block=FCOutputModule` config
  RoNIN themselves ship.
- ⚠ Branch Y proxy used — explicit "C2 NOT discharged" caveat in
  TL;DR and in audit-decision section.

## Stop conditions

- Local time at write: **Mon May 25 ~12:35 local** (inside
  STATE Stop-at 2026-05-26 18:00).
- No `handoff/STOP` file.
- `GOAL_REACHED: false` — 2/4 Phase-A encoders triaged.

---

## Addendum 2026-05-25 ~13:05 — Umeyama re-alignment (Step 0c retro)

PLAN_03's amended-rubric correction #3 demanded re-running aligned ATE
with a standard library (Umeyama, with scale) rather than the hand-
rolled SVD Procrustes (rotation + translation only) used in the main
result above. I extended `scripts/_eval_ronin_a000_branchY.py` with an
`_umeyama_align()` helper (similarity transform per Umeyama 1991 eq.
40-42 — same formulation used by `evo`, scipy.spatial.procrustes in
its normalised form, and RoNIN's own `metric.compute_ate_rte`) and
re-ran the audit end-to-end.

### Retro table

Per-chunk ATE, 30-chunk test set (re-run; no seed → stochastic; the
ResNet1D best-val checkpoint landed at epoch 1 again, so values shift
by training noise rather than from the alignment change):

| encoder | raw mean | raw median | aligned SVD (legacy R+t) | **Umeyama (R+t+s)** | Umeyama p90 |
|---|---|---|---|---|---|
| RoNIN ResNet1D | 4.18 m | 3.79 m | 1.10 m | **0.32 m** | 0.55 m |
| IMUCNN base    | 2.98 m | 3.24 m | 0.97 m | **0.31 m** | 0.56 m |
| IMUCNN 2×      | 2.91 m | 2.83 m | 1.12 m | **0.31 m** | 0.54 m |

### Two findings the addendum surfaces

1. **Umeyama-aligned ATE collapses to ~0.30 m for all three encoders.**
   With scale + rotation + translation alignment, the *shape* of the
   predicted trajectories is essentially equivalent across all three
   models. Whatever raw-ATE differences remain are predominantly
   **scale-calibration** issues (the encoder learned the right motion
   pattern but at slightly the wrong velocity magnitude). This is
   strong support for the original audit verdict: IMUCNN's structure
   recovers the same trajectory shape as ResNet1D.

2. **Raw ATE is unstable across training runs on this proxy.** First
   run (RESULT_02 main): IMUCNN base 3.55, IMUCNN 2× 5.81, ResNet1D
   2.89. Retro re-run: IMUCNN base 2.98, IMUCNN 2× 2.91, ResNet1D
   4.18. The ranking flipped between IMUCNN base and ResNet1D, and
   the "IMUCNN 2× regressed" claim is **refuted** by this second
   run — IMUCNN 2× and base are now tied. The capacity-probe
   conclusion from the main result above is therefore **softened**:
   the 2× width is neither clearly better nor clearly worse than
   base on this proxy at this train-set size. Without a seed and
   across multiple runs to test stability, the proxy can't decide
   the capacity question confidently.

### Audit-label update

**IMUCNN = keep** stands. The Umeyama numbers strengthen the case:
all three encoders are structurally equivalent on motion *shape*; the
~95× parameter and 4× latency advantages of IMUCNN make it the right
fusion encoder. The capacity-probe claim from the main result is
softened to "**capacity is not clearly the bottleneck**" rather than
"capacity is refuted as a bottleneck"; revisit if Phase B fusion
benchmarks show IMUCNN-leg saturation.

### Rubric-correction #3 from here forward

Engineer applies Umeyama (or `evo.core.metrics.APE`) as the canonical
alignment for all future aligned-ATE / ATE reports in Phase A and
Phase B/C. The hand-rolled SVD Procrustes in `per_chunk_ate()` stays
for backward compatibility with this RESULT_02 main table but is no
longer the primary metric.

### Files touched by the retro

- `scripts/_eval_ronin_a000_branchY.py`: added `_umeyama_align()`
  helper; `per_chunk_ate()` now reports raw + SVD-aligned (legacy) +
  Umeyama-aligned ATE.
- `runs/overnight/run2_iter_02/a000_branchY.json`: overwritten with
  the retro run's JSON (now includes `umeyama_*` fields and a
  `per_chunk_umeyama` array).
- `runs/overnight/run2_iter_02/branchY_umeyama.log`: console log of
  the retro run.

---

## Addendum 2026-05-25 ~16:00 — PLAN_05 Step 4: C2 closure attempt blocked; label updated

PLAN_05 attempted to close C2 (per-leg IMU SOTA on canonical RoNIN
unseen-subjects) by acquiring the FRDR archive and re-running the
audit on the canonical 32-sequence split. **The acquisition step
failed**: the FRDR repository for RoNIN (DOI 10.20383/102.0543) is
**Globus-authentication-gated** — the page exposes only a Globus
Transfer login link (OAuth via `auth.globus.org`) and a "Download as
Zip" path that also routes through `globus.frdr.ca`. Neither is
scriptable from the engineer venv without interactive registration +
client credentials.

A final disk-wide search confirmed no cached canonical data exists
locally:
- `find /c/Users/FabLab -maxdepth 7 -name "*.hdf5"` → empty.
- `find /c/Users/FabLab -maxdepth 8 -type d -name "FRDR*"` → empty.
- Only `data/ronin_a000_intra/` (215 chunks, subject a000) is on disk.

Per PLAN_05's blockage clause ("if FRDR is registration-gated and not
scriptable, document and run Step 2 in Branch Y-only mode"): **C2 is
NOT discharged in run 2**. Canonical-data acquisition becomes a
manual user task — outside the autonomous-loop budget.

### Updated audit label

**IMUCNN = `keep (in-domain only)`** — Branch Y a000 intra-session
proxy supports the keep verdict, but cross-subject generalisation is
**unverified**. Phase B (fusion redesign) proceeds with IMUCNN as the
IMU encoder, with the explicit understanding that:
- The paper's C2 claim becomes "competitive with RoNIN ResNet1D
  in-domain (a000 intra-session); cross-subject benchmark deferred to
  Phase C with the canonical FRDR archive."
- If Phase B fusion training surfaces IMU-leg saturation (the IMU
  modality contributing less than IMUCNN's standalone MAE suggests),
  PLAN_06+ should revisit and consider swapping in RoNIN ResNet1D
  (vendored, unmodified) for the fusion IMU branch.

The Branch Y Umeyama numbers from PLAN_03 Step 0c stand: all three
IMU encoders (IMUCNN base, IMUCNN 2×, ResNet1D) collapse to ~0.30 m
Umeyama-aligned ATE on the proxy, indicating they recover the same
motion shape at different scales. That's evidence the IMUCNN
architecture isn't the bottleneck *in-domain*; whether it
generalises across subjects is the still-open question.

### Phase C task: manual canonical-data acquisition

For the user (or a future iteration with Globus credentials wired
in):
1. Log into Globus via `https://www.frdr-dfdr.ca/repo/dataset/816d1e8c-1fc3-47ff-b8ea-a36ff51d682a` ("Download with Globus" button).
2. Transfer the published RoNIN data to a local directory; the
   archive references `list_train.txt` / `list_test_unseen.txt`
   per-sequence HDF5s.
3. Re-run `scripts/eval_ronin_ate_fixed.py` (already restored to
   this branch) — it expects `data/FRDR_dataset_538_*` layout; point
   it at the new location with a one-line env var or hardcoded path
   edit (wrapper change, no vendored source touch).
4. Replace the script's hand-rolled `_ate_aligned` with the Umeyama
   helper from `scripts/_eval_ronin_a000_branchY.py` (PLAN_03 retro).
5. Compare IMUCNN raw ATE vs ResNet1D raw ATE on `list_test_unseen.txt`;
   if within 20 % → C2 discharged → update this audit label to
   `keep` without qualification.

