# Result 05 — c2-closure-ronin-canonical (partial: blocked + retros)

## TL;DR

**C2 closure is BLOCKED in this iteration.** The canonical RoNIN
unseen-subjects FRDR archive (DOI 10.20383/102.0543) is gated behind
Globus authentication — the FRDR page exposes only a Globus
OAuth login flow and a Globus-routed "Download as Zip" link, neither
scriptable from the engineer venv without interactive registration +
client credentials. A final disk-wide search confirmed no cached
canonical data exists anywhere on this machine. **C2 is therefore
NOT discharged in run 2**; canonical-data acquisition is queued as a
**manual user task / Phase C deferral** with explicit instructions
written into RESULT_02's new addendum.

Step 0's three retros on RESULT_03 (Camera audit) are **all done**:
- **Step 0a (difficulty-matched probe)**: difficulty-normalised
  test-val gap is **+17.5 %** (val wins per-meter by 17.5 %, test is
  actually harder per meter). The `keep` label survives the 20 % gate
  but **at the edge**, not comfortably. RESULT_03 addendum written.
- **Step 0b (smoothness-debt reframe)**: verdict relabelled
  `DPVOMotionEncoder = keep with smoothness debt`; three Phase B
  follow-up candidates named (B-1 auxiliary velocity loss, B-2 EMA
  smoothing, B-3 fusion-transformer absorbs noise).
- **Step 0c (PLAN_06 queue)**: PLAN_06 (Camera external SOTA on
  TartanAir/EuRoC/KITTI) noted as the next-priority queue after this
  C2 closure attempt.

RESULT_02's IMU audit label is **updated** from `keep` to **`keep
(in-domain only)`**, with C2 explicitly deferred to Phase C. Phase A
remains effectively closed (4/4 encoders triaged); Phase B can begin
at PLAN_06 (Camera external SOTA) or PLAN_07 (Phase B bake-off)
depending on the scientist's prioritisation.

## Numbers

### Step-by-step acceptance

| step | acceptance | observed | pass? |
|---|---|---|---|
| 0a. Difficulty-matched probe | per-path table + per-meter gap | ran; per-m gap +17.5 % (val wins per-m); raw gap −7.4 % (test wins on raw). Verdict: keep survives the 20 % gate at the edge. | ✅ |
| 0b. Smoothness-debt reframe | RESULT_03 addendum with relabelled verdict | relabelled `keep with smoothness debt`; 3 Phase-B follow-ups named (B-1 / B-2 / B-3). | ✅ |
| 0c. Queue PLAN_06 | 3-line description in this RESULT | queued — see PLAN_06 section below. | ✅ |
| 1. Acquire canonical RoNIN | data downloaded OR documented obstacle | **BLOCKED** — FRDR is Globus-OAuth-gated (URL: https://auth.globus.org/v2/oauth2/authorize for `auth.globus.org` + transfer scopes). Direct HTTPS fallback also routes through `globus.frdr.ca`. No interactive auth possible from the engineer venv; no canonical data on disk. | ⏭ (blockage documented) |
| 2. ResNet1D on canonical | raw + Umeyama ATE within ±10 % of run-1 5.93 m | **SKIPPED** — Step 1 blocked. | ⏭ |
| 3. IMUCNN on canonical | raw + Umeyama ATE | **SKIPPED** — Step 1 blocked. | ⏭ |
| 4. C2 audit decision | discharged or NOT, with explicit gap %, raw-weighted | **C2 NOT discharged.** RESULT_02 audit label updated to `keep (in-domain only)`. Canonical re-eval queued as manual Phase-C task with explicit step-by-step instructions in RESULT_02 addendum. | ✅ (verdict written) |
| 5. Phase A close-out | updated table | table below, with C2 column populated explicitly | ✅ |

### Step 0a — difficulty-matched probe (RESULT_03 retro detail)

| path | split | length (m) | mean speed (m/s) | mean `|ω|` (rad/s) | n_pairs | MAE (m) | MAE / length |
|---|---|---|---|---|---|---|---|
| 2 | val | 25.83 | 0.313 | 0.147 | 424 | 2.327 | 0.0901 |
| 13 | val | 19.81 | 0.318 | 0.143 | 320 | 0.700 | 0.0354 |
| 14 | val | 25.14 | 0.328 | 0.104 | 395 | 2.272 | 0.0904 |
| 15 | test | 26.41 | 0.315 | 0.089 | 432 | 1.072 | 0.0406 |
| 16 | test | 18.14 | 0.320 | 0.070 | 293 | 1.849 | 0.1019 |
| 17 | test | 17.89 | 0.309 | 0.239 | 297 | 1.985 | 0.1110 |

- val aggregate (per-path-mean): MAE 1.766 m, length 23.59 m, MAE/m **0.0719**.
- test aggregate (per-path-mean): MAE 1.636 m, length 20.82 m, MAE/m **0.0845**.
- Raw test-val gap: **−7.4 %** (per-path-mean basis; the RESULT_03
  main-table frame-weighted basis gave −15.7 %).
- **Difficulty-normalised gap: +17.5 %** — test is 17.5 % harder per
  meter than val. The `keep` verdict survives the 20 % multi-condition
  gate but at the edge.

Path 17 stands out: 5× higher mean |ω| than other paths (0.239 vs
~0.10), MAE/m 0.111 (worst of the six). This is the curve-heavy test
path; the encoder degrades on high-curvature trajectories — a useful
diagnostic for Phase B (the fusion model should compensate via IMU/Odom
on high-omega segments).

Probe script: `scripts/_difficulty_probe_paths.py`. JSON:
`runs/overnight/run2_iter_05/camera_difficulty_probe.json`.

### Step 0b — smoothness-debt reframe (in RESULT_03 addendum)

Verdict: **`DPVOMotionEncoder = keep with smoothness debt`**.

Phase B follow-ups:
- (B-1) Auxiliary velocity loss on the camera head during fusion
  training.
- (B-2) EMA smoothing on per-instant camera tokens before the fusion
  transformer.
- (B-3) Let the fusion transformer absorb noise via temporal
  cross-attention (engineer's RESULT_03 recommendation).

Hard rule going forward: Phase B's bake-off iteration **must** report
per-modality per-trajectory smoothness in every 4-modality test run
so the debt is visible, not silent.

### Step 0c — PLAN_06 queue note

PLAN_06 (Camera external SOTA validation, queued behind this PLAN_05)
will:

1. Pick **one** public visual-odometry benchmark (TartanAir → EuRoC
   → KITTI, in preference order).
2. Pick **one** SOTA pipeline to reproduce unmodified (DPVO if
   `lietorch`/`altcorr` can be installed on Linux container/WSL2 →
   TartanVO MIT pure-PyTorch → DROID-SLAM fallback).
3. Run our `DPVOMotionEncoder` trunk on the SAME public sequence
   (motion-only, no SLAM tracker) and compare per-sample MAE against
   the SOTA's reported ATE on the same sequence.
4. Update RESULT_03's per-leg label using the public-benchmark
   evidence (currently the Webots-only audit gives a `keep` that
   isn't paper-strength for C2-style per-leg validation; PLAN_06 is
   the paper-strength step).

PLAN_06 timing is the scientist's call. If Phase B bake-off (PLAN_07)
gives a clear architecture pick, PLAN_06 can be deferred until after
Phase B without risk to the run-2 paper claim, because Camera shows
up in C3 (4-modality fusion on Webots) — not C1/C2 (per-leg public-
SOTA per modality).

### Step 1 — RoNIN acquisition (blocked)

Probe details:
- **FRDR DOI** (`https://doi.org/10.20383/102.0543`) → redirects to
  `https://www.frdr-dfdr.ca/repo/dataset/816d1e8c-1fc3-47ff-b8ea-a36ff51d682a`.
- Page exposes:
  - **Globus Transfer login** (`https://auth.globus.org/v2/oauth2/authorize`
    with `view_identities + transfer.api.globus.org + groups.api.globus.org`
    scopes) → interactive, not scriptable.
  - **"Download as Zip"** → routed through `globus.frdr.ca` (same
    Globus mechanism).
- No direct HTTPS file links; no `wget`-able URLs.
- `find` across `C:\Users\FabLab` for `*.hdf5` or `FRDR*` directory →
  empty. No cached download.

Conclusion: Step 1 cannot complete inside the autonomous loop.
Acquisition is a **manual user task** (Globus credentials + interactive
transfer); engineer documents the obstacle and skips to Step 4 with
the Branch-Y verdict reaffirmed.

### Step 4 — C2 audit decision

**C2 NOT discharged in run 2.** Branch Y a000 intra-session proxy
remains the only IMU evidence available, and it does not constitute
canonical RoNIN unseen-subjects validation. RESULT_02's audit label
is updated from `keep` to **`keep (in-domain only)`**.

The implication for Phase B (fusion redesign):
- **Default**: IMUCNN remains the IMU encoder. Phase B proceeds with
  the in-domain `keep` verdict.
- **Contingency**: if Phase B fusion training shows IMU-leg
  saturation (e.g., the IMU modality contributing markedly less than
  Anchor2Vec or DPVOMotion in per-modality ablations), revisit and
  swap in RoNIN ResNet1D (vendored, unmodified per Demand #3).

The PerCom paper's C2 claim becomes "competitive with RoNIN ResNet1D
in-domain (a000 intra-session); cross-subject benchmark deferred to
Phase C with the canonical FRDR archive." This is a **softer** claim
than originally targeted — the run-2 paper will need either (a) C2
closure via the manual acquisition task before submission, or (b)
explicit framing of C2 as in-domain.

## Step 5 — Phase A close-out (with PLAN_05 updates)

Updated Phase A summary (extending RESULT_04's table with this
iteration's updates):

| modality | encoder | bench | dataset | best metric | nearest SOTA reference | label (PRE-PLAN_05) | label (POST-PLAN_05) | paper claim status |
|---|---|---|---|---|---|---|---|---|
| WiFi | **Anchor2Vec** | UJI val mean Euclid | UJI canonical val (1 111 samples) | 8.69 m | run-1 8.55 m / eAaT+ 8.16 m | keep | **keep** (unchanged) | **C1 ✓** (paper-defensible per-leg) |
| IMU | **IMUCNN** | Branch Y proxy raw ATE | data/ronin_a000_intra | 3.55 m raw / 0.31 m Umeyama | RoNIN ResNet1D on same proxy 2.89 m raw / 0.32 m Umeyama | keep | **keep (in-domain only)** — relabelled per PLAN_05 Step 4 | **C2 PARTIAL** — in-domain only; canonical FRDR unseen-subjects deferred to manual / Phase C |
| Camera | **DPVOMotion** (P-A) | Webots val/test mean Euclid | data/async_collection [2,13,14]/[15,16,17] | val 1.85 / test 1.56 | CLAUDE.md ACEVision ~3.5 linear-probe; DPVO no direct number | keep | **keep with smoothness debt** — relabelled per PLAN_05 Step 0b; difficulty-normalised gap +17.5 % at edge | **C3 pending fusion** — within-sim transfer marginal; public-benchmark validation queued as PLAN_06 |
| Odom | **OdomCNN** (P-B) | Webots val/test mean Euclid | same Webots split | val 4.62 / test 4.24 | trivial integration floor 8.27 m | keep (P-B) | **keep (P-B)** (unchanged) | **C3 sim-only by design** — no public SOTA exists for this modality |

### Three cross-cutting weaknesses (input to Phase B design)

1. **Per-trajectory smoothness debt** — DPVOMotion (r ≈ 0.07) and
   OdomCNN (r ≈ 0) are absolute-position predictors with poor
   motion-magnitude consistency. Trivial odom integration has near-
   perfect smoothness (r = 0.999). **Phase B's bake-off must report
   per-modality smoothness in every 4-modality test run.**
2. **Cross-session WiFi** — run-1 evidence (MSILN) shows Anchor2Vec
   saturates on real-world cross-session data. C1 is paper-defensible
   on UJI canonical, but C4 (cross-session real-world plausibility on
   Microsoft ILN 2.0) remains a Phase B/C question.
3. **C2 cross-subject IMU** — undischarged due to FRDR Globus gate.
   Manual user task / Phase C deferral.

### Open question for the scientist (Phase B kickoff)

The SCIENTIST_BRIEF roadmap names four Phase B fusion candidates:
- **Set-transformer** (current direction, run-1 default).
- **TCN-based temporal fusion** (small, fast, robust).
- **LSTM-with-attention hybrid** (strong for variable-length async).
- **Late fusion + learned modality gate** (directly fixes "IMU
  injects noise" run-1 weakness AND would let trivial-integration
  odom flow through unmolested for smoothness).

**My read**: late fusion + learned modality gate is the strongest
fit for run-2's findings:
- It addresses the smoothness debt (gates can preferentially trust
  trivial-integration odom on smooth segments).
- It addresses the IMU-domain-shift concern (gate can attenuate IMU
  when cross-subject mismatch is detected — Phase B's IMU branch
  could even be a small ensemble of IMUCNN + ResNet1D once C2 is
  closed).
- It keeps each per-leg encoder's audit verdict honest at the
  architecture level — no implicit "fusion masks weak encoders"
  effect that would be untraceable in a 4-modality run.

But the bake-off should still run the four candidates on 10 % Webots
subset to validate. **Scientist call.**

## What was changed

- `scripts/_difficulty_probe_paths.py` — **new**. Step 0a probe;
  computes per-path difficulty features + difficulty-normalised
  test-val gap; reads `data/async_collection/path_*/ground_truth.csv`
  and the RESULT_03 P-A per-path JSON.
- `handoff/results/RESULT_02_imu-encoder-audit-ronin.md` — addendum
  appended (PLAN_05 Step 4 — audit label updated to `keep (in-domain
  only)` with C2 deferral instructions).
- `handoff/results/RESULT_03_camera-encoder-audit-webots.md` —
  addendum appended (PLAN_05 Step 0a/0b/0c — difficulty-matched
  probe, smoothness-debt relabel, PLAN_06 queue).
- `runs/overnight/run2_iter_05/camera_difficulty_probe.json` —
  difficulty-probe output (gitignored).
- `runs/overnight/run2_iter_05/difficulty_probe.log` — console log.

## What was reverted

Nothing.

## Logs

All under `runs/overnight/run2_iter_05/`:
- `difficulty_probe.log` — Step 0a probe console output.
- `camera_difficulty_probe.json` — Step 0a probe JSON.

No new training logs (Steps 1–3 were blocked / skipped).

## Demand #3 specifics

No vendored sources touched. FRDR-Globus auth was probed via a public
HTML fetch (no credentials, no API calls), and the obstacle is
documented; no automation written.

## Cycle-rules compliance

- ✅ Step 0a probe ran; per-path / per-meter table reported.
- ✅ Memory budget not relevant (no new training).
- ⚠ Day-1 SOTA reproduction (Step 2) BLOCKED by acquisition gate;
  documented per blockage clause in PLAN_05.
- ✅ Per-path distribution + per-trajectory features reported (Step
  0a).
- ✅ Multi-condition + preprocessing + raw-weighted alignment in
  retro-style addenda.
- ✅ No silent stalls; the FRDR blockage was probed, documented, and
  C2 was explicitly relabelled rather than hand-waved.
- ✅ Iteration well inside 90-min budget (~30 min wall clock).

## Stop conditions

- Local time at write: **Mon May 25 ~16:05 local** (inside STATE
  Stop-at 2026-05-26 18:00).
- No `handoff/STOP` file.
- `GOAL_REACHED: false` — Phase A close-out with C2 relabelled;
  Phase B can begin at PLAN_06 (Camera external SOTA, queued) or
  PLAN_07 (fusion bake-off, scientist's call).
