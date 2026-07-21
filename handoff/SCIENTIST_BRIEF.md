# Scientist Brief — NavLoRI-Fusion (run 2, 2026-05-25)

You are the **research strategist** for an indoor-localization PhD project.
This brief is the contract for **run 2** of the overnight scientist+engineer
loop. Run 1 is archived under `handoff/archive/run1/` — read its `README.md`
if you want the history, but do not let its conclusions anchor you.

---

## 1. Project in 30 seconds

- **Author:** Mohamed Bachar, PhD, CESI LINEACT.
- **System:** robot indoor localization via fusion of **4 modalities** —
  WiFi RSSI (1 Hz, absolute), IMU (~31 Hz, motion), Odometry (~15 Hz, motion),
  Camera (~5 Hz, visual). All four exist together only in the Webots
  simulation (TIAGO++ robot). Real-world datasets are typically 2-modality
  (smartphone WiFi+IMU).
- **Encoders** (committed under `src/pipeline/encoders/`):
  WiFi → `WiFi-Net`; WiFi alt → `WiFiSetTransformer` (built in run 1);
  IMU → `IMUCNN`; Odom → `OdomCNN`; Camera → `DPVOMotionEncoder` (DPVO trunk
  + head).
- **Fusion** (committed under `src/pipeline/fusion/`): single set-transformer
  with self-attention for cross-modal + cross-time fusion, cross-attention
  `PositionQuery(τ)` readout. Implementation was rushed in run 1 — open
  for redesign (transformer, TCN, LSTM-attention, late+gating are all on
  the table; transformer is letterature-preferred but not mandatory).
- **Repo:** `https://github.com/moebachar/navlori-fusion`. Branch for this
  run will be a fresh one off `main` — see STATE.md.

---

## 2. The publishable contribution this run targets

> **A 4-modality fusion architecture for indoor localization that gracefully
> handles missing/async modalities and matches per-leg published SOTA on
> each modality's canonical benchmark.**

This contribution has 4 supporting claims, each of which must be backed
by experiments before paper submission:

| claim | how to prove it | dataset |
|---|---|---|
| C1 — Our WiFi encoder is competitive with WiFi SOTA | reproduce CNNLoc + Locaris numbers; show our encoder within ~20 % | UJIIndoorLoc |
| C2 — Our IMU encoder is competitive with IMU SOTA | reproduce RoNIN ResNet18 ATE; show our encoder within ~20 % | RoNIN unseen-subjects |
| C3 — The 4-modality fusion architecture works end-to-end | full pipeline trained + evaluated on the only 4-modality dataset | Webots sim |
| C4 — The architecture degrades gracefully to 2 modalities on real data | run the architecture as WiFi+IMU subset with modality_dropout | Microsoft ILN 2.0 site1/B1 (cross-session) |

C1 and C2 are validation-grade ("our piece is not worse than the SOTA");
C3 is the novelty headline ("look what fusing all 4 gets you"); C4 is
the real-world plausibility check.

**Target venue:** PerCom 2026 (deadline ~11 Sept 2026); MDPI Sensors /
IEEE Sensors Journal as rolling fallbacks. IPIN 2026 deadline already
passed (10 May 2026); IPIN 2027 is the natural follow-up.

---

## 3. Acceptance criteria (the goal in numbers)

Stay locked. These are the bars; an iteration is judged against them.

(a) **Per-leg validation** — for each of WiFi and IMU:
    our encoder's published-protocol MAE is within **20 %** of the
    SOTA repo's number on the same dataset and same metric.

(b) **4-modality fusion on Webots sim** — full-fusion test MAE
    **≤ 0.5 m** (the existing run 1 baseline was ≈ 0.43 m, so this
    isn't a stretch; the bar exists so the architecture isn't broken).
    Per-modality subset eval reported.

(c) **Cross-session real-world plausibility** — on Microsoft ILN 2.0
    site1/B1, beat WiFi-kNN baseline by ≥ 1.5 m AND beat the open-source
    SOTA (CNNLoc or Locaris) by ≥ 0.5 m on the SAME data + metric. The
    1-3 m absolute bar from run 1 is **dropped** for this dataset — the
    physical achievable depends on AP density per session and was not
    quantified honestly in run 1.

(d) **Per-path distribution + per-trajectory smoothness** reported for
    every evaluation, not just aggregate mean.

(e) **Real-time** — < 100 ms per sample on the project GPU (Quadro
    P4000, 8 GB). Already met by the run-1 architecture; protect this
    if you change fusion.

---

## 4. Run-2 strategy — the iteration ordering

This is your starting roadmap, but you own the strategy and can shift
as evidence demands. Each iteration follows the new cycle rules in
`PROTOCOL.md` (small-subset pre-test, memory budget check, SOTA-baseline
day-1 rule).

### Phase A — Encoder audit (PLAN_01 → PLAN_04, ~1 iter each)

One encoder per iteration. For each:
1. Clone the SOTA repo (or use already-vendored at
   `C:\Users\FabLab\AppData\Local\Temp\`).
2. Reproduce its published number on its native benchmark.
3. Run our encoder on the SAME data, SAME metric, SAME protocol.
4. Compute the 6-metric harness (linear probe, kNN, alignment,
   uniformity, eff. dim, trustworthiness, temporal smoothness — already
   in `src/pipeline/evaluation/encoder_eval.py`).
5. Decision: **keep** (within 20 % of SOTA), **modify** (close but
   identified bottleneck), **replace** (gap > 20 %, name the alternative).

**Order:** WiFi (PLAN_01) → IMU (PLAN_02) → Camera (PLAN_03) → Odom (PLAN_04).
Odom has no public SOTA; the audit is internal (kNN, linear probe).

| iter | encoder | SOTA repo | benchmark |
|---|---|---|---|
| 01 | WiFi (`WiFi-Net` and/or `WiFiSetTransformer`) | `sharan-naribole/wlan_localization` + `Sachini/niloc` (Locaris) | UJIIndoorLoc |
| 02 | IMU (`IMUCNN`) | `Sachini/ronin` (ResNet1D) | RoNIN unseen-subjects |
| 03 | Camera (`DPVOMotionEncoder`) | DPVO published numbers | Webots sim (no public real-data fits) |
| 04 | Odom (`OdomCNN`) | — (internal: kNN, linear probe) | Webots sim |

### Phase B — Fusion redesign (PLAN_05 → PLAN_07ish)

After Phase A, you know which encoders to keep / modify / replace. Phase
B redesigns the fusion stack. Candidates:

- **Set-transformer** (current direction, fix run-1 issues: memory,
  IMU-noise-injection at higher dim) — letterature-preferred.
- **TCN-based temporal fusion** — small, fast, robust; less novel.
- **LSTM-with-attention hybrid** — strong for variable-length async.
- **Late fusion + learned modality gate** — directly fixes the
  "IMU injects noise" problem from run 1.

Plan a small bake-off (1 iter per candidate, on 10 % subset of Webots
sim) before committing the full training budget to one architecture.

### Phase C — Validation + ablations (PLAN_08+)

- Full 4-modality fusion on Webots sim (C3).
- Per-modality subset eval (`only:X`, `drop:X`).
- Cross-session real-world subset on Microsoft ILN 2.0 (C4).
- Per-path distribution + per-trajectory plots + latency.
- Conformal coverage on val/test.

---

## 5. What run 1 produced that is still useful

- `scripts/convert_msiln.py` + `data/msiln_site1_b1/` (untracked) +
  `configs/data/msiln_site1_b1.yaml` — the Microsoft ILN 2.0 integration.
  Cross-session day-based split already written (`split.json`).
- `src/pipeline/encoders/wifi_set.py` — `WiFiSetTransformer` (sparse-
  observed forward after iter_06; included in the encoder audit).
- `runs/baselines/msiln_site1_b1/` — trivial baselines (centroid /
  WiFi-kNN / IMU Kalman). Per-path distributions + waypoint metric
  validated.
- `runs/fusion_*/test_paths/*.png` — per-trajectory plots template.
- `src/pipeline/evaluation/encoder_eval.py` — the 6-metric harness.

---

## 6. Operational reminders (unchanged from run 1)

- **Demand #3:** baseline SOTA from open-source code unmodified. Shims
  (`np.int = int`, `importlib` workarounds for broken `__init__` chains)
  live in OUR wrapper scripts, never in vendored sources. Already-vendored
  repos at `C:\Users\FabLab\AppData\Local\Temp\`:
  `wlan_localization\` (MIT, sharan-naribole),
  `ronin\` (MIT, Sachini),
  `msiln20\` (location-competition starter, ships 2.1 GB of real data).
- **Windows + project venv only.** No WSL/bash scripts. Always use
  `.venv\Scripts\python.exe`.
- **Hardware:** Quadro P4000 8 GB, PyTorch < 2.7 (Pascal sm_61).
- **No `git push`** — engineer's token is denied. User pushes manually.

---

## 7. What's different about this run

- **SOTA baselines day-1.** Every new benchmark, the named SOTA repo is
  cloned and reproduced FIRST. No method comparisons until baseline
  numbers are in.
- **Small-subset pre-test gate.** Every training iteration runs on 10 %
  data / 5 epochs first. Full training only if the small-scale signal
  is clear.
- **Memory budget check.** Every new architecture proves it fits in
  6 GB on synthetic forward+backward before launching training.
- **Per-modality subset eval mandatory** in every RESULT.
- **Per-path distribution + per-trajectory smoothness** in every RESULT.
- **No silent stalls.** Engineer writes partial RESULT within 15 min of
  any blockage. Scientist writes override note if engineer silent > 60 min.
- **No anchoring on a single ablation as "the bottleneck"** — every
  bottleneck claim needs 3 orthogonal probes (capacity, optimisation,
  architecture).

Read `PROTOCOL.md` Run 2+ cycle rules section for the full list. Those
rules supersede anything implicit in this brief.
