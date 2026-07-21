# Experiments + Results — Writer Brief (from paper_results.ipynb)

**Source of truth:** `notebooks/paper_results.ipynb` (director-refined). Full digest:
`paper-workspace/_nb_digest.txt`. Every number below is from that notebook's live/saved
outputs — use these, NOT the older `draft.tex` experiment bullets (they are stale).

**Director's 5 takes (obey):**
1. Tables name the **competitive methods in columns** — never a generic "SOTA"/"best baseline" column.
2. WiFi datasets list the **number of APs**.
3. Drive the whole section from the notebook.
4. **Split into two sections: `Experiments` (setup) + `Results`.**
5. Figures come from the notebook (high-res PDFs already exported — see Figures section).

---

## SECTION 1 — `\section{Experiments}` (setup only)

### Datasets (build `tab:datasets`, full-width `table*`)
Columns: Dataset | Modalities | Setting | #APs | Unit | Train | Val | Test

| Dataset | Modalities | Setting | #APs | Unit | Train | Val | Test |
|---|---|---|---|---|---|---|---|
| Webots sim | WiFi+IMU | controlled lab (sim) | 117 | paths | 11 | 3 | 3 |
| MSILN site1/B1 | WiFi+IMU | real, cross-session | 1419 | paths | 94 | 34 | 5 |
| UJIIndoorLoc | WiFi | real, per-leg | 520 | scans | 19937 | 1111 | — |
| RoNIN canonical | IMU | real, per-leg (unseen subj.) | — | sequences | 73 | — | 32 |

- **Unit matters** (director's earlier note): UJI counts are **scans/fingerprints**, not sequences;
  Webots/MSILN are **paths**; RoNIN are **sequences**. State the unit, don't blur them.
- MSILN is the **headline cross-session** dataset (train Nov-24 / val Nov-25 / test Dec-05/06).
- Webots = controlled lab where all ablations run. UJI/RoNIN = per-leg encoder validation only.
- Cite datasets: UJIIndoorLoc `\cite{torressospedra2014ujiindoorloc}`, RoNIN `\cite{yan2019ronin}`.
  MSILN = Microsoft Indoor Localisation competition data `[[VERIFY cite]]`.

### Baselines (NAME each; this is take #1)
- **wlan\_localization** `\cite{wlanloc}` — kNN RSSI fingerprinting (k=3, Manhattan, distance-weighted). WiFi baseline.
- **ResNet1D (RoNIN)** `\cite{yan2019ronin}` — 4.63 M-param learned-inertial regressor. IMU baseline.
- **PDR-from-start** — IMU-only pedestrian dead-reckoning (step detection), anchored at the first
  ground-truth waypoint. Classical inertial baseline `[[VERIFY cite optional]]`.
- **IMUWiFine** — clean-room reimplementation of **Nurpeiissov et al. 2022** (4-layer LSTM that fuses
  WiFi+IMU). Learned fusion baseline `[[VERIFY: add nurpeiissov2022 to refs.bib]]`.
- Do NOT call any of these "SOTA" in a table header — use their names as columns/rows.

### Metrics (policy)
MAE = mean Euclidean position error (m); ATE = absolute trajectory error (m), reported **raw**.
Umeyama-aligned ATE appears **only** in Limitations. No RMSE in the main tables.

### Implementation / training
90 epochs, AdamW (lr 1.3e-3, wd 1e-4), OneCycleLR, Huber (δ=0.5), grad-clip 1.0, B=128, K=4,
seed 42; PyTorch on a single NVIDIA Quadro P4000. (Training config may be a small `tab:train` OR
folded into prose — keep one, not both. The architecture lives in §3, reference `Section~\ref{sec:method}`.)

---

## SECTION 2 — `\section{Results}`

### 2.1 Per-leg encoder validation (build `tab:perleg`, named columns)
| Modality / Dataset | metric | `wlan_localization` / `ResNet1D` | **WiFi-Net / IMUCNN (ours)** | Δ |
|---|---|---|---|---|
| WiFi / UJIIndoorLoc (val) | MAE | wlan\_localization **15.17** | WiFi-Net **8.69** | **−42.7%** |
| IMU / RoNIN canonical | raw ATE | ResNet1D **5.14** | IMUCNN **9.72** | **+89.2%** |
- **Honest framing (keep):** WiFi-Net beats wlan\_localization on UJI; the IMU encoder is **behind**
  ResNet1D on canonical RoNIN — state it plainly as the in-domain-vs-cross-subject trade-off of a
  ~95×-smaller encoder (IMUCNN ~0.05 M vs ResNet1D 4.63 M). Do not spin it.

### 2.2 End-to-end fusion (build `tab:fusion`, named columns — this is the headline)
| Dataset | `wlan_localization` | `PDR-from-start` | `IMUWiFine` | **Ours** |
|---|---|---|---|---|
| Webots sim test | — | — | — | **0.441** |
| MSILN val | 21.26 | 16.88 | 65.15 | **16.67** |
| MSILN test ⭐ | 28.31 | 12.49 | 52.69 | **10.90** |
- Bold the lowest per row. On **MSILN test**, Ours wins against all three: **−61.5%** vs
  wlan\_localization, **−79.3%** vs IMUWiFine, **−12.7%** vs PDR-from-start.
- On **MSILN val**, Ours (16.67) only **narrowly** beats PDR-from-start (16.88) — say so honestly;
  the decisive margin is on the held-out test session.
- **IMUWiFine fails to generalize cross-session** here (52–65 m) despite being a learned WiFi+IMU
  fusion — an honest, informative negative result, not hidden.
- Webots sim (0.441 m) is the controlled-lab sanity check; no public 2-modality baseline exists there.

### 2.3 Robustness & ablations (Webots; build small tables or prose)
- **Modality dropout (test MAE):** wifi-only 0.561 · imu-only 3.491 · **wifi+imu 0.441** → graceful
  degradation; WiFi anchors absolute position, IMU alone drifts.
- **Staleness sweep (WiFi token age 0→4, test MAE):** 0.441 → 0.587 → 0.772 → 0.980 → 3.491 →
  cliff-to-slope: small lags cost little, only a fully stale WiFi degrades sharply.
- **Latency (Quadro P4000):** b=1 **4.77 ms/sample**, b=32 **0.146 ms/sample** — real-time.

### 2.4 Limitations (honest; keep brief)
- IMU canonical gap: raw **+89.2%**, Umeyama-aligned **+48.2%** (9.72 / 7.62 vs ResNet1D 5.14).
- MSILN **path-130** = 786 samples (**28% of test**), WiFi-dense — note it dominates the test split.

---

## Figures (high-res vector PDFs, ready in `paper/figures/`)
Stable filenames, exported from the notebook (all vector PDF — use `\includegraphics`, no svg package):
- **`fig_msiln_cdf.pdf`** — CDF of per-sample error, MSILN test (Ours vs `wlan_localization` vs `IMUWiFine`).
  **HEADLINE results figure**: Ours reaches 90% within ~15 m; `wlan_localization` has a long tail to ~140 m;
  `IMUWiFine` is shifted right (starts ~30 m). → §Results, end-to-end fusion.
- **`fig_ronin_traj.pdf`** — RoNIN canonical a051\_3 trajectory: GT vs `ResNet1D` vs IMUCNN (Ours).
  Honest per-leg IMU qualitative (Ours drifts more than ResNet1D). → §Results, per-leg.
- **`fig_webots_scatter.pdf`** — Webots sim test, GT vs Ours; sub-metre tracking. → §Results, fusion (controlled lab).
- **`fig_uji_scatter.pdf`** (optional) — UJI val, GT vs `wlan_localization` vs WiFi-Net (Ours). → §Results, per-leg WiFi.
- **`fig_msiln_perpath.pdf`** (optional, weak — only 2 paths) — MSILN per-path MAE, PDR-start vs Ours.

**Recommended minimal set:** `fig_msiln_cdf` (headline) + `fig_ronin_traj` (per-leg IMU) + `fig_webots_scatter`
(fusion qualitative). Place single-column `[!hb]` or full-width `figure*[tb]` as fits; captions below.

---

## Writing directives
- Single-blind ("we"/"the proposed method"); scientific, declarative; anti-AI tone
  (`style-anti-ai.md`) — no "Furthermore/Moreover" chains, no hype.
- **No project-internal trivia** (no RESULT_NN, no Webots path indices in prose, no notebook cell refs).
- Every number traceable: keep a `% src:` comment (cell / value) on load-bearing numbers.
- Tables: `booktabs`-style or the template's rules; bold the best per row; named-method columns only.
- Keep the IMUWiFine-fl.4 extra dataset OUT unless the director opts in (pending decision).
- Do not invent citations; flag `[[VERIFY]]` for IMUWiFine/Nurpeiissov + MSILN + PDR.
