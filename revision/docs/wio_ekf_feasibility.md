# WIO-EKF feasibility note (M4c)

**Question** — can we add WIO-EKF (Zhou et al., IEEE IoT-J 2024) as a baseline
on MSILN site1 / B1 for the ICINCO 2026 revision?

## 1. Paper citation (BibTeX, verified)

```bibtex
@ARTICLE{zhou2024wioekf,
  author  = {Zhou, Pan and Wang, Hao and Gravina, Raffaele and Sun, Fangmin},
  title   = {{WIO-EKF}: Extended {Kalman} Filtering-Based {Wi-Fi} and Inertial
             Odometry Fusion Method for Indoor Localization},
  journal = {IEEE Internet of Things Journal},
  year    = {2024},
  volume  = {11},
  number  = {13},
  pages   = {23592--23603},
  doi     = {10.1109/JIOT.2024.3386889}
}
```

Title, journal, volume/issue/pages, year, DOI cross-checked against IEEE Xplore
(document 10496456) and the IEEE-pub-list metadata that quotes pp. 23592-23603.
Author list (Pan Zhou, Hao Wang, Raffaele Gravina, Fangmin Sun; SIAT) confirmed
via Google Scholar and the SIAT author page for the corresponding author
(fm.sun@siat.ac.cn). Already present in `paper-workspace/refs-raw/fusion.bib`
with the same DOI; only the first author's given name was wrong there
(`Pan`, not `Peng`).

## 2. Code availability — NOT PUBLIC

Searched (2026-06-18): IEEE Xplore record, ResearchGate, Google Scholar,
GitHub repo search (`WIO-EKF`, `CDAELoc`, `DbDIO`), GitHub code search,
author web pages (SIAT, Univ. of Calabria). No public reference
implementation. The IEEE paper carries no data-availability statement
naming a repo, and the related arXiv 2509.23118 (Li & Tang, 2025) cites
WIO-EKF as related work without pointing to code.

**Verdict: no public code as of 2026-06.**

## 3. What WIO-EKF needs vs what MSILN exposes

WIO-EKF is two deep regressors loosely coupled by an EKF:

| Sub-model | Inputs the paper uses | Outputs |
|---|---|---|
| **CDAELoc** (WiFi) | (a) per-scan RSSI vector aligned to a **fixed BSSID vocabulary**, (b) **AP-location sorting** for the data-augmentation step ("sorted by AP position"), (c) per-AP masking-noise schedule. Trained as a regression network on a *labelled fingerprint database* (RSSI -> (x,y)) | per-scan (x,y) |
| **DbDIO** (IMU) | raw 9-DoF IMU sampled at fixed rate, in **fixed 1 s windows (200 samples)** | per-window (Delta-x, Delta-y) |
| **EKF** | the two regressor outputs as observations; constant-velocity state | filtered (x,y) |

What MSILN site1/B1 actually exposes (verified against `data/msiln_site1_b1/`
and the Microsoft `indoor-location-competition-20` starter we use):

| Need | Available on MSILN? | Notes |
|---|---|---|
| Per-scan RSSI over a fixed BSSID vocab | YES | `ap_vocab.json` (1419 BSSIDs), per-path `wifi.csv` |
| Per-path raw 9-DoF IMU at ~50 Hz | YES | `imu.csv` |
| (x,y) labels at 10 Hz | YES | `ground_truth.csv` |
| **AP coordinates (BSSID -> (x,y))** | **NO** | Microsoft never published AP locations; only waypoint-anchor labels along surveyor traces |
| **Pre-built site-survey fingerprint heatmap on a grid** | **NO** | MSILN is *trace*-based, not a gridded survey; no offline radio map |
| Floorplan / wall geometry | partial | A floor image exists but no metric polygon set; not used by WIO-EKF anyway |

The two `NO` rows are the bottleneck. CDAELoc's *AP-location-sorting* data
augmentation needs BSSID -> (x,y); MSILN does not give that. Even if we
ignore the augmentation, training CDAELoc and DbDIO on MSILN traces is
*possible in principle* (the RSSI -> (x,y) regression and the IMU -> (Delta-x,
Delta-y) regression both have labels), but the two would have to be
re-implemented from scratch with no reference numbers to anchor against
and no authors' hyperparameters.

## 4. Re-implementation effort estimate

To run a *faithful* WIO-EKF on MSILN we would need to:

1. Re-implement CDAELoc (convolutional DAE + regression head) and the
   AP-position-sorted augmentation -> ~3 person-days, but the augmentation
   step would have to be redesigned for the no-AP-coords case (an honest
   degradation, not a reproduction).
2. Re-implement DbDIO (dual-branch 1-D CNN, two kernel scales) on our
   normalised IMU windows -> ~2 person-days.
3. Re-implement the EKF coupling (state, observation model for both
   sub-models, initial-heading correction step) -> ~2 person-days.
4. Hyperparameter sweep + sanity checking against the paper's
   UJIIndoorLoc/RoNIN numbers as a smoke test -> ~3 person-days.

**Total: ~10 person-days for a degraded reproduction** (no AP coords, no
gridded fingerprint DB, augmentation step neutered). The result would
*not* be a fair WIO-EKF — it would be a from-scratch reimplementation
trained on a dataset WIO-EKF's authors did not evaluate, with one of the
three pillars (the AP-sorted augmentation) missing. Reviewers comparing
against the IEEE-reported 2.53 m would (rightly) reject the comparison.

**Verdict: INFEASIBLE for a faithful baseline within the revision window.**
Within scope: cite WIO-EKF as the closest cross-session WiFi+IMU work and
explain the omission honestly.

## 5. Recommended paper text (drop-in, 2-3 sentences)

> The closest cross-session WiFi+IMU baseline, WIO-EKF
> \citep{zhou2024wioekf}, has no public reference implementation and its
> CDAELoc sub-model relies on per-AP location coordinates for its data-
> augmentation step; MSILN exposes BSSIDs and RSSI scans but no AP
> positions, so a faithful re-implementation is not possible on this
> dataset. We therefore report against the strongest WiFi-only and learned
> fusion references for which code is available or for which a faithful
> protocol-only baseline can be run on MSILN (LSTM concat, single-modality
> ablations), and we discuss WIO-EKF qualitatively in
> Section~\ref{sec:related-work}.

## 6. Bottom line

- **Citation**: verified, already in `refs-raw/fusion.bib`; fix only the
  first author's given name (`Pan`, not `Peng`).
- **Code**: not public as of 2026-06.
- **Faithful reproduction on MSILN**: infeasible (no AP coordinates, no
  gridded fingerprint database; ~10 person-days for a *degraded*
  reimplementation that would not be a fair comparison anyway).
- **Action**: cite, qualitatively position, omit numerical comparison —
  using the 3-sentence paragraph above.
