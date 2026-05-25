# Handoff — NavLoRI-Fusion (Stage B+C onwards)

You are picking up an active research project on **indoor localisation by
multi-modal sensor fusion**. This file gets you productive fast. Read it once,
then dip into `docs/fusion_pipeline.md` for depth and `CLAUDE.md` for the
day-to-day rules.

---

## 1. What this project is — in 30 seconds

A robot (Webots simulation of a TIAGO++, plus three real-world datasets) drives
around an indoor space. Up to **four async sensors** report at different rates:
**WiFi RSSI ~1 Hz · IMU ~31 Hz · Odom ~15 Hz · Camera ~5 Hz**. We predict its
position `(x, y)` in metres. The system must keep working when some sensors are
missing, late, or stale.

Author: **Mohamed Bachar**, PhD / CESI LINEACT. The work is research, not
production. Honest negative findings are part of the deliverable.

---

## 2. Where things stand right now

| Stage | What it is | Status |
|---|---|---|
| A — Encoders | per-modality networks → 128-d token | **done** (trained; in `src/pipeline/encoders/`) |
| B + C — Fusion | one set-transformer, three attentions | **done** — see `src/pipeline/fusion/transformer.py` |
| D — State filter (KalmanNet) | trajectory smoothing | **subsumed by temporal attention** (Step 7 of the docs) |
| E — Uncertainty | conformal `(x,y) ± r` | **done** — `src/pipeline/uncertainty/conformal.py` |
| Optuna search | TPE over depth/dropouts/lr/K/stride | **done** — best 0.409 m, current config defaults are its outcome |
| Multi-dataset | run on Webots sim **or** IMUWiFine / IPIN 2024 / RoNIN | **done** — `DATASET = '...'` in the notebook |

**The Stage B+C design** (the one idea everything rests on):

> One universal token = `encoder_embedding + modality_embedding + time_encoding(Δt)`.
> One transformer over that token set. **Self-attention = cross-modal fusion**;
> spanning K instants → the same layers also do **temporal fusion**. A
> **cross-attention `PositionQuery(τ)`** reads out the answer. A padding mask
> kills absent tokens; **modality + instant dropout** during training make the
> system dynamic and asynchronous-robust.

---

## 3. Quick start

```powershell
# Windows PowerShell — this is a Windows machine; no WSL/bash for scripts
cd x:\navlori-fusion

# Sanity-check (CPU): builder + config load
.venv\Scripts\python.exe -c "from src.pipeline.fusion.builder import available_datasets, load_config; print(available_datasets()); print(load_config('simulation').model.depth)"

# Phased smoke harness (GPU, ~minutes per phase)
.venv\Scripts\python.exe scripts\_smoke_fusion.py --phase 1   # shape / NaN sanity
.venv\Scripts\python.exe scripts\_smoke_fusion.py --phase 3   # full train + subset table
.venv\Scripts\python.exe scripts\_smoke_fusion.py --phase 5   # 4-modality incl. DPVO vision

# Optuna search (any dataset)
.venv\Scripts\python.exe scripts\optuna_fusion.py --dataset simulation        # ~50 min
.venv\Scripts\python.exe scripts\optuna_fusion.py --dataset ronin_a000_intra  # real-world

# Live demo notebook (picks dataset by name in cell c03)
jupyter notebook notebooks\fusion_workbench.ipynb
```

Always use `.venv\Scripts\python.exe`. Never fall back to system Python.

---

## 4. The architecture, one diagram

```
RAW ASYNC SENSORS (multi-rate)
  │
  ▼
Stage-A encoders (warm-start; heavy parts frozen + cached)
  WiFi → Anchor2Vec ─┐
  IMU  → IMUCNN     ─┤
  Odom → OdomCNN    ─┤  each → 128-d embedding
  Cam  → DPVOMotion ─┘  (frozen trunk + correlation → 64×132 patch tokens cached;
                         only _MotionHead trains end-to-end)
  │
  ▼
universal token = embedding + modality_emb + time_enc(Δt)
  │           K instants × M modalities, + padding mask
  ▼
FusionTransformer  —  N self-attention encoder layers
   self over modalities  → cross-modal fusion
   self over instants    → temporal fusion (same layers)
  │
  ▼
PositionQuery(τ) ─► CROSS-attention readout (Q ∉ set) ─► MLP → (x, y)
  │
  ▼
ConformalPosition  →  (x, y)  ±  90% radius
```

The three attentions are the **same operation** used three ways. There is no
separate "temporal module."

---

## 5. The honest findings (do not paper over these)

1. **WiFi dominates fresh-data accuracy.** Single-instant fusion reaches ≈ 0.43 m
   in simulation; only:wifi alone is ≈ 0.46 m. IMU/Odom/Vision add a few cm.

2. **Temporal fusion's value is robustness, not fresh accuracy.** Naïve temporal
   (Iteration 2) **regressed** to 0.69 m — diagnosed as overfitting. Per-instant
   dropout (Iteration 4) fixed it back to ≈ 0.44 m *and* unlocked the real
   payoff: under stale WiFi a single-instant model jumps to ~4 m (cliff); the
   temporal model degrades gracefully (0.8 m @ 2 s, 1.8 m @ 4 s).

3. **The Webots WiFi is GPR-synthesised**, not measured — its RSSI is the output
   of a Gaussian-Process model fitted to real data and queried during
   collection. So the simulation WiFi number is **optimistic**. On real data:

   | Dataset | Split | Single-instant MAE |
   |---|---|---|
   | simulation | cross-path | ≈ 0.4 m |
   | ronin_a000_intra | within-session | ≈ 10 m |
   | ipin2024_floor-2_intra | within-session | ≈ 13 m |
   | ipin2024_floor0 / imuwifine | cross-session | **diverges (train ↓, val ↑)** |

   **The architecture is fine** — the set-transformer, temporal fusion, and
   conformal all still work. The bottleneck is the **WiFi encoder**: real RSSI
   is session-specific (AP power, device, visible-AP set drift between
   recordings), and cross-session WiFi fingerprints don't transfer. Solving this
   is a Stage-A problem (session-invariant WiFi representation), not a Stage-C
   one.

4. **`drop:wifi` stays ~4 m and that is correct** — with no absolute reference
   at any instant the position is genuinely unobservable. Fusion cannot invent
   an anchor that was never measured.

5. **Conformal coverage holds only under exchangeability** — random halves of
   one pool give ~90–92%; calibrating on val and testing on test (different
   physical paths) under-covers. Methodologically the notebook uses random
   halves of the test pool.

---

## 6. File map (the things you'll actually touch)

```
configs/
  stage_c/fusion.yaml         # every model/train/temporal/optuna knob — TUNED defaults
  data/<dataset>.yaml         # per-dataset registry (modalities, split, wifi_pca, windows)

src/pipeline/
  data/dataset.py             # FusionDataset — async windows, caches, normalisation
  data/datamodule.py          # train/val/test splits, shared stats (no leakage)
  encoders/                   # Anchor2Vec / IMUCNN / OdomCNN / DPVOMotionEncoder
  fusion/transformer.py       # FusionTransformer + ContinuousTimeEncoding
  fusion/builder.py           # load_config / build_datamodule / build_model / build_trainer
  training/fusion_trainer.py  # FusionTrainer — dropout, temporal index, eval_all_subsets, eval_staleness
  uncertainty/conformal.py    # ConformalPosition

scripts/
  _smoke_fusion.py            # 5-phase smoke / profile harness
  optuna_fusion.py            # TPE search; --dataset <name>

notebooks/
  fusion_workbench.ipynb      # runnable end-to-end demo (DATASET selector in cell c03)

docs/
  fusion_pipeline.md          # 13-step walkthrough (intuition → formula → example → code)
  dpvo_motion_encoder.md      # vision encoder, by the project author
```

---

## 7. Critical rules (from CLAUDE.md — read it; these will bite you)

1. **Never push to GitHub directly.** Give the user git commands to run.
2. **Windows PowerShell** — no `&&` chaining, no `/dev/null`, no WSL scripts. Use
   PowerShell syntax or Bash via the WSL-less bash tool.
3. **Use the project venv.** `.venv\Scripts\python.exe`. Install new deps into
   it (and add them to `pyproject.toml`); never fall back to system Python.
4. **Webots needs Parsec** — cameras return NULL in SSH sessions (no GPU
   context). Run Webots in a real desktop via Parsec only.
5. **GitHub remote is HTTPS** — `https://github.com/moebachar/navlori-fusion.git`.
   The SSH deploy key was for an old repo.
6. After every dev task: update `requirements.txt`/`pyproject.toml`, `.gitignore`,
   `README.md` as needed, **then** give the user git commands.
7. **`CLAUDE.md` is gitignored** (deliberately). Update it locally for your own
   context; do not rely on it being committed.

---

## 8. Environment & data

- **GPU**: Quadro P4000 8 GB, PyTorch 2.4.1+cu124 (capped at `torch<2.7` because
  Pascal binaries were dropped from 2.7).
- **Python**: 3.11, single venv `.venv\`.
- **Webots**: R2025a, world at `src/simulation/worlds/Tiago++'s world.wbt`.
- **Services**: InfluxDB :8086 + Grafana :3000 — `scripts\services.ps1`.
- **DVC**: data versioned, remote `D:\dvc_store`.
- **Hydra + OmegaConf**: config composition.
- **Optuna**: hyperparameter search (added this stage).

Data:

- `data/async_collection/` — Webots simulation, 18 paths (path_00 empty);
  4-modality. **DVC-tracked.** WiFi is GPR-synthesised — see Finding 3.
- `data/imuwifine_floor4/` — real, 80 paths, WiFi+IMU. Cross-session split.
- `data/ipin2024_floor*/` — real, smartphone, WiFi+IMU. Both cross-session and
  `_intra` (within-session) variants exist.
- `data/ronin_a000_intra/` — real, 215 chunks, WiFi+IMU. Within-session.

All external datasets were converted to the project's `async_collection` format
by `scripts/convert_*.py`.

---

## 9. What's open / where to spend time next

In rough priority:

1. **Session-invariant WiFi encoder.** This is the single biggest research item.
   Cross-session WiFi divergence is the bottleneck on every real dataset.
   Candidates: AP-set–agnostic features (per-AP embeddings keyed by BSSID +
   masked-attention pooling), RSSI calibration (per-session offset/scale
   learned online), domain-adversarial training between sessions.

2. **Cross-dataset evaluation of fixed encoders.** Train encoders on one
   dataset, evaluate Stage-A metrics (linear probe, kNN, alignment) on another.
   Useful as a representation-quality measure independent of (x,y) targets,
   which are environment-specific.

3. **The vision encoder iteration.** `DPVOMotionEncoder` works (≈ 2.9 m alone,
   0.055 m per-pair displacement) but the author flagged it as a work in
   progress in this conversation. The framework treats it as a 4th modality;
   any better vision encoder drops in via the same `build_encoders` hook.

4. **Per-instant timestamp augmentation.** Current "instant dropout" zeros
   individual tokens; jittering each token's Δt at train time would harden
   genuine async robustness further. The masking and time-encoding machinery is
   ready for it.

5. **Webots WiFi.** The GPR-synthesis pipeline lives in the simulator's
   `wifi_module` (see `src/simulation/controllers/async_collector/`). If you
   want simulation to test cross-session robustness, the GPR needs noise
   injection / session-style variation. As-is, simulation WiFi is too clean to
   stress the encoder.

---

## 10. Etiquette for working with this user

- **Be honest about results.** Negative findings (Iteration-2 regression, WiFi
  bias, cross-session divergence) were welcomed and surfaced explicitly. Do not
  hide them.
- **Recommend rather than ask** for trivial choices. Ask explicitly only for
  forks that change what you build.
- **Iterate small.** The successful pattern this session was build → smoke test
  → profile → analyse → debug → move. One mechanism per iteration; gate it
  with a measurable test before adding the next.
- **The user knows the research.** They built the dataset, the encoders, the
  DPVO motion encoder. Trust their factual claims; verify code/results before
  asserting yours.

---

## 11. Just enough git context

Working branch: `stage-bc-fusion-transformer` (created this session). Files
ready to commit on it (the fusion work):

```
src/pipeline/fusion/transformer.py
src/pipeline/fusion/builder.py
src/pipeline/fusion/__init__.py
src/pipeline/training/fusion_trainer.py
src/pipeline/training/__init__.py
src/pipeline/uncertainty/conformal.py
src/pipeline/uncertainty/__init__.py
scripts/_smoke_fusion.py
scripts/optuna_fusion.py
configs/stage_c/fusion.yaml
notebooks/fusion_workbench.ipynb
docs/fusion_pipeline.md
README.md
pyproject.toml
```

Unrelated working-tree changes (`motion.py`, `scr_trainer.py`, simulation
files) were left out so the user commits them separately. Do not stage them
without asking.

---

## 12. If you read only one other file, read `docs/fusion_pipeline.md`

That document walks through the entire pipeline in 13 steps —
**intuition → relation → formula → worked example → in the code** — and was
written specifically to make this codebase comprehensible without re-deriving
everything. After it, the code itself is short.
