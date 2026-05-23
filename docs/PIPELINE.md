# NavLoRI-Fusion — Full Pipeline Diagram

State as of the `audit-baseline-2026-05-20` branch (Actions 1-6 applied). Read this with [handoff/HANDOFF_LOG.md](../handoff/HANDOFF_LOG.md) for the why.

```
                       NavLoRI-Fusion — End-to-End System
═══════════════════════════════════════════════════════════════════════════════════

   Webots sim       IPIN 2024 T3      IMUWiFine        RoNIN (FRDR)
   18 paths,        3 floors,         floor 4,         per-subject,
   4 modalities     wifi+imu          wifi+imu         wifi+imu
       │                │                 │                  │
       │ async_         │ convert_        │ convert_         │ convert_
       │ collector.py   │ ipin2024.py     │ imuwifine.py     │ ronin.py
       │                │  ⚠ default      │                  │  ⚠ default
       │                │   trial-out     │                  │   was within-
       │                │                 │                  │   trial (leaky!)
       ▼                ▼                 ▼                  ▼
 ┌──────────────────────────────────────────────────────────────────────────┐
 │      data/<dataset>/path_NN/{imu,odom,wifi,camera}.csv + ground_truth    │
 │      uniform "async_collection" schema, GT @ 10 Hz, world-frame (x,y)    │
 └──────────────────────────────────────────────────────────────────────────┘
                                  │
            configs/data/<dataset>.yaml selects modalities + split
            (trial-out = honest; _intra = LEAKY, DEV USE ONLY)
                                  ▼
                ┌────────────────────────────────────┐
                │  FusionDataModule (datamodule.py)  │
                │   train_ds / val_ds / test_ds      │
                │   stats + WiFi PCA from train only │
                │   test_ds may be None (Action 8)   │
                └────────────────┬───────────────────┘
                                 │ per-sample dict of windows + GT
                                 ▼
 ┌──────────────────────────────────────────────────────────────────────────┐
 │                       FusionDataset (dataset.py)                         │
 │  ┌──────────┐  ┌─────────┐  ┌─────────┐  ┌──────────┐  ┌──────────────┐  │
 │  │  IMU     │  │  Odom   │  │  WiFi   │  │  Camera  │  │  GT (x,y) +  │  │
 │  │ 32 × 9   │  │ 16 × 5  │  │ 1 × Nap │  │  frame/  │  │  timestamp + │  │
 │  │ ≈ 1 s    │  │ ≈ 1 s   │  │  scan   │  │  pair    │  │  path_id     │  │
 │  └──────────┘  └─────────┘  └─────────┘  └──────────┘  └──────────────┘  │
 │  ODOM_COLS no longer contains odom_x / odom_y  ← Action 2                │
 │                                                                          │
 │  get_targets(mode, lookback_s):                                          │
 │    mode="position"      → y = gt[t]                                      │
 │    mode="displacement"  → y = gt[t] − gt[t − lookback]  ← Action 2       │
 └──────────────────────────────────────────────────────────────────────────┘
                                 │
       ╔═════════════════════════╧═════════════════════════════════════════╗
       ║  STAGE A · per-modality encoders → 128-d shared embedding         ║
       ║  one EncoderTrainer per modality; target_mode chooses objective   ║
       ╚═════════════════════════╤═════════════════════════════════════════╝
                                 │
   ┌──────────┬──────────┬───────┴─────────┬──────────────────────┐
   ▼          ▼          ▼                 ▼                      ▼
 ┌──────┐ ┌──────┐ ┌────────────┐ ┌──────────────┐         ┌──────────────┐
 │IMUCNN│ │Odom  │ │ Anchor2Vec │ │ DPVOMotion   │         │  (removed    │
 │1D-CNN│ │CNN   │ │ k=64       │ │ DPVO trunk   │         │  2026-05-20: │
 │      │ │      │ │ anchors→   │ │ + patch flow │         │  ACEVision,  │
 │      │ │      │ │ softmax    │ │ + head MLP   │         │  VisionViT   │
 │128-d │ │128-d │ │ 128-d      │ │ 128-d        │         │  — place-    │
 └───┬──┘ └───┬──┘ └─────┬──────┘ └──────┬───────┘         │  recognition │
     │        │          │              │                  │  encoders;   │
 target=Δxy target=Δxy target=(x,y)  target=Δxy            │  trained on  │
 (motion)  (motion)  (place recog)  (motion)               │  (x,y) →     │
     │        │          │              │                  │  memorizing) │
     └────────┴──────────┴──────┬───────┘                  └──────────────┘
                                │
                       runs/<mod>_<ts>/encoder.pt           ← saved by
                       runs/<mod>_<ts>/eval.json              EncoderTrainer
                                │
       ┌────────────────────────┘
       │  cfg.stage_a.pretrained.<mod>: path/to/encoder.pt   ← Action 5
       │   strict_load=True, null = train fresh
       ▼
       ╔══════════════════════════════════════════════════════════════════╗
       ║  STAGE B + C · FusionTransformer (one set-transformer)           ║
       ║  three attentions: self (cross-modal) · self (across-time)       ║
       ║                    · cross (readout query)                       ║
       ╚════════════════════════╤═════════════════════════════════════════╝
                                ▼
       For each sample: build the universal token set

           token_{k,m} = encoder_m(window_{k,m})       ← from Stage A
                       + modality_emb[m]
                       + time_enc(Δt_{k,m})            ← ContinuousTimeEncoding

           avail_{k,m} = was this modality observed at instant k?

       Training-time augmentation (FusionTrainer):
         • modality_dropout (raised 0.16 → 0.4)        ← Action 3
         • instant_dropout
         • modality_balanced_loss: extra leave-one-out loss term  ← Action 3
                                ▼
                  ┌──────────────────────────────┐
                  │  [CLS]  +  M · K  tokens     │  CLS never masked
                  │                              │   ⇒ no all-(-inf)
                  │  pad_mask from avail{k,m}    │   softmax → no NaN
                  └──────────────┬───────────────┘
                                 ▼
                  ┌──────────────────────────────┐
                  │  nn.TransformerEncoder       │
                  │  depth × MHA + GELU FFN      │   cross-modal AND
                  │  norm_first                  │   cross-time fusion
                  └──────────────┬───────────────┘
                                 │
                  ┌──────────────┴──────────────┐
                  ▼                             ▼
          readout = "cls"               readout = "query"
          pooled = x[:, 0]              PositionQuery(τ) cross-attends
                                        over [CLS, tokens]
                                        ⇒ asynchronous prediction at τ
                                 │
                                 ▼
                  ┌──────────────────────────────┐
                  │  MLP head  (D → 2)           │
                  └──────────────┬───────────────┘
                                 ▼
                         (x, y)  prediction
                         val_mae = Euclidean MAE meters  ← Action 4
                                 │
       ╔═════════════════════════╪═════════════════════════════════════════╗
       ║  EVALUATION · GATING                                              ║
       ╚═════════════════════════╪═════════════════════════════════════════╝
                                 │
   ┌─────────────────────────────┼─────────────────────────────┐
   ▼                             ▼                             ▼
 ┌──────────────┐    ┌──────────────────────────────┐   ┌─────────────────┐
 │ Conformal-   │    │ Post-fit diagnostics block   │   │ scripts/        │
 │ Position     │    │  (Action 3, auto after fit)  │   │  baselines.py   │
 │ split conf,  │    │  • subsets table             │   │                 │
 │ α=0.1 → r    │    │  • drop:X gap < 0.1m         │   │  mean train pos │
 │ 90% target   │    │      ⇒  <- UNUSED  flag      │   │  WiFi-kNN k=5   │
 │ coverage     │    │  • vs best baseline          │   │  IMU Kalman     │
 │              │    │      ⇒  PASS / FAIL gate     │◀──┤  outputs to     │
 │  ⊕  (x,y)±r  │    │                              │   │  runs/baselines/│
 └──────────────┘    └──────────────┬───────────────┘   │  <dataset>/     │
                                    ▼                   └─────────────────┘
                  ┌──────────────────────────────┐               ▲
                  │  scripts/optuna_fusion.py    │               │
                  │  TPE sampler                 │               │
                  │  per-dataset output dir      │  Action 6     │
                  │  honors stage_a.pretrained   │───────────────┘
                  │  baseline gate at end        │
                  └──────────────────────────────┘

       ╔════════════════════════════════════════════════════════════════════╗
       ║  CROSS-CUTTING CONVENTIONS                                         ║
       ║                                                                    ║
       ║  • Metric: every `mae` in any module / json / log is               ║
       ║    Euclidean MAE in meters (encoder_eval.euclidean_mae).           ║
       ║    Old per-axis L1 stays available under `mae_component`.          ║
       ║                                                                    ║
       ║  • Splits: trial-out is honest. `_intra` configs labelled          ║
       ║    LEAKY — DEV USE ONLY; never published as a result.              ║
       ║                                                                    ║
       ║  • Notebook (fusion_workbench.ipynb): cells use                    ║
       ║    EVAL_SPLIT = 'test' if dm.test_ds is not None else 'val'        ║
       ║    so they degrade gracefully on no-test datasets.                 ║
       ║                                                                    ║
       ║  • Branch: audit-baseline-2026-05-20. main is untouched.           ║
       ╚════════════════════════════════════════════════════════════════════╝
```

## Action mapping

| Action | What changed | Code |
|---|---|---|
| 1 | Trial-out splits + 3 baselines computed per dataset | `scripts/baselines.py`, `configs/data/ronin_a000.yaml` |
| 2 | Displacement targets for motion encoders; `odom_x/y` removed | `dataset.py`, `trainer.py:target_mode`, `odom.py:in_features=5` |
| 3 | Modality-balanced loss + post-fit diagnostics + dropout 0.16→0.4 | `fusion_trainer.py`, `configs/stage_c/fusion.yaml` |
| 4 | Euclidean MAE everywhere; trustworthiness wired | `encoder_eval.py:euclidean_mae`, `trainer.py:evaluate` |
| 5 | `stage_a.pretrained` block strict-loads Stage A weights | `builder.py:build_encoders`, `configs/stage_c/fusion.yaml` |
| 6 | Optuna per-dataset output + baseline gate | `scripts/optuna_fusion.py` |
