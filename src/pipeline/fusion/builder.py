"""Wiring helpers — build the fusion stack from a single config file.

Centralises data / encoder / model / trainer assembly so the notebook, the
smoke harness and the Optuna study all construct the pipeline identically.

Two config files combine:

* ``configs/stage_c/fusion.yaml`` — model / training / temporal / Optuna
  hyperparameters (dataset-independent).
* ``configs/data/<dataset>.yaml`` — the chosen dataset: which modalities it
  has, its path split, WiFi-PCA, window sizes.

So the *same* pipeline runs on the Webots simulation or on the real-world
datasets (IMUWiFine / IPIN 2024 / RoNIN) — only the dataset name changes,
and only the modalities that dataset actually has are used.
"""

from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from src.pipeline.data.datamodule import FusionDataModule
from src.pipeline.encoders import (
    WiFiNet,
    DPVOMotionEncoder,
    IMUCNN,
    OdomCNN,
    WiFiSetTransformer,
)
from src.pipeline.fusion.transformer import FusionTransformer
from src.pipeline.training.fusion_trainer import FusionTrainer


def repo_root() -> Path:
    """Repository root (……/src/pipeline/fusion/builder.py → ……)."""
    return Path(__file__).resolve().parents[3]


def available_datasets() -> list[str]:
    """Names of selectable datasets.

    A configs/data/*.yaml qualifies if it declares a split + modalities and
    its converted data directory actually exists on disk.
    """
    out = []
    for p in sorted((repo_root() / "configs" / "data").glob("*.yaml")):
        try:
            d = OmegaConf.load(p).get("data", {})
            if not (d and "split" in d and "modalities" in d):
                continue
            cdir = repo_root() / str(d.get("root", "data")) / d.get(
                "collection_dir", "")
            if cdir.is_dir():
                out.append(p.stem)
        except Exception:
            pass
    return out


def load_config(dataset: str | None = None):
    """Load fusion.yaml and merge the chosen dataset config under ``cfg.dataset``.

    ``dataset`` selects ``configs/data/<dataset>.yaml`` (default:
    ``cfg.data.default_dataset``). ``cfg.dataset`` then carries that dataset's
    modalities / split / preprocessing / windows.
    """
    cfg = OmegaConf.load(repo_root() / "configs" / "stage_c" / "fusion.yaml")
    name = dataset or cfg.data.default_dataset
    dpath = repo_root() / "configs" / "data" / f"{name}.yaml"
    if not dpath.exists():
        raise FileNotFoundError(
            f"Unknown dataset '{name}'. Available: {available_datasets()}")
    cfg.dataset = OmegaConf.load(dpath).data
    cfg.dataset.selected = name
    return cfg


def build_datamodule(cfg) -> FusionDataModule:
    """FusionDataModule for ``cfg.dataset`` (``setup()`` done)."""
    d = cfg.dataset
    mods = list(d.modalities)
    windows = dict(d.windows) if d.get("windows") else {}
    if "camera" in mods:                       # DPVO needs a frame pair
        windows["camera"] = cfg.data.camera_window
    pre = d.get("preprocessing", {}) or {}
    dm = FusionDataModule(
        data_dir=repo_root() / str(d.root) / d.collection_dir,
        train_paths=list(d.split.train_paths),
        val_paths=list(d.split.val_paths),
        test_paths=list(d.split.test_paths),
        modalities=mods,
        windows=windows or None,
        normalize=pre.get("normalize", True),
        batch_size=cfg.data.batch_size,
        camera_stride=cfg.data.camera_stride,
        wifi_pca=pre.get("wifi_pca", None),
        wifi_norm=pre.get("wifi_norm", "whiten"),
        wifi_max_stale_s=pre.get("wifi_max_stale_s", None),
        imu_frame=pre.get("imu_frame", "body"),
    )
    dm.setup()
    return dm


def pretrained_paths_from_cfg(cfg) -> dict:
    """Read ``cfg.stage_a.pretrained.<modality>`` and return a non-null map.

    Returns an empty dict when the block is absent or all-null, so callers
    can always do ``pretrained = pretrained_paths_from_cfg(cfg)`` without
    a None check.
    """
    block = cfg.get("stage_a", {}) or {}
    pre = block.get("pretrained", {}) or {}
    out = {}
    for mod, path in dict(pre).items():
        if path:
            p = Path(path)
            if not p.is_absolute():
                p = repo_root() / p
            out[mod] = p
    return out


def build_encoders(
    cfg,
    dm,
    pretrained_paths: dict | None = None,
) -> tuple[dict, DPVOMotionEncoder | None]:
    """Return ``(encoders_for_model, vision_encoder)`` for ``cfg.dataset``.

    Only the modalities the dataset actually has get an encoder.
    ``encoders_for_model['camera']`` is the DPVO ``_MotionHead`` (trains
    end-to-end on cached patch tokens); ``vision_encoder`` is the full
    ``DPVOMotionEncoder`` used once to extract those tokens (``None`` if the
    dataset has no camera). WiFi input width is read from the dataset itself
    so WiFi-PCA datasets (128-d) and the raw simulation (117-d) both work.

    Parameters
    ----------
    pretrained_paths
        Optional ``{modality: Path}`` mapping. Each value points to a Stage
        A ``encoder.pt`` (the state dict ``EncoderTrainer`` writes); when
        provided, that modality's encoder is initialised from disk before
        being handed to the fusion model. Strict-loaded — shape mismatches
        raise. Missing keys (e.g. a new feature in the dataset) raise too,
        so silent staleness can't happen.
    """
    import torch as _torch

    embed = cfg.model.embed_dim
    mods = list(cfg.dataset.modalities)
    enc: dict = {}
    if "imu" in mods:
        # IMU feature count depends on imu_frame (body=9, world=5 — M4).
        imu_dim = int(dm.train_ds.feature_dims["imu"])
        enc["imu"] = IMUCNN(in_features=imu_dim, embed_dim=embed)
    if "odom" in mods:
        enc["odom"] = OdomCNN(in_features=5, embed_dim=embed)
    if "wifi" in mods:
        wifi_kind = str(
            cfg.dataset.get("wifi_encoder_type", "wifi_net")
        ).lower()
        if wifi_kind == "anchor2vec":  # pre-PLAN_39 run metadata / configs
            wifi_kind = "wifi_net"
        n_aps = int(dm.train_ds.feature_dims["wifi"])
        if wifi_kind == "set_transformer":
            enc["wifi"] = WiFiSetTransformer(n_aps=n_aps, embed_dim=embed)
        elif wifi_kind == "wifi_net":
            enc["wifi"] = WiFiNet(n_aps=n_aps, embed_dim=embed)
        else:
            raise ValueError(
                f"Unknown wifi_encoder_type={wifi_kind!r}; "
                "expected 'wifi_net' or 'set_transformer'."
            )
    vision = None
    if "camera" in mods:
        vision = DPVOMotionEncoder(
            embed_dim=embed,
            weights_path=repo_root() / "runs" / "_weights" / "dpvo.pth",
        )
        enc["camera"] = vision.head

    # Optionally load Stage A pretrained weights.
    if pretrained_paths:
        for mod, path in pretrained_paths.items():
            if mod not in enc:
                raise KeyError(
                    f"pretrained_paths includes '{mod}' but the dataset's "
                    f"modalities are {mods}")
            state = _torch.load(path, weights_only=True, map_location="cpu")
            # EncoderTrainer saves the FULL encoder state dict, including
            # any head MLP. For tabular modalities (IMU/Odom/WiFi) the
            # encoder is the Stage A module itself and load_state_dict
            # works directly. For camera the fusion side uses dpvo.head
            # while Stage A trains the full DPVOMotionEncoder — caller
            # must point at ``head.pt`` not ``encoder.pt`` for camera.
            enc[mod].load_state_dict(state, strict=True)
    return {m: enc[m] for m in mods}, vision


def extract_vision_tokens(dm, vision_encoder, device: str = "cuda") -> dict:
    """Extract & cache DPVO patch tokens for every staged split.

    The DPVO trunk + correlation tracker is frozen / parameter-free, so the
    ``(N, 64, 132)`` per-patch tokens are cached to disk once; only the
    ``_MotionHead`` then trains. Returns the ``extra_inputs`` dict expected
    by :class:`FusionTrainer`: ``{'camera': {'train': T, 'val': T, 'test': T}}``.
    """
    cache = dm.data_dir / ".dpvomotion_fusion_cache"
    splits = {"train": dm.train_ds, "val": dm.val_ds, "test": dm.test_ds}
    out: dict = {}
    for split, ds in splits.items():
        if ds is None:
            continue
        loader = DataLoader(ds, batch_size=128, shuffle=False)
        feats, _ = vision_encoder.extract_backbone_features(
            loader, device, cache_path=cache / f"{split}.pt")
        out[split] = feats
    return {"camera": out}


def extract_vision_tokens_fast(dm, vision_encoder, device: str = "cuda",
                               subsample_stride: int = 1,
                               flush_batch: int = 128) -> dict:
    """Like extract_vision_tokens, but runs the expensive DPVO forward only
    ONCE per unique camera frame-pair and scatters the token to every sample
    that shares it. On this data the camera is 5 Hz but samples are 10 Hz, so
    ~2x of the frame-pairs are exact duplicates -> ~2x fewer DPVO forwards,
    zero quality loss (identical input tensors give identical frozen tokens).

    subsample_stride > 1 adds a lossy speedup: only every Nth UNIQUE pair is
    actually run through DPVO; intermediate uniques reuse the previous token.
    At walking pace adjacent (~0.2 s) pairs are near-identical, so stride 2-3
    costs little. stride 1 = exact dedup only (default, lossless).

    Cache format + return value match extract_vision_tokens (drop-in).
    """
    import torch as _t
    cache = dm.data_dir / ".dpvomotion_fusion_cache"
    tag = "" if subsample_stride <= 1 else f"_s{subsample_stride}"
    splits = {"train": dm.train_ds, "val": dm.val_ds, "test": dm.test_ds}
    out: dict = {}
    vision_encoder.to(device).eval()

    def key_of(xi: _t.Tensor) -> bytes:
        # strided subsample -> cheap, collision-safe content key; identical
        # frames (same PNGs + same transform) hash identically.
        return xi[:, :, ::16, ::16].contiguous().numpy().tobytes()

    for split, ds in splits.items():
        if ds is None:
            continue
        cpath = cache / f"{split}{tag}.pt"
        if cpath.exists():
            saved = _t.load(cpath, weights_only=True)
            out[split] = saved["features"]
            print(f"  [vision] {split}: cache hit", flush=True)
            continue
        loader = DataLoader(ds, batch_size=128, shuffle=False)
        order_keys: list = []
        key_to_tok: dict = {}
        pend_x: list = []
        pend_k: list = []
        pend_set: set = set()
        tgts = []
        n_uniq_run = 0
        uniq_seen = 0

        def flush():
            nonlocal n_uniq_run
            if not pend_x:
                return
            with _t.no_grad():
                toks = vision_encoder._frozen_tokens(
                    _t.stack(pend_x).to(device)).cpu()
            for k, tk in zip(pend_k, toks):
                key_to_tok[k] = tk
            n_uniq_run += len(pend_x)
            pend_x.clear(); pend_k.clear()

        with _t.no_grad():
            for batch in loader:
                x, y = batch["camera"], batch["target"]
                tgts.append(y)
                for i in range(x.shape[0]):
                    k = key_of(x[i])
                    order_keys.append(k)
                    if k in key_to_tok or k in pend_set:
                        continue
                    uniq_seen += 1
                    # lossy subsample: only run every Nth unique pair; others
                    # copy the most recent run token
                    if subsample_stride > 1 and (uniq_seen % subsample_stride) != 1 \
                            and key_to_tok:
                        key_to_tok[k] = next(reversed(key_to_tok.values()))
                        continue
                    pend_x.append(x[i]); pend_k.append(k); pend_set.add(k)
                    if len(pend_x) >= flush_batch:
                        flush()
                        pend_set = set(key_to_tok.keys())
            flush()
        features = _t.stack([key_to_tok[k] for k in order_keys])
        targets = _t.cat(tgts)
        cpath.parent.mkdir(parents=True, exist_ok=True)
        _t.save({"features": features, "targets": targets}, cpath)
        print(f"  [vision] {split}: {len(order_keys)} samples, "
              f"{uniq_seen} unique pairs, {n_uniq_run} DPVO forwards "
              f"({100*n_uniq_run/max(1,len(order_keys)):.0f}% of samples)",
              flush=True)
        out[split] = features
    return {"camera": out}


def build_model(cfg, encoders) -> FusionTransformer:
    """FusionTransformer from the ``model`` config block."""
    m = cfg.model
    abs_mods = m.get("absolute_modalities", None)
    return FusionTransformer(
        encoders, embed_dim=m.embed_dim, depth=m.depth, n_heads=m.n_heads,
        ff_mult=m.ff_mult, dropout=m.dropout, use_time=m.use_time,
        readout=m.readout,
        absolute_modalities=list(abs_mods) if abs_mods is not None else None,
        time_enc_mode=m.get("time_enc_mode", "learned_continuous"),
        time_min_period=float(m.get("time_min_period", 0.05)),
        time_max_period=float(m.get("time_max_period", 120.0)),
    )


def build_trainer(cfg, model, dm, extra_inputs=None, **overrides) -> FusionTrainer:
    """FusionTrainer from the ``train`` / ``temporal`` config blocks."""
    t = cfg.train
    # Default the baselines.json path to runs/baselines/<dataset>/baselines.json
    # if it exists; the post-fit diagnostics will print "vs baseline" when so.
    dataset_name = cfg.dataset.get("selected") or cfg.dataset.get("name")
    baselines = (repo_root() / "runs" / "baselines" / str(dataset_name)
                 / "baselines.json")
    kw = dict(
        lr=t.lr, weight_decay=t.weight_decay, huber_delta=t.huber_delta,
        grad_clip=t.grad_clip, patience=t.patience,
        batch_size=cfg.data.batch_size,
        modality_dropout=t.modality_dropout, instant_dropout=t.instant_dropout,
        n_instants=cfg.temporal.n_instants,
        instant_stride=cfg.temporal.instant_stride,
        modality_balanced_loss=bool(t.get("modality_balanced_loss", False)),
        modality_balanced_weight=float(t.get("modality_balanced_weight", 0.5)),
        aux_abs_weight=float(t.get("aux_abs_weight", 0.5)),
        baselines_path=str(baselines) if baselines.exists() else None,
    )
    kw.update(overrides)
    return FusionTrainer(model, dm, list(cfg.dataset.modalities),
                         extra_inputs=extra_inputs, **kw)
