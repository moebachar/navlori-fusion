"""FusionTrainer — trains the FusionTransformer end-to-end.

Mirrors :class:`~src.pipeline.training.trainer.EncoderTrainer` but for the
multi-modal fusion model: AdamW, OneCycleLR, Huber loss, early stopping,
JSONL metric logging.

Two things it adds over EncoderTrainer:

1. **Modality dropout** — every training step independently drops each
   modality with probability ``modality_dropout`` (always keeping ≥1).
   This is the entire "dynamic / works-with-any-subset" property.
2. **Subset evaluation** — reports MAE for the all-modalities case, every
   single modality alone, and every leave-one-out combination. That table
   is the robustness proof.

Tabular modality windows are small enough to live on the GPU in full, so
training batches by ``torch.randperm`` index slicing — no DataLoader, no
collate, zero per-step Python overhead.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# Metric convention: ``val_mae`` and every subset/staleness ``mae`` below is
# the Euclidean mean error in meters (``mean(||pred - y||_2)``), matching
# :func:`src.pipeline.evaluation.encoder_eval.euclidean_mae`. Reporting one
# definition across stages avoids the encoder-vs-fusion unit mismatch.


@dataclass
class FusionHistory:
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    val_mae: list[float] = field(default_factory=list)
    best_epoch: int = 0
    best_val_mae: float = float("inf")
    elapsed_sec: float = 0.0


def _avail_from_cache(x: torch.Tensor) -> torch.Tensor:
    """Per-sample availability: True iff the window is not all-zeros.

    A zero window means the dataset found no observation before the
    anchor time (front-padded early sample) — genuinely missing data.
    """
    return x.flatten(1).abs().sum(dim=1) > 0


class FusionTrainer:
    """Train a :class:`FusionTransformer` to predict (x, y).

    Parameters
    ----------
    model : FusionTransformer
    dm : FusionDataModule
        Already ``setup()``-ed. Tabular modality caches are read directly.
    modalities : list[str]
        Modalities to use (must match ``model.modalities``).
    extra_inputs : dict[str, dict[str, torch.Tensor]] | None
        Optional pre-computed per-split tensors for modalities not in the
        tabular cache (e.g. camera DPVO patch tokens). Shape per split:
        ``{"train": tensor(N,*), "val": tensor(N,*)}``.
    modality_dropout : float
        Probability of dropping a whole modality (all K instants) per
        sample during training — robustness to a sensor being absent.
    instant_dropout : float
        Probability of dropping an individual (instant, modality) token —
        async / intermittent-observation augmentation. Forces the model to
        propagate across time instead of leaning on the freshest token,
        which both builds staleness robustness and regularises the larger
        temporal token set. Only meaningful with ``n_instants > 1``.
    n_instants : int
        Number of recent GT instants per sample (Iteration 2). ``1`` =
        single-instant fusion (Iteration 1). Each instant contributes one
        token per modality; the transformer attends across modality *and*
        time jointly.
    instant_stride : int
        Spacing between instants, in GT rows (~10 Hz → stride 10 ≈ 1 s).
    """

    def __init__(
        self,
        model,
        dm,
        modalities: list[str],
        extra_inputs: dict | None = None,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        huber_delta: float = 0.5,
        grad_clip: float = 1.0,
        patience: int = 25,
        batch_size: int = 128,
        modality_dropout: float = 0.3,
        instant_dropout: float = 0.0,
        n_instants: int = 1,
        instant_stride: int = 1,
        device: str = "auto",
        run_dir: str | Path = "runs",
        modality_balanced_loss: bool = False,
        modality_balanced_weight: float = 0.5,
        baselines_path: str | Path | None = None,
        aux_abs_weight: float = 0.5,
    ):
        self.device = ("cuda" if torch.cuda.is_available() else "cpu") \
            if device == "auto" else device
        self.model = model.to(self.device)
        self.dm = dm
        self.modalities = modalities
        self.lr = lr
        self.weight_decay = weight_decay
        self.criterion = nn.HuberLoss(delta=huber_delta)
        self.grad_clip = grad_clip
        self.patience = patience
        self.batch_size = batch_size
        self.modality_dropout = modality_dropout
        self.instant_dropout = instant_dropout
        self.n_instants = max(1, n_instants)
        self.instant_stride = max(1, instant_stride)
        # Modality-balanced loss: on each step, with probability
        # ``modality_balanced_weight`` we additionally compute the loss
        # after zeroing one randomly-chosen modality (leave-one-out), so
        # gradients explicitly reward improving every leave-one-out
        # subset. This is the direct fix for "the model leans on WiFi
        # and ignores everything else" (audit finding B2).
        self.modality_balanced_loss = bool(modality_balanced_loss)
        self.modality_balanced_weight = float(modality_balanced_weight)
        # Auxiliary anchor loss for the decomposed readout: pushes p_abs to
        # be the best WiFi-only estimate of the current position, so the
        # motion path's delta becomes a genuine residual correction. Only
        # used when model.readout == "decomposed".
        self.aux_abs_weight = float(aux_abs_weight)
        self.is_decomposed = getattr(model, "readout", None) == "decomposed"
        # Optional path to scripts/baselines.py output for this dataset;
        # if provided, the run prints "vs baseline" next to val_mae.
        self.baselines_path = (Path(baselines_path)
                               if baselines_path is not None else None)
        # Iteration 3 — when the model reads out via a cross-attention
        # query, train it at *random* query times so the time-parameterised
        # query genuinely learns to route by τ (→ asynchronous prediction).
        self.query_readout = getattr(model, "readout", "cls") == "query"

        # ---- Stage tensors onto the device (sim data is small) ----
        # Always train + val; add test when available so evaluate_subsets /
        # predict / evaluate_staleness can report on the held-out split.
        # For modalities served via extra_inputs, a split is only staged if
        # that modality provides a tensor for it.
        splits = [("train", dm.train_ds), ("val", dm.val_ds)]
        if getattr(dm, "test_ds", None) is not None and (
            not extra_inputs
            or all("test" in extra_inputs[m] for m in extra_inputs)
        ):
            splits.append(("test", dm.test_ds))
        self.splits = [s for s, _ in splits]

        self.X = {s: {} for s in self.splits}
        self.A = {s: {} for s in self.splits}
        self.y, self.n = {}, {}
        self.inst_idx, self.inst_av, self.inst_dt = {}, {}, {}
        for split, ds in splits:
            for mod in modalities:
                if extra_inputs and mod in extra_inputs:
                    x = extra_inputs[mod][split]
                else:
                    x = ds.get_tensors(mod)[0]
                x = x.to(self.device)
                self.X[split][mod] = x
                self.A[split][mod] = _avail_from_cache(x).to(self.device)
            self.y[split] = ds._targets.to(self.device)
            self.n[split] = len(self.y[split])
            # Temporal index: for each sample, the K recent instants;
            # instant K-1 is the anchor / query.
            ii, av, dt = self._build_temporal_index(ds)
            self.inst_idx[split] = ii
            self.inst_av[split] = av
            self.inst_dt[split] = dt

        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        self.scheduler = None

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_id = f"fusion_{ts}"
        self.run_path = Path(run_dir) / self.run_id
        self.run_path.mkdir(parents=True, exist_ok=True)
        self._metrics_file = self.run_path / "metrics.jsonl"

    # ------------------------------------------------------------------
    def _build_temporal_index(self, ds):
        """Per-sample (instant_idx, instant_avail, instant_dt) tensors.

        For each sample the K instants are ``[L-(K-1)s, ..., L-s, L]`` in
        GT-row local position within the *same path*; instant K-1 is the
        anchor (= the query instant). Instants that fall before the start
        of the path are marked unavailable (masked, not clamped).

        Returns
        -------
        inst_idx : (N, K) long  — global cache index of each instant
        inst_av  : (N, K) bool  — False where the instant predates the path
        inst_dt  : (N, K) float — Δt seconds, t_instant − t_anchor (≤ 0)
        """
        K, step = self.n_instants, self.instant_stride
        N = len(ds._gt_rows)
        times = ds._timestamps.to(torch.float64)
        inst_idx = torch.zeros(N, K, dtype=torch.long)
        inst_av = torch.zeros(N, K, dtype=torch.bool)
        inst_dt = torch.zeros(N, K, dtype=torch.float32)

        by_path: dict[int, list[int]] = {}
        for g, r in enumerate(ds._gt_rows):
            by_path.setdefault(r["path_id"], []).append(g)

        for gidx in by_path.values():
            gidx = sorted(gidx)
            for L, g in enumerate(gidx):
                for k in range(K):
                    Lp = L - (K - 1 - k) * step
                    if Lp >= 0:
                        gp = gidx[Lp]
                        inst_idx[g, k] = gp
                        inst_av[g, k] = True
                        inst_dt[g, k] = float(times[gp] - times[g])
                    else:
                        inst_idx[g, k] = g  # dummy — masked out by inst_av
        return (inst_idx.to(self.device), inst_av.to(self.device),
                inst_dt.to(self.device))

    def _batch(self, split: str, idx: torch.Tensor, drop: bool):
        """Assemble a (inputs, avail, dt, target) batch for index tensor idx.

        Tokens are shaped ``(B, K, *window)``. With ``n_instants=1`` this
        is single-instant fusion (Iteration 1); with K>1 the transformer
        also attends across time (Iteration 2). ``dt`` carries each
        instant's Δt to the query instant so timing is honoured.
        """
        ii = self.inst_idx[split][idx]                   # (B, K)
        inputs, avail = {}, {}
        for mod in self.modalities:
            inputs[mod] = self.X[split][mod][ii]         # (B, K, *win)
            avail[mod] = self.A[split][mod][ii] & self.inst_av[split][idx]
        if drop and (self.modality_dropout > 0 or self.instant_dropout > 0):
            avail = self._apply_dropout(avail)
        dt = {m: self.inst_dt[split][idx] for m in self.modalities}
        y_inst = self.y[split][ii]                       # (B, K, 2) GT per instant
        return inputs, avail, dt, self.y[split][idx], y_inst

    def _resolve_query(self, idx, dt, y_anchor, y_inst, randomize: bool):
        """Pick the query time τ and matching target for a batch.

        With the cross-attention readout the query can ask for position at
        any instant. Training randomises τ over the K instants so the query
        learns to route by time; evaluation fixes τ at the anchor (Δt=0).
        Returns ``(target, query_dt)`` — ``query_dt`` is None for CLS readout.
        """
        if not self.query_readout:
            return y_anchor, None
        B = idx.shape[0]
        anchor_dt = dt[self.modalities[0]]               # (B, K)
        if not randomize:
            return y_anchor, torch.zeros(B, device=self.device)
        ar = torch.arange(B, device=self.device)
        q = torch.randint(0, self.n_instants, (B,), device=self.device)
        return y_inst[ar, q], anchor_dt[ar, q]

    def _apply_dropout(self, avail: dict) -> dict:
        """Drop tokens for training-time robustness; always keep ≥1 live.

        Two independent mechanisms:

        * **modality dropout** — zero a whole modality across all K
          instants (a sensor is absent for this sample).
        * **instant dropout** — zero individual (instant, modality)
          tokens (a sensor's observation is intermittently missing).

        A rescue pass restores the anchor token of a random
        originally-available modality for any sample that lost everything.
        """
        mods = self.modalities
        B, K = avail[mods[0]].shape
        keep = {m: avail[m].clone() for m in mods}

        if self.modality_dropout > 0:
            roll = torch.rand(B, len(mods), device=self.device) \
                < self.modality_dropout
            for j, m in enumerate(mods):
                keep[m] = keep[m] & ~roll[:, j:j + 1]      # broadcast over K

        if self.instant_dropout > 0:
            for m in mods:
                roll = torch.rand(B, K, device=self.device) \
                    < self.instant_dropout
                keep[m] = keep[m] & ~roll

        # Rescue: any sample with no live token → restore one anchor token.
        stacked = torch.stack([keep[m] for m in mods], dim=-1)     # (B,K,M)
        dead = ~stacked.any(dim=(1, 2))                            # (B,)
        if dead.any():
            anchor = torch.stack([avail[m][:, -1].float() for m in mods],
                                 dim=-1)                           # (B,M)
            pick = torch.multinomial(anchor[dead] + 1e-6, 1).squeeze(1)
            dead_idx = torch.where(dead)[0]
            for j, m in enumerate(mods):
                keep[m][dead_idx[pick == j], -1] = True
        return keep

    # ------------------------------------------------------------------
    def fit(self, epochs: int = 100, verbose: bool = True) -> FusionHistory:
        n_tr = self.n["train"]
        steps = max(1, n_tr // self.batch_size)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer, max_lr=self.lr, epochs=epochs,
            steps_per_epoch=steps, pct_start=0.3,
        )
        hist = FusionHistory()
        patience_ctr = 0
        best_state = None
        t0 = time.time()
        self._metrics_file.unlink(missing_ok=True)
        (self.run_path / "meta.json").write_text(json.dumps({
            "run_id": self.run_id, "modalities": self.modalities,
            "device": self.device, "epochs": epochs, "lr": self.lr,
            "modality_dropout": self.modality_dropout,
            "instant_dropout": self.instant_dropout,
            "modality_balanced_loss": self.modality_balanced_loss,
            "modality_balanced_weight": self.modality_balanced_weight,
            "readout": getattr(self.model, "readout", "?"),
            "aux_abs_weight": self.aux_abs_weight if self.is_decomposed else None,
            "n_instants": self.n_instants, "instant_stride": self.instant_stride,
            "n_train": n_tr, "n_val": self.n["val"],
            "started_at": datetime.now().isoformat(),
        }, indent=2))

        for epoch in range(epochs):
            train_loss = self._train_epoch(steps)
            val_loss, val_mae = self._val_epoch()
            hist.train_loss.append(train_loss)
            hist.val_loss.append(val_loss)
            hist.val_mae.append(val_mae)

            if val_mae < hist.best_val_mae:
                hist.best_val_mae, hist.best_epoch = val_mae, epoch
                patience_ctr = 0
                best_state = {k: v.detach().cpu().clone()
                              for k, v in self.model.state_dict().items()}
            else:
                patience_ctr += 1

            lr = self.scheduler.get_last_lr()[0]
            with open(self._metrics_file, "a") as f:
                f.write(json.dumps({
                    "epoch": epoch, "train_loss": train_loss,
                    "val_loss": val_loss, "val_mae": val_mae, "lr": lr,
                    "is_best": patience_ctr == 0, "t": time.time() - t0,
                }) + "\n")
            if verbose and (epoch % 10 == 0 or epoch == epochs - 1
                            or patience_ctr == self.patience):
                print(f"  Epoch {epoch:3d}/{epochs}  train={train_loss:.4f}  "
                      f"val={val_loss:.4f}  val_mae={val_mae:.3f}m  "
                      f"lr={lr:.2e}  {'*' if patience_ctr == 0 else ''}",
                      flush=True)
            if patience_ctr >= self.patience:
                if verbose:
                    print(f"  Early stopping at epoch {epoch} "
                          f"(best={hist.best_epoch})")
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.model.to(self.device)
        hist.elapsed_sec = time.time() - t0
        if verbose:
            print(f"  Done in {hist.elapsed_sec:.1f}s — best val MAE: "
                  f"{hist.best_val_mae:.3f}m (epoch {hist.best_epoch})")
        (self.run_path / "history.json").write_text(json.dumps({
            "run_id": self.run_id, "best_epoch": hist.best_epoch,
            "best_val_mae": hist.best_val_mae, "elapsed_sec": hist.elapsed_sec,
            "train_loss": hist.train_loss, "val_loss": hist.val_loss,
            "val_mae": hist.val_mae,
            "finished_at": datetime.now().isoformat(),
        }, indent=2))
        torch.save(self.model.state_dict(), self.run_path / "model.pt")

        # Always run subsets + report unused modalities + (if available)
        # how far we are from the baselines. This is the audit-mandated
        # "no fusion run ships without these numbers" diagnostic.
        if verbose:
            self._print_post_fit_diagnostics(hist.best_val_mae)
        return hist

    # ------------------------------------------------------------------
    def _print_post_fit_diagnostics(self, best_val_mae: float) -> None:
        """Subset-MAE table, unused-modality flags, baseline comparison.

        Lifted out of ``fit`` so it can be called standalone after loading a
        saved model. ``best_val_mae`` is the all-modalities MAE the run
        finished at; we use it as the reference point for "drop X" gaps.
        """
        print("\n  Post-fit diagnostics", flush=True)
        print("  " + "-" * 70, flush=True)
        subsets = self.evaluate_subsets("val")
        all_mae = subsets.get("all", {}).get("mae", best_val_mae)
        flag_thresh = 0.10                              # m
        for label, m in subsets.items():
            mark = ""
            if label.startswith("drop:"):
                gap = m["mae"] - all_mae
                if abs(gap) < flag_thresh:
                    mark = "  <- UNUSED (gap < 0.1m)"
            print(f"    {label:14s}  MAE={m['mae']:.3f}m  RMSE={m['rmse']:.3f}m{mark}",
                  flush=True)

        # Baseline comparison
        if self.baselines_path and self.baselines_path.exists():
            try:
                rec = json.loads(self.baselines_path.read_text())
                vsplit = rec.get("splits", {}).get("val", {})
                if vsplit:
                    print(f"\n  Baselines on this dataset (val):", flush=True)
                    for b in vsplit.get("baselines", []):
                        gap = all_mae - b["mae"]
                        sign = "BEATEN by fusion" if gap < 0 else "STILL BEATING fusion"
                        print(f"    {b['name']:14s}  MAE={b['mae']:.3f}m  "
                              f"(fusion - baseline = {gap:+.3f}m, {sign})",
                              flush=True)
                    best = vsplit.get("best")
                    bm = vsplit.get("best_mae")
                    if best and bm is not None:
                        gap = all_mae - bm
                        ok = "PASS" if gap < 0 else "FAIL"
                        print(f"  Best baseline = {best} @ {bm:.3f}m | fusion {all_mae:.3f}m | gap {gap:+.3f}m  [{ok}]",
                              flush=True)
            except Exception as e:
                print(f"  (baseline comparison skipped: {e})", flush=True)
        print("  " + "-" * 70, flush=True)

    def _train_epoch(self, steps: int) -> float:
        self.model.train()
        perm = torch.randperm(self.n["train"], device=self.device)
        total, seen = 0.0, 0
        mods = self.modalities
        for s in range(steps):
            idx = perm[s * self.batch_size:(s + 1) * self.batch_size]
            if len(idx) == 0:
                continue
            inputs, avail, dt, y_anchor, y_inst = self._batch(
                "train", idx, drop=True)
            y, query_dt = self._resolve_query(
                idx, dt, y_anchor, y_inst, randomize=True)
            if self.is_decomposed:
                pred, parts = self.model(inputs, avail, dt, query_dt=query_dt,
                                         return_parts=True)
                loss = self.criterion(pred, y)
                # Anchor aux loss: p_abs should itself predict the position,
                # so delta is forced to be a correction, not the whole signal.
                loss = loss + self.aux_abs_weight * self.criterion(parts["p_abs"], y)
            else:
                pred = self.model(inputs, avail, dt, query_dt=query_dt)
                loss = self.criterion(pred, y)

            # Modality-balanced loss: with probability
            # `modality_balanced_weight`, add a leave-one-out loss on a
            # uniformly-random modality. Forces the model to also do well
            # without that modality, so gradient pressure spreads across
            # modalities instead of collapsing onto the dominant one.
            if (self.modality_balanced_loss and len(mods) > 1
                    and torch.rand(1).item() < self.modality_balanced_weight):
                drop_mod = mods[int(torch.randint(0, len(mods), (1,)).item())]
                avail_loo = {m: (avail[m].clone() if m != drop_mod
                                 else torch.zeros_like(avail[m]))
                             for m in mods}
                # The model has a CLS rescue path so all-modalities-gone
                # samples don't NaN; the leave-one-out still has the
                # other modalities available.
                pred_loo = self.model(inputs, avail_loo, dt, query_dt=query_dt)
                loss = loss + self.criterion(pred_loo, y)

            self.optimizer.zero_grad()
            loss.backward()
            if self.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()
            self.scheduler.step()
            total += loss.item() * len(idx)
            seen += len(idx)
        return total / max(seen, 1)

    @torch.no_grad()
    def _val_epoch(self) -> tuple[float, float]:
        """Validation with all modalities present (the reference case)."""
        self.model.eval()
        total, errs = 0.0, []
        for s in range(0, self.n["val"], self.batch_size):
            idx = torch.arange(s, min(s + self.batch_size, self.n["val"]),
                               device=self.device)
            inputs, avail, dt, y_anchor, y_inst = self._batch(
                "val", idx, drop=False)
            y, query_dt = self._resolve_query(
                idx, dt, y_anchor, y_inst, randomize=False)
            pred = self.model(inputs, avail, dt, query_dt=query_dt)
            total += self.criterion(pred, y).item() * len(idx)
            errs.append(torch.linalg.norm(pred - y, dim=1).cpu())
        errs = torch.cat(errs)
        return total / len(errs), float(errs.mean())

    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict(self, split: str = "val",
                active: set | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Predictions and targets for a split (CPU tensors).

        ``active`` optionally restricts which modalities are available
        (the rest are masked) — handy for "what if sensor X is missing"
        demos. Query time is fixed at the anchor instant.
        """
        self.model.eval()
        preds, tgts = [], []
        n = self.n[split]
        for s in range(0, n, self.batch_size):
            idx = torch.arange(s, min(s + self.batch_size, n),
                               device=self.device)
            inputs, avail, dt, y_anchor, y_inst = self._batch(
                split, idx, drop=False)
            y, query_dt = self._resolve_query(
                idx, dt, y_anchor, y_inst, randomize=False)
            if active is not None:
                avail = {m: (avail[m] if m in active
                             else torch.zeros_like(avail[m]))
                         for m in self.modalities}
            pred = self.model(inputs, avail, dt, query_dt=query_dt)
            preds.append(pred.cpu())
            tgts.append(y.cpu())
        return torch.cat(preds), torch.cat(tgts)

    @torch.no_grad()
    def log_attribution(self, split: str = "val", path_id: int | None = None,
                        max_samples: int | None = None,
                        save: bool = True, verbose: bool = True) -> list[dict]:
        """Per-prediction modality attribution along a sequence.

        For each sample (optionally restricted to one ``path_id`` and sorted
        by timestamp) record: the prediction, the GT, the Euclidean error,
        the cross-attention mass the readout query put on each modality
        (and on CLS), and the fraction of each modality's K instants that
        were actually available. The attention percentages answer "what did
        THIS prediction rely on" — the per-sample version of the subset
        table. Requires ``readout='query'``.

        Returns the list of records; also writes
        ``attribution_<split>[_path<id>].json`` to the run dir when ``save``.
        """
        self.model.eval()
        mods = self.modalities

        ds = {"train": self.dm.train_ds, "val": self.dm.val_ds,
              "test": getattr(self.dm, "test_ds", None)}[split]
        if ds is None:
            raise ValueError(f"split '{split}' has no dataset")

        # Order samples: by timestamp within a chosen path, else split order.
        gt_rows = ds._gt_rows
        order = list(range(self.n[split]))
        if path_id is not None:
            order = [i for i in order if gt_rows[i]["path_id"] == path_id]
            order.sort(key=lambda i: gt_rows[i]["time"])
        if max_samples is not None:
            order = order[:max_samples]

        records: list[dict] = []
        for s in range(0, len(order), self.batch_size):
            chunk = order[s:s + self.batch_size]
            idx = torch.tensor(chunk, device=self.device)
            inputs, avail, dt, y_anchor, y_inst = self._batch(
                split, idx, drop=False)
            y, query_dt = self._resolve_query(
                idx, dt, y_anchor, y_inst, randomize=False)
            pred, attr = self.model.forward_attribution(
                inputs, avail, dt, query_dt=query_dt)
            err = torch.linalg.norm(pred - y, dim=1)
            attn = attr["attn"].cpu()                 # (b, M)
            cls = attr["cls"].cpu()                   # (b,)
            avail_frac = attr["avail_frac"].cpu()     # (b, M)
            pred_c, y_c = pred.cpu(), y.cpu()
            # Decomposed readout exposes the gate + motion contribution.
            gate = attr["gate"].cpu() if "gate" in attr else None
            motion_frac = attr["motion_frac"].cpu() if "motion_frac" in attr else None
            for bi, gi in enumerate(chunk):
                rec = {
                    "sample": int(gi),
                    "path_id": int(gt_rows[gi]["path_id"]),
                    "time": float(gt_rows[gi]["time"]),
                    "pred": [float(pred_c[bi, 0]), float(pred_c[bi, 1])],
                    "gt": [float(y_c[bi, 0]), float(y_c[bi, 1])],
                    "err_m": float(err[bi]),
                    "attn_pct": {m: round(float(attn[bi, mi]) * 100, 1)
                                 for mi, m in enumerate(mods)},
                    "attn_cls_pct": round(float(cls[bi]) * 100, 1),
                    "avail_frac": {m: round(float(avail_frac[bi, mi]), 3)
                                   for mi, m in enumerate(mods)},
                }
                if gate is not None:
                    rec["gate"] = round(float(gate[bi]), 3)
                    rec["motion_frac"] = round(float(motion_frac[bi]), 3)
                records.append(rec)

        if verbose:
            tag = f" path_{path_id:02d}" if path_id is not None else ""
            print(f"\n  Per-prediction modality attribution — {split}{tag}",
                  flush=True)
            cols = "  ".join(f"{m[:5]:>6s}" for m in mods)
            print(f"  {'t(s)':>7s} {'err':>6s} | {cols}   {'CLS':>6s}",
                  flush=True)
            print("  " + "-" * (20 + 9 * (len(mods) + 1)), flush=True)
            # Print up to ~25 rows evenly spaced so long paths stay readable.
            show = records
            if len(records) > 25:
                step = len(records) // 25
                show = records[::step]
            for r in show:
                attn = "  ".join(f"{r['attn_pct'][m]:5.1f}%" for m in mods)
                print(f"  {r['time']:7.2f} {r['err_m']:5.2f}m | {attn}   "
                      f"{r['attn_cls_pct']:5.1f}%", flush=True)
            # Sequence-mean attribution
            mean_attn = {m: round(
                float(np.mean([r["attn_pct"][m] for r in records])), 1)
                for m in mods}
            mean_cls = round(
                float(np.mean([r["attn_cls_pct"] for r in records])), 1)
            mean_err = float(np.mean([r["err_m"] for r in records]))
            print("  " + "-" * (20 + 9 * (len(mods) + 1)), flush=True)
            attn = "  ".join(f"{mean_attn[m]:5.1f}%" for m in mods)
            print(f"  {'MEAN':>7s} {mean_err:5.2f}m | {attn}   "
                  f"{mean_cls:5.1f}%", flush=True)

        if save:
            tag = f"_path{path_id:02d}" if path_id is not None else ""
            (self.run_path / f"attribution_{split}{tag}.json").write_text(
                json.dumps(records, indent=2))
        return records

    @torch.no_grad()
    def evaluate_subsets(self, split: str = "val") -> dict:
        """MAE for every meaningful modality subset — the robustness table.

        Returns ``{subset_label: {"mae": float, "rmse": float}}`` for the
        all-modalities case, each single modality, and each leave-one-out.
        """
        self.model.eval()
        mods = self.modalities
        subsets: list[tuple[str, set]] = [("all", set(mods))]
        for m in mods:
            subsets.append((f"only:{m}", {m}))
        if len(mods) > 2:
            for m in mods:
                subsets.append((f"drop:{m}", set(mods) - {m}))

        results = {}
        n = self.n[split]
        for label, active in subsets:
            errs = []
            for s in range(0, n, self.batch_size):
                idx = torch.arange(s, min(s + self.batch_size, n),
                                   device=self.device)
                inputs, avail, dt, y_anchor, y_inst = self._batch(
                    split, idx, drop=False)
                y, query_dt = self._resolve_query(
                    idx, dt, y_anchor, y_inst, randomize=False)
                avail = {m: (avail[m] if m in active
                             else torch.zeros_like(avail[m]))
                         for m in mods}
                pred = self.model(inputs, avail, dt, query_dt=query_dt)
                errs.append(torch.linalg.norm(pred - y, dim=1).cpu())
            errs = torch.cat(errs)
            results[label] = {
                "mae": float(errs.mean()),
                "rmse": float((errs ** 2).mean().sqrt()),
            }
        (self.run_path / "subsets.json").write_text(json.dumps(results, indent=2))
        return results

    @torch.no_grad()
    def evaluate_all_subsets(self, split: str = "val") -> dict:
        """MAE for *every* non-empty modality subset (2**M − 1 of them).

        The exhaustive version of :meth:`evaluate_subsets` — keyed by a
        ``'+'``-joined label, e.g. ``'imu+wifi'``.
        """
        self.model.eval()
        mods = self.modalities
        results = {}
        n = self.n[split]
        for r in range(1, len(mods) + 1):
            for combo in combinations(mods, r):
                active = set(combo)
                errs = []
                for s in range(0, n, self.batch_size):
                    idx = torch.arange(s, min(s + self.batch_size, n),
                                       device=self.device)
                    inputs, avail, dt, y_anchor, y_inst = self._batch(
                        split, idx, drop=False)
                    y, query_dt = self._resolve_query(
                        idx, dt, y_anchor, y_inst, randomize=False)
                    avail = {m: (avail[m] if m in active
                                 else torch.zeros_like(avail[m]))
                             for m in mods}
                    pred = self.model(inputs, avail, dt, query_dt=query_dt)
                    errs.append(torch.linalg.norm(pred - y, dim=1).cpu())
                errs = torch.cat(errs)
                results["+".join(combo)] = {
                    "mae": float(errs.mean()),
                    "rmse": float((errs ** 2).mean().sqrt()),
                    "n_modalities": r,
                }
        (self.run_path / f"all_subsets_{split}.json").write_text(
            json.dumps(results, indent=2))
        return results

    @torch.no_grad()
    def evaluate_staleness(self, modality: str = "wifi",
                           split: str = "val") -> dict:
        """MAE as a modality goes stale — the test temporal fusion exists for.

        ``modality`` is masked at the ``stale`` most-recent instants and
        kept at the older ones. ``stale=0`` is all-fresh; ``stale=K`` is
        the modality fully gone. A single-instant model can only ever see
        the anchor, so its error jumps to the no-modality level the moment
        ``stale>0``; a temporal model should degrade gracefully by
        propagating motion from the last good fix.
        """
        self.model.eval()
        K = self.n_instants
        mods = self.modalities
        levels = sorted({0, K // 4, K // 2, (3 * K) // 4, K - 1, K})
        levels = [s for s in levels if 0 <= s <= K]
        results = {}
        n = self.n[split]
        for stale in levels:
            errs = []
            for s in range(0, n, self.batch_size):
                idx = torch.arange(s, min(s + self.batch_size, n),
                                   device=self.device)
                inputs, avail, dt, y_anchor, y_inst = self._batch(
                    split, idx, drop=False)
                y, query_dt = self._resolve_query(
                    idx, dt, y_anchor, y_inst, randomize=False)
                am = avail[modality].clone()
                if stale > 0:
                    am[:, K - stale:] = False
                avail = {**avail, modality: am}
                pred = self.model(inputs, avail, dt, query_dt=query_dt)
                errs.append(torch.linalg.norm(pred - y, dim=1).cpu())
            errs = torch.cat(errs)
            results[f"stale={stale}"] = {
                "mae": float(errs.mean()),
                "rmse": float((errs ** 2).mean().sqrt()),
            }
        (self.run_path / f"staleness_{modality}.json").write_text(
            json.dumps(results, indent=2))
        return results
