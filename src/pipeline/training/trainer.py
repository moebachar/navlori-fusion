"""Single-modality encoder trainer.

Trains an encoder + linear position head end-to-end, then runs the
full evaluation harness. Follows configs/training/default.yaml defaults.

Usage::

    trainer = EncoderTrainer(encoder, modality="wifi", dm=datamodule)
    history = trainer.fit(epochs=100)
    results = trainer.evaluate()
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.pipeline.evaluation.encoder_eval import evaluate_encoder, print_report


@dataclass
class TrainHistory:
    """Stores per-epoch metrics."""
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    val_mae: list[float] = field(default_factory=list)
    best_epoch: int = 0
    best_val_mae: float = float("inf")
    elapsed_sec: float = 0.0


class PositionHead(nn.Module):
    """Simple linear head: embed_dim → (x, y)."""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.head = nn.Linear(embed_dim, 2)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.head(z)


class EncoderTrainer:
    """Train a single-modality encoder to predict (x, y) position.

    Parameters
    ----------
    encoder : nn.Module
        The modality encoder (e.g. Anchor2Vec, IMUCNN, OdomCNN).
    modality : str
        Which key to read from the dataloader batch (e.g. "wifi", "imu").
    dm : FusionDataModule
        DataModule with setup() already called.
    lr : float
        Learning rate (default 1e-3).
    weight_decay : float
        AdamW weight decay (default 1e-4).
    loss_fn : str
        Loss function: "huber" or "mse" (default "huber").
    huber_delta : float
        Delta for Huber loss (default 0.5).
    grad_clip : float
        Gradient clipping max norm (default 1.0).
    patience : int
        Early stopping patience in epochs (default 25).
    device : str
        Device: "auto", "cuda", "cpu" (default "auto").
    """

    def __init__(
        self,
        encoder: nn.Module,
        modality: str,
        dm,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        loss_fn: str = "huber",
        huber_delta: float = 0.5,
        grad_clip: float = 1.0,
        patience: int = 25,
        device: str = "auto",
        run_dir: str | Path = "runs",
    ):
        self.modality = modality
        self.dm = dm

        # Device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Model: encoder + position head
        self.encoder = encoder.to(self.device)
        self.head = PositionHead(encoder.embed_dim).to(self.device)

        # Optimizer — only train parameters that require grad
        # Build once, reuse every batch (avoid per-step list allocation)
        self._all_params = list(self.encoder.parameters()) + list(self.head.parameters())
        trainable = [p for p in self._all_params if p.requires_grad]
        self.optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=weight_decay)

        # Loss
        if loss_fn == "huber":
            self.criterion = nn.HuberLoss(delta=huber_delta)
        elif loss_fn == "mse":
            self.criterion = nn.MSELoss()
        else:
            raise ValueError(f"Unknown loss: {loss_fn}")

        self.grad_clip = grad_clip
        self.patience = patience
        self.scheduler = None  # set in fit()

        # Run persistence
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_id = f"{modality}_{ts}"
        self.run_path = Path(run_dir) / self.run_id
        self.run_path.mkdir(parents=True, exist_ok=True)
        self._metrics_file = self.run_path / "metrics.jsonl"

    def _make_loaders(self) -> tuple[DataLoader, DataLoader]:
        """Build DataLoaders.

        - Tabular modalities: TensorDataset from pre-stacked cache (pure tensor
          slice, zero disk I/O per batch).
        - Camera: extract frozen ViT backbone features once, then use a
          TensorDataset of (768-dim features, targets) for training only the
          projection head. Eliminates repeated image decoding from disk.
        """
        bs = self.dm.batch_size
        pin = torch.cuda.is_available()

        if self.modality == "camera":
            from src.pipeline.encoders.vision import VisionViT
            if not isinstance(self.encoder, VisionViT):
                # Fallback: generic camera encoder, load images normally
                return self.dm.train_dataloader(), self.dm.val_dataloader()

            # Pre-extract frozen backbone features.
            # Features are cached to disk so extraction only runs once ever.
            cache_dir = self.dm.data_dir / ".vit_cache"
            cache_tr = str(cache_dir / "train_features.pt")
            cache_va = str(cache_dir / "val_features.pt")

            if not hasattr(self, "_vision_cache"):
                print("  Extracting backbone features (one-time, saved to disk)...", flush=True)
                raw_tr = self.dm.train_dataloader()
                raw_va = self.dm.val_dataloader()
                feat_tr, y_tr = self.encoder.extract_backbone_features(
                    raw_tr, self.device, cache_path=cache_tr,
                )
                feat_va, y_va = self.encoder.extract_backbone_features(
                    raw_va, self.device, cache_path=cache_va,
                )
                self._vision_cache = (feat_tr, y_tr, feat_va, y_va)
                print(f"  Ready: {len(feat_tr)} train + {len(feat_va)} val vectors.", flush=True)
            else:
                feat_tr, y_tr, feat_va, y_va = self._vision_cache

            train_loader = DataLoader(
                TensorDataset(feat_tr, y_tr), batch_size=bs, shuffle=True,
                num_workers=0, pin_memory=pin,
            )
            val_loader = DataLoader(
                TensorDataset(feat_va, y_va), batch_size=bs, shuffle=False,
                num_workers=0, pin_memory=pin,
            )

            # Patch encoder.forward to skip the frozen backbone during training
            # (input is now 768-dim features, not raw images)
            self._vision_head_only = True
            return train_loader, val_loader

        # Tabular: TensorDataset from pre-stacked cache
        X_tr, y_tr = self.dm.train_ds.get_tensors(self.modality)
        X_va, y_va = self.dm.val_ds.get_tensors(self.modality)
        train_loader = DataLoader(
            TensorDataset(X_tr, y_tr), batch_size=bs, shuffle=True,
            num_workers=0, pin_memory=pin,
        )
        val_loader = DataLoader(
            TensorDataset(X_va, y_va), batch_size=bs, shuffle=False,
            num_workers=0, pin_memory=pin,
        )
        self._vision_head_only = False
        return train_loader, val_loader

    def fit(self, epochs: int = 100, verbose: bool = True) -> TrainHistory:
        """Train the encoder for N epochs with early stopping.

        Returns:
            TrainHistory with per-epoch metrics.
        """
        train_loader, val_loader = self._make_loaders()

        # OneCycleLR scheduler
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.optimizer.defaults["lr"],
            epochs=epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.3,
        )

        history = TrainHistory()
        patience_counter = 0
        best_state = None
        t0 = time.time()

        # Write run metadata
        meta = {
            "run_id": self.run_id,
            "modality": self.modality,
            "device": self.device,
            "epochs": epochs,
            "lr": self.optimizer.defaults["lr"],
            "started_at": datetime.now().isoformat(),
            "n_train": len(self.dm.train_ds),
            "n_val": len(self.dm.val_ds),
        }
        (self.run_path / "meta.json").write_text(json.dumps(meta, indent=2))
        self._metrics_file.unlink(missing_ok=True)  # fresh log

        for epoch in range(epochs):
            # --- Train ---
            train_loss = self._train_epoch(train_loader)
            history.train_loss.append(train_loss)

            # --- Validate ---
            val_loss, val_mae = self._val_epoch(val_loader)
            history.val_loss.append(val_loss)
            history.val_mae.append(val_mae)

            # --- Early stopping ---
            if val_mae < history.best_val_mae:
                history.best_val_mae = val_mae
                history.best_epoch = epoch
                patience_counter = 0
                best_state = {
                    "encoder": {k: v.cpu().clone() for k, v in self.encoder.state_dict().items()},
                    "head": {k: v.cpu().clone() for k, v in self.head.state_dict().items()},
                }
            else:
                patience_counter += 1

            # Write epoch metrics to JSONL (dashboard reads this live)
            lr = self.scheduler.get_last_lr()[0]
            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_mae": val_mae,
                "lr": lr,
                "is_best": patience_counter == 0,
                "t": time.time() - t0,
            }
            with open(self._metrics_file, "a") as f:
                f.write(json.dumps(row) + "\n")

            if verbose and (epoch % 10 == 0 or epoch == epochs - 1 or patience_counter == self.patience):
                print(
                    f"  Epoch {epoch:3d}/{epochs}  "
                    f"train_loss={train_loss:.4f}  "
                    f"val_loss={val_loss:.4f}  "
                    f"val_mae={val_mae:.3f}m  "
                    f"lr={lr:.2e}  "
                    f"{'*' if patience_counter == 0 else ''}"
                )

            if patience_counter >= self.patience:
                if verbose:
                    print(f"  Early stopping at epoch {epoch} (best={history.best_epoch})")
                break

        # Restore best weights
        if best_state is not None:
            self.encoder.load_state_dict(best_state["encoder"])
            self.head.load_state_dict(best_state["head"])
            self.encoder.to(self.device)
            self.head.to(self.device)

        history.elapsed_sec = time.time() - t0
        if verbose:
            print(f"  Done in {history.elapsed_sec:.1f}s — best val MAE: {history.best_val_mae:.3f}m (epoch {history.best_epoch})")

        # Save history + encoder weights
        hist_dict = {
            "run_id": self.run_id,
            "modality": self.modality,
            "best_epoch": history.best_epoch,
            "best_val_mae": history.best_val_mae,
            "elapsed_sec": history.elapsed_sec,
            "train_loss": history.train_loss,
            "val_loss": history.val_loss,
            "val_mae": history.val_mae,
            "finished_at": datetime.now().isoformat(),
        }
        (self.run_path / "history.json").write_text(json.dumps(hist_dict, indent=2))
        torch.save(self.encoder.state_dict(), self.run_path / "encoder.pt")
        torch.save(self.head.state_dict(), self.run_path / "head.pt")

        return history

    def evaluate(self) -> dict:
        """Run the full 6-metric evaluation harness on the trained encoder."""
        train_loader, val_loader = self._make_loaders()

        # For vision: evaluation uses cached backbone features + trained head.
        # We wrap encoder.head as the "encoder" so extract_embeddings works correctly.
        if getattr(self, "_vision_head_only", False):
            eval_encoder = self.encoder.head
        else:
            eval_encoder = self.encoder

        results = evaluate_encoder(
            encoder=eval_encoder,
            train_loader=train_loader,
            val_loader=val_loader,
            modality=self.modality,
            device=self.device,
        )
        print_report(results, modality=self.modality)

        # Persist eval results (convert numpy floats for JSON serialisation)
        def _to_python(obj):
            if hasattr(obj, "item"):
                return obj.item()
            if isinstance(obj, dict):
                return {k: _to_python(v) for k, v in obj.items()}
            return obj

        (self.run_path / "eval.json").write_text(
            json.dumps(_to_python(results), indent=2)
        )
        return results

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _unpack(batch, modality: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Unpack a batch from either TensorDataset (tuple) or FusionDataset (dict)."""
        if isinstance(batch, (list, tuple)):
            return batch[0], batch[1]
        return batch[modality], batch["target"]

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """Forward through encoder. When vision features are pre-cached,
        skip the frozen backbone and call only the projection head."""
        if getattr(self, "_vision_head_only", False):
            z = self.encoder.head(x)  # x is already 768-dim backbone output
        else:
            z = self.encoder(x)
        if z.ndim == 3:
            z = z.mean(dim=1)
        return z

    def _train_epoch(self, loader: DataLoader) -> float:
        self.encoder.train()
        self.head.train()
        total_loss = 0.0
        n = 0

        for batch in loader:
            x, y = self._unpack(batch, self.modality)
            x, y = x.to(self.device), y.to(self.device)

            z = self._encode(x)
            pred = self.head(z)
            loss = self.criterion(pred, y)

            self.optimizer.zero_grad()
            loss.backward()
            if self.grad_clip > 0:
                nn.utils.clip_grad_norm_(self._all_params, self.grad_clip)
            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item() * len(y)
            n += len(y)

        return total_loss / n

    def _val_epoch(self, loader: DataLoader) -> tuple[float, float]:
        self.encoder.eval()
        self.head.eval()
        total_loss = 0.0
        all_err = []

        with torch.no_grad():
            for batch in loader:
                x, y = self._unpack(batch, self.modality)
                x, y = x.to(self.device), y.to(self.device)

                z = self._encode(x)
                pred = self.head(z)

                loss = self.criterion(pred, y)
                total_loss += loss.item() * len(y)

                err = torch.sqrt(((pred - y) ** 2).sum(dim=1))
                all_err.append(err.cpu())

        all_err = torch.cat(all_err)
        return total_loss / len(all_err), float(all_err.mean())
