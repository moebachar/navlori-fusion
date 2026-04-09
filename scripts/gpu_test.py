"""Quick GPU training smoke test — watch nvidia-smi while this runs."""
import sys
sys.path.insert(0, ".")

import torch
from src.pipeline.data import FusionDataModule
from src.pipeline.encoders import Anchor2Vec
from src.pipeline.training import EncoderTrainer

print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

dm = FusionDataModule(
    data_dir="data/async_collection",
    train_paths=[1, 3, 4, 5],
    val_paths=[2],
    test_paths=[13],
    modalities=["wifi"],
    batch_size=512,
    normalize=True,
    wifi_pca=32,
)
dm.setup()
print(f"Train: {len(dm.train_ds)} samples, Val: {len(dm.val_ds)} samples")

enc = Anchor2Vec(n_aps=32, embed_dim=128, n_anchors=64)
trainer = EncoderTrainer(enc, modality="wifi", dm=dm)
print(f"Device: {trainer.device}")

history = trainer.fit(epochs=20, verbose=True)
print(f"\nDone — best val MAE: {history.best_val_mae:.3f}m")

if torch.cuda.is_available():
    print(f"Peak GPU mem: {torch.cuda.max_memory_allocated() / 1e6:.1f} MB")
