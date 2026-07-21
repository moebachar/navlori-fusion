"""Profile every step of the training pipeline to find bottlenecks."""
import sys, time
sys.path.insert(0, ".")

import torch
import torch.nn as nn
from src.pipeline.data import FusionDataModule
from src.pipeline.encoders import WiFiNet

print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# --- 1. DataModule setup ---
t0 = time.time()
dm = FusionDataModule(
    data_dir="data/async_collection",
    train_paths=[1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    val_paths=[2, 13, 14],
    test_paths=[15, 16, 17],
    modalities=["wifi"],
    batch_size=512,
    num_workers=0,
    normalize=True,
    wifi_pca=32,
)
dm.setup()
print(f"\n[1] DataModule setup: {time.time()-t0:.2f}s")
print(f"    Train: {len(dm.train_ds)} samples, Val: {len(dm.val_ds)} samples")

# --- 2. DataLoader creation ---
t0 = time.time()
train_loader = dm.train_dataloader()
val_loader = dm.val_dataloader()
print(f"[2] DataLoader creation: {time.time()-t0:.4f}s")
print(f"    Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

# --- 3. Iterate one full epoch of batches (no model) ---
t0 = time.time()
n_batches = 0
for batch in train_loader:
    n_batches += 1
print(f"[3] Iterate train loader (data only): {time.time()-t0:.4f}s ({n_batches} batches)")

# --- 4. Iterate + move to GPU ---
device = "cuda" if torch.cuda.is_available() else "cpu"
t0 = time.time()
for batch in train_loader:
    x = batch["wifi"].to(device)
    y = batch["target"].to(device)
print(f"[4] Iterate + .to({device}): {time.time()-t0:.4f}s")

# --- 5. Model creation ---
enc = WiFiNet(n_aps=32, embed_dim=128, n_anchors=64).to(device)
head = nn.Linear(128, 2).to(device)
criterion = nn.HuberLoss(delta=0.5)
params = list(enc.parameters()) + list(head.parameters())
optimizer = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-4)

# --- 6. Forward pass only ---
t0 = time.time()
for batch in train_loader:
    x = batch["wifi"].to(device)
    y = batch["target"].to(device)
    with torch.no_grad():
        z = enc(x)
        if z.ndim == 3:
            z = z.mean(dim=1)
        pred = head(z)
        loss = criterion(pred, y)
if torch.cuda.is_available():
    torch.cuda.synchronize()
print(f"[5] Forward pass (1 epoch): {time.time()-t0:.4f}s")

# --- 7. Full train step (fwd + bwd + optim) ---
t0 = time.time()
for batch in train_loader:
    x = batch["wifi"].to(device)
    y = batch["target"].to(device)
    z = enc(x)
    if z.ndim == 3:
        z = z.mean(dim=1)
    pred = head(z)
    loss = criterion(pred, y)
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(params, 1.0)
    optimizer.step()
if torch.cuda.is_available():
    torch.cuda.synchronize()
print(f"[6] Full train step (1 epoch): {time.time()-t0:.4f}s")

# --- 8. Validation epoch ---
t0 = time.time()
enc.eval()
head.eval()
with torch.no_grad():
    for batch in val_loader:
        x = batch["wifi"].to(device)
        y = batch["target"].to(device)
        z = enc(x)
        if z.ndim == 3:
            z = z.mean(dim=1)
        pred = head(z)
        loss = criterion(pred, y)
if torch.cuda.is_available():
    torch.cuda.synchronize()
print(f"[7] Validation epoch: {time.time()-t0:.4f}s")

# --- 9. Evaluation harness ---
from src.pipeline.evaluation import evaluate_encoder, print_report
t0 = time.time()
results = evaluate_encoder(
    encoder=enc,
    train_loader=train_loader,
    val_loader=val_loader,
    modality="wifi",
    device=device,
)
print(f"[8] Full evaluation harness: {time.time()-t0:.2f}s")

# --- 10. OneCycleLR scheduler overhead ---
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=1e-3, epochs=100, steps_per_epoch=len(train_loader), pct_start=0.3,
)
enc.train()
head.train()
t0 = time.time()
for batch in train_loader:
    x = batch["wifi"].to(device)
    y = batch["target"].to(device)
    z = enc(x)
    if z.ndim == 3:
        z = z.mean(dim=1)
    pred = head(z)
    loss = criterion(pred, y)
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(params, 1.0)
    optimizer.step()
    scheduler.step()
if torch.cuda.is_available():
    torch.cuda.synchronize()
print(f"[9] Train step + scheduler (1 epoch): {time.time()-t0:.4f}s")

print(f"\nExpected 82 epochs at step [6] rate: {82 * float(time.time()-t0):.1f}s total")
