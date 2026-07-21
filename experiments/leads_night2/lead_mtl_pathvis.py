"""MTL with path-classification + WiFi AP visibility aux heads on CLS pooling.

Mechanistic intent: train the FusionTransformer's main (x,y) head together
with two auxiliary self-supervised heads on the pooled CLS embedding:
  - path_cls: predict which training path this sample came from
  - ap_vis  : predict next-step AP presence (binary multi-label)

Both auxiliaries are computed from the LayerNorm output of the readout
('self.base.norm' in FusionTransformer) which is the (B, embed_dim)
representation just before the (x, y) head.
"""
from __future__ import annotations
import os, random, sys, argparse, time
from pathlib import Path
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
REPO = Path('x:/navlori-fusion'); os.chdir(REPO); sys.path.insert(0, str(REPO))
from src.pipeline.fusion.builder import (
    build_datamodule, build_trainer, load_config, build_encoders, build_model,
)

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'


class MTLHeads(nn.Module):
    def __init__(self, d, n_paths, n_aps):
        super().__init__()
        self.path_cls = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, n_paths))
        self.ap_vis = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, n_aps))


def precompute_aux(ds, n_aps):
    pids = np.array([r['path_id'] for r in ds._gt_rows])
    uniq = sorted(set(pids.tolist()))
    p2i = {p: i for i, p in enumerate(uniq)}
    plabel = torch.tensor([p2i[p] for p in pids], dtype=torch.long)
    rssi, _ = ds.get_tensors('wifi')   # (N, win, n_aps) — wifi window=1
    # Last instant of window; presence = any non-zero across last-instant features
    last = rssi[:, -1, :]              # (N, n_aps)
    presence = (last.abs() > 1e-3).float()
    # nxt_presence[i] = presence[i+1] when same path; else fall back to current
    nxt = presence.clone()
    for i in range(len(ds) - 1):
        if pids[i] == pids[i + 1]:
            nxt[i] = presence[i + 1]
    return plabel, nxt, len(uniq)


def run_mtl_epoch(trainer, heads, plab, nxt, lam_path, lam_vis, opt, sched=None):
    """One training epoch that mirrors FusionTrainer._train_epoch but adds aux losses.

    Uses the trainer's cached tensors (X/A/y/inst_*) directly so we don't
    need a DataLoader.
    """
    model = trainer.model
    model.train()
    heads.train()
    n_tr = trainer.n["train"]
    perm = torch.randperm(n_tr, device=trainer.device)
    steps = max(1, n_tr // trainer.batch_size)
    total, seen = 0.0, 0

    # Hook self.norm output to capture (B, embed_dim) just before the head.
    captured = {}

    def _hook(_m, _i, o):
        captured['z'] = o

    handle = model.norm.register_forward_hook(_hook)

    try:
        for s in range(steps):
            idx = perm[s * trainer.batch_size:(s + 1) * trainer.batch_size]
            if len(idx) == 0:
                continue
            inputs, avail, dt, y_anchor, y_inst = trainer._batch(
                "train", idx, drop=True)
            y, query_dt = trainer._resolve_query(
                idx, dt, y_anchor, y_inst, randomize=True)
            captured.clear()
            pred = model(inputs, avail, dt, query_dt=query_dt)
            loss_main = trainer.criterion(pred, y)

            # Aux heads on the captured pooled embedding (B, D).
            loss = loss_main
            if 'z' in captured:
                z = captured['z']
                cls_logits = heads.path_cls(z)
                vis_logits = heads.ap_vis(z)
                loss_path = F.cross_entropy(cls_logits, plab[idx])
                loss_vis = F.binary_cross_entropy_with_logits(vis_logits, nxt[idx])
                loss = loss + lam_path * loss_path + lam_vis * loss_vis

            opt.zero_grad()
            loss.backward()
            if trainer.grad_clip > 0:
                nn.utils.clip_grad_norm_(
                    list(model.parameters()) + list(heads.parameters()),
                    trainer.grad_clip,
                )
            opt.step()
            if sched is not None:
                try:
                    sched.step()
                except Exception:
                    pass
            total += loss.item() * len(idx)
            seen += len(idx)
    finally:
        handle.remove()

    return total / max(seen, 1)


def main(dataset='msiln_site1_b1', seed=42, epochs=12,
         lam_path=0.1, lam_vis=0.2):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    cfg = load_config(dataset)
    # Keep temporal modest for time budget; use config default for stride/etc.
    cfg.temporal.n_instants = 4
    cfg.train.modality_balanced_loss = False

    dm = build_datamodule(cfg)
    encs, vision = build_encoders(cfg, dm)
    extra_inputs = {}
    if vision is not None:
        from src.pipeline.fusion.builder import extract_vision_tokens
        extra_inputs = extract_vision_tokens(dm, vision, device=DEV)
    base = build_model(cfg, encs)

    n_aps = int(dm.train_ds.feature_dims['wifi'])
    plab, nxt, n_paths = precompute_aux(dm.train_ds, n_aps)
    plab = plab.to(DEV); nxt = nxt.to(DEV)

    heads = MTLHeads(cfg.model.embed_dim, n_paths, n_aps).to(DEV)

    run_dir = REPO / 'runs' / 'experiments' / f'mtl_pathvis_{dataset}_s{seed}'
    run_dir.mkdir(parents=True, exist_ok=True)
    trainer = build_trainer(
        cfg, base, dm,
        extra_inputs=extra_inputs or None,
        run_dir=str(run_dir),
    )

    opt = torch.optim.AdamW(
        list(base.parameters()) + list(heads.parameters()),
        lr=cfg.train.lr, weight_decay=cfg.train.weight_decay,
    )
    # Replace trainer optimizer's scheduler with a simple OneCycleLR over our opt
    steps_per_epoch = max(1, trainer.n["train"] // trainer.batch_size)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=cfg.train.lr, epochs=epochs,
        steps_per_epoch=steps_per_epoch, pct_start=0.3,
    )
    # Inject scheduler stepping by wrapping run_mtl_epoch — keep simple here.

    best_val = float('inf'); best_state = None
    t0 = time.time()
    for ep in range(epochs):
        train_loss = run_mtl_epoch(
            trainer, heads, plab, nxt, lam_path, lam_vis, opt, sched=sched,
        )
        pv, tv = trainer.predict('val')
        vm = float((pv - tv).norm(dim=1).mean())
        if vm < best_val:
            best_val = vm
            best_state = {k: v.detach().cpu().clone()
                          for k, v in base.state_dict().items()}
        print(f'[mtl_pathvis] ep={ep} train_loss={train_loss:.4f} val_mae={vm:.3f}',
              flush=True)

    if best_state is not None:
        base.load_state_dict(best_state)
        base.to(DEV)
    pt, tt = trainer.predict('test')
    tm = float((pt - tt).norm(dim=1).mean())
    pv, tv = trainer.predict('val')
    vm = float((pv - tv).norm(dim=1).mean())
    print(f'RESULT: dataset={dataset} seed={seed} val={best_val:.3f} test={tm:.3f}',
          flush=True)
    print(f'[mtl_pathvis] elapsed_sec={time.time() - t0:.1f}', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--epochs', type=int, default=12)
    ap.add_argument('--dataset', type=str, default='msiln_site1_b1')
    args = ap.parse_args()
    main(dataset=args.dataset, seed=args.seed, epochs=args.epochs)
