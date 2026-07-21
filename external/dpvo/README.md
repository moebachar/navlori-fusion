# Vendored DPVO subset

This directory contains a minimal subset of [Princeton-VL/DPVO][upstream]
copied at commit-time from `dpvo/extractor.py`. We use it as a **frozen
feature extractor** inside `src/pipeline/encoders/dpvo_motion.py`; nothing
else from DPVO is needed here on the host (no CUDA extensions, no
`lietorch`, no bundle adjustment).

The full DPVO pipeline is still used **offline in Docker** via
`scripts/run_dpvo_paths.py` — that's a separate concern and stays as-is.

## What's in the file

- `BasicEncoder4` — the stride-4, 128-channel matching-feature extractor.
  This is `Patchifier.fnet` upstream; we initialise it with the matching
  weights from `dpvo.pth` and freeze it.
- `BasicEncoder`, `ResidualBlock` — companion classes the above depends on.

## Pretrained weights

We do **not** vendor `dpvo.pth` (~30 MB binary). Run

    python scripts/fetch_dpvo_weights.py

once to download it into `runs/_weights/dpvo.pth`. The script pulls from
the same Google Drive link the upstream Dockerfile uses.

[upstream]: https://github.com/princeton-vl/DPVO
