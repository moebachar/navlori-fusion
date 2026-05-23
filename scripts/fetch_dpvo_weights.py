"""Download DPVO pretrained weights into ``runs/_weights/dpvo.pth``.

The Princeton-VL DPVO release ships a single ``models.zip`` on Google Drive
containing the inference weights. This script pulls that zip via ``gdown``
(same source the upstream Dockerfile uses), extracts ``dpvo.pth``, and
drops it next to the other vendored weights (``ace_encoder_pretrained.pt``
etc.).

Usage (from repo root):

    python scripts/fetch_dpvo_weights.py
    python scripts/fetch_dpvo_weights.py --force   # re-download even if present

Idempotent: skips download if ``runs/_weights/dpvo.pth`` already exists.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS_DIR = ROOT / "runs" / "_weights"
TARGET = WEIGHTS_DIR / "dpvo.pth"

# Same Drive ID the DPVO Dockerfile uses (`download_models_and_data.sh`).
GDRIVE_ID = "1dRqftpImtHbbIPNBIseCv9EvrlHEnjhX"


def _ensure_gdown() -> None:
    try:
        import gdown  # noqa: F401
    except ImportError:
        print("gdown not installed; installing into the active venv...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--force", action="store_true",
                   help="Re-download even if dpvo.pth already exists")
    args = p.parse_args()

    if TARGET.exists() and not args.force:
        size_mb = TARGET.stat().st_size / (1024 * 1024)
        print(f"{TARGET} already present ({size_mb:.1f} MB). Use --force to re-download.")
        return

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    _ensure_gdown()
    import gdown

    with tempfile.TemporaryDirectory() as td:
        zip_path = Path(td) / "models.zip"
        print(f"Downloading models.zip from gdrive (id={GDRIVE_ID})...")
        gdown.download(id=GDRIVE_ID, output=str(zip_path), quiet=False)

        if not zip_path.exists():
            sys.exit(
                "Download failed. Check network/gdrive access. You can also "
                "extract dpvo.pth manually from the navlori_dpvo Docker image "
                "with: docker cp <container>:/DPVO/dpvo.pth runs/_weights/dpvo.pth"
            )

        print(f"Extracting dpvo.pth from {zip_path}...")
        with zipfile.ZipFile(zip_path) as zf:
            members = zf.namelist()
            # The release ships dpvo.pth at the zip root, but be defensive.
            cands = [m for m in members if m.endswith("dpvo.pth")]
            if not cands:
                sys.exit(f"dpvo.pth not found in archive. Members: {members}")
            zf.extract(cands[0], path=td)
            extracted = Path(td) / cands[0]
            shutil.move(str(extracted), TARGET)

    size_mb = TARGET.stat().st_size / (1024 * 1024)
    print(f"\nDone. Wrote {TARGET}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
