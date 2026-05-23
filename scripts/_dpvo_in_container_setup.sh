#!/bin/bash
# Run inside the pytorch CUDA dev container. Builds DPVO upstream and runs the
# patch-extraction script. Idempotent: re-running only redoes the inference.
set -e

echo '[setup] apt deps'
apt-get update -qq >/dev/null
apt-get install -y --no-install-recommends \
    libeigen3-dev libgl1-mesa-glx libglib2.0-0 git wget ninja-build >/dev/null

echo '[setup] pip deps'
pip install --no-cache-dir --quiet \
    "torch-scatter==2.1.2" -f https://data.pyg.org/whl/torch-2.3.0+cu121.html
pip install --no-cache-dir --quiet \
    tensorboard numba tqdm einops pypose kornia "numpy==1.26.4" \
    plyfile evo opencv-python yacs pandas

cd /work/dpvo

# DPVO's setup.py expects thirdparty/eigen-3.4.0. Use the system package
# instead by symlinking the system include path.
if [ ! -d thirdparty/eigen-3.4.0 ]; then
    mkdir -p thirdparty
    # Eigen is header-only — symlink the system headers in.
    ln -sfn /usr/include/eigen3 thirdparty/eigen-3.4.0
fi

# Build the three CUDA extensions in place (cuda_corr, cuda_ba, lietorch_backends)
if ! python -c 'import lietorch_backends' 2>/dev/null; then
    echo '[setup] building DPVO CUDA extensions (this takes a few minutes)'
    TORCH_CUDA_ARCH_LIST='6.1' \
        pip install --no-cache-dir --no-build-isolation -e . 2>&1 | tail -8
fi

echo '[setup] sanity import'
python -c 'from dpvo.dpvo import DPVO; from dpvo import altcorr, fastba, lietorch; import torch; print("DPVO ok, cuda:", torch.cuda.is_available(), torch.cuda.get_device_name(0))'

echo '[setup] running inference script'
mkdir -p /work/output
python /work/scripts/_dpvo_in_container_extract.py
