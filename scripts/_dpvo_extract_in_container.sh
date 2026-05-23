#!/bin/bash
# Run inside an Ubuntu/CUDA dev container. Installs Python+PyTorch+DPVO on top
# of the bare nvidia/cuda base, then runs per-path feature extraction over
# every async_collection path. Idempotent: re-running re-uses the cached state
# if the container is kept alive.
#
# Mounts assumed: project root at /work
set -e

echo '[setup] apt deps (python + build tools)'
apt-get update -qq >/dev/null
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev \
    libeigen3-dev libgl1-mesa-glx libglib2.0-0 \
    git wget ninja-build build-essential >/dev/null

if ! command -v python >/dev/null; then
    ln -sf /usr/bin/python3 /usr/local/bin/python
fi

echo '[setup] pip deps'
if ! python -c 'import torch' 2>/dev/null; then
    echo '[setup] installing torch 2.3.0+cu121 (large download, ~2 GB)'
    pip install --no-cache-dir --quiet \
        torch==2.3.0 torchvision==0.18.0 \
        --index-url https://download.pytorch.org/whl/cu121
fi

pip install --no-cache-dir --quiet \
    "torch-scatter==2.1.2" -f https://data.pyg.org/whl/torch-2.3.0+cu121.html
pip install --no-cache-dir --quiet \
    tensorboard numba tqdm einops pypose kornia "numpy==1.26.4" \
    plyfile evo opencv-python yacs pandas

cd /work/external/dpvo_upstream

# DPVO's setup.py expects thirdparty/eigen-3.4.0. Use the system package
# instead by symlinking the system include path.
if [ ! -d thirdparty/eigen-3.4.0 ]; then
    mkdir -p thirdparty
    ln -sfn /usr/include/eigen3 thirdparty/eigen-3.4.0
fi

# CUDA extensions: previous container left compiled .so files in place. We
# avoid `pip install -e .` entirely (which would re-trigger CUDAExtension
# build in a base image without nvcc); the extraction script puts the dir
# on sys.path and the .so files are siblings.
echo '[setup] verifying cached CUDA extensions import'
PYTHONPATH=/work/external/dpvo_upstream:$PYTHONPATH \
    python -c 'import torch; import lietorch_backends, cuda_corr, cuda_ba; print("cached extensions import OK (torch", torch.__version__, ")")'

echo '[setup] sanity import'
PYTHONPATH=/work/external/dpvo_upstream:$PYTHONPATH \
    python -c 'import torch; from dpvo.dpvo import DPVO; from dpvo import altcorr, fastba, lietorch; print("DPVO ok, cuda:", torch.cuda.is_available(), torch.cuda.get_device_name(0))'

echo '[setup] running per-path feature extraction'
PYTHONPATH=/work/external/dpvo_upstream:$PYTHONPATH \
    python /work/scripts/extract_dpvo_features.py "$@"
