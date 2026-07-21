# Headless DPVO build container — drops Pangolin/DPViewer (visualizers).
#
# Built for NavLoRI evaluation: produce a container that can run DPVO
# inference on a directory of RGB frames + a calibration file, and write
# a TUM-format trajectory out.
#
# Pascal-friendly: TORCH_CUDA_ARCH_LIST set so cuda_corr / cuda_ba /
# lietorch_backends compile for sm_61 (and a few neighbours, in case a
# different host GPU runs the same image later).

ARG CUDA_VERSION=12.1.1
FROM nvidia/cuda:${CUDA_VERSION}-cudnn8-devel-ubuntu20.04

ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    DEBIAN_FRONTEND=noninteractive \
    PATH="/root/miniconda3/bin:${PATH}" \
    TORCH_CUDA_ARCH_LIST="6.1;7.0;7.5;8.0;8.6"

RUN apt-get update && apt-get install -y \
        git wget unzip ca-certificates build-essential \
    && rm -rf /var/lib/apt/lists/*

# --- miniconda ----------------------------------------------------------
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/mc.sh \
    && bash /tmp/mc.sh -b -p /root/miniconda3 \
    && rm /tmp/mc.sh

# --- DPVO source + Eigen -----------------------------------------------
WORKDIR /
RUN git clone --recursive https://github.com/princeton-vl/DPVO.git /DPVO
WORKDIR /DPVO
RUN wget -q https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip \
    && unzip -q eigen-3.4.0.zip -d thirdparty \
    && rm eigen-3.4.0.zip

# --- DPVO conda env (pytorch 2.3.1 + cuda 12.1) ------------------------
# Anaconda added a TOS prompt for the default channels in 2024 that blocks
# `conda env create`; accept up-front so the layer can proceed unattended.
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main \
 && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r \
 && conda env create -f environment.yml

# Make the env the default for subsequent RUN/CMD layers.
SHELL ["conda", "run", "--no-capture-output", "-n", "dpvo", "/bin/bash", "-c"]

# --- build the three custom CUDA extensions ----------------------------
# (cuda_corr, cuda_ba, lietorch_backends — all defined in DPVO/setup.py)
RUN pip install --no-build-isolation .

# --- model checkpoint --------------------------------------------------
# `download_models_and_data.sh` pulls models.zip + a small demo MOV from
# Google Drive via gdown. We only really need models.zip for inference.
RUN pip install --no-cache-dir gdown \
    && gdown 1dRqftpImtHbbIPNBIseCv9EvrlHEnjhX -O /DPVO/models.zip \
    && unzip -q /DPVO/models.zip -d /DPVO \
    && rm /DPVO/models.zip

# Fall back to bash by default; runtime will pass the actual command.
SHELL ["/bin/bash", "-c"]
CMD ["bash", "-lc", "conda run --no-capture-output -n dpvo bash"]
