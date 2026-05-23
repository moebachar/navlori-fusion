# External Dependencies

Things that the project depends on but does **not** vendor into git. Each one needs an explicit step before the relevant scripts will run.

---

## DPVO (Deep Patch Visual Odometry)

The DPVO research branch (`scripts/run_dpvo_paths.py`, `scripts/extract_dpvo_features.py`, `scripts/diagnostic_dpvo_*.py`, `notebooks/fusion_workbench.ipynb` DPVO sections) uses Teed et al.'s upstream DPVO repo plus a few project-local patches in `external/dpvo/`.

The full upstream repo (~336 MB) is gitignored. Clone it into `external/dpvo_upstream/` before running anything DPVO-related:

```powershell
mkdir external -ErrorAction SilentlyContinue
git clone https://github.com/princeton-vl/DPVO.git external\dpvo_upstream
# pin to a known-good commit (the one this project was tested against)
git -C external\dpvo_upstream checkout <commit-sha>
```

> Pin the upstream commit if you care about reproducibility. The project was last validated against `princeton-vl/DPVO` commit `<set-this-when-you-clone>`. Update this line when you re-pin.

### DPVO weights

```powershell
.\.venv\Scripts\python.exe scripts\fetch_dpvo_weights.py
```

The weights file is large; it is also gitignored and lives under `runs/_weights/`.

### Docker (only for paths that need NVIDIA's CUDA stack the venv lacks)

A self-contained Docker image is used for full DPVO trajectory extraction. The image build/run scripts (`scripts/_dpvo_in_container_*.sh`) assume a Linux host or WSL2; they are not required for the rest of the project.

---

## External datasets

All datasets below are tracked by DVC alongside `data/async_collection/`. If you have access to the project's DVC remote (`D:\dvc_store`), a single `dvc pull` will fetch everything. If not, download the upstream archives and re-run the conversion scripts.

### UJIIndoorLoc

- Source: <https://www.kaggle.com/datasets/giantuji/UjiIndoorLoc>
- License: CC-BY 4.0 (Torres-Sospedra et al., IPIN 2014)
- Place `trainingData.csv` and `validationData.csv` under `data/uji_indoorloc/`. The WiFi encoder evaluation (`scripts/eval_uji_wifi.py`) reads these directly.

### RoNIN

- Source: <https://ronin.cs.sfu.ca/> (download FRDR dataset 538)
- License: CC-BY-NC 4.0 (Yan et al., ICRA 2020)
- Subject `a000` is the default; place the raw extraction under `data/FRDR_dataset_538_download_259_<date>/` and convert:

```powershell
.\.venv\Scripts\python.exe scripts\convert_ronin.py --subject a000
.\.venv\Scripts\python.exe scripts\convert_ronin.py --subject a000 --intra
```

Outputs land under `data/ronin_a000/` and `data/ronin_a000_intra/`.

### IPIN 2024 Track 3

- Source: <https://evarilos.eu/ipin2024> (competition data, registration required)
- Place the raw archive under `data/2024_IPIN_Competition_Track03/` and convert:

```powershell
.\.venv\Scripts\python.exe scripts\convert_ipin2024.py --floor 0
.\.venv\Scripts\python.exe scripts\convert_ipin2024.py --floor -2
.\.venv\Scripts\python.exe scripts\convert_ipin2024.py --floor -2 --intra
```

Outputs land under `data/ipin2024_floor*/`.

### IMUWiFine

- Source: <https://github.com/lixufa/IMUWiFine>
- License: check the upstream repo
- Floor 4 is the converted default:

```powershell
.\.venv\Scripts\python.exe scripts\convert_imuwifine.py --floor 4
```

Output lands under `data/imuwifine_floor4/`.

---

## Open-source baseline code

The SOTA-comparison scripts (`scripts/eval_wlanloc_*.py`, `scripts/eval_ronin_ipin.py`, `scripts/eval_cnnloc_*.py`) load open-source baseline code from its installed location — they do not reimplement it. Install the baselines once:

```powershell
# wlan_localization (sharan-naribole)
pip install git+https://github.com/sharan-naribole/wlan_localization.git

# RoNIN — clone and put on PYTHONPATH (their loader is imported pure)
git clone https://github.com/Sachini/ronin.git external\ronin_upstream
$env:PYTHONPATH += ";$PWD\external\ronin_upstream\source"
```

A documented runtime shim (`np.int` ↔ `np.int_` for numpy ≥ 1.20) is applied at import time rather than patching upstream files.

---

## Webots

The simulation half of the project (data collection only — not required for training or evaluation) targets Webots R2025a. Install separately:

- Source: <https://cyberbotics.com/>
- License: Apache 2.0

The world file lives at `src/simulation/worlds/Tiago++'s world.wbt`. Open it in the Webots GUI and follow the data-collection section in the main README.
