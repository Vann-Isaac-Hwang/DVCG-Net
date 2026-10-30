# DVCG-Net

Research code for DVCG-Net, a dual-view contrastive graph network for ECG classification, and its baseline evaluation.
The repository contains code only; ECG datasets, processed tensors, trained
weights, logs, plots, and third-party baseline source repositories are excluded.

## Setup

Use Python 3.9 or newer. Install a PyTorch build appropriate for your CPU/GPU,
then install the remaining dependencies:

```bash
python -m pip install -r requirements.txt
cp config.example.yaml config.yaml
```

Put datasets under `data/`, or set `DUAL_ECG_DATA_ROOT` to a data directory.
The default layout expected by the registry is:

```text
data/
  CPSC_Data/processed_sr100_freq10/
  PTBXL_Data/processed_sr100_freq10/
  Chapman_Data/processed_sr100_freq10/
```

Dataset files are not distributed with this repository. Obtain them from their
official sources and follow each dataset's terms of use. Some preprocessing
steps require the corresponding raw dataset files.

## External Baselines

The official baseline adapters load source repositories at runtime. Download
those sources separately into the ignored `external_baselines/` directory:

```bash
bash scripts/clone_official_baselines.sh
python check_official_baseline_sources.py
python smoke_test_models.py --models official_patchtst --device cpu --batch_size 2 --seq_len 1000
```

The clone script retrieves upstream repositories and does not bundle or modify
their source in this project. Review each upstream license and record exact
commit hashes when reporting results. Some baselines have additional upstream
Python dependencies.

## Training

Check the CPSC runner's Python environment and dataset path first:

```bash
./run_cpsc2018_5fold.sh --check-env
MODELS=official_patchtst START_FOLD=0 END_FOLD=5 EPOCHS=50 PATIENCE=10 \
  ./run_cpsc2018_5fold.sh
```

Set `DUAL_ECG_DATA_ROOT`, `CPSC_DATA_DIR`, or `CHAPMAN_DATA_DIR` to override
the default data locations. Training outputs are written under ignored local
`checkpoints/`, `logs_*`, and `results/` directories.
