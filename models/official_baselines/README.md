# Official / public-source baseline adapters

These adapters are for paper-grade baseline comparisons. They do **not** include
project-local `xxx-style` fallback models.

## Source repositories

Clone public repositories into `external_baselines/`:

```bash
bash scripts/clone_official_baselines.sh
python check_official_baseline_sources.py
```

Alternatively set environment variables:

```bash
export ECG_BASELINE_MEDFORMER_ROOT=/path/to/Medformer
export ECG_BASELINE_PATCHTST_ROOT=/path/to/PatchTST
export ECG_BASELINE_ITRANSFORMER_ROOT=/path/to/iTransformer
export ECG_BASELINE_FEDFORMER_ROOT=/path/to/FEDformer
export ECG_BASELINE_FOURIERGNN_ROOT=/path/to/FourierGNN
export ECG_BASELINE_MEDGNN_ROOT=/path/to/MedGNN
```

## Registered names

- `official_medformer`
- `official_patchtst`
- `official_itransformer`
- `official_fedformer`
- `official_fouriergnn`
- `official_medgnn`

All wrappers reconstruct ECG using `x = x_hf + x_lf`, convert `[B, 12, T]` to
`[B, T, 12]`, preserve the external backbone, and adapt only the final task head
when the original repository is forecasting-oriented.

## Paper note template

> We adapted publicly available implementations of baseline models to the
> PTB-XL multi-label diagnosis setting. Only the input interface and final
> classification head were modified, while the core backbone architectures were
> preserved. All models were trained using the same patient-level splits,
> BCEWithLogitsLoss, and validation-calibrated per-class thresholds.

Before submission, record the exact commit of every external repository.
