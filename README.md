# Bayesian Uncertainty Quantification for Exoplanet Detection

Individual research component of SLIIT project J26-IT-449,
"Explainable AI for Astronomical Discovery using XAI and Uncertainty Quantification".

The goal: train a 1D-CNN / LSTM to classify Kepler transit signals, add
Monte Carlo Dropout so each prediction also has an uncertainty, and check
whether the model's confidence is honest (reliability diagrams, ECE,
Brier score, NLL, temperature scaling).

## Set up

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -e ".[dev]"
```

## Run the tests

```bash
pytest
ruff check .
```

## Pipeline steps

| Step | Script | Output |
|---|---|---|
| 1. Download the KOI catalogue and assign labels | `python scripts/01_download_koi.py` | `metadata/koi_*.csv`, `metadata/koi_download_info.json` |
| 2. Pick the pilot stars (150, seed 42) and download their Kepler long cadence light curves | `python scripts/02_download_lightcurves.py` | `data/raw/kepler/<kepid>/*.fits`, `metadata/lightcurve_manifest.csv`, `metadata/lightcurve_download_info.json` |

Step 2 can be stopped and run again: stars that are already complete are skipped.

## Folders

| Folder | What is inside | In git? |
|---|---|---|
| `src/exoplanet_uq/` | Reusable project code | Yes |
| `tests/` | Automatic checks for the code | Yes |
| `scripts/` | Pipeline steps you run by hand | Yes |
| `metadata/` | Small tables: catalogue, labels, splits | Yes |
| `data/` | Downloaded and processed light curves | No (too big) |
| `Docs/` | Private learning notes and documents | No (private) |
