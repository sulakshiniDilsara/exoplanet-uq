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

| 3. Clean each light curve (quality flags, quarter normalising, upward outliers, masked flatten) and make a global (2001 bins) and local (201 bins) view per signal | `python scripts/03_make_views.py` | `data/processed/views.npz`, `data/processed/plots/`, `metadata/processed_signals.csv`, `metadata/processed_info.json` |

Step 2 can be stopped and run again: stars that are already complete are skipped.
Step 3 needs no internet and never uses the labels, so planets and false positives are cleaned the same way.

## Folders

| Folder | What is inside | In git? |
|---|---|---|
| `src/exoplanet_uq/` | Reusable project code | Yes |
| `tests/` | Automatic checks for the code | Yes |
| `scripts/` | Pipeline steps you run by hand | Yes |
| `metadata/` | Small tables: catalogue, labels, splits | Yes |
| `data/` | Downloaded and processed light curves | No (too big) |
| `Docs/` | Private learning notes and documents | No (private) |
