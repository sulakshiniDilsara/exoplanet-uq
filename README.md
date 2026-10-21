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
pip install -r requirements-lock.txt   # exact versions, CPU-only PyTorch
pip install -e . --no-deps
```

The lock file points pip to PyTorch's CPU package index, so it installs the
small CPU-only build of PyTorch (no GPU is needed for this project).

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
| 4. Split the kept signals into train / val / test (70 / 15 / 15) by star, stratified by planet, seed 42 | `python scripts/04_split.py` | `metadata/split.csv`, `metadata/split_info.json` |
| 5. Train the two-branch 1D-CNN (global + local view) on train, early stopping on val | `python scripts/05_train.py` | `data/models/cnn_pilot.pt`, `data/models/loss_curve.png`, `metadata/train_history.csv`, `metadata/train_info.json` |
| 6. MC Dropout on val: pick T (smallest of 10/20/50/100 within 0.01 of T = 1000), save mean, std and entropies per signal | `python scripts/06_mc_dropout.py` | `metadata/mc_val_predictions.csv`, `metadata/mc_info.json`, `data/models/mc_*.png` |
| 7. Calibration on val: reliability diagram, ECE (10 bins, bootstrap range), Brier, NLL; temperature scaling only if it can be fitted honestly | `python scripts/07_calibration.py` | `metadata/calibration_val.json`, `data/models/reliability_val.png` |
| 8. Classification scores (accuracy, precision, recall, F1, ROC-AUC, confusion matrix), follow-up ranking by uncertainty, correct-vs-wrong uncertainty (val) | `python scripts/08_evaluate.py` | `metadata/evaluation_val.json`, `metadata/follow_up_val.csv`, `data/models/confusion_val.png` |

Step 2 can be stopped and run again: stars that are already complete are skipped.
Step 3 needs no internet and never uses the labels, so planets and false positives are cleaned the same way.
Step 4 splits stars, not signals, so no star is in two sets (no data leakage). Model work uses only train and val; the test set is opened once, at the end.

## Research dashboard (Step 9)

A local web page that shows the model and the saved results. Run it **from the project root**, so Streamlit finds `.streamlit/config.toml`:

```bash
streamlit run app/dashboard.py
```

Then open http://localhost:8501. It needs no internet and no new data: it uses the committed model (`data/models/cnn_pilot.pt`), views (`data/processed/views.npz`) and `metadata/` files.

| Page | What it shows |
|---|---|
| Overview | Key numbers, workflow, current status |
| Signal Explorer | Pick a train or val signal; the model runs live with MC Dropout (T from Step 6) and shows its answer, its uncertainty and the light-curve views |
| Uncertainty | Answer vs spread for the val set, how T was chosen, aleatoric / epistemic parts |
| Follow-up List | Val signals ranked by uncertainty (downloadable CSV) |
| Calibration | ECE, Brier, NLL, reliability diagram, and why temperature scaling was refused |
| Research Evidence | Loss curve, scores, confusion matrix, data split, limitations |

Safety: the page listens on 127.0.0.1 only, has no file upload, sends no usage statistics, and cannot load the test set (`models/inference.py` refuses it). Live val numbers equal the saved Step 6 table (checked by `tests/test_inference.py`).

## Folders

| Folder | What is inside | In git? |
|---|---|---|
| `src/exoplanet_uq/` | Reusable project code | Yes |
| `tests/` | Automatic checks for the code | Yes |
| `scripts/` | Pipeline steps you run by hand | Yes |
| `app/` | The research dashboard (web page only) | Yes |
| `metadata/` | Small tables: catalogue, labels, splits | Yes |
| `data/` | Downloaded and processed light curves | No (too big) |
| `Docs/` | Private learning notes and documents | No (private) |
