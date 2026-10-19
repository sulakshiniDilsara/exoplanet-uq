"""Project-wide paths.

All paths are built from the project root, so the code works no matter
which folder you run it from.
"""

from pathlib import Path

# src/exoplanet_uq/config.py -> go up three levels to reach the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

METADATA_DIR = PROJECT_ROOT / "metadata"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Kepler light curve files, one folder per star (see data/lightcurves.py).
KEPLER_RAW_DIR = RAW_DATA_DIR / "kepler"
