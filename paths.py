"""Repo root by marker, so nesting depth never matters. Top-level dirs only."""
import pathlib

ROOT = next(p for p in pathlib.Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
DATA = ROOT / "data"    # generated datasets and source downloads
RUNS = ROOT / "runs"    # checkpoints and final artifacts
OUT = ROOT / "out"      # sweep results, logs, figures
