"""Shared helpers for the paper experiments.

Run from the repo root, e.g.
    OTEL_SDK_DISABLED=true PYTHONPATH=backend/src/main/python \
      .venv/bin/python reports/paper/quantumfpo-arxiv/experiments/e1_montecarlo.py
"""

from __future__ import annotations

import json
import math
import platform
import random
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIGURES = ROOT.parent / "figures"
RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)


def provenance() -> Dict[str, Any]:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001
        commit = "unknown"
    return {"commit": commit, "python": platform.python_version(), "machine": platform.machine()}


def save(name: str, payload: Dict[str, Any]) -> Path:
    path = RESULTS / f"{name}.json"
    payload = {"provenance": provenance(), **payload}
    path.write_text(json.dumps(payload, indent=2, default=float))
    print(f"wrote {path}")
    return path


def loglog_slope(x: Sequence[float], y: Sequence[float]) -> float:
    lx, ly = np.log10(np.asarray(x, float)), np.log10(np.asarray(y, float))
    return float(np.polyfit(lx, ly, 1)[0])


def bootstrap_slope(
    x: Sequence[float], errors_by_point: Sequence[Sequence[float]], n_boot: int = 2000, seed: int = 0
) -> Tuple[float, float, float]:
    """RMSE log-log slope with a percentile CI from resampling seeds per point."""
    rng = np.random.default_rng(seed)
    errs = [np.asarray(e, float) for e in errors_by_point]
    point = loglog_slope(x, [math.sqrt(np.mean(e ** 2)) for e in errs])
    boots = []
    for _ in range(n_boot):
        rmse = [math.sqrt(np.mean(rng.choice(e, size=len(e)) ** 2)) for e in errs]
        boots.append(loglog_slope(x, rmse))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return point, float(lo), float(hi)


def wilson(successes: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return centre - half, centre + half


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def mpl():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "figure.dpi": 150,
        "pdf.fonttype": 42, "axes.grid": True, "grid.alpha": 0.3,
    })
    return plt


def mean_ci(values: Sequence[float]) -> Dict[str, float]:
    v = np.asarray(values, float)
    if len(v) < 2:
        return {"mean": float(v.mean()) if len(v) else float("nan"), "ci95": 0.0, "n": int(len(v))}
    return {"mean": float(v.mean()), "ci95": float(1.96 * v.std(ddof=1) / math.sqrt(len(v))), "n": int(len(v))}


def as_list(x: Any) -> List[Any]:
    return list(x) if isinstance(x, (list, tuple)) else [x]
