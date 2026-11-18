"""E5 — Fleet routing: assignment-QUBO samplers vs the classical referee.

Ten independently generated instances per preset, five sampler seeds on each.
Runs on one instance share that instance and are not independent, so the unit
of analysis is the instance: the headline is the mean over instances of the
per-instance mean gap, with a Student-t interval over the ten instances. The
sampler-seed spread within an instance is reported separately.
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ProcessPoolExecutor

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
from common import mean_ci, pool_workers, save  # noqa: E402

from core.optimizers.fleet.service import solve_scenario  # noqa: E402

PRESETS = ["regional_food_redistributor_demo", "metro_grocery_last_mile"]
INSTANCE_SEEDS = [42, 7, 11, 101, 102, 103, 104, 105, 106, 107]
SOLVER_SEEDS = [0, 1, 2, 3, 4]
METHODS = ["qubo-annealing", "simulated-annealing"]


def _cell(args):
    preset, inst, method, s = args
    started = time.time()
    r = solve_scenario(scenario=preset, method=method, seed=inst, config={"seed": s})
    cmp = r.get("comparison") or {}
    return {
        "preset": preset, "instance_seed": inst, "method": method, "solver_seed": s,
        "distance_delta_pct": cmp.get("distance_delta_pct"),
        "cost_delta_pct": cmp.get("cost_delta_pct"),
        "unassigned_delta": cmp.get("unassigned_delta"),
        "wins": cmp.get("wins"), "wall_s": round(time.time() - started, 2),
    }


def main() -> None:
    t0 = time.time()
    jobs = [(p, i, m, s) for p in PRESETS for i in INSTANCE_SEEDS for m in METHODS for s in SOLVER_SEEDS]
    with ProcessPoolExecutor(max_workers=pool_workers()) as ex:
        rows = list(ex.map(_cell, jobs))
    summary = {}
    for preset in PRESETS:
        for method in METHODS:
            sel = [r for r in rows if r["preset"] == preset and r["method"] == method]
            assert all(r["distance_delta_pct"] is not None for r in sel), "a run returned no comparison"
            per_instance = [[r["distance_delta_pct"] for r in sel if r["instance_seed"] == i] for i in INSTANCE_SEEDS]
            instance_means = [float(np.mean(v)) for v in per_instance]
            summary[f"{preset}|{method}"] = {
                "distance_delta_pct": mean_ci(instance_means),  # interval over instances
                "instance_means": instance_means,
                "within_instance_sd": float(np.sqrt(np.mean([np.var(v, ddof=1) for v in per_instance]))),
                "best": min(r["distance_delta_pct"] for r in sel),
                "worst": max(r["distance_delta_pct"] for r in sel),
                "wins": sum(bool(r["wins"]) for r in sel), "runs": len(sel), "instances": len(INSTANCE_SEEDS),
                "instances_with_a_win": sum(any(bool(r["wins"]) for r in sel if r["instance_seed"] == i) for i in INSTANCE_SEEDS),
                "any_unassigned": sum(1 for r in sel if (r["unassigned_delta"] or 0) > 0),
            }
            s = summary[f"{preset}|{method}"]
            print(f"{preset[:22]:22s} {method:20s} gap {s['distance_delta_pct']['mean']:+.1f}±{s['distance_delta_pct']['ci95']:.1f}% "
                  f"range {s['best']:+.1f}..{s['worst']:+.1f} wins {s['wins']}/{s['runs']} dropping stops {s['any_unassigned']}")
    save("e5_fleet", {"instance_seeds": INSTANCE_SEEDS, "solver_seeds": SOLVER_SEEDS, "rows": rows,
                      "summary": summary, "wall_s": round(time.time() - t0, 1)})


if __name__ == "__main__":
    main()
