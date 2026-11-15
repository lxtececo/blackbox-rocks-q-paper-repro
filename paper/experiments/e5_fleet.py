"""E5 — Fleet routing: assignment-QUBO samplers vs the classical referee.

Varies both the instance (scenario seed) and the sampler seed, and reports the
distance gap to the classical baseline on the same instance.
"""

from __future__ import annotations

import time

from common import mean_ci, save

from core.optimizers.fleet.service import solve_scenario

PRESETS = ["regional_food_redistributor_demo", "metro_grocery_last_mile"]
INSTANCE_SEEDS = [42, 7, 11]
SOLVER_SEEDS = [0, 1, 2, 3, 4]
METHODS = ["qubo-annealing", "simulated-annealing"]


def main() -> None:
    t0 = time.time()
    rows = []
    for preset in PRESETS:
        for inst in INSTANCE_SEEDS:
            for method in METHODS:
                for s in SOLVER_SEEDS:
                    started = time.time()
                    r = solve_scenario(scenario=preset, method=method, seed=inst, config={"seed": s})
                    cmp = r.get("comparison") or {}
                    rows.append({
                        "preset": preset, "instance_seed": inst, "method": method, "solver_seed": s,
                        "distance_delta_pct": cmp.get("distance_delta_pct"),
                        "cost_delta_pct": cmp.get("cost_delta_pct"),
                        "unassigned_delta": cmp.get("unassigned_delta"),
                        "wins": cmp.get("wins"), "wall_s": round(time.time() - started, 2),
                    })
                    print(f"{preset[:12]} inst={inst} {method:20s} s={s} d={cmp.get('distance_delta_pct')}% "
                          f"unassigned+{cmp.get('unassigned_delta')} {rows[-1]['wall_s']}s")
    summary = {}
    for preset in PRESETS:
        for method in METHODS:
            sel = [r for r in rows if r["preset"] == preset and r["method"] == method and r["distance_delta_pct"] is not None]
            summary[f"{preset}|{method}"] = {
                "distance_delta_pct": mean_ci([r["distance_delta_pct"] for r in sel]),
                "best": min(r["distance_delta_pct"] for r in sel),
                "worst": max(r["distance_delta_pct"] for r in sel),
                "wins": sum(bool(r["wins"]) for r in sel), "runs": len(sel),
                "any_unassigned": sum(1 for r in sel if (r["unassigned_delta"] or 0) > 0),
            }
    save("e5_fleet", {"rows": rows, "summary": summary, "wall_s": round(time.time() - t0, 1)})


if __name__ == "__main__":
    main()
