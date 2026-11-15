"""E6 — QAOA on the 18-variable facility-location QUBO, with a random control.

The platform's QAOA sampler returns the lowest-energy bitstring among its final
shots. Best-of-N selection alone can find good states, so every QAOA run is
paired with uniform random sampling at the same shot budget. Quality is
reported as the energy rank of the returned state among all 2^n states.
"""

from __future__ import annotations

import time

import numpy as np
from common import mean_ci, save

from core.optimizers.network.qubo import build_network_qubo
from core.optimizers.network.quantum_solver import solve_quantum
from core.optimizers.network.scenario import build_scenario

REPS = [1, 2, 3]
SEEDS = [0, 1, 2, 3, 4]
SHOTS = 1024


def all_energies(qubo) -> np.ndarray:
    n = qubo.n_vars
    states = ((np.arange(2 ** n)[:, None] >> np.arange(n)) & 1).astype(np.int8)
    q, Q = np.asarray(qubo.q, float), np.asarray(qubo.Q, float)
    e = states @ q + np.einsum("bi,ij,bj->b", states, Q, states, optimize=True) + float(qubo.const)
    for _ in range(5):  # cross-check against the platform's own energy
        k = np.random.randint(2 ** n)
        assert abs(e[k] - qubo.energy(states[k])) < 1e-6 * max(1.0, abs(e[k]))
    return e, states


def rank_pct(sorted_e: np.ndarray, value: float) -> float:
    return 100.0 * np.searchsorted(sorted_e, value + 1e-9, side="right") / len(sorted_e)


def main() -> None:
    t0 = time.time()
    problem = build_scenario("pilot_2of4")
    qubo = build_network_qubo(problem)
    energies, states = all_energies(qubo)
    order = np.sort(energies)
    ground = float(order[0])
    uniform_mean = float(energies.mean())
    feasible_energy_cut = None
    print(f"n={qubo.n_vars} ground={ground:.1f}")

    rows = []
    for p in REPS:
        for s in SEEDS:
            started = time.time()
            sol, diag = solve_quantum(problem, backend="gate", seed=s, sampler_options={"reps": p, "shots": SHOTS})
            rng = np.random.default_rng(1000 + s)
            rand_best = float(energies[rng.integers(0, 2 ** qubo.n_vars, SHOTS)].min())
            trace = diag.get("energy_trace") or []
            best_expect = float(min(trace)) if trace else float("nan")
            rows.append({
                "reps": p, "seed": s, "qaoa_energy": float(diag["energy"]),
                "qaoa_expectation": best_expect,
                "qaoa_expectation_ratio": (uniform_mean - best_expect) / (uniform_mean - ground),
                "qaoa_rank_pct": rank_pct(order, float(diag["energy"])),
                "qaoa_is_ground": abs(float(diag["energy"]) - ground) < 1e-6,
                "random_energy": rand_best, "random_rank_pct": rank_pct(order, rand_best),
                "random_is_ground": abs(rand_best - ground) < 1e-6,
                "raw_feasible": bool(diag.get("raw_feasible")),
                "refined_cost": float(sol.total_cost), "iterations": len(diag.get("energy_trace") or []),
                "wall_s": round(time.time() - started, 1),
            })
            print(f"p={p} s={s} qaoa rank {rows[-1]['qaoa_rank_pct']:.4f}% ground={rows[-1]['qaoa_is_ground']} | "
                  f"random rank {rows[-1]['random_rank_pct']:.4f}% | {rows[-1]['wall_s']}s")

    summary = {}
    for p in REPS:
        sel = [r for r in rows if r["reps"] == p]
        summary[str(p)] = {
            "qaoa_rank_pct": mean_ci([r["qaoa_rank_pct"] for r in sel]),
            "qaoa_expectation_ratio": mean_ci([r["qaoa_expectation_ratio"] for r in sel]),
            "random_rank_pct": mean_ci([r["random_rank_pct"] for r in sel]),
            "qaoa_ground_runs": sum(r["qaoa_is_ground"] for r in sel),
            "random_ground_runs": sum(r["random_is_ground"] for r in sel),
            "qaoa_beats_random": sum(r["qaoa_energy"] < r["random_energy"] - 1e-9 for r in sel),
            "ties": sum(abs(r["qaoa_energy"] - r["random_energy"]) < 1e-9 for r in sel),
            "runs": len(sel),
        }
    save("e6_qaoa_control", {
        "scenario": "pilot_2of4", "n_vars": int(qubo.n_vars), "shots": SHOTS, "ground_energy": ground,
        "uniform_mean_energy": uniform_mean,
        "states": int(2 ** qubo.n_vars), "rows": rows, "summary": summary,
        "feasible_energy_cut": feasible_energy_cut, "wall_s": round(time.time() - t0, 1),
    })


if __name__ == "__main__":
    main()
