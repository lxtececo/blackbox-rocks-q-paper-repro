"""E2 — Capacitated facility location: referee strength and sampler variance.

Adds an exact MILP optimum (HiGHS via scipy) as a stronger referee than the
platform's greedy + local-search baseline, and runs both simulated-annealing
samplers over independent seeds, recording raw (pre-refinement) and refined
costs.
"""

from __future__ import annotations

import time

import numpy as np
from common import mean_ci, save
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

from core.optimizers.network.classical import solve_classical
from core.optimizers.network.qubo import build_cost_matrix, build_network_qubo
from core.optimizers.network.quantum_solver import solve_quantum
from core.optimizers.network.scenario import build_scenario

SCENARIOS = ["pilot_2of4", "regional_3of6", "metro_5of12", "national_8of40"]
SEEDS = list(range(10))


def exact_cflp(problem, time_limit=600.0):
    c = build_cost_matrix(problem)  # (demand, sites)
    nd, ns = c.shape
    f = np.array([s.fixed_cost for s in problem.sites])
    cap = np.array([s.capacity for s in problem.sites])
    d = np.array([p.demand for p in problem.demand_points])
    nv = ns + nd * ns
    xi = lambda i, j: ns + i * ns + j  # noqa: E731
    obj = np.concatenate([f, c.reshape(-1)])
    rows = nd + nd * ns + ns + 1
    A = lil_matrix((rows, nv))
    lb, ub = np.full(rows, -np.inf), np.zeros(rows)
    r = 0
    for i in range(nd):  # each demand served once
        for j in range(ns):
            A[r, xi(i, j)] = 1
        lb[r] = ub[r] = 1
        r += 1
    for i in range(nd):  # linking
        for j in range(ns):
            A[r, xi(i, j)] = 1
            A[r, j] = -1
            r += 1
    for j in range(ns):  # capacity
        for i in range(nd):
            A[r, xi(i, j)] = d[i]
        A[r, j] = -cap[j]
        r += 1
    A[r, :ns] = 1
    ub[r] = problem.max_open_sites if problem.max_open_sites else ns
    started = time.perf_counter()
    res = milp(obj, constraints=LinearConstraint(A.tocsr(), lb, ub), integrality=np.ones(nv),
               bounds=Bounds(0, 1), options={"time_limit": time_limit, "mip_rel_gap": 1e-9})
    return {
        "status": int(res.status), "message": res.message, "objective": float(res.fun),
        "mip_gap": float(getattr(res, "mip_gap", 0.0) or 0.0),
        "open_sites": [problem.sites[j].id for j in range(ns) if res.x[j] > 0.5],
        "wall_s": round(time.perf_counter() - started, 2),
    }


def main() -> None:
    out = []
    for sid in SCENARIOS:
        p = build_scenario(sid)
        q = build_network_qubo(p)
        exact = exact_cflp(p)
        ref, _ = solve_classical(p)
        ref_cost = float(ref.total_cost)
        row = {"scenario": sid, "sites": len(p.sites), "demand": len(p.demand_points),
               "qubo_vars": int(q.n_vars), "exact": exact, "referee_cost": ref_cost,
               "referee_overflow": float(ref.metrics.get("capacity_overflow", 0.0) or 0.0),
               "referee_gap_to_exact_pct": 100 * (ref_cost - exact["objective"]) / exact["objective"],
               "samplers": {}}
        print(f"{sid}: exact {exact['objective']:.0f} ({exact['wall_s']}s) referee {ref_cost:.0f}")
        for backend in ("dwave", "numpy"):
            if backend == "numpy" and q.n_vars > 400:
                continue
            raw, fin, ms, raw_feasible = [], [], [], 0
            for s in SEEDS:
                sol, diag = solve_quantum(p, backend=backend, seed=s)
                raw.append(100 * (diag["raw_total_cost"] - exact["objective"]) / exact["objective"])
                fin.append(100 * (sol.total_cost - exact["objective"]) / exact["objective"])
                ms.append(diag.get("elapsed_ms") or 0)
                raw_feasible += int(bool(diag.get("raw_feasible")))
            row["samplers"][backend] = {
                "raw_gap_pct": mean_ci(raw), "refined_gap_pct": mean_ci(fin),
                "refined_optimal_runs": int(sum(1 for g in fin if g < 1e-6)),
                "raw_feasible_runs": raw_feasible, "best_raw_gap_pct": float(min(raw)),
                "worst_refined_gap_pct": float(max(fin)), "elapsed_ms": mean_ci(ms),
            }
            print(f"  {backend}: raw {np.mean(raw):.1f}% refined {np.mean(fin):.2f}% optimal {row['samplers'][backend]['refined_optimal_runs']}/10")
        out.append(row)
    save("e2_facility", {"seeds": len(SEEDS), "rows": out})


if __name__ == "__main__":
    main()
