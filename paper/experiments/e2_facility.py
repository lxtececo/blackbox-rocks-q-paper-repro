"""E2 — Capacitated facility location: referee strength and sampler variance.

Adds an exact MILP optimum (HiGHS via scipy) as a stronger referee than the
platform's greedy + local-search baseline, and runs both simulated-annealing
samplers over independent seeds, recording raw (pre-refinement) and refined
costs.

The refined annealing result is tested against the heuristic referee, which
shares its local search (one-sample t-test on the per-seed gaps; the referee is
deterministic). A budget arm repeats the 255-variable instance with 4x and 16x
the default sweeps (NumPy annealer) and 10x the default reads (D-Wave
annealer), to show whether the refined result depends on annealing effort.
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ProcessPoolExecutor

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
from common import mean_ci, pool_workers, save  # noqa: E402
from scipy.optimize import Bounds, LinearConstraint, milp  # noqa: E402
from scipy.sparse import lil_matrix  # noqa: E402
from scipy.stats import ttest_1samp  # noqa: E402

from core.optimizers.network.classical import solve_classical  # noqa: E402
from core.optimizers.network.qubo import build_cost_matrix, build_network_qubo  # noqa: E402
from core.optimizers.network.quantum_solver import solve_quantum  # noqa: E402
from core.optimizers.network.scenario import build_scenario  # noqa: E402

SCENARIOS = ["pilot_2of4", "regional_3of6", "metro_5of12", "national_8of40"]
SEEDS = list(range(10))
BUDGET_SCENARIO = "metro_5of12"
BUDGET_FACTORS = {"numpy": ("sweeps", [4, 16]), "dwave": ("reads", [10])}
DEFAULT_READS = 200  # sample_qubo default for the D-Wave annealer


def exact_cflp(problem, time_limit=600.0, capacitated=True):
    """Proven optimum by MILP. ``capacitated=False`` drops the capacity rows: the problem the QUBO encodes."""
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
        if not capacitated:
            ub[r] = np.inf
        r += 1
    A[r, :ns] = 1
    ub[r] = problem.max_open_sites if problem.max_open_sites else ns
    started = time.perf_counter()
    res = milp(obj, constraints=LinearConstraint(A.tocsr(), lb, ub), integrality=np.ones(nv),
               bounds=Bounds(0, 1), options={"time_limit": time_limit, "mip_rel_gap": 1e-9})
    x = np.round(res.x[ns:]).reshape(nd, ns)
    overflow = float(max(0.0, (d @ x - cap).max()))
    return {
        "status": int(res.status), "message": res.message, "objective": float(res.fun),
        "mip_gap": float(getattr(res, "mip_gap", 0.0) or 0.0),
        "open_sites": [problem.sites[j].id for j in range(ns) if res.x[j] > 0.5],
        "max_capacity_overflow": overflow, "respects_capacity": overflow <= 1e-6,
        "wall_s": round(time.perf_counter() - started, 2),
    }


def _sampler_cell(args):
    sid, backend, seed, options, exact_obj = args
    sol, diag = solve_quantum(build_scenario(sid), backend=backend, seed=seed, sampler_options=options or None)
    return {"raw": 100 * (diag["raw_total_cost"] - exact_obj) / exact_obj,
            "refined": 100 * (sol.total_cost - exact_obj) / exact_obj,
            "elapsed_ms": diag.get("elapsed_ms") or 0, "raw_feasible": bool(diag.get("raw_feasible"))}


def _summarise(cells, referee_gap):
    raw = [c["raw"] for c in cells]
    fin = [c["refined"] for c in cells]
    diff = np.asarray(fin) - referee_gap
    # a constant difference has no variance; the t-test is undefined there
    p_value = None if np.ptp(diff) < 1e-9 else float(ttest_1samp(diff, 0.0).pvalue)
    return {
        "raw_gap_pct": mean_ci(raw), "refined_gap_pct": mean_ci(fin),
        "raw_gap_pct_runs": raw, "refined_gap_pct_runs": fin,
        "refined_minus_referee_pct": mean_ci(diff), "refined_vs_referee_p": p_value,
        "refined_optimal_runs": int(sum(1 for g in fin if g < 1e-6)),
        "raw_feasible_runs": int(sum(c["raw_feasible"] for c in cells)), "best_raw_gap_pct": float(min(raw)),
        "worst_refined_gap_pct": float(max(fin)), "elapsed_ms": mean_ci([c["elapsed_ms"] for c in cells]),
    }


def main() -> None:
    t0 = time.time()
    out, jobs = [], []
    for sid in SCENARIOS:
        p = build_scenario(sid)
        q = build_network_qubo(p)
        exact = exact_cflp(p)
        assert exact["status"] == 0, f"{sid}: MILP did not prove optimality: {exact['message']}"
        target = exact_cflp(p, capacitated=False)  # the QUBO has no capacity term
        assert target["status"] == 0, f"{sid}: uncapacitated MILP did not prove optimality"
        ref, _ = solve_classical(p)
        ref_cost = float(ref.total_cost)
        row = {"scenario": sid, "sites": len(p.sites), "demand": len(p.demand_points),
               "qubo_vars": int(q.n_vars), "exact": exact, "referee_cost": ref_cost,
               "qubo_target": {**target, "gap_to_exact_pct": 100 * (target["objective"] - exact["objective"]) / exact["objective"]},
               "referee_overflow": float(ref.metrics.get("capacity_overflow", 0.0) or 0.0),
               "referee_gap_to_exact_pct": 100 * (ref_cost - exact["objective"]) / exact["objective"],
               "samplers": {}}
        print(f"{sid}: exact {exact['objective']:.0f} ({exact['wall_s']}s) referee {ref_cost:.0f}; "
              f"QUBO target {target['objective']:.0f} respects capacity: {target['respects_capacity']}", flush=True)
        out.append(row)
        for backend in ("dwave", "numpy"):
            if backend == "numpy" and q.n_vars > 400:
                continue
            jobs += [((sid, backend, "default"), (sid, backend, s, None, exact["objective"])) for s in SEEDS]
            if sid == BUDGET_SCENARIO:
                key, factors = BUDGET_FACTORS[backend]
                base = DEFAULT_READS if key == "reads" else int(min(max(8 * q.n_vars, 500), 4000))
                for f in factors:
                    jobs += [((sid, backend, f"{key}x{f}"), (sid, backend, s, {key: base * f}, exact["objective"]))
                             for s in SEEDS]

    with ProcessPoolExecutor(max_workers=pool_workers()) as ex:
        cells = list(ex.map(_sampler_cell, [j[1] for j in jobs]))

    budget = {}
    for row in out:
        for (sid, backend, label), cell in zip([j[0] for j in jobs], cells):
            if sid != row["scenario"]:
                continue
            target = row["samplers"] if label == "default" else budget
            target.setdefault(backend if label == "default" else f"{backend}|{label}", []).append(cell)
        for backend, group in list(row["samplers"].items()):
            row["samplers"][backend] = _summarise(group, row["referee_gap_to_exact_pct"])
            s = row["samplers"][backend]
            print(f"  {row['scenario']} {backend}: raw {s['raw_gap_pct']['mean']:.1f}% refined "
                  f"{s['refined_gap_pct']['mean']:.2f}±{s['refined_gap_pct']['ci95']:.2f}% vs referee "
                  f"{row['referee_gap_to_exact_pct']:.2f}% (p={s['refined_vs_referee_p']}) "
                  f"raw feasible {s['raw_feasible_runs']}/{len(group)}")
    referee_gap = next(r["referee_gap_to_exact_pct"] for r in out if r["scenario"] == BUDGET_SCENARIO)
    budget = {k: _summarise(v, referee_gap) for k, v in budget.items()}
    for k, s in budget.items():
        print(f"  budget {k}: raw {s['raw_gap_pct']['mean']:.1f}% refined {s['refined_gap_pct']['mean']:.2f}"
              f"±{s['refined_gap_pct']['ci95']:.2f}% raw feasible {s['raw_feasible_runs']}/{len(SEEDS)}")
    save("e2_facility", {"seeds": len(SEEDS), "rows": out, "budget_scenario": BUDGET_SCENARIO,
                         "budget": budget, "wall_s": round(time.time() - t0, 1)})


if __name__ == "__main__":
    main()
