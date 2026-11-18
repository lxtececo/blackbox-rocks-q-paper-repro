"""E1 - Monte Carlo option pricing: MLAE vs classical sampling (v2 design).

Design (revised after the 2026-11-16 audit):
  * every (method, point, replicate) cell gets its own SeedSequence seed, so
    points are statistically independent and per-point bootstrap is valid;
  * the MLAE estimator is checked against its Cramer-Rao bound at the true
    amplitude (efficiency ratio RMSE/CRLB) and its reported Fisher standard
    error is checked for coverage;
  * query-cost ratios are read off log-log interpolation strictly inside the
    measured range of both methods - no extrapolation.
Simulator: noiseless statevector, independent shot noise per circuit.
"""

from __future__ import annotations

import math
import os
import time
from concurrent.futures import ProcessPoolExecutor

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
from common import FIGURES, bootstrap_slope, mpl, point_seed, pool_workers, rmse_ci, save, wilson  # noqa: E402

from core.optimizers.montecarlo.blackscholes import black_scholes_price  # noqa: E402
from core.optimizers.montecarlo.classical import classical_monte_carlo_price  # noqa: E402
from core.optimizers.montecarlo.grid import build_price_grid  # noqa: E402
from core.optimizers.montecarlo.qae import (  # noqa: E402
    _fisher_standard_error,
    _measurement_circuits,
    estimate_price_via_qae,
    exponential_schedule,
)

S0, K, R, SIGMA, T = 100.0, 100.0, 0.05, 0.2, 1.0
N_QUBITS = 5
REPS = 50
MAX_POWERS = [1, 2, 4, 8, 16, 32, 64, 128, 256]
SHOTS = 100
PATHS = [100, 316, 1000, 3162, 10000, 31623, 100000, 316228, 1000000, 3162278, 10000000]
EPS = [0.1, 0.03, 0.01]


def _grid():
    return build_price_grid(S0, K, R, SIGMA, T, "call", N_QUBITS)


def _mlae_cell(args):
    m, rep = args
    r = estimate_price_via_qae(_grid(), shots_per_power=SHOTS, max_grover_power=m,
                               seed=point_seed(1, m, rep))
    return m, rep, r["price"], r["std_error"], r["num_oracle_queries"]


def _mc_cell(args):
    n, rep = args
    return n, rep, classical_monte_carlo_price(S0, K, R, SIGMA, T, n_paths=n, seed=point_seed(2, n, rep))["price"]


def _disc_cell(args):
    n, rep = args
    g = _grid()
    payoffs = np.asarray(g.discounted_payoffs, float)
    probs = np.asarray(g.probabilities, float)
    probs = probs / probs.sum()
    rng = np.random.default_rng(point_seed(3, n, rep))
    # multinomial counts == n i.i.d. draws, without materialising them
    counts = rng.multinomial(n, probs)
    return n, rep, float(counts @ payoffs / n)


def _interp_cost(xs, rmses, eps):
    """log-log linear interpolation of cost at RMSE eps; None outside the measured range."""
    lx, ly = np.log10(np.asarray(xs, float)), np.log10(np.asarray(rmses, float))
    le = math.log10(eps)
    for i in range(len(lx) - 1):
        a, b = ly[i], ly[i + 1]
        if min(a, b) <= le <= max(a, b) and a != b:
            return float(10 ** (lx[i] + (le - a) * (lx[i + 1] - lx[i]) / (b - a)))
    return None


def main() -> None:
    t0 = time.time()
    bs = black_scholes_price(S0, K, R, SIGMA, T, "call")
    grid = _grid()
    target = float(grid.exact_discretized_price)
    ceiling = float(grid.payoff_ceiling)
    a_true = target / ceiling
    theta_true = math.asin(math.sqrt(a_true))

    bias = []
    for n in range(2, 9):
        g = build_price_grid(S0, K, R, SIGMA, T, "call", n)
        bias.append({"n_qubits": n, "discretized_price": float(g.exact_discretized_price),
                     "bias": float(g.exact_discretized_price - bs)})

    workers = pool_workers()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        mlae_raw = list(ex.map(_mlae_cell, [(m, r) for m in MAX_POWERS for r in range(REPS)]))
        mc_raw = list(ex.map(_mc_cell, [(n, r) for n in PATHS for r in range(REPS)]))
        disc_raw = list(ex.map(_disc_cell, [(n, r) for n in PATHS for r in range(REPS)]))

    qae_rows = []
    for m in MAX_POWERS:
        cells = [c for c in mlae_raw if c[0] == m]
        p = np.asarray([c[2] for c in cells])
        se = np.asarray([c[3] for c in cells])
        err = p - target
        crlb = _fisher_standard_error(theta_true, exponential_schedule(m), SHOTS) * ceiling
        rc = rmse_ci(err, seed=m)
        covered = int(np.sum(np.abs(err) <= 1.96 * se))
        lo, hi = wilson(covered, len(err))
        qae_rows.append({
            "max_power": m, "queries": int(cells[0][4]), "err_vs_target": err.tolist(),
            "rmse_target": rc["rmse"], "rmse_ci": [rc["lo"], rc["hi"]],
            "rmse_bs": float(np.sqrt(np.mean((p - bs) ** 2))),
            "bias_mean": float(err.mean()), "empirical_sd": float(p.std(ddof=1)),
            "crlb": float(crlb), "efficiency_ratio": rc["rmse"] / crlb,
            "efficiency_ratio_ci": [rc["lo"] / crlb, rc["hi"] / crlb],
            "mean_reported_stderr": float(se.mean()),
            "coverage_95": covered / len(err), "coverage_wilson": [lo, hi],
        })
        print(f"MLAE m={m:3d} q={cells[0][4]:7d} rmse={rc['rmse']:.5f} crlb={crlb:.5f} "
              f"ratio={rc['rmse'] / crlb:.2f} cover={covered}/{len(err)}")

    payoffs = np.asarray(grid.discounted_payoffs, float)
    probs = np.asarray(grid.probabilities, float)
    probs = probs / probs.sum()
    disc_sd = float(math.sqrt(probs @ payoffs ** 2 - (probs @ payoffs) ** 2))

    def _rows(raw, ref, key, theory_sd=None):
        rows = []
        for n in PATHS:
            err = np.asarray([c[2] for c in raw if c[0] == n]) - ref
            rc = rmse_ci(err, seed=n % 100_003)
            row = {key: n, "err": err.tolist(), "rmse": rc["rmse"], "rmse_ci": [rc["lo"], rc["hi"]],
                   "bias_mean": float(err.mean())}
            if theory_sd is not None:
                row["theory_se"] = theory_sd / math.sqrt(n)
            rows.append(row)
        return rows

    mc_rows = _rows(mc_raw, bs, "n_paths")
    disc_rows = _rows(disc_raw, target, "n_samples", disc_sd)
    for r in mc_rows:
        print(f"MC   n={r['n_paths']:9d} rmse={r['rmse']:.5f}")

    slopes = {
        "mlae_all": bootstrap_slope([r["queries"] for r in qae_rows], [r["err_vs_target"] for r in qae_rows]),
        "mlae_m_ge_4": bootstrap_slope([r["queries"] for r in qae_rows[2:]], [r["err_vs_target"] for r in qae_rows[2:]]),
        "classical_vs_bs": bootstrap_slope([r["n_paths"] for r in mc_rows], [r["err"] for r in mc_rows]),
        "classical_discretized": bootstrap_slope([r["n_samples"] for r in disc_rows], [r["err"] for r in disc_rows]),
    }
    # slope of the bound itself over the same schedules: what an efficient estimator should show
    log_q, log_b = np.log([r["queries"] for r in qae_rows]), np.log([r["crlb"] for r in qae_rows])
    slopes["cramer_rao_bound_all"] = float(np.polyfit(log_q, log_b, 1)[0])
    slopes["cramer_rao_bound_m_ge_4"] = float(np.polyfit(log_q[2:], log_b[2:], 1)[0])
    covered = sum(round(r["coverage_95"] * REPS) for r in qae_rows)
    coverage_pooled = {"covered": int(covered), "runs": REPS * len(qae_rows), "wilson95": list(wilson(int(covered), REPS * len(qae_rows)))}

    q = [r["queries"] for r in qae_rows]
    cost = {}
    for eps in EPS:
        mq = _interp_cost(q, [r["rmse_target"] for r in qae_rows], eps)
        cd = _interp_cost(PATHS, [r["rmse"] for r in disc_rows], eps)
        cm = _interp_cost(PATHS, [r["rmse"] for r in mc_rows], eps)
        cost[str(eps)] = {
            "mlae_queries": mq, "classical_discretized_samples": cd, "classical_antithetic_paths": cm,
            "ratio_discretized_over_mlae": (cd / mq) if (cd and mq) else None,
            "ratio_antithetic_over_mlae": (cm / mq) if (cm and mq) else None,
        }

    from qiskit import transpile

    powers = exponential_schedule(64)
    circs = _measurement_circuits(grid, powers)
    resources = []
    for k, c in zip(powers, circs):
        tc = transpile(c, basis_gates=["cx", "rz", "sx", "x"], optimization_level=1, seed_transpiler=7)
        ops = tc.count_ops()
        resources.append({"grover_power": k, "depth": tc.depth(), "cx": int(ops.get("cx", 0)),
                          "total_gates": int(sum(ops.values()))})

    save("e1_montecarlo", {
        "design": "v2: independent SeedSequence seed per (method, point, replicate); "
                  "statevector sampler with independent per-circuit streams",
        "instance": {"S0": S0, "K": K, "r": R, "sigma": SIGMA, "T": T, "n_qubits": N_QUBITS,
                     "black_scholes": bs, "discretized_target": target, "payoff_ceiling": ceiling,
                     "true_amplitude": a_true, "discretized_payoff_sd": disc_sd},
        "replicates": REPS, "shots_per_power": SHOTS,
        "bias": bias, "mlae": qae_rows, "classical": mc_rows, "classical_discretized": disc_rows,
        "slopes": slopes, "coverage_pooled": coverage_pooled, "cost_at_rmse_interpolated": cost, "resources": resources,
        "wall_s": round(time.time() - t0, 1),
    })

    plt = mpl()
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.6))
    a0 = ax[0]

    def band(x, rows, key_lo_hi="rmse_ci"):
        return [r[key_lo_hi][0] for r in rows], [r[key_lo_hi][1] for r in rows]

    lo, hi = band(q, qae_rows)
    a0.fill_between(q, lo, hi, alpha=0.2, color="tab:blue")
    a0.loglog(q, [r["rmse_target"] for r in qae_rows], "o-", color="tab:blue", ms=3, label="MLAE vs grid target")
    a0.loglog(q, [r["crlb"] for r in qae_rows], "k--", lw=0.8, label="MLAE Cramér–Rao bound")
    lo, hi = band(PATHS, disc_rows)
    a0.fill_between(PATHS, lo, hi, alpha=0.2, color="tab:green")
    a0.loglog(PATHS, [r["rmse"] for r in disc_rows], "v-", color="tab:green", ms=3, label="Classical, same 32-bin grid")
    a0.loglog(PATHS, [r["rmse"] for r in mc_rows], "^:", color="tab:red", ms=3, label="Classical antithetic vs Black–Scholes")
    a0.axhline(abs(target - bs), color="grey", lw=0.8, ls=":", label="|discretisation bias|, n=5")
    a0.set_xlabel("oracle queries / samples")
    a0.set_ylabel(f"RMSE ({REPS} independent replicates)")
    a0.legend(loc="lower left", fontsize=6)
    a1 = ax[1]
    a1.errorbar([str(r["max_power"]) for r in qae_rows], [r["efficiency_ratio"] for r in qae_rows],
                yerr=[[r["efficiency_ratio"] - r["efficiency_ratio_ci"][0] for r in qae_rows],
                      [r["efficiency_ratio_ci"][1] - r["efficiency_ratio"] for r in qae_rows]],
                fmt="o", ms=3, capsize=2)
    a1.axhline(1.0, color="k", lw=0.8, ls="--")
    a1.set_xlabel("largest Grover power $k_{\\max}$")
    a1.set_ylabel("RMSE / CRLB")
    fig.tight_layout()
    fig.savefig(FIGURES / "mlae_convergence.pdf")
    print("figure written", round(time.time() - t0, 1), "s")


if __name__ == "__main__":
    main()
