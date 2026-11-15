"""E1 — Monte Carlo option pricing: MLAE vs classical sampling.

Measures (i) RMSE against the estimator's own target and against Black-Scholes
as a function of oracle queries, over independent seeds, with bootstrap CIs on
the log-log slope; (ii) discretisation bias vs qubit count; (iii) transpiled
circuit cost of each Grover power, which bounds what current hardware can run.
"""

from __future__ import annotations

import math
import time

import numpy as np
from common import FIGURES, bootstrap_slope, mpl, save

from core.optimizers.montecarlo.blackscholes import black_scholes_price
from core.optimizers.montecarlo.classical import classical_monte_carlo_price
from core.optimizers.montecarlo.grid import build_price_grid
from core.optimizers.montecarlo.qae import _measurement_circuits, estimate_price_via_qae, exponential_schedule

S0, K, R, SIGMA, T = 100.0, 100.0, 0.05, 0.2, 1.0
N_QUBITS = 5
SEEDS = list(range(20))
MAX_POWERS = [1, 2, 4, 8, 16, 32, 64]
SHOTS = 100
PATHS = [100, 316, 1000, 3162, 10000, 31623, 100000, 316228, 1000000]


def main() -> None:
    t0 = time.time()
    bs = black_scholes_price(S0, K, R, SIGMA, T, "call")
    grid = build_price_grid(S0, K, R, SIGMA, T, "call", N_QUBITS)
    target = float(grid.exact_discretized_price)

    bias = []
    for n in range(2, 9):
        g = build_price_grid(S0, K, R, SIGMA, T, "call", n)
        bias.append({"n_qubits": n, "discretized_price": float(g.exact_discretized_price),
                     "bias": float(g.exact_discretized_price - bs)})

    qae_rows = []
    for m in MAX_POWERS:
        prices, queries, stderr = [], None, []
        for s in SEEDS:
            r = estimate_price_via_qae(grid, shots_per_power=SHOTS, max_grover_power=m, seed=s)
            prices.append(r["price"])
            stderr.append(r["std_error"])
            queries = r["num_oracle_queries"]
        p = np.asarray(prices)
        qae_rows.append({
            "max_power": m, "queries": int(queries),
            "err_vs_target": (p - target).tolist(), "err_vs_bs": (p - bs).tolist(),
            "rmse_target": float(np.sqrt(np.mean((p - target) ** 2))),
            "rmse_bs": float(np.sqrt(np.mean((p - bs) ** 2))),
            "mean_reported_stderr": float(np.mean(stderr)),
            "empirical_sd": float(p.std(ddof=1)),
        })
        print(f"MLAE m={m:3d} queries={queries:7d} rmse_target={qae_rows[-1]['rmse_target']:.4f}")

    mc_rows = []
    for n in PATHS:
        p = np.asarray([classical_monte_carlo_price(S0, K, R, SIGMA, T, n_paths=n, seed=s)["price"] for s in SEEDS])
        mc_rows.append({"n_paths": n, "err_vs_bs": (p - bs).tolist(),
                        "rmse_bs": float(np.sqrt(np.mean((p - bs) ** 2)))})
        print(f"MC n={n:8d} rmse={mc_rows[-1]['rmse_bs']:.4f}")

    qae_slope = bootstrap_slope([r["queries"] for r in qae_rows], [r["err_vs_target"] for r in qae_rows])
    mc_slope = bootstrap_slope([r["n_paths"] for r in mc_rows], [r["err_vs_bs"] for r in mc_rows])
    # m=1 sits in the pre-asymptotic regime of the likelihood fit and steepens the slope.
    qae_slope_m2 = bootstrap_slope([r["queries"] for r in qae_rows[1:]], [r["err_vs_target"] for r in qae_rows[1:]])

    # Like-for-like baseline: classical sampling of the same 2^n-bin distribution MLAE encodes.
    disc_rows = []
    payoffs = np.asarray(grid.discounted_payoffs, float)
    probs = np.asarray(grid.probabilities, float)
    probs = probs / probs.sum()
    for n in PATHS:
        p = []
        for s in SEEDS:
            rng = np.random.default_rng(10_000 + s)
            p.append(float(payoffs[rng.choice(len(probs), size=n, p=probs)].mean()))
        p = np.asarray(p)
        disc_rows.append({"n_samples": n, "err_vs_target": (p - target).tolist(),
                          "rmse_target": float(np.sqrt(np.mean((p - target) ** 2)))})
    disc_slope = bootstrap_slope([r["n_samples"] for r in disc_rows], [r["err_vs_target"] for r in disc_rows])

    # Queries needed to reach a target RMSE, interpolated on the fitted lines.
    def fit(xs, ys):
        b, a = np.polyfit(np.log10(xs), np.log10(ys), 1)
        return a, b

    a_q, b_q = fit([r["queries"] for r in qae_rows], [r["rmse_target"] for r in qae_rows])
    a_c, b_c = fit([r["n_paths"] for r in mc_rows], [r["rmse_bs"] for r in mc_rows])
    a_d, b_d = fit([r["n_samples"] for r in disc_rows], [r["rmse_target"] for r in disc_rows])
    crossover = {}
    for eps in (0.1, 0.03, 0.01):
        crossover[str(eps)] = {
            "mlae_queries": float(10 ** ((math.log10(eps) - a_q) / b_q)),
            "classical_samples": float(10 ** ((math.log10(eps) - a_c) / b_c)),
            "classical_discretized_samples": float(10 ** ((math.log10(eps) - a_d) / b_d)),
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
        print(f"power {k:2d}: depth {tc.depth()} cx {ops.get('cx', 0)}")

    save("e1_montecarlo", {
        "instance": {"S0": S0, "K": K, "r": R, "sigma": SIGMA, "T": T, "n_qubits": N_QUBITS,
                     "black_scholes": bs, "discretized_target": target},
        "seeds": len(SEEDS), "shots_per_power": SHOTS,
        "bias": bias, "mlae": qae_rows, "classical": mc_rows, "classical_discretized": disc_rows,
        "slopes": {"mlae_vs_target": qae_slope, "mlae_vs_target_excluding_m1": qae_slope_m2,
                   "classical_vs_bs": mc_slope, "classical_discretized_vs_target": disc_slope},
        "fit": {"mlae": [a_q, b_q], "classical": [a_c, b_c], "classical_discretized": [a_d, b_d]},
        "queries_to_reach_rmse": crossover, "resources": resources,
        "wall_s": round(time.time() - t0, 1),
    })

    plt = mpl()
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.5))
    q = [r["queries"] for r in qae_rows]
    ax[0].loglog(q, [r["rmse_target"] for r in qae_rows], "o-", label=f"MLAE vs grid target (slope {qae_slope[0]:.2f})")
    ax[0].loglog(q, [r["rmse_bs"] for r in qae_rows], "s--", label="MLAE vs Black–Scholes")
    ax[0].loglog(PATHS, [r["rmse_bs"] for r in mc_rows], "^-", label=f"Classical vs Black–Scholes (slope {mc_slope[0]:.2f})")
    ax[0].loglog(PATHS, [r["rmse_target"] for r in disc_rows], "v:", label=f"Classical on same grid vs grid target (slope {disc_slope[0]:.2f})")
    ax[0].axhline(abs(target - bs), color="grey", lw=0.8, ls=":", label="|discretisation bias|, n=5")
    ax[0].set_xlabel("oracle queries / samples")
    ax[0].set_ylabel("RMSE over 20 seeds")
    ax[0].legend(loc="upper right")
    ax[1].bar([str(r["grover_power"]) for r in resources], [r["cx"] for r in resources], color="tab:orange")
    ax[1].set_yscale("log")
    ax[1].set_xlabel("Grover power k")
    ax[1].set_ylabel("CX count after transpilation")
    fig.tight_layout()
    fig.savefig(FIGURES / "mlae_convergence.pdf")
    print("figure written")


if __name__ == "__main__":
    main()
