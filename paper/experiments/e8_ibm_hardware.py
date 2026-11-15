"""E8 — The same experiments on IBM quantum hardware.

Runs three workloads through the platform's real-hardware path (least-busy IBM
device, dynamical decoupling and gate twirling as configured by the platform):

  (a) MLAE at shallow Grover schedules, where the noiseless analysis predicts
      the circuits might still run;
  (b) SqDRIFT + SQD on N2 at coverage << 1, the regime where E4/E7 showed the
      sampler matters;
  (c) QAOA p=1 on the 18-qubit facility-location QUBO with the random control.

Usage:  e8_ibm_hardware.py [--dry-run]   (dry run = same code on the simulator)
QPU budget is kept small on purpose; every job id is recorded.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
from common import save

DRY = "--dry-run" in sys.argv
MODE = "simulator" if DRY else "real"
PLATFORM = None if DRY else "ibm"


def load_token() -> None:
    env = Path(__file__).resolve().parents[4] / ".env"
    if "IBM_QUANTUM_TOKEN" in os.environ or not env.exists():
        return
    for line in env.read_text().splitlines():
        if line.startswith("IBM_QUANTUM_TOKEN="):
            os.environ["IBM_QUANTUM_TOKEN"] = line.split("=", 1)[1].strip().strip('"').strip("'")


def mlae() -> list:
    from core.optimizers.montecarlo.blackscholes import black_scholes_price
    from core.optimizers.montecarlo.grid import build_price_grid
    from core.optimizers.montecarlo.qae import estimate_price_via_qae

    grid = build_price_grid(100.0, 100.0, 0.05, 0.2, 1.0, "call", 5)
    target = float(grid.exact_discretized_price)
    bs = black_scholes_price(100.0, 100.0, 0.05, 0.2, 1.0, "call")
    rows = []
    for m in (1, 2, 4, 8):
        started = time.time()
        r = estimate_price_via_qae(grid, shots_per_power=200, max_grover_power=m, seed=11,
                                   platform=PLATFORM, execution_mode=MODE)
        rows.append({
            "max_power": m, "queries": r["num_oracle_queries"], "price": r["price"],
            "error_vs_target": r["price"] - target, "error_vs_bs": r["price"] - bs,
            "hits_per_power": r.get("hits_per_power"), "grover_schedule": r.get("grover_schedule"),
            "backend": r.get("backend_name"), "physical_qubits": r.get("physical_qubits"),
            "vendor_runs": r.get("vendor_runs"), "wall_s": round(time.time() - started, 1),
        })
        print(f"MLAE m={m}: price {r['price']:.3f} (target {target:.3f}) err {r['price'] - target:+.3f} on {r.get('backend_name')}")
    return rows


def sqdrift() -> list:
    from core.optimizers.chemistry.problem import build_problem, score_energy
    from core.optimizers.chemistry.sqdrift import run_sqdrift

    rows = []
    for bond, cpt in ((None, 25), (1.8, 25), (1.8, 100)):
        problem = build_problem("n2", bond)
        started = time.time()
        r = run_sqdrift(problem, circuits_per_time=cpt, shots=100, seed=0, platform=PLATFORM, execution_mode=MODE)
        sc = score_energy(r["sqd"]["energy"], problem)
        s = r["sampling"]
        rows.append({
            "molecule": "n2", "bond_length": bond, "circuits_per_time": cpt,
            "error_mha": sc["error_mha"], "within_chemical_accuracy": sc["within_chemical_accuracy"],
            "subspace_dimension": r["sqd"]["subspace_dimension"], "coverage": r["sqd"]["subspace_dimension"] / 14400,
            "valid_fraction": s["valid_fraction"], "distinct_valid_configurations": s["distinct_valid_configurations"],
            "distinct_bitstrings": s["distinct_bitstrings"], "total_shots": s["total_shots"],
            "backend": r.get("backend_name") or r.get("backend"), "vendor_runs": r.get("vendor_runs"),
            "timings": r.get("timings"), "wall_s": round(time.time() - started, 1),
        })
        print(f"SqDRIFT N2 {bond} cpt={cpt}: err {sc['error_mha']:.2f} mHa coverage {rows[-1]['coverage']:.2f} "
              f"valid {s['valid_fraction']:.2f} distinct {s['distinct_valid_configurations']}")
    return rows


def qaoa() -> list:
    from core.optimizers.network.qubo import build_network_qubo
    from core.optimizers.network.quantum_solver import solve_quantum
    from core.optimizers.network.scenario import build_scenario
    from e6_qaoa_control import all_energies, rank_pct

    problem = build_scenario("pilot_2of4")
    qubo = build_network_qubo(problem)
    energies, _ = all_energies(qubo)
    order = np.sort(energies)
    rows = []
    for s in (0, 1, 2):
        started = time.time()
        sol, diag = solve_quantum(problem, backend="gate", platform=PLATFORM or "cpu", execution_mode=MODE,
                                  seed=s, sampler_options={"reps": 1, "shots": 1024})
        rng = np.random.default_rng(1000 + s)
        rand_best = float(energies[rng.integers(0, 2 ** qubo.n_vars, 1024)].min())
        rows.append({
            "reps": 1, "seed": s, "qaoa_energy": float(diag["energy"]),
            "qaoa_rank_pct": rank_pct(order, float(diag["energy"])),
            "random_rank_pct": rank_pct(order, rand_best),
            "backend": diag.get("backend_name"), "platform": diag.get("platform"),
            "execution_mode": diag.get("execution_mode"), "wall_s": round(time.time() - started, 1),
        })
        print(f"QAOA p=1 s={s}: rank {rows[-1]['qaoa_rank_pct']:.4f}% vs random {rows[-1]['random_rank_pct']:.4f}% on {diag.get('backend_name')}")
    return rows


def main() -> None:
    load_token()
    t0 = time.time()
    parts = sys.argv[1:]
    which = [p for p in parts if not p.startswith("--")] or ["mlae", "sqdrift", "qaoa"]
    out = {"mode": MODE}
    if "mlae" in which:
        out["mlae"] = mlae()
    if "sqdrift" in which:
        out["sqdrift"] = sqdrift()
    if "qaoa" in which:
        out["qaoa"] = qaoa()
    out["wall_s"] = round(time.time() - t0, 1)
    save("e8_ibm_hardware" + ("_dryrun" if DRY else "") + ("_" + "_".join(which) if len(which) < 3 else ""), out)


if __name__ == "__main__":
    main()
