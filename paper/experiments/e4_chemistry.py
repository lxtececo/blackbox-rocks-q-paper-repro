"""E4 — Molecular ground-state energy with SqDRIFT + SQD.

Error vs FCI and, crucially, the SQD subspace dimension relative to the full
FCI space: when the subspace covers the whole space, SQD is exact regardless of
what the quantum sampler produced, and the run says nothing about the sampler.

Five seeds per setting; every run is one single-threaded worker process, which
makes a seed bit-reproducible (E12 re-runs the N2 settings and must agree).
"""

from __future__ import annotations

import math
import os
import time
from concurrent.futures import ProcessPoolExecutor

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "AER_MAX_THREADS"):
    os.environ.setdefault(_v, "1")

from common import FIGURES, mean_ci, mpl, pool_workers, save  # noqa: E402

from core.optimizers.chemistry.problem import build_problem, score_energy  # noqa: E402
from core.optimizers.chemistry.sqdrift import run_sqdrift  # noqa: E402

SEEDS = [0, 1, 2, 3, 4]
PLAN = [
    ("h2", None, [25, 100]),
    ("lih", None, [25, 100]),
    ("h2o", None, [25, 100]),
    ("n2", None, [25, 100]),
    ("n2", 1.8, [25, 100, 200]),
]
SHOTS = 100


def _cell(args):
    mol, bond, cpt, s = args
    problem = build_problem(mol, bond)
    na, nb = problem.nelec
    fci_dim = math.comb(problem.norb, na) * math.comb(problem.norb, nb)
    started = time.time()
    try:
        r = run_sqdrift(problem, circuits_per_time=cpt, shots=SHOTS, seed=s)
    except Exception as exc:  # noqa: BLE001 - recorded, not hidden
        return {"molecule": mol, "bond_length": bond, "circuits_per_time": cpt, "seed": s, "failed": str(exc)[:200]}
    score = score_energy(r["sqd"]["energy"], problem)
    sampling = r.get("sampling", {})
    return {
        "molecule": mol, "bond_length": bond, "norb": problem.norb, "nelec": [na, nb],
        "qubits": 2 * problem.norb, "fci_dimension": fci_dim,
        "circuits_per_time": cpt, "seed": s, "error_mha": score["error_mha"],
        "within_chemical_accuracy": score["within_chemical_accuracy"],
        "correlation_recovered_pct": score.get("correlation_recovered_pct"),
        "subspace_dimension": r["sqd"]["subspace_dimension"],
        "subspace_fraction": r["sqd"]["subspace_dimension"] / fci_dim,
        "sqd_iterations": len(r["sqd"]["iterations"]),
        "sampling": {k: v for k, v in sampling.items() if isinstance(v, (int, float))},
        "wall_s": round(time.time() - started, 1),
    }


def main() -> None:
    t0 = time.time()
    jobs = [(mol, bond, cpt, s) for mol, bond, efforts in PLAN for cpt in efforts for s in SEEDS]
    with ProcessPoolExecutor(max_workers=pool_workers()) as ex:
        rows = list(ex.map(_cell, jobs))
    failed = [r for r in rows if "failed" in r]
    assert not failed, failed
    summary = []
    for mol, bond, efforts in PLAN:
        for cpt in efforts:
            sel = [r for r in rows if (r["molecule"], r["bond_length"], r["circuits_per_time"]) == (mol, bond, cpt)]
            summary.append({
                "molecule": mol, "bond_length": bond, "circuits_per_time": cpt, "runs": len(sel),
                "qubits": sel[0]["qubits"], "fci_dimension": sel[0]["fci_dimension"],
                "abs_error_mha": mean_ci([abs(r["error_mha"]) for r in sel]),
                "max_abs_error_mha": max(abs(r["error_mha"]) for r in sel),
                "coverage": mean_ci([r["subspace_fraction"] for r in sel]),
                "within_chemical_accuracy": sum(bool(r["within_chemical_accuracy"]) for r in sel),
                "correlation_recovered_pct": mean_ci([r["correlation_recovered_pct"] for r in sel]),
                "distinct_valid_configurations": mean_ci([r["sampling"].get("distinct_valid_configurations", float("nan")) for r in sel]),
            })
            s = summary[-1]
            print(f"{mol:4s} {bond} cpt={cpt:3d}: |err|={s['abs_error_mha']['mean']:.4f}±{s['abs_error_mha']['ci95']:.4f} mHa "
                  f"coverage={s['coverage']['mean']:.3f} acc={s['within_chemical_accuracy']}/{len(sel)}")
    save("e4_chemistry", {"shots_per_circuit": SHOTS, "seeds": SEEDS, "summary": summary, "rows": rows,
                          "wall_s": round(time.time() - t0, 1)})

    plt = mpl()
    fig, ax = plt.subplots(figsize=(3.3, 2.4))
    labels = {("h2", None): "H$_2$", ("lih", None): "LiH", ("h2o", None): "H$_2$O",
              ("n2", None): "N$_2$ eq.", ("n2", 1.8): "N$_2$ 1.8 Å"}
    for key, label in labels.items():
        sel = [r for r in rows if (r["molecule"], r["bond_length"]) == key]
        ax.scatter([r["subspace_fraction"] for r in sel], [max(abs(r["error_mha"]), 1e-4) for r in sel], s=10, label=label)
    ax.axhline(1.6, color="grey", ls=":", lw=0.8)
    ax.text(0.02, 1.9, "chemical accuracy", fontsize=6, color="grey", transform=ax.get_yaxis_transform())
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("SQD subspace / FCI dimension")
    ax.set_ylabel("|error vs FCI| (mHa)")
    ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(FIGURES / "chemistry_subspace.pdf")
    print("done", round(time.time() - t0, 1), "s")


if __name__ == "__main__":
    main()
