"""E4 — Molecular ground-state energy with SqDRIFT + SQD.

Error vs FCI and, crucially, the SQD subspace dimension relative to the full
FCI space: when the subspace covers the whole space, SQD is exact regardless of
what the quantum sampler produced, and the run says nothing about the sampler.
"""

from __future__ import annotations

import math
import time

from common import FIGURES, mpl, save

from core.optimizers.chemistry.problem import build_problem, score_energy
from core.optimizers.chemistry.sqdrift import run_sqdrift

PLAN = [
    ("h2", None, [25, 100], [0, 1, 2]),
    ("lih", None, [25, 100], [0, 1, 2]),
    ("h2o", None, [25, 100], [0, 1, 2]),
    ("n2", None, [25, 100], [0, 1]),
    ("n2", 1.8, [25, 100, 200], [0, 1]),
]
SHOTS = 100


def main() -> None:
    t0 = time.time()
    rows = []
    for mol, bond, efforts, seeds in PLAN:
        problem = build_problem(mol, bond)
        na, nb = problem.nelec
        fci_dim = math.comb(problem.norb, na) * math.comb(problem.norb, nb)
        for cpt in efforts:
            for s in seeds:
                started = time.time()
                try:
                    r = run_sqdrift(problem, circuits_per_time=cpt, shots=SHOTS, seed=s)
                except Exception as exc:  # noqa: BLE001 - recorded, not hidden
                    rows.append({"molecule": mol, "bond_length": bond, "circuits_per_time": cpt,
                                 "seed": s, "failed": str(exc)[:200]})
                    print(f"{mol} {bond} cpt={cpt} s={s} FAILED: {exc}")
                    continue
                score = score_energy(r["sqd"]["energy"], problem)
                sampling = r.get("sampling", {})
                rows.append({
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
                })
                print(f"{mol:4s} {bond} cpt={cpt:3d} s={s} err={score['error_mha']:8.3f} mHa "
                      f"sub={rows[-1]['subspace_dimension']}/{fci_dim} t={rows[-1]['wall_s']}s")
    save("e4_chemistry", {"shots_per_circuit": SHOTS, "rows": rows, "wall_s": round(time.time() - t0, 1)})

    plt = mpl()
    fig, ax = plt.subplots(figsize=(3.3, 2.4))
    labels = {("h2", None): "H$_2$", ("lih", None): "LiH", ("h2o", None): "H$_2$O",
              ("n2", None): "N$_2$ eq.", ("n2", 1.8): "N$_2$ 1.8 Å"}
    for key, label in labels.items():
        sel = [r for r in rows if (r["molecule"], r["bond_length"]) == key and "failed" not in r]
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
