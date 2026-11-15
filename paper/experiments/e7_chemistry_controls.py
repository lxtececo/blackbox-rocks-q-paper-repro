"""E7 — Is the quantum sampler better than classical configuration selection?

SQD diagonalizes in the product of the distinct alpha- and beta-strings it is
given. We therefore compare *subspaces*: for each N2 setting of E4, build
classical subspaces and diagonalize them with the same routine, without
subsampling (one batch containing every configuration).

  random_matched: HF strings plus uniformly random alpha/beta strings, sized
                  so |A'| x |B'| matches the mean SqDRIFT subspace dimension;
  cisd_strings:   all alpha/beta strings up to double excitation from HF.

SqDRIFT figures are taken from E4 (platform defaults, frequency-weighted
subsampling of 300 configurations per batch, 3 batches).
"""

from __future__ import annotations

import itertools
import json
import math
import time

import numpy as np
from common import RESULTS, mean_ci, save

from core.optimizers.chemistry.problem import build_problem, score_energy
from core.optimizers.chemistry.sqdrift import diagonalize

SETTINGS = [("n2", None, 25), ("n2", None, 100), ("n2", 1.8, 25), ("n2", 1.8, 100), ("n2", 1.8, 200)]
SEEDS = [0, 1, 2]


def bitstring(alpha, beta):
    # Matches sqdrift._occupation: qubit q is character -1-q; alpha qubits first, then beta.
    return "".join(str(b) for b in reversed(beta)) + "".join(str(a) for a in reversed(alpha))


def strings(norb, n_occ):
    out = []
    for occ in itertools.combinations(range(norb), n_occ):
        v = [0] * norb
        for i in occ:
            v[i] = 1
        out.append(tuple(v))
    return out


def excitation(v, ref):
    return sum(1 for a, b in zip(v, ref) if a and not b)


def diag_product(problem, alphas, betas, seed):
    counts = {bitstring(a, b): 1 for a in alphas for b in betas}
    d = diagonalize(problem, counts, samples_per_batch=len(counts), num_batches=1, seed=seed)
    return score_energy(d["energy"], problem)["error_mha"], d["subspace_dimension"]


def main() -> None:
    t0 = time.time()
    e4 = json.loads((RESULTS / "e4_chemistry.json").read_text())
    rows = []
    cache = {}
    for mol, bond, cpt in SETTINGS:
        if (mol, bond) not in cache:
            cache[(mol, bond)] = build_problem(mol, bond)
        problem = cache[(mol, bond)]
        norb = problem.norb
        na, nb = problem.nelec
        A, B = strings(norb, na), strings(norb, nb)
        fci = len(A) * len(B)
        ha, hb = tuple([1] * na + [0] * (norb - na)), tuple([1] * nb + [0] * (norb - nb))
        sq = [r for r in e4["rows"] if r["molecule"] == mol and r["bond_length"] == bond
              and r["circuits_per_time"] == cpt and "failed" not in r]
        target_dim = float(np.mean([r["subspace_dimension"] for r in sq]))

        ca = [a for a in A if excitation(a, ha) <= 2]
        cb = [b for b in B if excitation(b, hb) <= 2]
        err, dim = diag_product(problem, ca, cb, 0)
        rows.append({"molecule": mol, "bond_length": bond, "circuits_per_time": cpt, "control": "cisd_strings",
                     "error_mha": err, "coverage": dim / fci})

        k = max(2, int(round(math.sqrt(target_dim))))
        errs, covs = [], []
        for s in SEEDS:
            rng = np.random.default_rng(700 + s)
            ra = [ha] + [A[i] for i in rng.choice([i for i, a in enumerate(A) if a != ha], size=k - 1, replace=False)]
            rb = [hb] + [B[i] for i in rng.choice([i for i, b in enumerate(B) if b != hb], size=k - 1, replace=False)]
            e, d = diag_product(problem, ra, rb, s)
            errs.append(e)
            covs.append(d / fci)
        rows.append({"molecule": mol, "bond_length": bond, "circuits_per_time": cpt, "control": "random_matched",
                     "error_mha": {"mean": float(np.mean(errs)), "min": float(min(errs)), "max": float(max(errs))},
                     "coverage": mean_ci(covs)})
        sq_err = [r["error_mha"] for r in sq]
        rows.append({"molecule": mol, "bond_length": bond, "circuits_per_time": cpt, "control": "sqdrift",
                     "error_mha": {"mean": float(np.mean(sq_err)), "min": float(min(sq_err)), "max": float(max(sq_err))},
                     "coverage": mean_ci([r["subspace_fraction"] for r in sq])})
        print(f"{mol} {bond} cpt={cpt}: SqDRIFT {np.mean(sq_err):.2f} mHa @ {rows[-1]['coverage']['mean']:.2f} | "
              f"random {np.mean(errs):.2f} [{min(errs):.2f},{max(errs):.2f}] @ {np.mean(covs):.2f} | "
              f"CISD-strings {err:.2f} @ {dim / fci:.2f}")
    save("e7_chemistry_controls", {"rows": rows, "wall_s": round(time.time() - t0, 1)})


if __name__ == "__main__":
    main()
