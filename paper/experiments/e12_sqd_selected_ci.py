"""E12 - SqDRIFT subspaces against a proper classical selected-CI referee.

E7 compared SqDRIFT with random and singles-and-doubles string selection and
named heat-bath selected CI as "the proper referee". This experiment runs it.

For N2 (10 orbitals, 14 electrons, FCI dimension 14 400) at equilibrium and at
1.8 A, and for each SqDRIFT effort level (circuits per time step) over 5 seeds:

  platform   the shipped pipeline: subsampled SQD with configuration recovery
             (qiskit-addon-sqd); energy and subspace dimension D as reported.
  sq_full    every distinct alpha/beta string SqDRIFT sampled (valid Hamming
             weight, spin-symmetrised union), diagonalised exactly in the
             product space - no subsampling, no recovery.
  random     |A| = |B| = round(sqrt(D)) strings: HF plus uniform random
             (5 draws), exact diagonalisation in the product space.
  sd         every alpha/beta string within two excitations of Hartree-Fock
             (85 strings, 50% of the FCI space): a fixed classical selection.
  oracle     the round(sqrt(D)) strings with the largest marginal weight in
             the exact FCI ground state. It uses the answer, so it is a
             reference for what a product space of that size can reach, not
             a competitor - and not a proven optimum either.
  sci        PySCF selected CI (pyscf.fci.selected_ci, which grows a product
             of alpha and beta string sets with a heat-bath-style criterion)
             over a grid of selection cutoffs; its error at dimension D is
             interpolated (log-log) from the measured (dimension, error)
             curve, never extrapolated. A determinant-based selected CI is
             not confined to product spaces and would be more compact still.

All product-space diagonalisations use PySCF selected_ci.kernel_fixed_space,
so every selection rule is scored by the same solver.
"""

from __future__ import annotations

import math
import os
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "AER_MAX_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
from common import FIGURES, mean_ci, mpl, point_seed, pool_workers, save  # noqa: E402
from pyscf import fci  # noqa: E402
from pyscf.fci import cistring, selected_ci  # noqa: E402

from core.optimizers.chemistry.problem import build_problem  # noqa: E402
from core.optimizers.chemistry.sqdrift import _occupation, build_sqdrift_circuits, diagonalize  # noqa: E402
from core.optimizers.quantum.shared.circuit_runner import run_measured_circuits  # noqa: E402

SETTINGS = [(None, 25), (None, 100), (1.8, 25), (1.8, 100), (1.8, 200)]
SEEDS = [0, 1, 2, 3, 4]
SHOTS = 100
RANDOM_DRAWS = 5
SCI_CUTOFFS = list(np.geomspace(3e-2, 1e-4, 13))
SOLVER_FLOOR_MHA = 1e-4  # eigensolver convergence; smaller differences are not resolved

_PROBLEMS = {}


def problem_for(bond):
    if bond not in _PROBLEMS:
        _PROBLEMS[bond] = build_problem("n2", bond)
    return _PROBLEMS[bond]


def to_int(occ):
    return sum(1 << i for i, v in enumerate(occ) if v)


def sd_strings(p):
    na = p.nelec[0]
    hf = to_int([1] * na + [0] * (p.norb - na))
    return [int(s) for s in cistring.make_strings(range(p.norb), na) if bin(int(s) & ~hf).count("1") <= 2]


def fixed_space_error(p, strs_a, strs_b):
    sa = np.asarray(sorted(set(strs_a)), dtype=np.int64)
    sb = np.asarray(sorted(set(strs_b)), dtype=np.int64)
    e, _ = selected_ci.kernel_fixed_space(selected_ci.SCI(), p.h1e, p.h2e, p.norb, p.nelec,
                                          (sa, sb), ecore=p.ecore)
    return 1000.0 * (float(e) - p.fci_energy), int(len(sa) * len(sb))


def _sqdrift_cell(args):
    bond, cpt, seed = args
    p = problem_for(bond)
    na, nb = p.nelec
    started = time.time()
    circuits, _ = build_sqdrift_circuits(p, circuits_per_time=cpt, seed=seed)
    run = run_measured_circuits(circuits, shots=SHOTS, execution_mode="simulator", seed=seed,
                                job_label="e12", min_qubits=2 * p.norb, simulator="aer")
    combined = Counter()
    for c in run.counts:
        combined.update(c)
    plat = diagonalize(p, dict(combined), seed=seed)
    plat_err = 1000.0 * (plat["energy"] - p.fci_energy)
    strings = set()
    for bits in combined:
        a, b = _occupation(bits, p.norb)
        if sum(a) == na and sum(b) == nb:
            strings.add(to_int(a))
            strings.add(to_int(b))
    full_err, full_dim = fixed_space_error(p, strings, strings)
    return {"bond": bond, "cpt": cpt, "seed": seed, "platform_err": plat_err,
            "platform_dim": plat["subspace_dimension"], "sq_full_err": full_err, "sq_full_dim": full_dim,
            "distinct_strings": len(strings), "wall_s": round(time.time() - started, 1)}


def _sci_cell(args):
    bond, cut = args
    p = problem_for(bond)
    sci = selected_ci.SCI()
    sci.select_cutoff = float(cut)
    sci.ci_coeff_cutoff = float(cut)
    e, c = sci.kernel(p.h1e, p.h2e, p.norb, p.nelec, ecore=p.ecore)
    return {"bond": bond, "cutoff": float(cut), "dim": int(len(c._strs[0]) * len(c._strs[1])),
            "err": 1000.0 * (float(e) - p.fci_energy)}


def fci_marginal_order(p):
    e, civec = fci.direct_spin1.FCI().kernel(p.h1e, p.h2e, p.norb, p.nelec, ecore=p.ecore)
    assert abs(e - p.fci_energy) < 1e-6, (e, p.fci_energy)
    strs = cistring.make_strings(range(p.norb), p.nelec[0])
    w = (np.abs(civec) ** 2).sum(axis=1) + (np.abs(civec) ** 2).sum(axis=0)  # closed shell: symmetrise
    return [int(strs[i]) for i in np.argsort(-w, kind="stable")]


def interp_err(curve, dim):
    pts = sorted((c["dim"], c["err"]) for c in curve if c["err"] > SOLVER_FLOOR_MHA)
    for (d0, e0), (d1, e1) in zip(pts, pts[1:]):
        if d0 <= dim <= d1 and d1 > d0:
            t = (math.log(dim) - math.log(d0)) / (math.log(d1) - math.log(d0))
            return float(math.exp(math.log(e0) + t * (math.log(e1) - math.log(e0))))
    return None  # outside the measured range of resolvable errors: not extrapolated


def main() -> None:
    t0 = time.time()
    for bond in (None, 1.8):  # solver sanity: the full product space reproduces FCI
        p = problem_for(bond)
        allstr = [int(s) for s in cistring.make_strings(range(p.norb), p.nelec[0])]
        err, dim = fixed_space_error(p, allstr, allstr)
        assert abs(err) < SOLVER_FLOOR_MHA and dim == math.comb(p.norb, p.nelec[0]) ** 2, (err, dim)
        # string convention: the lowest-orbital determinant alone gives the Hartree-Fock energy
        hf = to_int([1] * p.nelec[0] + [0] * (p.norb - p.nelec[0]))
        hf_err, _ = fixed_space_error(p, [hf], [hf])
        assert abs(hf_err - 1000.0 * (p.hf_energy - p.fci_energy)) < SOLVER_FLOOR_MHA, (hf_err, p.hf_energy, p.fci_energy)
    workers = pool_workers()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        sq_rows = list(ex.map(_sqdrift_cell, [(b, c, s) for b, c in SETTINGS for s in SEEDS]))
        sci = list(ex.map(_sci_cell, [(b, c) for b in (None, 1.8) for c in SCI_CUTOFFS]))
        repeat = ex.submit(_sqdrift_cell, tuple(SETTINGS[0]) + (SEEDS[0],)).result()

    first = sq_rows[0]
    reproducibility = {
        "setting": [SETTINGS[0][0], SETTINGS[0][1], SEEDS[0]],
        "platform_err": [first["platform_err"], repeat["platform_err"]],
        "platform_dim": [first["platform_dim"], repeat["platform_dim"]],
        "bit_identical": first["platform_err"] == repeat["platform_err"] and first["platform_dim"] == repeat["platform_dim"],
    }
    print("same seed, second process:", reproducibility)

    fci_dim = math.comb(10, 7) ** 2
    sd = {}
    for bond in (None, 1.8):
        p = problem_for(bond)
        strs = sd_strings(p)
        err, dim = fixed_space_error(p, strs, strs)
        sd["eq" if bond is None else str(bond)] = {"err": err, "dim": dim, "strings": len(strs), "coverage": dim / fci_dim}
        print(f"N2 {bond} S+D strings: {err:.4f} mHa at dim {dim}")

    oracle_order = {b: fci_marginal_order(problem_for(b)) for b in (None, 1.8)}
    for r in sq_rows:
        p = problem_for(r["bond"])
        k = max(2, int(round(math.sqrt(r["platform_dim"]))))
        top = oracle_order[r["bond"]][:k]
        r["oracle_err"], r["oracle_dim"] = fixed_space_error(p, top, top)
        hf = to_int([1] * p.nelec[0] + [0] * (p.norb - p.nelec[0]))
        allstr = [int(s) for s in cistring.make_strings(range(p.norb), p.nelec[0]) if int(s) != hf]
        errs = []
        for d in range(RANDOM_DRAWS):
            rng = np.random.default_rng(point_seed(12, int(10 * (r["bond"] or 0)), r["cpt"], r["seed"], d))
            pick = [hf] + [allstr[i] for i in rng.choice(len(allstr), size=k - 1, replace=False)]
            errs.append(fixed_space_error(p, pick, pick)[0])
        r["random_err"] = float(np.mean(errs))
        curve = [h for h in sci if h["bond"] == r["bond"]]
        r["sci_err_at_dim"] = interp_err(curve, r["platform_dim"])
        r["sci_err_at_full_dim"] = interp_err(curve, r["sq_full_dim"])
        print(f"N2 {r['bond']} cpt={r['cpt']:3d} s={r['seed']} D={r['platform_dim']:5d} "
              f"platform={r['platform_err']:.2f} full={r['sq_full_err']:.2f}@{r['sq_full_dim']} "
              f"random={r['random_err']:.1f} sci={r['sci_err_at_dim']} oracle={r['oracle_err']:.3f}")

    summary = []
    for bond, cpt in SETTINGS:
        sel = [r for r in sq_rows if r["bond"] == bond and r["cpt"] == cpt]
        sci_vals = [r["sci_err_at_dim"] for r in sel if r["sci_err_at_dim"] is not None]
        full_pairs = [(r["sq_full_err"], r["sci_err_at_full_dim"]) for r in sel if r["sci_err_at_full_dim"] is not None]
        summary.append({
            "bond": bond, "cpt": cpt, "runs": len(sel),
            "coverage": mean_ci([r["platform_dim"] / fci_dim for r in sel]),
            "platform_err": mean_ci([r["platform_err"] for r in sel]),
            "platform_within_chemical_accuracy": sum(r["platform_err"] <= 1.6 for r in sel),
            "sq_full_err": mean_ci([r["sq_full_err"] for r in sel]),
            "sq_full_coverage": mean_ci([r["sq_full_dim"] / fci_dim for r in sel]),
            "random_err": mean_ci([r["random_err"] for r in sel]),
            "sci_err": mean_ci(sci_vals) if sci_vals else None,
            "sci_err_at_full_dim": mean_ci([s for _, s in full_pairs]) if full_pairs else None,
            "sci_resolved_at_full_dim_runs": len(full_pairs),
            "oracle_err": mean_ci([r["oracle_err"] for r in sel]),
            "platform_over_sci": mean_ci([r["platform_err"] / r["sci_err_at_dim"] for r in sel
                                          if r["sci_err_at_dim"]]) if sci_vals else None,
            "platform_worse_than_sci_runs": sum(r["platform_err"] > r["sci_err_at_dim"] for r in sel if r["sci_err_at_dim"]),
            "sq_full_over_sci": mean_ci([f / s for f, s in full_pairs]) if full_pairs else None,
        })
    save("e12_sqd_selected_ci", {"settings": SETTINGS, "seeds": SEEDS, "shots": SHOTS,
                                 "sci_curve": sci, "sd_strings": sd, "reproducibility": reproducibility,
                                 "rows": sq_rows, "summary": summary,
                                 "wall_s": round(time.time() - t0, 1)})

    plt = mpl()
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.5), sharey=True)
    for a, bond, title in ((ax[0], None, "N$_2$ equilibrium"), (ax[1], 1.8, "N$_2$ 1.8 Å")):
        curve = sorted((h["dim"], max(h["err"], 1e-4)) for h in sci if h["bond"] == bond)
        a.loglog([c[0] for c in curve], [c[1] for c in curve], "k-", lw=0.9, label="selected CI (PySCF)")
        sel = [r for r in sq_rows if r["bond"] == bond]
        a.loglog([r["platform_dim"] for r in sel], [r["platform_err"] for r in sel], "o", ms=3, label="SqDRIFT + SQD (platform)")
        a.loglog([r["platform_dim"] for r in sel], [r["random_err"] for r in sel], "x", ms=3, label="random strings")
        a.loglog([r["oracle_dim"] for r in sel], [max(r["oracle_err"], 1e-4) for r in sel], "^", ms=3, label="FCI-weight oracle")
        point = sd["eq" if bond is None else str(bond)]
        a.loglog([point["dim"]], [point["err"]], "s", ms=4, label="S+D strings")
        a.axhline(1.6, color="grey", ls=":", lw=0.8)
        a.set_title(title)
        a.set_xlabel("subspace dimension")
    ax[0].set_ylabel("error above FCI (mHa)")
    ax[0].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(FIGURES / "sqd_selected_ci.pdf")
    print("done", round(time.time() - t0, 1), "s")


if __name__ == "__main__":
    main()
