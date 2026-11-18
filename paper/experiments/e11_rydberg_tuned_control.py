"""E11 - Rydberg MIS: static-radius mapping against a *tuned* dynamic-radius control.

E3 compared the platform's mapping, which places the spacing threshold at the
static blockade radius (C6/Delta_end)^(1/6) of the final detuning, against a
mapping that places it at the dynamic blockade radius (C6/Omega)^(1/6) and
sweeps to a fixed Delta_end/Omega = 6 (both radii as defined in the Aquila
whitepaper, arXiv:2306.11727). A reviewer can fairly ask whether that control
was simply mis-tuned. Here the control's only free parameter, Delta_end/Omega,
is tuned on one set of random instances and both mappings are then compared
on a disjoint test set (paired by instance), so the tuning cannot overfit the
comparison.

Metrics per instance, from 200 shots before any classical repair:
  raw optimal fraction      share of shots whose excited set is a maximum
                            independent set (the tuning criterion);
  raw independent fraction  share of shots that violate no spacing rule.
"""

from __future__ import annotations

import math
import os
import time
from concurrent.futures import ProcessPoolExecutor

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
from common import FIGURES, mean_ci, mpl, point_seed, pool_workers, save  # noqa: E402
from e3_rydberg import omega_mapping_embedding, random_points, run, sites_from_km  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

from core.optimizers.network import rydberg_mis as rm  # noqa: E402

RATIOS = [1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0]
N_TUNE, N_TEST = 24, 24


def make_instances(seed: int, count: int):
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < count:
        pts = random_points(rng, 8)
        if pts is None:
            continue
        d_min = min(math.dist(a, b) for i, a in enumerate(pts) for b in pts[i + 1:])
        spacing = d_min * float(rng.uniform(1.4, 2.4))
        g = rm.conflict_graph(rm.SpacingProblem(sites_from_km(pts), spacing))
        if not g.edges:
            continue
        try:
            rm.embed(g, spacing, max_atoms=rm.MAX_LOCAL_ATOMS)
        except rm.EmbeddingError:
            continue
        out.append((pts, spacing))
    return out


def _cell(args):
    idx, pts, spacing, mapping, ratio, seed = args
    g = rm.conflict_graph(rm.SpacingProblem(sites_from_km(pts), spacing))
    opt = len(rm.exact_mis(g))
    if mapping == "cutoff":
        emb = rm.embed(g, spacing, max_atoms=rm.MAX_LOCAL_ATOMS)
    else:
        base = omega_mapping_embedding(g, spacing)
        emb = rm.Embedding(positions_m=base.positions_m, scale_um_per_km=base.scale_um_per_km,
                           threshold_km=base.threshold_km, blockade_radius_m=base.blockade_radius_m,
                           omega_max=base.omega_max, delta_max=min(rm.DELTA_MAX, ratio * base.omega_max))
    d = run(g, emb, opt, seed=seed)
    return idx, mapping, ratio, d["raw_optimal_fraction"], d["raw_independent_fraction"], d["shots_used"]


def main() -> None:
    t0 = time.time()
    tune = make_instances(7001, N_TUNE)
    test = make_instances(7002, N_TEST)
    jobs = []
    for split_id, (split, insts) in enumerate((("tune", tune), ("test", test))):
        for i, (pts, sp) in enumerate(insts):
            key = f"{split}{i}"
            jobs.append((key, pts, sp, "cutoff", None, point_seed(11, split_id, i, 0)))
            for k, r in enumerate(RATIOS):
                jobs.append((key, pts, sp, "omega", r, point_seed(11, split_id, i, 1 + k)))
    workers = pool_workers()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        cells = list(ex.map(_cell, jobs))

    def frac(split, mapping, ratio=None, which=3):
        n = N_TUNE if split == "tune" else N_TEST
        return [next(c[which] for c in cells if c[0] == f"{split}{i}" and c[1] == mapping and c[2] == ratio)
                for i in range(n)]

    tune_curve = [{"ratio": r, "raw_optimal": mean_ci(frac("tune", "omega", r)),
                   "raw_independent": mean_ci(frac("tune", "omega", r, 4))} for r in RATIOS]
    best = max(tune_curve, key=lambda row: row["raw_optimal"]["mean"])["ratio"]
    test_cut = np.asarray(frac("test", "cutoff"))
    test_omega = np.asarray(frac("test", "omega", best))
    diff = test_cut - test_omega
    ind_cut = np.asarray(frac("test", "cutoff", None, 4))
    ind_omega = np.asarray(frac("test", "omega", best, 4))

    def paired_p(a, b):
        try:
            return float(wilcoxon(a, b).pvalue)
        except ValueError:  # all differences zero
            return None

    p_value = paired_p(test_cut, test_omega)
    test_curve = [{"ratio": r, "raw_optimal": mean_ci(frac("test", "omega", r)),
                   "raw_independent": mean_ci(frac("test", "omega", r, 4))} for r in RATIOS]
    out = {
        "ratios": RATIOS, "n_tune": N_TUNE, "n_test": N_TEST, "shots": 200,
        "tune_curve": tune_curve, "tune_cutoff": mean_ci(frac("tune", "cutoff")),
        "tune_cutoff_raw_independent": mean_ci(frac("tune", "cutoff", None, 4)),
        "best_ratio_on_tune": best,
        "test": {"cutoff": mean_ci(test_cut), "omega_tuned": mean_ci(test_omega),
                 "paired_difference": mean_ci(diff), "wilcoxon_p": p_value,
                 "cutoff_wins": int((diff > 0).sum()), "ties": int((diff == 0).sum()),
                 "omega_wins": int((diff < 0).sum()),
                 "cutoff_raw_independent": mean_ci(ind_cut),
                 "omega_tuned_raw_independent": mean_ci(ind_omega),
                 "independent_paired_difference": mean_ci(ind_cut - ind_omega),
                 "independent_wilcoxon_p": paired_p(ind_cut, ind_omega),
                 "independent_cutoff_wins": int((ind_cut > ind_omega).sum()),
                 "independent_ties": int((ind_cut == ind_omega).sum())},
        "test_curve_all_ratios": test_curve,
        "cells": [list(c) for c in cells], "wall_s": round(time.time() - t0, 1),
    }
    save("e11_rydberg_tuned_control", out)
    print(f"best ratio {best}; test cutoff {out['test']['cutoff']} omega {out['test']['omega_tuned']} "
          f"diff {out['test']['paired_difference']} p={p_value}")
    for row in tune_curve:
        print(row["ratio"], round(row["raw_optimal"]["mean"], 3), round(row["raw_independent"]["mean"], 3))

    plt = mpl()
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.4), sharex=True)
    for a, key, name in ((ax[0], "raw_optimal", "raw optimal fraction"), (ax[1], "raw_independent", "raw independent fraction")):
        a.errorbar(RATIOS, [r[key]["mean"] for r in tune_curve], yerr=[r[key]["ci95"] for r in tune_curve],
                   fmt="o-", ms=3, capsize=2, label="dynamic-radius control (tuning set)")
        a.errorbar(RATIOS, [r[key]["mean"] for r in test_curve], yerr=[r[key]["ci95"] for r in test_curve],
                   fmt="s--", ms=3, capsize=2, label="dynamic-radius control (test set)")
        ref = out["test"]["cutoff" if key == "raw_optimal" else "cutoff_raw_independent"]["mean"]
        a.axhline(ref, color="k", lw=0.8, label="static-radius mapping (test set)")
        a.set_xscale("log")
        a.set_xlabel("control sweep ratio Δ$_{end}$/Ω")
        a.set_ylabel(name)
    ax[0].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(FIGURES / "rydberg_tuned_control.pdf")
    print("independent:", out["test"]["cutoff_raw_independent"], out["test"]["omega_tuned_raw_independent"],
          "p=", out["test"]["independent_wilcoxon_p"])
    print("done", round(time.time() - t0, 1), "s")


if __name__ == "__main__":
    main()
