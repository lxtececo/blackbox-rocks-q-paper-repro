"""E13 - Statistical re-analysis of the IBM hardware runs (E8) and the noise control (E9).

No new hardware time. Uses the stored per-power hit counts and valid-shot
fractions to answer two precise questions:

 (a) MLAE on ibm_marrakesh: at which Grover powers is the measured P(1)
     statistically distinguishable from the ideal value, and from 1/2 (the
     fully depolarised value)? Exact two-sided binomial tests, Wilson 95% CIs.
 (b) SqDRIFT on hardware: is the valid-configuration fraction (correct alpha
     and beta electron counts) different from what uniformly random 20-bit
     strings give, C(10,7)^2 / 2^20? Exact binomial tests.
"""

from __future__ import annotations

import json
import math

from common import RESULTS, save, wilson
from scipy.stats import binomtest

SHOTS_MLAE = 200


def main() -> None:
    e1 = json.loads((RESULTS / "e1_montecarlo.json").read_text())
    a = e1["instance"]["true_amplitude"]
    theta = math.asin(math.sqrt(a))
    mlae = json.loads((RESULTS / "e8_ibm_hardware_mlae.json").read_text())
    seen = {}
    for run in mlae["mlae"]:
        assert run["queries"] == SHOTS_MLAE * sum(2 * k + 1 for k in run["grover_schedule"])
        for k, hits in zip(run["grover_schedule"], run["hits_per_power"]):
            seen.setdefault(k, []).append((run["max_power"], hits))
    rows = []
    for k in sorted(seen):
        ideal = math.sin((2 * k + 1) * theta) ** 2
        for job, hits in seen[k]:
            lo, hi = wilson(hits, SHOTS_MLAE)
            rows.append({
                "grover_power": k, "job_max_power": job, "hits": hits, "shots": SHOTS_MLAE,
                "p_hat": hits / SHOTS_MLAE, "wilson95": [lo, hi], "ideal": ideal,
                "p_vs_ideal": binomtest(hits, SHOTS_MLAE, ideal).pvalue,
                "p_vs_half": binomtest(hits, SHOTS_MLAE, 0.5).pvalue,
                # can 200 shots tell ideal from 1/2 at all? (ideal outside the CI a fair coin would give)
                "ideal_resolvable_from_half": abs(ideal - 0.5) > 1.96 * math.sqrt(0.25 / SHOTS_MLAE),
            })
            print(f"k={k} (job m={job}) {hits}/200 = {hits / 200:.3f} [{lo:.3f},{hi:.3f}] ideal {ideal:.3f} "
                  f"p(ideal)={rows[-1]['p_vs_ideal']:.1e} p(1/2)={rows[-1]['p_vs_half']:.3f}")

    p_uniform = math.comb(10, 7) ** 2 / 2 ** 20
    hw = json.loads((RESULTS / "e8_ibm_hardware_sqdrift.json").read_text())
    valid = []
    for r in hw["sqdrift"]:
        n = r["total_shots"]
        k = round(r["valid_fraction"] * n)
        lo, hi = wilson(k, n)
        valid.append({
            "bond_length": r["bond_length"], "circuits_per_time": r["circuits_per_time"], "shots": n,
            "valid_shots": k, "valid_fraction": k / n, "wilson95": [lo, hi], "uniform_rate": p_uniform,
            "expected_under_uniform": p_uniform * n,
            "ratio_to_uniform": (k / n) / p_uniform,
            "p_vs_uniform": binomtest(k, n, p_uniform).pvalue,
            "z_vs_uniform": (k - p_uniform * n) / math.sqrt(n * p_uniform * (1 - p_uniform)),
        })
        print(f"N2 {r['bond_length']} cpt={r['circuits_per_time']}: {k}/{n} valid vs {p_uniform * n:.1f} "
              f"uniform, z={valid[-1]['z_vs_uniform']:.2f}, p={valid[-1]['p_vs_uniform']:.1e}")

    save("e13_hardware_statistics", {"mlae_true_amplitude": a, "mlae": rows,
                                      "sqdrift_valid_fraction": valid, "uniform_valid_rate": p_uniform})


if __name__ == "__main__":
    main()
