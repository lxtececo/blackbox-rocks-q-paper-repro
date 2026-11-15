"""E9 — Does the hardware SqDRIFT result carry quantum signal, or is it SQD on noise?

On ibm_marrakesh (E8) only 1-2% of shots had the right particle number, yet SQD
reached sub-0.1 mHa because configuration recovery expanded the noisy shots to
60-70% of the FCI space. The control is a fully depolarised device: uniformly
random bitstrings, the same shot count, the same `diagonalize` call (platform
defaults) and the same seed for SQD's internal subsampling.
"""

from __future__ import annotations

import json
import time
from collections import Counter

import numpy as np
from common import RESULTS, save

from core.optimizers.chemistry.problem import build_problem, score_energy
from core.optimizers.chemistry.sqdrift import diagonalize

SEEDS = [0, 1, 2]


def main() -> None:
    t0 = time.time()
    hw = json.loads((RESULTS / "e8_ibm_hardware_sqdrift.json").read_text())["sqdrift"]
    rows = []
    for h in hw:
        problem = build_problem(h["molecule"], h["bond_length"])
        nq = 2 * problem.norb
        errs, covs, valid = [], [], []
        for s in SEEDS:
            rng = np.random.default_rng(900 + s)
            bits = rng.integers(0, 2, size=(h["total_shots"], nq))
            counts = Counter("".join(map(str, b)) for b in bits)
            d = diagonalize(problem, dict(counts), seed=0)
            errs.append(score_energy(d["energy"], problem)["error_mha"])
            covs.append(d["subspace_dimension"] / 14400)
            ones = bits.sum(axis=1)
            valid.append(float(np.mean(ones == sum(problem.nelec))))
        row = {
            "molecule": h["molecule"], "bond_length": h["bond_length"], "circuits_per_time": h["circuits_per_time"],
            "total_shots": h["total_shots"],
            "hardware": {"error_mha": h["error_mha"], "coverage": h["coverage"], "valid_fraction": h["valid_fraction"]},
            "uniform_noise": {"error_mha": {"mean": float(np.mean(errs)), "min": float(min(errs)), "max": float(max(errs))},
                              "coverage": float(np.mean(covs)), "valid_fraction_total_n": float(np.mean(valid))},
        }
        rows.append(row)
        print(f"{h['molecule']} {h['bond_length']} cpt={h['circuits_per_time']}: hardware {h['error_mha']:.3f} mHa "
              f"@ {h['coverage']:.2f} | uniform noise {np.mean(errs):.3f} [{min(errs):.3f},{max(errs):.3f}] "
              f"@ {np.mean(covs):.2f}")
    save("e9_sqd_noise_control", {"rows": rows, "wall_s": round(time.time() - t0, 1)})


if __name__ == "__main__":
    main()
