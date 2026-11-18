"""E24, post hoc - why no selection displayed a lower J than the classical steps alone.

Written after the result of E24 was known; not part of its design. The second
classical step minimises J over the assets the first step left with a positive
weight. If, for the selection "every asset", the first step leaves every asset
with a positive weight, the second step minimises J over the whole universe
under the caps, and a selection of three or more assets that the guard does not
replace can only restrict that minimisation. This script checks the premise on
the 30 instances with the product's own first step, and lists the exact
differences of J where the QAOA display differs from that of the classical
steps alone (read from the stored result of E24; nothing is re-run for them).

Usage: e24_posthoc.py
"""

from __future__ import annotations

import json

import numpy as np
from common import RESULTS, save

import e24_displayed_allocation as e24


def main() -> None:
    from api._executor_helpers import min_variance_weights

    stored = json.loads((RESULTS / "e24_displayed_allocation.json").read_text())
    support, differing = [], []
    for cell in stored["instances"]:
        n, instance = cell["n"], cell["instance"]
        _, prices = e24.instance_rows(n, instance)
        weights = np.array(list(min_variance_weights(prices, list(prices.columns)).values()))
        support.append({"n": n, "instance": instance, "assets": n, "positive_after_step_1": int((weights > 0).sum()),
                        "smallest_weight_after_step_1": float(weights.min()),
                        "held_above_1pct_after_all_steps": cell["all"]["held_above_1pct"]})
        runs = [r for r in cell["runs"] if r["arm"] == "method"]
        other = [r for r in runs if not r["same_as_all"]]
        if other:
            differing.append({"n": n, "instance": instance, "runs_that_differ": len(other), "runs": len(runs),
                              "J_minus_J_all_of_those_runs": [r["J"] - cell["all"]["J"] for r in other],
                              "largest_weight_difference": [float(np.max(np.abs(np.asarray(r["weights"]) - np.asarray(cell["all"]["weights"]))))
                                                            for r in other],
                              "optimum_same_as_all": cell["optimum"]["same_as_all"]})
    summary = {str(n): {"instances": sum(s["n"] == n for s in support),
                        "instances_where_step_1_keeps_every_asset": sum(s["n"] == n and s["positive_after_step_1"] == n for s in support),
                        "smallest_weight_after_step_1": min(s["smallest_weight_after_step_1"] for s in support if s["n"] == n)}
               for n in stored["sizes"]}
    for n, cell in summary.items():
        print(f"{n:>2} assets: step 1 on every asset keeps every asset on {cell['instances_where_step_1_keeps_every_asset']}"
              f" of {cell['instances']} instances (smallest weight {cell['smallest_weight_after_step_1']:.4f})")
    for d in differing:
        print(f"  differs: n={d['n']} instance {d['instance']}: {d['runs_that_differ']}/{d['runs']} runs,"
              f" J - J_all {[f'{x:.2e}' for x in d['J_minus_J_all_of_those_runs']]},"
              f" largest weight difference {[round(x, 4) for x in d['largest_weight_difference']]}")
    save("e24_posthoc", {"design": __doc__.split("Usage:")[0].strip(), "e24_commit": stored["provenance"]["commit"],
                         "summary": summary, "step_1_on_every_asset": support, "qaoa_runs_that_differ": differing})


if __name__ == "__main__":
    main()
