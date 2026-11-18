"""E17 - Derived statistics quoted in the paper that the experiments do not store themselves.

No simulation and no hardware time. Reads the result files of E1, E2, E5,
E10, E12, E15 and E19 and writes every derived number the text quotes (counts of
runs better or worse than a referee, per-run ratio ranges, enrichment
factors, extrapolated gate counts), so that each can be traced to a file
instead of to a calculation done by hand.

Post hoc by construction: these are descriptions of stored results, not tests
planned in advance. The one test added here, a two-sided sign test of the
refined facility-location gap against the referee's gap, is reported next to
the t-test that E2 planned, because the refined gaps take only a handful of
distinct values and are far from normal.
"""

from __future__ import annotations

import json
import math

import numpy as np
from common import RESULTS, save
from scipy.stats import binomtest

CHEMICAL_ACCURACY_MHA = 1.6
N2_FCI_DIMENSION = math.comb(10, 7) ** 2
TIE = 1e-6  # percentage points


def load(name: str) -> dict:
    return json.loads((RESULTS / f"{name}.json").read_text())


def against_referee(cell: dict, referee_gap: float) -> dict:
    runs = np.array(cell["refined_gap_pct_runs"], float)
    better = int((runs < referee_gap - TIE).sum())
    worse = int((runs > referee_gap + TIE).sum())
    equal = int(len(runs) - better - worse)
    return {
        "runs": int(len(runs)), "better_than_referee": better, "equal_to_referee": equal, "worse_than_referee": worse,
        "sign_test_p": float(binomtest(worse, worse + better, 0.5).pvalue) if worse + better else None,
        "t_test_p": cell["refined_vs_referee_p"],
        "distinct_refined_gaps_pct": sorted({round(float(v), 2) for v in runs}),
        "best_refined_gap_pct": float(runs.min()), "mean_refined_gap_pct": float(runs.mean()),
        "mean_raw_cost_vs_optimum_pct": cell["raw_gap_pct"]["mean"], "raw_feasible_runs": cell["raw_feasible_runs"],
        "mean_elapsed_s": cell["elapsed_ms"]["mean"] / 1000.0,
    }


def facility() -> dict:
    e2 = load("e2_facility")
    out = {"default_budget": {}, "budget_arm": {}}
    referee = {}
    for row in e2["rows"]:
        referee[row["scenario"]] = row["referee_gap_to_exact_pct"]
        for name, cell in row["samplers"].items():
            if isinstance(cell, dict) and "refined_gap_pct_runs" in cell:
                out["default_budget"][f"{row['scenario']}|{name}"] = against_referee(cell, referee[row["scenario"]])
    scenario = e2["budget_scenario"]
    for key, cell in e2["budget"].items():
        out["budget_arm"][f"{scenario}|{key}"] = against_referee(cell, referee[scenario])

    def total(cells):
        return {k: int(sum(c[k] for c in cells)) for k in ("runs", "better_than_referee", "equal_to_referee", "worse_than_referee")}

    default_cells = [v for k, v in out["default_budget"].items() if k.startswith(scenario)]
    out["budget_scenario"] = scenario
    out["budget_scenario_default_total"] = total(default_cells)
    out["budget_scenario_all_configurations_total"] = total(default_cells + list(out["budget_arm"].values()))
    out["referee_gap_pct"] = referee
    out["elapsed_s_range_default_budget"] = [min(c["mean_elapsed_s"] for c in out["default_budget"].values()),
                                             max(c["mean_elapsed_s"] for c in out["default_budget"].values())]
    return out


def chemistry() -> dict:
    e12 = load("e12_sqd_selected_ci")
    settings, every_ratio, full_better = {}, [], 0
    for s in e12["summary"]:
        rows = [r for r in e12["rows"] if r["bond"] == s["bond"] and r["cpt"] == s["cpt"]]
        pipeline = np.array([r["platform_err"] for r in rows], float)
        random_ = np.array([r["random_err"] for r in rows], float)
        sci = np.array([r["sci_err_at_dim"] for r in rows], float)
        full = np.array([r["sq_full_err"] for r in rows], float)
        every_ratio += list(pipeline / sci)
        full_better += int((full < pipeline).sum())
        label = f"{'eq' if s['bond'] is None else s['bond']}|{s['cpt']}"
        settings[label] = {
            "runs": len(rows),
            "random_over_pipeline": {"ratio_of_means": float(random_.mean() / pipeline.mean()),
                                     "per_run_min": float((random_ / pipeline).min()),
                                     "per_run_max": float((random_ / pipeline).max())},
            "pipeline_over_selected_ci": {"mean_of_per_run_ratios": float((pipeline / sci).mean()),
                                          "per_run_median": float(np.median(pipeline / sci)),
                                          "per_run_min": float((pipeline / sci).min()),
                                          "per_run_max": float((pipeline / sci).max()),
                                          "ratio_of_means": float(pipeline.mean() / sci.mean()),
                                          "runs_pipeline_worse": int((pipeline > sci).sum())},
            "pipeline_over_all_strings": {"ratio_of_means": float(pipeline.mean() / full.mean()),
                                          "per_run_min": float((pipeline / full).min()),
                                          "per_run_max": float((pipeline / full).max()),
                                          "runs_all_strings_better": int((full < pipeline).sum())},
        }
    first_accurate = {}
    for bond in sorted({str(c["bond"]) for c in e12["sci_curve"]}):
        curve = sorted((c["dim"], c["err"]) for c in e12["sci_curve"] if str(c["bond"]) == bond)
        index = next(i for i, (_, err) in enumerate(curve) if err <= CHEMICAL_ACCURACY_MHA)
        first_accurate["eq" if bond == "None" else bond] = {
            "dimension": int(curve[index][0]), "coverage": curve[index][0] / N2_FCI_DIMENSION, "error_mha": float(curve[index][1]),
            "previous_point": {"dimension": int(curve[index - 1][0]), "coverage": curve[index - 1][0] / N2_FCI_DIMENSION,
                               "error_mha": float(curve[index - 1][1])} if index else None,
            "thresholds_on_curve": len(curve),
        }
    key = lambda name: [v[name]["ratio_of_means"] for v in settings.values()]
    return {
        "settings": settings,
        "all_runs": {"runs": len(every_ratio), "pipeline_over_selected_ci_min": float(min(every_ratio)),
                     "pipeline_over_selected_ci_max": float(max(every_ratio)),
                     "runs_pipeline_worse_than_selected_ci": int(sum(r > 1 for r in every_ratio)),
                     "runs_all_strings_better_than_pipeline": full_better},
        "random_over_pipeline_ratio_of_means_range": [min(key("random_over_pipeline")), max(key("random_over_pipeline"))],
        "pipeline_over_all_strings_ratio_of_means_range": [min(key("pipeline_over_all_strings")), max(key("pipeline_over_all_strings"))],
        "pipeline_over_selected_ci_mean_ratio_range": [min(v["pipeline_over_selected_ci"]["mean_of_per_run_ratios"] for v in settings.values()),
                                                       max(v["pipeline_over_selected_ci"]["mean_of_per_run_ratios"] for v in settings.values())],
        "selected_ci_first_point_within_chemical_accuracy": first_accurate,
    }


def qaoa() -> dict:
    e10 = load("e10_qaoa_exact")
    uniform = e10["uniform"]["p_feasible"]
    exact = {e["p"]: e for e in e10["exact_opt"]}
    arms = {}
    for name, cell in e10["summary"].items():
        arm, depth = name.rsplit("_p", 1)
        arms[name] = {"feasible_enrichment_over_uniform": cell["p_feasible"]["mean"] / uniform,
                      "fraction_of_exact_objective_ratio": cell["approx_ratio"]["mean"] / exact[int(depth)]["approx_ratio"],
                      "runs_better_than_uniform": cell["runs_better_than_uniform"], "runs": cell["approx_ratio"]["n"]}
    span = lambda arm, field: [min(v[field] for k, v in arms.items() if k.startswith(arm)),
                               max(v[field] for k, v in arms.items() if k.startswith(arm))]
    return {
        "uniform_p_feasible": uniform, "arms": arms,
        "exact_objective_feasible_enrichment": {str(p): e["p_feasible"] / uniform for p, e in exact.items()},
        "exact_objective_starts_converged": {str(p): e["starts_converged"] for p, e in exact.items()},
        "feasible_enrichment_range": {arm: span(arm, "feasible_enrichment_over_uniform") for arm in ("raw", "norm", "platform")},
        "fraction_of_exact_objective_ratio_range": {arm: span(arm, "fraction_of_exact_objective_ratio") for arm in ("raw", "norm", "platform")},
        "raw_runs_better_than_uniform": int(sum(v["runs_better_than_uniform"] for k, v in arms.items() if k.startswith("raw"))),
    }


def amplitude_estimation() -> dict:
    e1 = load("e1_montecarlo")
    measured = {r["grover_power"]: r["cx"] for r in e1["resources"]}
    powers = sorted(measured)
    slope = (measured[powers[-1]] - measured[powers[0]]) / (powers[-1] - powers[0])
    law = {"intercept": measured[0], "per_grover_power": slope,
           "exact_for_all_measured_powers": all(measured[k] == measured[0] + slope * k for k in powers),
           "measured_powers": powers}
    rows = []
    for r in e1["mlae"]:
        k = r["max_power"]
        cx = measured.get(k, measured[0] + slope * k)
        rows.append({"max_power": k, "cx_deepest_circuit": int(cx), "cx_measured": k in measured,
                     "p_no_two_qubit_error_at_1e-3": float((1 - 1e-3) ** cx),
                     "efficiency_interval_contains_1": bool(r["efficiency_ratio_ci"][0] <= 1 <= r["efficiency_ratio_ci"][1])})
    return {"cx_law": law, "schedules": rows,
            "schedules_whose_efficiency_interval_contains_1": int(sum(r["efficiency_interval_contains_1"] for r in rows)),
            "efficiency_ratio_range": [min(r["efficiency_ratio"] for r in e1["mlae"]), max(r["efficiency_ratio"] for r in e1["mlae"])]}


def routing() -> dict:
    e5 = load("e5_fleet")
    rows = e5["rows"]
    shorter = [r for r in rows if r["distance_delta_pct"] < 0]
    wins = [r for r in rows if r["wins"]]
    return {
        "runs": len(rows),
        "runs_shorter_than_referee": len(shorter),
        "runs_shorter_than_referee_that_left_more_stops_unserved": int(sum(r["unassigned_delta"] > 0 for r in shorter)),
        "wins": len(wins), "win_instances": sorted({(r["preset"], r["instance_seed"]) for r in wins}),
        "win_extra_stops_served": sorted(-r["unassigned_delta"] for r in wins),
        "win_distance_delta_pct": sorted(r["distance_delta_pct"] for r in wins),
        "within_instance_sd_range": [min(s["within_instance_sd"] for s in e5["summary"].values()),
                                     max(s["within_instance_sd"] for s in e5["summary"].values())],
    }


def screened_qubits() -> dict:
    e15 = load("e15_mlae_screened_qubits")
    limit10, limit01 = e15["thresholds"]["max_p_read1_given0"], e15["thresholds"]["max_p_read0_given1"]
    selected, above = 0, []
    for entry in e15["sets"]:
        readout = entry["readout"]
        for qubit, p10, p01 in zip(readout["physical_qubits"], readout["p_read1_given0"], readout["p_read0_given1"]):
            selected += 1
            if p10 > limit10 or p01 > limit01:
                above.append({"qubit": qubit, "p_read1_given0": p10, "p_read0_given1": p01})
    return {"selected_qubits": selected, "above_screen_threshold_in_probe_job_calibration": above,
            "screen_submitted_utc": e15["screen"]["run"]["submitted_utc"], "probe_submitted_utc": e15["run"]["submitted_utc"]}


def portfolio() -> dict:
    e19 = load("e19_portfolio")
    summary, sizes = e19["summary"], e19["sizes"]

    def cells(arm):
        return [summary[f"{arm}|{n}"] for n in sizes]

    def total(arm, side, field):
        return int(sum(c[side][field] for c in cells(arm)))

    def means(arm, side, field):
        return [c[side][field]["mean"] for c in cells(arm)]

    out = {"sizes": sizes, "instances_per_size": e19["instances"], "seeds_per_instance": e19["seeds"]}
    tests = {f"{arm}|{n}": summary[f"{arm}|{n}"]["method_vs_null"]["wilcoxon_p"]
             for arm in ("qaoa", "cvar_all", "cvar_free", "bfdcqo_shipped", "bfdcqo_fixed", "pce") for n in sizes}
    threshold = 0.05 / len(tests)
    out["primary_tests"] = {
        "count": len(tests), "bonferroni_threshold": threshold, "p_values": tests,
        "untestable_all_differences_zero": sorted(k for k, p in tests.items() if p is None),
        "below_0.05_unadjusted": sorted(k for k, p in tests.items() if p is not None and p < 0.05),
        "below_bonferroni": sorted(k for k, p in tests.items() if p is not None and p < threshold),
        "smallest_possible_p_with_10_pairs": 2.0 / 2 ** 10,
    }
    q = cells("qaoa")
    out["qaoa"] = {
        "runs": total("qaoa", "method", "runs"),
        "optimal_runs": total("qaoa", "method", "optimal_runs"), "null_optimal_runs": total("qaoa", "null", "optimal_runs"),
        "gap": means("qaoa", "method", "gap"), "null_gap": means("qaoa", "null", "gap"),
        "rank_pct": means("qaoa", "method", "rank_pct"), "null_rank_pct": means("qaoa", "null", "rank_pct"),
        "instances_method_better": [c["method_vs_null"]["method_better"] for c in q],
        "returned_is_most_frequent_sample_runs": total("qaoa", "method", "returned_is_most_frequent_sample_runs"),
        "null_returned_is_most_frequent_sample_runs": total("qaoa", "null", "returned_is_most_frequent_sample_runs"),
        "final_state_approx_ratio": means("qaoa", "method", "state_approx_ratio"),
        "initial_state_approx_ratio": means("qaoa", "method", "state_initial_approx_ratio"),
        "runs_where_training_raised_approx_ratio": total("qaoa", "method", "runs_where_training_raised_approx_ratio"),
        "p_optimum_over_uniform_median": [c["method"]["state_p_optimum_over_uniform_median"] for c in q],
        "lowest_cost_rule_optimal_rate": means("qaoa", "method", "lowest_cost_rule_optimal"),
        "null_lowest_cost_rule_optimal_rate": means("qaoa", "null", "lowest_cost_rule_optimal"),
        "most_frequent_rule_optimal_rate": means("qaoa", "method", "optimal"),
        "lowest_cost_rule_p": [c["lowest_cost_rule_method_vs_null"]["wilcoxon_p"] for c in q],
        "distinct_final_samples": means("qaoa", "method", "distinct_final_samples"),
        "largest_sample_count": means("qaoa", "method", "largest_sample_count"),
    }
    out["cvar_vqe"] = {}
    for arm in ("cvar_all", "cvar_free"):
        c = cells(arm)
        evaluations = [x["method"]["objective_evaluations"] for x in c]
        parameters = [x["method"]["parameters"] for x in c]
        out["cvar_vqe"][arm] = {
            "runs": total(arm, "method", "runs"),
            "optimal_runs": total(arm, "method", "optimal_runs"), "null_optimal_runs": total(arm, "null", "optimal_runs"),
            "gap": means(arm, "method", "gap"), "null_gap": means(arm, "null", "gap"),
            "p_values": [x["method_vs_null"]["wilcoxon_p"] for x in c],
            "instances_method_better": [x["method_vs_null"]["method_better"] for x in c],
            "final_state_approx_ratio": means(arm, "method", "state_approx_ratio"),
            "initial_state_approx_ratio": means(arm, "method", "state_initial_approx_ratio"),
            "objective_evaluations": evaluations, "parameters": parameters,
            # COBYLA spends parameters + 1 evaluations on its first simplex
            "evaluations_after_first_simplex": [e[0] - (p[0] + 1) for e, p in zip(evaluations, parameters)],
        }
    out["bfdcqo"] = {}
    for arm in ("bfdcqo_shipped", "bfdcqo_fixed"):
        c = cells(arm)
        before = [x[side]["sampler_found_optimum_before_refinement"] for x in c for side in ("method", "null")]
        out["bfdcqo"][arm] = {
            "runs": total(arm, "method", "runs"), "null_runs": total(arm, "null", "runs"),
            "first_k_assets_runs": total(arm, "method", "first_k_assets_runs"),
            "null_first_k_assets_runs": total(arm, "null", "first_k_assets_runs"),
            "optimal_runs": total(arm, "method", "optimal_runs"), "null_optimal_runs": total(arm, "null", "optimal_runs"),
            "instances_identical_to_null": int(sum(x["method_vs_null"]["ties"] for x in c)),
            "instances": int(sum(x["method_vs_null"]["instances"] for x in c)),
            "rank_pct": means(arm, "method", "rank_pct"),
            "single_cluster_runs_checked_before_refinement": int(sum(b["runs"] for b in before)),
            "single_cluster_runs_optimal_before_refinement": int(sum(b["optimal"] for b in before)),
        }
    p = cells("pce")
    out["pce"] = {
        "excess": means("pce", "method", "excess"), "null_excess": means("pce", "null", "excess"),
        "equal_weights_excess": means("pce", "method", "equal_weights_excess"),
        "excess_over_equal_weights_excess": [a / b for a, b in zip(means("pce", "method", "excess"),
                                                                  means("pce", "method", "equal_weights_excess"))],
        "p_values": [x["method_vs_null"]["wilcoxon_p"] for x in p],
        "convex_runs": total("pce", "method", "convex_runs"), "runs": total("pce", "method", "runs"),
        "qubits": [x["method"]["qubits"] for x in p],
    }
    walls = {arm: [c["method"]["mean_wall_s"] for c in cells(arm)] for arm in ("qaoa", "cvar_all", "cvar_free", "bfdcqo_fixed", "pce")}
    out["timing"] = {
        "method_mean_wall_s_range": [min(min(v) for v in walls.values()), max(max(v) for v in walls.values())],
        "exact_enumeration_seconds": {n: v["seconds"] for n, v in e19["exact_enumeration"].items()},
    }
    engine = e19["audit"]["refinement_engine"]
    out["refinement_engine"] = {**engine, "relabelling": e19["relabelling"]}
    return out


def main() -> None:
    payload = {"facility_location": facility(), "chemistry": chemistry(), "qaoa": qaoa(),
               "amplitude_estimation": amplitude_estimation(), "routing": routing(), "screened_qubits": screened_qubits(),
               "portfolio": portfolio()}
    save("e17_derived_statistics", payload)
    f = payload["facility_location"]
    for name, cell in {**f["default_budget"], **f["budget_arm"]}.items():
        print(f"{name:34s} better/equal/worse {cell['better_than_referee']}/{cell['equal_to_referee']}/{cell['worse_than_referee']}"
              f"  sign p={cell['sign_test_p']}  t p={cell['t_test_p']}  values {cell['distinct_refined_gaps_pct']}"
              f"  raw {cell['mean_raw_cost_vs_optimum_pct']:+.1f}%  {cell['mean_elapsed_s']:.2f}s")
    print("metro default:", f["budget_scenario_default_total"], "all:", f["budget_scenario_all_configurations_total"])
    c = payload["chemistry"]
    for name, cell in c["settings"].items():
        print(name, {k: {a: round(b, 2) for a, b in v.items()} for k, v in cell.items() if isinstance(v, dict)})
    print({k: v for k, v in c.items() if k != "settings"})
    q = payload["qaoa"]
    print({k: v for k, v in q.items() if k != "arms"})
    a = payload["amplitude_estimation"]
    print(a["cx_law"], [(r["max_power"], r["cx_deepest_circuit"], f"{r['p_no_two_qubit_error_at_1e-3']:.3g}") for r in a["schedules"]],
          a["schedules_whose_efficiency_interval_contains_1"], a["efficiency_ratio_range"])
    print(payload["routing"])
    print(payload["screened_qubits"])
    for key, value in payload["portfolio"].items():
        print("portfolio", key, value)


if __name__ == "__main__":
    main()
