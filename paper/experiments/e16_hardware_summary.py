"""E16 - Descriptive summary of the hardware depth probes (E14-B, E15) and the hardware QAOA (E14-A).

Post hoc: these quantities were defined after the data were seen and are
reported as descriptions, not as tests.

 signal fraction w   as measured in E14-B/E15: the maximum-likelihood weight of
                     the ideal output distribution in a mixture with uniform.
 effective error     slope of ln w against the transpiled two-qubit gate count
                     N within one qubit set, ln w = ln w_SPAM - eps_eff N
                     (weighted least squares; points whose 95% interval reaches
                     below 0.02 are left out). eps_eff lumps every error source
                     into a rate per two-qubit gate.
 information bound   Under the depolarising contrast model
                       P_k = 1/2 + F_k (sin^2((2k+1) theta) - 1/2)
                     one shot at Grover power k carries Fisher information
                       I_k = 4 (2k+1)^2 F_k^2 sin^2(x) / (1 - F_k^2 cos^2(x)),
                       x = 2 (2k+1) theta,
                     about theta, so I_k <= 4 (2k+1)^2 F_k^2 for every theta.
                     An ideal unamplified shot (a noiseless Bernoulli draw of
                     the amplitude a = sin^2 theta, k = 0, F = 1) carries
                     exactly 4 and costs one oracle query; a shot at power k
                     costs 2k+1. Hence the information per query, in units of
                     an ideal unamplified shot, is at most
                       Gamma_k = (2k + 1) F_k^2.
                     Noiseless: Gamma_k = 2k + 1. Gamma_k <= 1 for every k
                     means that on these qubits no Grover power can reduce the
                     number of queries below unamplified noiseless sampling,
                     whatever the amplitude. Gamma_k > 1 is necessary, not
                     sufficient, for a gain. We take F_k = w_k, which is exact
                     for global depolarising noise.
 relative sensitivity  G_k = (2k + 1) w_k / w_0: the same envelope relative to
                     the device's own k = 0 shot, per shot and in standard
                     error (kept for the record; G_k^2 / (2k+1) is the
                     per-query information ratio to the device's own k = 0).
"""

from __future__ import annotations

import json
import math

import numpy as np
from common import FIGURES, RESULTS, mpl, save


def load_sets():
    sets = []
    e14 = json.loads((RESULTS / "e14_ibm_hardware_controls_mlae.json").read_text())
    for reg in e14["mlae"]["registers"]:
        sets.append({"label": f"n={reg['n_price_qubits']} unscreened", "n": reg["n_price_qubits"], "screened": False,
                     "qubits": reg["initial_layout"], "rows": reg["rows"], "fit": reg.get("contrast_fit"),
                     "amplitudes": reg.get("amplitude_estimates"), "job": e14["mlae"]["run"]["job_id"]})
    e15 = json.loads((RESULTS / "e15_mlae_screened_qubits.json").read_text())
    for entry in e15["sets"]:
        sets.append({"label": f"n={entry['n_price_qubits']} screened set {entry['set']}", "n": entry["n_price_qubits"],
                     "screened": True, "qubits": entry["initial_layout"], "rows": entry["rows"],
                     "fit": entry.get("contrast_fit"), "amplitudes": entry.get("amplitude_estimates"),
                     "job": e15["run"]["job_id"]})
    return sets, e14, e15


def log_sd(row) -> float:
    lo, hi = row["signal_fraction_ci95"]
    return (hi - lo) / 3.92 / max(row["signal_fraction"], 1e-9)


def effective_error(rows):
    use = [r for r in rows if r["signal_fraction_ci95"][0] > 0.02]
    if len(use) < 3:
        return None
    x = np.array([r["two_qubit_gates"] for r in use], float)
    y = np.log([r["signal_fraction"] for r in use])
    w = 1.0 / np.array([log_sd(r) for r in use]) ** 2
    design = np.vstack([np.ones_like(x), -x]).T
    cov = np.linalg.inv(design.T @ (design * w[:, None]))
    beta = cov @ design.T @ (w * y)
    resid = y - design @ beta
    scale = float(resid @ (w * resid) / max(len(x) - 2, 1))  # reduced chi-square
    se = np.sqrt(np.diag(cov) * max(scale, 1.0))  # inflated when the line does not fit within the errors
    return {"eps_eff": float(beta[1]), "eps_eff_se": float(se[1]), "w_spam": float(math.exp(beta[0])),
            "points": len(use), "reduced_chi2": scale}


def main() -> None:
    sets, e14, e15 = load_sets()
    out_sets = []
    for s in sets:
        rows = s["rows"]
        w0 = rows[0]["signal_fraction"]
        lo0, hi0 = rows[0]["signal_fraction_ci95"]
        gains = []
        for r in rows:
            scale = 2 * r["k"] + 1
            g = scale * r["signal_fraction"] / w0
            lo, hi = r["signal_fraction_ci95"]
            # conservative: both intervals at their unfavourable ends
            interval = [g, g] if r["k"] == rows[0]["k"] else [scale * lo / hi0, scale * hi / lo0]
            gains.append({"k": r["k"], "two_qubit_gates": r["two_qubit_gates"], "signal_fraction": r["signal_fraction"],
                          "signal_fraction_ci95": r["signal_fraction_ci95"],
                          "query_information_bound": scale * r["signal_fraction"] ** 2,
                          "query_information_bound_ci95": [scale * lo ** 2, scale * hi ** 2],
                          "noiseless_query_information": scale,
                          "relative_sensitivity": g, "relative_sensitivity_ci95": interval,
                          "p_no_two_qubit_error_reported": r["calibration"]["p_no_two_qubit_error"]})
        best = max(gains, key=lambda g: g["query_information_bound"])
        best_amplified = max((g for g in gains if g["k"] > 0), key=lambda g: g["query_information_bound"])
        entry = {"label": s["label"], "n_price_qubits": s["n"], "screened": s["screened"], "qubits": s["qubits"],
                 "job_id": s["job"], "rows": gains,
                 "max_query_information_bound": best["query_information_bound"],
                 "max_query_information_bound_ci95": best["query_information_bound_ci95"],
                 "max_query_information_bound_at_k": best["k"],
                 "best_amplified": {key: best_amplified[key] for key in
                                    ("k", "query_information_bound", "query_information_bound_ci95",
                                     "noiseless_query_information")},
                 "amplification_can_pay": bool(best_amplified["query_information_bound_ci95"][0] > 1.0),
                 "effective_error": effective_error(rows)}
        if s["fit"]:
            entry["f_A"], entry["f_Q"], entry["f_Q_ci95"] = s["fit"]["f_A"], s["fit"]["f_Q"], s["fit"]["f_Q_ci95"]
            entry["contrast_model_fit_p"] = s["fit"]["goodness_of_fit_p"]
            entry["free_floor"] = s["fit"]["free_floor"]["floor"]
            a = s["amplitudes"]
            entry["amplitude"] = {"true": a["true"], "noiseless_likelihood": a["noiseless_likelihood_mle"],
                                  "noise_aware": a["noise_aware_mle"], "noise_aware_sd": a["noise_aware_bootstrap_sd"],
                                  "noiseless_relative_error": a["noiseless_likelihood_mle"] / a["true"] - 1,
                                  "noise_aware_relative_error": a["noise_aware_mle"] / a["true"] - 1}
        out_sets.append(entry)
        eff = entry["effective_error"]
        print(f"{s['label']:24s} qubits {s['qubits']}: max Gamma {best['query_information_bound']:.3f} at k={best['k']}"
              f" (best k>0: {best_amplified['query_information_bound']:.3f}"
              f" [{best_amplified['query_information_bound_ci95'][0]:.3f}, {best_amplified['query_information_bound_ci95'][1]:.3f}]"
              f" at k={best_amplified['k']}, noiseless {best_amplified['noiseless_query_information']})"
              + (f"; eps_eff {eff['eps_eff'] * 100:.3f}±{eff['eps_eff_se'] * 100:.3f}% w_SPAM {eff['w_spam']:.3f}" if eff else "")
              + (f"; f_Q {entry['f_Q']:.3f}" if s["fit"] else ""))

    screened_eps = [e["effective_error"]["eps_eff"] for e in out_sets if e["screened"] and e["effective_error"]]
    qaoa = json.loads((RESULTS / "e14_ibm_hardware_controls_qaoa.json").read_text())["qaoa"]
    arms, tests = qaoa["arms"], qaoa["tests"]
    null_r = (arms["null_beta"]["approx_ratio"] + arms["null_gamma"]["approx_ratio"]) / 2
    null_shots = arms["null_beta"]["shots"] + arms["null_gamma"]["shots"]
    null_top = (arms["null_beta"]["top"]["count"] + arms["null_gamma"]["top"]["count"]) / null_shots
    readout = qaoa["calibration"]["measured"]
    all_bits_right = float(np.prod([1 - (a + b) / 2 for a, b in zip(readout["p_read1_given0"], readout["p_read0_given1"])]))
    qaoa_summary = {
        "job_id": qaoa["run"]["job_id"], "two_qubit_gates": qaoa["two_qubit_gates"],
        "p_no_two_qubit_error_reported": qaoa["calibration"]["p_no_two_qubit_error"],
        "p_all_bits_read_correctly": all_bits_right,
        "approx_ratio": arms["qaoa"]["approx_ratio"], "approx_ratio_null_mean": null_r,
        "approx_ratio_ideal": arms["qaoa"]["ideal_approx_ratio"],
        "approx_ratio_retained": (arms["qaoa"]["approx_ratio"] - null_r) / arms["qaoa"]["ideal_approx_ratio"],
        "feasible_rate": arms["qaoa"]["feasible"]["rate"], "feasible_rate_null": tests["pooled_null_feasible"]["rate"],
        "feasible_ratio_to_null": arms["qaoa"]["feasible"]["rate"] / tests["pooled_null_feasible"]["rate"],
        "feasible_retained": tests["retained_fraction_of_feasible_excess"]["estimate"],
        "top_rate": arms["qaoa"]["top"]["rate"], "top_rate_null": null_top,
        "top_retained": (arms["qaoa"]["top"]["rate"] - null_top) / (arms["qaoa"]["top"]["ideal_rate"] - arms["null_beta"]["top"]["ideal_rate"]),
    }
    print("QAOA:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in qaoa_summary.items()})

    screen = e15["screen"]
    p01, p10 = np.array(screen["p_read0_given1"]), np.array(screen["p_read1_given0"])
    reported = np.array([np.nan if v is None else v for v in screen["reported_readout_error"]], float)
    measured = (p01 + p10) / 2
    ok = ~np.isnan(reported)
    screen_summary = {
        "job_id": screen["run"]["job_id"], "qubits": int(len(p01)), "rejected": len(screen["rejected_qubits"]),
        "calibration_last_update": screen["calibration_last_update"],
        "p_read0_given1_quantiles_10_25_50_75_90": [float(v) for v in np.percentile(p01, [10, 25, 50, 75, 90])],
        "p_read1_given0_quantiles_10_25_50_75_90": [float(v) for v in np.percentile(p10, [10, 25, 50, 75, 90])],
        "median_measured": float(np.median(measured)), "median_reported": float(np.nanmedian(reported)),
        "pearson_reported_vs_measured": float(np.corrcoef(reported[ok], measured[ok])[0, 1]),
        "qubits_measured_over_3x_reported": int((measured[ok] > 3 * reported[ok]).sum()),
        "qubits_measured_over_2x_reported": int((measured[ok] > 2 * reported[ok]).sum()),
    }
    print("screen:", screen_summary)

    save("e16_hardware_summary", {
        "sets": out_sets, "qaoa": qaoa_summary, "screen": screen_summary,
        "screened_eps_eff_range": [min(screened_eps), max(screened_eps)] if screened_eps else None,
        "reported_median_two_qubit_error": e14["mlae"]["registers"][0]["rows"][0]["calibration"]["median_two_qubit_error"],
    })

    plt = mpl()
    fig, ax = plt.subplots(1, 2, figsize=(6.8, 2.6))
    colours = {2: "C0", 3: "C1", 5: "C2"}
    for e in out_sets:
        x = [r["two_qubit_gates"] for r in e["rows"]]
        y = [max(r["signal_fraction"], 2e-3) for r in e["rows"]]
        ax[0].plot(x, y, "o-" if e["screened"] else "x:", ms=3, lw=0.8, color=colours[e["n_price_qubits"]])
        ax[1].plot([r["k"] for r in e["rows"]], [max(r["query_information_bound"], 1e-3) for r in e["rows"]],
                   "o-" if e["screened"] else "x:", ms=3, lw=0.8, color=colours[e["n_price_qubits"]])
    grid = np.linspace(0, 800, 100)
    median = e14["mlae"]["registers"][0]["rows"][0]["calibration"]["median_two_qubit_error"]
    ax[0].plot(grid, (1 - median) ** grid, "k--", lw=0.8)
    ax[0].set_yscale("log")
    ax[0].set_xlabel("two-qubit gates after transpilation")
    ax[0].set_ylabel("signal fraction")
    ks = np.arange(0, 7)
    ax[1].plot(ks, 2 * ks + 1, "k--", lw=0.8)
    ax[1].axhline(1.0, color="grey", lw=0.6)
    ax[1].set_yscale("log")
    ax[1].set_xlabel("Grover power k")
    ax[1].set_ylabel("information per query, $(2k+1)\\,w_k^2$")
    handles = [plt.Line2D([], [], color=c, marker="o", ms=3, lw=0.8, label=f"{n + 1} qubits") for n, c in colours.items()]
    handles += [plt.Line2D([], [], color="grey", marker="x", ls=":", ms=3, lw=0.8, label="placed by reported calibration"),
                plt.Line2D([], [], color="k", ls="--", lw=0.8, label="gate-error model / noiseless")]
    ax[0].legend(handles=handles, fontsize=5.5)
    fig.tight_layout()
    fig.savefig(FIGURES / "hardware_depth.pdf")


if __name__ == "__main__":
    main()
