"""Table bodies of the paper's section on the corrected portfolio methods, generated from the stored results.

The rows are printed for pasting into main.tex. With ``--check`` the script
verifies instead that every generated row is in main.tex character for
character, so a number retyped wrongly, or a result file that changed after
the paper was written, is caught.

Sources: results/e20_portfolio_corrected.json, results/e21_qaoa_fresh.json,
results/e22_hardware_portfolio_qaoa.json, results/e24_displayed_allocation.json
and results/e25_qaoa_initial_angles.json.
Offline; runs nothing.
"""

from __future__ import annotations

import json
import sys

from common import RESULTS

MAIN_TEX = RESULTS.parent.parent / "main.tex"
SIZES = (8, 12, 16)
PRIMARY = [("qaoa", "QAOA"), ("cvar_free", "CVaR-VQE, no $K$"), ("cvar_k", "CVaR-VQE, $K$"), ("bfdcqo", "BF-DCQO"),
           ("bfdcqo_lowshot", "BF-DCQO, 32 shots"), ("pce_free", "PCE, no $K$"), ("pce_k", "PCE, $K$")]


def load(name: str):
    path = RESULTS / f"{name}.json"
    return json.loads(path.read_text()) if path.exists() else None


def pm(c: dict, digits: int = 2) -> str:
    """Mean and half-width of the 95% interval; a plain 0 when both round to zero."""
    if abs(c["mean"]) < 0.5 * 10 ** -digits and abs(c["ci95"]) < 0.5 * 10 ** -digits:
        return "0"
    return f"${c['mean']:.{digits}f} \\pm {c['ci95']:.{digits}f}$"


def pval(test: dict) -> str:
    """Enough digits to compare with the thresholds: four below 0.01, three below 0.1, two above."""
    p = test["wilcoxon_p"]
    if p is None:
        return "tie"
    return f"{p:.4f}" if p < 0.01 else f"{p:.3f}" if p < 0.1 else f"{p:.2f}"


def bwt(test: dict) -> str:
    return f"{test['method_better']}/{test['null_better']}/{test['ties']}"


def primary_rows(e20: dict, e21: dict) -> list:
    S = e20["summary"]
    rows = []
    for arm, label in PRIMARY:
        cells = [S[f"{arm}|{n}"] for n in SIZES]
        if all(c["method"]["optimal_runs"] == c["method"]["runs"] == c["null"]["optimal_runs"] for c in cells):
            total = sum(c["method"]["runs"] for c in cells)
            rows.append(f"{label} & 8--16 & {total}/{total} & {total}/{total} & 0 & 0 & tie \\\\")
            continue
        for index, (n, c) in enumerate(zip(SIZES, cells)):
            m, u = c["method"], c["null"]
            field = "admissible_rank_pct" if "admissible_rank_pct" in m else "rank_pct"
            rows.append(f"{label if index == 0 else ''} & {n} & {m['optimal_runs']}/{m['runs']} & {u['optimal_runs']}/{u['runs']}"
                        f" & {pm(m[field])} & {pm(u[field])} & {pval(c['method_vs_null'])} \\\\")
    for index, n in enumerate(SIZES):
        c = e21["summary"][str(n)]
        m, u = c["method"], c["null"]
        rows.append(f"{'QAOA, new instances' if index == 0 else ''} & {n} & {m['optimal_runs']}/{m['runs']} & {u['optimal_runs']}/{u['runs']}"
                    f" & {pm(m['rank_pct'])} & {pm(u['rank_pct'])} & {pval(c['method_vs_null'])} \\\\")
    return rows


def secondary_rows(e20: dict) -> list:
    S = e20["summary"]
    rows = []
    for arm, label in (("pce_free", "PCE decoded, no $K$"), ("pce_k", "PCE decoded, $K$")):
        for index, n in enumerate(SIZES):
            c = S[f"{arm}|{n}"]
            m, u, t = c["method"], c["null"], c["decoded_method_vs_null"]
            rows.append(f"{label if index == 0 else ''} & {n} & {m['decoded_optimal_runs']}/{m['runs']} & {u['decoded_optimal_runs']}/{u['runs']}"
                        f" & {pm(t['method_minus_null'], 2)} & {bwt(t)} & {pval(t)} \\\\")
    for depth in (1, 2, 3):
        for index, n in enumerate(SIZES):
            c = S[f"bfdcqo_sampler_d{depth}|{n}"]
            m, u, t = c["method"], c["null"], c["sampler_vs_null"]
            found = "sampler_found_optimum_before_any_classical_step"
            rows.append(f"{f'BF-DCQO shots, depth {depth}' if index == 0 else ''} & {n} & {m[found]['optimal']}/{m[found]['runs']}"
                        f" & {u[found]['optimal']}/{u[found]['runs']} & {pm(t['method_minus_null'], 4)} & {bwt(t)} & {pval(t)} \\\\")
    control = e20["sampler_arms_against_depth_1_control"]
    for depth in (2, 3):
        for index, n in enumerate(SIZES):
            t = control[f"d{depth}_vs_d1|{n}"]
            rows.append(f"{f'depth {depth} against depth 1' if index == 0 else ''} & {n} & & & {pm(t['method_minus_null'], 4)}"
                        f" & {bwt(t)} & {pval(t)} \\\\")
    return rows


def hardware_rows(e22: dict) -> list:
    rows = []
    for n in SIZES:
        c = e22["summary"][str(n)]
        b, p, r, k = c["batch"], c["p_optimum"], c["approx_ratio"], c["circuits"]
        sim = c["simulator_e21"]
        gates = k["two_qubit_gates_range"]
        rows.append(
            f"{n} & {gates[0] if gates[0] == gates[1] else f'{gates[0]}--{gates[1]}'} & {k['p_no_two_qubit_error']['mean']:.2f}"
            f" & {100 * b['qaoa']['optimal_share']:.1f} & {100 * b['null']['optimal_share']:.1f} & {100 * b['uniform_exact_optimal_share']:.1f}"
            f" & {100 * sim['qaoa_optimal_runs'] / sim['runs']:.0f}"
            f" & {pm(b['qaoa']['gap'], 3)} & {pm(b['null']['gap'], 3)} & {pval(b['qaoa_vs_null'])}"
            f" & {r['qaoa']['mean']:.2f} & {r['ideal']['mean']:.2f} & {p['retained_fraction']['estimate']:.3f}"
            f" [{p['retained_fraction']['ci95'][0]:.3f}, {p['retained_fraction']['ci95'][1]:.3f}] \\\\")
    return rows


def displayed_rows(e24: dict) -> list:
    """E24: the displayed allocation against the one the classical steps produce alone."""
    def signs(test: dict) -> str:
        return f"{test['first_lower']}/{test['first_higher']}/{test['ties']}"

    rows = []
    for n in SIZES:
        c = e24["summary"][str(n)]
        m, u, o, uni = c["method"], c["null"], c["optimum"], c["uniform"]
        t1, t2 = c["J_tests"]["T1_method_vs_null"], c["J_tests"]["T2_method_vs_all"]
        rows.append(
            f"{n} & {m['same_allocation_as_all']}/{m['runs']} & {u['same_allocation_as_all']}/{u['runs']}"
            f" & {o['same_allocation_as_all']}/{c['instances']} & {100 * uni['share_same_allocation_as_all']['median']:.2f}"
            f" & {100 * uni['share_J_above_all']['median']:.2f}"
            f" & {m['selected_mean']:.1f} & {m['held_above_1pct_mean']:.1f}"
            f" & {signs(t1)} & {pval(t1)} & {signs(t2)} & {pval(t2)} \\\\")
    return rows


def initial_angle_rows(e25: dict) -> list:
    """E25: the start the comment of the code describes against the start the code uses."""
    def signs(test: dict) -> str:
        return f"{test['intended_better']}/{test['shipped_better']}/{test['instances'] - test['untied']}"

    rows = []
    for family, label in (("e21", "E21"), ("e23", "E23")):
        for index, n in enumerate(SIZES):
            c = e25["summary"][f"{family}|{n}"]
            i, s = c["intended"], c["shipped"]
            g, a = c["gap_intended_vs_shipped"], c["approx_ratio_intended_vs_shipped"]
            rows.append(
                f"{label if index == 0 else ''} & {n} & {i['optimal_runs']}/{i['runs']} & {s['optimal_runs']}/{s['runs']}"
                f" & {signs(g)} & {pval(g)} & {i['approx_ratio']['mean']:.2f} & {s['approx_ratio']['mean']:.2f}"
                f" & {signs(a)} & {pval(a)} & ${i['state_initial_approx_ratio']['mean']:.2f}$"
                f" & {s['state_initial_approx_ratio']['mean']:.2f} \\\\")
    return rows


def main() -> None:
    e20, e21, e22 = load("e20_portfolio_corrected"), load("e21_qaoa_fresh"), load("e22_hardware_portfolio_qaoa")
    e24 = load("e24_displayed_allocation")
    e25 = load("e25_qaoa_initial_angles")
    tables = {"primary (returned selection)": primary_rows(e20, e21), "secondary (before the classical search)": secondary_rows(e20)}
    if e22 is not None:
        tables["hardware"] = hardware_rows(e22)
    if e24 is not None:
        tables["displayed allocation"] = displayed_rows(e24)
    if e25 is not None:
        tables["initial angles"] = initial_angle_rows(e25)
    if "--check" in sys.argv:
        text = MAIN_TEX.read_text()
        missing = [(name, row) for name, rows in tables.items() for row in rows if row not in text]
        for name, row in missing:
            print(f"NOT IN main.tex [{name}]: {row}")
        total = sum(len(rows) for rows in tables.values())
        print(f"{total - len(missing)} of {total} generated rows are in main.tex")
        raise SystemExit(1 if missing else 0)
    for name, rows in tables.items():
        print(f"% --- {name}")
        print("\n".join(rows))


if __name__ == "__main__":
    main()
