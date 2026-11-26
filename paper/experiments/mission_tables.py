"""Table bodies of the paper's section on mission resource allocation, generated from the stored result.

The rows are printed for pasting into main.tex, followed by the figures the text quotes. With
``--check`` the script verifies instead that every generated row is in main.tex character for
character, so a number retyped wrongly, or a result file that changed after the paper was
written, is caught.

Source: results/e31_mission_allocation.json and results/e32_mission_equal_time.json.
Offline; runs nothing.
"""

from __future__ import annotations

import json
import sys

from common import RESULTS

MAIN_TEX = RESULTS.parent.parent / "main.tex"
SIZES = ("4x12", "6x18", "8x24", "10x30", "12x36", "14x40")
LARGER = ("20x60", "40x120", "80x240")
ARMS = (("uniform", "Uniform"), ("qaoa_1", "QAOA, depth 1"), ("qaoa_2", "QAOA, depth 2"))


def pm(c: dict, digits: int) -> str:
    """Mean and half-width of the 95% interval; the mean alone when the half-width rounds to zero."""
    if abs(c["ci95"]) < 0.5 * 10 ** -digits:
        return f"${c['mean']:.{digits}f}$"
    return f"${c['mean']:.{digits}f} \\pm {c['ci95']:.{digits}f}$"


def part_a_rows(a: dict) -> list:
    rows = []
    for key in SIZES + ("all",):
        c = a["all"] if key == "all" else a["by_size"][key]
        label = f"All {c['instances']}" if key == "all" else "$" + key.replace("x", " \\times ") + "$"
        optimal = f"{c['S']['optimal_runs']} / {c['U']['optimal_runs']} / {c['U16_optimal_runs']}"
        rows.append(
            f"{label} & {c['pairs'][0]}--{c['pairs'][1]} & {pm(c['S']['best_filled'], 3)} & {pm(c['U']['best_filled'], 3)}"
            f" & {pm(c['U16_best_filled'], 3)} & {pm(c['greedy'], 2)} & {optimal}{' of ' + str(c['runs']) if key == 'all' else ''} \\\\")
    return rows


def part_b_rows(b: dict) -> list:
    rows = []
    for key, label in ARMS:
        c = b["arms"][key]
        ratio = "--" if key == "uniform" else pm(c["approx_ratio"], 3)
        rows.append(
            f"{label} & {pm(c['p_opt'], 3)} & {pm(c['gap1'], 2)} & {pm(c['p_raw'], 4)} & {pm(c['p_feasible'], 3)}"
            f" & {ratio} & {c['p_opt_64']['mean']:.3f} \\\\")
    c = b["arms"]["annealer"]
    rows.append(f"Annealer, 320 reads & {pm(c['p_opt'], 3)} & {pm(c['gap1'], 2)} & {pm(c['p_raw'], 4)} & -- & -- & -- \\\\")
    return rows


def seconds(value: float) -> str:
    """A time in seconds with the decimals its size calls for."""
    if value < 0.01:
        return f"{value:.4f}"
    if value < 1:
        return f"{value:.3f}"
    return f"{value:.1f}" if value < 100 else f"{value:.0f}"


def equal_time_rows(e: dict) -> list:
    rows = []
    for key in SIZES + ("all",) + LARGER:
        c = e["workspace"] if key == "all" else e["by_size"][key]
        label = f"All {c['instances']}" if key == "all" else "$" + key.replace("x", " \\times ") + "$"
        if c["proven"] < c["instances"]:
            label += "$^\\ast$"
        digits = 2 if key in LARGER else 3
        t = c["seconds"]
        pairs = "--".join(f"{n:,}".replace(",", "\\,") for n in c["pairs"])
        rows.append(
            f"{label} & {pairs} & {seconds(t['exact']['median'])} & {seconds(t['exact']['max'])}"
            f" & {seconds(t['t_S']['mean'])} & {c['N_eq']['mean']:.0f} & {pm(c['S'], digits)} & {pm(c['U_eq'], digits)}"
            f" & {pm(c['U_half'], digits)} & {pm(c['U_double'], digits)} \\\\")
    return rows


def quoted_equal_time(e: dict) -> list:
    """The figures of E32 the running text quotes, for transcription; not part of the check."""
    lines = []
    for name, test in e["tests"].items():
        lines.append(
            f"E32 {name}: first smaller {test['first_smaller']}, larger {test['first_larger']}, equal {test['zero']} of "
            f"{test['pairs']}; median difference {test['median_difference']:.4f}; p = {test['p']:.3g}; "
            f"threshold {test['alpha']:.4f}; rejects {test['rejects']}; {test['method']}")
    for key in ("workspace", "larger") + SIZES + LARGER:
        c = e[key] if key in ("workspace", "larger") else e["by_size"][key]
        t = c["seconds"]
        optimal = c.get("optimal_runs", {})
        lines.append(
            f"E32 {key}: proven {c['proven']} of {c['instances']}, not proven {c['not_proven']}; runs {c['runs']}; "
            f"optimal runs S {optimal.get('S')}, half {optimal.get('U_half')}, eq {optimal.get('U_eq')}, "
            f"double {optimal.get('U_double')}; N_eq {c['N_eq']['min']}..{c['N_eq']['max']} (mean {c['N_eq']['mean']:.0f}), "
            f"N_half mean {c['N_half']['mean']:.0f}, N_double mean {c['N_double']['mean']:.0f}; capped runs {c['capped_runs']}; "
            f"U64 {pm(c['U64'], 3)}, U1024 {pm(c['U1024'], 3)}, U_all {pm(c['U_all'], 3)}, greedy {pm(c['greedy'], 2)}")
        lines.append(
            f"E32 {key} seconds: t_S mean {t['t_S']['mean']:.4f} median {t['t_S']['median']:.4f} min {t['t_S']['min']:.4f} "
            f"max {t['t_S']['max']:.4f}; anneal {t['anneal']['mean']:.4f}, stages {t['stages']['mean']:.4f}, "
            f"build {t['build_qubo']['mean']:.5f}; exact mean {t['exact']['mean']:.4f} median {t['exact']['median']:.4f} "
            f"min {t['exact']['min']:.4f} max {t['exact']['max']:.4f}; greedy {t['greedy']['mean']:.5f}; "
            f"exact faster than t_S on {c['exact_faster_than_t_S']}, within 20 s on {c['exact_within_default_limit']}; "
            f"exact / t_S median {c['exact_over_t_S']['median']:.3f} min {c['exact_over_t_S']['min']:.3f} "
            f"max {c['exact_over_t_S']['max']:.3f}")
    for key in LARGER:
        part = [r for r in e["rows"] if f"{r['assets']}x{r['tasks']}" == key and r["proven"]]
        mean = lambda r, f: sum(run[f] for run in r["runs"]) / len(r["runs"])  # noqa: E731
        closer = sum(mean(r, "S") < mean(r, "U_eq") for r in part)
        lines.append(f"E32 {key}: of {len(part)} proven instances the annealed reads are closer than U_eq on {closer}; "
                     f"exact seconds on the proven {sorted(round(r['exact']['seconds'], 3) for r in part)}")
    return lines


def quoted(result: dict) -> list:
    """The figures the running text quotes, for transcription; not part of the check."""
    a, b = result["part_a"], result["part_b"]
    lines = []
    for name, test in list(a["tests"].items()) + list(b["tests"].items()):
        lines.append(
            f"{name}: first smaller {test['first_smaller']}, larger {test['first_larger']}, equal {test['zero']} of "
            f"{test['pairs']}; median difference {test['median_difference']:.4f}; p = {test['p']:.3g}; "
            f"threshold {test['alpha']:.4f}; rejects {test['rejects']}; {test['method']}")
    whole = a["all"]
    for arm in ("S", "U"):
        lines.append(f"{arm}: " + "; ".join(f"{field} {pm(value, 3)}" for field, value in whole[arm].items()
                                               if isinstance(value, dict)))
    lines.append(f"greedy optimal on {whole['greedy_optimal_instances']} instances; fill of the empty plan {pm(whole['fill_only'], 3)}; "
                 f"duty limit binding on {whole['duty_limit_binding_instances']}; proven {a['proven']}, not proven {len(a['not_proven'])}")
    lines.append(f"seconds per run: annealing {pm(whole['anneal_seconds'], 4)}, repair and fill {pm(whole['stages_seconds'], 4)}")
    for key in SIZES:
        c = a["by_size"][key]
        lines.append(f"{key}: raw feasible share S {c['S']['raw_feasible_share']['mean']:.3f}, U {c['U']['raw_feasible_share']['mean']:.4f}; "
                     f"duty limit binding on {c['duty_limit_binding_instances']}")
    for row in a["presets"]:
        lines.append(f"{row['scenario']} ({row['pairs']} pairs): S {row['S_best_filled']:.3f}, U {row['U_best_filled']:.3f}, "
                     f"U16 {row['U16_best_filled']:.4f}, greedy {row['greedy']:.3f}")
    lines.append(f"Part B: {b['proven']} instances, {b['pairs'][0]} to {b['pairs'][1]} pairs, largest circuit check "
                 f"{b['largest_circuit_check']:.1e}")
    return lines


def main() -> None:
    result = json.loads((RESULTS / "e31_mission_allocation.json").read_text())
    equal_time = json.loads((RESULTS / "e32_mission_equal_time.json").read_text())
    tables = {"part A (by size)": part_a_rows(result["part_a"]), "part B (by distribution)": part_b_rows(result["part_b"]),
              "equal time (by size)": equal_time_rows(equal_time)}
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
    print("% --- quoted in the text")
    print("\n".join("% " + line for line in quoted(result) + quoted_equal_time(equal_time)))


if __name__ == "__main__":
    main()
