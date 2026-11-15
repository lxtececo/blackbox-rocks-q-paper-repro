"""E3 — Spaced site selection on Rydberg arrays: ablation and robustness.

(a) Ablation: Omega-blockade radius vs final-detuning cutoff on random
    unit-disk instances.
(b) Sweep duration on the seeded Rhine-Ruhr instance.
(c) Embeddability of random instances vs spacing / closest-pair ratio.
"""

from __future__ import annotations

import math
import time

import numpy as np
from common import FIGURES, mean_ci, mpl, save, wilson

from core.optimizers.network import rydberg_mis as rm
from core.optimizers.network.models import CandidateSite

KM_PER_DEG = math.pi / 180.0 * rm.EARTH_RADIUS_KM
LAT0, LON0 = 51.0, 7.0
SHOTS = 200


def sites_from_km(points):
    return [
        CandidateSite(id=f"P{i}", name=f"P{i}", lat=LAT0 + y / KM_PER_DEG,
                      lon=LON0 + x / (KM_PER_DEG * math.cos(math.radians(LAT0))))
        for i, (x, y) in enumerate(points)
    ]


def random_points(rng, n, box_km=40.0, d_min=6.0, tries=20000):
    pts = []
    for _ in range(tries):
        p = rng.uniform(0, box_km, 2)
        if all(math.dist(p, q) >= d_min for q in pts):
            pts.append(p)
            if len(pts) == n:
                return pts
    return None


def omega_mapping_embedding(graph, spacing):
    """The first-cut mapping: spacing threshold -> Omega blockade radius."""
    n = graph.n
    pair = [graph.distances_km[i][j] for i in range(n) for j in range(i + 1, n)]
    edge = [d for d in pair if d < spacing]
    free = [d for d in pair if d >= spacing]
    thr = math.sqrt(max(edge) * min(free)) if edge and free else spacing
    rb = (rm.C6 / rm.OMEGA_MAX) ** (1 / 6)
    scale = max(rb / thr, rm.MIN_ATOM_SPACING_M / min(pair))
    blockade = thr * scale
    omega = min(rm.OMEGA_MAX, rm.C6 / blockade ** 6)
    xs = [x for x, _ in graph.coords_km]
    ys = [y for _, y in graph.coords_km]
    return rm.Embedding(
        positions_m=[((x - min(xs)) * scale, (y - min(ys)) * scale) for x, y in graph.coords_km],
        scale_um_per_km=scale * 1e6, threshold_km=thr, blockade_radius_m=blockade,
        omega_max=omega, delta_max=min(rm.DELTA_MAX, 6.0 * omega),
    )


def run(graph, emb, optimum, duration=rm.MAX_DURATION_S, shots=SHOTS):
    shots_rows = rm.run_local(rm.build_program(emb, duration_s=duration), shots=shots, steps=80)
    d = rm.decode(graph, shots_rows, optimum)
    return d


def main() -> None:
    t0 = time.time()
    rng = np.random.default_rng(2026)

    # (a) ablation on random instances, n = 8
    ablation = []
    while len(ablation) < 24:
        pts = random_points(rng, 8)
        if pts is None:
            continue
        d_min = min(math.dist(a, b) for i, a in enumerate(pts) for b in pts[i + 1:])
        spacing = d_min * float(rng.uniform(1.4, 2.4))
        problem = rm.SpacingProblem(sites_from_km(pts), spacing)
        g = rm.conflict_graph(problem)
        if not g.edges:
            continue
        try:
            emb_new = rm.embed(g, spacing, max_atoms=rm.MAX_LOCAL_ATOMS)
        except rm.EmbeddingError:
            continue
        opt = len(rm.exact_mis(g))
        greedy = len(rm.greedy_mis(g))
        new = run(g, emb_new, opt)
        old = run(g, omega_mapping_embedding(g, spacing), opt)
        ablation.append({
            "edges": len(g.edges), "ratio": spacing / d_min, "optimum": opt, "greedy": greedy,
            "ambiguous_pairs": len(emb_new.ambiguous_pairs),
            "cutoff": {k: new[k] for k in ("raw_independent_fraction", "raw_optimal_fraction", "raw_mean_size")}
            | {"final_optimal": len(new["best_set"]) == opt},
            "omega": {k: old[k] for k in ("raw_independent_fraction", "raw_optimal_fraction", "raw_mean_size")}
            | {"final_optimal": len(old["best_set"]) == opt},
        })
        print(f"inst {len(ablation):2d} E={len(g.edges):2d} opt={opt} "
              f"cutoff indep={new['raw_independent_fraction']:.2f} opt={new['raw_optimal_fraction']:.2f} | "
              f"omega indep={old['raw_independent_fraction']:.2f} opt={old['raw_optimal_fraction']:.2f}")

    # (b) sweep duration on Rhine-Ruhr
    durations = [0.5e-6, 1e-6, 2e-6, 3e-6, 4e-6]
    sweep = []
    for spacing in (15.0, 20.0):
        p = rm.rhine_ruhr_problem(spacing)
        g = rm.conflict_graph(p)
        emb = rm.embed(g, spacing, max_atoms=rm.MAX_LOCAL_ATOMS)
        opt = len(rm.exact_mis(g))
        for T in durations:
            d = run(g, emb, opt, duration=T)
            k = round(d["raw_optimal_fraction"] * d["shots_used"])
            lo, hi = wilson(k, d["shots_used"])
            sweep.append({"spacing_km": spacing, "duration_us": T * 1e6,
                          "raw_independent_fraction": d["raw_independent_fraction"],
                          "raw_optimal_fraction": d["raw_optimal_fraction"], "ci": [lo, hi]})
            print(f"RR {spacing} km T={T * 1e6:.1f}us opt={d['raw_optimal_fraction']:.3f}")

    # (c) embeddability vs ratio, n = 10 (no simulation)
    embed_rows = []
    for _ in range(2000):
        pts = random_points(rng, 10, box_km=60.0, d_min=4.0)
        if pts is None:
            continue
        d_min = min(math.dist(a, b) for i, a in enumerate(pts) for b in pts[i + 1:])
        ratio = float(rng.uniform(1.0, 4.0))
        g = rm.conflict_graph(rm.SpacingProblem(sites_from_km(pts), ratio * d_min))
        try:
            rm.embed(g, ratio * d_min, max_atoms=rm.MAX_AQUILA_ATOMS)
            ok, reason = True, ""
        except rm.EmbeddingError as exc:
            ok, reason = False, "field" if "atom field" in str(exc) else "ratio"
        embed_rows.append({"ratio": ratio, "ok": ok, "reason": reason})

    bins = np.arange(1.0, 4.01, 0.25)
    embeddable = []
    for lo_b, hi_b in zip(bins[:-1], bins[1:]):
        sel = [r for r in embed_rows if lo_b <= r["ratio"] < hi_b]
        if sel:
            embeddable.append({"ratio_mid": float((lo_b + hi_b) / 2), "n": len(sel),
                               "fraction": sum(r["ok"] for r in sel) / len(sel)})

    def summary(key):
        return {m: mean_ci([a[key][m] for a in ablation]) for m in ("raw_independent_fraction", "raw_optimal_fraction")} | {
            "final_optimal_instances": sum(a[key]["final_optimal"] for a in ablation)}

    amb = [a["ambiguous_pairs"] for a in ablation]
    opt_frac = [a["cutoff"]["raw_optimal_fraction"] for a in ablation]
    corr = float(np.corrcoef(amb, opt_frac)[0, 1]) if np.std(amb) > 0 else None
    save("e3_rydberg", {
        "shots": SHOTS, "instances": len(ablation), "ablation": ablation,
        "ablation_summary": {"cutoff": summary("cutoff"), "omega": summary("omega"),
                             "greedy_optimal_instances": sum(a["greedy"] == a["optimum"] for a in ablation),
                             "corr_ambiguous_vs_raw_optimal": corr},
        "duration_sweep": sweep, "embeddability": embeddable,
        "embed_failures": {"ratio": sum(r["reason"] == "ratio" for r in embed_rows),
                           "field": sum(r["reason"] == "field" for r in embed_rows), "total": len(embed_rows)},
        "wall_s": round(time.time() - t0, 1),
    })

    plt = mpl()
    fig, ax = plt.subplots(1, 3, figsize=(6.8, 2.2))
    x = np.arange(len(ablation))
    order = np.argsort([a["cutoff"]["raw_optimal_fraction"] for a in ablation])
    ax[0].plot(x, [ablation[i]["cutoff"]["raw_optimal_fraction"] for i in order], "o", ms=3, label="Δ-end cutoff")
    ax[0].plot(x, [ablation[i]["omega"]["raw_optimal_fraction"] for i in order], "x", ms=3, label="Ω blockade radius")
    ax[0].set_xlabel("random instance (sorted)")
    ax[0].set_ylabel("raw optimal fraction")
    ax[0].legend()
    for spacing, mk in ((15.0, "o-"), (20.0, "s-")):
        rows = [r for r in sweep if r["spacing_km"] == spacing]
        ax[1].plot([r["duration_us"] for r in rows], [r["raw_optimal_fraction"] for r in rows], mk, ms=3, label=f"{spacing:.0f} km")
    ax[1].set_xlabel("sweep duration (µs)")
    ax[1].set_ylabel("raw optimal fraction")
    ax[1].legend()
    ax[2].plot([e["ratio_mid"] for e in embeddable], [e["fraction"] for e in embeddable], "o-", ms=3)
    ax[2].set_xlabel("spacing / closest pair")
    ax[2].set_ylabel("embeddable fraction")
    fig.tight_layout()
    fig.savefig(FIGURES / "rydberg_mis.pdf")
    print("figure written", round(time.time() - t0, 1), "s")


if __name__ == "__main__":
    main()
