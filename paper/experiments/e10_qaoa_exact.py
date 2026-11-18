"""E10 - QAOA output distribution, computed exactly, against uniform sampling.

E6 judged QAOA by the best of 1024 shots and by the minimum of a shot-noisy
training trace. Both are noisy, and the minimum is biased low. Here the final
trained state of every run is evaluated *exactly* (2^18 amplitudes), and every
metric has a closed-form uniform reference:

  r       = (E_unif - <E>) / (E_unif - E_0)   (0 = uniform, 1 = ground state)
  P(gs)   = probability of the QUBO ground level (all degenerate states)
  P(feas) = probability of a constraint-satisfying assignment (one-hot,
            linking, site cap; slack bits ignored)
  P(cap)  = ... that also respects site capacities, which the QUBO does not
            encode (the platform enforces them in classical repair)
  P(opt)  = probability of the true facility-location optimum, i.e. the
            cheapest capacity-respecting assignment; it is NOT the QUBO
            ground level
  P(top)  = probability mass on the lowest-energy 0.1% of states
  E[rank of best of N] for N = 1024 shots, from the exact CDF.

Arms, each at p = 1, 2, 3:
  raw       un-normalised cost Hamiltonian, platform training procedure
            (the configuration behind E6)
  norm      cost Hamiltonian divided by its largest coefficient, same training
  platform  the shipped sampler itself (``sample_qubo(backend="gate")``, now
            normalised); its final parameters are evaluated exactly. It must
            agree in distribution with ``norm`` if the emulation is faithful.
  exact-opt normalised, exact-<E> objective, 50 random starts in [0, pi]^{2p},
            L-BFGS-B to convergence, best kept: a lower bound on what the
            ansatz can reach, free of shot noise and of the 60-step budget
Platform training procedure: COBYLA (maxiter 60), x0 ~ U[0, pi]^{2p}, objective
= sample mean of 1024 shots, common random numbers across evaluations - the
behaviour of ``_sample_gate``.

At p = 1 the exact <E> is also tabulated on a (gamma, beta) grid over
[0, pi] x [0, pi) for both Hamiltonians: an optimiser-free reference for the
best value inside the initialisation box, and a direct picture of what the
optimiser sees.
"""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "AER_MAX_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
from common import FIGURES, RESULTS, mean_ci, mpl, point_seed, pool_workers, save  # noqa: E402
from e6_qaoa_control import all_energies  # noqa: E402
from scipy.optimize import minimize  # noqa: E402
from scipy.stats import mannwhitneyu  # noqa: E402

from core.optimizers.fleet.samplers import build_qaoa_circuit, ising_scale, sample_qubo  # noqa: E402
from core.optimizers.network.qubo import build_network_qubo  # noqa: E402
from core.optimizers.network.scenario import build_scenario  # noqa: E402

SCENARIO = "pilot_2of4"
REPS = [1, 2, 3]
SEEDS = 20
SHOTS = 1024
MAXITER = 60
EXACT_STARTS = 50
TOP_FRACTION = 0.001
GRID_GAMMA = 257  # gamma in [0, pi], endpoints included
GRID_BETA = 128  # beta in [0, pi); the mixer has period pi
_CHUNK = 6

_CTX = {}


def _context():
    if not _CTX:
        problem = build_scenario(SCENARIO)
        qubo = build_network_qubo(problem)
        energies, states = all_energies(qubo)
        h, J, _ = qubo.to_ising()
        z = 1.0 - 2.0 * states
        iu = np.triu_indices(qubo.n_vars, 1)
        ising = z @ h + ((z[:, iu[0]] * z[:, iu[1]]) * J[iu]).sum(axis=1)
        _CTX.update(problem=problem, qubo=qubo, energies=energies, states=states,
                    ising=ising, scale=ising_scale(qubo), n=qubo.n_vars)
        valid = _feasible_mask()
        within = valid & _capacity_mask()
        best = float(energies[within].min())
        _CTX.update(feasible=valid, capacity_feasible=within, cflp_optimum=best,
                    ground=np.abs(energies - energies.min()) < 1e-6,
                    optimum=within & (np.abs(energies - best) < 1e-6))
    return _CTX


def _apply_mixer(psi: np.ndarray, beta: float, n: int) -> np.ndarray:
    """exp(-i beta X) on every qubit, _CHUNK qubits at a time as one matrix product."""
    cb, sb = np.cos(beta), -1j * np.sin(beta)
    r = np.array([[cb, sb], [sb, cb]])
    blocks = {}
    done = 0
    while done < n:
        k = min(_CHUNK, n - done)
        if k not in blocks:
            m = np.ones((1, 1), dtype=complex)
            for _ in range(k):
                m = np.kron(m, r)
            blocks[k] = m
        m = blocks[k]
        if done + k == n:
            psi = psi.reshape(-1, 2 ** k) @ m.T
        else:
            psi = np.matmul(m, psi.reshape(2 ** done, 2 ** k, -1))
        done += k
    return psi.reshape(-1)


def qaoa_state(params: np.ndarray, p: int, normalize: bool) -> np.ndarray:
    c = _context()
    n = c["n"]
    diag = c["ising"] / (c["scale"] if normalize else 1.0)
    psi = np.full(2 ** n, 2 ** (-n / 2), dtype=complex)
    for layer in range(p):
        psi = psi * np.exp(-1j * params[layer] * diag)
        psi = _apply_mixer(psi, float(params[p + layer]), n)
    return psi


def _validate_simulator() -> float:
    from qiskit.quantum_info import Statevector

    c = _context()
    worst = 0.0
    rng = np.random.default_rng(0)
    for normalize in (True, False):
        for p in (1, 2):
            x = rng.uniform(0, np.pi, 2 * p)
            ref = Statevector(build_qaoa_circuit(c["qubo"], x, p, measure=False, normalize=normalize)).data
            ours = qaoa_state(x, p, normalize)
            worst = max(worst, 1.0 - abs(np.vdot(ref, ours)) ** 2)
    assert worst < 1e-8, f"numpy QAOA simulator disagrees with qiskit: infidelity {worst}"
    return worst


def _feasible_mask() -> np.ndarray:
    c = _context()
    q, st = c["qubo"], c["states"]
    n_sites, n_dem = len(q.site_ids), len(q.demand_ids)
    opened = np.stack([st[:, q.open_index[j]] for j in range(n_sites)], axis=1)
    ok = opened.sum(axis=1) <= int(c["problem"].max_open_sites or n_sites)
    for d in range(n_dem):
        row = np.stack([st[:, q.assign_index[(d, j)]] for j in range(n_sites)], axis=1)
        ok &= row.sum(axis=1) == 1
        ok &= (row * opened).sum(axis=1) == 1
    return ok


def _capacity_mask() -> np.ndarray:
    c = _context()
    q, st, problem = c["qubo"], c["states"], c["problem"]
    demand = [float(p.demand) for p in problem.demand_points]
    ok = np.ones(len(st), bool)
    for j, site in enumerate(problem.sites):
        load = sum(demand[d] * st[:, q.assign_index[(d, j)]] for d in range(len(demand)))
        ok &= load <= float(site.capacity) + 1e-9
    return ok


def metrics(prob: np.ndarray) -> dict:
    c = _context()
    e = c["energies"]
    order = np.argsort(e, kind="stable")
    pe = prob[order]
    cdf = np.cumsum(pe)
    n_states = len(e)
    e0, eu = float(e[order[0]]), float(e.mean())
    exp_e = float(prob @ e)
    k_top = max(1, int(round(TOP_FRACTION * n_states)))
    # P(best of N lands at rank <= k) = 1 - (1 - cdf_k)^N ; E[rank] = sum_k P(rank > k)
    surv = (1.0 - np.clip(cdf, 0, 1)) ** SHOTS
    exp_rank = 1.0 + float(surv[:-1].sum())
    p_ground = float(prob[c["ground"]].sum())
    p_optimum = float(prob[c["optimum"]].sum())
    return {
        "approx_ratio": (eu - exp_e) / (eu - e0),
        "p_ground": p_ground,
        "p_feasible": float(prob[c["feasible"]].sum()),
        "p_capacity_feasible": float(prob[c["capacity_feasible"]].sum()),
        "p_optimum": p_optimum,
        "p_top": float(cdf[k_top - 1]),
        "best_of_n_rank_pct": 100.0 * exp_rank / n_states,
        "best_of_n_ground_prob": float(1.0 - (1.0 - p_ground) ** SHOTS),
        "best_of_n_optimum_prob": float(1.0 - (1.0 - p_optimum) ** SHOTS),
    }


def _train(args):
    arm, p, rep = args
    c = _context()
    normalize = arm != "raw"
    seed = point_seed(10, {"raw": 0, "norm": 1}[arm], p, rep)
    e = c["energies"]
    trace = []

    def objective(x):
        prob = np.abs(qaoa_state(x, p, normalize)) ** 2
        counts = np.random.default_rng(seed).multinomial(SHOTS, prob / prob.sum())
        val = float(counts @ e / SHOTS)
        trace.append(val)
        return val

    x0 = np.random.default_rng(seed).uniform(0.0, np.pi, 2 * p)
    res = minimize(objective, x0, method="COBYLA", options={"maxiter": MAXITER})
    prob = np.abs(qaoa_state(res.x, p, normalize)) ** 2
    m = metrics(prob)
    m.update(arm=arm, p=p, rep=rep, evaluations=len(trace), trace_min=min(trace),
             exact_expectation=float(prob @ e), params=[float(v) for v in res.x])
    return m


def _exact_start(args):
    p, start = args
    c = _context()
    e = c["energies"]
    e0, eu = float(e.min()), float(e.mean())
    x0 = np.random.default_rng(point_seed(11, p, start)).uniform(0.0, np.pi, 2 * p)
    # minimise -r: dimensionless, so finite-difference gradients are well conditioned
    res = minimize(lambda x: (float((np.abs(qaoa_state(x, p, True)) ** 2) @ e) - eu) / (eu - e0), x0,
                   method="L-BFGS-B", options={"maxiter": 1000, "ftol": 1e-13, "gtol": 1e-7})
    return p, start, float(-res.fun), [float(v) for v in res.x], int(res.nfev), bool(res.success)


def _platform(args):
    p, seed = args
    c = _context()
    out = sample_qubo(c["qubo"], backend="gate", shots=SHOTS, reps=p, seed=seed, maxiter=MAXITER)
    x = np.asarray(out.extra["final_parameters"])
    m = metrics(np.abs(qaoa_state(x, p, True)) ** 2)
    m.update(arm="platform", p=p, rep=seed, returned_energy=float(out.energy), params=[float(v) for v in x],
             backend=out.backend_name, fallback=out.fallback_reason)
    return m


def _grid_row(args):
    normalize, gamma = args
    c = _context()
    n, e = c["n"], c["energies"]
    diag = c["ising"] / (c["scale"] if normalize else 1.0)
    phased = np.full(2 ** n, 2 ** (-n / 2), dtype=complex) * np.exp(-1j * gamma * diag)
    betas = np.linspace(0.0, np.pi, GRID_BETA, endpoint=False)
    return [float((np.abs(_apply_mixer(phased, float(b), n)) ** 2) @ e) for b in betas]


def _grid_summary(exp_e: np.ndarray, e0: float, eu: float) -> dict:
    """Summary of the p=1 landscape r(gamma, beta); rows are gamma, columns beta."""
    r = (eu - exp_e) / (eu - e0)
    i, j = np.unravel_index(int(np.argmax(r)), r.shape)
    return {
        "max_r": float(r.max()),
        "argmax_gamma": float(np.linspace(0.0, np.pi, GRID_GAMMA)[i]),
        "argmax_beta": float(np.linspace(0.0, np.pi, GRID_BETA, endpoint=False)[j]),
        "min_r": float(r.min()),
        "mean_r": float(r.mean()),
        "sd_r": float(r.std(ddof=1)),
        "fraction_above_uniform": float((r > 0).mean()),
        # Pearson correlation of r between neighbouring gamma grid points (spacing pi/256)
        "neighbour_correlation_gamma": float(np.corrcoef(r[:-1].ravel(), r[1:].ravel())[0, 1]),
        "neighbour_correlation_beta": float(np.corrcoef(r[:, :-1].ravel(), r[:, 1:].ravel())[0, 1]),
    }


def main() -> None:
    t0 = time.time()
    c = _context()
    infid = _validate_simulator()
    exact_file = RESULTS / "e2_facility.json"
    if exact_file.exists():  # the cheapest capacity-respecting state must be the MILP optimum of E2
        milp = next(r for r in json.loads(exact_file.read_text())["rows"] if r["scenario"] == SCENARIO)
        assert abs(c["cflp_optimum"] - milp["exact"]["objective"]) < 1e-3, (c["cflp_optimum"], milp["exact"]["objective"])
    n_states = 2 ** c["n"]
    uniform = metrics(np.full(n_states, 1.0 / n_states))
    print(f"n={c['n']} scale={c['scale']:.1f} feasible states={int(c['feasible'].sum())} "
          f"sim infidelity={infid:.1e}\nuniform {uniform}", flush=True)

    e0, eu = float(c["energies"].min()), float(c["energies"].mean())
    gammas = np.linspace(0.0, np.pi, GRID_GAMMA)
    jobs = [(arm, p, r) for arm in ("raw", "norm") for p in REPS for r in range(SEEDS)]
    workers = pool_workers()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        trained = list(ex.map(_train, jobs))
        print("trained arms done", round(time.time() - t0, 1), "s", flush=True)
        grid_rows = list(ex.map(_grid_row, [(norm, float(g)) for norm in (False, True) for g in gammas]))
        print("p=1 grids done", round(time.time() - t0, 1), "s", flush=True)
        starts = list(ex.map(_exact_start, [(p, s) for p in REPS for s in range(EXACT_STARTS)]))
        print("exact-objective starts done", round(time.time() - t0, 1), "s", flush=True)
        platform_runs = list(ex.map(_platform, [(p, s) for p in REPS for s in range(SEEDS)]))
        print("platform sampler done", round(time.time() - t0, 1), "s", flush=True)
    assert all(m["fallback"] is None for m in platform_runs), "platform sampler fell back"

    grids = {"raw": np.asarray(grid_rows[:GRID_GAMMA]), "norm": np.asarray(grid_rows[GRID_GAMMA:])}
    grid_summary = {arm: _grid_summary(g, e0, eu) for arm, g in grids.items()}
    for arm, s in grid_summary.items():
        print(f"grid {arm:4s}: max r={s['max_r']:.4f} at gamma={s['argmax_gamma']:.4f} beta={s['argmax_beta']:.4f} "
              f"sd={s['sd_r']:.4f} neighbour corr(gamma)={s['neighbour_correlation_gamma']:.3f}")

    exact_rows = []
    for p in REPS:
        mine = [s for s in starts if s[0] == p]
        best = max(mine, key=lambda s: s[2])
        prob = np.abs(qaoa_state(np.asarray(best[3]), p, True)) ** 2
        m = metrics(prob)
        start_r = [s[2] for s in mine]
        assert abs(m["approx_ratio"] - best[2]) < 1e-9
        m.update(p=p, params=best[3], start_approx_ratios=start_r,
                 starts_within_0p01_of_best=sum(v >= m["approx_ratio"] - 0.01 for v in start_r),
                 starts_converged=sum(s[5] for s in mine), evaluations=[s[4] for s in mine])
        exact_rows.append(m)
        print(f"exact-opt p={p}: r={m['approx_ratio']:.4f} P(gs)={m['p_ground']:.2e} "
              f"P(feas)={m['p_feasible']:.4f} starts within 0.01 of best: {m['starts_within_0p01_of_best']}/{len(mine)} "
              f"converged {m['starts_converged']}/{len(mine)} max evals {max(m['evaluations'])}")

    keys = ["approx_ratio", "p_ground", "p_feasible", "p_capacity_feasible", "p_optimum", "p_top",
            "best_of_n_rank_pct", "best_of_n_ground_prob", "best_of_n_optimum_prob"]
    summary = {}
    for arm in ("raw", "norm", "platform"):
        for p in REPS:
            sel = [t for t in trained + platform_runs if t["arm"] == arm and t["p"] == p]
            s = {k: mean_ci([t[k] for t in sel]) for k in keys}
            s["median_approx_ratio"] = float(np.median([t["approx_ratio"] for t in sel]))
            s["median_p_feasible"] = float(np.median([t["p_feasible"] for t in sel]))
            if arm != "platform":
                # trace-min vs exact expectation: the winner's-curse bias of E6's metric
                s["trace_min_bias"] = mean_ci([t["trace_min"] - t["exact_expectation"] for t in sel])
                s["trace_min_ratio"] = mean_ci([(eu - t["trace_min"]) / (eu - e0) for t in sel])
            s["runs_better_than_uniform"] = sum(t["approx_ratio"] > 0 for t in sel)
            summary[f"{arm}_p{p}"] = s
            print(f"{arm:8s} p={p}: r={s['approx_ratio']['mean']:.3f}±{s['approx_ratio']['ci95']:.3f} "
                  f"(median {s['median_approx_ratio']:.3f}) P(feas)={s['p_feasible']['mean']:.4f} "
                  f"best-rank={s['best_of_n_rank_pct']['mean']:.4f}% >unif {s['runs_better_than_uniform']}/{len(sel)}")

    # Is the emulated procedure the shipped one? Two-sided Mann-Whitney U per depth.
    emulation_check = {}
    for p in REPS:
        a = [t for t in trained if t["arm"] == "norm" and t["p"] == p]
        b = [t for t in platform_runs if t["p"] == p]
        emulation_check[f"p{p}"] = {
            k: float(mannwhitneyu([t[k] for t in a], [t[k] for t in b], alternative="two-sided").pvalue)
            for k in ("approx_ratio", "p_feasible")}
    # Does normalisation change the trained state? One-sided (norm > raw) Mann-Whitney U per depth.
    normalisation_effect = {}
    for p in REPS:
        a = [t for t in trained if t["arm"] == "norm" and t["p"] == p]
        b = [t for t in trained if t["arm"] == "raw" and t["p"] == p]
        normalisation_effect[f"p{p}"] = {
            k: float(mannwhitneyu([t[k] for t in a], [t[k] for t in b], alternative="greater").pvalue)
            for k in ("approx_ratio", "p_feasible")}
    print("emulation vs platform (two-sided p):", emulation_check)
    print("norm > raw (one-sided p):", normalisation_effect)

    save("e10_qaoa_exact", {
        "scenario": SCENARIO, "n_vars": c["n"], "states": n_states, "shots": SHOTS, "maxiter": MAXITER,
        "seeds": SEEDS, "exact_starts": EXACT_STARTS, "top_fraction": TOP_FRACTION,
        "hamiltonian_scale": c["scale"], "feasible_states": int(c["feasible"].sum()),
        "capacity_feasible_states": int(c["capacity_feasible"].sum()),
        "ground_degeneracy": int(c["ground"].sum()), "optimum_states": int(c["optimum"].sum()),
        "ground_state_respects_capacity": bool((c["ground"] & c["capacity_feasible"]).any()),
        "cflp_optimum_energy": c["cflp_optimum"],
        "energy_levels_below_optimum": int(len(np.unique(np.round(c["energies"][c["energies"] < c["cflp_optimum"] - 1e-6], 6)))),
        "ground_energy": float(c["energies"].min()), "uniform_mean_energy": float(c["energies"].mean()),
        "simulator_max_infidelity_vs_qiskit": infid,
        "uniform": uniform, "summary": summary, "exact_opt": exact_rows,
        "p1_grid": {"gamma_points": GRID_GAMMA, "beta_points": GRID_BETA, **grid_summary},
        "emulation_vs_platform_mannwhitney_p": emulation_check,
        "norm_greater_than_raw_mannwhitney_p": normalisation_effect,
        "runs": trained, "platform_runs": platform_runs, "wall_s": round(time.time() - t0, 1),
    })

    plt = mpl()
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.4))
    width = 0.2
    labels = {"raw": "un-normalised, emulated", "norm": "normalised, emulated", "platform": "normalised, shipped sampler"}
    for key, a in (("approx_ratio", ax[0]), ("p_feasible", ax[1])):
        for i, arm in enumerate(("raw", "norm", "platform")):
            mu = [summary[f"{arm}_p{p}"][key]["mean"] for p in REPS]
            ci = [summary[f"{arm}_p{p}"][key]["ci95"] for p in REPS]
            a.bar(np.arange(3) + (i - 1.5) * width, mu, width, yerr=ci, capsize=2, label=labels[arm])
        a.bar(np.arange(3) + 1.5 * width, [r[key] for r in exact_rows], width, label="normalised, exact objective (best of 50)")
        a.set_xticks(range(3), [f"p={p}" for p in REPS])
    ax[0].axhline(0, color="k", lw=0.8)
    ax[0].set_ylabel("approximation ratio r")
    ax[0].legend(fontsize=5.5)
    ax[1].axhline(uniform["p_feasible"], color="grey", ls=":", lw=0.8)
    ax[1].set_yscale("log")
    ax[1].set_ylabel("P(constraint-satisfying)")
    fig.tight_layout()
    fig.savefig(FIGURES / "qaoa_exact.pdf")

    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.5))
    titles = {"raw": "un-normalised", "norm": "normalised"}
    for a, arm in zip(ax, ("raw", "norm")):
        r = (eu - grids[arm]) / (eu - e0)
        lim = float(np.abs(r).max())
        im = a.imshow(r.T, origin="lower", aspect="auto", extent=(0, np.pi, 0, np.pi),
                      cmap="RdBu", vmin=-lim, vmax=lim, interpolation="nearest")
        a.set_title(titles[arm], fontsize=8)
        a.set_xlabel(r"$\gamma$")
        fig.colorbar(im, ax=a, label="r")
    ax[0].set_ylabel(r"$\beta$")
    fig.tight_layout()
    fig.savefig(FIGURES / "qaoa_landscape.pdf")
    print("done", round(time.time() - t0, 1), "s")


if __name__ == "__main__":
    main()
