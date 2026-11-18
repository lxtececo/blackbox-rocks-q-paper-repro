"""E15 - Amplitude-estimation depth probe on qubits chosen by a calibration measured in the session.

Follow-up to E14-B, designed AFTER its result was seen, and labelled as such
wherever it is reported. E14-B let the transpiler place the circuits with the
device's reported calibration. That calibration turned out to be three days
old, and two of the three qubits it chose read out |1> as 0 in 8% and 5% of
shots against a reported 0.6% and 0.2%. The Grover iterate lost half its
contrast per application there. This run asks what the same circuits do on
qubits that pass a readout check taken minutes before.

Plan, fixed before the run:
 1. Screen (one job): all qubits prepared in |0> and in |1>, 2000 shots each;
    gives P(read 1 | 0) and P(read 0 | 1) for every qubit.
 2. Selection uses the screen only, never probe data. Qubits with
    P(read 0 | 1) > 0.03 or P(read 1 | 0) > 0.02 are rejected. A circuit is
    transpiled against the largest connected group of accepted qubits not
    yet used, with each qubit's measurement error set to its measured mean
    assignment error. Everything else is as in E14-B: optimisation level 3,
    best of 8 seeds by two-qubit gate count, all powers of a register on the
    qubits chosen for k = 1. Qubit sets are chosen largest register first; a
    register that no remaining group can hold is skipped and reported.
    (Added after the first attempt, before any probe circuit had run: the
    screen rejected 80 of 156 qubits, the accepted ones form groups of at
    most 8, and Qiskit's layout search failed on the fragmented target.)
 3. Registers: n = 5 price qubits on one set, k = 0, 1, 2; n = 3 on one set,
    k = 0..6; n = 2 on three mutually disjoint sets, k = 0..6. 4000 shots per
    circuit, XY4 dynamical decoupling, gate twirling. Readout calibration of
    every used set in the same job as the circuits.
 4. Analysis exactly as E14-B (same functions): objective-qubit probability,
    contrast model (f_A, f_Q), full-register signal fraction, noise-aware
    amplitude estimate. Every selected set is reported.

Usage: e15_mlae_screened_qubits.py [--dry-run]
"""

from __future__ import annotations

import inspect
import math
import os
import sys
import time

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "AER_MAX_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
from common import point_seed, save  # noqa: E402
from scipy.stats import binomtest, chi2  # noqa: E402

import e14_ibm_hardware_controls as e14  # noqa: E402

DRY = "--dry-run" in sys.argv
e14.SIM = DRY
SCREEN_SHOTS = 2000
MAX_P01 = 0.03  # P(read 0 | prepared 1)
MAX_P10 = 0.02  # P(read 1 | prepared 0)
PLAN = [(5, [0, 1, 2], 1), (3, [0, 1, 2, 3, 4, 5, 6], 1), (2, [0, 1, 2, 3, 4, 5, 6], 3)]  # (n, powers, qubit sets)
SHOTS = e14.MLAE_SHOTS


def largest_group(backend, removed) -> set:
    """Largest connected set of qubits not in ``removed`` (ties: the one with the lowest qubit index)."""
    neighbours = {}
    for a, b in backend.coupling_map.get_edges():
        if a not in removed and b not in removed:
            neighbours.setdefault(a, set()).add(b)
            neighbours.setdefault(b, set()).add(a)
    seen, best = set(), set()
    for start in sorted(neighbours):
        if start in seen:
            continue
        group, stack = {start}, [start]
        while stack:
            for nxt in neighbours[stack.pop()]:
                if nxt not in group:
                    group.add(nxt)
                    stack.append(nxt)
        seen |= group
        if len(group) > len(best):
            best = group
    return best


def screened_target(backend, p10, p01, removed):
    """A compact target over the largest connected group of accepted qubits, with measured readout errors.

    Returns ``(target, group)``; target qubit i is physical qubit ``group[i]``.
    """
    from qiskit.transpiler import InstructionProperties, Target

    src = backend.target
    group = sorted(largest_group(backend, removed))
    pos = {q: i for i, q in enumerate(group)}
    out = Target(num_qubits=len(group), dt=src.dt)
    for name in src.operation_names:
        op = src.operation_from_name(name)
        if inspect.isclass(op):
            out.add_instruction(op, name=name)
            continue
        props = {}
        for qargs, prop in src[name].items():
            if qargs is None:
                props[None] = prop
                continue
            if not all(q in pos for q in qargs):
                continue
            if name == "measure":
                prop = InstructionProperties(duration=getattr(prop, "duration", None),
                                             error=float((p10[qargs[0]] + p01[qargs[0]]) / 2))
            props[tuple(pos[q] for q in qargs)] = prop
        if props:
            out.add_instruction(op, props, name=name)
    return out, group


def prepare(circuit, target, group, total, initial_layout=None):
    """Best of several seeds on the compact target, placed on the physical qubits. Returns (circuit, compact layout)."""
    from qiskit import ClassicalRegister, QuantumCircuit
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

    best = None
    for seed in e14.TRANSPILER_SEEDS:
        pm = generate_preset_pass_manager(optimization_level=3, target=target, seed_transpiler=seed,
                                          initial_layout=initial_layout)
        isa = pm.run(circuit)
        if best is None or e14.two_qubit_gates(isa) < e14.two_qubit_gates(best):
            best = isa
    placed = QuantumCircuit(total)
    for creg in best.cregs:
        placed.add_register(ClassicalRegister(creg.size, creg.name))
    placed.compose(best, qubits=[group[i] for i in range(best.num_qubits)], clbits=list(range(best.num_clbits)), inplace=True)
    return placed, best.layout.initial_index_layout(filter_ancillas=True)


def build_register(n, powers):
    from qiskit import ClassicalRegister, QuantumCircuit
    from qiskit.quantum_info import Statevector

    from core.optimizers.montecarlo.grid import build_price_grid
    from core.optimizers.montecarlo.qae import build_grover_operator, build_state_preparation

    o = e14.OPTION
    grid = build_price_grid(o["S0"], o["K"], o["r"], o["sigma"], o["T"], "call", n)
    amplitude = float(grid.exact_discretized_price / grid.payoff_ceiling)
    theta = math.asin(math.sqrt(amplitude))
    prep = build_state_preparation(grid)
    grover = build_grover_operator(prep, n)
    circuits, ideals = [], []
    for k in powers:
        qc = QuantumCircuit(n + 1)
        creg = ClassicalRegister(n + 1, "m")
        qc.add_register(creg)
        qc.compose(prep, inplace=True)
        for _ in range(k):
            qc.compose(grover, inplace=True)
        prob = np.abs(Statevector(qc).data) ** 2
        assert abs(float(prob[e14.objective_mask(n)].sum()) - math.sin((2 * k + 1) * theta) ** 2) < 1e-9
        qc.measure(range(n + 1), creg)
        circuits.append(qc)
        ideals.append(prob)
    return {"n": n, "powers": powers, "amplitude": amplitude, "circuits": circuits, "ideals": ideals,
            "price_ceiling": float(grid.payoff_ceiling)}


def analyse(reg, isa_list, count_list, backend, rng):
    """Same statistics as E14-B, through the same functions."""
    from core.optimizers.montecarlo.qae import _maximize_likelihood

    n, powers = reg["n"], reg["powers"]
    rows, hits = [], []
    for k, ideal, isa, c in zip(powers, reg["ideals"], isa_list, count_list):
        v = e14.counts_vector(c, n + 1)
        one = int(v[e14.objective_mask(n)].sum())
        hits.append(one)
        p_ideal = float(ideal[e14.objective_mask(n)].sum())
        boots = [e14.mixture_weight(rng.multinomial(SHOTS, v / v.sum()).astype(float), ideal) for _ in range(300)]
        row = {"k": k, "two_qubit_gates": e14.two_qubit_gates(isa), "depth": isa.depth(),
               "objective_hits": one, "p_hat": one / SHOTS, "p_hat_ci95": e14.exact_ci(one, SHOTS),
               "p_ideal": p_ideal, "p_vs_ideal": float(binomtest(one, SHOTS, p_ideal).pvalue),
               "p_vs_half": float(binomtest(one, SHOTS, 0.5).pvalue),
               "signal_fraction": e14.mixture_weight(v, ideal),
               "signal_fraction_ci95": [float(x) for x in np.percentile(boots, [2.5, 97.5])],
               "tv_to_ideal": 0.5 * float(np.abs(v / v.sum() - ideal).sum()),
               "tv_to_uniform": 0.5 * float(np.abs(v / v.sum() - 1.0 / len(v)).sum())}
        if backend is not None:
            row["calibration"] = e14.gate_errors(backend, isa)
            row["measured_physical"] = e14.measured_physical(isa)
        rows.append(row)
    entry = {"n_price_qubits": n, "qubits": n + 1, "true_amplitude": reg["amplitude"], "rows": rows}
    if len(powers) >= 4:
        ideal_p = [r["p_ideal"] for r in rows]
        x, fitted, nll = e14.fit_contrast(powers, hits, SHOTS, ideal_p)
        xb = np.array([e14.fit_contrast(powers, rng.binomial(SHOTS, fitted), SHOTS, ideal_p)[0] for _ in range(e14.BOOT // 4)])
        dev = e14.deviance(hits, SHOTS, fitted)
        x3, fitted3, nll3 = e14.fit_contrast(powers, hits, SHOTS, ideal_p, free_floor=True)
        entry["contrast_fit"] = {
            "f_A": float(x[0]), "f_Q": float(x[1]),
            "f_A_ci95": [float(v) for v in np.percentile(xb[:, 0], [2.5, 97.5])],
            "f_Q_ci95": [float(v) for v in np.percentile(xb[:, 1], [2.5, 97.5])],
            "fitted_p": [float(v) for v in fitted], "deviance": dev, "dof": len(powers) - 2,
            "goodness_of_fit_p": float(chi2.sf(dev, len(powers) - 2)),
            "free_floor": {"f_A": float(x3[0]), "f_Q": float(x3[1]), "floor": float(x3[2]),
                           "deviance": e14.deviance(hits, SHOTS, fitted3),
                           "likelihood_ratio_p": float(chi2.sf(2 * (nll - nll3), 1))},
        }
        schedule = [k for k in (0, 1, 2, 4) if k in powers]
        naive = math.sin(_maximize_likelihood(schedule, SHOTS, [hits[powers.index(k)] for k in schedule])) ** 2
        aware, fa, fq = e14.noise_aware_amplitude(powers, hits, SHOTS)
        aware_boot = [e14.noise_aware_amplitude(powers, rng.binomial(SHOTS, fitted), SHOTS)[0] for _ in range(60)]
        entry["amplitude_estimates"] = {
            "true": reg["amplitude"], "noiseless_likelihood_mle": naive, "noiseless_schedule": schedule,
            "noise_aware_mle": aware, "noise_aware_f_A": fa, "noise_aware_f_Q": fq,
            "noise_aware_bootstrap_sd": float(np.std(aware_boot, ddof=1)), "price_ceiling": reg["price_ceiling"],
        }
    return entry


def main() -> None:
    t0 = time.time()
    runner = e14.Runner()
    backend = runner.backend
    out = {"mode": "dry-run" if DRY else "hardware", "backend": runner.name, "shots": SHOTS,
           "thresholds": {"max_p_read0_given1": MAX_P01, "max_p_read1_given0": MAX_P10}}

    removed, p10, p01 = set(), None, None
    if not DRY:
        total = backend.num_qubits
        every = [q for q in range(total) if (q,) in backend.target["measure"] and (q,) in backend.target["x"]]
        counts, meta = runner.run([(c, None, SCREEN_SHOTS) for c in e14.calibration_circuits(backend, every)], "E15-screen readout")
        screen = e14.readout_errors(counts[0][0], counts[1][0], len(every), SCREEN_SHOTS)
        p10, p01 = np.ones(total), np.ones(total)  # qubits that cannot be screened count as rejected
        p10[every], p01[every] = screen["p_read1_given0"], screen["p_read0_given1"]
        removed = {q for q in range(total) if p01[q] > MAX_P01 or p10[q] > MAX_P10}
        reported = [backend.target["measure"][(q,)].error for q in every]
        out["screen"] = {"run": meta, "shots": SCREEN_SHOTS, "screened_qubits": every,
                         "p_read1_given0": p10[every].tolist(), "p_read0_given1": p01[every].tolist(),
                         "rejected_qubits": sorted(removed), "reported_readout_error": reported,
                         "median_measured_assignment_error": float(np.median((p10[every] + p01[every]) / 2)),
                         "median_reported_readout_error": float(np.median([r for r in reported if r is not None])),
                         "calibration_last_update": str(backend.properties().last_update_date)}
        print(f"screen: {len(removed)} of {total} qubits rejected; median measured assignment error "
              f"{out['screen']['median_measured_assignment_error']:.4f} (reported {out['screen']['median_reported_readout_error']:.4f})",
              flush=True)

    pubs, index, sets, skipped = [], [], [], []
    for n, powers, n_sets in PLAN:
        reg = build_register(n, powers)
        for s in range(n_sets):
            if DRY:
                isa = [runner.prepare(c) for c in reg["circuits"]]
                layout = None
            else:
                target, group = screened_target(backend, p10, p01, removed)
                if len(group) < n + 1:
                    skipped.append({"n_price_qubits": n, "set": s, "largest_remaining_group": len(group)})
                    print(f"n={n} set {s}: skipped, largest remaining group has {len(group)} qubits", flush=True)
                    continue
                _, compact = prepare(reg["circuits"][min(1, len(powers) - 1)], target, group, backend.num_qubits)
                isa = [prepare(c, target, group, backend.num_qubits, initial_layout=compact)[0] for c in reg["circuits"]]
                layout = [group[i] for i in compact]
                touched = {c.find_bit(q).index for c in isa for inst in c.data for q in inst.qubits}
                assert not touched & removed, "a rejected or already used qubit was used"
                used = sorted(set(q for c in isa for q in e14.measured_physical(c)))
                removed |= touched  # the next set must be disjoint
            sets.append({"reg": reg, "isa": isa, "layout": layout, "set": s})
            for k, circuit in zip(powers, isa):
                pubs.append((circuit, None, SHOTS))
                index.append((len(sets) - 1, k))
            if not DRY:
                sets[-1]["calibrated"] = used
                for c in e14.calibration_circuits(backend, used):
                    pubs.append((c, None, e14.CAL_SHOTS))
                    index.append((len(sets) - 1, "cal"))
            print(f"n={n} set {s}: layout {layout}, two-qubit gates {[e14.two_qubit_gates(c) for c in isa]}", flush=True)

    counts, meta = runner.run(pubs, "E15-probe mlae screened qubits")
    out["run"] = meta
    out["skipped"] = skipped
    out["sets"] = []
    rng = np.random.default_rng(point_seed(15, 1))
    for i, st in enumerate(sets):
        reg = st["reg"]
        count_list = [counts[index.index((i, k))][0] for k in reg["powers"]]
        entry = analyse(reg, st["isa"], count_list, backend, rng)
        entry.update(set=st["set"], initial_layout=st["layout"])
        if not DRY:
            cal = [j for j, key in enumerate(index) if key == (i, "cal")]
            entry["readout"] = {"physical_qubits": st["calibrated"],
                                **e14.readout_errors(counts[cal[0]][0], counts[cal[1]][0], len(st["calibrated"]), e14.CAL_SHOTS)}
        out["sets"].append(entry)
        fit = entry.get("contrast_fit")
        head = f"n={reg['n']} set {st['set']} qubits {st['layout']}"
        if fit:
            a = entry["amplitude_estimates"]
            print(f"{head}: f_A={fit['f_A']:.3f} f_Q={fit['f_Q']:.3f} [{fit['f_Q_ci95'][0]:.3f}, {fit['f_Q_ci95'][1]:.3f}] "
                  f"fit p={fit['goodness_of_fit_p']:.3f}; amplitude true {a['true']:.4f} naive {a['noiseless_likelihood_mle']:.4f} "
                  f"noise-aware {a['noise_aware_mle']:.4f}±{a['noise_aware_bootstrap_sd']:.4f}")
        else:
            print(head)
        for r in entry["rows"]:
            print(f"   k={r['k']}: 2q={r['two_qubit_gates']:4d} P(1)={r['p_hat']:.3f} ideal {r['p_ideal']:.3f} "
                  f"signal fraction {r['signal_fraction']:.3f} [{r['signal_fraction_ci95'][0]:.3f}, {r['signal_fraction_ci95'][1]:.3f}]")
    out["wall_s"] = round(time.time() - t0, 1)
    save("e15_mlae_screened_qubits" + ("_dryrun" if DRY else ""), out)


if __name__ == "__main__":
    main()
