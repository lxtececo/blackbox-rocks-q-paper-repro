"""Sampling path for mission allocation: anneal, repair, fill, null (F7).

Every read and every null bitstring goes through raw, repaired and filled."""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from core.optimizers.mission.feasibility import check_plan, objective
from core.optimizers.mission.greedy import fill
from core.optimizers.mission.models import TOL, MissionProblem, Pair, Structure, build_structure, sort_pairs
from core.optimizers.mission.qubo import DEFAULT_PENALTY_FACTOR, MissionQubo, build_qubo, energy

RETURNED = "best filled plan over the reads"


class SamplerFailed(RuntimeError):
    """The annealer could not be run or returned unusable reads."""


def repair(problem: MissionProblem, pairs: Iterable[Pair], structure: Optional[Structure] = None) -> List[Pair]:
    """Make a plan feasible by dropping pairs only; deterministic."""
    st = structure or build_structure(problem)
    a_pos = {a.id: i for i, a in enumerate(problem.assets)}
    t_pos = {t.id: i for i, t in enumerate(problem.tasks)}
    start = {t.id: float(t.start_hour) for t in problem.tasks}
    hours = {t.id: float(t.duration_hours) for t in problem.tasks}

    plan = {(str(a), str(t)) for a, t in pairs}
    plan = sorted((p for p in plan if p in st.weight), key=lambda p: (a_pos[p[0]], t_pos[p[1]]))

    # Step 1: one asset per task, the largest weight, ties to the earlier asset.
    keep: Dict[str, Pair] = {}
    for pair in plan:
        current = keep.get(pair[1])
        if current is None or st.weight[pair] > st.weight[current]:
            keep[pair[1]] = pair
    held: Dict[str, List[str]] = {a.id: [] for a in problem.assets}
    for asset_id, task_id in keep.values():
        held[asset_id].append(task_id)

    for asset in problem.assets:
        mine = held[asset.id]
        near = st.conflicts[asset.id]

        def low_weight(task_id: str) -> Tuple[float, float, int]:
            return (st.weight[(asset.id, task_id)], -start[task_id], -t_pos[task_id])

        def low_weight_per_hour(task_id: str) -> Tuple[float, float, int]:
            rate = st.weight[(asset.id, task_id)] / hours[task_id] if hours[task_id] > 0 else math.inf
            return (rate, -start[task_id], -t_pos[task_id])

        # Step 2: among the tasks in conflict drop the smallest weight, ties to the later start.
        while True:
            clashing = [t for t in mine if any(u in near[t] for u in mine)]
            if not clashing:
                break
            mine.remove(min(clashing, key=low_weight))
        # Step 3: over the duty limit drop the smallest weight per hour.
        while math.fsum(hours[t] for t in mine) > asset.duty_limit_hours + TOL:
            mine.remove(min(mine, key=low_weight_per_hour))

    repaired = sort_pairs(problem, [(a, t) for a, mine in held.items() for t in mine])
    verdict = check_plan(problem, repaired)
    assert verdict["feasible"], f"repair left violations: {verdict['violations']}"
    return repaired


def stages(problem: MissionProblem, bits: Sequence[int], structure: Optional[Structure] = None) -> Dict[str, Any]:
    """One bitstring through raw, repaired and filled."""
    st = structure or build_structure(problem)
    if len(bits) != len(st.pairs):
        raise ValueError(f"expected {len(st.pairs)} bits, got {len(bits)}")
    raw = [pair for pair, bit in zip(st.pairs, bits) if int(bit)]
    raw_feasible = bool(check_plan(problem, raw)["feasible"])
    repaired = repair(problem, raw, st)
    filled = fill(problem, repaired, st)
    return {
        "raw_feasible": raw_feasible,
        "raw_objective": objective(problem, raw) if raw_feasible else None,
        "repaired_pairs": repaired,
        "repaired_objective": objective(problem, repaired),
        "filled_pairs": filled,
        "filled_objective": objective(problem, filled),
    }


def _annealer() -> Any:
    """The simulated annealer the logistics workspaces use in simulator mode."""
    try:
        from dwave.samplers import SimulatedAnnealingSampler
    except ImportError:
        from neal import SimulatedAnnealingSampler
    return SimulatedAnnealingSampler()


def _to_bqm(qubo: MissionQubo) -> Any:
    """The binary (0/1) model handed to the annealing library."""
    import dimod

    n = len(qubo.variables)
    linear = {i: float(qubo.terms.get((i, i), 0.0)) for i in range(n)}
    quadratic = {key: float(value) for key, value in qubo.terms.items() if key[0] != key[1]}
    return dimod.BinaryQuadraticModel(linear, quadratic, 0.0, dimod.BINARY)


def sample_qubo(qubo: MissionQubo, reads: int = 64, sweeps: int = 1000, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Anneal the QUBO; returns every read (reads x variables) and its energy."""
    n = len(qubo.variables)
    if n == 0:
        return np.zeros((int(reads), 0), dtype=np.int8), np.zeros(int(reads))
    try:
        bqm = _to_bqm(qubo)
        sampleset = _annealer().sample(bqm, num_reads=int(reads), num_sweeps=int(sweeps), seed=int(seed))
        column = {variable: k for k, variable in enumerate(sampleset.variables)}
        order = [column[i] for i in range(n)]
        bits = np.asarray(sampleset.record.sample)[:, order].astype(np.int8)
        energies = np.asarray(sampleset.record.energy, dtype=float)
    except Exception as exc:  # noqa: BLE001 - reported as a failure, never replaced
        raise SamplerFailed(f"simulated annealing failed: {type(exc).__name__}: {exc}") from exc
    if bits.shape != (int(reads), n):
        raise SamplerFailed(f"the annealer returned {bits.shape[0]} reads of {int(reads)} requested")
    return bits, energies


def _best_mean(values: List[float]) -> Dict[str, float]:
    return {"best": round(max(values), 3), "mean": round(math.fsum(values) / len(values), 3)}


def _distribution(sampled: List[float], null: List[float], bins: int = 12) -> Dict[str, Any]:
    """Counts of both sets of objectives over shared, equally wide bins."""
    low, high = min(sampled + null), max(sampled + null)
    if high - low < 1e-9:
        return {"bin_edges": [round(low, 3), round(high, 3)], "sampler": [len(sampled)], "null": [len(null)]}
    edges = np.linspace(low, high, bins + 1)
    return {
        "bin_edges": [round(float(e), 3) for e in edges],
        "sampler": [int(c) for c in np.histogram(sampled, bins=edges)[0]],
        "null": [int(c) for c in np.histogram(null, bins=edges)[0]],
    }


def solve_sampling(
    problem: MissionProblem,
    reads: int = 64,
    sweeps: int = 1000,
    seed: int = 0,
    penalty_factor: float = DEFAULT_PENALTY_FACTOR,
    qubo: Optional[MissionQubo] = None,
) -> Tuple[List[Pair], Dict[str, Any]]:
    """Sample the QUBO; return the best filled plan and the sampler block."""
    st = build_structure(problem)
    model = qubo or build_qubo(problem, penalty_factor, st)
    bits, _ = sample_qubo(model, reads=reads, sweeps=sweeps, seed=seed)
    sampled = [stages(problem, row, st) for row in bits]
    energies = [energy(model, row) for row in bits]
    feasible = [s["raw_objective"] for s in sampled if s["raw_feasible"]]

    # The null: uniform random bitstrings through the same stages.
    rng = np.random.default_rng(int(seed) + 1)
    null_bits = rng.integers(0, 2, size=(len(sampled), len(model.variables)))
    null = [stages(problem, row, st) for row in null_bits]

    filled = [s["filled_objective"] for s in sampled]
    null_filled = [s["filled_objective"] for s in null]
    best = max(range(len(sampled)), key=lambda r: (filled[r], -r))
    block = {
        "reads": int(reads),
        "sweeps": int(sweeps),
        "seed": int(seed),
        "raw": {
            "feasible_share": round(len(feasible) / len(sampled), 4),
            "best_feasible_objective": round(max(feasible), 3) if feasible else None,
            "lowest_energy": round(min(energies), 3),
        },
        "repaired": _best_mean([s["repaired_objective"] for s in sampled]),
        "filled": _best_mean(filled),
        "null": {
            "draws": len(null),
            "seed": int(seed) + 1,
            "repaired": _best_mean([s["repaired_objective"] for s in null]),
            "filled": _best_mean(null_filled),
        },
        "filled_distribution": _distribution(filled, null_filled),
        "returned": RETURNED,
    }
    return list(sampled[best]["filled_pairs"]), block
