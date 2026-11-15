"""Spaced site selection on a neutral-atom array (F34 Phase A).

Business question: open as many candidate sites as possible such that no two
open sites are closer than ``min_spacing_km`` (franchise cannibalisation,
charger spacing, tower interference). That is a maximum independent set (MIS)
on the unit-disk conflict graph of the sites.

Rydberg atoms solve it natively: atoms closer than the blockade radius cannot
both be excited, so an adiabatic sweep of the detuning drives the array towards
a maximum independent set of the graph the atom positions define (Pichler et
al. 2018; Ebadi et al., Science 2022). No QUBO and no penalty weights.

Runs on the Amazon Braket local AHS simulator, or on QuEra Aquila through
Braket. Every result is scored against the exact MIS.
"""

from __future__ import annotations

import logging
import math
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from core.optimizers.network.models import CandidateSite

logger = logging.getLogger(__name__)

#: Rb-87 70S van der Waals coefficient used by Aquila, rad·m⁶/s.
C6 = 5.42e-24
#: Aquila device limits (Braket device capabilities, 2024-2026).
OMEGA_MAX = 1.58e7          # rad/s
DELTA_MAX = 1.25e8          # rad/s
MAX_DURATION_S = 4.0e-6
FIELD_WIDTH_M = 75e-6
FIELD_HEIGHT_M = 76e-6
MIN_ATOM_SPACING_M = 4.0e-6
POSITION_RESOLUTION_M = 1e-7
#: Weakest final detuning we accept: below it the 4 µs sweep is not adiabatic.
DELTA_END_MIN = 2.5e6
#: Preferred cutoff radius; smaller means larger energy scales (more adiabatic),
#: bounded below by the detuning limit, (C6 / DELTA_MAX)^(1/6) ≈ 5.9 µm.
TARGET_CUTOFF_M = 7.0e-6

MAX_LOCAL_ATOMS = 12
MAX_AQUILA_ATOMS = 256
MAX_EXACT_SITES = 40
EARTH_RADIUS_KM = 6371.0088

AQUILA_ARN = "arn:aws:braket:us-east-1::device/qpu/quera/Aquila"

#: Seeded instance: ten Rhine-Ruhr cities as candidate hub sites.
RHINE_RUHR_10 = [
    ("C1", "Duisburg", 51.4344, 6.7623), ("C2", "Oberhausen", 51.4963, 6.8638),
    ("C3", "Essen", 51.4556, 7.0116), ("C4", "Bochum", 51.4818, 7.2162),
    ("C5", "Dortmund", 51.5136, 7.4653), ("C6", "Dusseldorf", 51.2277, 6.7735),
    ("C7", "Wuppertal", 51.2562, 7.1508), ("C8", "Solingen", 51.1652, 7.0671),
    ("C9", "Hagen", 51.3671, 7.4633), ("C10", "Gelsenkirchen", 51.5177, 7.0857),
]


def rhine_ruhr_problem(min_spacing_km: float = 15.0) -> "SpacingProblem":
    sites = [CandidateSite(id=i, name=n, lat=la, lon=lo) for i, n, la, lo in RHINE_RUHR_10]
    return SpacingProblem(sites, min_spacing_km)


class RydbergExecutionError(RuntimeError):
    """The neutral-atom run failed; never substituted by a classical answer."""


class EmbeddingError(ValueError):
    """The site geometry cannot be placed on the device faithfully."""


@dataclass
class SpacingProblem:
    sites: List[CandidateSite]
    min_spacing_km: float

    def __post_init__(self) -> None:
        if len(self.sites) < 2:
            raise ValueError("spaced site selection needs at least 2 candidate sites")
        if self.min_spacing_km <= 0:
            raise ValueError("min_spacing_km must be positive")


@dataclass
class ConflictGraph:
    coords_km: List[Tuple[float, float]]
    distances_km: List[List[float]]
    edges: List[Tuple[int, int]]
    adjacency: List[int]  # bitmask of neighbours per vertex

    @property
    def n(self) -> int:
        return len(self.coords_km)

    def degree(self, v: int) -> int:
        return bin(self.adjacency[v]).count("1")

    def is_independent(self, vertices: Sequence[int]) -> bool:
        mask = 0
        for v in vertices:
            mask |= 1 << v
        return all(not (self.adjacency[v] & mask) for v in vertices)


@dataclass
class Embedding:
    positions_m: List[Tuple[float, float]]
    scale_um_per_km: float
    threshold_km: float
    blockade_radius_m: float
    omega_max: float
    delta_max: float
    ambiguous_pairs: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


def _project(sites: Sequence[CandidateSite]) -> List[Tuple[float, float]]:
    """Equirectangular projection around the centroid, in km.

    Distances in this plane define both the business conflict graph and the
    atom geometry, so the two are the same graph by construction.
    """
    lat0 = math.radians(sum(s.lat for s in sites) / len(sites))
    lon0 = sum(s.lon for s in sites) / len(sites)
    lat_c = sum(s.lat for s in sites) / len(sites)
    k = math.pi / 180.0 * EARTH_RADIUS_KM
    return [
        ((s.lon - lon0) * k * math.cos(lat0), (s.lat - lat_c) * k)
        for s in sites
    ]


def conflict_graph(problem: SpacingProblem) -> ConflictGraph:
    coords = _project(problem.sites)
    n = len(coords)
    dist = [[0.0] * n for _ in range(n)]
    edges: List[Tuple[int, int]] = []
    adjacency = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            d = math.dist(coords[i], coords[j])
            dist[i][j] = dist[j][i] = d
            if d < problem.min_spacing_km:
                edges.append((i, j))
                adjacency[i] |= 1 << j
                adjacency[j] |= 1 << i
    return ConflictGraph(coords, dist, edges, adjacency)


def exact_mis(graph: ConflictGraph) -> List[int]:
    """Maximum independent set by branch and bound on bitmasks."""
    if graph.n > MAX_EXACT_SITES:
        raise ValueError(f"exact MIS is limited to {MAX_EXACT_SITES} sites")
    adj = graph.adjacency
    best = [0, 0]  # size, mask

    def popcount(x: int) -> int:
        return bin(x).count("1")

    def search(candidates: int, chosen: int, size: int) -> None:
        if size + popcount(candidates) <= best[0]:
            return
        if not candidates:
            best[0], best[1] = size, chosen
            return
        # Branch on the highest-degree candidate: include it, or drop it.
        v = max(
            (i for i in range(graph.n) if candidates >> i & 1),
            key=lambda i: popcount(adj[i] & candidates),
        )
        if not adj[v] & candidates:
            search(candidates & ~(1 << v), chosen | (1 << v), size + 1)
            return
        search(candidates & ~(1 << v) & ~adj[v], chosen | (1 << v), size + 1)
        search(candidates & ~(1 << v), chosen, size)

    search((1 << graph.n) - 1, 0, 0)
    return [i for i in range(graph.n) if best[1] >> i & 1]


def greedy_mis(graph: ConflictGraph, start: Sequence[int] = ()) -> List[int]:
    """Minimum-degree greedy, optionally extending an independent ``start``."""
    chosen = set(start)
    blocked = 0
    for v in chosen:
        blocked |= graph.adjacency[v] | (1 << v)
    remaining = [v for v in range(graph.n) if not blocked >> v & 1]
    while remaining:
        v = min(remaining, key=lambda i: (bin(graph.adjacency[i] & _mask(remaining)).count("1"), i))
        chosen.add(v)
        blocked |= graph.adjacency[v] | (1 << v)
        remaining = [u for u in remaining if not blocked >> u & 1]
    return sorted(chosen)


def _mask(vertices: Sequence[int]) -> int:
    m = 0
    for v in vertices:
        m |= 1 << v
    return m


def embed(graph: ConflictGraph, min_spacing_km: float, *, max_atoms: int) -> Embedding:
    """Scale the km plane onto the atom field so the interaction cutoff separates
    conflicting from non-conflicting pairs.

    With Ω ramped to zero, the end of the sweep favours exciting an atom unless
    a neighbour costs more than the detuning, so the effective cutoff is where
    C6 / r⁶ = Δ_end, not the Ω-defined blockade radius. Tuning on the seeded
    Rhine-Ruhr instance moved raw independent readouts from ~1% to ~100% with
    this rule and Ω = Δ_end / 2.

    The interaction is a soft 1/r⁶, so the cutoff is placed at the geometric
    mean of the longest conflicting and shortest non-conflicting distance — the
    widest gap available — and pairs close to it are reported as ambiguous.
    """
    n = graph.n
    if n > max_atoms:
        raise EmbeddingError(f"{n} sites exceed the {max_atoms}-atom limit for this back-end")

    pair_d = [graph.distances_km[i][j] for i in range(n) for j in range(i + 1, n)]
    d_min = min(pair_d)
    if d_min <= 0:
        raise EmbeddingError("two candidate sites share a location")
    edge_d = [d for d in pair_d if d < min_spacing_km]
    free_d = [d for d in pair_d if d >= min_spacing_km]
    if edge_d and free_d:
        threshold = math.sqrt(max(edge_d) * min(free_d))
    else:
        threshold = min_spacing_km

    target = max(TARGET_CUTOFF_M, (C6 / DELTA_MAX) ** (1.0 / 6.0))
    scale = max(target / threshold, MIN_ATOM_SPACING_M / d_min)  # m per km
    cutoff = threshold * scale
    delta_end = C6 / cutoff ** 6
    if delta_end < DELTA_END_MIN:
        raise EmbeddingError(
            "sites are too unevenly spaced to embed: the spacing rule is "
            f"{threshold / d_min:.1f}x the distance between the two closest sites, "
            "and the atom array supports about 2.8x. Drop near-duplicate sites or "
            "reduce the minimum spacing."
        )

    xs = [x for x, _ in graph.coords_km]
    ys = [y for _, y in graph.coords_km]
    width, height = (max(xs) - min(xs)) * scale, (max(ys) - min(ys)) * scale
    if width > FIELD_WIDTH_M or height > FIELD_HEIGHT_M:
        raise EmbeddingError(
            f"layout needs {width * 1e6:.0f} x {height * 1e6:.0f} µm but the atom field "
            f"is {FIELD_WIDTH_M * 1e6:.0f} x {FIELD_HEIGHT_M * 1e6:.0f} µm. The region is "
            "too large relative to the spacing; raise the spacing or split the region."
        )

    def snap(v: float) -> float:
        return round(v / POSITION_RESOLUTION_M) * POSITION_RESOLUTION_M

    positions = [
        (snap((x - min(xs)) * scale), snap((y - min(ys)) * scale))
        for x, y in graph.coords_km
    ]

    ambiguous = []
    for i in range(n):
        for j in range(i + 1, n):
            ratio = graph.distances_km[i][j] / threshold
            if 0.85 <= ratio <= 1.15:
                ambiguous.append({
                    "pair": [i, j],
                    "distance_km": round(graph.distances_km[i][j], 1),
                    "ratio_to_blockade": round(ratio, 3),
                    "conflict": graph.distances_km[i][j] < min_spacing_km,
                })
    warnings = []
    if ambiguous:
        warnings.append(
            f"{len(ambiguous)} site pair(s) sit within 15% of the blockade radius; the "
            "atoms may treat them differently from the spacing rule. Results are still "
            "scored against the exact answer for the spacing rule."
        )
    return Embedding(
        positions_m=positions,
        scale_um_per_km=scale * 1e6,
        threshold_km=threshold,
        blockade_radius_m=cutoff,
        omega_max=min(OMEGA_MAX, delta_end / 2.0),
        delta_max=delta_end,
        ambiguous_pairs=ambiguous,
        warnings=warnings,
    )


def build_program(embedding: Embedding, duration_s: float = MAX_DURATION_S):
    """Adiabatic MIS sweep: Ω ramps up and down, Δ sweeps negative to positive."""
    from braket.ahs.analog_hamiltonian_simulation import AnalogHamiltonianSimulation
    from braket.ahs.atom_arrangement import AtomArrangement
    from braket.ahs.driving_field import DrivingField
    from braket.timings.time_series import TimeSeries

    duration_s = min(duration_s, MAX_DURATION_S)
    ramp = 0.1 * duration_s
    register = AtomArrangement()
    for x, y in embedding.positions_m:
        register.add([x, y])

    def series(points):
        ts = TimeSeries()
        for t, v in points:
            ts.put(round(t, 9), v)
        return ts

    om, de = embedding.omega_max, embedding.delta_max
    amplitude = series([(0.0, 0.0), (ramp, om), (duration_s - ramp, om), (duration_s, 0.0)])
    detuning = series([(0.0, -de), (ramp, -de), (duration_s - ramp, de), (duration_s, de)])
    phase = series([(0.0, 0.0), (duration_s, 0.0)])
    drive = DrivingField(amplitude=amplitude, phase=phase, detuning=detuning)
    return AnalogHamiltonianSimulation(register=register, hamiltonian=drive)


def run_local(program, *, shots: int, steps: int = 80) -> List[Tuple[List[int], List[int]]]:
    from braket.devices import LocalSimulator

    result = LocalSimulator("braket_ahs").run(program, shots=shots, steps=steps).result()
    return [
        (list(map(int, m.pre_sequence)), list(map(int, m.post_sequence)))
        for m in result.measurements
        if str(m.status).lower().endswith("success")
    ]


def _repair(graph: ConflictGraph, excited: List[int]) -> List[int]:
    """Drop the most-conflicted vertex until independent, then extend greedily."""
    chosen = set(excited)
    while True:
        mask = _mask(list(chosen))
        worst = max(
            chosen,
            key=lambda v: (bin(graph.adjacency[v] & mask).count("1"), graph.degree(v), v),
            default=None,
        )
        if worst is None or not graph.adjacency[worst] & mask:
            break
        chosen.discard(worst)
    return greedy_mis(graph, sorted(chosen))


def decode(graph: ConflictGraph, shots: List[Tuple[List[int], List[int]]], optimum: int) -> Dict[str, Any]:
    """Read Rydberg excitations (post-sequence 0 after a filled pre-sequence)."""
    kept = [post for pre, post in shots if all(b == 1 for b in pre)]
    if not kept:
        raise RydbergExecutionError("no shot had all atoms loaded; nothing to decode")
    raw_sets = [[i for i, b in enumerate(post) if b == 0] for post in kept]
    independent = [s for s in raw_sets if graph.is_independent(s)]
    repaired = [_repair(graph, s) for s in raw_sets]
    best = max(repaired, key=len)
    counts = Counter("".join("1" if b == 0 else "0" for b in post) for post in kept)
    return {
        "shots_used": len(kept),
        "shots_discarded": len(shots) - len(kept),
        "raw_independent_fraction": round(len(independent) / len(kept), 4),
        "raw_optimal_fraction": round(
            sum(1 for s in independent if len(s) == optimum) / len(kept), 4
        ),
        "raw_mean_size": round(sum(len(s) for s in raw_sets) / len(kept), 3),
        "best_raw_independent_size": max((len(s) for s in independent), default=0),
        "best_set": best,
        "repair_gain": len(best) - max((len(s) for s in independent), default=0),
        "top_bitstrings": [
            {"bitstring": b, "count": c} for b, c in counts.most_common(8)
        ],
    }


def solve_spacing(
    problem: SpacingProblem,
    *,
    method: str = "rydberg-mis",
    shots: int = 100,
    steps: int = 80,
    backend: str = "local",
) -> Dict[str, Any]:
    """Solve and score. ``method`` is ``classical-mis`` or ``rydberg-mis``."""
    started = time.perf_counter()
    graph = conflict_graph(problem)
    optimum_set = exact_mis(graph)
    optimum = len(optimum_set)
    greedy_set = greedy_mis(graph)
    site_ids = [s.id for s in problem.sites]

    result: Dict[str, Any] = {
        "method": method,
        "n_sites": graph.n,
        "n_conflicts": len(graph.edges),
        "min_spacing_km": problem.min_spacing_km,
        "exact": {"size": optimum, "site_ids": [site_ids[i] for i in optimum_set]},
        "greedy": {"size": len(greedy_set), "site_ids": [site_ids[i] for i in greedy_set]},
        "conflicts": [[site_ids[i], site_ids[j]] for i, j in graph.edges],
        "caveats": [],
    }

    if method == "classical-mis":
        chosen = optimum_set
        result["engine"] = "branch-and-bound"
    elif method == "rydberg-mis":
        if backend != "local":
            raise ValueError("only the local AHS simulator is wired in Phase A")
        embedding = embed(graph, problem.min_spacing_km, max_atoms=MAX_LOCAL_ATOMS)
        program = build_program(embedding)
        try:
            shot_rows = run_local(program, shots=shots, steps=steps)
        except Exception as exc:  # noqa: BLE001 - surfaced, never replaced
            logger.warning("[rydberg-mis] local AHS simulation failed: %s", exc)
            raise RydbergExecutionError(f"neutral-atom simulation failed: {exc}") from exc
        decoded = decode(graph, shot_rows, optimum)
        chosen = decoded.pop("best_set")
        result["engine"] = "braket-ahs-local"
        result["quantum"] = decoded
        result["embedding"] = {
            "scale_um_per_km": round(embedding.scale_um_per_km, 4),
            "threshold_km": round(embedding.threshold_km, 2),
            "interaction_cutoff_um": round(embedding.blockade_radius_m * 1e6, 2),
            "omega_max_rad_s": round(embedding.omega_max, 1),
            "delta_max_rad_s": round(embedding.delta_max, 1),
            "positions_um": [
                [round(x * 1e6, 1), round(y * 1e6, 1)] for x, y in embedding.positions_m
            ],
            "ambiguous_pairs": embedding.ambiguous_pairs,
        }
        result["caveats"].extend(embedding.warnings)
        if decoded["repair_gain"] > 0:
            result["caveats"].append(
                "The best plan needed classical repair: raw atom readouts were extended "
                f"by {decoded['repair_gain']} site(s). Raw quality is reported separately."
            )
    else:
        raise ValueError(f"unknown method '{method}'")

    if not graph.is_independent(chosen):
        raise RydbergExecutionError("decoded plan violates the spacing rule")
    result["solution"] = {
        "size": len(chosen),
        "site_ids": [site_ids[i] for i in chosen],
        "approximation_ratio": round(len(chosen) / optimum, 4) if optimum else 1.0,
        "is_optimal": len(chosen) == optimum,
    }
    result["wall_s"] = round(time.perf_counter() - started, 3)
    return result
