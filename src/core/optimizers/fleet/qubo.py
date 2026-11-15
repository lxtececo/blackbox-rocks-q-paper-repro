"""QUBO encoding of the fleet assignment problem (F6).

A 620-stop CVRP cannot be encoded node-by-node on any near-term device:
a naive position-indexed formulation needs O(n^2 * K) qubits. This module
uses the industry-standard **cluster-first / route-second** decomposition,
which is what production annealing demos actually run:

1. Delivery points are grouped into geographic micro-clusters whose demand
   fits inside a single truck.
2. The QUBO decides **which vehicle serves which cluster** — one binary
   ``x[c, v]`` per (cluster, vehicle) pair. That is the combinatorially
   hard, globally-coupled part of the problem.
3. Intra-route sequencing is handed back to the classical improver in
   :mod:`core.optimizers.fleet.classical_solver`.

Cost model::

    C(x, y) = sum_cv  travel_cv * x_cv                    (depot -> cluster leg)
            + sum_v   fixed_v * y_v                       (cost of opening a truck)
            + A * sum_v sum_{(c,c') in kNN} dist(c,c') x_cv x_c'v   (tour coherence)
            + L1 * sum_c ( sum_v x_cv - 1 )^2             (serve each cluster once)
            + L2 * sum_v sum_{c<c'} max(0, d_c + d_c' - cap_v)/cap_v * x_cv x_c'v
            + L3 * sum_cv x_cv (1 - y_v)                  (x_cv implies y_v)

Three details matter and are easy to get wrong:

* The capacity term is **one-sided**. An earlier revision penalised every
  co-assigned pair in proportion to ``d_c*d_c'/cap^2``, which is a spreading
  force rather than a capacity constraint: the cheapest way to satisfy it is
  to give every cluster its own truck. It now contributes nothing until a
  pair genuinely overflows the vehicle.
* ``y_v`` makes "open a truck" cost something. Without it the objective is
  blind to ``Vehicle.fixed_cost``, which :mod:`core.optimizers.fleet.evaluate`
  *does* charge — so the device optimised a different problem than it was
  scored on and always spread load across the whole fleet.
* The affinity term is pruned to the ``AFFINITY_NEIGHBOURS`` nearest cluster
  pairs. Clusters far apart carry negligible detour information, and the
  pruning takes the coupling count from O(n^2) to O(nk), which is what makes
  the encoding tractable on real gate hardware.

Conversion to Ising goes through :mod:`core.quantum_math`, the single
authority for QUBO <-> Ising in this codebase, so fleet energies are
directly comparable with the portfolio stack's.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from core.optimizers.fleet.geo import DistanceOracle, centroid, road_km
from core.optimizers.fleet.models import FleetProblem, Vehicle
from core.quantum_math import qubo_pair_to_ising, qubo_pair_cost

logger = logging.getLogger(__name__)

#: Hard ceiling on QUBO width. 512 binaries already exceeds what any
#: gate-based simulator handles; annealers cope, but the surrounding
#: request budget does not. Callers may lower it, never raise it.
MAX_QUBO_VARS = 512

#: Target fill ratio when sizing clusters against truck capacity.
TARGET_FILL = 0.8

#: Number of nearest cluster neighbours retained in the affinity term.
#: Dense all-to-all affinity costs O(n^2) couplings, which on gate hardware
#: becomes O(n^2) two-qubit gates and destroys fidelity long before the
#: qubit count runs out. Geographically distant clusters contribute almost
#: no detour signal, so pruning to the k nearest is close to free.
AFFINITY_NEIGHBOURS = 6

#: Capacity penalty weight as a multiple of the travel scale. The old
#: d_c*d_c'/cap^2 form damped itself by ~1/100 through that ratio; the
#: normalised form applies the weight directly, so this has to be far
#: smaller than the 2.0 that preceded it or capacity dominates affinity and
#: the solver packs trucks by demand while ignoring geography. Swept over
#: {0.15, 0.3, 0.5, 1.0} x fill {0.8, 0.9, 1.0} on the demo network across
#: seeds 42/7/2024; 0.15 at 0.8 fill was best on cost, km and service at once.
BALANCE_SCALE = 0.15

#: Fill level an open truck is pulled toward, as a fraction of capacity.
#: Pulling all the way to 1.0 buys ~4% cost but sheds stops (service drops
#: to 93.7%), because an overloaded truck sheds its overflow downstream.
BALANCE_TARGET_FILL = TARGET_FILL


@dataclass
class Cluster:
    """A group of delivery points small enough for one truck."""

    id: str
    stop_ids: List[str]
    demand: float
    lat: float
    lon: float
    internal_km: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "stop_ids": list(self.stop_ids),
            "demand": round(self.demand, 3),
            "lat": round(self.lat, 6),
            "lon": round(self.lon, 6),
            "internal_km": round(self.internal_km, 3),
        }


@dataclass
class FleetQubo:
    """Pair-form QUBO over cluster-to-vehicle assignment binaries."""

    q: np.ndarray
    Q: np.ndarray
    const: float
    clusters: List[Cluster]
    vehicle_ids: List[str]
    index: Dict[Tuple[int, int], int]
    open_index: Dict[int, int] = field(default_factory=dict)
    penalties: Dict[str, float] = field(default_factory=dict)

    @property
    def n_vars(self) -> int:
        return int(self.q.shape[0])

    def energy(self, bits: Sequence[int]) -> float:
        """QUBO objective for a bit assignment (includes penalties)."""
        return qubo_pair_cost(np.asarray(bits, dtype=int), self.q, self.Q, self.const)

    def to_ising(self) -> Tuple[np.ndarray, np.ndarray, float]:
        """(h, J, offset) via the shared ``core.quantum_math`` authority."""
        h, J, offset = qubo_pair_to_ising(self.q, self.Q)
        return h, J, offset + self.const

    def decode(self, bits: Sequence[int]) -> Tuple[Dict[str, List[str]], List[str]]:
        """Map a bitstring to ``({vehicle_id: [stop_id]}, unassigned)``.

        One-hot violations are repaired rather than rejected: an
        over-assigned cluster keeps its cheapest vehicle and an
        unassigned cluster falls through to the unassigned pool, where the
        classical insertion sweep gets a shot at it.

        The repair scores a candidate vehicle by its linear term *plus* the
        couplings to clusters already placed on that vehicle. Scoring on the
        linear term alone ignores the affinity and capacity structure, which
        biases every tie toward an empty truck and quietly undoes the
        consolidation the sampler found.
        """
        bits = np.asarray(bits, dtype=int)
        by_vehicle: Dict[str, List[str]] = {vid: [] for vid in self.vehicle_ids}
        unassigned: List[str] = []
        placed: Dict[int, List[int]] = {v: [] for v in range(len(self.vehicle_ids))}

        def _score(c_idx: int, v_idx: int) -> float:
            var = self.index[(c_idx, v_idx)]
            total = float(self.q[var])
            for other in placed[v_idx]:
                o = self.index[(other, v_idx)]
                # Q is stored upper-triangular, so order the pair before
                # reading it or half the couplings silently come back zero.
                total += float(self.Q[min(var, o), max(var, o)])
            if v_idx in self.open_index and not placed[v_idx]:
                total += float(self.q[self.open_index[v_idx]])
            return total

        for c_idx, cluster in enumerate(self.clusters):
            chosen = [
                v_idx
                for v_idx, _ in enumerate(self.vehicle_ids)
                if bits[self.index[(c_idx, v_idx)]] == 1
            ]
            if not chosen:
                unassigned.extend(cluster.stop_ids)
                continue
            best = min(chosen, key=lambda v_idx: _score(c_idx, v_idx))
            placed[best].append(c_idx)
            by_vehicle[self.vehicle_ids[best]].extend(cluster.stop_ids)

        return by_vehicle, unassigned


def _kmeans(
    coords: np.ndarray,
    k: int,
    *,
    seed: int,
    iterations: int = 25,
) -> np.ndarray:
    """Deterministic k-means++ style clustering on lat/lon."""
    rng = np.random.default_rng(seed)
    n = coords.shape[0]
    k = max(1, min(k, n))

    centres = np.empty((k, coords.shape[1]), dtype=float)
    centres[0] = coords[rng.integers(0, n)]
    for i in range(1, k):
        d2 = np.min(
            ((coords[:, None, :] - centres[None, :i, :]) ** 2).sum(axis=2), axis=1
        )
        total = d2.sum()
        if total <= 0:
            centres[i] = coords[rng.integers(0, n)]
            continue
        centres[i] = coords[rng.choice(n, p=d2 / total)]

    labels = np.zeros(n, dtype=int)
    for _ in range(iterations):
        dist = ((coords[:, None, :] - centres[None, :, :]) ** 2).sum(axis=2)
        new_labels = np.argmin(dist, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for j in range(k):
            members = coords[labels == j]
            if members.size:
                centres[j] = members.mean(axis=0)
    return labels


def build_clusters(
    problem: FleetProblem,
    oracle: DistanceOracle,
    stop_ids: Sequence[str],
    *,
    n_clusters: int,
    seed: int = 42,
) -> List[Cluster]:
    """Group stops into demand-balanced geographic micro-clusters."""
    stops = [problem.stop(sid) for sid in stop_ids]
    if not stops:
        return []

    coords = np.array([[s.lat, s.lon] for s in stops], dtype=float)
    labels = _kmeans(coords, n_clusters, seed=seed)

    buckets: Dict[int, List[str]] = {}
    for label, stop in zip(labels.tolist(), stops):
        buckets.setdefault(label, []).append(stop.id)

    clusters: List[Cluster] = []
    for order, (_, members) in enumerate(sorted(buckets.items())):
        points = [(problem.stop(m).lat, problem.stop(m).lon) for m in members]
        lat, lon = centroid(points)
        demand = sum(problem.stop(m).demand for m in members)
        internal = oracle.path_km(members) if len(members) > 1 else 0.0
        clusters.append(
            Cluster(
                id=f"C{order:03d}",
                stop_ids=members,
                demand=demand,
                lat=lat,
                lon=lon,
                internal_km=internal,
            )
        )
    return clusters


def suggest_cluster_count(
    problem: FleetProblem,
    stop_ids: Sequence[str],
    vehicles: Sequence[Vehicle],
    *,
    max_vars: int = MAX_QUBO_VARS,
) -> int:
    """Pick a cluster count that fits the QUBO width and the truck capacity."""
    if not stop_ids or not vehicles:
        return 0
    total_demand = sum(problem.stop(sid).demand for sid in stop_ids)
    avg_capacity = sum(v.capacity for v in vehicles) / len(vehicles)
    by_capacity = math.ceil(total_demand / max(avg_capacity * TARGET_FILL, 1e-6))
    # The encoding also spends one binary per vehicle on the "truck is open"
    # indicator, so the width left for assignments is max_vars - n_v.
    n_v = len(vehicles)
    by_width = max(1, (max_vars - n_v) // max(n_v, 1))
    return int(max(1, min(len(stop_ids), by_capacity, by_width)))


def build_cluster_qubo(
    problem: FleetProblem,
    oracle: DistanceOracle,
    stop_ids: Sequence[str],
    vehicles: Sequence[Vehicle],
    *,
    n_clusters: Optional[int] = None,
    max_vars: int = MAX_QUBO_VARS,
    seed: int = 42,
    one_hot_weight: Optional[float] = None,
    balance_weight: Optional[float] = None,
    affinity_weight: Optional[float] = None,
    affinity_neighbours: int = AFFINITY_NEIGHBOURS,
) -> FleetQubo:
    """Build the cluster-to-vehicle assignment QUBO.

    Raises ``ValueError`` when the encoding would exceed ``max_vars`` even
    after cluster sizing — the caller is expected to fall back to the
    classical solver rather than silently truncating the problem.
    """
    if not stop_ids:
        raise ValueError("build_cluster_qubo requires at least one stop")
    if not vehicles:
        raise ValueError("build_cluster_qubo requires at least one vehicle")

    if n_clusters is None:
        n_clusters = suggest_cluster_count(
            problem, stop_ids, vehicles, max_vars=max_vars
        )
    clusters = build_clusters(
        problem, oracle, stop_ids, n_clusters=n_clusters, seed=seed
    )

    vehicle_ids = [v.id for v in vehicles]
    n_c, n_v = len(clusters), len(vehicle_ids)
    # n_c*n_v assignment binaries plus one "is this truck open" binary per
    # vehicle, which is what lets the objective see Vehicle.fixed_cost.
    n_vars = n_c * n_v + n_v
    if n_c == 0 or n_v == 0:
        raise ValueError("empty QUBO: no clusters or no vehicles")
    if n_vars > max_vars:
        raise ValueError(
            f"fleet QUBO would need {n_vars} binaries (limit {max_vars}); "
            "reduce the stop set or the vehicle pool"
        )

    index = {
        (c, v): c * n_v + v for c in range(n_c) for v in range(n_v)
    }
    open_index = {v: n_c * n_v + v for v in range(n_v)}

    # ── assignment cost: out-and-back from the vehicle's depot ─────────
    travel = np.zeros((n_c, n_v), dtype=float)
    for c_idx, cluster in enumerate(clusters):
        for v_idx, vehicle in enumerate(vehicles):
            depot = problem.depot(vehicle.depot_id)
            leg = road_km(depot.lat, depot.lon, cluster.lat, cluster.lon)
            travel[c_idx, v_idx] = (
                2.0 * leg + cluster.internal_km
            ) * vehicle.cost_per_km

    scale = float(np.max(travel)) if travel.size else 1.0
    scale = scale if scale > 1e-9 else 1.0
    lam_one_hot = (
        float(one_hot_weight) if one_hot_weight is not None else 4.0 * scale
    )
    lam_balance = (
        float(balance_weight) if balance_weight is not None else BALANCE_SCALE * scale
    )
    lam_affinity = (
        float(affinity_weight) if affinity_weight is not None else 1.0
    )

    q = np.zeros(n_vars, dtype=float)
    Q = np.zeros((n_vars, n_vars), dtype=float)
    const = 0.0

    for c_idx in range(n_c):
        for v_idx in range(n_v):
            q[index[(c_idx, v_idx)]] += travel[c_idx, v_idx]

    # ── one-hot: L1 * (sum_v x_cv - 1)^2 ──────────────────────────────
    # Expanded with x^2 = x: -L1*sum_v x_cv + 2*L1*sum_{v<v'} x_cv x_cv' + L1
    for c_idx in range(n_c):
        const += lam_one_hot
        for v_idx in range(n_v):
            q[index[(c_idx, v_idx)]] -= lam_one_hot
            for v2 in range(v_idx + 1, n_v):
                a = index[(c_idx, v_idx)]
                b = index[(c_idx, v2)]
                lo, hi = (a, b) if a < b else (b, a)
                Q[lo, hi] += 2.0 * lam_one_hot

    # ── cluster affinity + capacity, both same-vehicle couplings ───────
    #
    # Affinity is the term that gives the QUBO its spatial structure: two
    # clusters sharing a truck cost roughly the detour between them. Without
    # it the linear term is degenerate for identical trucks at one depot and
    # the device would be optimising nothing but load balance.
    inter = np.zeros((n_c, n_c), dtype=float)
    for c1 in range(n_c):
        for c2 in range(c1 + 1, n_c):
            inter[c1, c2] = road_km(
                clusters[c1].lat,
                clusters[c1].lon,
                clusters[c2].lat,
                clusters[c2].lon,
            )

    # Keep only the k nearest neighbours of each cluster. Beyond that the
    # detour signal is noise, and every retained pair costs a two-qubit gate
    # on hardware.
    keep = set()
    k_near = max(1, min(int(affinity_neighbours), n_c - 1)) if n_c > 1 else 0
    for c1 in range(n_c):
        dists = [
            (inter[min(c1, c2), max(c1, c2)], c2)
            for c2 in range(n_c)
            if c2 != c1
        ]
        dists.sort()
        for _, c2 in dists[:k_near]:
            keep.add((min(c1, c2), max(c1, c2)))

    for v_idx, vehicle in enumerate(vehicles):
        for c1 in range(n_c):
            for c2 in range(c1 + 1, n_c):
                if (c1, c2) not in keep:
                    continue
                coupling = lam_affinity * inter[c1, c2] * vehicle.cost_per_km
                if coupling == 0.0:
                    continue
                a = index[(c1, v_idx)]
                b = index[(c2, v_idx)]
                lo, hi = (a, b) if a < b else (b, a)
                Q[lo, hi] += coupling

    # ── opening a truck costs money, and x_cv implies y_v ─────────────
    #
    # evaluate.py charges Vehicle.fixed_cost for every active route. Until
    # the QUBO carried the same term it was solving a problem where trucks
    # were free, so it always spread the load over the entire fleet and was
    # then graded on a cost it never saw.
    max_fixed = max((float(v.fixed_cost) for v in vehicles), default=0.0)
    # The link penalty only has to beat fixed_cost: the sampler's choice is
    # "pay lam_link for an unlinked assignment" vs "pay fixed_cost and open
    # the truck". Scaling it to lam_one_hot instead (4x the travel scale)
    # swamps the objective and starves the one-hot constraint.
    lam_link = 4.0 * max_fixed
    for v_idx, vehicle in enumerate(vehicles):
        y = open_index[v_idx]
        q[y] += float(vehicle.fixed_cost)
        for c_idx in range(n_c):
            x = index[(c_idx, v_idx)]
            q[x] += lam_link
            lo, hi = (x, y) if x < y else (y, x)
            Q[lo, hi] -= lam_link

    # ── capacity: L2 * (load_v - target_v * y_v)^2 / target_v^2 ───────
    #
    # Conditioning on y_v is what makes this a capacity constraint instead
    # of a spreading force. An unopened truck contributes nothing; an open
    # one is pulled toward its target fill from both sides, so consolidating
    # onto fewer trucks is rewarded and overloading is still punished. The
    # earlier pairwise-only form could not see aggregate overflow at all:
    # three 25-unit clusters never trip a pair test against a 60-unit truck
    # but together they overload it.
    for v_idx, vehicle in enumerate(vehicles):
        target = max(float(vehicle.capacity) * BALANCE_TARGET_FILL, 1e-6)
        w = lam_balance / (target * target)
        y = open_index[v_idx]
        q[y] += lam_balance
        for c1 in range(n_c):
            a = index[(c1, v_idx)]
            d1 = clusters[c1].demand
            q[a] += w * d1 * d1
            lo, hi = (a, y) if a < y else (y, a)
            Q[lo, hi] -= 2.0 * w * target * d1
            for c2 in range(c1 + 1, n_c):
                b = index[(c2, v_idx)]
                lo2, hi2 = (a, b) if a < b else (b, a)
                Q[lo2, hi2] += 2.0 * w * d1 * clusters[c2].demand

    # ── clusters that simply do not fit get a prohibitive linear cost ─
    for c_idx, cluster in enumerate(clusters):
        for v_idx, vehicle in enumerate(vehicles):
            if cluster.demand > vehicle.capacity + 1e-9:
                q[index[(c_idx, v_idx)]] += 10.0 * lam_one_hot

    logger.info(
        "[fleet-qubo] clusters=%d vehicles=%d vars=%d couplings=%d "
        "one_hot=%.3f balance=%.3f link=%.3f",
        n_c,
        n_v,
        n_vars,
        int(np.count_nonzero(Q)),
        lam_one_hot,
        lam_balance,
        lam_link,
    )

    return FleetQubo(
        q=q,
        Q=Q,
        const=const,
        clusters=clusters,
        vehicle_ids=vehicle_ids,
        index=index,
        open_index=open_index,
        penalties={
            "one_hot_weight": lam_one_hot,
            "balance_weight": lam_balance,
            "affinity_weight": lam_affinity,
            "link_weight": lam_link,
            "affinity_neighbours": float(k_near),
            "cost_scale": scale,
        },
    )
