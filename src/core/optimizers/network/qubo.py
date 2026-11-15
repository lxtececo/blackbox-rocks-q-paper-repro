"""Capacitated facility-location QUBO (F32).

Encodes the network-design decision directly as binaries, with no clustering
or decomposition pass: unlike the fleet CVRP (F6), the CFLP decision variables
*are* the answer, so the sampler's output decodes straight into a network.

Variable layout (one flat index space so the pair-form arrays stay dense):

    y_j              site j is open                    n_sites vars
    x_ij             demand point i served by site j   n_demand * n_sites vars

Objective and constraint penalties follow
``reports/architecture/enhancements/F32_facility_location_network_design.md``.
Ising conversion is delegated to :mod:`core.quantum_math` so the F29
correctness lock covers this vertical too.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from core.optimizers.fleet.geo import road_km
from core.optimizers.network.models import NetworkProblem
from core.quantum_math import qubo_pair_cost, qubo_pair_to_ising


@dataclass
class NetworkQubo:
    """Pair-form QUBO over (site-open, demand-assignment) binaries.

    Exposes exactly the surface ``fleet.samplers.sample_qubo`` consumes
    (``q``, ``Q``, ``const``, ``n_vars``, ``to_ising``), which is why this
    vertical needs no sampler module of its own.
    """

    q: np.ndarray
    Q: np.ndarray
    const: float
    site_ids: List[str]
    demand_ids: List[str]
    #: (demand_index, site_index) -> flat variable index
    assign_index: Dict[Tuple[int, int], int] = field(default_factory=dict)
    #: site_index -> flat variable index
    open_index: Dict[int, int] = field(default_factory=dict)
    penalties: Dict[str, float] = field(default_factory=dict)
    cost_matrix: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))

    @property
    def n_vars(self) -> int:
        return int(self.q.shape[0])

    def energy(self, bits: Sequence[int]) -> float:
        return qubo_pair_cost(np.asarray(bits, dtype=int), self.q, self.Q, self.const)

    def to_ising(self) -> Tuple[np.ndarray, np.ndarray, float]:
        h, J, offset = qubo_pair_to_ising(self.q, self.Q)
        return h, J, offset + self.const

    def decode(self, bits: Sequence[int]) -> Tuple[Dict[str, List[str]], List[str]]:
        """Map a bitstring to ``({site_id: [demand_id]}, unassigned)``.

        A demand point whose row is all-zero, or whose chosen site is closed,
        is reported as unassigned rather than silently reassigned: the caller
        needs to see that the penalty weights failed to enforce the one-hot.
        """
        arr = np.asarray(bits, dtype=int)
        opened = {
            s_idx for s_idx, v_idx in self.open_index.items() if arr[v_idx] == 1
        }
        assignment: Dict[str, List[str]] = {
            self.site_ids[s]: [] for s in sorted(opened)
        }
        unassigned: List[str] = []

        for d_idx, demand_id in enumerate(self.demand_ids):
            chosen = [
                s_idx
                for s_idx in range(len(self.site_ids))
                if arr[self.assign_index[(d_idx, s_idx)]] == 1
            ]
            # Exactly one open site required; anything else is a violation.
            valid = [s for s in chosen if s in opened]
            if len(valid) != 1:
                unassigned.append(demand_id)
                continue
            assignment[self.site_ids[valid[0]]].append(demand_id)

        return assignment, unassigned


def _slack_weights(m: int) -> List[int]:
    """Binary weights that represent every integer in ``[0, m]`` exactly once.

    Plain powers of two would over-shoot (``[1,2,4]`` reaches 7, not 5), which
    would let the slack absorb more than the constraint allows and silently
    widen the bound.
    """
    if m <= 0:
        return []
    weights: List[int] = []
    k = 1
    while (2 * k - 1) <= m:
        weights.append(k)
        k *= 2
    remainder = m - (sum(weights))
    if remainder > 0:
        weights.append(remainder)
    return weights


def build_cost_matrix(problem: NetworkProblem) -> np.ndarray:
    """``cost[i][j]`` = transport cost of serving demand point i from site j."""
    n_demand = len(problem.demand_points)
    n_sites = len(problem.sites)
    cost = np.zeros((n_demand, n_sites), dtype=float)
    for i, dp in enumerate(problem.demand_points):
        for j, site in enumerate(problem.sites):
            km = road_km(dp.lat, dp.lon, site.lat, site.lon, problem.detour_factor)
            cost[i, j] = km * dp.demand * problem.cost_per_km_per_unit
    return cost


def build_network_qubo(
    problem: NetworkProblem,
    *,
    assign_penalty: float | None = None,
    link_penalty: float | None = None,
) -> NetworkQubo:
    """Build the pair-form CFLP QUBO.

    Penalty weights default to values *derived from the instance's own cost
    magnitudes*. Hard-coding them is the failure mode F6 hit with its fixed
    annealing beta range: a weight that is sane for one instance silently
    stops binding on another whose costs are an order of magnitude larger.
    """
    sites = problem.sites
    demand_points = problem.demand_points
    n_sites = len(sites)
    n_demand = len(demand_points)

    cost = build_cost_matrix(problem)

    # Reference scale: the worst single assignment plus the priciest site.
    max_transport = float(cost.max()) if cost.size else 1.0
    max_fixed = max((float(s.fixed_cost) for s in sites), default=1.0)
    scale = max(max_transport, max_fixed, 1.0)

    # Penalties are sized against *what a violation could actually save*, not
    # against the global cost scale. Oversized penalties are not "safe": they
    # dominate the coefficient magnitudes the annealer derives its beta
    # schedule from, so real differences between feasible networks (tens of
    # thousands) become noise next to the penalty scale (hundreds of
    # thousands) and the sampler returns a feasible-but-mediocre answer.
    #
    #   one-hot: dropping a demand point saves its transport cost AND can
    #            save a whole site's fixed cost if it was the last customer.
    #   linking: serving from a closed site saves that site's fixed cost.
    #   site cap: opening past the cap saves at most one site's fixed cost.
    margin = 1.5
    lam_assign = float(
        assign_penalty
        if assign_penalty is not None
        else margin * (max_fixed + max_transport) + 1.0
    )
    lam_link = float(
        link_penalty if link_penalty is not None else margin * max_fixed + 1.0
    )
    # The site-count cap is structural, not elastic: it must outweigh the
    # fixed cost of any site the sampler might be tempted to open anyway.
    lam_site_cap = margin * max_fixed + 1.0

    open_index: Dict[int, int] = {}
    assign_index: Dict[Tuple[int, int], int] = {}
    cursor = 0
    for j in range(n_sites):
        open_index[j] = cursor
        cursor += 1
    for i in range(n_demand):
        for j in range(n_sites):
            assign_index[(i, j)] = cursor
            cursor += 1

    # Slack binaries for the `max_open_sites` upper bound are part of the
    # layout, so allocate them before the arrays are sized.
    slack_weights = (
        _slack_weights(int(problem.max_open_sites))
        if problem.max_open_sites is not None
        else []
    )
    slack_index: Dict[int, int] = {}
    for k in range(len(slack_weights)):
        slack_index[k] = cursor
        cursor += 1

    n_vars = cursor
    q = np.zeros(n_vars, dtype=float)
    Q = np.zeros((n_vars, n_vars), dtype=float)
    const = 0.0

    def add_pair(a: int, b: int, coeff: float) -> None:
        """Accumulate into the upper triangle, which is what pair-form reads."""
        if a == b:
            q[a] += coeff
            return
        lo, hi = (a, b) if a < b else (b, a)
        Q[lo, hi] += coeff

    # --- Objective: fixed opening cost -----------------------------------
    for j, site in enumerate(sites):
        vj = open_index[j]
        if site.must_open:
            # Pinned open: fix the decision by making closure prohibitive
            # rather than removing the variable, so the index layout stays
            # uniform and decode() needs no special case.
            q[vj] -= lam_assign
        q[vj] += float(site.fixed_cost)

    # --- Objective: transport cost ---------------------------------------
    for i in range(n_demand):
        for j in range(n_sites):
            q[assign_index[(i, j)]] += float(cost[i, j])

    # --- Constraint: each demand point served exactly once ----------------
    # lam * (sum_j x_ij - 1)^2 = lam * (-sum_j x_ij + 2*sum_{j<k} x_ij x_ik + 1)
    for i in range(n_demand):
        const += lam_assign
        for j in range(n_sites):
            add_pair(assign_index[(i, j)], assign_index[(i, j)], -lam_assign)
            for k in range(j + 1, n_sites):
                add_pair(assign_index[(i, j)], assign_index[(i, k)], 2.0 * lam_assign)

    # --- Constraint: cannot serve from a closed site ----------------------
    # lam * x_ij * (1 - y_j) = lam * x_ij - lam * x_ij * y_j
    for i in range(n_demand):
        for j in range(n_sites):
            vij = assign_index[(i, j)]
            q[vij] += lam_link
            add_pair(vij, open_index[j], -lam_link)

    # --- Constraint: soft capacity ---------------------------------------
    # Deliberately NOT encoded as a quadratic penalty. A capacity limit is a
    # one-sided (<=) constraint, and the natural quadratic `(load/cap - 1)^2`
    # is two-sided: it penalises an UNDER-filled site just as hard as an
    # overfull one, which pushes the sampler to over-concentrate demand.
    # Expressing <= properly needs ~log2(cap) slack bits per site, roughly
    # doubling the problem to softly enforce something the decode step repairs
    # anyway. Capacity is therefore enforced in the classical assignment
    # repair and any residual overflow is reported in the result.

    # --- Optional cap on how many sites may open --------------------------
    # `max_open_sites` is an upper bound, so it needs slack: encoding it as
    # the equality `(sum_j y_j - m)^2` would force EXACTLY m sites open even
    # when fewer are cheaper. Slack binaries with weights summing to m let
    # `sum_j y_j` land anywhere in [0, m] at zero penalty.
    if problem.max_open_sites is not None:
        m = int(problem.max_open_sites)
        terms: List[Tuple[int, float]] = [(open_index[j], 1.0) for j in range(n_sites)]
        terms += [(slack_index[k], float(w)) for k, w in enumerate(slack_weights)]

        # lam * (sum_t c_t z_t - m)^2
        const += lam_site_cap * m * m
        for idx, (vi, ci) in enumerate(terms):
            q[vi] += lam_site_cap * ci * (ci - 2.0 * m)
            for vj, cj in terms[idx + 1 :]:
                add_pair(vi, vj, 2.0 * lam_site_cap * ci * cj)

    return NetworkQubo(
        q=q,
        Q=Q,
        const=const,
        site_ids=[s.id for s in sites],
        demand_ids=[d.id for d in demand_points],
        assign_index=assign_index,
        open_index=open_index,
        penalties={
            "assign": lam_assign,
            "link": lam_link,
            "site_cap": lam_site_cap,
        },
        cost_matrix=cost,
    )


def qubo_summary(qubo: NetworkQubo) -> Dict[str, Any]:
    """Diagnostics block for the execution-steps envelope."""
    return {
        "n_vars": qubo.n_vars,
        "n_sites": len(qubo.site_ids),
        "n_demand_points": len(qubo.demand_ids),
        "open_vars": len(qubo.open_index),
        "assignment_vars": len(qubo.assign_index),
        "nonzero_pairs": int(np.count_nonzero(qubo.Q)),
        "penalties": {k: round(v, 4) for k, v in qubo.penalties.items()},
    }
