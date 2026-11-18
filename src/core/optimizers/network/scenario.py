"""Seeded facility-location scenarios (F32).

Four presets sized against the solver ceilings rather than picked for round
numbers: ``pilot_2of4`` is the only one that fits ``MAX_GATE_QUBITS``, so it is
what makes QAOA selectable at all; the rest target annealing. The encoding
costs ``n_sites + n_sites * n_demand + slack`` binaries, so the assignment
block dominates and even a six-site network is far past a state-vector budget.

Geography is real German/Benelux logistics corridors so the map reads as a
plausible network rather than random dots.
"""

from __future__ import annotations

from typing import Any, Dict, List

from core.optimizers.network.models import CandidateSite, DemandPoint, NetworkProblem

DEFAULT_SCENARIO = "regional_3of6"


def _sites(rows: List[tuple]) -> List[CandidateSite]:
    return [
        CandidateSite(
            id=r[0], name=r[1], lat=r[2], lon=r[3], fixed_cost=r[4], capacity=r[5]
        )
        for r in rows
    ]


def _demand(rows: List[tuple]) -> List[DemandPoint]:
    return [
        DemandPoint(id=r[0], name=r[1], lat=r[2], lon=r[3], demand=r[4]) for r in rows
    ]


def _pilot_2of4() -> NetworkProblem:
    """4 candidate sites, 3 demand points - 18 binaries, the only gate fit."""
    sites = _sites([
        ("S1", "Duisburg", 51.4344, 6.7623, 42000.0, 1400.0),
        ("S2", "Dortmund", 51.5136, 7.4653, 38000.0, 1200.0),
        ("S3", "Koln", 50.9375, 6.9603, 45000.0, 1500.0),
        ("S4", "Kassel", 51.3127, 9.4797, 31000.0, 900.0),
    ])
    demand = _demand([
        ("D1", "Essen", 51.4556, 7.0116, 620.0),
        ("D2", "Bonn", 50.7374, 7.0982, 410.0),
        ("D3", "Gottingen", 51.5413, 9.9158, 260.0),
    ])
    return NetworkProblem(sites=sites, demand_points=demand, max_open_sites=2)


def _regional_3of6() -> NetworkProblem:
    """6 candidate sites, 8 demand points - 56 binaries, annealing only."""
    sites = _sites([
        ("S1", "Duisburg", 51.4344, 6.7623, 42000.0, 2600.0),
        ("S2", "Dortmund", 51.5136, 7.4653, 38000.0, 2200.0),
        ("S3", "Koln", 50.9375, 6.9603, 45000.0, 2800.0),
        ("S4", "Hannover", 52.3759, 9.7320, 36000.0, 2000.0),
        ("S5", "Frankfurt", 50.1109, 8.6821, 52000.0, 3200.0),
        ("S6", "Kassel", 51.3127, 9.4797, 31000.0, 1700.0),
    ])
    demand = _demand([
        ("D1", "Essen", 51.4556, 7.0116, 620.0),
        ("D2", "Dusseldorf", 51.2277, 6.7735, 580.0),
        ("D3", "Bonn", 50.7374, 7.0982, 410.0),
        ("D4", "Munster", 51.9607, 7.6261, 350.0),
        ("D5", "Bielefeld", 52.0302, 8.5325, 300.0),
        ("D6", "Gottingen", 51.5413, 9.9158, 260.0),
        ("D7", "Wiesbaden", 50.0782, 8.2398, 440.0),
        ("D8", "Siegen", 50.8748, 8.0243, 230.0),
    ])
    return NetworkProblem(sites=sites, demand_points=demand, max_open_sites=3)


def _metro_5of12() -> NetworkProblem:
    """12 candidate sites, 20 demand points - annealing territory."""
    sites = _sites([
        ("S1", "Duisburg", 51.4344, 6.7623, 42000.0, 2600.0),
        ("S2", "Dortmund", 51.5136, 7.4653, 38000.0, 2200.0),
        ("S3", "Koln", 50.9375, 6.9603, 45000.0, 2800.0),
        ("S4", "Hannover", 52.3759, 9.7320, 36000.0, 2000.0),
        ("S5", "Frankfurt", 50.1109, 8.6821, 52000.0, 3200.0),
        ("S6", "Kassel", 51.3127, 9.4797, 31000.0, 1700.0),
        ("S7", "Bremen", 53.0793, 8.8017, 34000.0, 1900.0),
        ("S8", "Hamburg", 53.5511, 9.9937, 58000.0, 3400.0),
        ("S9", "Leipzig", 51.3397, 12.3731, 33000.0, 1800.0),
        ("S10", "Nurnberg", 49.4521, 11.0767, 39000.0, 2100.0),
        ("S11", "Stuttgart", 48.7758, 9.1829, 47000.0, 2700.0),
        ("S12", "Mannheim", 49.4875, 8.4660, 35000.0, 1900.0),
    ])
    demand = _demand([
        ("D1", "Essen", 51.4556, 7.0116, 620.0),
        ("D2", "Dusseldorf", 51.2277, 6.7735, 580.0),
        ("D3", "Bonn", 50.7374, 7.0982, 410.0),
        ("D4", "Munster", 51.9607, 7.6261, 350.0),
        ("D5", "Bielefeld", 52.0302, 8.5325, 300.0),
        ("D6", "Gottingen", 51.5413, 9.9158, 260.0),
        ("D7", "Wiesbaden", 50.0782, 8.2398, 440.0),
        ("D8", "Siegen", 50.8748, 8.0243, 230.0),
        ("D9", "Oldenburg", 53.1435, 8.2146, 270.0),
        ("D10", "Kiel", 54.3233, 10.1228, 310.0),
        ("D11", "Rostock", 54.0924, 12.0991, 240.0),
        ("D12", "Magdeburg", 52.1205, 11.6276, 290.0),
        ("D13", "Dresden", 51.0504, 13.7373, 420.0),
        ("D14", "Erfurt", 50.9848, 11.0299, 260.0),
        ("D15", "Wurzburg", 49.7913, 9.9534, 250.0),
        ("D16", "Regensburg", 49.0134, 12.1016, 280.0),
        ("D17", "Augsburg", 48.3705, 10.8978, 330.0),
        ("D18", "Karlsruhe", 49.0069, 8.4037, 360.0),
        ("D19", "Freiburg", 47.9990, 7.8421, 290.0),
        ("D20", "Ulm", 48.4011, 9.9876, 240.0),
    ])
    return NetworkProblem(sites=sites, demand_points=demand, max_open_sites=5)


def _national_8of40() -> NetworkProblem:
    """A deliberately oversized instance: 40 sites x 60 demand points.

    Past every device ceiling; exists so the UI can demonstrate the classical
    baseline at a realistic national scale and so the API's guard rails have
    something to refuse.
    """
    import math

    sites = []
    for i in range(40):
        angle = 2.0 * math.pi * i / 40.0
        sites.append(
            CandidateSite(
                id=f"S{i + 1}",
                name=f"Site {i + 1}",
                lat=51.0 + 3.0 * math.sin(angle) * (0.4 + 0.6 * ((i % 7) / 7.0)),
                lon=9.5 + 4.5 * math.cos(angle) * (0.4 + 0.6 * ((i % 5) / 5.0)),
                fixed_cost=28000.0 + 900.0 * (i % 11),
                capacity=1500.0 + 120.0 * (i % 9),
            )
        )
    demand = []
    for i in range(60):
        angle = 2.0 * math.pi * i / 60.0
        demand.append(
            DemandPoint(
                id=f"D{i + 1}",
                name=f"Region {i + 1}",
                lat=51.0 + 3.4 * math.sin(angle) * (0.3 + 0.7 * ((i % 6) / 6.0)),
                lon=9.5 + 5.0 * math.cos(angle) * (0.3 + 0.7 * ((i % 8) / 8.0)),
                demand=180.0 + 22.0 * (i % 13),
            )
        )
    return NetworkProblem(sites=sites, demand_points=demand, max_open_sites=8)


SCENARIOS: Dict[str, Dict[str, Any]] = {
    "pilot_2of4": {
        "label": "Pilot - 2 of 4 sites",
        "description": (
            "Rhine-Ruhr pilot: pick 2 of 4 candidate DCs to serve 3 cities. "
            "18 binaries, the only preset a gate-based simulator can run."
        ),
        "builder": _pilot_2of4,
    },
    "regional_3of6": {
        "label": "Regional - 3 of 6 sites",
        "description": (
            "Rhine-Ruhr consolidation: pick 3 of 6 candidate DCs to serve 8 "
            "cities. 56 binaries - annealing methods only."
        ),
        "builder": _regional_3of6,
    },
    "metro_5of12": {
        "label": "National - 5 of 12 sites",
        "description": (
            "Germany-wide: pick 5 of 12 candidate DCs to serve 20 cities. "
            "Annealing territory."
        ),
        "builder": _metro_5of12,
    },
    "national_8of40": {
        "label": "Large scale - 8 of 40 sites",
        "description": (
            "40 candidate sites, 60 demand regions. Beyond "
            "every device ceiling; classical baseline only."
        ),
        "builder": _national_8of40,
    },
}


def list_scenarios() -> List[Dict[str, Any]]:
    """Scenario catalogue for the UI picker."""
    from core.optimizers.fleet.samplers import MAX_GATE_QUBITS  # noqa: PLC0415
    from core.optimizers.network.qubo import build_network_qubo  # noqa: PLC0415

    out = []
    for key, spec in SCENARIOS.items():
        problem = spec["builder"]()
        # Build the real QUBO rather than recomputing the layout by hand: the
        # cardinality cap adds slack binaries, and a hand-rolled formula
        # silently under-reported the qubit cost the moment that was added.
        n_vars = build_network_qubo(problem).n_vars
        out.append({
            "id": key,
            "label": spec["label"],
            "description": spec["description"],
            "site_count": len(problem.sites),
            "demand_point_count": len(problem.demand_points),
            "qubo_vars": n_vars,
            # Lets the picker disable gate methods up front instead of letting
            # the user submit a job that can only fail in the worker.
            "gate_ready": n_vars <= MAX_GATE_QUBITS,
            "max_open_sites": problem.max_open_sites,
            "default": key == DEFAULT_SCENARIO,
        })
    return out


def build_scenario(scenario_id: str | None) -> NetworkProblem:
    """Instantiate a seeded scenario by id."""
    key = (scenario_id or DEFAULT_SCENARIO).strip()
    if key not in SCENARIOS:
        raise ValueError(
            f"unknown network scenario '{scenario_id}'; "
            f"expected one of {sorted(SCENARIOS)}"
        )
    return SCENARIOS[key]["builder"]()


# --------------------------------------------------------------------------
# Custom instances
# --------------------------------------------------------------------------

CUSTOM_SCENARIO_ID = "custom"

CUSTOM_LIMITS: Dict[str, Any] = {
    "sites": (3, 40),
    "demand_points": (3, 60),
    "max_open_sites": (1, 20),
    "capacity_headroom": (1.05, 4.0),
    "cost_spread": (0.0, 0.6),
}

DEFAULT_CAPACITY_HEADROOM = 1.6
DEFAULT_COST_SPREAD = 0.15


def _clamp(value: float, bounds: tuple) -> float:
    return max(bounds[0], min(bounds[1], value))


def build_custom_scenario(
    *,
    sites: int = 12,
    demand_points: int = 20,
    max_open_sites: int = 5,
    capacity_headroom: float = DEFAULT_CAPACITY_HEADROOM,
    cost_spread: float = DEFAULT_COST_SPREAD,
    seed: int = 42,
) -> NetworkProblem:
    """A seeded random instance with the difficulty knobs exposed.

    On the two small presets greedy + drop/swap local search reaches the
    optimum, so a comparison shows nothing there. This builder exists so a
    user can try other instances.

    ``capacity_headroom`` is the ratio of the capacity available across the
    ``max_open_sites`` largest sites to total demand. It is floored at 1.05
    so a feasible plan always exists: below 1.0 no solver can serve all
    demand and the run would only measure who overflows least.
    """
    import math  # noqa: PLC0415
    import random  # noqa: PLC0415

    n_sites = int(_clamp(sites, CUSTOM_LIMITS["sites"]))
    n_demand = int(_clamp(demand_points, CUSTOM_LIMITS["demand_points"]))
    n_open = int(_clamp(max_open_sites, CUSTOM_LIMITS["max_open_sites"]))
    n_open = max(1, min(n_open, n_sites))
    headroom = _clamp(float(capacity_headroom), CUSTOM_LIMITS["capacity_headroom"])
    spread = _clamp(float(cost_spread), CUSTOM_LIMITS["cost_spread"])

    rng = random.Random(int(seed))

    demand = []
    for i in range(n_demand):
        angle = 2.0 * math.pi * i / n_demand
        demand.append(
            DemandPoint(
                id=f"D{i + 1}",
                name=f"Region {i + 1}",
                lat=51.0 + 3.4 * math.sin(angle) * (0.35 + 0.65 * rng.random()),
                lon=9.5 + 5.0 * math.cos(angle) * (0.35 + 0.65 * rng.random()),
                demand=round(rng.uniform(150.0, 650.0), 1),
            )
        )

    total_demand = sum(d.demand for d in demand)
    max_demand = max(d.demand for d in demand)
    # Size each site so the n_open largest together cover demand * headroom.
    # Jitter is one-sided so this is a floor, never less: symmetric jitter let
    # an unlucky draw undershoot and made the instance infeasible outright.
    base_capacity = total_demand * headroom / n_open
    # The greedy baseline assigns demand points whole, so a site with ample
    # spare capacity can still fail to fit the next point. Flooring at a few
    # times the largest point halves that fragmentation without flattening
    # the instance into a one-site answer.
    base_capacity = max(base_capacity, 3.0 * max_demand)

    site_rows = []
    for i in range(n_sites):
        angle = 2.0 * math.pi * i / n_sites + 0.4
        site_rows.append(
            CandidateSite(
                id=f"S{i + 1}",
                name=f"Site {i + 1}",
                lat=51.0 + 3.0 * math.sin(angle) * (0.35 + 0.65 * rng.random()),
                lon=9.5 + 4.5 * math.cos(angle) * (0.35 + 0.65 * rng.random()),
                fixed_cost=round(35000.0 * (1.0 + rng.uniform(-spread, spread)), 1),
                capacity=round(base_capacity * (1.0 + rng.uniform(0.0, 0.25)), 1),
            )
        )

    return NetworkProblem(
        sites=site_rows, demand_points=demand, max_open_sites=n_open
    )
