"""QUBO of the mission allocation problem, one binary per allowed pair (F7).

Energy = -weights + P per two assets on one task + P per conflicting task pair."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from core.optimizers.mission.models import MissionProblem, Pair, Structure, build_structure

DEFAULT_PENALTY_FACTOR = 1.05
NOT_ENCODED = "duty limit per asset"


@dataclass
class MissionQubo:
    """Upper-triangular QUBO terms and the pair behind every variable."""

    #: {(i, j): value} with i <= j; (i, i) holds the linear term.
    terms: Dict[Tuple[int, int], float]
    #: Variable i stands for the allowed pair variables[i].
    variables: List[Pair]
    penalty: float
    penalty_factor: float
    max_weight: float

    @property
    def index(self) -> Dict[Pair, int]:
        return {pair: i for i, pair in enumerate(self.variables)}


def build_qubo(
    problem: MissionProblem,
    penalty_factor: float = DEFAULT_PENALTY_FACTOR,
    structure: Optional[Structure] = None,
) -> MissionQubo:
    """Encode weights and pair exclusions; the duty limit is not encoded."""
    if not float(penalty_factor) > 1.0:
        raise ValueError("penalty_factor must be larger than 1")
    st = structure or build_structure(problem)
    index = {pair: i for i, pair in enumerate(st.pairs)}
    max_weight = max(st.weight.values(), default=0.0)
    penalty = float(penalty_factor) * max_weight

    terms: Dict[Tuple[int, int], float] = {(i, i): -st.weight[pair] for pair, i in index.items()}
    for task in problem.tasks:
        takers = st.assets_of[task.id]
        for k, first in enumerate(takers):
            for second in takers[k + 1:]:
                terms[(index[(first, task.id)], index[(second, task.id)])] = penalty
    for asset in problem.assets:
        for t, u in st.conflict_pairs(asset.id):
            terms[(index[(asset.id, t)], index[(asset.id, u)])] = penalty
    return MissionQubo(
        terms=terms,
        variables=list(st.pairs),
        penalty=penalty,
        penalty_factor=float(penalty_factor),
        max_weight=max_weight,
    )


def qubo_summary(qubo: MissionQubo) -> Dict[str, Any]:
    """Size and penalty of the QUBO, and what it leaves out."""
    return {
        "variables": len(qubo.variables),
        "interactions": sum(1 for i, j in qubo.terms if i != j),
        "penalty": round(qubo.penalty, 3),
        "penalty_factor": qubo.penalty_factor,
        "max_weight": round(qubo.max_weight, 3),
        "not_encoded": NOT_ENCODED,
    }


def energy(qubo: MissionQubo, bits: Sequence[int]) -> float:
    """QUBO energy of one bitstring."""
    x = [int(b) for b in bits]
    if len(x) != len(qubo.variables):
        raise ValueError(f"expected {len(qubo.variables)} bits, got {len(x)}")
    return math.fsum(value for (i, j), value in qubo.terms.items() if x[i] and x[j])
