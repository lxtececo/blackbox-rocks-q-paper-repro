"""Pauli Correlation Encoding (PCE) portfolio optimizer.

Encoding (Sciorilli et al., Nat. Commun. 16, 476 (2025), arXiv:2401.09421):
the m selection variables are the signs of the expectation values of m Pauli
strings on n qubits. Each string is a product of k identical Pauli matrices
(X..X, Y..Y or Z..Z) on k of the qubits, so up to 3*C(n, k) variables fit on
n qubits. Strings of one type commute: three measurement settings give all m
expectation values.

Training minimises the Ising form of the portfolio QUBO with every spin z_i
replaced by tanh(alpha <P_i>), plus the paper's regulariser. The selection is
z_i = sgn <P_i> at the trained parameters, followed by classical single-bit
flips and, with a target number of assets, swaps.

The paper states its loss for MaxCut. The linear term, the scale nu of the
regulariser and the swap step are this module's extension to a QUBO with
fields and a cardinality penalty; the rounds of perturbed local optimisation
are this module's training schedule, not the paper's.
"""

from __future__ import annotations

import itertools
import logging
import math
import os
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pypfopt import expected_returns
from pypfopt.risk_models import CovarianceShrinkage
from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import SamplerV2 as Sampler
from scipy.optimize import OptimizeResult, minimize

from core.portfolio_referee import referee
from core.postprocessing.refinement_engine import single_flip_descent, swap_local_search
from core.quantum_math import cardinality_penalty_influence, cardinality_penalty_params, qubo_pair_to_ising

logger = logging.getLogger(__name__)

SHOTS_FLOOR = 32
BASES = "XYZ"


@dataclass
class PCEConfig:
    """Configuration of the PCE optimizer."""
    compression_factor: int = 3           # at most this many assets per qubit
    correlation_depth: int = 2            # k: Pauli matrices per string
    ansatz_depth: int = 3
    alpha: Optional[float] = None         # tanh scale; None -> n_qubits ** (k // 2), as in the paper
    beta: float = 0.5                     # weight of the regulariser
    max_iterations: int = 50              # optimiser iterations per round
    perturbation_factor: float = 0.1
    exploration_factor: float = 2.0
    decay_factor: float = 0.9
    max_no_improvement_rounds: int = 6    # rounds of local optimisation
    restart_threshold: int = 3            # failed rounds before a random restart
    convergence_threshold: float = 1e-5
    convergence_patience: int = 6
    shots: int = 1024                     # per measurement setting
    optimization_timeout: float = 3600.0
    risk_aversion: float = 0.5
    cardinality: Optional[int] = None     # assets to hold; None = no target
    seed: Optional[int] = None


class PCEMethod(Enum):
    QUBO_LOSS = "qubo_loss"
    MAX_CUT = "max_cut"
    HYBRID = "hybrid"


def required_qubits(num_variables: int, k: int, compression_factor: int) -> int:
    """Fewest qubits that hold ``num_variables`` k-body strings, with at most ``compression_factor`` per qubit."""
    n = k
    while 3 * math.comb(n, k) < num_variables:
        n += 1
    return max(n, math.ceil(num_variables / max(1, int(compression_factor))))


def pauli_strings(num_qubits: int, num_variables: int, k: int) -> List[Tuple[str, Tuple[int, ...]]]:
    """The first ``num_variables`` strings, ordered by qubit subset and then by Pauli type."""
    strings = [(pauli, subset) for subset in itertools.combinations(range(num_qubits), k) for pauli in BASES]
    if len(strings) < num_variables:
        raise ValueError(f"{num_qubits} qubits hold {len(strings)} {k}-body strings, {num_variables} needed")
    return strings[:num_variables]


def correlators(counts_by_basis: Dict[str, Dict[int, int]], strings: List[Tuple[str, Tuple[int, ...]]],
                num_qubits: int) -> np.ndarray:
    """Expectation value of each string, from the counts measured in that string's own basis.

    Count keys are integers whose bit q is the outcome of qubit q.
    """
    parsed = {}
    for basis, counts in counts_by_basis.items():
        keys = np.fromiter(counts.keys(), dtype=np.int64, count=len(counts))
        weights = np.fromiter(counts.values(), dtype=float, count=len(counts))
        bits = (keys[:, None] >> np.arange(num_qubits, dtype=np.int64)) & 1
        parsed[basis] = (bits, weights / weights.sum())
    values = np.empty(len(strings))
    for i, (pauli, subset) in enumerate(strings):
        bits, probability = parsed[pauli]
        values[i] = float(probability @ (1 - 2 * (bits[:, list(subset)].sum(axis=1) & 1)))
    return values


class PCEPortfolioOptimizer:
    """Asset selection with m variables encoded in the Pauli correlators of fewer qubits."""

    @classmethod
    def create_deterministic(cls, seed: int, config: PCEConfig = None):
        """An optimizer whose parameters, perturbations and simulator shots all follow ``seed``."""
        return cls(replace(config or PCEConfig(), seed=int(seed)))

    def __init__(self, config: PCEConfig = None, use_real_hardware: bool = False, backend=None):
        self.config = config or PCEConfig()
        self.logger = logger

        self._shots_original = self.config.shots
        self._shots_clamped = self.config.shots < SHOTS_FLOOR
        if self._shots_clamped:
            logger.warning("[PCE] shots=%d is below %d; the sign of a correlator would be shot noise. Using %d.",
                           self.config.shots, SHOTS_FLOOR, SHOTS_FLOOR)
            self.config.shots = SHOTS_FLOOR

        self.simulator = backend if backend is not None else self._resolve_backend(use_real_hardware)
        self.pm = generate_preset_pass_manager(backend=self.simulator, optimization_level=1)
        self._backend_name = str(getattr(self.simulator, "name", self.simulator))
        self._is_real_ibm = backend is None and not isinstance(self.simulator, AerSimulator)
        self.vendor_runs: list[dict] = []

        seed = self.config.seed if self.config.seed is not None else int(time.time() * 1000) % 2 ** 32
        self.random_seed = int(seed)
        self.rng = np.random.default_rng(self.random_seed)
        self._shot_seeds = np.random.default_rng([self.random_seed, 1])

        self.optimization_history: list[dict] = []
        self.refinement_telemetry: dict = {}
        self.prices = None
        self.num_assets = 0
        self.num_qubits = 0
        self.pauli_correlations: List[Tuple[str, Tuple[int, ...]]] = []
        self._circuits: List[QuantumCircuit] = []
        self._evaluations = 0
        self._optimizer_name = ""

    def _resolve_backend(self, use_real_hardware: bool):
        if not use_real_hardware:
            return AerSimulator()
        from core.optimizers.quantum.shared.circuit_runner import HardwareUnavailableError
        try:
            from core.quantum_credential_context import tenant_token_for
            token = tenant_token_for("ibm_quantum") or os.getenv("IBM_QUANTUM_TOKEN")
        except Exception:  # noqa: BLE001
            token = os.getenv("IBM_QUANTUM_TOKEN")
        if not token:
            raise HardwareUnavailableError("PCE on IBM hardware needs an IBM Quantum credential; none was resolved")
        try:
            from qiskit_ibm_runtime import QiskitRuntimeService
            service = QiskitRuntimeService(channel="ibm_quantum_platform", token=token)
            backend = service.least_busy(min_num_qubits=127, simulator=False)
            logger.info("[PCE] using IBM backend %s", getattr(backend, "name", backend))
            return backend
        except Exception as exc:  # noqa: BLE001
            raise HardwareUnavailableError(f"IBM Quantum hardware is unavailable: {exc}") from exc

    def _capture_runtime_job(self, job, stage: str) -> None:
        """Record the IBM Runtime job id for the vendor-portal link (real backend only)."""
        if not self._is_real_ibm:
            return
        try:
            jid = job.job_id()
        except Exception:  # noqa: BLE001
            return
        if jid:
            self.vendor_runs.append({
                "vendor": "ibm", "vendor_label": "IBM Quantum", "job_id": jid, "stage": stage,
                "backend": self._backend_name, "submitted_at": datetime.now(timezone.utc).isoformat(),
            })

    # ------------------------------------------------------------------ problem and encoding

    def calculate_required_qubits(self, num_assets: int) -> int:
        return required_qubits(num_assets, self.config.correlation_depth, self.config.compression_factor)

    def generate_pauli_correlations(self, num_qubits: int, num_assets: int) -> List[Tuple[str, Tuple[int, ...]]]:
        return pauli_strings(num_qubits, num_assets, self.config.correlation_depth)

    def build_portfolio_qubo(self, prices: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, float, Dict[str, Any]]:
        """Pair-form selection QUBO: cost(x) = -mu.x + gamma x.S.x (+ penalty (sum x - K)^2 with a target K)."""
        self.prices = prices
        self.num_assets = prices.shape[1]
        mu = expected_returns.mean_historical_return(prices).to_numpy()
        cov = CovarianceShrinkage(prices).ledoit_wolf().to_numpy()
        gamma = float(self.config.risk_aversion)
        q = -mu + gamma * np.diag(cov)
        Q = np.triu(2.0 * gamma * cov, 1)
        const, penalty = 0.0, None
        cardinality = self.config.cardinality
        if cardinality is not None and not 0 < cardinality < self.num_assets:
            cardinality = None  # a target of all assets constrains nothing
        if cardinality is not None:
            penalty = cardinality_penalty_influence(q, Q)
            q, Q, const = cardinality_penalty_params(q, Q, cardinality, penalty)
        self._cardinality = cardinality
        metadata = {
            "num_assets": self.num_assets,
            "expected_returns": dict(zip(prices.columns, (float(v) for v in mu))),
            "risk_matrix_trace": float(np.trace(cov)),
            "cardinality": cardinality,
            "penalty_lambda": penalty,
        }
        return q, Q, const, metadata

    def create_brickwork_ansatz(self, num_qubits: int) -> Tuple[QuantumCircuit, List[Parameter]]:
        """Layers of RX, RY, RZ on every qubit and RXX on alternating neighbouring pairs; no measurement."""
        circuit = QuantumCircuit(num_qubits)
        parameters: List[Parameter] = []

        def new(name: str) -> Parameter:
            parameters.append(Parameter(name))
            return parameters[-1]

        for layer in range(self.config.ansatz_depth):
            for qubit in range(num_qubits):
                circuit.rx(new(f"rx_{layer}_{qubit}"), qubit)
                circuit.ry(new(f"ry_{layer}_{qubit}"), qubit)
                circuit.rz(new(f"rz_{layer}_{qubit}"), qubit)
            for i in range(layer % 2, num_qubits - 1, 2):
                circuit.rxx(new(f"rxx_{layer}_{i}_{i + 1}"), i, i + 1)
        return circuit, parameters

    @staticmethod
    def measurement_circuits(ansatz: QuantumCircuit) -> Dict[str, QuantumCircuit]:
        """The ansatz followed by the basis change of each measurement setting."""
        out = {}
        for basis in BASES:
            circuit = ansatz.copy()
            if basis == "Y":
                circuit.sdg(range(circuit.num_qubits))
            if basis in "XY":
                circuit.h(range(circuit.num_qubits))
            circuit.measure_all()
            out[basis] = circuit
        return out

    # ------------------------------------------------------------------ loss and decoding

    def _estimate(self, params: np.ndarray, stage: str) -> np.ndarray:
        """All correlators at ``params``: one job, three measurement settings."""
        sampler = Sampler(mode=self.simulator)
        if not self._is_real_ibm:
            sampler.options.simulator.seed_simulator = int(self._shot_seeds.integers(1, 2 ** 31 - 1))
        values = np.asarray(params, dtype=float)
        job = sampler.run([(circuit, values) for circuit in self._circuits], shots=self.config.shots)
        self._capture_runtime_job(job, stage=stage)
        result = job.result()
        counts = {basis: result[i].data.meas.get_int_counts() for i, basis in enumerate(BASES)}
        return correlators(counts, self.pauli_correlations, self.num_qubits)

    def loss(self, expectations: np.ndarray) -> float:
        """Ising energy of the relaxed spins tanh(alpha <P_i>) plus beta nu (mean tanh^2)^2."""
        t = np.tanh(self._alpha * np.asarray(expectations, dtype=float))
        return float(self._h @ t + t @ self._J @ t + self.config.beta * self._nu * np.mean(t ** 2) ** 2)

    @staticmethod
    def decode(expectations: np.ndarray) -> np.ndarray:
        """x_i = 1 where <P_i> is negative (z_i = sgn <P_i>, x_i = (1 - z_i) / 2)."""
        return (np.asarray(expectations) < 0).astype(int)

    def _objective(self, params: np.ndarray) -> float:
        self._evaluations += 1
        return self.loss(self._estimate(params, stage=f"sampler loss evaluation #{self._evaluations}"))

    # ------------------------------------------------------------------ training

    def _spsa_minimize(self, objective, theta0: np.ndarray, maxiter: int) -> OptimizeResult:
        """SPSA with the gain sequences of Spall (1998): two evaluations per iteration."""
        theta = theta0.copy()
        best_theta, best_cost = theta.copy(), float("inf")
        a, c, stability, decay_a, decay_c = 0.2, 0.1, max(1.0, 0.1 * maxiter), 0.602, 0.101
        for k in range(maxiter):
            ak = a / (k + 1 + stability) ** decay_a
            ck = c / (k + 1) ** decay_c
            delta = self.rng.choice([-1.0, 1.0], size=len(theta))
            plus, minus = objective(theta + ck * delta), objective(theta - ck * delta)
            theta = theta - ak * (plus - minus) / (2.0 * ck * delta)
            if 0.5 * (plus + minus) < best_cost:
                best_cost, best_theta = 0.5 * (plus + minus), theta.copy()
        return OptimizeResult(x=best_theta, fun=best_cost, nfev=2 * maxiter, success=True)

    def multi_step_optimization(self) -> Tuple[np.ndarray, float, int]:
        """Rounds of local optimisation from perturbed parameters; the lowest loss seen wins.

        Returns (parameters, loss, rounds).
        """
        cfg = self.config
        n_params = len(self._circuits[0].parameters)
        # COBYLA spends n_params + 1 evaluations on its first simplex; below that budget SPSA is used.
        use_spsa = cfg.max_iterations < n_params + 2
        self._optimizer_name = "SPSA" if use_spsa else "COBYLA"

        theta = self.rng.uniform(-np.pi, np.pi, n_params)
        best_theta, best_cost = theta.copy(), float("inf")
        trend = np.zeros(n_params)
        perturbation = cfg.perturbation_factor
        failures = idle = rounds = 0
        started = time.time()

        for round_num in range(cfg.max_no_improvement_rounds):
            rounds = round_num + 1
            round_start = time.perf_counter()
            scale = (cfg.exploration_factor if round_num % 3 == 0 else 1.0) * perturbation
            start = theta + self.rng.normal(0, scale * (1 + failures / 5), n_params) + scale * np.sign(trend)
            if use_spsa:
                outcome = self._spsa_minimize(self._objective, start, cfg.max_iterations)
            else:
                outcome = minimize(self._objective, start, method="COBYLA",
                                   options={"maxiter": cfg.max_iterations, "rhobeg": 0.5})
            cost = float(outcome.fun)
            self.optimization_history.append({
                "round": round_num,
                "cost": cost,
                "success": bool(outcome.success),
                "parameter_norm": float(np.linalg.norm(outcome.x)),
                "failure_count": failures,
                "params": [float(x) for x in outcome.x],
                "wall_s": round(time.perf_counter() - round_start, 6),
            })

            if cost < best_cost:
                improvement = best_cost - cost
                best_cost, best_theta = cost, outcome.x.copy()
                failures = 0
                perturbation *= cfg.decay_factor
                idle = 0 if improvement >= cfg.convergence_threshold else idle + 1
                trend = (outcome.x - theta) if not trend.any() else 0.8 * trend + 0.2 * (outcome.x - theta)
                theta = outcome.x.copy()
            else:
                failures += 1
                idle += 1
                if failures >= cfg.restart_threshold:
                    theta = self.rng.uniform(-np.pi, np.pi, n_params)
                    trend = np.zeros(n_params)
                    failures = 0
                elif failures % 2 == 0:
                    theta = 0.7 * best_theta + 0.3 * outcome.x
                else:
                    theta = best_theta + self.rng.normal(0, 2 * perturbation, n_params)

            if idle >= cfg.convergence_patience or time.time() - started > cfg.optimization_timeout:
                break
        return best_theta, best_cost, rounds

    # ------------------------------------------------------------------ classical post-processing

    def _postprocess(self, bits: np.ndarray, q: np.ndarray, Q: np.ndarray) -> np.ndarray:
        """Single-bit flips, then swaps at the target cardinality, each only if the QUBO cost falls."""
        cost = lambda x: float(x @ q + x @ Q @ x)  # noqa: E731
        h, J, _ = qubo_pair_to_ising(q, Q)
        start = np.asarray(bits, dtype=int)
        _, out = single_flip_descent(start, h, J + J.T, 10 * len(start), self.rng)
        forced = False
        if not out.any():
            # The QUBO prefers to hold nothing; a portfolio needs one asset.
            out[int(np.argmin(q))] = 1
            forced = True
        swaps = {"swaps_accepted": 0}
        if self._cardinality is not None:
            out, swaps = swap_local_search(out, q, Q, rng=self.rng)
        self.refinement_telemetry = {
            "engine": "single_flip_descent+swap_local_search",
            "initial_qubo_cost": round(cost(start), 6),
            "final_qubo_cost": round(cost(out), 6),
            "cost_delta": round(cost(start) - cost(out), 6),
            "improved": bool(cost(out) < cost(start) - 1e-12),
            "bits_changed": int(np.sum(start != out)),
            "selected_before": int(start.sum()),
            "selected_after": int(out.sum()),
            "swaps_accepted": int(swaps["swaps_accepted"]),
            "forced_nonempty": forced,
        }
        return out

    # ------------------------------------------------------------------ entry point

    def optimize(self, prices: pd.DataFrame) -> Dict[str, Any]:
        started = time.time()
        try:
            q, Q, const, metadata = self.build_portfolio_qubo(prices)
            k = self.config.correlation_depth
            self.num_qubits = self.calculate_required_qubits(self.num_assets)
            self.pauli_correlations = self.generate_pauli_correlations(self.num_qubits, self.num_assets)

            h, J, _ = qubo_pair_to_ising(q, Q)
            scale = max(float(np.abs(h).max(initial=0.0)), float(np.abs(J).max(initial=0.0))) or 1.0
            self._h, self._J = h / scale, J / scale
            # Half the total coupling weight: the role the cut-value estimate plays for MaxCut in the paper.
            self._nu = 0.5 * (float(np.abs(self._h).sum()) + float(np.abs(self._J).sum()))
            self._alpha = float(self.config.alpha) if self.config.alpha else float(self.num_qubits ** (k // 2))

            ansatz, parameters = self.create_brickwork_ansatz(self.num_qubits)
            self._circuits = [self.pm.run(circuit) for circuit in self.measurement_circuits(ansatz).values()]
            self.optimization_history = []
            self._evaluations = 0

            best_theta, best_cost, rounds = self.multi_step_optimization()
            final = self._estimate(best_theta, stage="sampler final estimate")
            decoded = self.decode(final)
            selection = self._postprocess(decoded, q, Q)

            symbols = list(prices.columns)
            weights = selection / selection.sum()
            portfolio_weights = {symbol: float(weights[i]) for i, symbol in enumerate(symbols)}
            relaxed = np.tanh(self._alpha * final)
            standard_error = np.sqrt(np.clip(1.0 - final ** 2, 0.0, None) / self.config.shots)

            return {
                "success": True,
                "portfolio_weights": portfolio_weights,
                "selection": [int(b) for b in selection],
                "decoded_selection": [int(b) for b in decoded],
                "qubo_cost": float(selection @ q + selection @ Q @ selection + const),
                # Standing of the returned selection among all admissible ones (None above the enumeration limit).
                "exact_reference": referee(q, Q, selection, const=const, cardinality=self._cardinality, min_assets=1),
                "best_cost": best_cost,
                "optimization_rounds": rounds,
                "optimization_time": time.time() - started,
                "final_parameters": [float(x) for x in best_theta],
                "resource_usage": {
                    "qubits_used": self.num_qubits,
                    "ansatz_depth": self.config.ansatz_depth,
                    "total_parameters": len(parameters),
                    "shots_per_evaluation": self.config.shots,
                    "measurement_settings": len(BASES),
                    "loss_evaluations": self._evaluations,
                    "compression_achieved": f"{100 * (1 - self.num_qubits / self.num_assets):.1f}%",
                },
                "encoding": {
                    "variables": self.num_assets,
                    "qubits": self.num_qubits,
                    "k": k,
                    "strings": [f"{pauli}{list(subset)}" for pauli, subset in self.pauli_correlations],
                },
                "performance_metrics": self._calculate_performance_metrics(prices, portfolio_weights),
                "convergence_metrics": self._calculate_convergence_metrics(),
                "relaxation_metrics": {
                    "alpha": self._alpha,
                    "shots_floor_applied": bool(self._shots_clamped),
                    "shots_requested": self._shots_original,
                    "shots_used": self.config.shots,
                    "mean_abs_tanh": float(np.mean(np.abs(relaxed))),
                    "min_abs_correlator": float(np.min(np.abs(final))),
                    # Variables whose sign the final estimate does not resolve from shot noise.
                    "signs_within_two_standard_errors": int(np.sum(np.abs(final) < 2.0 * standard_error)),
                },
                "refinement": dict(self.refinement_telemetry),
                "metadata": metadata,
                "randomness_info": {
                    "random_seed": self.random_seed,
                    "parameter_initialization": "uniform(-pi, pi)",
                    "quantum_sampling_shots": self.config.shots,
                    "classical_optimizer": self._optimizer_name,
                    "sources_of_randomness": [
                        "Initial parameters",
                        "Measurement outcomes",
                        "Perturbations and restarts between rounds",
                        "Scan order of the classical post-processing",
                    ],
                },
                "optimization_history": self.optimization_history,
                "vendor_runs": list(self.vendor_runs),
                "total_optimization_time": time.time() - started,
                "method": "PCE_QUBO",
                "configuration": {
                    "compression_factor": self.config.compression_factor,
                    "correlation_depth": k,
                    "ansatz_depth": self.config.ansatz_depth,
                    "shots": self.config.shots,
                    "alpha": self._alpha,
                    "beta": self.config.beta,
                    "risk_aversion": self.config.risk_aversion,
                    "cardinality": self._cardinality,
                    "seed": self.random_seed,
                },
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as exc:  # noqa: BLE001
            logger.exception("[PCE] optimisation failed: %s", exc)
            return {"success": False, "error": str(exc), "method": "PCE_QUBO",
                    "timestamp": datetime.now().isoformat()}

    # ------------------------------------------------------------------ reporting

    def _calculate_performance_metrics(self, prices: pd.DataFrame,
                                       portfolio_weights: Dict[str, float]) -> Dict[str, float]:
        weights = np.array([portfolio_weights.get(symbol, 0.0) for symbol in prices.columns])
        mu = expected_returns.mean_historical_return(prices).to_numpy()
        cov = CovarianceShrinkage(prices).ledoit_wolf().to_numpy()
        expected_return = float(weights @ mu)
        volatility = float(np.sqrt(max(weights @ cov @ weights, 0.0)))
        return {
            "expected_annual_return": expected_return,
            "annual_volatility": volatility,
            "sharpe_ratio": expected_return / volatility if volatility > 0 else 0.0,
            "concentration_index": float(np.sum(weights ** 2)),
            "max_weight": float(weights.max()),
            "num_assets_held": int(np.sum(weights > 0.01)),
        }

    def _calculate_convergence_metrics(self) -> Dict[str, Any]:
        costs = [entry["cost"] for entry in self.optimization_history]
        if not costs:
            return {"total_evaluations": 0}
        improvements = [a - b for a, b in zip(costs, costs[1:]) if b < a]
        variance = float(np.var(costs)) if len(costs) > 1 else 0.0
        return {
            "total_evaluations": len(costs),
            "loss_evaluations": self._evaluations,
            "cost_improvements": len(improvements),
            "average_improvement": float(np.mean(improvements)) if improvements else 0.0,
            "cost_variance": variance,
            "cost_trend_slope": float(np.polyfit(range(len(costs)), costs, 1)[0]) if len(costs) > 2 else 0.0,
            "success_rate": float(np.mean([entry["success"] for entry in self.optimization_history])),
            "convergence_efficiency": float((costs[0] - costs[-1]) / len(costs)),
            "best_cost_achieved": float(min(costs)),
            "final_cost": float(costs[-1]),
            "optimization_stability": self._classify_stability(variance),
        }

    @staticmethod
    def _classify_stability(variance: float) -> str:
        """Spread of the per-round losses (unit-free, the loss is normalised)."""
        if variance < 0.01:
            return "high"
        return "medium" if variance < 0.1 else "low"
