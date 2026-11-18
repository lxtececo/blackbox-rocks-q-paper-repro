"""CVaR-VQE: variational optimisation of a diagonal Hamiltonian with a CVaR objective.

Definition (Barkoutsos, Nannicini, Robert, Tavernelli, Woerner, Quantum 4, 256
(2020), Eq. 3): measure the ansatz K times; every shot is a bitstring with a
definite energy E_k; sort the energies ascending and average the lowest
ceil(alpha * K) of them. alpha = 1 is the sample mean, alpha -> 0 the sample
minimum. The parameters are chosen to minimise that average.
"""

from __future__ import annotations

import logging
import math
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import SamplerV2 as Sampler
from scipy.optimize import minimize

logger = logging.getLogger(__name__)

SIMULATOR_SHOTS = 1024
HARDWARE_SHOTS = 2000
EVALUATIONS_PER_PARAMETER = 10
MIN_EVALUATIONS = 100


@dataclass
class CVaRConfig:
    """Configuration for CVaR-VQE.

    ``adaptive_alpha`` and ``noise_mitigation`` are accepted for compatibility
    with stored requests and have no effect: alpha is fixed during a run, and
    the L2 penalty on the parameters that ``noise_mitigation`` used to add is
    not part of the objective.
    """
    simulator_mode: bool = True
    alpha: float = 0.25
    adaptive_alpha: bool = False
    energy_threshold: float = 1e-6
    max_cvar_iterations: Optional[int] = None   # objective evaluations; None -> from the parameter count
    measurement_budget: Optional[int] = None    # shots per objective evaluation
    noise_mitigation: bool = False
    seed: Optional[int] = None                  # simulator only: makes a run repeatable

    def __post_init__(self):
        if not 0.0 < float(self.alpha) <= 1.0:
            raise ValueError(f"alpha must be in (0, 1], got {self.alpha}")
        if self.measurement_budget is None:
            self.measurement_budget = SIMULATOR_SHOTS if self.simulator_mode else HARDWARE_SHOTS
        self.measurement_budget = max(1, int(self.measurement_budget))

    @property
    def shots_per_circuit(self) -> int:
        return self.measurement_budget

    @property
    def measurement_rounds(self) -> int:
        return 1

    def evaluations(self, n_parameters: int) -> int:
        """Objective evaluations for ``n_parameters``; never fewer than COBYLA's first simplex needs."""
        wanted = self.max_cvar_iterations
        if wanted is None:
            wanted = max(MIN_EVALUATIONS, EVALUATIONS_PER_PARAMETER * (n_parameters + 1))
        return max(int(wanted), n_parameters + 2)


def diagonal_energies(hamiltonian: SparsePauliOp, keys: np.ndarray) -> np.ndarray:
    """Energy of each computational basis state under a Z-only Hamiltonian.

    ``keys`` are integers whose bit i is the measured value of qubit i.
    """
    if hamiltonian.paulis.x.any():
        raise ValueError("CVaR over sampled energies needs a diagonal (Z-only) Hamiltonian")
    zmask = hamiltonian.paulis.z.astype(np.int64)
    keys = np.asarray(keys, dtype=np.int64)
    bits = (keys[:, None] >> np.arange(zmask.shape[1], dtype=np.int64)) & 1
    parity = (bits @ zmask.T) & 1
    return (1 - 2 * parity) @ np.real(hamiltonian.coeffs)


def cvar_of_counts(energies: np.ndarray, counts: np.ndarray, alpha: float) -> Tuple[float, Dict[str, float]]:
    """Mean of the lowest ceil(alpha * K) of K shots, given distinct outcomes and their counts."""
    energies = np.asarray(energies, dtype=float)
    counts = np.asarray(counts, dtype=np.int64)
    total = int(counts.sum())
    if total <= 0:
        raise ValueError("no shots")
    order = np.argsort(energies, kind="stable")
    e, c = energies[order], counts[order]
    keep = max(1, math.ceil(alpha * total - 1e-12))
    before = np.cumsum(c) - c
    take = np.clip(keep - before, 0, c)
    cvar = float((e * take).sum() / keep)
    var = float(e[int(np.searchsorted(np.cumsum(c), keep))])
    mean = float((e * c).sum() / total)
    # Asymptotic standard error of the CVaR estimator: sd(min(E - VaR, 0)) / (alpha * sqrt(K)).
    tail = np.minimum(e - var, 0.0)
    tail_mean = float((tail * c).sum() / total)
    tail_var = float((((tail - tail_mean) ** 2) * c).sum() / max(total - 1, 1))
    se = math.sqrt(tail_var) / ((keep / total) * math.sqrt(total))
    return cvar, {
        "cvar_value": cvar,
        "var_value": var,
        "standard_expectation": mean,
        "minimum_sampled": float(e[0]),
        "n_samples": total,
        "worst_fraction_size": int(keep),
        "alpha_used": float(alpha),
        "cvar_std_error": se,
        "cvar_confidence_interval": [cvar - 1.959964 * se, cvar + 1.959964 * se],
    }


class CVaRVQEOptimizer:
    """Minimise the CVaR of the sampled energies of a parameterised circuit."""

    def __init__(self, config: CVaRConfig = None, use_real_hardware: bool = False, backend=None):
        self.config = config or CVaRConfig()
        self.use_real_hardware = bool(use_real_hardware)
        self.simulator = backend if backend is not None else self._resolve_backend()
        self.pm = generate_preset_pass_manager(backend=self.simulator, optimization_level=1)
        self._backend_name = str(getattr(self.simulator, "name", self.simulator))
        self._is_real_ibm = backend is None and not isinstance(self.simulator, AerSimulator)
        self.vendor_runs: list[dict] = []

        self.energy_history: list[float] = []
        self.cvar_history: list[float] = []
        self.alpha_history: list[float] = []
        self.params_history: list[list[float]] = []
        self.wall_history: list[float] = []
        self.convergence_metrics: Dict[str, Any] = {}

        seed = self.config.seed
        self.rng = np.random.default_rng(42 if seed is None else seed)
        self._shot_seeds = None if seed is None else np.random.default_rng([int(seed), 1])
        self._isa: Optional[QuantumCircuit] = None
        self._hamiltonian: Optional[SparsePauliOp] = None

        logger.info("[CVaR-VQE] %s alpha=%s shots_per_evaluation=%d backend=%s",
                    "SIMULATOR" if self.config.simulator_mode else "PRODUCTION",
                    self.config.alpha, self.config.shots_per_circuit, self._backend_name)

    def _resolve_backend(self):
        if not (self.use_real_hardware or not self.config.simulator_mode):
            return AerSimulator()
        from core.optimizers.quantum.shared.circuit_runner import HardwareUnavailableError
        try:
            from core.quantum_credential_context import tenant_token_for
            token = tenant_token_for("ibm_quantum") or os.getenv("IBM_QUANTUM_TOKEN")
        except Exception:  # noqa: BLE001
            token = os.getenv("IBM_QUANTUM_TOKEN")
        if not token:
            raise HardwareUnavailableError("CVaR-VQE on IBM hardware needs an IBM Quantum credential; none was resolved")
        try:
            from qiskit_ibm_runtime import QiskitRuntimeService
            service = QiskitRuntimeService(channel="ibm_quantum_platform", token=token)
            backend = service.least_busy(min_num_qubits=127, simulator=False)
            logger.info("[CVaR-VQE] using IBM backend %s", getattr(backend, "name", backend))
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
        if not jid:
            return
        from datetime import datetime, timezone
        self.vendor_runs.append({
            "vendor": "ibm", "vendor_label": "IBM Quantum", "job_id": jid, "stage": stage,
            "backend": self._backend_name, "submitted_at": datetime.now(timezone.utc).isoformat(),
        })

    # ------------------------------------------------------------------ sampling and objective

    def prepare(self, ansatz: QuantumCircuit, hamiltonian: SparsePauliOp) -> None:
        """Transpile the measured ansatz once; parameters are bound per evaluation."""
        if hamiltonian.paulis.x.any():
            raise ValueError("CVaR over sampled energies needs a diagonal (Z-only) Hamiltonian")
        measured = ansatz.copy()
        if not measured.cregs:
            measured.measure_all()
        self._isa = self.pm.run(measured)
        self._hamiltonian = hamiltonian

    def sample_counts(self, params: np.ndarray, shots: int, stage: str = "sampler") -> Dict[int, int]:
        """Counts of the measured ansatz at ``params``; key bit i is qubit i."""
        if self._isa is None:
            raise RuntimeError("prepare() must be called before sampling")
        sampler = Sampler(mode=self.simulator)
        if self._shot_seeds is not None and not self._is_real_ibm:
            sampler.options.simulator.seed_simulator = int(self._shot_seeds.integers(1, 2 ** 31 - 1))
        job = sampler.run([(self._isa, np.asarray(params, dtype=float))], shots=int(shots))
        self._capture_runtime_job(job, stage=stage)
        data = job.result()[0].data
        register = getattr(data, "meas", None)
        if register is None:
            register = next(iter(data.values()))
        return register.get_int_counts()

    def evaluate(self, params: np.ndarray, shots: Optional[int] = None,
                 stage: str = "sampler") -> Tuple[float, Dict[str, Any]]:
        """CVaR of the energies of the bitstrings sampled at ``params``."""
        counts = self.sample_counts(params, shots or self.config.shots_per_circuit, stage=stage)
        keys = np.fromiter(counts.keys(), dtype=np.int64, count=len(counts))
        weights = np.fromiter(counts.values(), dtype=np.int64, count=len(counts))
        energies = diagonal_energies(self._hamiltonian, keys)
        cvar, metrics = cvar_of_counts(energies, weights, self.config.alpha)
        metrics["distinct_bitstrings"] = int(len(counts))
        return cvar, metrics

    def cvar_cost_function(self, params: np.ndarray, ansatz: QuantumCircuit,
                           hamiltonian: SparsePauliOp) -> float:
        if self._isa is None:
            self.prepare(ansatz, hamiltonian)
        return self.evaluate(params)[0]

    # ------------------------------------------------------------------ optimisation

    def analyze_convergence(self) -> Dict[str, Any]:
        if len(self.cvar_history) < 5:
            return {"status": "insufficient_data", "is_converged": False,
                    "total_iterations": len(self.cvar_history)}
        cvar = np.asarray(self.cvar_history)
        recent = cvar[-5:]
        slope = float(np.polyfit(np.arange(len(cvar)), cvar, 1)[0]) if len(cvar) >= 10 else 0.0
        analysis = {
            "is_converged": bool(recent.std() < self.config.energy_threshold),
            "total_iterations": int(len(cvar)),
            "cvar_improvement": float(cvar[0] - cvar.min()),
            "recent_std": float(recent.std()),
            "cvar_slope": slope,
            "best_cvar": float(cvar.min()),
            "best_standard_energy": float(np.min(self.energy_history)),
            "alpha_adaptation_active": False,
            "current_alpha": float(self.config.alpha),
        }
        self.convergence_metrics = analysis
        return analysis

    def enhanced_vqe_optimization(self, ansatz: QuantumCircuit, hamiltonian: SparsePauliOp,
                                  initial_params: np.ndarray = None,
                                  classical_optimizer: str = "COBYLA") -> Dict[str, Any]:
        started = time.time()
        self.prepare(ansatz, hamiltonian)
        n_parameters = ansatz.num_parameters
        if initial_params is None:
            initial_params = self.rng.uniform(-np.pi, np.pi, n_parameters)
        budget = self.config.evaluations(n_parameters)
        budget_raised = self.config.max_cvar_iterations is not None and budget > int(self.config.max_cvar_iterations)
        if budget_raised:
            logger.warning("[CVaR-VQE] %d evaluations requested, %d parameters: raised to %d "
                           "(the optimiser's first simplex alone needs %d)",
                           self.config.max_cvar_iterations, n_parameters, budget, n_parameters + 1)

        self.energy_history, self.cvar_history, self.alpha_history = [], [], []
        self.params_history, self.wall_history = [], []
        last = [time.perf_counter()]

        def objective(params):
            cvar, metrics = self.evaluate(params, stage=f"sampler #{len(self.cvar_history) + 1}")
            self.cvar_history.append(cvar)
            self.energy_history.append(metrics["standard_expectation"])
            self.alpha_history.append(float(self.config.alpha))
            self.params_history.append([float(p) for p in params])
            now = time.perf_counter()
            self.wall_history.append(round(now - last[0], 6))
            last[0] = now
            return cvar

        options: Dict[str, Any] = {"maxiter": budget}
        if classical_optimizer.upper() == "COBYLA":
            options["rhobeg"] = 0.5 if self.config.simulator_mode else 0.25
        outcome = minimize(objective, np.asarray(initial_params, dtype=float),
                           method=classical_optimizer, options=options)
        evaluations = len(self.cvar_history)

        final_cvar, final_metrics = self.evaluate(outcome.x, stage="sampler final evaluation")
        shots = self.config.shots_per_circuit
        result = {
            "success": bool(np.isfinite(final_cvar)),
            "converged": bool(outcome.success),
            "cvar_energy": final_cvar,
            "standard_energy": final_metrics["standard_expectation"],
            "optimal_parameters": [float(p) for p in outcome.x],
            "optimization_iterations": evaluations,
            "optimization_time": time.time() - started,
            "convergence_analysis": self.analyze_convergence(),
            "final_evaluation_metrics": {
                "cvar_energy": final_cvar,
                "standard_energy": final_metrics["standard_expectation"],
                "alpha_used": float(self.config.alpha),
                "total_shots": shots,
                "measurement_rounds": 1,
                "cvar_metrics": final_metrics,
            },
            "cvar_improvement_vs_standard": final_metrics["standard_expectation"] - final_cvar,
            "method": "CVaR-VQE",
            "cvar_history": [float(v) for v in self.cvar_history],
            "energy_history": [float(v) for v in self.energy_history],
            "alpha_history": [float(v) for v in self.alpha_history],
            "params_history": [list(p) for p in self.params_history],
            "wall_history": [float(w) for w in self.wall_history],
            "configuration": {
                "alpha": float(self.config.alpha),
                "alpha_schedule": "fixed",
                "objective": "cvar_of_sampled_bitstring_energies",
                "measurement_budget": shots,
                "evaluation_budget": budget,
                "evaluation_budget_raised": bool(budget_raised),
                "seed": self.config.seed,
            },
            "resource_usage": {
                "total_measurements": shots * (evaluations + 1),
                "measurement_rounds": evaluations,
                "parameter_count": int(n_parameters),
                "circuit_depth": int(ansatz.depth()),
            },
        }
        logger.info("[CVaR-VQE] %d evaluations in %.2fs, CVaR=%.6f, mean=%.6f",
                    evaluations, result["optimization_time"], final_cvar, result["standard_energy"])
        return result
