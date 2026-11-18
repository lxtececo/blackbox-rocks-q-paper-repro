"""
Conditional Value-at-Risk VQE (CVaR-VQE) Enhancement Module
Based on "A Comparative Study of Quantum Optimization Techniques" (arXiv:2503.12121v2)

Implements CVaR-VQE approach for improved convergence and noise resilience:
1. CVaR aggregation with confidence level α=0.25
2. Focus on lowest-energy measurement outcomes  
3. Enhanced noise resilience for NISQ devices
4. Adaptive confidence level tuning
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import time
from scipy.stats import norm
from collections import defaultdict

from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import SamplerV2 as Sampler, EstimatorV2 as Estimator


@dataclass
class CVaRConfig:
    """Configuration for CVaR-VQE optimization"""
    # Execution mode — set True for Aer simulator, False for real IBM hardware
    simulator_mode: bool = True      # True = simulator (fast/cheap), False = production (real QPU)

    # CVaR Parameters
    alpha: float = 0.25              # Confidence level (0 < α ≤ 1)
    adaptive_alpha: bool = True      # Enable adaptive α tuning
    alpha_min: float = 0.1           # Minimum confidence level
    alpha_max: float = 0.5           # Maximum confidence level

    # Convergence Enhancement
    energy_threshold: float = 1e-6   # Energy convergence criterion
    # max_cvar_iterations: simulator default 20, production default 100 (overridden in __post_init__)
    max_cvar_iterations: int = None  # None → resolved in __post_init__ based on simulator_mode
    # measurement_budget (shots): simulator default 1000, production default 8000
    measurement_budget: int = None   # None → resolved in __post_init__ based on simulator_mode

    # Noise Handling
    noise_mitigation: bool = True    # Enable noise mitigation
    statistical_analysis: bool = True # Enable statistical convergence analysis
    confidence_interval: float = 0.95 # Statistical confidence interval

    def __post_init__(self):
        """Apply mode-aware defaults for unset fields."""
        if self.max_cvar_iterations is None:
            # Simulator: keep low for interactive demo latency. COBYLA does
            # ~2-3 cost evaluations per iter; at 23 qubits each Aer Estimator
            # call is ~2-3s, so 8 iters keeps a full run well under a minute.
            self.max_cvar_iterations = 8 if self.simulator_mode else 100
        if self.measurement_budget is None:
            self.measurement_budget = 1000 if self.simulator_mode else 8000

    @property
    def shots_per_circuit(self) -> int:
        """Derive per-circuit shot count from measurement_budget and mode.

        Simulator: keep shots low; production: up to measurement_budget / 5
        but never exceeding 2000 per circuit (hardware queue cost control).
        """
        if self.simulator_mode:
            return min(100, self.measurement_budget)
        return min(2000, max(100, self.measurement_budget // 5))

    @property
    def measurement_rounds(self) -> int:
        """Number of independent measurement rounds for CVaR aggregation."""
        return 1 if self.simulator_mode else max(3, min(10, self.measurement_budget // max(1, self.shots_per_circuit)))


class CVaRVQEOptimizer:
    """
    Conditional Value-at-Risk Variational Quantum Eigensolver
    
    Enhances standard VQE by focusing on the lowest α fraction of measurement outcomes,
    providing better convergence and noise resilience for portfolio optimization.
    """
    
    def __init__(self, config: CVaRConfig = None, use_real_hardware: bool = False):
        self.config = config or CVaRConfig()
        self.use_real_hardware = bool(use_real_hardware)

        # F24 Phase A.2 — real IBM Runtime backend wiring (parity with PCE).
        # Honour ``use_real_hardware`` *and* ``config.simulator_mode=False``;
        # either one promotes the inner backend off AerSimulator. Falls back
        # to AerSimulator cleanly if BYOK is unavailable.
        want_real = self.use_real_hardware or (not self.config.simulator_mode)
        if want_real:
            try:
                from core.quantum_credential_context import tenant_token_for
                ibm_token = tenant_token_for("ibm_quantum") or os.getenv('IBM_QUANTUM_TOKEN')
            except Exception:  # noqa: BLE001
                ibm_token = os.getenv('IBM_QUANTUM_TOKEN')
            if ibm_token:
                try:
                    from qiskit_ibm_runtime import QiskitRuntimeService
                    service = QiskitRuntimeService(channel='ibm_quantum_platform', token=ibm_token)
                    self.simulator = service.least_busy(min_num_qubits=127, simulator=False)
                    print(f"[CVaR-VQE] Using REAL IBM backend: {getattr(self.simulator, 'name', self.simulator)}")
                except Exception as e:
                    print(f"[CVaR-VQE] Real IBM hardware unavailable ({e}); falling back to AerSimulator")
                    self.simulator = AerSimulator()
                    self.use_real_hardware = False
            else:
                print("[CVaR-VQE] IBM_QUANTUM_TOKEN not set; using AerSimulator")
                self.simulator = AerSimulator()
                self.use_real_hardware = False
        else:
            self.simulator = AerSimulator()
        self.pm = generate_preset_pass_manager(backend=self.simulator, optimization_level=1)

        # F24 Phase A.2 — backend label + vendor-side IBM Runtime job IDs
        # for the JobHistoryView "View on IBM Quantum portal" deep-links.
        # Stays empty when AerSimulator is in use.
        self._backend_name = str(getattr(self.simulator, "name", self.simulator))
        self._is_real_ibm = not isinstance(self.simulator, AerSimulator)
        self.vendor_runs: list[dict] = []

        # CVaR optimization state
        self.energy_history = []
        self.cvar_history = []
        self.alpha_history = []
        # 7-QAP.25 — per-iter ansatz params + per-iter wallclock so the
        # P3 (parameter trajectory) and P9 (wallclock-per-iter) charts
        # have real data instead of only P1.
        self.params_history: list[list[float]] = []
        self.wall_history: list[float] = []
        self.convergence_metrics = {}

        # Random number generator for reproducibility
        self.rng = np.random.default_rng(seed=42)

        mode_label = "SIMULATOR" if self.config.simulator_mode else "PRODUCTION"
        print(f"[CVaR-VQE] Initialized [{mode_label}] α={self.config.alpha}, "
              f"max_iter={self.config.max_cvar_iterations}, "
              f"shots_per_circuit={self.config.shots_per_circuit}, "
              f"measurement_budget={self.config.measurement_budget}")

    def _capture_runtime_job(self, job, stage: str) -> None:
        """F24 Phase A.2 — record IBM Runtime job_id for vendor-portal deep-link.

        Only records when running against a real IBM backend (not AerSimulator).
        All errors are swallowed so a transient SDK quirk never breaks the
        optimisation loop.
        """
        if not self._is_real_ibm:
            return
        try:
            jid = job.job_id()
        except Exception:  # noqa: BLE001
            return
        if not jid:
            return
        try:
            from datetime import datetime, timezone
            self.vendor_runs.append({
                "vendor": "ibm",
                "vendor_label": "IBM Quantum",
                "job_id": jid,
                "stage": stage,
                "backend": self._backend_name,
                "submitted_at": datetime.now(timezone.utc).isoformat(),
            })
        except Exception:  # noqa: BLE001  # nosec B110 - best-effort cleanup; exception intentionally swallowed
            pass

    def calculate_cvar(self, energy_samples: np.ndarray, alpha: float = None) -> Tuple[float, Dict[str, float]]:
        """
        Calculate Conditional Value-at-Risk (CVaR) from energy samples
        
        CVaR_α(E) = (1/α) × E[E | E ≤ VaR_α(E)]
        
        Args:
            energy_samples: Array of energy measurements
            alpha: Confidence level (uses config default if None)
            
        Returns:
            CVaR value and statistical metrics
        """
        if alpha is None:
            alpha = self.config.alpha
        
        if len(energy_samples) == 0:
            return 0.0, {}
        
        # Sort energy samples (ascending)
        sorted_energies = np.sort(energy_samples)
        n_samples = len(sorted_energies)
        
        # Calculate VaR (Value-at-Risk) - α-quantile
        var_index = max(1, int(alpha * n_samples))
        var_value = sorted_energies[var_index - 1]
        
        # Calculate CVaR - expected value of worst α fraction
        worst_energies = sorted_energies[:var_index]
        cvar_value = np.mean(worst_energies)
        
        # Statistical analysis
        metrics = {
            'var_value': float(var_value),
            'cvar_value': float(cvar_value),
            'n_samples': n_samples,
            'alpha_used': alpha,
            'worst_fraction_size': len(worst_energies),
            'standard_expectation': float(np.mean(sorted_energies)),
            'cvar_improvement': float(np.mean(sorted_energies) - cvar_value)
        }
        
        # Add confidence interval for CVaR estimate
        if len(worst_energies) > 1:
            std_error = np.std(worst_energies) / np.sqrt(len(worst_energies))
            ci_factor = norm.ppf((1 + self.config.confidence_interval) / 2)
            metrics['cvar_std_error'] = float(std_error)
            metrics['cvar_confidence_interval'] = [
                float(cvar_value - ci_factor * std_error),
                float(cvar_value + ci_factor * std_error)
            ]
        
        return cvar_value, metrics
    
    def adaptive_alpha_tuning(self, energy_samples: np.ndarray, 
                            iteration: int) -> float:
        """
        Adaptive confidence level tuning based on convergence behavior
        
        Adjusts α dynamically to balance exploration vs exploitation
        """
        if not self.config.adaptive_alpha or len(energy_samples) < 10:
            return self.config.alpha
        
        # Analyze energy distribution characteristics
        energy_std = np.std(energy_samples)
        energy_range = np.max(energy_samples) - np.min(energy_samples)
        
        # Calculate sample efficiency metric
        n_samples = len(energy_samples)
        expected_samples_at_alpha = max(1, int(self.config.alpha * n_samples))
        
        # Adaptive logic
        if iteration < 20:
            # Early iterations: broader exploration (higher α)
            adaptive_alpha = min(self.config.alpha_max, 
                               self.config.alpha * (1.5 - iteration / 40))
        else:
            # Later iterations: focused exploitation (lower α)  
            if energy_std / (abs(np.mean(energy_samples)) + 1e-10) < 0.1:
                # Low variance - can use lower α
                adaptive_alpha = max(self.config.alpha_min, self.config.alpha * 0.8)
            else:
                # High variance - use higher α for stability
                adaptive_alpha = min(self.config.alpha_max, self.config.alpha * 1.2)
        
        # Ensure within bounds
        adaptive_alpha = np.clip(adaptive_alpha, self.config.alpha_min, self.config.alpha_max)
        
        self.alpha_history.append(adaptive_alpha)
        
        return adaptive_alpha
    
    def enhanced_energy_evaluation(self, circuit: QuantumCircuit,
                                 hamiltonian: SparsePauliOp,
                                 shots_per_circuit: int = None) -> Tuple[float, Dict[str, Any]]:
        """
        Enhanced energy evaluation with CVaR aggregation

        Performs multiple measurements and applies CVaR to get robust energy estimate.
        shots_per_circuit defaults to config.shots_per_circuit (mode-aware).
        """
        if shots_per_circuit is None:
            shots_per_circuit = self.config.shots_per_circuit
        try:
            # Prepare circuit for execution
            isa_circuit = self.pm.run(circuit)

            # Execute multiple measurement rounds for statistical analysis
            num_measurement_rounds = self.config.measurement_rounds
            all_energy_samples = []
            round_results = []
            
            for round_idx in range(num_measurement_rounds):
                # EstimatorV2 takes precision via the PUB, not a top-level
                # ``shots`` kwarg. Default shots can also be set on the
                # primitive; we pass precision = 1/sqrt(shots) per-PUB so
                # the per-round shot budget is honoured exactly.
                precision = float(1.0 / np.sqrt(max(int(shots_per_circuit), 1)))
                estimator = Estimator(mode=self.simulator)
                job = estimator.run(
                    [(isa_circuit, hamiltonian, [], precision)]
                )
                self._capture_runtime_job(job, stage=f"estimator round #{round_idx + 1}")
                result = job.result()

                # Extract energy value (EstimatorV2 returns ``evs`` as a
                # 0-d ndarray for a single observable).
                energy_value = float(np.asarray(result[0].data.evs).reshape(-1)[0])
                all_energy_samples.append(energy_value)

                round_results.append({
                    'round': round_idx,
                    'energy': float(energy_value),
                    'shots': shots_per_circuit
                })
            
            # Apply adaptive α tuning
            current_alpha = self.adaptive_alpha_tuning(
                np.array(all_energy_samples), 
                len(self.energy_history)
            )
            
            # Calculate CVaR energy
            cvar_energy, cvar_metrics = self.calculate_cvar(
                np.array(all_energy_samples), 
                current_alpha
            )
            
            # Enhanced metrics
            evaluation_metrics = {
                'cvar_energy': cvar_energy,
                'standard_energy': float(np.mean(all_energy_samples)),
                'energy_std': float(np.std(all_energy_samples)),
                'measurement_rounds': num_measurement_rounds,
                'total_shots': shots_per_circuit * num_measurement_rounds,
                'alpha_used': current_alpha,
                'cvar_metrics': cvar_metrics,
                'round_results': round_results
            }
            
            # Store for convergence analysis
            self.energy_history.append(float(np.mean(all_energy_samples)))
            self.cvar_history.append(cvar_energy)
            
            return cvar_energy, evaluation_metrics
            
        except Exception as e:
            print(f"[CVaR-VQE] Energy evaluation error: {e}")
            return float('inf'), {'error': str(e)}
    
    def analyze_convergence(self) -> Dict[str, Any]:
        """
        Analyze CVaR-VQE convergence behavior
        
        Provides detailed convergence statistics and recommendations
        """
        if len(self.cvar_history) < 5:
            return {'status': 'insufficient_data', 'recommendation': 'continue_optimization'}
        
        cvar_array = np.array(self.cvar_history)
        energy_array = np.array(self.energy_history)
        
        # Convergence analysis
        recent_cvar = cvar_array[-5:]
        cvar_improvement = cvar_array[0] - cvar_array[-1] if len(cvar_array) > 1 else 0
        recent_std = np.std(recent_cvar)
        
        # Trend analysis
        if len(cvar_array) >= 10:
            x = np.arange(len(cvar_array))
            cvar_slope = np.polyfit(x, cvar_array, 1)[0]
        else:
            cvar_slope = 0
        
        # Convergence determination
        is_converged = (recent_std < self.config.energy_threshold and 
                       abs(cvar_slope) < self.config.energy_threshold)
        
        # Generate recommendation
        if is_converged:
            recommendation = 'converged'
        elif cvar_improvement < 0:
            recommendation = 'increase_iterations'  # Getting worse
        elif recent_std > 0.01:
            recommendation = 'reduce_alpha'  # High variance
        else:
            recommendation = 'continue_optimization'
        
        convergence_analysis = {
            'is_converged': is_converged,
            'recommendation': recommendation,
            'total_iterations': len(cvar_array),
            'cvar_improvement': float(cvar_improvement),
            'recent_std': float(recent_std),
            'cvar_slope': float(cvar_slope),
            'best_cvar': float(np.min(cvar_array)),
            'best_standard_energy': float(np.min(energy_array)),
            'cvar_vs_standard_improvement': float(np.mean(energy_array) - np.mean(cvar_array)),
            'alpha_adaptation_active': self.config.adaptive_alpha,
            'current_alpha': self.alpha_history[-1] if self.alpha_history else self.config.alpha
        }
        
        self.convergence_metrics = convergence_analysis
        return convergence_analysis
    
    def cvar_cost_function(self, params: np.ndarray, ansatz: QuantumCircuit,
                          hamiltonian: SparsePauliOp) -> float:
        """
        CVaR-enhanced cost function for VQE optimization

        Returns CVaR energy instead of standard expectation value.
        Shot count is derived from config.shots_per_circuit (mode-aware).
        """
        try:
            # Assign parameters to ansatz
            circuit = ansatz.assign_parameters(params)

            # Get CVaR energy evaluation — uses mode-aware shots_per_circuit from config
            cvar_energy, metrics = self.enhanced_energy_evaluation(
                circuit,
                hamiltonian,
                shots_per_circuit=self.config.shots_per_circuit
            )
            
            # Optional: Add parameter regularization for noise resilience
            if self.config.noise_mitigation:
                param_penalty = 0.001 * np.sum(params**2)  # L2 regularization
                cvar_energy += param_penalty
            
            return cvar_energy
            
        except Exception as e:
            print(f"[CVaR-VQE] Cost function error: {e}")
            return float('inf')
    
    def enhanced_vqe_optimization(self, ansatz: QuantumCircuit, 
                                hamiltonian: SparsePauliOp,
                                initial_params: np.ndarray = None,
                                classical_optimizer: str = 'SLSQP') -> Dict[str, Any]:
        """
        Enhanced VQE optimization with CVaR aggregation
        
        Provides improved convergence and noise resilience compared to standard VQE
        """
        print(f"[CVaR-VQE] Starting enhanced optimization with α={self.config.alpha}")
        
        start_time = time.time()
        
        # Initialize parameters
        if initial_params is None:
            initial_params = self.rng.uniform(-np.pi, np.pi, ansatz.num_parameters)
        
        # Reset optimization state
        self.energy_history = []
        self.cvar_history = []
        self.alpha_history = []
        self.params_history = []
        self.wall_history = []
        
        # Define CVaR cost function
        _last_t = [time.perf_counter()]
        def objective(params):
            val = self.cvar_cost_function(params, ansatz, hamiltonian)
            # Capture params + per-iter wallclock alongside the cost.
            # `cvar_cost_function` already appended into `cvar_history`,
            # so we mirror its length here.
            try:
                self.params_history.append([float(p) for p in params])
            except Exception:
                self.params_history.append([])
            now = time.perf_counter()
            self.wall_history.append(round(now - _last_t[0], 6))
            _last_t[0] = now
            return val
        
        # Classical optimization with enhanced monitoring
        from scipy.optimize import minimize
        
        # COBYLA-specific options: a larger initial trust radius (rhobeg)
        # and a looser tolerance (rhoend) reduce the number of cost-function
        # evaluations, which matters for simulator-mode demos where each
        # evaluation triggers a full Aer Estimator run on ~20+ qubits.
        opt_options: Dict[str, Any] = {
            'maxiter': self.config.max_cvar_iterations,
            'disp': True,
        }
        if classical_optimizer.upper() == 'COBYLA':
            opt_options.update({
                'rhobeg': 0.5 if self.config.simulator_mode else 0.25,
                'catol': 1e-2,
            })

        optimization_result = minimize(
            objective,
            initial_params,
            method=classical_optimizer,
            options=opt_options,
        )
        
        # Final energy evaluation with detailed statistics
        # Production: use full measurement_budget for the final circuit; simulator: keep it cheap
        final_shots = self.config.measurement_budget if not self.config.simulator_mode else self.config.shots_per_circuit
        final_circuit = ansatz.assign_parameters(optimization_result.x)
        final_cvar_energy, final_metrics = self.enhanced_energy_evaluation(
            final_circuit,
            hamiltonian,
            shots_per_circuit=final_shots
        )
        
        # Convergence analysis
        convergence_analysis = self.analyze_convergence()
        
        optimization_time = time.time() - start_time
        
        # Compile comprehensive result. NB: not all scipy optimisers
        # populate ``nit`` (COBYLA only exposes ``nfev``); fall back so
        # gradient-free methods don't crash here.
        _nit = int(getattr(optimization_result, 'nit',
                           getattr(optimization_result, 'nfev', 0)))
        result = {
            'success': optimization_result.success,
            'cvar_energy': final_cvar_energy,
            'standard_energy': final_metrics.get('standard_energy', final_cvar_energy),
            'optimal_parameters': optimization_result.x.tolist(),
            'optimization_iterations': _nit,
            'optimization_time': optimization_time,
            'convergence_analysis': convergence_analysis,
            'final_evaluation_metrics': final_metrics,
            'cvar_improvement_vs_standard': final_metrics.get('cvar_metrics', {}).get('cvar_improvement', 0),
            'method': 'CVaR-VQE',
            # 7-QAP.23 Slice B — expose per-iteration trajectories so adapters
            # can render convergence + adaptive-α charts.
            'cvar_history': [float(v) for v in self.cvar_history],
            'alpha_history': [float(v) for v in self.alpha_history],
            # 7-QAP.25 — per-iter ansatz params + wallclock (P3 + P9).
            'params_history': [list(p) for p in self.params_history],
            'wall_history': [float(w) for w in self.wall_history],
            'configuration': {
                'alpha': self.config.alpha,
                'adaptive_alpha': self.config.adaptive_alpha,
                'measurement_budget': self.config.measurement_budget,
                'noise_mitigation': self.config.noise_mitigation
            },
            'resource_usage': {
                'total_measurements': sum([m.get('total_shots', 0) for m in final_metrics.get('round_results', [])]),
                'measurement_rounds': len(self.cvar_history),
                'parameter_count': len(optimization_result.x),
                'circuit_depth': ansatz.depth()
            }
        }
        
        print(f"[CVaR-VQE] Optimization complete: {optimization_time:.2f}s, "
              f"CVaR_energy={final_cvar_energy:.6f}, iterations={_nit}")
        
        return result
    
    def compare_with_standard_vqe(self, ansatz: QuantumCircuit,
                                 hamiltonian: SparsePauliOp,
                                 initial_params: np.ndarray = None) -> Dict[str, Any]:
        """
        Compare CVaR-VQE performance with standard VQE
        
        Provides side-by-side comparison for method validation
        """
        print("[CVaR-VQE] Running comparative analysis with standard VQE")
        
        # Run CVaR-VQE
        cvar_result = self.enhanced_vqe_optimization(ansatz, hamiltonian, initial_params)
        
        # Run standard VQE for comparison
        def standard_vqe_cost(params):
            circuit = ansatz.assign_parameters(params)
            isa_circuit = self.pm.run(circuit)

            estimator = Estimator(mode=self.simulator)
            # Use mode-aware shot count from config
            shots = self.config.shots_per_circuit
            job = estimator.run([(isa_circuit, hamiltonian)], shots=shots)
            self._capture_runtime_job(job, stage="standard-vqe-compare")
            result = job.result()
            
            return result[0].data.evs
        
        from scipy.optimize import minimize
        
        standard_result = minimize(
            standard_vqe_cost,
            initial_params if initial_params is not None else self.rng.uniform(-np.pi, np.pi, ansatz.num_parameters),
            method='SLSQP',
            options={'maxiter': self.config.max_cvar_iterations}
        )
        
        # Comparative analysis
        comparison = {
            'cvar_vqe_result': cvar_result,
            'standard_vqe_result': {
                'energy': float(standard_result.fun),
                'iterations': standard_result.nit,
                'success': standard_result.success
            },
            'improvement_metrics': {
                'energy_improvement': float(standard_result.fun - cvar_result['cvar_energy']),
                'convergence_advantage': cvar_result['convergence_analysis']['is_converged'],
                'noise_resilience': cvar_result['configuration']['noise_mitigation'],
                'statistical_robustness': len(cvar_result['final_evaluation_metrics']['round_results'])
            },
            'recommendation': 'CVaR-VQE' if cvar_result['cvar_energy'] < standard_result.fun else 'Standard-VQE'
        }
        
        print(f"[CVaR-VQE] Comparison complete - Energy improvement: "
              f"{comparison['improvement_metrics']['energy_improvement']:.6f}")
        
        return comparison


# Integration functions for existing portfolio optimization system

def enhance_dynamic_portfolio_with_cvar(config, hamiltonian, ansatz):
    """
    Enhance existing dynamic portfolio optimization with CVaR-VQE
    
    Drop-in replacement for standard VQE optimization in enhanced_dynamic_portfolio_opt.py
    """
    cvar_config = CVaRConfig(
        alpha=0.25,
        adaptive_alpha=True,
        max_cvar_iterations=config.num_generations * 2,  # More iterations for CVaR
        measurement_budget=config.sampler_shots * 10,    # Higher measurement budget
        noise_mitigation=True
    )
    
    cvar_optimizer = CVaRVQEOptimizer(cvar_config)
    
    # Run enhanced optimization
    result = cvar_optimizer.enhanced_vqe_optimization(ansatz, hamiltonian)
    
    # Convert to expected format for portfolio system
    portfolio_result = {
        'success': result['success'],
        'optimal_energy': result['cvar_energy'],
        'optimal_parameters': result['optimal_parameters'],
        'iterations': result['optimization_iterations'],
        'convergence_metrics': result['convergence_analysis'],
        'method': 'CVaR-VQE_Enhanced',
        'cvar_advantage': result.get('cvar_improvement_vs_standard', 0)
    }
    
    return portfolio_result


def create_cvar_enhanced_cost_function(hamiltonian, cvar_config: CVaRConfig = None):
    """
    Create CVaR-enhanced cost function for integration with existing optimizers
    """
    if cvar_config is None:
        cvar_config = CVaRConfig()
    
    cvar_optimizer = CVaRVQEOptimizer(cvar_config)
    
    def cvar_cost_function(params, ansatz):
        return cvar_optimizer.cvar_cost_function(params, ansatz, hamiltonian)
    
    return cvar_cost_function, cvar_optimizer


# Example usage and testing
if __name__ == "__main__":
    """
    Example CVaR-VQE optimization for testing
    """
    from qiskit.circuit.library import TwoLocal
    import numpy as np
    
    # Create test Hamiltonian (simple portfolio optimization)
    num_qubits = 4
    coeffs = [1.0, -0.5, -0.3, 0.2, 0.1]
    pauli_strings = ['IIII', 'ZIII', 'IZII', 'IIZI', 'IIIZ']
    hamiltonian = SparsePauliOp.from_list(list(zip(pauli_strings, coeffs)))
    
    # Create test ansatz
    ansatz = TwoLocal(num_qubits, 'ry', 'cz', reps=2)
    
    # Run CVaR-VQE optimization
    cvar_config = CVaRConfig(alpha=0.25, max_cvar_iterations=50)
    optimizer = CVaRVQEOptimizer(cvar_config)
    
    result = optimizer.enhanced_vqe_optimization(ansatz, hamiltonian)
    
    print("\n=== CVaR-VQE Test Result ===")
    print(f"Success: {result['success']}")
    print(f"CVaR Energy: {result['cvar_energy']:.6f}")
    print(f"Standard Energy: {result['standard_energy']:.6f}")
    print(f"CVaR Improvement: {result['cvar_improvement_vs_standard']:.6f}")
    print(f"Convergence: {result['convergence_analysis']['is_converged']}")
    print(f"Optimization Time: {result['optimization_time']:.2f}s")