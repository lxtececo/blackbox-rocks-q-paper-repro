"""Simulator-mode defaults of the four portfolio selection methods.

What a job gets when its request carries no method configuration. One table,
read by the worker executors and checked by experiment E20
(reports/paper/quantumfpo-arxiv/experiments/e20_portfolio_corrected.py), which
stops if the values it evaluates differ from these. A value changed here is a
value that has not been measured.

QAOA and BF-DCQO: the executor values that E19 measured. CVaR-VQE and PCE:
chosen on held-out instances (results/e20_tuning.json and
results/e20_tuning_followup.json next to the experiment).
"""

# Largest number of assets at which the four methods have been measured against
# the exact optimum and a null (E19, E20). The executors that give a method one
# qubit or one variable per asset cut a larger universe down to this size first.
MAX_MEASURED_ASSETS = 16

SIMULATOR_DEFAULTS = {
    "qaoa": {
        "risk_aversion": 0.5,
        "estimator_shots": 100,      # per objective evaluation
        "sampler_shots": 200,        # final sample the selection is taken from
        "max_iterations": 15,
        "reps": 2,
    },
    "cvar_vqe": {
        "alpha": 0.25,
        "max_cvar_iterations": None,  # None: ten objective evaluations per parameter, at least 100
        "measurement_budget": 1024,   # shots per objective evaluation
        "risk_aversion": 0.5,
        "ansatz_depth": 2,
        "sampling_shots": 2048,       # final sample the selection is taken from
    },
    "bfdcqo": {
        "qmax": 8,
        "n_iterations": 3,
        "shots": 8192,
        "ansatz_depth": 2,
        "noise_model": "aria-1",
        "risk_aversion": 0.5,
        "n_candidates": 100,
    },
    "pce": {
        "compression_factor": 3,
        "ansatz_depth": 3,
        "shots": 1024,                # per measurement setting
        "max_iterations": 6,          # optimiser iterations per round
        "max_no_improvement_rounds": 6,
        "restart_threshold": 3,
        "optimization_timeout": 300.0,
        "risk_aversion": 0.5,
    },
}
