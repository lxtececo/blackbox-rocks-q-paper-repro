"""E8b — Two-qubit gate cost of the MLAE circuits on the actual IBM device.

Transpiles each Grover power for the backend used in E8 (heavy-hex coupling,
native gate set, optimisation level 3 as the platform does) and reports the
two-qubit gate count and an error-free probability from the device's own
reported median two-qubit error. No QPU time is used.
"""

from __future__ import annotations

import json
import math
import os
from statistics import median

from common import RESULTS, save
from e8_ibm_hardware import load_token


def main() -> None:
    load_token()
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from qiskit_ibm_runtime import QiskitRuntimeService

    from core.optimizers.montecarlo.grid import build_price_grid
    from core.optimizers.montecarlo.qae import _measurement_circuits, exponential_schedule

    hw = json.loads((RESULTS / "e8_ibm_hardware_mlae.json").read_text())
    backend_name = hw["mlae"][0]["backend"]
    svc = QiskitRuntimeService(channel="ibm_quantum_platform", token=os.environ["IBM_QUANTUM_TOKEN"])
    backend = svc.backend(backend_name)
    target = backend.target
    two_q = [op for op in ("cz", "ecr", "cx") if op in target.operation_names]
    errs = [p.error for op in two_q for p in target[op].values() if p is not None and p.error is not None]
    eps = median(errs)
    pm = generate_preset_pass_manager(optimization_level=3, backend=backend, seed_transpiler=7)

    grid = build_price_grid(100.0, 100.0, 0.05, 0.2, 1.0, "call", 5)
    powers = exponential_schedule(8)
    rows = []
    for k, circ in zip(powers, _measurement_circuits(grid, powers)):
        isa = pm.run(circ)
        ops = isa.count_ops()
        n2 = sum(int(ops.get(op, 0)) for op in two_q)
        rows.append({"grover_power": k, "two_qubit_gates": n2, "depth": isa.depth(),
                     "p_error_free": math.exp(n2 * math.log1p(-eps))})
        print(f"k={k}: {n2} two-qubit gates, depth {isa.depth()}, P(no 2q error) = {rows[-1]['p_error_free']:.2e}")

    measured = hw["mlae"][-1]
    shots = 200
    rows_measured = [{"grover_power": k, "p1_measured": h / shots}
                     for k, h in zip(measured["grover_schedule"], measured["hits_per_power"])]
    theta = math.asin(math.sqrt(sum(p * v for p, v in zip(grid.probabilities, grid.normalized_payoffs))))
    for r in rows_measured:
        r["p1_ideal"] = math.sin((2 * r["grover_power"] + 1) * theta) ** 2
    save("e8b_ibm_transpiled_cost", {
        "backend": backend_name, "native_two_qubit_ops": two_q, "median_two_qubit_error": eps,
        "rows": rows, "measured_vs_ideal": rows_measured,
    })


if __name__ == "__main__":
    main()
