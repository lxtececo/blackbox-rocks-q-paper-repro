"""Shared execution substrate for measured circuits.

The finance paradigm methods each grew their own copy of "pick a backend,
transpile, run, read counts". The fleet, network and Monte Carlo verticals
need the same thing, so it lives here once instead of three more times.

The contract is deliberately narrow: hand over a list of circuits that
already carry their own measurements, get back per-circuit counts plus the
provenance needed to prove where they ran. Anything richer (Estimator
expectation values, session batching, error mitigation knobs) belongs to
the caller that needs it.

Execution modes
---------------
``simulator``  Local Qiskit Aer. Seeded, free, state-vector bound.
``real``       Vendor hardware via the tenant's BYOK credential. IBM goes
               through Qiskit Runtime ``SamplerV2``; IonQ through the
               ``qiskit-ionq`` provider.

Real-hardware runs submit every circuit as ONE job wherever the provider
allows it. A QAOA outer loop that queues one job per objective evaluation
would spend hours in a device queue for a solve worth seconds of QPU time.
"""

from __future__ import annotations

import logging
import math
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

from core.vertical_hardware import SIM_BYOK_PLATFORMS

logger = logging.getLogger(__name__)

#: Platforms that mean "no device, run it locally".
LOCAL_PLATFORMS = frozenset({"", "cpu", "local", "aer", "simulator", "none"})

#: Platform -> BYOK vendor name in ``tenant_quantum_credentials``.
PLATFORM_VENDOR = {
    "ibm": "ibm_quantum",
    "ionq": "ionq",
}


class HardwareUnavailableError(RuntimeError):
    """Real-hardware execution was requested but cannot be performed.

    Separate from a generic ``RuntimeError`` so callers can decide between
    failing the job and degrading to a simulator, and so the message can be
    surfaced verbatim to the user - these are almost always actionable
    (missing credential, device too small, provider outage).
    """


@dataclass
class CircuitRunResult:
    """Counts plus the provenance that proves where they came from."""

    counts: List[Dict[str, int]]
    backend_name: str
    platform: str
    execution_mode: str
    shots: int
    elapsed_ms: int = 0
    #: Vendor-side job identifiers, so Job History can deep-link into the
    #: provider's portal. Empty for simulator runs.
    vendor_runs: List[Dict[str, Any]] = field(default_factory=list)
    #: Qubit count after transpilation onto the device coupling map. Real
    #: devices route through SWAPs, so this exceeds the logical width.
    physical_qubits: Optional[int] = None

    def provenance(self) -> Dict[str, Any]:
        return {
            "backend_name": self.backend_name,
            "platform": self.platform,
            "execution_mode": self.execution_mode,
            "shots": self.shots,
            "vendor_runs": self.vendor_runs,
            "physical_qubits": self.physical_qubits,
        }


def resolve_vendor_token(vendor: str, env_var: str) -> Optional[str]:
    """Tenant BYOK credential first, pod-wide env var second."""
    try:
        from core.quantum_credential_context import tenant_token_for  # noqa: PLC0415

        token = tenant_token_for(vendor)
    except Exception:  # noqa: BLE001 - context module is optional at import time
        token = None
    return token or os.getenv(env_var)


def resolve_tenant_token(vendor: str) -> Optional[str]:
    """Tenant BYOK credential only, never the pod-wide platform key.

    The cloud-simulator path uses this so a simulator run can never spend the
    Enterprise platform pool. Mirrors ``_SIM_BYOK_VENDORS`` in the finance
    dispatcher.
    """
    try:
        from core.quantum_credential_context import tenant_token_for  # noqa: PLC0415

        return tenant_token_for(vendor)
    except Exception:  # noqa: BLE001 - context module is optional at import time
        return None


def _creg_name(circuit: Any) -> str:
    cregs = getattr(circuit, "cregs", None)
    if cregs:
        return cregs[0].name
    return "c"


def _counts_from_databin(data: Any, creg: str) -> Dict[str, int]:
    """Read counts out of a SamplerV2 ``DataBin``.

    The register is addressed by name because a circuit built with an
    explicit ``ClassicalRegister`` does not use the default ``c``, and
    guessing wrong yields an empty histogram rather than an error.
    """
    register = getattr(data, creg, None)
    if register is None:
        # Single-register circuits are the norm; fall back to whatever the
        # only field is before giving up.
        fields = [k for k in getattr(data, "keys", lambda: [])()]
        if len(fields) == 1:
            register = getattr(data, fields[0])
    if register is None:
        raise HardwareUnavailableError(
            f"sampler result carries no classical register named '{creg}'"
        )
    return {str(k): int(v) for k, v in register.get_counts().items()}


# ---------------------------------------------------------------------------
# Simulator path
# ---------------------------------------------------------------------------


def _cgroup_cpu_quota() -> Optional[int]:
    """CPU count this process may actually use, or ``None`` if unlimited.

    ``os.cpu_count()`` reports the host's cores, not the cgroup's quota, so
    inside a container it is wrong by the ratio between the node and the pod
    limit. Reads cgroup v2 first, then v1.
    """
    for path, parse in (
        ("/sys/fs/cgroup/cpu.max", lambda t: t.split()),
        ("/sys/fs/cgroup/cpu/cpu.cfs_quota_us", None),
    ):
        try:
            with open(path, encoding="utf-8") as fh:
                text = fh.read().strip()
        except OSError:
            continue
        try:
            if parse is not None:
                quota_s, period_s = parse(text)
                if quota_s == "max":
                    return None
                quota, period = int(quota_s), int(period_s)
            else:
                quota = int(text)
                with open(
                    "/sys/fs/cgroup/cpu/cpu.cfs_period_us", encoding="utf-8"
                ) as fh:
                    period = int(fh.read().strip())
                if quota <= 0:
                    return None
            if period > 0:
                return max(1, quota // period)
        except (ValueError, OSError):
            continue
    return None


def _aer_threads() -> Optional[int]:
    """Thread cap for Aer, honouring an explicit override then the cgroup."""
    override = os.getenv("AER_MAX_THREADS", "").strip()
    if override:
        try:
            return max(1, int(override))
        except ValueError:
            logger.warning("[circuit-runner] ignoring non-numeric AER_MAX_THREADS=%r", override)
    quota = _cgroup_cpu_quota()
    if quota is None:
        return None
    return min(quota, os.cpu_count() or quota)


def _cgroup_memory_limit() -> Optional[int]:
    """Bytes of RAM this process may actually use, or ``None`` if unlimited.

    Same blind spot as the CPU quota: ``SC_PHYS_PAGES`` reports the node, not
    the pod. Reads cgroup v2 first, then v1, then the host.
    """
    for path in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            with open(path, encoding="utf-8") as fh:
                text = fh.read().strip()
        except OSError:
            continue
        if text == "max":
            return None
        try:
            value = int(text)
        except ValueError:
            continue
        # cgroup v1 spells "unlimited" as a near-word-size sentinel.
        if value <= 0 or value >= 2**62:
            return None
        return value
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        return None


#: A complex128 amplitude, and the working copy Aer holds while it applies a
#: gate. The rest of the pod's RAM has to hold the QUBO, the interpreter and
#: the runtime, so only a fraction of the limit is ours to spend.
_AMPLITUDE_BYTES = 16
_STATEVECTOR_COPIES = 2
_STATEVECTOR_HEADROOM = 0.45


def statevector_qubit_cap() -> int:
    """Widest state vector that fits in this container's memory limit.

    This is the hard physical bound: it says nothing about how long the
    simulation would take, so callers that need to stay interactive apply a
    time budget on top.
    """
    override = os.getenv("STATEVECTOR_MAX_QUBITS", "").strip()
    if override:
        try:
            return max(1, int(override))
        except ValueError:
            logger.warning(
                "[circuit-runner] ignoring non-numeric STATEVECTOR_MAX_QUBITS=%r", override
            )
    limit = _cgroup_memory_limit()
    if limit is None:
        limit = 4 * 1024**3
    budget = limit * _STATEVECTOR_HEADROOM
    amplitudes = budget / (_AMPLITUDE_BYTES * _STATEVECTOR_COPIES)
    if amplitudes < 2:
        return 1
    return max(1, int(math.floor(math.log2(amplitudes))))


#: Advertised widths of the devices we dispatch to, by backend name. Used to
#: refuse a circuit the device could never accept before it reaches a queue.
#: IBM picks a backend dynamically, so its entry is the narrowest of the
#: machines `least_busy` can return rather than a specific one.
_DEVICE_QUBITS = {
    "ionq_simulator": 29,
    "ionq_qpu.aria-1": 25,
    "ionq_qpu.aria-2": 25,
    "ionq_qpu.forte-1": 36,
    "ionq_qpu.forte-enterprise-1": 36,
}
_IBM_MIN_DEVICE_QUBITS = 127


def ionq_qpu_backend() -> str:
    """IonQ QPU for hardware runs. IonQ retired Aria-1/2, so default to Forte-1."""
    return os.getenv("IONQ_BACKEND", "ionq_qpu.forte-1").strip()


def device_qubit_width(platform: Optional[str], *, hardware: bool = True) -> Optional[int]:
    """Qubit count of the device a request would land on, if we know it."""
    key = (platform or "").strip().lower()
    if not key:
        return None
    if key.startswith("ibm"):
        override = os.getenv("IBM_DEVICE_QUBITS", "").strip()
        if override.isdigit():
            return int(override)
        return _IBM_MIN_DEVICE_QUBITS
    if key == "ionq":
        backend = ionq_qpu_backend() if hardware else "ionq_simulator"
        return _DEVICE_QUBITS.get(backend.strip().lower())
    return _DEVICE_QUBITS.get(key)


def _run_aer(circuits: Sequence[Any], shots: int, seed: int) -> List[Dict[str, int]]:
    from qiskit import transpile  # noqa: PLC0415
    from qiskit_aer import AerSimulator  # noqa: PLC0415

    # Aer sizes its OpenMP pool from the HOST core count. In a 2-CPU pod on a
    # 32-core node that is 32 threads fighting over 2 cores, and the thrash is
    # not marginal: a 16-qubit QAOA layer measured 6.41s unpinned against
    # 0.015s pinned, and 20 qubits 8.94s against 0.23s. Pin it to the quota.
    options: Dict[str, Any] = {"seed_simulator": seed}
    threads = _aer_threads()
    if threads is not None:
        options["max_parallel_threads"] = threads
        options["max_parallel_experiments"] = 1

    backend = AerSimulator(**options)
    bound = transpile(list(circuits), backend)
    result = backend.run(bound, shots=shots).result()
    return [
        {str(k): int(v) for k, v in result.get_counts(idx).items()}
        for idx in range(len(circuits))
    ]


def _run_statevector(
    circuits: Sequence[Any], shots: int, seed: int
) -> List[Dict[str, int]]:
    """Reference simulator for circuits Aer cannot assemble.

    Aer segfaults on the uniformly-controlled rotations and composed Grover
    operators the amplitude-estimation vertical builds. ``StatevectorSampler``
    evaluates them natively, at the cost of materialising the full state -
    fine at the qubit counts these circuits use, unusable for a wide QAOA.
    """
    import numpy as np  # noqa: PLC0415
    from qiskit.primitives import StatevectorSampler  # noqa: PLC0415

    # StatevectorSampler(seed=s) reuses one random stream for every circuit in
    # a batch, which makes the shot noise of different circuits perfectly
    # correlated. MLAE's likelihood assumes independent rounds, so one sampler
    # per circuit, each with its own spawned seed.
    children = np.random.SeedSequence(seed).spawn(len(circuits))
    out: List[Dict[str, int]] = []
    for circuit, child in zip(circuits, children):
        sampler = StatevectorSampler(seed=int(child.generate_state(1)[0]))
        outcome = sampler.run([circuit], shots=shots).result()[0]
        out.append({
            str(k): int(v)
            for k, v in getattr(outcome.data, _creg_name(circuit)).get_counts().items()
        })
    return out


_LOCAL_SIMULATORS = {"aer": _run_aer, "statevector": _run_statevector}


# ---------------------------------------------------------------------------
# Real-hardware paths
# ---------------------------------------------------------------------------


def _run_ibm(
    circuits: Sequence[Any],
    *,
    shots: int,
    job_label: str,
    min_qubits: int,
) -> CircuitRunResult:
    from qiskit.transpiler.preset_passmanagers import (  # noqa: PLC0415
        generate_preset_pass_manager,
    )
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2  # noqa: PLC0415

    token = resolve_vendor_token("ibm_quantum", "IBM_QUANTUM_TOKEN")
    if not token:
        raise HardwareUnavailableError(
            "No IBM Quantum API token available. Add one under IAM & Admin -> "
            "Quantum Credentials, or upgrade to Enterprise for the platform "
            "credit pool."
        )

    started = time.monotonic()
    service = QiskitRuntimeService(channel="ibm_quantum_platform", token=token)
    try:
        backend = service.least_busy(
            operational=True, simulator=False, min_num_qubits=min_qubits
        )
    except Exception as exc:  # noqa: BLE001 - provider-specific failure modes
        raise HardwareUnavailableError(
            f"no operational IBM Quantum device with at least {min_qubits} "
            f"qubits is currently available: {exc}"
        ) from exc

    backend_name = str(getattr(backend, "name", backend))
    logger.info(
        "[circuit-runner] ibm real backend=%s circuits=%d shots=%d label=%s",
        backend_name, len(circuits), shots, job_label,
    )

    pm = generate_preset_pass_manager(optimization_level=3, backend=backend)
    isa_circuits = [pm.run(c) for c in circuits]

    sampler = SamplerV2(mode=backend)
    # Dynamical decoupling idles the qubits that are waiting on the critical
    # path, which is where most of the decoherence in a QAOA-depth circuit
    # accumulates. Free to enable, materially better counts.
    try:
        sampler.options.dynamical_decoupling.enable = True
        sampler.options.dynamical_decoupling.sequence_type = "XY4"
        sampler.options.twirling.enable_gates = True
    except Exception:  # noqa: BLE001 - option surface varies by runtime version  # nosec B110
        pass

    job = sampler.run(isa_circuits, shots=shots)
    try:
        job_id = job.job_id()
    except Exception:  # noqa: BLE001  # nosec B110
        job_id = None
    result = job.result()

    counts = [
        _counts_from_databin(result[idx].data, _creg_name(circuits[idx]))
        for idx in range(len(circuits))
    ]
    vendor_runs = []
    if job_id:
        vendor_runs.append({
            "vendor": "ibm",
            "vendor_label": "IBM Quantum",
            "job_id": job_id,
            "stage": job_label,
            "backend": backend_name,
            "submitted_at": datetime.now(timezone.utc).isoformat(),
        })
    return CircuitRunResult(
        counts=counts,
        backend_name=backend_name,
        platform="ibm",
        execution_mode="real",
        shots=shots,
        elapsed_ms=int((time.monotonic() - started) * 1000),
        vendor_runs=vendor_runs,
        physical_qubits=max((c.num_qubits for c in isa_circuits), default=None),
    )


def _run_ionq(
    circuits: Sequence[Any],
    *,
    shots: int,
    job_label: str,
    min_qubits: int,
    hardware: bool = True,
) -> CircuitRunResult:
    from qiskit_ionq import IonQProvider  # noqa: PLC0415

    if hardware:
        token = resolve_vendor_token("ionq", "IONQ_API_KEY")
        if not token:
            raise HardwareUnavailableError(
                "No IonQ API key available. Add one under IAM & Admin -> Quantum "
                "Credentials, or upgrade to Enterprise for the platform credit pool."
            )
    else:
        token = resolve_tenant_token("ionq")
        if not token:
            raise HardwareUnavailableError(
                "No tenant IonQ API key is in scope for a cloud-simulator run."
            )

    started = time.monotonic()
    provider = IonQProvider(token)
    # Same backend vocabulary the finance BF-DCQO optimizer uses: the QPU for
    # hardware, IonQ's hosted simulator otherwise. Both mint a real cloud job
    # id, which is what makes the portal deep-link resolve.
    backend_name = ionq_qpu_backend() if hardware else "ionq_simulator"
    backend = provider.get_backend(backend_name)
    # qiskit-ionq 1.x is BackendV2: `configuration()` is gone, the width is on
    # `num_qubits`. Fall back to the V1 accessor so older pins keep working.
    n_qubits = getattr(backend, "num_qubits", None)
    if n_qubits is None:
        legacy_config = getattr(backend, "configuration", None)
        if callable(legacy_config):
            n_qubits = getattr(legacy_config(), "n_qubits", None)
    if n_qubits is not None and int(n_qubits) < min_qubits:
        raise HardwareUnavailableError(
            f"IonQ backend '{backend_name}' is too small: needs {min_qubits} "
            f"qubits, device has {int(n_qubits)}"
        )

    logger.info(
        "[circuit-runner] ionq %s backend=%s circuits=%d shots=%d label=%s",
        "real" if hardware else "cloud-sim",
        backend_name, len(circuits), shots, job_label,
    )

    # QAE state-prep emits ucry/multiplexer gates that IonQ's native gateset
    # rejects outright. Level 1 because IonQ warns the qiskit default of 2
    # re-synthesises too aggressively for their hardware.
    from qiskit import transpile  # noqa: PLC0415

    bound = transpile(list(circuits), backend, optimization_level=1)

    # IonQ's provider takes one circuit per job; it has no batched pub API.
    counts: List[Dict[str, int]] = []
    vendor_runs: List[Dict[str, Any]] = []
    for idx, circuit in enumerate(bound):
        job = backend.run(circuit, shots=shots)
        counts.append({str(k): int(v) for k, v in job.get_counts().items()})
        job_id = getattr(job, "job_id", lambda: None)()
        if job_id:
            vendor_runs.append({
                "vendor": "ionq",
                "vendor_label": "IonQ",
                "job_id": str(job_id),
                "stage": f"{job_label} #{idx + 1}",
                "backend": backend_name,
                "submitted_at": datetime.now(timezone.utc).isoformat(),
            })
    return CircuitRunResult(
        counts=counts,
        backend_name=backend_name,
        platform="ionq",
        execution_mode="real" if hardware else "simulator",
        shots=shots,
        elapsed_ms=int((time.monotonic() - started) * 1000),
        vendor_runs=vendor_runs,
    )


_REAL_RUNNERS = {"ibm": _run_ibm, "ionq": _run_ionq}


def run_measured_circuits(
    circuits: Sequence[Any],
    *,
    shots: int,
    platform: Optional[str] = None,
    execution_mode: str = "simulator",
    seed: int = 42,
    job_label: str = "sample",
    min_qubits: int = 1,
    simulator: str = "aer",
) -> CircuitRunResult:
    """Execute ``circuits`` and return per-circuit counts plus provenance.

    Every circuit must already contain its own measurements. Raises
    :class:`HardwareUnavailableError` when ``execution_mode='real'`` cannot
    be honoured; the caller decides whether that is fatal.
    """
    key = (platform or "").strip().lower()
    mode = (execution_mode or "simulator").strip().lower()

    if mode == "real":
        runner = _REAL_RUNNERS.get(key)
        if runner is None:
            raise HardwareUnavailableError(
                f"real-hardware execution is not wired for platform '{platform}'; "
                f"supported: {', '.join(sorted(_REAL_RUNNERS))}"
            )
        return runner(
            circuits, shots=shots, job_label=job_label, min_qubits=min_qubits
        )

    # IonQ's hosted simulator is a real cloud job that lands in the tenant's
    # IonQ portal, so simulator mode goes there when the tenant brought their
    # own key. Every other vendor's simulator is a local process with no portal
    # footprint. Mirrors _SIM_BYOK_VENDORS in the finance dispatcher.
    if key in SIM_BYOK_PLATFORMS and resolve_tenant_token(
        PLATFORM_VENDOR.get(key, key)
    ):
        try:
            return _run_ionq(
                circuits,
                shots=shots,
                job_label=job_label,
                min_qubits=min_qubits,
                hardware=False,
            )
        except Exception as exc:  # noqa: BLE001 - a sim run must not hard-fail
            logger.warning(
                "[circuit-runner] ionq cloud simulator unavailable (%s); "
                "falling back to the local simulator", exc,
            )

    local = _LOCAL_SIMULATORS.get(simulator)
    if local is None:
        raise ValueError(
            f"unknown local simulator '{simulator}'; "
            f"expected one of {sorted(_LOCAL_SIMULATORS)}"
        )
    started = time.monotonic()
    counts = local(circuits, shots, seed)
    return CircuitRunResult(
        counts=counts,
        backend_name="aer_simulator" if simulator == "aer" else "statevector_sampler",
        # A local run is a local run, whichever vendor was asked for. Echoing
        # the request here would file the job under that vendor in Job History
        # and offer a portal link to a job the vendor never saw.
        platform="cpu",
        execution_mode="simulator",
        shots=shots,
        elapsed_ms=int((time.monotonic() - started) * 1000),
    )


def vendor_for(platform: Optional[str]) -> Optional[str]:
    """BYOK vendor name for a gate platform, or ``None`` if it has no gate."""
    return PLATFORM_VENDOR.get((platform or "").strip().lower())


def runs_on_vendor_cloud(
    platform: Optional[str], execution_mode: Optional[str]
) -> bool:
    """Whether this ``(platform, mode)`` pair will actually reach the vendor.

    Callers use this to decide whether to attribute a run to the vendor or
    relabel it as local. Real mode always reaches the vendor; simulator mode
    only does on vendors that host their own simulator, and only when a
    tenant credential is in scope to submit it under.
    """
    key = (platform or "").strip().lower()
    mode = (execution_mode or "simulator").strip().lower()
    if mode == "real":
        return key in _REAL_RUNNERS
    if key not in SIM_BYOK_PLATFORMS:
        return False
    return bool(resolve_tenant_token(PLATFORM_VENDOR.get(key, key)))
