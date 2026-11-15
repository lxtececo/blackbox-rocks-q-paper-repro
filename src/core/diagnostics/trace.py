"""7-QAP.25 Slice 1 — Shared optimization-trace accumulator.

Method-agnostic ring-buffered recorder that every iterative optimiser
(SciPy, Qiskit VQE/QAOA, PennyLane, IonQ BF-DCQO, …) can push iteration
records into without each method having to invent its own bookkeeping.
The resulting :class:`OptimizationTrace` is JSON-serialisable so the
chart builders in :mod:`core.diagnostics.artifacts` (Slice 2) can read
the same shape regardless of which method produced it.

Design contract
---------------
* **No-op when disabled.** :class:`OptimizationTrace.disabled()` returns
  a sentinel that silently accepts ``add`` / ``scipy_callback`` / etc.
  and serialises to ``{"enabled": false, "iters": []}``. Callers can
  therefore unconditionally pass a trace into any optimiser without
  paying the cost when diagnostics are off.
* **Bounded.** ``max_iters`` (env ``DIAGNOSTICS_TRACE_MAX_ITERS``,
  default 500) caps memory; the ring buffer keeps the most recent
  iterations. The total iteration counter still reflects the true
  number of pushes so the chart axis is honest.
* **Side-effect free.** No DB / file I/O; the accumulator is purely
  in-memory. Persistence is the executor's job via
  ``ExecutionStep.add_artifact``.
* **Opt-in.** Disabled by default unless ``DIAGNOSTICS_TRACE_ENABLED``
  is truthy *or* the caller explicitly constructs an enabled trace.
* **Adapter-friendly.** Provides three integration shims:
    - :meth:`scipy_callback` — drop-in ``callback=`` for
      ``scipy.optimize.minimize``.
    - :meth:`qiskit_callback` — matches Qiskit VQE/QAOA's
      ``(eval_count, params, mean, std)`` signature.
    - :meth:`record` — generic per-iter push used by hand-rolled loops.

The schema is intentionally narrow so charts can be added without a
recorder change:

    {
      "enabled": true,
      "method": "markowitz" | "vqe" | "qaoa" | "bfdcqo" | "<custom>",
      "total_iters": 273,
      "captured_iters": 273,            # may be < total when truncated
      "wallclock_s": 4.812,
      "iters": [
        {"i": 0,   "cost": 0.0421, "grad_norm": 0.31, "params": [...],
         "constraint_violations": {"budget": 0.0}, "wall_s": 0.011},
        ...
      ]
    }

Slice 2 (chart builders) reads ``iters[*].cost`` for P1, ``grad_norm``
for P2, ``params`` for P3, and the final ``cost`` against an external
reference for P4. Missing keys are tolerated — every chart skips when
its source field is absent.
"""

from __future__ import annotations

import os
import time
from collections import deque
from typing import Any, Iterable, Optional, Sequence


__all__ = [
    "OptimizationTrace",
    "DEFAULT_MAX_ITERS",
    "is_globally_enabled",
]


DEFAULT_MAX_ITERS = 500


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        v = int(raw)
        return v if v > 0 else default
    except ValueError:
        return default


def _env_truthy(name: str) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def is_globally_enabled() -> bool:
    """Return True iff ``DIAGNOSTICS_TRACE_ENABLED`` is truthy."""
    return _env_truthy("DIAGNOSTICS_TRACE_ENABLED")


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    # Reject NaN/inf at the boundary so the JSON serializer never
    # emits non-conforming numbers (chart libs reject them silently).
    if f != f or f in (float("inf"), float("-inf")):
        return None
    return f


def _coerce_params(params: Any, max_len: int = 64) -> Optional[list[float]]:
    """Coerce an ansatz parameter vector to a bounded list of floats.

    ``max_len`` caps the length so a 10 000-parameter ansatz doesn't
    explode the per-iter record. Chart P3 already clamps to ≤ 12 lines
    so the extra headroom is cheap insurance for downstream tooling.
    """
    if params is None:
        return None
    try:
        seq: Iterable[Any] = list(params)
    except TypeError:
        return None
    out: list[float] = []
    for item in seq:
        f = _to_float(item)
        if f is None:
            # Don't drop the whole vector for one bad entry — emit 0.0
            # so the index alignment of P3 (per-param lines) is stable.
            f = 0.0
        out.append(f)
        if len(out) >= max_len:
            break
    return out or None


class OptimizationTrace:
    """Bounded, JSON-serialisable iteration recorder."""

    __slots__ = (
        "_enabled",
        "method",
        "max_iters",
        "_iters",
        "_total",
        "_t0",
        "_last_t",
    )

    def __init__(
        self,
        method: str = "unknown",
        max_iters: Optional[int] = None,
        enabled: bool = True,
    ) -> None:
        self._enabled = bool(enabled)
        self.method = method or "unknown"
        cap = max_iters if max_iters is not None else _env_int(
            "DIAGNOSTICS_TRACE_MAX_ITERS", DEFAULT_MAX_ITERS,
        )
        self.max_iters = max(1, int(cap))
        self._iters: deque[dict[str, Any]] = deque(maxlen=self.max_iters)
        self._total = 0
        self._t0 = time.perf_counter()
        self._last_t = self._t0

    # ─── Factory helpers ────────────────────────────────────────────

    @classmethod
    def disabled(cls, method: str = "unknown") -> "OptimizationTrace":
        """Return a no-op trace. Cheap to pass into any optimiser."""
        return cls(method=method, enabled=False)

    @classmethod
    def from_env(
        cls,
        method: str = "unknown",
        force: Optional[bool] = None,
    ) -> "OptimizationTrace":
        """Construct a trace gated on ``DIAGNOSTICS_TRACE_ENABLED``.

        ``force=True`` forces enabled regardless of the env var (useful
        for tests and explicit opt-in via wizard config). ``force=False``
        forces disabled. ``None`` defers to the env var.
        """
        if force is True:
            enabled = True
        elif force is False:
            enabled = False
        else:
            enabled = is_globally_enabled()
        return cls(method=method, enabled=enabled)

    # ─── Status ─────────────────────────────────────────────────────

    @property
    def enabled(self) -> bool:
        return self._enabled

    def __len__(self) -> int:
        return len(self._iters)

    def __bool__(self) -> bool:
        return self._enabled

    # ─── Recording ──────────────────────────────────────────────────

    def record(
        self,
        cost: Any,
        *,
        grad_norm: Any = None,
        params: Any = None,
        constraint_violations: Optional[dict[str, Any]] = None,
        extras: Optional[dict[str, Any]] = None,
    ) -> None:
        """Push one iteration record.

        Silently no-ops when the trace is disabled. ``cost`` is the
        only field assumed present; everything else is optional.
        """
        if not self._enabled:
            return
        now = time.perf_counter()
        rec: dict[str, Any] = {
            "i": self._total,
            "cost": _to_float(cost),
            "wall_s": round(now - self._last_t, 6),
        }
        gn = _to_float(grad_norm)
        if gn is not None:
            rec["grad_norm"] = gn
        p = _coerce_params(params)
        if p is not None:
            rec["params"] = p
        if constraint_violations:
            cv: dict[str, float] = {}
            for k, v in constraint_violations.items():
                f = _to_float(v)
                if f is not None:
                    cv[str(k)] = f
            if cv:
                rec["constraint_violations"] = cv
        if extras:
            for k, v in extras.items():
                if k in rec:
                    continue
                f = _to_float(v)
                if f is not None:
                    rec[str(k)] = f
        self._iters.append(rec)
        self._total += 1
        self._last_t = now

    # ─── Optimiser adapters ─────────────────────────────────────────

    def scipy_callback(self, xk: Sequence[float], *_args: Any, **_kw: Any) -> None:
        """Drop-in for ``scipy.optimize.minimize(..., callback=...)``.

        SciPy invokes the callback with the current parameter vector
        only (no objective value), so we record ``cost=None`` and
        capture ``params``. The wrapping executor should *also* push
        ``cost`` via :meth:`record` after each successful function
        evaluation if it wants P1 populated.
        """
        if not self._enabled:
            return
        self.record(cost=None, params=xk)

    def qiskit_callback(
        self,
        eval_count: int,
        params: Sequence[float],
        mean: float,
        std: float,
    ) -> None:
        """Matches Qiskit ``VQE`` / ``QAOA`` ``callback=`` signature.

        Records ``cost=mean`` and ``params``; ``std`` is preserved in
        ``extras`` for an optional confidence-band overlay on P1.
        """
        if not self._enabled:
            return
        self.record(
            cost=mean,
            params=params,
            extras={"eval_count": eval_count, "std": std},
        )

    # ─── Serialisation ──────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe snapshot of the trace."""
        if not self._enabled:
            return {
                "enabled": False,
                "method": self.method,
                "total_iters": 0,
                "captured_iters": 0,
                "wallclock_s": 0.0,
                "iters": [],
            }
        wall = round(time.perf_counter() - self._t0, 6)
        return {
            "enabled": True,
            "method": self.method,
            "total_iters": self._total,
            "captured_iters": len(self._iters),
            "wallclock_s": wall,
            "iters": list(self._iters),
        }
