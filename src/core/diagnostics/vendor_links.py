"""F24 — Vendor portal deep-link URL templates and artifact builders.

Each quantum vendor (IBM Runtime, IonQ Cloud, D-Wave Leap, AWS Braket,
Google Quantum Engine) exposes a cloud console where users can inspect
the original job (circuit compilation, calibration data, queue position,
billing, raw shot counts, noise model). This module centralises the
per-vendor URL format and the artifact shape used by the Job History
detail view to render "View on vendor portal" buttons.

Design captured in
``reports/architecture/enhancements/F24_vendor_portal_deep_links.md``.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional


VENDOR_LABELS: Dict[str, str] = {
    "ibm": "IBM Quantum",
    "ionq": "IonQ Cloud",
    "dwave": "D-Wave Leap",
    "braket": "AWS Braket",
    "google": "Google Quantum Engine",
}


def build_portal_url(
    vendor: str,
    job_id: str,
    *,
    region: Optional[str] = None,
    project: Optional[str] = None,
) -> str:
    """Build the vendor-portal URL for a given ``(vendor, job_id)``.

    Args:
        vendor: one of ``ibm`` | ``ionq`` | ``dwave`` | ``braket`` | ``google``.
        job_id: vendor-side identifier (UUID, ARN suffix, problem id).
        region: AWS region for braket (defaults to ``us-east-1``).
        project: GCP project id for google (required for google).

    Returns:
        Absolute HTTPS URL to the vendor's console for this job.

    Raises:
        ValueError: vendor not recognised, missing job_id, or required
            region/project missing.
    """
    if not job_id:
        raise ValueError("job_id is required")
    v = (vendor or "").lower().strip()
    if v == "ibm":
        return f"https://quantum.ibm.com/jobs/{job_id}"
    if v == "ionq":
        return f"https://cloud.ionq.com/jobs/{job_id}"
    if v == "dwave":
        return f"https://cloud.dwavesys.com/leap/problems/{job_id}"
    if v == "braket":
        r = region or "us-east-1"
        return (
            f"https://{r}.console.aws.amazon.com/braket/home"
            f"?region={r}#/tasks/{job_id}"
        )
    if v == "google":
        if not project:
            raise ValueError("google portal URL requires `project`")
        return (
            f"https://console.cloud.google.com/quantum/jobs/{job_id}"
            f"?project={project}"
        )
    raise ValueError(f"unknown vendor {vendor!r}")


def _safe_id_fragment(job_id: str) -> str:
    """Reduce a job id to an artifact-name-safe fragment (≤ 80 chars)."""
    cleaned = "".join(
        c for c in str(job_id).lower() if c.isalnum() or c in "._-"
    )
    return cleaned[:80] or "job"


def vendor_link_artifact(
    vendor: str,
    job_id: str,
    *,
    stage: str = "post",
    backend: Optional[str] = None,
    submitted_at: Optional[str] = None,
    region: Optional[str] = None,
    project: Optional[str] = None,
    title_suffix: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a single artifact dict suitable for ``attach_diagnostics_step``.

    The returned dict carries ``format="link"`` — the frontend
    ``StepArtifact`` component routes this to its ``LinkBlock`` renderer
    instead of trying to render a Chart.js spec.
    """
    url = build_portal_url(vendor, job_id, region=region, project=project)
    label = VENDOR_LABELS.get((vendor or "").lower(), vendor)
    title = f"{label} — {title_suffix}" if title_suffix else f"{label} job"
    name_prefix = (vendor or "vendor").lower()
    name_stage = (
        "".join(c for c in str(title_suffix).lower() if c.isalnum() or c in "._-")
        if title_suffix
        else ""
    )
    safe_id = _safe_id_fragment(job_id)
    name = (
        f"{name_prefix}_{name_stage}_{safe_id}"
        if name_stage
        else f"{name_prefix}_{safe_id}"
    )
    return {
        "name": name,
        "kind": "vendor_link",
        "title": title,
        "stage": stage,
        "format": "link",
        "spec": {
            "url": url,
            "label": f"View on {label} portal",
            "vendor": (vendor or "").lower(),
            "vendor_label": label,
            "job_id": job_id,
            "backend": backend,
            "submitted_at": submitted_at,
            "stage_tag": title_suffix,
        },
    }


def vendor_link_artifacts_from_runs(
    vendor_runs: Optional[List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """Bulk-convert a list of ``vendor_runs`` entries into artifact dicts.

    Entries with missing ``vendor`` or ``job_id`` are skipped. Per-entry
    errors are swallowed so a single bad row never poisons the others.
    """
    out: List[Dict[str, Any]] = []
    for run in vendor_runs or []:
        if not isinstance(run, dict):
            continue
        vendor = run.get("vendor")
        job_id = run.get("job_id")
        if not vendor or not job_id:
            continue
        try:
            out.append(vendor_link_artifact(
                vendor,
                job_id,
                stage="post",
                backend=run.get("backend"),
                submitted_at=run.get("submitted_at"),
                region=run.get("region"),
                project=run.get("project"),
                title_suffix=run.get("stage"),
            ))
        except Exception:  # noqa: BLE001 — best-effort, never break the run  # nosec B112 - auto-applied by hardening agent; reviewer must verify before merging
            continue
    return out


__all__ = [
    "VENDOR_LABELS",
    "build_portal_url",
    "vendor_link_artifact",
    "vendor_link_artifacts_from_runs",
]
