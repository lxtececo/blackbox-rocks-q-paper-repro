"""E18 - The device's published calibration at the time of each round-2 job, next to what the job measured.

No QPU time: reads backend properties as published at each job's submission
time (IBM keeps the history) and the readout calibrations stored by E14 and
E15. Written after the runs, to make the comparison "published against
measured" traceable per qubit and per direction of the readout error; E14
stored only the published mean assignment error.

Usage: e18_published_calibration.py        (needs the IBM token, as E14)
"""

from __future__ import annotations

import datetime as dt
import json
import sys

from common import RESULTS, save

BACKEND = "ibm_marrakesh"
FIELDS = ("readout_error", "prob_meas0_prep1", "prob_meas1_prep0")


def published(props, qubit: int) -> dict:
    out = {}
    for name in FIELDS:
        value, when = props.qubit_property(qubit, name)
        out[name] = float(value)
        out[f"{name}_measured_utc"] = when.astimezone(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return out


def compare(props, readout: dict) -> list:
    rows = []
    for qubit, p10, p01 in zip(readout["physical_qubits"], readout["p_read1_given0"], readout["p_read0_given1"]):
        pub = published(props, qubit)
        rows.append({"qubit": qubit, "measured_p_read0_given1": p01, "measured_p_read1_given0": p10,
                     "measured_mean_assignment_error": (p01 + p10) / 2, "published": pub,
                     "p_read0_given1_measured_over_published": p01 / pub["prob_meas0_prep1"] if pub["prob_meas0_prep1"] else None})
    return rows


def main() -> None:
    argv, sys.argv = sys.argv, [sys.argv[0], "--dry-run"]  # import E14 without touching its hardware path
    import e14_ibm_hardware_controls as e14
    sys.argv = argv
    backend = e14.ibm_service().backend(BACKEND)

    def props_at(stamp: str):
        when = dt.datetime.strptime(stamp, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.timezone.utc)
        props = backend.properties(datetime=when)
        return props, {"job_submitted_utc": stamp,
                       "published_calibration_last_update_utc": props.last_update_date.astimezone(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}

    out = {"backend": BACKEND, "jobs": []}
    e14b = json.loads((RESULTS / "e14_ibm_hardware_controls_mlae.json").read_text())["mlae"]
    props, meta = props_at(e14b["run"]["submitted_utc"])
    out["jobs"].append({"experiment": "E14-B", "job_id": e14b["run"]["job_id"], **meta,
                        "registers": [{"n_price_qubits": r["n_price_qubits"], "placement": "published calibration",
                                       "qubits": compare(props, r["readout"])} for r in e14b["registers"]]})
    e15 = json.loads((RESULTS / "e15_mlae_screened_qubits.json").read_text())
    props, meta = props_at(e15["run"]["submitted_utc"])
    out["jobs"].append({"experiment": "E15", "job_id": e15["run"]["job_id"], **meta,
                        "registers": [{"n_price_qubits": s["n_price_qubits"], "placement": f"screened set {s['set']}",
                                       "qubits": compare(props, s["readout"])} for s in e15["sets"]]})
    ages = []
    for job in out["jobs"]:
        submitted = dt.datetime.strptime(job["job_submitted_utc"], "%Y-%m-%dT%H:%M:%SZ")
        stamps = [dt.datetime.strptime(q["published"]["prob_meas0_prep1_measured_utc"], "%Y-%m-%dT%H:%M:%SZ")
                  for r in job["registers"] for q in r["qubits"]]
        job["published_readout_age_days"] = [min((submitted - s).total_seconds() for s in stamps) / 86400,
                                             max((submitted - s).total_seconds() for s in stamps) / 86400]
        ages += job["published_readout_age_days"]
    out["published_readout_age_days_range"] = [min(ages), max(ages)]
    save("e18_published_calibration", out)
    for job in out["jobs"]:
        print(job["experiment"], job["job_id"], "published readout age (days)", [round(a, 2) for a in job["published_readout_age_days"]])
        for reg in job["registers"]:
            print(" ", reg["n_price_qubits"] + 1, "qubits,", reg["placement"])
            for q in reg["qubits"]:
                print(f"    q{q['qubit']:3d}  P(0|1) measured {q['measured_p_read0_given1'] * 100:5.2f}% published {q['published']['prob_meas0_prep1'] * 100:5.2f}%"
                      f"   P(1|0) measured {q['measured_p_read1_given0'] * 100:5.2f}% published {q['published']['prob_meas1_prep0'] * 100:5.2f}%")


if __name__ == "__main__":
    main()
