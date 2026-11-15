# Every Quantum Run Needs a Referee — reproduction package

Code, raw results and paper source for

> A. Nachtigall, *Every Quantum Run Needs a Referee: Design and Empirical
> Lessons from a Vendor-Neutral Hybrid Quantum–Classical Platform*, 2026.

The experiments call the platform's own solver modules (`src/`), not a
separate reimplementation. Only those modules are included; the platform's
API, tenancy, billing and credential code is not part of this package.

## Contents

| Path | What |
|---|---|
| `src/core/optimizers/` | Solver modules used by the experiments (Monte Carlo/MLAE, SqDRIFT/SQD, facility location incl. exact MILP, fleet QUBO, QAOA, Rydberg MIS) |
| `paper/experiments/eN_*.py` | One script per experiment (see the paper's appendix) |
| `paper/experiments/results/*.json` | Raw results as published; each records commit, Python version and, for hardware runs, every IBM job ID |
| `paper/main.tex`, `references.bib`, `figures/` | Paper source; `main.pdf` is the built paper |
| `paper/PEER_REVIEW_2026-11-15.md` | Internal referee report and how it was addressed |

## Run

Docker (recommended, pinned environment):

```bash
docker build -t qfpo-repro .
docker run --rm -v "$PWD/paper/experiments/results:/repro/paper/experiments/results" qfpo-repro e2_facility
docker run --rm qfpo-repro all          # every simulator experiment, several hours
```

Local Python 3.12:

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
./run.sh e1_montecarlo                   # or: ./run.sh all
```

| Script | Runtime (16-core workstation) | Needs |
|---|---|---|
| `e1_montecarlo` | ~40 min | — |
| `e2_facility` | ~10 min | — |
| `e3_rydberg` | ~30 min | — |
| `e4_chemistry` | ~2 h | — |
| `e5_fleet` | ~20 min | — |
| `e6_qaoa_control` | ~15 min | — |
| `e7_chemistry_controls` | ~10 min | `e4` results |
| `e8_ibm_hardware` | ~45 min, ~45 s QPU | `IBM_QUANTUM_TOKEN`; `--dry-run` runs on the simulator |
| `e8b_ibm_transpiled_cost` | ~2 min | `IBM_QUANTUM_TOKEN` (backend properties only, no QPU time) |
| `e9_sqd_noise_control` | ~5 min | `e8` results |

`./run.sh all` skips the two IBM scripts unless `IBM_QUANTUM_TOKEN` is set.

## Reproducibility notes

- Seeds are fixed in every script. SQD results are not bit-reproducible under
  multithreading (see the paper, Section 8.2); set `OMP_NUM_THREADS=1` for
  bit-identical reruns at the cost of speed.
- Hardware results depend on device calibration on the day; the published
  JSON files keep the job IDs so the raw counts can be retrieved from IBM.
- Re-running a script overwrites its JSON in `paper/experiments/results/`.
  Compare against the published version with `git diff`.

## Licence and citation

Code: Apache-2.0 (`LICENSE`). Paper, figures and results: CC-BY-4.0
(`LICENSE-CC-BY-4.0`). Cite via `CITATION.cff` or the Zenodo DOI.
