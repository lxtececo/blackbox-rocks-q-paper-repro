# Every Quantum Run Needs a Referee — reproduction package

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24247106.svg)](https://doi.org/10.5281/zenodo.24247106)

Code, raw results and paper source for

> A. Nachtigall, *Every Quantum Run Needs a Referee: Design and Empirical
> Lessons from a Vendor-Neutral Hybrid Quantum–Classical Platform*, 2026.

**Version 2.1.0** (26 November 2026): experiments E1 to E32 and eight
hardware rounds on three IBM Heron devices (`ibm_marrakesh`, `ibm_kingston`
and `ibm_fez`). New since version 2.0.1: hardware rounds 4 to 8, the seventh
workload (mission resource allocation) with its solver modules and two
experiments, and a paper that reports the platform as it stands.
[`CHANGES.md`](CHANGES.md) says what changed from version to version and
keeps the record of corrections to earlier versions. Version 1.0.0 held E1
to E9 of the first version of the paper and is kept on Zenodo for the
record; later checks invalidated part of its results.

The experiments call the platform's own solver modules (`src/`), not a
separate reimplementation. Only those modules are included; the platform's
API, tenancy, billing and credential code is not part of this package.

## Contents

| Path | What |
|---|---|
| `src/core/` | The solver modules the experiments import, and nothing else of the platform: amplitude estimation, SqDRIFT and SQD, facility location with its exact MILP, the routing QUBO and its samplers, QAOA, Rydberg MIS, the portfolio methods (QAOA, CVaR-VQE, PCE, BF-DCQO) with the refinement step and the exact referee, and mission resource allocation (model, exact MILP, greedy baseline, QUBO, sampler with repair and fill) |
| `paper/experiments/eN_*.py` | One script per experiment (table under [Experiments](#experiments)) |
| `paper/experiments/results/*.json` | Raw results as published; each records commit, Python version and, for hardware runs, every IBM job ID |
| `paper/experiments/results/raw/` | Raw counts of the hardware rounds from round 2 on, the circuits of rounds 3 to 8 as submitted, and the calibration snapshots stored for `e30` (it compares three of them) |
| `paper/experiments/data/` | The simulated market data of the platform that E23 to E25 use, as fetched once |
| `paper/experiments/legacy/` | Portfolio modules of an earlier version of the platform, which E19 measured |
| `paper/main.tex`, `references.bib`, `figures/` | Paper source; `main.pdf` is the built paper |
| `CHANGES.md` | Release notes: what changed from version to version, and the record of corrections |
| `paper/PEER_REVIEW_2026-11-15.md` | The internal referee report on the first version of the paper and how it was addressed |

## Experiments

What each script measures. The wall times are
those recorded in the stored results, on a 16-core workstation (the
portfolio experiments with 12 worker processes; set `QFPO_WORKERS`). A dash
means that the result records none. The paper reports all of them except
`e6`, `e7` and `e19`, which measured earlier versions of the code.

| Script | What it measures | Wall time | Needs |
|---|---|---|---|
| `e1_montecarlo` | Option pricing: amplitude estimation (MLAE) against classical Monte Carlo, 50 replicates, Cramér–Rao bound and coverage; bias against qubits; transpiled resources | 104 s | |
| `e2_facility` | Facility location: exact MILP (capacitated and uncapacitated), heuristic referee, two simulated annealers, 10 seeds, budget arm | 216 s | |
| `e3_rydberg` | Rydberg MIS: static- against untuned dynamic-radius mapping on 24 instances, sweep duration, embeddability of 2,000 instances | 80 s | |
| `e4_chemistry` | SqDRIFT + SQD on H₂, LiH, H₂O, N₂ (equilibrium and 1.8 Å), five seeds | 890 s | |
| `e5_fleet` | Routing: two presets × 10 instances × 5 seeds × 2 samplers | 161 s | |
| `e6_qaoa_control` | QAOA against a random control (first version of the paper) | 116 s | superseded by `e10` |
| `e7_chemistry_controls` | SQD against simple classical selections (first version of the paper) | 24 s | results of `e4`; superseded by `e12` |
| `e8_ibm_hardware` | Hardware round 1: MLAE, SqDRIFT on N₂ and QAOA through the platform path | hardware round 1 | `IBM_QUANTUM_TOKEN`; `--dry-run` on the simulator |
| `e8b_ibm_transpiled_cost` | MLAE circuits transpiled for `ibm_marrakesh` | – | `IBM_QUANTUM_TOKEN` (backend properties, no QPU time) |
| `e9_sqd_noise_control` | Uniform random bitstrings through the SQD call (round 1; extended by `e14`) | 194 s | results of `e8` |
| `e10_qaoa_exact` | QAOA output distribution evaluated exactly: Hamiltonian in problem units and normalised, the shipped sampler, exact-objective optimum; depth-1 landscape | 526 s | |
| `e11_rydberg_tuned_control` | Static-radius mapping against a dynamic-radius control tuned on held-out instances | 41 s | |
| `e12_sqd_selected_ci` | SqDRIFT subspaces against random strings, an FCI-weight oracle and PySCF selected CI, one solver | 917 s | |
| `e13_hardware_statistics` | Exact tests on the stored round-1 counts | – | stored counts of round 1 |
| `e14_ibm_hardware_controls` | Hardware round 2 (pre-registered): QAOA with matched nulls, amplitude-estimation depth probe, SQD on raw counts against shuffled and uniform input | hardware round 2 | `IBM_QUANTUM_TOKEN`; `--dry-run` |
| `e15_mlae_screened_qubits` | Follow-up to `e14`: depth probe on qubits screened by an in-session readout calibration | hardware | `IBM_QUANTUM_TOKEN`; `--dry-run` |
| `e16_hardware_summary` | Effective error per two-qubit gate and information-per-query bound from `e14` and `e15` | – | results of `e14`, `e15` |
| `e17_derived_statistics` | Derived numbers quoted in the text, computed from the stored results | – | stored results |
| `e18_published_calibration` | The device's published calibration at each round-2 job, per qubit, next to the readout errors the job measured | – | `IBM_QUANTUM_TOKEN` (calibration history, no QPU time) |
| `e19_portfolio` | The four portfolio methods of an earlier version of the platform against the exact optimum of their own objectives and a uniform null through the same code; not reported in the paper | 103 s | not runnable here, see below |
| `e20_portfolio_corrected` | The four portfolio methods on 30 instances: seven arms against the exact optimum and their nulls; run twice | 365 s | |
| `e21_qaoa_fresh` | QAOA on 30 instances that no earlier run had touched, through the product's own seed argument | 31 s | |
| `e22_hardware_portfolio_qaoa` | Hardware round 3 (design committed before the first job): portfolio QAOA with transferred angles at 8, 12 and 16 qubits against two null bindings | hardware round 3 | `IBM_QUANTUM_TOKEN`; `--dry-run` |
| `e23_qaoa_product_data` | `e21` on the product's simulated market data | 32 s | the stored data file |
| `e24_displayed_allocation`, `e24_posthoc` | The product's executor on the instances of `e23`: what is displayed for the QAOA selection, its null, every asset, the exact optimum and uniform selections | 78 s | not runnable here, see below |
| `e25_qaoa_initial_angles` | The start of the portfolio QAOA's training against the opposite assignment of its two ramps, on the instances and seeds of `e21` and `e23` | 74 s | results of `e21` and `e23` |
| `e26_hardware_qaoa_replication` | Hardware round 4 (plan and device rule committed before the job): the three bindings of `e14`'s QAOA experiment on a second device, with a readout calibration in the job | hardware round 4 | `IBM_QUANTUM_TOKEN`; `--dry-run` |
| `e27_hardware_amplitude_estimation` | Hardware round 5 (criterion committed before the first job): amplitude estimation at 2,450 oracle queries on three screened 3-qubit sets against ideal unamplified sampling | hardware round 5 | `IBM_QUANTUM_TOKEN`; `--dry-run` |
| `e28_hardware_qaoa_device_training` | Hardware round 6 (plan committed before the first job, amended once before any training job): the product's hardware path for portfolio QAOA, training of ten 8-asset instances on the device, and one evaluation job with the start, the device-trained and the transferred angles and two nulls | hardware round 6 | `IBM_QUANTUM_TOKEN`; `--dry-run` |
| `e29_hardware_sqd_active_space` | Hardware round 7 (plan committed before the first job): the product's SqDRIFT pipeline for N₂ in a 24-qubit active space on the device, against strings drawn from the samples' marginals, uniform bits and noiseless samples of the same circuits | hardware round 7 | `IBM_QUANTUM_TOKEN`; `--dry-run` |
| `e30_hardware_calibration_staleness` | Hardware round 8 (plan committed before the first snapshot, amended once before any job): one QAOA circuit placed with the calibration data the device reports at submission and with data hours and days old, in one job with a null | hardware round 8 | `IBM_QUANTUM_TOKEN`; `--dry-run`; `--plan` runs offline |
| `e31_mission_allocation` | Mission resource allocation (plan committed before the evaluation run): 64 annealed reads against 64 and 1,024 uniform bitstrings through the product's repair and fill, the greedy baseline and the exact optimum on 60 generated instances and the four built-in scenarios; QAOA at depth 1 and 2 against uniform bitstrings, evaluated exactly on 40 small instances | 853 s | |
| `e32_mission_equal_time` | Mission resource allocation at equal time (plan committed before the evaluation run): the annealed arm against uniform bitstrings given its time, half of it and twice it, on 60 further instances; the time of the exact solver; 15 larger instances, described only | 4,893 s | its times depend on the machine, see below |
| `hardware_jobs` | Used by `e26` to `e30`: the record of submitted jobs, write-once archives of raw counts, the estimate of QPU seconds | – | |
| `portfolio_tables` | Generates the rows of five tables of the paper from the stored results | – | stored results; `--check` compares its rows with `paper/main.tex` |
| `mission_tables` | The same for the paper's three mission tables | – | stored results; `--check` |

## What cannot be re-run from this package

- **`e19_portfolio`** measured the four portfolio methods of an earlier
  version of the platform. `src/` holds the current modules, and the
  script does not run against them. The seven modules it measured are under
  `paper/experiments/legacy/portfolio_as_shipped_f8d99e6b/`, byte for byte,
  for reading; they did not change between the commit the result records and
  that one. The stored result is the record of the run; the paper no longer
  reports it, and `CHANGES.md` says what it found. `e20` to `e25`
  measure the current modules and run here.
- **`e24_displayed_allocation` and `e24_posthoc`** call the platform's result
  assembly (the executor that turns a selection into the displayed
  portfolio). That code is not part of this package. The two scripts, their
  inputs and their raw results are included; the scripts do not run here.
- **Hardware rounds** need an IBM Quantum account and QPU time, and a device
  is not the same on another day. The raw counts and every job identifier are
  in the result files; `--dry-run` runs the same analysis on a simulator.
- **`e32_mission_equal_time`** runs here, but it compares methods at equal
  time, and its times are those of the workstation it ran on. A re-run
  measures other times and therefore gives the random arm other numbers of
  bitstrings; its stored result is the record of the run the paper reports.

One substitution was made in two copied files: the address of the development
gateway from which the simulated market data were fetched is replaced by
`https://dev-gateway.invalid` in
`paper/experiments/results/e23_qaoa_product_data.json` and
`paper/experiments/data/e23_simulated_market.json`. No number is affected.

## Run

Docker (recommended, pinned environment):

```bash
docker build -t qfpo-repro .
docker run --rm -v "$PWD/paper/experiments/results:/repro/paper/experiments/results" qfpo-repro e21_qaoa_fresh
docker run --rm qfpo-repro all          # every simulator experiment
```

Local Python 3.12:

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
./run.sh e1_montecarlo                   # or: ./run.sh all
```

`./run.sh all` runs the simulator experiments. `./run.sh analyses` runs the
scripts that only read stored results. Hardware scripts are never part of
either: name them explicitly, with `IBM_QUANTUM_TOKEN` set.

## Reproducibility notes

- Seeds are fixed in every script, and every script except `e3_rydberg` pins
  the numerical libraries to one thread per process (paper, Section 7).
- `e20` to `e23`, `e25`, `e31`, `e32` and the hardware scripts from `e26` on
  stop when the code they load differs from the
  committed state of a git checkout. In an archive download, which is no git
  checkout, they say so and run; the result then records the commit as
  unknown.
- `e25_qaoa_initial_angles` reports how many of its runs return the stored
  results of `e21` and `e23` exactly. In the platform repository that was 300
  of 300.
- Hardware results depend on device calibration on the day; the published
  JSON files keep the job IDs so the raw counts can be retrieved from IBM.
- Re-running a script overwrites its JSON in `paper/experiments/results/`.
  Compare against the published version with `git diff`.

## Licence and citation

Code: Apache-2.0 (`LICENSE`). Paper, figures and results: CC-BY-4.0
(`LICENSE-CC-BY-4.0`). Cite via `CITATION.cff` or the Zenodo DOI:
[10.5281/zenodo.24247106](https://doi.org/10.5281/zenodo.24247106) (all
versions; it resolves to the latest, and its page lists the DOI of each
version). Version 2.0.1,
[10.5281/zenodo.24368162](https://doi.org/10.5281/zenodo.24368162), holds
experiments E1 to E25, three hardware rounds and the paper as it stood on
19 November 2026. Version 2.0.0,
[10.5281/zenodo.24347819](https://doi.org/10.5281/zenodo.24347819), holds the
same code, results and figures as 2.0.1; its copy of the paper has an
earlier wording of two passages. Version 1.0.0,
[10.5281/zenodo.24247107](https://doi.org/10.5281/zenodo.24247107), holds the
results of the first version of the paper, including those that
[`CHANGES.md`](CHANGES.md) lists as corrected; it is kept for the record.
