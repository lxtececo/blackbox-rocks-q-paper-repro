# Release notes

What changed from version to version of this archive, and the record of
corrections. The paper reports the platform as it is and the experiments run
on that code. Where an earlier version of the paper or of this archive
reported a result that a later check invalidated, the correction is listed
here, under [Corrections](#corrections). Every version stays on Zenodo.

## Version 2.1.1 (26 November 2026)

Adds `paper/summary-deck.pdf`, a summary of the paper in plain words: an
overview, one or more slides per section, and an appendix for readers with
a background in quantum computing. The deck is built from the paper and
holds no result of its own. It is under CC-BY-4.0; the logo and wordmark on
its cover are excluded. The paper names the deck in its availability
paragraph and cites this version. Code, results and figures are those of
version 2.1.0.

## Version 2.1.0 (26 November 2026)

New experiments and code:

- Hardware rounds 4 to 8 on three IBM Heron devices (`e26` to `e30`): the
  scripts, which contain the plans, the records of every submitted job, the
  raw counts and, for `e30`, the calibration snapshots it compares.
  `hardware_jobs.py` holds what these scripts share.
- The seventh workload, mission resource allocation: its solver modules
  (`src/core/optimizers/mission`), two experiments planned before their runs
  (`e31`, `e32`) and `mission_tables.py`, which generates the paper's three
  mission tables.

The paper:

- It now reports the platform as it stands. The account of earlier defects
  and their corrections that versions 2.0.0 and 2.0.1 of the paper contained
  (its audit of the first version, the portfolio methods as they first
  shipped, and what the product had stated without a measurement) is no
  longer in the paper. It is kept below.
- `e19_portfolio` measured the portfolio methods of an earlier version of the
  platform. Its script, inputs and result stay in the archive; the paper no
  longer reports it.
- The sections of the results are renumbered: portfolio selection 8.5,
  vehicle routing 8.6, Rydberg arrays 8.7, mission resource allocation 8.8,
  hardware 8.9.
- The abstract is rewritten as a summary for readers outside the field.

Code that changed in files version 2.0.1 already contained:

- `src/core/optimizers/chemistry/sqdrift.py`: the SQD call states the width
  of a sample (`num_bits`). Before, the width was taken from the largest
  outcome sampled, so that a set of samples in which none sets the highest
  qubit was read with too few bits. Found in the simulator pilot of hardware
  round 7, before the plan of that round was committed. The stored results
  of `e4` and `e12` were produced before this change. Run again with it on
  26 November 2026, both scripts return every stored value unchanged (1,326
  and 641 values; only the provenance record and the run time differ).
- `paper/experiments/e14_ibm_hardware_controls.py`: restructured so that
  `e26` and `e30` reuse its circuit and its scoring of a placement; a gate
  that the device reports with error 1 now gives a placement the score 0.
  The stored result of `e14` is unchanged.
- `run.sh all` includes `e31` and `e32`; `paper/build.sh` stops when a page
  of the paper is overfull.

## Version 2.0.1 (19 November 2026)

Text of the paper only, in two passages of its methodology section. Code,
results and figures are those of version 2.0.0.

## Version 2.0.0 (18 November 2026)

Experiments E1 to E25 and three hardware rounds on `ibm_marrakesh`. It
supersedes version 1.0.0; the corrections are listed below.

## Version 1.0.0 (15 November 2026)

Experiments E1 to E9 of the first version of the paper.

## Corrections

### Results of version 1.0.0 that an audit invalidated

The audit recomputed each headline number of the first version of the paper
from the code and checked each statistical statement against the stored
data. `paper/PEER_REVIEW_2026-11-15.md` is the internal referee report on
that version and says how it was addressed. In
all four cases the encoding was correct, which is why the encoding tests had
passed.

| What was wrong | What version 1.0.0 reported | What the corrected experiment shows |
|---|---|---|
| The simulator path passed one seed to the sampler for a whole batch of circuits, so that all circuits of a batch drew their shots from one random stream | Amplitude estimation with an error 2.4 times below its Cramér–Rao bound, a convergence slope of −1.14 and a "70×" saving of queries | Each circuit has its own generator. The estimator sits on its bound, with a slope of −0.96 and 38 times fewer queries than classical sampling (`e1`) |
| The cost Hamiltonian of QAOA was not normalised (largest coefficient 9 × 10⁴) | QAOA "no better than random sampling", deeper circuits worse | The Hamiltonian is divided by its largest coefficient. Evaluated exactly, the ansatz reaches approximation ratios of 0.66 to 0.83; the platform's training reaches 0.17 to 0.31 (`e10`, which replaces `e6`) |
| The control of the Rydberg mapping had a fixed, untuned sweep ratio | The mapping "raises valid readouts from 42% to 97%" | Against a control tuned on held-out instances the mapping shows no gain in optimal readouts and seven points in valid ones (`e11`) |
| Selected CI was named as the proper classical referee and not run | SqDRIFT "beats classical selection" for stretched N₂ | At matched dimension selected CI is 6 to 41 times more accurate (`e12`, which replaces `e7`) |

Statistics of version 1.0.0 that were corrected: confidence intervals
computed with z = 1.96 from two to ten runs, which are 1.2 to 6.5 times too
narrow; routing runs on the same instance counted as independent; seeds
reused across the points of a convergence curve; an unseeded simulator; and
statements about hardware that the stored counts did not support.

The QAOA job of hardware round 1 ran the circuit without normalisation and
without a control. It is uninformative; round 2A replaces it.

An internal figure that predates version 1.0.0 gave the annealing pipeline
a distance gap of +1.24% on the 90-stop routing instance. It was obsolete:
the classical construction had since improved by 14% on the same instance,
and the quantum figure had not been regenerated. Versions 1.0.0 to 2.0.1 of
the paper mentioned it for that reason. `e5` is the regenerated comparison:
the annealing pipelines are 13% to 60% longer on average.

### Encoding errors found by the exhaustive test

Before the conversion from QUBO to Ising existed once and was tested on
every bitstring, independently written conversions in different vendor paths
contained: a sign flip of the linear field; pair coefficients at half
weight; QUBO coefficients put on Pauli terms without conversion; big-endian
strings decoded as little-endian; a dropped variance diagonal; an equality
where an inequality was intended; and oversized penalty weights. None
crashed. The paper lists them as the classes of error the test catches.

### The portfolio methods as they shipped until November 2026

`e19_portfolio` measured the four gate-model portfolio methods as they then
were, against the exact optimum of each method's own QUBO and against
uniform bits pushed through the same code. Version 2.0.0 of the paper
reported what it found; the corrected methods are measured in `e20` to
`e25`. The modules `e19` measured are under
`paper/experiments/legacy/portfolio_as_shipped_f8d99e6b/`.

- **Refinement step of the BF-DCQO path.** A private copy of the conversion
  from QUBO to Ising, outside the tested one, assigned the diagonal term over
  pair shares it had already accumulated, flipped the sign of the linear
  field and halved the pair energy. With it the path returned the first K
  assets of the input in 150 of 150 runs, and so did its null. It was in
  production from May to November 2026. The step now uses the tested
  conversion, and a test lists every module that builds an Ising Hamiltonian
  together with the test that covers it.
- **BF-DCQO.** The circuit contained no counterdiabatic term, and its QUBO
  was a minimum-variance selection: the risk-aversion parameter was never
  read, and expected returns did not enter. Re-implemented after its source.
- **CVaR-VQE.** The objective took its CVaR over per-round estimates of the
  mean energy, which in simulator mode is the mean itself; the budget of 20
  evaluations was smaller than the optimiser's first simplex at 12 and 16
  assets; the selection was read from a single shot; and with a cap on the
  number of assets the cardinality target was set to the number of assets
  passed in. Re-implemented after its source.
- **PCE.** The method measured in the computational basis only and
  evaluated X and Y strings as if they were Z. Re-implemented after its
  source.
- **QAOA.** The method returned the most frequent of its 200 final samples,
  an exact optimum in 26%, 20% and 6% of runs at 8, 12 and 16 assets, where
  the cheapest of the same samples is one in 98%, 84% and 76%. It now
  returns the cheapest.
- **Seeds and defaults.** The methods exposed no seed for their simulators.
  Four sets of default values existed (executor, gateway, user interface,
  assistant), so that a run started from the interface was not the run the
  experiment measured. The methods now take a seed, and a test compares the
  interface's simulator presets with the measured table.
- **Start of the QAOA training (`e25`).** A comment in the code assigned the
  rising ramp of the initial angles to γ and the falling one to β; the code
  assigns them the other way round. `e25` measured the order the comment
  described, with a decision rule committed before the run. It was better in
  no comparison and worse in four, so the product keeps its order and the
  comment was corrected.

### What the product had stated without a measurement

Found while the product was evaluated in November 2026, by following
statements on the screen back to the line that produced them, and corrected
with tests:

- a per-result "quantum advantage" that three executors computed from the
  Sharpe ratio of the returned portfolio;
- a template that answered a failed annealing run with an equal-weight
  portfolio and invented key figures, an annealing method that returned
  random weights on the simulator, and a worker that stored a run reporting
  failure as completed;
- a method catalogue that listed a "true quantum advantage potential", and
  two recommenders that ranked the paradigms by fixed scores;
- a selector that preferred QAOA for 11 to 50 assets, annealing above 50
  and CVaR-VQE under a tail-risk constraint, held in place by a regression
  test that a selector returning the classical method would have failed;
- a request for real market data that was answered with generated prices
  under the real ticker symbols and recorded as a run on market data;
- a chemistry result that reported a "mean-field error closed by the quantum
  run: 100.0%" for a subspace that covers the whole space;
- runtimes and break-even sizes that nobody had measured, and the quadratic
  speed-up of amplitude estimation without the condition under which it was
  measured;
- a job detail saying that the QAOA circuit had been warm-started from a
  classical solution, which the code reports and does not use;
- eleven panel options that no back-end read;
- a reference of the programming interface that described calls the public
  hosts do not serve and gave 16 portfolio methods where 14 execute.

### Two defects of the platform outside the experiments

- A run was recorded as executed on a vendor while the worker fell back to
  a local simulator, because a credential resolved in the API process is not
  available in the worker process. Workers now resolve credentials again and
  fail a hardware run for which none resolves.
- Twice a field sent by one component was dropped by the next: once because
  a schema library ignores undeclared keys, once because the gateway nested
  a field that the receiver declares at the top level. In the second case a
  check of the tenant's hardware credential did not run for hardware
  requests routed through the gateway; it was repaired the same day.
