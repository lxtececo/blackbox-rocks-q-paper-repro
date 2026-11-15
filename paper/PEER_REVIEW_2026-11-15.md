# Referee report — "Every Quantum Run Needs a Referee" (draft of 2026-11-15)

**Role:** internal review written from the perspective of a physicist working in
quantum algorithms and quantum simulation. **Recommendation:** major revision.
Every point is answered below, either by a new experiment or by a text change,
in the revision of 2026-11-15.

## Summary of the submission

The paper describes a multi-tenant platform that runs five application
workloads on gate-based, annealing and (in integration) analog back-ends, and
argues that every quantum result must be shown with a same-instance classical
referee. It reports six experiments: MLAE vs classical Monte Carlo, SqDRIFT/SQD
vs FCI, CFLP annealing vs an exact MILP, QAOA vs a random control, routing, and
an adiabatic Rydberg MIS protocol. The candour is welcome and the negative
results are useful to the community. Several comparisons, however, are not yet
like-for-like, and some physical interpretations are under-argued.

## Major points

1. **MLAE vs classical is not like-for-like (Sec. 8.1).** MLAE is scored against
   the discretised target while classical Monte Carlo samples the continuous
   log-normal and is scored against Black–Scholes. The clean comparison is
   classical sampling of the *same* $2^n$-bin distribution against the same
   target. *Addressed:* E1 now includes this baseline; slopes and crossover
   points are reported for it.
2. **A slope of $-1.14$ is steeper than the Heisenberg-limited $-1$.** This cannot
   hold asymptotically and points to a pre-asymptotic first point ($m=1$),
   where the likelihood fit is poorly conditioned. *Addressed:* fit without
   $m=1$ is still $-1.16$ $[-1.27,-1.04]$, so the hypothesis is only partly
   right; the text now attributes the excess to a pre-asymptotic
   maximum-likelihood fit and reads the result as consistent with $N^{-1}$.
3. **The resource argument is order-of-magnitude only.** $P_{\mathrm{ok}}=(1-\varepsilon)^{N_{\mathrm{CX}}}$
   ignores SWAP routing on restricted connectivity, single-qubit and readout
   errors, coherent errors and error mitigation. State preparation via generic
   uniformly controlled rotations scales as $O(2^n)$ CX, and more efficient
   loaders exist. A query-count comparison also hides that one quantum query
   costs hundreds of two-qubit gates while one classical sample costs
   nanoseconds. *Addressed in text.*
4. **The key chemistry question is not asked.** Is the quantum sampler better
   than *classical* configuration selection at the same subspace size? Without
   that control, exactness or failure says nothing about the sampler.
   *Addressed:* new E7 compares SqDRIFT with uniformly random configurations
   (matched subspace size) and with all singles-and-doubles strings, through
   the same SQD diagonalization. Result: SqDRIFT is 20–45× better than random
   and, for stretched N₂, 3× better than singles-and-doubles with a smaller
   subspace. Heat-bath or perturbative selected CI remains the proper referee
   and is listed as a limitation.
5. **"Coverage" needs a definition.** SQD diagonalizes in the Cartesian product
   of the distinct $\alpha$- and $\beta$-strings. A few hundred sampled
   configurations can span thousands of determinants. *Addressed in text.*
6. **QAOA metric conflates sampler and post-selection.** Best-of-$N$ rank
   measures selection as much as the state. Report the energy expectation of
   the optimised state, normalised between the uniform mean and the ground
   energy. Also discuss initialisation, optimiser budget and mixer choice.
   *Addressed:* E6 extended; text discusses.
7. **The Rydberg ablation is confounded and the mechanism is asserted.** The
   two mappings differ in both $\Omega$ and $\Delta_{\mathrm{end}}$. A
   quantitative mechanism is needed: under the blockade mapping with
   $\Delta_{\mathrm{end}}=6\Omega$, $r_c = 6^{-1/6}R_b \approx 0.74\,R_b$, so
   conflicting pairs at $0.74$–$1.0$ of the threshold are not excluded at the
   end of the sweep. Under the cutoff mapping with $\Omega=\Delta_{\mathrm{end}}/2$,
   $R_b = 2^{1/6} r_c \approx 1.12\,r_c$, so non-conflicting pairs just beyond
   the threshold are blockaded during the sweep, which predicts valid but
   sub-optimal outputs and the observed correlation with near-threshold pairs.
   *Addressed in text.*
8. **Noiseless simulation of an analog device.** The local AHS simulator omits
   decoherence, atom loss, position disorder, Rabi inhomogeneity and readout
   error, and the 4 µs sweep is quasi-adiabatic at best. Claims must be
   restricted to the ideal model. *Addressed in text and limitations.*
9. **Statistics with two seeds.** A 95% normal interval from $n=2$ is not
   meaningful. *Addressed:* ranges reported where $n\le3$.

## Minor points

1. Table 5 (chemistry) floats into Sec. 8.1. *Fixed:* result tables are placed
   in their subsections.
2. Define an oracle query: one application of the state-preparation unitary
   $\mathcal A$; a shot at Grover power $k$ costs $2k+1$. *Fixed.*
3. Antithetic sampling counts payoff evaluations, not independent pairs; state
   this. *Fixed.*
4. State basis and active space: STO-3G, no frozen core; N$_2$ has 10 spatial
   orbitals and (7,7) electrons, FCI dimension $\binom{10}{7}^2=14{,}400$.
   *Fixed.*
5. Say "noiseless state-vector sampling" rather than "simulator". *Fixed.*
6. Facility location: give sampler wall time next to MILP time. *Fixed.*
7. "Adiabatic" should be "quasi-adiabatic" for a 4 µs sweep. *Fixed.*
8. QAOA training on a noiseless simulator overstates hardware performance; say
   so where the result is stated. *Fixed.*

## Assessment after revision

The revised manuscript is a careful negative-results and methodology paper.
Its main contribution is not any single number but the demonstration that
stronger referees and cheap controls overturn plausible claims. The remaining
weakness is the absence of hardware data for the QUBO and analog workloads,
which the authors acknowledge.

## Follow-up (hardware round, 2026-11-15)

The remaining weakness was partly addressed. Section 8.7 adds MLAE, SqDRIFT on
N2 and QAOA p=1 on `ibm_marrakesh` (IBM Heron r2, 41 s QPU, nine jobs; job IDs
in the ancillary results), plus the transpiled cost of the MLAE circuits on
the device (E8b).

The new noise control (E9) is the most important addition. Without it, the
hardware SqDRIFT result would read as better than simulation, by 100–5,000×.
With it, the result is a pipeline artefact: uniform random bitstrings at the
same shot count reach the same or better error, because configuration
recovery fills 60–75% of the FCI space. The paper reports this as a negative
result and adds the noise control to the methodology.

Still open:
- Annealing and analog hardware.
- QAOA trained on the device rather than on a simulator.
- A chemistry instance large enough that recovery cannot fill the space.
- Repeat runs across calibration days.
