"""E27 - Amplitude estimation against ideal unamplified sampling at equal oracle queries, on screened qubits.

Analysis plan fixed before any device job of this script; only --plan and --dry-run were run before.

Question. Does an estimate of amplitude estimation with Grover powers up to 6 on screened qubits
lie within one standard error of ideal (noiseless) unamplified sampling of the amplitude with a
larger probability than an estimate of ideal unamplified sampling with the same number of oracle
queries? Unamplified sampling on the same qubits in the same job is measured as well and
reported descriptively. Hardware round 2B (E14 part B and E15, ibm_marrakesh) measured on
three screened 3-qubit sets a contrast of 0.905 to 0.924 per Grover iterate, and a bound
Gamma_k = (2k+1) w_k^2 on the information per oracle query that peaked at 3.3 to 4.8 at k = 5 to 6.
Gamma was defined after those data had been seen. This script is the test fixed in advance.

Design
 1. Register. The 3-qubit register of Round 2B, e15.build_register(2, ...): two price qubits and the
    objective qubit, state preparation A followed by k Grover iterates, every qubit measured. Its
    amplitude a = 0.14126 scores the estimates, sets the tolerance of decision 5 and enters the
    parts marked "uses a". Neither estimator reads it.
 2. Two kinds of job, both with the sampler settings of E14: XY4 dynamical decoupling, gate
    twirling, at most 300 s of execution.
    --screen  The readout screen of E15: every qubit prepared in |0> and in |1>, 2 000 shots each. A
              qubit with P(read 0 | 1) > 0.03 or P(read 1 | 0) > 0.02 is rejected, and so is a qubit
              that cannot be screened. Manifest kind "screen".
    --submit  ONE job, manifest kind "main". It needs a collected screen of the same device whose
              job finished at most 30 minutes before the main job is submitted; the collected
              screen that finished last is used. Without such a screen nothing is submitted, and
              the age is stated when a screen is listed. The finish time is the time "finished"
              that --collect stores with the screen (hardware_jobs.job_times). If the entry of a
              screen has no finish time, its submission time is used in its place, which
              overstates the age. The age is counted in whole seconds; a fraction of a second of
              a stored time is dropped. It is checked for the last time immediately before the
              job is handed to hardware_jobs.submit, which asks the service for the state of the
              listed jobs, serialises the circuits and then sends the job. The main job may
              wait in a queue: the age of the screen when the main job is executed is not
              controlled by the script. It is reported with the result: the screen's finish time,
              the main job's submission time, its execution start and stop, and the age of the
              screen at the main job's submission and at the start of its execution.
              S disjoint qubit sets (--sets, default 3) are placed by E15's rule, one after the
              other: the circuits are transpiled against the largest connected group of accepted
              qubits not yet used, with each qubit's measurement error set to its screened mean
              assignment error; optimisation level 3, transpiler seeds 0 to 7, the fewest two-qubit
              gates; every power starts from the qubits chosen for k = 1. If fewer than S sets can
              be placed, nothing is submitted.
 3. Circuits per set: the powers k = 0..8, and the readout calibration (all 0, all 1) on the qubits
    the set's circuits measure, 4 000 shots each. A shot at power k costs 2k+1 oracle queries. The
    calibration uses no oracle queries and is not counted in Q; all repetitions of a set share it.
 4. Repetitions: Q = 2 450 oracle queries per repetition and estimator. Per set, R1 = 200
    repetitions (--primary-repetitions) of AE with the primary schedule and R2 = 20 repetitions
    (--repetitions) of the direct estimator and of each schedule that is reported descriptively
    (decision 6). R1 = 200 and R2 = 20 were fixed before any device data. The primary schedule
    takes 350 shots per repetition, so its repetitions are cheap.
    Below, p10 = P(read 1 | 0) and p01 = P(read 0 | 1) are those of the physical qubit on
    which the objective qubit is read at that power, from the calibration of the same job.
    direct  2 450 shots at k = 0. Estimate: (p - p10) / (1 - p01 - p10) clipped to [0, 1], p the
            fraction of 1 on the objective qubit.
    AE      Primary schedule: 50 shots at each power 0..6 (50 x 49 = 2 450 queries). Maximum
            likelihood over (theta, c, f) of the binomial counts of 1 on the objective qubit, with
              P(read 1 at power k) = p10 + (1 - p01 - p10) P_k,
              P_k = 1/2 + c f^k (sin^2((2k+1) theta) - 1/2),
            theta in [0, pi/2], c and f in [0, 1]. The likelihood has several maxima. It is
            evaluated on a grid of 400 values of theta and 21 values each of c and f; the three
            highest local maxima over theta of its largest value over (c, f) are each refined
            within the bounds (L-BFGS-B), and the highest result is taken. Estimate: sin^2(theta).
    The shots of the two estimators are disjoint, and so are the shots of different schedules
    (decision 6). The job has one run per set and power. The outcomes of a run are put in a seeded
    random order (common.point_seed(27, 1, set, k)) and cut into consecutive blocks: at k = 0 first
    the R2 blocks of the direct estimator; then, at each of its powers, the R1 blocks of the
    primary schedule; then, at each of its powers, the R2 blocks of every other schedule, the
    schedules in the order 2, 3, 4, 5, 8. This is the procedure of E22 and it assumes that
    the device is stable over the job: if the shots of a run are independent draws from one
    distribution, disjoint blocks are independent experiments. A drift within the job is averaged
    into every block and cannot be seen.
 5. Primary criterion (uses a): an exact binomial test. The tolerance is t = sqrt(a (1 - a) / Q) =
    0.00704, the standard error of ideal (noiseless) unamplified sampling with Q = 2 450 shots.
    q0 = P(|X / Q - a| <= t) for X ~ Binomial(Q, a), computed exactly (scipy.stats.binom) as the
    sum of the probabilities of the integers x with |x / Q - a| <= t: x = 329 to 363, q0 = 0.69001.
    hits is the number of the S x R1 AE estimates of the primary schedule with
    |estimate - a| <= t. Hypothesis: an AE estimate lies within t of the amplitude with a larger
    probability than an estimate of ideal unamplified sampling with the same number of oracle
    queries. Test: the one-sided exact binomial test, p = P(Binomial(S x R1, q0) >= hits)
    (scipy.stats.binomtest, alternative "greater"), alpha = 0.01. The hypothesis is confirmed only
    if p <= alpha; with 3 x 200 repetitions that is 441 hits or more. The hit rate is reported
    with its exact (Clopper-Pearson) 95% interval, and the hits of every set.
    Validity. The test is exact if the repetitions are independent. If their probabilities
    differ, for example between sets, it remains valid for the mean of the probabilities
    (Hoeffding 1956, "On the distribution of the number of successes in independent trials";
    its condition, critical number of hits - 1 >= n q0, holds: 440 against 414.0). The
    repetitions of a set are disjoint blocks of shots and share the set's qubits and its readout
    calibration (2 x 4 000 shots, no oracle queries, not counted in Q): the result holds for these
    sets with this job's calibration, not for other qubits, devices or days.
 6. Reported descriptively only, whatever the outcome; none of it is a criterion.
    - rho = MSE_AE / (a (1 - a) / Q), per set and pooled: MSE_AE is the mean of (estimate - a)^2
      over the repetitions of AE with the primary schedule, a (1 - a) / Q the mean squared error of
      ideal unamplified sampling with Q shots. With it a two-sided 95% BCa bootstrap interval
      (scipy.stats.bootstrap, 9 999 resamples, the repetitions resampled within their set; seeds
      common.point_seed(27, 3) pooled and (27, 3, set) per set) and 1 / rho, the information per
      query realised. An interval of a mean squared error cannot exclude rare large errors that
      the repetitions did not contain. If an interval cannot be computed, the result states why.
    - Per set and pooled over the sets: bias, standard deviation and RMSE of the direct estimator
      (R2 repetitions per set) and of AE with the primary schedule (R1 per set), each with
      bootstrap 95% intervals (percentiles of 4 000 resamples, the repetitions resampled within
      their set), and the ratio MSE_AE / MSE_direct with a bootstrap 95% interval (the two
      estimators resampled independently within each set). The ratio does not isolate
      amplification: the direct estimator keeps the state-preparation bias (1 - c)(1/2 - a). Also
      the value of ideal unamplified sampling, sqrt(a (1 - a) / Q) = 0.00704.
    - The schedules with largest power m = 2, 3, 4, 5 and 8 at no more than Q queries, per set
      and pooled: floor(2450 / (m+1)^2) shots at each power 0..m (2 448, 2 448, 2 450, 2 448 and
      2 430 queries), R2 repetitions per set, cut from the same
      runs. Bias, standard deviation and RMSE with the same intervals; the ratio of the mean
      squared error to that of the direct estimator and to a (1 - a) / Q, with percentile
      intervals for which the R2 repetitions of the schedule and of the direct estimator are
      resampled together, by their number in the set; the number of repetitions in which the
      schedule has a smaller squared error than the direct estimator with the same number. For
      every schedule, the primary one included: the number of estimates within t of the
      amplitude, with the exact 95% interval of the rate.
    - The depth probe of Round 2B on all shots of every power (uses a): the signal fraction w_k
      with a bootstrap interval, the contrast per iterate f_Q fitted to the objective qubit as in
      Round 2B, Gamma_k = (2k+1) w_k^2 and its maximum, and for every schedule the query-weighted
      mean of Gamma_k over its powers, sum (2k+1) Gamma_k / (m+1)^2: the largest information per
      query, in units of an ideal unamplified shot, that Gamma allows the schedule.
    - An "oracle-contrast" AE estimate (uses a, so it is only an upper reference): the counts of
      the R1 repetitions of the primary schedule with c and f fixed at the values fitted to the
      set's probe, theta alone free. c and f are fitted to the objective-qubit counts of all
      powers with a known and with the readout terms of decision 4. They are not derived from
      w_k, which also contains the readout errors of the price qubits; the fit of c f^k to w_k
      is reported next to them.
    - The measured readout errors of every set, and the qubits that read above a screen threshold
      in the job's own calibration.
    - Whether the result agrees with what Gamma predicted: the information per query that AE
      realised, 1 / rho with the interval of rho, against the largest that Gamma allows the
      primary schedule and against 1, the value of ideal unamplified sampling. Gamma_k
      is taken from w_k as in Round 2B. w_k contains the readout errors of all three qubits,
      which the AE estimate corrects on the objective qubit, so the allowed value is somewhat low
      for it.
 7. Shots per set and power: for every block that uses the power, its repetitions times its
    shots, and at least 2 000 for the probe. With R1 = 200 and R2 = 20: 71 420, 22 420, 22 420,
    16 980, 13 920, 11 960, 10 600, 2 000 and 2 000 shots at k = 0..8; with the calibration
    181 720 shots per set, 545 160 in three sets. --plan prints the table, the total and the
    QPU-seconds estimate with its basis. --submit refuses a job estimated above 200 s or a run
    above the device's shot limit.
 8. --plan also prints, for the S, R1 and R2 it is given:
    - t, q0 with its integers x, and the critical number of hits: the smallest number of hits
      among the S x R1 repetitions with p <= alpha, with the exact size of the test there.
    - Power, analytic, for AE errors that are normally distributed without bias: for rho = 0.4,
      0.5, 0.6, 0.7, 0.8 and 0.9 the probability of a hit, 2 Phi(1 / sqrt(rho)) - 1, and the exact
      binomial probability that S x R1 repetitions reach the critical number of hits.
    - Size, by simulation, as a check of the code only (the test is exact): ideal unamplified
      sampling in the place of AE, S x R1 estimates Binomial(Q, a) / Q per simulated job, 10 000
      simulated jobs. Printed: the mean hit rate, and the rate at which the criterion is met with
      an exact (Clopper-Pearson) 95% interval; it must be at or below alpha up to simulation noise.
    - Power, by a seeded Monte Carlo under the model of decision 4, with readout errors 0.017 and
      0.019 estimated per set from 4 000 calibration shots each as in the job; 100 simulated jobs
      per assumption, each with S sets, R1 repetitions of the primary schedule and R2 of the
      direct estimator: the Round 2B contrast (c = 0.90, f = 0.92), a pessimistic one (c = 0.90,
      f = 0.58, the 4-qubit register of Round 2B) and a state preparation without loss (c = 1.00,
      f = 0.92). Printed: bias, standard deviation and RMSE of both estimators, the mean hit rate,
      the mean of rho, and the share of the simulated jobs in which the criterion is met with an
      exact 95% interval.
    - The wall time of the simulations and of the whole plan.
 9. --dry-run runs the whole pipeline without a device: the coupling map and gates of the packaged
    model fake_fez, and a synthetic screen whose counts are drawn from seeded readout errors
    (P(read 1 | 0) = 0.006 exp(0.8 z) and P(read 0 | 1) = 0.012 exp(0.8 z'), z and z' standard
    normal, per qubit). The circuits of the main job run on Aer, each on the qubits it uses, without
    gate errors or, with --noise P, with a depolarising error P on every two-qubit gate; the
    synthetic readout errors are then applied to every measured bit. --quick: 2 sets, R1 = 20,
    R2 = 5 and a tenth of the resamples of the descriptive bootstraps other than that of rho,
    which keeps its 9 999. Dry-run counts carry no information about hardware; the result says so
    and its name ends in _dryrun.

What the primary criterion does not show
 - The test compares AE on the device with a computed value, q0, not with a measurement.
   Unamplified sampling on the device (the direct estimator) is reported descriptively: it
   corrects the readout only, and a state preparation that keeps a contrast c biases it by
   (1 - c)(1/2 - a), 0.036 at c = 0.90.
 - The test concerns the probability that an estimate lies within t of the amplitude, at one
   amplitude, a = 0.14126, with Q = 2 450 and the primary schedule. It says nothing about the
   size of the errors outside t: the mean squared error can be large while most estimates lie
   within t. rho, the bias and the standard deviation of AE are reported descriptively.
 - The repetitions of a set are blocks of one run per power and share the set's readout
   calibration (decisions 4 and 5). The result therefore holds for these sets with this job's
   calibration, not for other qubits, devices or days.

No run is excluded. Every submitted job is listed in results/e27_submission.json and in every
result. Raw counts are archived before any analysis. A device result is never replaced. A device
that already has a finished main job of this experiment gets a further one only with --again.

Usage
  e27_hardware_amplitude_estimation.py --plan [--sets S] [--primary-repetitions R1] [--repetitions R2]
        offline: jobs, pubs, shots per power, the QPU-seconds estimate with its basis, and the
        tolerance, q0, critical number of hits, size and power of the criterion (decision 8)
  e27_hardware_amplitude_estimation.py --dry-run [--noise P] [--quick] [--sets S] [--primary-repetitions R1]
        [--repetitions R2]
        packaged device model and Aer (decision 9). The result name ends in _dryrun.
  e27_hardware_amplitude_estimation.py --status      device, account usage, jobs, the collected screen that finished last
  e27_hardware_amplitude_estimation.py --screen      submit the readout screen
  e27_hardware_amplitude_estimation.py --submit [--sets S] [--primary-repetitions R1] [--repetitions R2] [--again]
  e27_hardware_amplitude_estimation.py --collect     archive the counts and times of finished jobs, then analyse
  e27_hardware_amplitude_estimation.py --analyse     analyse archived counts that have no result yet
  e27_hardware_amplitude_estimation.py -h, --help    the command line
Order on a device: --screen; --collect once the screen has finished; --submit within 30 minutes of
the screen's finish; --collect once the main job has finished.
Environment: QFPO_IBM_BACKEND names the device; there is no default.
Device, chosen on 20 November 2026 before any run of this script: ibm_marrakesh, the device of
Round 2B, on which the contrast behind the plan's power figures was measured.
"""

from __future__ import annotations

import calendar
import json
import math
import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor

import hardware_jobs as hj
import numpy as np
from common import RESULTS, point_seed, pool_workers
from scipy.optimize import minimize
from scipy.stats import beta, binom, binomtest, bootstrap, chi2, norm

e14 = hj.e14
e15 = hj._import_script("e15_mlae_screened_qubits")      # screen thresholds, placement rule, register of Round 2B

EXPERIMENT = "e27"
STEM = "e27_hardware_amplitude_estimation"
ALPHA = 0.01
N_PRICE = 2                                      # price qubits; the objective qubit is qubit N_PRICE
POWERS = list(range(9))                          # Grover powers of the job
PRIMARY = 6                                      # largest power of the tested schedule
OTHER = (2, 3, 4, 5, 8)                          # largest powers of the schedules reported descriptively
QUERIES = 2_450                                  # oracle queries per repetition and estimator
SETS, PRIMARY_REPETITIONS, REPETITIONS = 3, 200, 20    # S, R1 and R2 of decision 4
PROBE_SHOTS = 2_000                              # least shots of a power
CAL_SHOTS = e14.CAL_SHOTS                        # 4 000 per calibration circuit, as in Round 2B
SCREEN_SHOTS = e15.SCREEN_SHOTS                  # 2 000 per screen circuit, as in Round 2B
MAX_SCREEN_AGE_S = 1_800
MAX_JOB_SECONDS = e14.MAX_JOB_SECONDS
THETA_GRID = np.linspace(1e-3, math.pi / 2 - 1e-3, 400)
CONTRAST_GRID = np.linspace(0.0, 1.0, 21)
STARTS = 3                                       # local maxima over theta that are refined
RHO_RESAMPLES = 9_999                            # bootstrap resamples of the interval of rho (decision 6)
RESAMPLES = {"errors": 4_000, "signal_fraction": 300, "contrast": 200}    # the other descriptive bootstraps
QUICK = {"sets": 2, "primary_repetitions": 20, "repetitions": 5,
         "resamples": {"errors": 400, "signal_fraction": 30, "contrast": 20}}
SCENARIOS = (("Round 2B contrast", 0.90, 0.92), ("pessimistic contrast (4-qubit register)", 0.90, 0.58),
             ("reference: state preparation without loss", 1.00, 0.92))
PLAN_READOUT = (0.017, 0.019)                    # P(read 1 | 0), P(read 0 | 1): medians of the objective qubits of Round 2B
PLAN_EXPERIMENTS = 100                           # simulated jobs per assumption
PLAN_NULL_EXPERIMENTS = 10_000                   # simulated jobs of the size check
PLAN_RHO = (0.4, 0.5, 0.6, 0.7, 0.8, 0.9)        # the rho of the analytic power table
DRY_READOUT = ((0.006, 0.8), (0.012, 0.8))       # median and log-spread of P(read 1 | 0) and of P(read 0 | 1)
FACT_KEYS = ("amplitude", "powers", "plan", "calibration_shots", "sets")
COMPARED = ("direct", "ae", "mse_ratio_ae_over_direct")    # what a set and the pooled part take from the primary schedule
DESIGN = __doc__.split("\nUsage\n")[0].strip()


# --------------------------------------------------------------------------- shots


def shot_plan(primary: int, repetitions: int) -> dict:
    """Decisions 4, 6 and 7: where the blocks of every estimator start in the permuted shots of a power, and the shots.

    primary: R1, the repetitions of the primary schedule; repetitions: R2, those of the direct estimator and the others.
    """
    used = [0] * len(POWERS)
    used[0] = repetitions * QUERIES
    schedules = []
    for m in (PRIMARY, *OTHER):
        shots, blocks = QUERIES // (m + 1) ** 2, primary if m == PRIMARY else repetitions
        schedules.append({"largest_power": m, "shots_per_power": shots, "queries": shots * (m + 1) ** 2,
                          "repetitions": blocks, "offsets": used[:m + 1]})
        for k in range(m + 1):
            used[k] += blocks * shots
    return {"primary_repetitions": primary, "repetitions": repetitions, "queries": QUERIES, "schedules": schedules,
            "needed": used, "shots": [max(n, PROBE_SHOTS) for n in used]}


def expand(counts: dict, seed: int) -> np.ndarray:
    """The outcomes of one run as integers, one per shot, in a seeded random order (decision 4)."""
    keys = sorted(counts)
    shots = np.repeat([int(key, 2) for key in keys], [counts[key] for key in keys])
    np.random.default_rng(seed).shuffle(shots)
    return shots


def cut(ones: np.ndarray, offset: int, size: int, repetitions: int) -> np.ndarray:
    """Counts of 1 in consecutive blocks of the permuted shots, one block per repetition."""
    return ones[offset:offset + size * repetitions].reshape(repetitions, size).sum(axis=1)


# --------------------------------------------------------------------------- the two estimators


def model_p(theta, c, f, powers, p10, p01):
    """Decision 4: the probability of reading 1 on the objective qubit at each power."""
    return p10 + (1 - p01 - p10) * (0.5 + c * f ** powers * (np.sin((2 * powers + 1) * theta) ** 2 - 0.5))


def direct_estimate(hits, shots: int, p10: float, p01: float) -> np.ndarray:
    """Decision 4: the fraction of 1 at power 0, corrected for the readout and clipped to [0, 1]."""
    return np.clip((np.asarray(hits, float) / shots - p10) / (1 - p01 - p10), 0.0, 1.0)


def ae_fit(powers, hits, shots: int, p10, p01, contrast=None) -> np.ndarray:
    """Decision 4: maximum-likelihood (amplitude, c, f) per row of hits, one row per repetition.

    hits: counts of 1 on the objective qubit, one column per power; shots: per power.
    contrast: (c, f) held fixed, theta alone free (the oracle-contrast estimate of decision 6).
    """
    powers, hits = np.asarray(powers, float), np.atleast_2d(np.asarray(hits, float))
    if len(hits) > 25:                           # 25 repetitions at a time: the grid of the likelihood is large
        return np.concatenate([ae_fit(powers, hits[i:i + 25], shots, p10, p01, contrast) for i in range(0, len(hits), 25)])
    c_grid, f_grid = (CONTRAST_GRID, CONTRAST_GRID) if contrast is None else ([contrast[0]], [contrast[1]])
    pairs = np.array([(c, f) for c in c_grid for f in f_grid])
    ideal = np.sin((2 * powers[None, :] + 1) * THETA_GRID[:, None]) ** 2
    p = p10 + (1 - p01 - p10) * (0.5 + (pairs[:, :1] * pairs[:, 1:] ** powers[None, :])[None] * (ideal[:, None, :] - 0.5))
    p = np.clip(p, 1e-12, 1 - 1e-12).reshape(-1, len(powers))
    surface = (hits @ np.log(p).T + (shots - hits) @ np.log1p(-p).T).reshape(len(hits), len(THETA_GRID), len(pairs))
    bounds = [(0.0, math.pi / 2), (0.0, 1.0), (0.0, 1.0)][:3 if contrast is None else 1]
    out = []
    for row, grid in zip(hits, surface):
        def nll(x):
            c, f = (x[1], x[2]) if contrast is None else contrast
            return e14.binomial_nll(model_p(x[0], c, f, powers, p10, p01), row, shots)

        profile = grid.max(axis=1)
        peaks = np.flatnonzero(np.r_[True, profile[1:] >= profile[:-1]] & np.r_[profile[:-1] >= profile[1:], True])
        best = None
        for t in peaks[np.argsort(-profile[peaks], kind="stable")][:STARTS]:
            start = np.array([THETA_GRID[t], *pairs[int(grid[t].argmax())]])[:len(bounds)]
            found = minimize(nll, start, method="L-BFGS-B", bounds=bounds)
            x, value = (found.x, float(found.fun)) if found.fun <= nll(start) else (start, nll(start))
            if best is None or value < best[1]:
                best = (x, value)
        c, f = (best[0][1], best[0][2]) if contrast is None else contrast
        out.append([math.sin(best[0][0]) ** 2, float(c), float(f)])
    return np.array(out)


def exact_interval(successes: int, n: int) -> tuple:
    """Exact (Clopper-Pearson) two-sided 95% interval of a proportion."""
    return (float(beta.ppf(0.025, successes, n - successes + 1)) if successes else 0.0,
            float(beta.ppf(0.975, successes + 1, n - successes)) if successes < n else 1.0)


def tolerance_facts(amplitude: float) -> dict:
    """Decision 5: the tolerance t, q0 = P(|X / Q - a| <= t) for X ~ Binomial(Q, a), the first and last integer it sums."""
    t = math.sqrt(amplitude * (1 - amplitude) / QUERIES)
    x = np.arange(QUERIES + 1)
    inside = x[np.abs(x / QUERIES - amplitude) <= t]
    return {"tolerance": t, "q0": float(binom.pmf(inside, QUERIES, amplitude).sum()),
            "q0_range": [int(inside[0]), int(inside[-1])]}


def within(estimates, amplitude: float, tolerance: float) -> dict:
    """The estimates within the tolerance of the amplitude, and their rate with its exact 95% interval."""
    estimates = np.asarray(estimates, float)
    hits = int((np.abs(estimates - amplitude) <= tolerance).sum())
    return {"hits": hits, "repetitions": int(estimates.size), "hit_rate": hits / estimates.size,
            "hit_rate_ci95": list(exact_interval(hits, estimates.size))}


def critical_hits(n: int, q0: float) -> int | None:
    """The smallest number of hits among n repetitions with p <= ALPHA in the test of decision 5; None if there is none."""
    enough = np.flatnonzero(binom.sf(np.arange(n + 1) - 1, n, q0) <= ALPHA)
    return int(enough[0]) if len(enough) else None


def criterion(estimates: list, amplitude: float) -> dict:
    """Decision 5: the exact binomial test of the hits. estimates: per set the AE estimates of the primary schedule."""
    facts = tolerance_facts(amplitude)
    found = within(np.concatenate(estimates), amplitude, facts["tolerance"])
    p = float(binomtest(found["hits"], found["repetitions"], facts["q0"], alternative="greater").pvalue)
    return {**facts, **found, "hits_per_set": [within(x, amplitude, facts["tolerance"])["hits"] for x in estimates],
            "p": p, "alpha": ALPHA, "confirmed": bool(p <= ALPHA),
            "method": "one-sided exact binomial test, p = P(Binomial(repetitions, q0) >= hits) (scipy.stats.binomtest, "
                      "alternative greater). Hypothesis: an AE estimate lies within the tolerance of the amplitude with a "
                      "larger probability than an estimate of ideal unamplified sampling with the same number of oracle "
                      "queries, for which that probability is q0",
            "verdict": f"confirmed: p <= {ALPHA}" if p <= ALPHA else f"not confirmed: p > {ALPHA}"}


def rho_interval(squared: list, amplitude: float, seed) -> dict:
    """Decision 6, descriptive: rho and its two-sided 95% BCa interval. squared: per set the squared errors of AE."""
    ideal = amplitude * (1 - amplitude) / QUERIES

    def rho(*samples, axis=-1):
        return np.concatenate(samples, axis=axis).mean(axis=axis) / ideal

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        found = bootstrap(tuple(squared), rho, n_resamples=RHO_RESAMPLES, confidence_level=0.95, method="BCa",
                          vectorized=True, random_state=np.random.default_rng(seed))
    value, ends = float(rho(*squared)), [float(found.confidence_interval.low), float(found.confidence_interval.high)]
    out = {"value": value, "ci95": ends if all(map(math.isfinite, ends)) else None,
           "information_per_query_realised": 1 / max(value, np.finfo(float).tiny)}
    if out["ci95"] is None:
        out["interval_not_computed"] = ("; ".join(sorted({str(w.message) for w in caught}))
                                        or "scipy.stats.bootstrap returned no finite limit")
    return out


# --------------------------------------------------------------------------- screen and placement


def screen_circuits(backend) -> tuple:
    """E15's screen: (the qubits that have an x gate and a measurement, the circuits all 0 and all 1 on them)."""
    every = [q for q in range(backend.num_qubits) if (q,) in backend.target["measure"] and (q,) in backend.target["x"]]
    return every, e14.calibration_circuits(backend, every)


def screen_summary(counts: list, every: list, total: int, shots: int) -> dict:
    """E15's screen on its counts: the readout errors of every qubit and the qubits its thresholds reject."""
    measured = e14.readout_errors(counts[0][0], counts[1][0], len(every), shots)
    p10, p01 = np.ones(total), np.ones(total)     # a qubit that cannot be screened counts as rejected
    p10[every], p01[every] = measured["p_read1_given0"], measured["p_read0_given1"]
    rejected = [q for q in range(total) if p01[q] > e15.MAX_P01 or p10[q] > e15.MAX_P10]
    return {"shots": shots, "num_qubits": total, "screened_qubits": list(every), "p_read1_given0": p10.tolist(),
            "p_read0_given1": p01.tolist(), "rejected_qubits": rejected,
            "thresholds": {"max_p_read0_given1": e15.MAX_P01, "max_p_read1_given0": e15.MAX_P10},
            "median_p_read1_given0": float(np.median(p10[every])), "median_p_read0_given1": float(np.median(p01[every]))}


def place_sets(backend, screen: dict, register: dict, sets: int) -> list:
    """Decision 2: E15's placement rule, one set after the other, on qubits the screen accepted."""
    p10, p01 = np.array(screen["p_read1_given0"]), np.array(screen["p_read0_given1"])
    removed, placed = set(screen["rejected_qubits"]), []
    for index in range(sets):
        largest = len(e15.largest_group(backend, removed))
        if largest < N_PRICE + 1:
            raise SystemExit(f"not submitted: {index} of {sets} sets can be placed; the largest connected group of accepted "
                             f"qubits not yet used has {largest} qubits")
        target, group = e15.screened_target(backend, p10, p01, removed)
        _, compact = e15.prepare(register["circuits"][1], target, group, backend.num_qubits)
        circuits = [e15.prepare(c, target, group, backend.num_qubits, initial_layout=compact)[0] for c in register["circuits"]]
        touched = {c.find_bit(q).index for c in circuits for inst in c.data for q in inst.qubits}
        if touched & removed:
            raise SystemExit(f"not submitted: set {index} uses a rejected or already used qubit: {sorted(touched & removed)}")
        facts = [hj.circuit_facts(backend, c) for c in circuits]
        placed.append({"circuits": circuits, "set": index, "layout": [group[i] for i in compact],
                       "measured_physical": [row["physical_qubits"] for row in facts],
                       "calibrated": sorted({q for row in facts for q in row["physical_qubits"]}),
                       "two_qubit_gates": [row["two_qubit_gates"] for row in facts], "depth": [row["depth"] for row in facts],
                       "p_no_two_qubit_error_reported": [row["p_no_two_qubit_error"] for row in facts],
                       "calibration_date": facts[0]["calibration_date"]})
        removed |= touched
    return placed


def build_job(backend, screen: dict, register: dict, sets: int, primary: int, repetitions: int) -> dict:
    """Decisions 2, 3 and 7: per set the nine powers and the two calibration circuits, and what the manifest keeps."""
    plan = shot_plan(primary, repetitions)
    pubs, rows = [], []
    for placed in place_sets(backend, screen, register, sets):
        pubs += [(circuit, None, shots) for circuit, shots in zip(placed.pop("circuits"), plan["shots"])]
        pubs += [(circuit, None, CAL_SHOTS) for circuit in e14.calibration_circuits(backend, placed["calibrated"])]
        rows.append(placed)
    return {"pubs": pubs, "facts": {"amplitude": register["amplitude"], "powers": list(POWERS), "plan": plan,
                                    "calibration_shots": CAL_SHOTS, "sets": rows}}


def describe(label: str, screen: dict, job: dict) -> None:
    plan = job["facts"]["plan"]
    print(f"{label}: the screen rejects {len(screen['rejected_qubits'])} of {screen['num_qubits']} qubits (median "
          f"P(read 1 | 0) {screen['median_p_read1_given0']:.4f}, P(read 0 | 1) {screen['median_p_read0_given1']:.4f})")
    for row in job["facts"]["sets"]:
        print(f"  set {row['set']}: qubits {row['layout']}, calibrated {row['calibrated']}; two-qubit gates at k = 0..8 "
              f"{row['two_qubit_gates']}; reported product of (1 - error) over them at k = 0, 6, 8: "
              + ", ".join(f"{row['p_no_two_qubit_error_reported'][k]:.3f}" for k in (0, PRIMARY, POWERS[-1])))
    print(f"  {len(job['facts']['sets'])} sets; per set {plan['primary_repetitions']} repetitions of the primary schedule and "
          f"{plan['repetitions']} of the direct estimator and of every other schedule; shots at k = 0..8 {plan['shots']} and "
          f"2 x {CAL_SHOTS:,} calibration shots per set; {len(job['pubs'])} pubs", flush=True)


# --------------------------------------------------------------------------- analysis (the same for device and Aer)


def interval(value: float, resampled) -> dict:
    return {"value": float(value), "ci95": [float(x) for x in np.percentile(resampled, [2.5, 97.5])]}


def compare(ae: list, direct: list, amplitude: float, resamples: int, seed: int, paired: bool) -> dict:
    """Decision 6: bias, standard deviation and RMSE of two estimators, with bootstrap intervals.

    ae, direct: per set the estimates of its repetitions, resampled within their set. paired: both have the same
    number of repetitions per set and the same ones are resampled for both; otherwise each is resampled independently.
    """
    rng = np.random.default_rng(seed)

    def drawn(samples: list) -> np.ndarray:
        sizes = [len(x) for x in samples]
        return np.concatenate([start + rng.integers(0, n, (resamples, n))
                               for start, n in zip(np.cumsum([0] + sizes[:-1]), sizes)], axis=1)

    index = {"direct": drawn(direct)}
    index["ae"] = index["direct"] if paired else drawn(ae)
    errors = {"ae": np.concatenate(ae) - amplitude, "direct": np.concatenate(direct) - amplitude}
    out, mse = {}, {}
    for name, e in errors.items():
        b = e[index[name]]
        mse[name] = (float((e ** 2).mean()), (b ** 2).mean(axis=1))
        out[name] = {"repetitions": len(e), "bias": interval(e.mean(), b.mean(axis=1)),
                     "sd": interval(e.std(ddof=1), b.std(axis=1, ddof=1)),
                     "rmse": interval(math.sqrt(mse[name][0]), np.sqrt(mse[name][1]))}
    floor = np.finfo(float).tiny
    out["mse_ratio_ae_over_direct"] = interval(mse["ae"][0] / max(mse["direct"][0], floor),
                                               mse["ae"][1] / np.maximum(mse["direct"][1], floor))
    if paired:
        ideal = amplitude * (1 - amplitude) / QUERIES
        out["ae_mse_over_ideal_unamplified"] = interval(mse["ae"][0] / ideal, mse["ae"][1] / ideal)
        out["pairs"] = len(errors["ae"])
        out["pairs_with_smaller_ae_error"] = int((errors["ae"] ** 2 < errors["direct"] ** 2).sum())
    return out


def contrast_with_readout(hits, shots, ideal_p, p10, p01) -> tuple:
    """(c, f) by maximum likelihood with the amplitude known and the readout terms of decision 4. It uses a."""
    powers = np.arange(len(hits), dtype=float)

    def nll(x):
        return e14.binomial_nll(p10 + (1 - p01 - p10) * (0.5 + x[0] * x[1] ** powers * (ideal_p - 0.5)), hits, shots)

    fits = [minimize(nll, start, method="L-BFGS-B", bounds=[(0.0, 1.0), (0.0, 1.0)])
            for start in ((0.9, 0.9), (0.5, 0.7), (0.3, 0.5), (0.8, 0.3))]
    best = min(fits, key=lambda found: found.fun)
    return float(best.x[0]), float(best.x[1])


def probe(register: dict, runs: list, p10, p01, plan: dict, rng, resamples: dict) -> dict:
    """Decision 6: the depth probe of Round 2B on all shots of every power of one set. It uses a."""
    mask = e14.objective_mask(N_PRICE)
    rows = []
    for k, (ideal, run) in enumerate(zip(register["ideals"], runs)):
        v = e14.counts_vector(run, N_PRICE + 1)
        total = int(v.sum())
        weight = e14.mixture_weight(v, ideal)
        low, high = np.percentile([e14.mixture_weight(rng.multinomial(total, v / total).astype(float), ideal)
                                   for _ in range(resamples["signal_fraction"])], [2.5, 97.5])
        rows.append({"k": k, "shots": total, "objective_hits": int(v[mask].sum()), "p_hat": float(v[mask].sum()) / total,
                     "p_ideal": float(ideal[mask].sum()), "signal_fraction": weight,
                     "signal_fraction_ci95": [float(low), float(high)], "gamma": (2 * k + 1) * weight ** 2,
                     "gamma_ci95": [(2 * k + 1) * float(low) ** 2, (2 * k + 1) * float(high) ** 2]})
    hits, shots = np.array([r["objective_hits"] for r in rows], float), np.array([r["shots"] for r in rows], float)
    ideal_p = np.array([r["p_ideal"] for r in rows])
    x, fitted, _ = e14.fit_contrast(POWERS, hits, shots, ideal_p)
    again = np.array([e14.fit_contrast(POWERS, rng.binomial(shots.astype(int), fitted), shots, ideal_p)[0]
                      for _ in range(resamples["contrast"])])
    deviance = e14.deviance(hits, shots, fitted)
    c, f = contrast_with_readout(hits, shots, ideal_p, p10, p01)
    slope, level = np.polyfit(POWERS, np.log(np.clip([r["signal_fraction"] for r in rows], 1e-6, None)), 1)
    best, amplified = max(rows, key=lambda r: r["gamma"]), max(rows[1:], key=lambda r: r["gamma"])
    allowed = {}
    for schedule in plan["schedules"]:
        m = schedule["largest_power"]
        allowed[str(m)] = {"value": sum((2 * r["k"] + 1) * r["gamma"] for r in rows[:m + 1]) / (m + 1) ** 2,
                           "ci95": [sum((2 * r["k"] + 1) * r["gamma_ci95"][end] for r in rows[:m + 1]) / (m + 1) ** 2
                                    for end in (0, 1)]}
    return {"rows": rows,
            "contrast_fit_objective_qubit": {
                "f_A": float(x[0]), "f_Q": float(x[1]), "f_A_ci95": [float(v) for v in np.percentile(again[:, 0], [2.5, 97.5])],
                "f_Q_ci95": [float(v) for v in np.percentile(again[:, 1], [2.5, 97.5])], "deviance": deviance,
                "dof": len(POWERS) - 2, "goodness_of_fit_p": float(chi2.sf(deviance, len(POWERS) - 2))},
            "contrast_with_readout": {"c": c, "f": f},
            "contrast_from_signal_fraction": {"c": float(math.exp(level)), "f": float(math.exp(slope))},
            "max_gamma": best["gamma"], "max_gamma_ci95": best["gamma_ci95"], "max_gamma_at_k": best["k"],
            "best_amplified": {"k": amplified["k"], "gamma": amplified["gamma"], "gamma_ci95": amplified["gamma_ci95"]},
            "amplification_can_pay": bool(amplified["gamma_ci95"][0] > 1.0),
            "largest_information_per_query_allowed": allowed}


def gamma_statement(allowed: float, rho: dict) -> dict:
    """Decision 6: the information per query AE realised, 1 / rho, against the largest Gamma allows and against 1."""
    tiny = np.finfo(float).tiny
    if rho["ci95"] is None:
        return {"largest_information_per_query_allowed": allowed,
                "information_per_query_realised": {"value": rho["information_per_query_realised"], "ci95": None},
                "statement": "not stated: the interval of rho could not be computed"}
    realised = {"value": rho["information_per_query_realised"],
                "ci95": [1 / max(rho["ci95"][1], tiny), 1 / max(rho["ci95"][0], tiny)]}
    low, high = realised["ci95"]
    if low > allowed:
        text = "disagrees: AE realised more information per query than Gamma allows"
    elif low > 1:
        text = "agrees: Gamma allows a gain over ideal unamplified sampling and AE realised one"
    elif allowed <= 1:
        text = "agrees: Gamma allows no gain over ideal unamplified sampling and AE shows none"
    elif high < 1:
        text = ("no contradiction, the gain was not realised: Gamma is an upper bound and AE stayed below ideal "
                "unamplified sampling")
    else:
        text = "open: the interval of the information per query that AE realised contains 1"
    return {"largest_information_per_query_allowed": allowed, "information_per_query_realised": realised, "statement": text}


def analyse_counts(register: dict, facts: dict, counts: list, resamples: dict | None = None) -> dict:
    """Decisions 4 to 6 on the counts of one main job. facts: what build_job() returned, as the manifest keeps it."""
    resamples = resamples or RESAMPLES
    plan, amplitude, per_set = facts["plan"], register["amplitude"], len(POWERS) + 2
    tolerance, sets, direct, oracle = tolerance_facts(amplitude)["tolerance"], [], [], []
    ae = {schedule["largest_power"]: [] for schedule in plan["schedules"]}

    def rows_of(estimates: dict, against: list, part: int) -> list:
        """Per schedule, the primary one first: AE against the direct estimator over the given repetitions."""
        rows = []
        for schedule in plan["schedules"]:
            m = schedule["largest_power"]
            found = [x[:, 0] for x in estimates[m]]
            rows.append({**{key: schedule[key] for key in ("largest_power", "shots_per_power", "queries")},
                         "repetitions_per_set": schedule["repetitions"],
                         **compare(found, against, amplitude, resamples["errors"], point_seed(27, 4, part, m), m != PRIMARY),
                         "within_tolerance": within(np.concatenate(found), amplitude, tolerance)})
        return rows

    for i, row in enumerate(facts["sets"]):
        runs = [counts[i * per_set + k][0] for k in range(len(POWERS))]
        readout = e14.readout_errors(counts[i * per_set + len(POWERS)][0], counts[i * per_set + len(POWERS) + 1][0],
                                     len(row["calibrated"]), facts["calibration_shots"])
        where = [row["calibrated"].index(measured[N_PRICE]) for measured in row["measured_physical"]]
        p10 = np.array([readout["p_read1_given0"][j] for j in where])
        p01 = np.array([readout["p_read0_given1"][j] for j in where])
        if (1 - p01 - p10).min() <= 0:
            raise SystemExit(f"set {row['set']} not analysed: the readout calibration of its objective qubit cannot be inverted")
        depth = probe(register, runs, p10, p01, plan, np.random.default_rng(point_seed(27, 2, i)), resamples)
        ones = [(expand(run, point_seed(27, 1, i, k)) >> N_PRICE) & 1 for k, run in enumerate(runs)]
        direct.append(direct_estimate(cut(ones[0], 0, plan["queries"], plan["repetitions"]), plan["queries"], p10[0], p01[0]))
        for schedule in plan["schedules"]:
            m, n = schedule["largest_power"], schedule["shots_per_power"]
            hits = np.stack([cut(ones[k], schedule["offsets"][k], n, schedule["repetitions"]) for k in range(m + 1)], axis=1)
            ae[m].append(ae_fit(range(m + 1), hits, n, p10[:m + 1], p01[:m + 1]))
            if m == PRIMARY:
                fixed = (depth["contrast_with_readout"]["c"], depth["contrast_with_readout"]["f"])
                oracle.append(ae_fit(range(m + 1), hits, n, p10[:m + 1], p01[:m + 1], fixed)[:, 0])
        rows = rows_of({m: [ae[m][i]] for m in ae}, [direct[i]], i)
        own = rho_interval([(ae[PRIMARY][i][:, 0] - amplitude) ** 2], amplitude, point_seed(27, 3, i))
        sets.append({
            "set": row["set"], "layout": row["layout"], "two_qubit_gates": row["two_qubit_gates"],
            "readout": {"physical_qubits": row["calibrated"], **readout,
                        "objective_qubit_at_each_power": [measured[N_PRICE] for measured in row["measured_physical"]],
                        "applied_p_read1_given0": p10.tolist(), "applied_p_read0_given1": p01.tolist(),
                        "above_a_screen_threshold": [q for q, a, b in zip(row["calibrated"], readout["p_read1_given0"],
                                                                          readout["p_read0_given1"])
                                                     if a > e15.MAX_P10 or b > e15.MAX_P01]},
            "estimates": {"direct": direct[i].tolist(), "oracle_contrast": oracle[i].tolist(),
                          "ae_amplitude_c_f": {str(m): ae[m][i].tolist() for m in ae}},
            **{key: rows[0][key] for key in COMPARED}, "rho": own, "schedules": rows,
            "oracle_contrast": compare([oracle[i]], [direct[i]], amplitude, resamples["errors"], point_seed(27, 5, i), False),
            "gamma": gamma_statement(depth["largest_information_per_query_allowed"][str(PRIMARY)]["value"], own),
            "probe": depth})
    schedules = rows_of(ae, direct, 99)
    primary = [x[:, 0] for x in ae[PRIMARY]]
    rho = rho_interval([(x - amplitude) ** 2 for x in primary], amplitude, point_seed(27, 3))
    allowed = len(sets) / sum(1 / max(s["gamma"]["largest_information_per_query_allowed"], 1e-12) for s in sets)
    return {"criterion": criterion(primary, amplitude),
            "rho": {"status": "descriptive, no criterion: an interval of a mean squared error cannot exclude rare large "
                              "errors that the repetitions did not contain",
                    "method": "rho = MSE_AE / (a (1 - a) / Q) over the repetitions of AE with the primary schedule; two-sided "
                              f"95% BCa bootstrap interval (scipy.stats.bootstrap, {RHO_RESAMPLES} resamples, the "
                              "repetitions resampled within their set)",
                    **rho, "per_set": [s["rho"] for s in sets]},
            "true_amplitude": amplitude, "ideal_unamplified_rmse": math.sqrt(amplitude * (1 - amplitude) / plan["queries"]),
            "pooled": {**{key: schedules[0][key] for key in COMPARED},
                       "mse_ratio_ae_over_direct_note": "descriptive; it does not isolate amplification: the direct estimator "
                                                        "keeps the state-preparation bias (1 - c)(1/2 - a)",
                       "oracle_contrast": compare(oracle, direct, amplitude, resamples["errors"], point_seed(27, 5, 99), False),
                       "gamma": {**gamma_statement(allowed, rho),
                                 "note": "the largest allowed value is the harmonic mean over the sets"},
                       "amplification_can_pay_sets": sum(s["probe"]["amplification_can_pay"] for s in sets)},
            "schedules": schedules, "sets": sets, "round_2b": round_2b(),
            "resamples": {**resamples, "rho": RHO_RESAMPLES}}


def round_2b() -> dict:
    """Round 2B as stored (E15, the screened 3-qubit sets): contrast per iterate, w_0 and the largest Gamma_k."""
    stored = json.loads((RESULTS / "e15_mlae_screened_qubits.json").read_text())
    sets = []
    for entry in stored["sets"]:
        if entry["n_price_qubits"] == N_PRICE:
            gamma = [(2 * r["k"] + 1) * r["signal_fraction"] ** 2 for r in entry["rows"]]
            sets.append({"qubits": entry["initial_layout"], "f_Q": entry["contrast_fit"]["f_Q"],
                         "w_0": entry["rows"][0]["signal_fraction"], "max_gamma": max(gamma),
                         "max_gamma_at_k": entry["rows"][int(np.argmax(gamma))]["k"]})
    return {"device": stored["backend"], "job_id": stored["run"]["job_id"], "sets": sets}


def errors_text(row: dict) -> str:
    bias, rmse = row["bias"], row["rmse"]
    return (f"bias {bias['value']:+.4f} [{bias['ci95'][0]:+.4f}, {bias['ci95'][1]:+.4f}], sd {row['sd']['value']:.4f}, "
            f"RMSE {rmse['value']:.4f} [{rmse['ci95'][0]:.4f}, {rmse['ci95'][1]:.4f}]")


def ratio_text(row: dict) -> str:
    if row["ci95"] is None:
        return f"{row['value']:.3f} [interval not computed]"
    return f"{row['value']:.3f} [{row['ci95'][0]:.3f}, {row['ci95'][1]:.3f}]"


def report(label: str, facts: dict, result: dict) -> None:
    plan, found, rho, pooled = facts["plan"], result["criterion"], result["rho"], result["pooled"]
    print(f"{label}: {len(result['sets'])} sets; per set {plan['primary_repetitions']} repetitions of AE with the primary "
          f"schedule and {plan['repetitions']} of the direct estimator and of every other schedule; {plan['queries']:,} oracle "
          f"queries per repetition and estimator; true amplitude {result['true_amplitude']:.5f}, ideal unamplified sampling "
          f"RMSE {result['ideal_unamplified_rmse']:.5f}")
    print(f"  criterion: {found['hits']} of {found['repetitions']} AE estimates within t = {found['tolerance']:.5f} of the "
          "amplitude (per set " + ", ".join(map(str, found["hits_per_set"])) + f"), hit rate {found['hit_rate']:.4f} "
          f"[{found['hit_rate_ci95'][0]:.4f}, {found['hit_rate_ci95'][1]:.4f}]; ideal unamplified sampling: q0 = "
          f"{found['q0']:.5f} (x = {found['q0_range'][0]} to {found['q0_range'][1]} of {plan['queries']:,} shots); one-sided "
          f"exact binomial test p = {found['p']:.3g}, alpha = {found['alpha']}; {found['verdict']}")
    print(f"  descriptive, no criterion: rho = mean squared error of AE / (a (1 - a) / Q) = {ratio_text(rho)} (per set "
          + ", ".join(ratio_text(x) for x in rho["per_set"]) + "; 95% BCa intervals), information per query realised "
          f"{rho['information_per_query_realised']:.2f}. An interval of a mean squared error cannot exclude rare large "
          "errors that the repetitions did not contain")
    for name, row in [("pooled", pooled)] + [(f"set {s['set']} {s['layout']}", s) for s in result["sets"]]:
        print(f"  {name}: direct {errors_text(row['direct'])}")
        print(f"  {' ' * len(name)}  AE     {errors_text(row['ae'])}")
        print(f"  {' ' * len(name)}  mean squared error AE / direct {ratio_text(row['mse_ratio_ae_over_direct'])} (does not "
              "isolate amplification: the direct estimator keeps the state-preparation bias); oracle-contrast AE RMSE "
              f"{row['oracle_contrast']['ae']['rmse']['value']:.4f}")
    print("  schedules, pooled (largest power: shots per power, queries, repetitions per set; AE errors; mean squared error "
          "AE / direct; estimates within t):")
    for row in result["schedules"]:
        hit = row["within_tolerance"]
        print(f"    m = {row['largest_power']}: {row['shots_per_power']} shots, {row['queries']:,} queries, "
              f"{row['repetitions_per_set']} repetitions; {errors_text(row['ae'])}; "
              f"{ratio_text(row['mse_ratio_ae_over_direct'])}; {hit['hits']} of {hit['repetitions']} within t, rate "
              f"{hit['hit_rate']:.3f} [{hit['hit_rate_ci95'][0]:.3f}, {hit['hit_rate_ci95'][1]:.3f}]")
    for s in result["sets"]:
        depth, fit, readout = s["probe"], s["probe"]["contrast_fit_objective_qubit"], s["readout"]
        print(f"  probe, set {s['set']}: w_k " + " ".join(f"{r['signal_fraction']:.3f}" for r in depth["rows"])
              + f"; f_Q {fit['f_Q']:.3f} [{fit['f_Q_ci95'][0]:.3f}, {fit['f_Q_ci95'][1]:.3f}]; with readout terms c "
              f"{depth['contrast_with_readout']['c']:.3f}, f {depth['contrast_with_readout']['f']:.3f}; largest Gamma "
              f"{depth['max_gamma']:.2f} at k = {depth['max_gamma_at_k']}; readout errors of the objective qubit at k = 0: "
              f"{readout['applied_p_read1_given0'][0]:.4f}, {readout['applied_p_read0_given1'][0]:.4f}; "
              f"above a screen threshold in this job: {readout['above_a_screen_threshold'] or 'none'}")
        print(f"    Gamma allows the primary schedule {s['gamma']['largest_information_per_query_allowed']:.2f} per query, AE "
              f"realised {ratio_text(s['gamma']['information_per_query_realised'])}: {s['gamma']['statement']}")
    gamma = pooled["gamma"]
    print(f"  Gamma, pooled: allows {gamma['largest_information_per_query_allowed']:.2f}, AE realised "
          f"{ratio_text(gamma['information_per_query_realised'])} (ideal unamplified sampling = 1): {gamma['statement']}",
          flush=True)


# --------------------------------------------------------------------------- --plan


def rate_text(met: int, n: int) -> str:
    low, high = exact_interval(met, n)
    return f"{met / n:.4f} [{low:.4f}, {high:.4f}] ({met} of {n} simulated jobs)"


def null_job(task) -> tuple:
    """Decision 8, size: one simulated job with ideal unamplified sampling in the place of AE: (hits, criterion met)."""
    amplitude, sets, primary, seed = task
    drawn = np.random.default_rng(seed).binomial(QUERIES, amplitude, (sets, primary))
    found = criterion(list(drawn / QUERIES), amplitude)
    return found["hits"], found["confirmed"]


def simulated_job(task) -> tuple:
    """Decision 8: one simulated main job under the model of decision 4.

    Returns (errors of the direct estimator, sets x R2; errors of AE, sets x R1; hit rate; rho; criterion met).
    """
    amplitude, c, f, sets, primary, repetitions, seed = task
    rng = np.random.default_rng(seed)
    powers, shots = np.arange(PRIMARY + 1.0), QUERIES // (PRIMARY + 1) ** 2
    truth = model_p(math.asin(math.sqrt(amplitude)), c, f, powers, *PLAN_READOUT)
    direct, ae = [], []
    for _ in range(sets):
        p10, p01 = (rng.binomial(CAL_SHOTS, p) / CAL_SHOTS for p in PLAN_READOUT)
        direct.append(direct_estimate(rng.binomial(QUERIES, truth[0], repetitions), QUERIES, p10, p01))
        ae.append(ae_fit(powers, rng.binomial(shots, truth, (primary, len(powers))), shots, p10, p01)[:, 0])
    found = criterion(ae, amplitude)
    errors = np.array(ae) - amplitude
    rho = float((errors ** 2).mean() / (amplitude * (1 - amplitude) / QUERIES))
    return np.array(direct) - amplitude, errors, found["hit_rate"], rho, found["confirmed"]


def plan(sets: int, primary: int, repetitions: int) -> None:
    began = time.time()
    register, basis, shots = e15.build_register(N_PRICE, [0, 1]), hj.usage_basis(), shot_plan(primary, repetitions)
    amplitude = register["amplitude"]
    ideal = math.sqrt(amplitude * (1 - amplitude) / QUERIES)
    print("E27, amplitude estimation against ideal unamplified sampling at equal oracle queries: two jobs per device")
    print(f"  screen: 2 pubs (every qubit in |0>, every qubit in |1>), {SCREEN_SHOTS:,} shots each, "
          f"{hj.qpu_seconds([(None, None, SCREEN_SHOTS)] * 2, basis)['seconds']:.1f} QPU seconds estimated")
    print(f"  main: {sets} sets x ({len(POWERS)} powers + 2 calibration circuits) = {sets * (len(POWERS) + 2)} pubs; per set "
          f"{primary} repetitions of AE with the primary schedule (m = {PRIMARY}) and {repetitions} of the direct estimator "
          f"and of every other schedule; {QUERIES:,} oracle queries per repetition and estimator")
    names = ["direct"] + [f"m={s['largest_power']}" for s in shots["schedules"]]
    print("  shots per set (per repetition: the block of each estimator; then the shots of the run):")
    print("    k  queries/shot " + " ".join(f"{name:>7}" for name in names) + "   needed     run")
    for k in POWERS:
        blocks = [QUERIES if k == 0 else 0] + [s["shots_per_power"] if k <= s["largest_power"] else 0 for s in shots["schedules"]]
        print(f"    {k}  {2 * k + 1:>12} " + " ".join(f"{b:>7,}" for b in blocks)
              + f" {shots['needed'][k]:>8,} {shots['shots'][k]:>7,}")
    print(f"    repetitions per set  {repetitions:>7,} " + " ".join(f"{s['repetitions']:>7,}" for s in shots["schedules"]))
    print("    queries per repetition: " + ", ".join(f"m={s['largest_power']} {s['queries']:,}" for s in shots["schedules"])
          + f"; a run has at least {PROBE_SHOTS:,} shots (probe)")
    per_set = sum(shots["shots"]) + 2 * CAL_SHOTS
    estimate = hj.qpu_seconds(([(None, None, n) for n in shots["shots"]] + [(None, None, CAL_SHOTS)] * 2) * sets, basis)
    print(f"  per set {sum(shots['shots']):,} shots and 2 x {CAL_SHOTS:,} calibration shots = {per_set:,}; {sets} sets: "
          f"{estimate['pubs']} pubs, {estimate['shots']:,} shots, largest run {max(shots['shots']):,} shots; at most "
          f"{MAX_JOB_SECONDS} s of execution allowed, a job estimated above {MAX_JOB_SECONDS / 1.5:.0f} s is not submitted")
    hj.print_estimate(estimate, basis)
    old = round_2b()
    print(f"Round 2B on {old['device']} (job {old['job_id']}), screened 3-qubit sets: f_Q "
          + ", ".join(f"{s['f_Q']:.3f}" for s in old["sets"]) + "; w_0 " + ", ".join(f"{s['w_0']:.2f}" for s in old["sets"])
          + "; largest Gamma_k " + ", ".join(f"{s['max_gamma']:.1f} at k = {s['max_gamma_at_k']}" for s in old["sets"]))
    facts, n = tolerance_facts(amplitude), sets * primary
    critical = critical_hits(n, facts["q0"])

    def power_at(q: float) -> float:
        return 0.0 if critical is None else float(binom.sf(critical - 1, n, q))

    print(f"Criterion (decision 5): tolerance t = sqrt(a (1 - a) / Q) = {facts['tolerance']:.5f}; q0 = P(|X / Q - a| <= t) for "
          f"X ~ Binomial({QUERIES:,}, {amplitude:.5f}) = {facts['q0']:.5f} (x = {facts['q0_range'][0]} to "
          f"{facts['q0_range'][1]}); {sets} x {primary} = {n} repetitions of AE with the primary schedule")
    print(f"  critical number of hits (the smallest with p <= {ALPHA}): "
          + ("none, no number of hits is enough" if critical is None else
             f"{critical} of {n} (hit rate {critical / n:.4f}); exact size of the test, P(Binomial({n}, q0) >= {critical}) = "
             f"{power_at(facts['q0']):.5f}"))
    print("  power, analytic, for normally distributed AE errors without bias: hit probability 2 Phi(1 / sqrt(rho)) - 1 and "
          f"the exact binomial power at {n} repetitions")
    for rho in PLAN_RHO:
        q = 2 * float(norm.cdf(1 / math.sqrt(rho))) - 1
        print(f"    rho = {rho}: hit probability {q:.4f}, power {power_at(q):.4f}")
    started = time.time()
    with ProcessPoolExecutor(max_workers=pool_workers()) as pool:
        found = list(pool.map(null_job, [(amplitude, sets, primary, point_seed(27, 10, job))
                                         for job in range(PLAN_NULL_EXPERIMENTS)], chunksize=50))
        print("  size, by simulation (a check of the code only: the test is exact): ideal unamplified sampling in the place of "
              f"AE, {n} estimates Binomial(Q, a) / Q per simulated job, seeds point_seed(27, 10, job); mean hit rate "
              f"{np.mean([hits for hits, _ in found]) / n:.4f}; rate at which the criterion is met (alpha = {ALPHA}): "
              + rate_text(sum(met for _, met in found), len(found)), flush=True)
        print(f"  power, Monte Carlo under the model of decision 4: {PLAN_EXPERIMENTS} simulated jobs per assumption, each "
              f"{sets} sets x {primary} repetitions of AE with the primary schedule and {repetitions} of the direct estimator; "
              f"readout errors {PLAN_READOUT[0]} and {PLAN_READOUT[1]}, estimated per set from {CAL_SHOTS:,} shots each; ideal "
              f"unamplified sampling RMSE {ideal:.5f}; seeds point_seed(27, 9, assumption, job)")
        for index, (name, c, f) in enumerate(SCENARIOS):
            tasks = [(amplitude, c, f, sets, primary, repetitions, point_seed(27, 9, index, job))
                     for job in range(PLAN_EXPERIMENTS)]
            found = list(pool.map(simulated_job, tasks))
            direct, ae = np.concatenate([row[0] for row in found]), np.concatenate([row[1] for row in found])
            allowed = sum((2 * k + 1) ** 2 * (c * f ** k) ** 2 for k in range(PRIMARY + 1)) / (PRIMARY + 1) ** 2
            print(f"  {name}, c = {c:.2f}, f = {f:.2f}: Gamma allows the primary schedule {allowed:.2f} per query, an RMSE of "
                  f"{ideal / math.sqrt(allowed):.4f} at best")
            for label, e in (("direct", direct), ("AE    ", ae)):
                print(f"    {label} bias {e.mean():+.4f}, sd {e.std(ddof=1):.4f}, RMSE {math.sqrt((e ** 2).mean()):.4f}")
            print(f"    mean hit rate {np.mean([row[2] for row in found]):.4f}; mean rho {np.mean([row[3] for row in found]):.3f}; "
                  f"share of the simulated jobs in which the criterion is met: "
                  + rate_text(sum(row[4] for row in found), len(found)), flush=True)
    print(f"  wall time: the simulations {time.time() - started:.0f} s, the whole plan {time.time() - began:.0f} s", flush=True)


# --------------------------------------------------------------------------- --dry-run


def compacted(circuit):
    """The same instructions on the qubits the circuit uses, for the simulator; classical bits are unchanged."""
    from qiskit import QuantumCircuit

    to = {q: i for i, q in enumerate(sorted({circuit.find_bit(q).index for inst in circuit.data for q in inst.qubits}))}
    out = QuantumCircuit(len(to), global_phase=circuit.global_phase)
    for creg in circuit.cregs:
        out.add_register(creg)
    for inst in circuit.data:
        out.append(inst.operation, [to[circuit.find_bit(q).index] for q in inst.qubits],
                   [circuit.find_bit(c).index for c in inst.clbits])
    return out


def as_counts(bits: np.ndarray) -> dict:
    """Counts of the rows of a 0/1 matrix; column i is classical bit i, the last character of the bitstring."""
    rows, numbers = np.unique(bits, axis=0, return_counts=True)
    return {"".join(map(str, row[::-1])): int(n) for row, n in zip(rows, numbers)}


def misread(counts: dict, qubits: list, p10: np.ndarray, p01: np.ndarray, seed: int) -> dict:
    """Decision 9: readout errors on simulated counts; bit i is read from qubits[i] and flips with that qubit's error."""
    keys = sorted(counts)
    bits = np.repeat(np.array([[int(ch) for ch in key[::-1]] for key in keys], dtype=np.int8),
                     [counts[key] for key in keys], axis=0)
    flips = np.random.default_rng(seed).random(bits.shape) < np.where(bits == 1, p01[qubits], p10[qubits])
    return as_counts(bits ^ flips)


def dry_run(args) -> None:
    started = time.time()
    model = hj.device_models()[0]
    sizes = QUICK if args.quick else {"sets": SETS, "primary_repetitions": PRIMARY_REPETITIONS, "repetitions": REPETITIONS}
    sets, primary, repetitions = (getattr(args, key) or sizes[key] for key in ("sets", "primary_repetitions", "repetitions"))
    rng = np.random.default_rng(point_seed(27, 7))
    p10, p01 = (np.minimum(median * np.exp(spread * rng.standard_normal(model.num_qubits)), 0.4)
                for median, spread in DRY_READOUT)
    every, _ = screen_circuits(model)
    drawn = rng.random((2, SCREEN_SHOTS, len(every)))
    screen = screen_summary([[as_counts((drawn[0] < p10[every]).astype(np.int8))],
                             [as_counts((drawn[1] >= p01[every]).astype(np.int8))]], every, model.num_qubits, SCREEN_SHOTS)
    register = e15.build_register(N_PRICE, list(POWERS))
    job = build_job(model, screen, register, sets, primary, repetitions)
    describe(f"dry run on the packaged model {model.name}, synthetic screen", screen, job)
    counts = hj.run_aer([(compacted(circuit), None, shots) for circuit, _, shots in job["pubs"]], 27, args.noise)
    counts = [[misread(pub[0], e14.measured_physical(circuit), p10, p01, point_seed(27, 8, index))]
              for index, (pub, (circuit, _, _)) in enumerate(zip(counts, job["pubs"]))]
    result = analyse_counts(register, job["facts"], counts, QUICK["resamples"] if args.quick else RESAMPLES)
    execution = (f"Aer on the circuits placed on {model.name}, two-qubit depolarising error {args.noise or 0}, synthetic "
                 "readout errors on every measured bit")
    report(f"dry run ({execution}; these counts carry no information about hardware)", job["facts"], result)
    hj.save_result(hj.dry_run_name(STEM, args), {
        "design": DESIGN, "execution": execution,
        "note": "dry run: a synthetic screen and simulated counts; they validate the pipeline and the analysis and carry "
                "no information about hardware",
        "screen": {"job_id": None, "rejected_qubits": len(screen["rejected_qubits"]), "screened_qubits": len(every)},
        **{key: job["facts"][key] for key in FACT_KEYS if key != "sets"}, "placement": job["facts"]["sets"], **result,
        "wall_s": round(time.time() - started, 1)})


# --------------------------------------------------------------------------- the device


def epoch(stamp: str) -> int:
    """Seconds since 1970 of a UTC time as the manifest writes it; a fraction of a second is dropped."""
    return calendar.timegm(time.strptime(stamp[:19], "%Y-%m-%dT%H:%M:%S"))


def screen_time(entry: dict) -> tuple:
    """Decision 2: (the time from which the age of a screen is counted, whether it is the finish time of its job)."""
    finished = (entry.get("timestamps") or {}).get("finished")
    return (finished, True) if finished else (entry["submitted_utc"], False)


def screen_age(entry: dict) -> int:
    return max(0, int(time.time()) - epoch(screen_time(entry)[0]))


def screens_of(device: str) -> list:
    """The screen jobs of a device in the manifest, in the order of their submission."""
    return sorted((entry for entry in hj.load_manifest(EXPERIMENT)["jobs"]
                   if entry.get("kind") == "screen" and entry["device"] == device), key=lambda entry: entry["submitted_utc"])


def last_finished(screens: list) -> dict | None:
    """Decision 2: the collected screen that finished last; None if no screen is collected."""
    collected = [entry for entry in screens if entry.get("archive")]
    return max(collected, key=lambda entry: epoch(screen_time(entry)[0])) if collected else None


def screen_text(entry: dict) -> str:
    stamp, finished = screen_time(entry)
    age = screen_age(entry)
    state = f"is {entry['state']}, {'collected' if entry.get('archive') else 'not collected'}"
    if finished:
        return (f"screen job {entry['job_id']} of {entry['device']} was submitted {entry['submitted_utc']}, finished {stamp}, "
                f"{age // 60} min {age % 60} s ago, and {state}")
    return (f"screen job {entry['job_id']} of {entry['device']} was submitted {entry['submitted_utc']}, {age // 60} min "
            f"{age % 60} s ago, and {state}" + ("; its entry has no finish time, so the age is counted from the submission, "
                                                "which overstates it" if entry.get("archive") else ""))


def usable_screen(device: str) -> dict:
    """Decision 2: the collected screen of the device that finished last, if it finished at most 30 minutes ago. Manifest only."""
    screens = screens_of(device)
    entry = last_finished(screens)
    if not screens:
        raise SystemExit(f"not submitted: no screen of {device} is listed. Run --screen, then --collect")
    if entry is None:
        raise SystemExit(f"not submitted: {screen_text(screens[-1])}. Run --collect once it is DONE, or --screen again")
    if screen_age(entry) > MAX_SCREEN_AGE_S:
        later = "" if screens[-1] is entry else f"; {screen_text(screens[-1])}"
        raise SystemExit(f"not submitted: {screen_text(entry)}; a screen may have finished at most {MAX_SCREEN_AGE_S // 60} "
                         f"minutes before the main job is submitted{later}. Run --screen again, then --collect")
    return entry


def named_device() -> str:
    name = os.environ.get("QFPO_IBM_BACKEND")
    if not name:
        raise SystemExit("not submitted: set QFPO_IBM_BACKEND to the name of the device")
    return name


def open_device(name: str) -> tuple:
    opened, backend = hj.device()
    if backend.name != name:
        raise SystemExit(f"not submitted: asked for {name}, the service returned {backend.name}")
    return opened, backend


def submit_screen() -> None:
    name = named_device()
    hj.e20.require_committed_code()
    opened, backend = open_device(name)
    every, circuits = screen_circuits(backend)
    facts = {"kind": "screen", "shots": SCREEN_SHOTS, "num_qubits": backend.num_qubits, "screened_qubits": every}
    hj.submit(EXPERIMENT, __file__, opened, backend, [(c, None, SCREEN_SHOTS) for c in circuits], MAX_JOB_SECONDS, facts,
              again=True)
    print(f"next: --collect once the screen has finished, then --submit within {MAX_SCREEN_AGE_S // 60} minutes of the "
          "screen's finish")


def stored_screen(entry: dict) -> dict:
    """The summary of a collected screen job, from its archived counts."""
    record = hj.read_archive(hj.OUT / "raw" / entry["archive"])
    if record["shots"] != [[entry["shots"]], [entry["shots"]]]:
        raise SystemExit(f"screen job {entry['job_id']} not used: shots per pub {record['shots']}, requested {entry['shots']}")
    return screen_summary(record["counts"], entry["screened_qubits"], entry["num_qubits"], entry["shots"])


def submit(args) -> None:
    name = named_device()
    hj.refuse_repeat(EXPERIMENT, name, args.again, "main")
    usable_screen(name)
    hj.e20.require_committed_code()
    opened, backend = open_device(name)
    hj.refuse_unfinished(EXPERIMENT, opened)
    entry = usable_screen(name)
    screen = stored_screen(entry)
    if backend.num_qubits != screen["num_qubits"]:
        raise SystemExit(f"not submitted: the screen has {screen['num_qubits']} qubits, {name} has {backend.num_qubits}")
    job = build_job(backend, screen, e15.build_register(N_PRICE, list(POWERS)), args.sets or SETS,
                    args.primary_repetitions or PRIMARY_REPETITIONS, args.repetitions or REPETITIONS)
    describe(f"E27 on {name}, screen job {entry['job_id']}", screen, job)
    try:
        limit = backend.configuration().max_shots
    except Exception:  # noqa: BLE001 - a device that states no limit is not limited here
        limit = None
    if limit and max(job["facts"]["plan"]["shots"]) > limit:
        raise SystemExit(f"not submitted: the run at k = 0 has {max(job['facts']['plan']['shots']):,} shots, but {name} "
                         f"accepts at most {limit:,} per circuit; reduce --primary-repetitions or --repetitions")
    estimate = hj.qpu_seconds(job["pubs"])
    if estimate["seconds"] > MAX_JOB_SECONDS / 1.5:
        raise SystemExit(f"not submitted: {estimate['seconds']:.0f} QPU seconds estimated, and a job is limited to "
                         f"{MAX_JOB_SECONDS} s of execution; reduce --sets, --primary-repetitions or --repetitions")
    usable_screen(name)                          # the placement took time: the age is checked again
    stamp, finished = screen_time(entry)
    print(f"screen accepted (it may have finished at most {MAX_SCREEN_AGE_S // 60} minutes ago): {screen_text(entry)}", flush=True)
    facts = {"kind": "main", **job["facts"],
             "screen": {"job_id": entry["job_id"], "submitted_utc": entry["submitted_utc"],
                        "finished_utc": stamp if finished else None, "archive": entry["archive"],
                        "age_s_at_submission": screen_age(entry),
                        "age_counted_from": "the finish of the screen job" if finished else
                        "the submission of the screen job, which overstates the age: its entry has no finish time",
                        "rejected_qubits": len(screen["rejected_qubits"]), "screened_qubits": len(screen["screened_qubits"])}}
    hj.submit(EXPERIMENT, __file__, opened, backend, job["pubs"], MAX_JOB_SECONDS, facts, again=args.again)


def recorded_times(entry: dict) -> dict:
    """Decision 2: the times of the screen and of the main job, and the age of the screen as the main job was sent and ran."""
    screen, stamps = entry["screen"], entry.get("timestamps") or {}
    since = screen["finished_utc"] or screen["submitted_utc"]

    def minutes(stamp):
        return None if not stamp else round((epoch(stamp) - epoch(since)) / 60, 1)

    return {"screen_finished_utc": screen["finished_utc"], "screen_age_counted_from": screen["age_counted_from"],
            "main_submitted_utc": entry["submitted_utc"], "main_execution_start_utc": stamps.get("execution_start"),
            "main_execution_stop_utc": stamps.get("execution_stop"),
            "screen_age_min_at_main_submission": minutes(entry["submitted_utc"]),
            "screen_age_min_at_main_execution_start": minutes(stamps.get("execution_start")),
            "note": "descriptive; the age of the screen at the execution of the main job is not controlled by the script"}


def analyse() -> None:
    todo = hj.to_analyse(EXPERIMENT, STEM)
    if not todo:
        print("E27: no archived job is without a result")
        return
    hj.e20.require_committed_code()
    register = e15.build_register(N_PRICE, list(POWERS))
    for entry, record in todo:
        started = time.time()
        if entry["kind"] == "screen":
            screen = stored_screen(entry)
            print(f"{entry['device']}, screen job {entry['job_id']}: {len(screen['rejected_qubits'])} of "
                  f"{screen['num_qubits']} qubits rejected; median P(read 1 | 0) {screen['median_p_read1_given0']:.4f}, "
                  f"P(read 0 | 1) {screen['median_p_read0_given1']:.4f}")
            hj.save_result(f"{STEM}_{entry['tag']}", {"kind": "screen", "design": DESIGN, "execution": entry["device"],
                                                      **screen, **hj.job_summary(EXPERIMENT, entry, __file__)})
            continue
        requested = [[n] for _ in entry["sets"] for n in entry["plan"]["shots"] + [entry["calibration_shots"]] * 2]
        if record["shots"] != requested:
            raise SystemExit(f"job {entry['job_id']} not analysed: shots per pub {record['shots']}, requested {requested}")
        if abs(entry["amplitude"] - register["amplitude"]) > 1e-12 or entry["powers"] != list(POWERS):
            raise SystemExit(f"job {entry['job_id']} not analysed: the register of e15.build_register() is not the one submitted")
        result = analyse_counts(register, entry, record["counts"])
        report(f"{entry['device']}, job {entry['job_id']}", entry, result)
        times = recorded_times(entry)
        ran = times["screen_age_min_at_main_execution_start"]
        print(f"  times (descriptive): screen finished {times['screen_finished_utc'] or 'not recorded'}; main job submitted "
              f"{times['main_submitted_utc']}, executed from {times['main_execution_start_utc'] or 'not recorded'} to "
              f"{times['main_execution_stop_utc'] or 'not recorded'}; age of the screen, counted from "
              f"{times['screen_age_counted_from']}: {times['screen_age_min_at_main_submission']} min at the submission of the "
              f"main job, {'not known' if ran is None else f'{ran} min'} at the start of its execution", flush=True)
        hj.save_result(f"{STEM}_{entry['tag']}", {
            "kind": "main", "design": DESIGN, "execution": entry["device"], "screen": entry["screen"], "times": times,
            **{key: entry[key] for key in FACT_KEYS if key != "sets"}, "placement": entry["sets"], **result,
            **hj.job_summary(EXPERIMENT, entry, __file__), "wall_s": round(time.time() - started, 1)})


def main(argv=None) -> None:
    p = hj.parser(__doc__, {"screen": "submit the readout screen and record it in the manifest"})
    p.add_argument("--sets", type=int, metavar="S", help=f"with --plan, --dry-run and --submit: qubit sets (default {SETS})")
    p.add_argument("--primary-repetitions", type=int, metavar="R1",
                   help="with --plan, --dry-run and --submit: repetitions per set of AE with the primary schedule "
                        f"(default {PRIMARY_REPETITIONS})")
    p.add_argument("--repetitions", type=int, metavar="R2",
                   help="with --plan, --dry-run and --submit: repetitions per set of the direct estimator and of every "
                        f"other schedule (default {REPETITIONS})")
    args = hj.parse(p, argv)
    sizes = (args.sets, args.primary_repetitions, args.repetitions)
    if any(x is not None for x in sizes) and not (args.plan or args.dry_run or args.submit):
        p.error("--sets, --primary-repetitions and --repetitions belong to --plan, --dry-run and --submit")
    if (args.sets is not None and args.sets < 1) or any(x is not None and x < 2 for x in sizes[1:]):
        p.error("--sets: at least 1; --primary-repetitions and --repetitions: at least 2")
    if args.plan:
        plan(args.sets or SETS, args.primary_repetitions or PRIMARY_REPETITIONS, args.repetitions or REPETITIONS)
    elif args.dry_run:
        dry_run(args)
    elif args.status:
        opened, backend = hj.device()
        hj.status(EXPERIMENT, opened, backend)
        screens = screens_of(backend.name)
        newest = last_finished(screens)
        if newest is None:
            print(f"no screen of {backend.name} is " + (f"collected; {screen_text(screens[-1])}" if screens else "listed"))
        else:
            print(f"the collected screen that finished last: {screen_text(newest)}; --submit accepts it up to "
                  f"{MAX_SCREEN_AGE_S // 60} minutes after its finish: "
                  + ("within the limit" if screen_age(newest) <= MAX_SCREEN_AGE_S else "too old"))
            if screens[-1] is not newest:
                print(f"the screen submitted last: {screen_text(screens[-1])}")
    elif args.screen:
        submit_screen()
    elif args.submit:
        submit(args)
    elif args.collect:
        if not hj.load_manifest(EXPERIMENT)["jobs"]:
            raise SystemExit("E27: no job has been submitted")
        hj.collect(EXPERIMENT, hj.service())
        analyse()
    else:
        analyse()


if __name__ == "__main__":
    main()
