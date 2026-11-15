"""Monte Carlo simulation vertical (finance): European option pricing by
classical Monte Carlo and by quantum amplitude estimation.

Module layout:

* ``blackscholes`` - exact closed-form price and Greeks, the ground truth
  every run is scored against.
* ``grid``         - discretization of the terminal-price distribution,
  and the resulting bias measured separately from estimator error.
* ``classical``    - Monte Carlo estimator with antithetic variates.
* ``qae``          - native maximum-likelihood amplitude estimation.
* ``comparison``   - accuracy-matched comparison of the two estimators.
* ``service``      - orchestration used by ``api/montecarlo_endpoints.py``;
  one estimator per run, one run per job.
"""
