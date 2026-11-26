#!/usr/bin/env bash
# ./run.sh <experiment> [args]   one experiment, e.g. ./run.sh e21_qaoa_fresh   or   ./run.sh e22_hardware_portfolio_qaoa --dry-run
# ./run.sh all                   every simulator experiment
# ./run.sh analyses              the scripts that only read stored results
# Hardware scripts are never part of "all": name them, with IBM_QUANTUM_TOKEN set.
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="$PWD/src:$PWD/paper/experiments" OTEL_SDK_DISABLED=true
PY="${PYTHON:-python3}"

ALL=(e1_montecarlo e2_facility e3_rydberg e4_chemistry e5_fleet e10_qaoa_exact e11_rydberg_tuned_control
     e12_sqd_selected_ci e20_portfolio_corrected e21_qaoa_fresh e23_qaoa_product_data
     e25_qaoa_initial_angles e31_mission_allocation e32_mission_equal_time)
ANALYSES=(e13_hardware_statistics e16_hardware_summary e17_derived_statistics)

if [[ $# -eq 0 ]]; then
  echo "usage: $0 <experiment> [args] | all | analyses"; exit 2
fi

run() {
  echo "=== $1 $(date +%T)"
  "$PY" "paper/experiments/${1%.py}.py" "${@:2}"
}

case "$1" in
  all)      for e in "${ALL[@]}"; do run "$e"; done ;;
  analyses) for e in "${ANALYSES[@]}"; do run "$e"; done ;;
  *)        run "$@" ;;
esac
