#!/usr/bin/env bash
# ./run.sh <experiment>...   e.g. ./run.sh e2_facility e6_qaoa_control
# ./run.sh all               every experiment; IBM scripts only if IBM_QUANTUM_TOKEN is set
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="$PWD/src:$PWD/paper/experiments" OTEL_SDK_DISABLED=true
PY="${PYTHON:-python3}"

ALL=(e1_montecarlo e2_facility e3_rydberg e4_chemistry e5_fleet e6_qaoa_control e7_chemistry_controls)
HW=(e8_ibm_hardware e8b_ibm_transpiled_cost e9_sqd_noise_control)

if [[ $# -eq 0 ]]; then
  echo "usage: $0 <experiment>... | all"; exit 2
fi
if [[ "$1" == "all" ]]; then
  set -- "${ALL[@]}"
  if [[ -n "${IBM_QUANTUM_TOKEN:-}" ]]; then set -- "$@" "${HW[@]}"; fi
fi

for e in "$@"; do
  echo "=== $e $(date +%T)"
  "$PY" "paper/experiments/${e%.py}.py"
done
