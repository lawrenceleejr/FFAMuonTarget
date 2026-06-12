#!/bin/bash
# Scan proton kinetic energy for pi- production on a graphite target.
# Runs target_yield.g4bl for each energy (PARALLEL jobs at a time, each in
# its own working directory), collects the detector files into
# out/scan/KE<energy>/, then prints the summary table.
#
# Usage: scripts/run_energy_scan.sh [NEV] [THICK_mm]
#   ENERGIES="800 1000 ..." scripts/run_energy_scan.sh 5000 20
set -e
cd "$(dirname "$0")/.."

NEV=${1:-2000}
THICK=${2:-20}
PARALLEL=${PARALLEL:-4}
ENERGIES=${ENERGIES:-"300 500 800 1000 1500 2000 3000 5000"}

run_point() {
  KE=$1
  d=runs/scan_KE$KE
  mkdir -p $d
  rm -f $d/VD*.txt
  ( cd $d && ../../scripts/g4bl-docker.sh ../../target_yield.g4bl \
        KE=$KE THICK=$THICK NEV=$NEV > g4bl.log 2>&1 )
  o=out/scan/KE$KE
  mkdir -p $o
  mv $d/VD*.txt $o/ 2>/dev/null || { echo "KE=$KE produced no output:"; tail -5 $d/g4bl.log; }
  echo "=== done KE=$KE"
}

n=0
for KE in $ENERGIES; do
  echo "=== launch KE=$KE MeV  THICK=$THICK mm  NEV=$NEV"
  run_point $KE &
  n=$((n+1))
  if [ $((n % PARALLEL)) -eq 0 ]; then wait; fi
done
wait

python3 scripts/analyze_yield.py --nev $NEV --thick $THICK
