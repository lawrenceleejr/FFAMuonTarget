#!/bin/bash
# Scan proton kinetic energy for pi- production on a graphite target.
# Runs target_yield.g4bl at each energy, moves the detector files into
# out/scan/KE<energy>/, then prints the summary table.
#
# Usage: scripts/run_energy_scan.sh [NEV] [THICK_mm]
set -e
cd "$(dirname "$0")/.."

NEV=${1:-2000}
THICK=${2:-20}
ENERGIES=${ENERGIES:-"300 500 800 1000 1500 2000 3000 5000"}

for KE in $ENERGIES; do
  echo "=== KE=$KE MeV  THICK=$THICK mm  NEV=$NEV ==="
  ./scripts/g4bl-docker.sh target_yield.g4bl KE=$KE THICK=$THICK NEV=$NEV \
      > out/ty_KE$KE.log 2>&1 || { tail -20 out/ty_KE$KE.log; exit 1; }
  # the virtualdetector placements write VDfwd.txt, VDbwd.txt, VDxp.txt,
  # VDxm.txt, VDyp.txt, VDym.txt in the working directory
  d=out/scan/KE$KE
  mkdir -p $d
  mv VD*.txt $d/ 2>/dev/null || true
done

python3 scripts/analyze_yield.py --nev $NEV --thick $THICK
