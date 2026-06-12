#!/bin/bash
# Run ffa_ring.g4bl in NJOBS parallel docker containers (different seeds),
# each in its own working directory runs/jobN/, then analyze them together:
#   scripts/run_ring_parallel.sh [NJOBS] [NEV_PER_JOB] [extra g4bl params...]
set -e
cd "$(dirname "$0")/.."

NJOBS=${1:-4}
NEV=${2:-50}
shift 2 2>/dev/null || shift $# || true
EXTRA="$@"

test -f out/ffa_cell_map.txt || python3 scripts/make_ffa_fieldmap.py --k 3.6 -o out/ffa_cell_map.txt

pids=()
for i in $(seq 1 $NJOBS); do
  d=runs/job$i
  mkdir -p $d/out
  cp -f out/ffa_cell_map.txt $d/out/ffa_cell_map.txt
  rm -f $d/out/VD*.txt
  ( cd $d && ../../scripts/g4bl-docker.sh ../../ffa_ring.g4bl \
        SEED=$((1000+i)) NEV=$NEV $EXTRA > g4bl.log 2>&1 ) &
  pids+=($!)
done

fail=0
for p in "${pids[@]}"; do wait $p || fail=1; done
if [ $fail -ne 0 ]; then echo "some jobs failed; check runs/job*/g4bl.log"; exit 1; fi

grep -h "simulation complete" runs/job*/g4bl.log | wc -l | xargs echo "jobs completed:"
python3 scripts/analyze_ring.py $(ls -d runs/job*/out)
