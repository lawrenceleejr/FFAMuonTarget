#!/bin/bash
# The OPTIMIZED chain: 3 GeV protons -> scaled FFA ring (R0 = 20.3 m, same
# k=3.6 lattice) with a 10 mm beryllium internal target and ~400 MHz
# energy-recovery RF -> waist-placed 0.6 m capture solenoid -> 40 m, 5 T
# decay channel -> liquid D-T target where the mu- range out (Bragg peak).
#
# Design constants below come from scripts/find_closed_orbit.py and
# scripts/tune_rf.py for the 3 GeV map (see README, "Optimized
# configuration").  Usage:
#   scripts/run_3gev_chain.sh [NBATCH=2] [NEV_PER_JOB=50]
# CPU: ~NBATCH x 5 min on 4 cores for the ring + seconds for the channel.
set -e
cd "$(dirname "$0")/.."

NBATCH=${1:-2}
NEV=${2:-50}

MAP3=out/ffa_cell_map_3gev.txt
P3="SCALE=2.25518 KE0=3000 RCO=20119.41 RSTR=19459.11 TREV=431.4673 \
    HARM=173 TMAT=Be TGT=10 SIGP=3.8 GRAD=17.4 PHID=-39.23 TMAX=54000"

test -f $MAP3 || python3 scripts/make_ffa_fieldmap.py --k 3.6 \
    --R0 20296.6 --pc 3824.87 -o $MAP3

# ---- ring production (4 parallel docker jobs per batch) ----
for b in $(seq 1 $NBATCH); do
  MAPFILE=$MAP3 SEEDBASE=$((5000+1000*b)) JOBBASE=$((20+4*(b-1))) \
      ./scripts/run_ring_parallel.sh 4 $NEV $P3
done

DIRS=$(ls -d runs/job2[1-9]/out runs/job3[0-9]/out 2>/dev/null | sort -V)
NP=$((4 * NBATCH * NEV))
python3 scripts/analyze_ring.py $DIRS

# ---- capture (waist-placed 0.6 m bore) + 40 m decay channel + D-T stop ----
python3 scripts/make_channel_input.py $DIRS --bore 600 -o out/pions_for_channel.txt
rm -f out/chan_ends.txt out/VDmuEnd.txt out/VDch*.txt
./scripts/g4bl-docker.sh pion_decay_channel.g4bl \
    LCH=40000 RCH=600 BSOL=5 STOP=1 DEG=1 CELL=3000 TRACE=40
python3 scripts/analyze_channel.py --nprotons $NP
python3 scripts/analyze_stops.py --lch 40000 --deg 1 --cell 3000 --rch 600 \
    --nprotons $NP

# ---- figures ----
python3 scripts/make_plots.py $DIRS --preset 3gev --nprotons $NP
