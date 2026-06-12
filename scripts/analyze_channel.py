#!/usr/bin/env python3
"""
Analyze pion_decay_channel.g4bl output: mu- (and surviving pi-) along and
at the end of the decay solenoid, normalized per injected proton when the
ring statistics are given.

Usage: python3 scripts/analyze_channel.py [--nprotons N]
  (N = how many protons were injected in the ring runs that produced the
   channel input; with run_ring_parallel.sh it is NJOBS*NEV)
"""

import argparse
import math
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from analyze_ring import read_bltrack  # noqa: E402

MMU = 105.658
MPI = 139.570


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nprotons", type=int, default=0)
    ap.add_argument("--out", default="out")
    args = ap.parse_args()

    nin = 0
    with open(f"{args.out}/pions_for_channel.txt") as f:
        nin = sum(1 for line in f if not line.startswith("#") and line.strip())
    print(f"channel input tracks: {nin}")

    print(f"{'z[m]':>6} {'pi-':>6} {'mu-':>6}")
    for det, z in (("VDch1", 5), ("VDch2", 10), ("VDch3", 15), ("VDch4", 20)):
        a = read_bltrack(f"{args.out}/{det}.txt")
        npi = int(np.sum(a[:, 7] == -211)) if len(a) else 0
        nmu = int(np.sum(a[:, 7] == 13)) if len(a) else 0
        print(f"{z:6d} {npi:6d} {nmu:6d}")

    end = read_bltrack(f"{args.out}/VDmuEnd.txt")
    nmu = int(np.sum(end[:, 7] == 13)) if len(end) else 0
    npi = int(np.sum(end[:, 7] == -211)) if len(end) else 0
    print(f"channel end: {nmu} mu-  (+{npi} undecayed pi-)")
    if nmu:
        mu = end[end[:, 7] == 13]
        p = np.sqrt(np.sum(mu[:, 3:6] ** 2, axis=1))
        ke = np.sqrt(p * p + MMU * MMU) - MMU
        print(f"   mu- momentum: mean={p.mean():.0f} MeV/c  "
              f"[{p.min():.0f}..{p.max():.0f}];  KE mean={ke.mean():.0f} MeV")
    if args.nprotons:
        print(f"mu- per injected 1 GeV proton: {nmu/args.nprotons:.5f}  "
              f"(+{npi/args.nprotons:.5f} pi- still to decay)")


if __name__ == "__main__":
    main()
