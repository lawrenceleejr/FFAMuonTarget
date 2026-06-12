#!/usr/bin/env python3
"""
Analyze the muon STOPPING stage of pion_decay_channel.g4bl (STOP=1).

Reads out/chan_ends.txt (beamlossntuple: the END point of every track) and
counts mu- that came to rest (|p| ~ 0, the Bragg-peak stop) inside the
stopping cell, plus the stop-depth distribution ("how neatly" they stop).

Usage:
  python3 scripts/analyze_stops.py --lch 30000 --deg 120 --cell 600 \
      --rch 500 [--nprotons N] [--quiet]
The geometry arguments must match the channel run.
"""

import argparse
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from analyze_ring import read_bltrack  # noqa: E402

RHO_LD2 = 0.169  # g/cm^3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lch", type=float, default=30000.0)
    ap.add_argument("--deg", type=float, default=120.0)
    ap.add_argument("--cell", type=float, default=600.0)
    ap.add_argument("--rch", type=float, default=500.0)
    ap.add_argument("--nprotons", type=int, default=0)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    ends = read_bltrack("out/chan_ends.txt")
    if len(ends) == 0:
        print("no out/chan_ends.txt entries")
        return

    z0 = args.lch + 100.0 + args.deg          # cell entrance
    z1 = z0 + args.cell
    p = np.sqrt(np.sum(ends[:, 3:6] ** 2, axis=1))
    r = np.hypot(ends[:, 0], ends[:, 1])
    mu = ends[:, 7] == 13
    pi = ends[:, 7] == -211
    stopped = mu & (p < 1.0)
    in_cell = stopped & (ends[:, 2] >= z0) & (ends[:, 2] <= z1) & (r < args.rch)
    in_deg = stopped & (ends[:, 2] < z0)

    n_in = int(np.sum(in_cell))
    nmu_end = int(np.sum(mu))
    if not args.quiet:
        print(f"track ends recorded            : {len(ends)}")
        print(f"mu- track ends                 : {nmu_end} "
              f"(stopped anywhere: {int(np.sum(stopped))}, "
              f"decayed in flight: {int(np.sum(mu & (p >= 1.0)))})")
        print(f"mu- stopped in degrader        : {int(np.sum(in_deg))}")
        print(f"pi- reaching degrader/cell (captured, lost): "
              f"{int(np.sum(pi & (ends[:, 2] > args.lch)))}")
    print(f"mu- STOPPED IN CELL            : {n_in}")
    if n_in:
        depth = (ends[in_cell, 2] - z0) / 10.0          # cm
        gcm2 = depth * RHO_LD2
        q = np.percentile(depth, [10, 50, 90])
        print(f"   stop depth in LD2: mean={depth.mean():.1f} cm "
              f"(10/50/90% = {q[0]:.0f}/{q[1]:.0f}/{q[2]:.0f} cm; "
              f"{gcm2.mean():.2f} g/cm2 mean)")
        hist, edges = np.histogram(depth, bins=12, range=(0, args.cell / 10.0))
        print("   depth profile [cm:count] "
              + " ".join(f"{int(e)}:{h}" for e, h in zip(edges[:-1], hist)))
    if args.nprotons:
        print(f"mu- stopped in cell per proton : {n_in/args.nprotons:.5f}")


if __name__ == "__main__":
    main()
