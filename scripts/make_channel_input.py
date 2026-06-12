#!/usr/bin/env python3
"""
Convert pi-/mu- recorded on the ring enclosure detectors into a
BLTrackFile that pion_decay_channel.g4bl can use as its input beam.

Selects pi- and mu- whose exit azimuth lies in a window around the target
(where the analysis shows they are concentrated), then rotates/translates
into the channel frame: origin at the selected-sample centroid, z along
the mean pi- momentum direction, y vertical.

Usage:
  python3 scripts/make_channel_input.py [rundirs ...] \
      [--window -15 45] [-o out/pions_for_channel.txt]
Also prints the channel-frame acceptance bookkeeping.
"""

import argparse
import math
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from analyze_ring import read_bltrack  # noqa: E402

TARGET_AZ = 15.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="*", default=None)
    ap.add_argument("--window", nargs=2, type=float, default=[-20.0, 30.0],
                    help="exit azimuth window relative to the target [deg]")
    ap.add_argument("--bore", type=float, default=500.0,
                    help="capture channel bore radius [mm] (acceptance filter)")
    ap.add_argument("-o", "--output", default="out/pions_for_channel.txt")
    args = ap.parse_args()
    dirs = args.dirs or ["out"]

    rows = []
    for i, d in enumerate(dirs):
        for det in ("VDouter", "VDinner", "VDtop", "VDbot"):
            a = read_bltrack(f"{d}/{det}.txt")
            if len(a):
                a[:, 8] += i * 1000000
                rows.append(a)
    allb = np.vstack(rows)
    sel = allb[np.isin(allb[:, 7], (-211.0, 13.0))]
    print(f"pi-/mu- on enclosure: {len(sel)}")

    phi = (np.degrees(np.arctan2(sel[:, 2], sel[:, 0])) - TARGET_AZ + 180.0) % 360.0 - 180.0
    inwin = sel[(phi >= args.window[0]) & (phi <= args.window[1])]
    print(f"in azimuth window {args.window} rel. to target: {len(inwin)} "
          f"({100.0*len(inwin)/max(len(sel),1):.0f}%)")

    # a real channel must point AWAY from the ring: keep only tracks with
    # outward radial momentum (inward-going pi- cross the ring interior
    # and are lost by design)
    if len(inwin):
        ur = inwin[:, 0:3].copy()
        ur[:, 1] = 0.0
        ur /= np.linalg.norm(ur, axis=1)[:, None]
        pr = np.sum(inwin[:, 3:6] * ur, axis=1)
        inwin = inwin[pr > 0]
    print(f"outward-going (radially): {len(inwin)}")
    if len(inwin) == 0:
        sys.exit("nothing to write")

    pos = inwin[:, 0:3].copy()
    mom = inwin[:, 3:6].copy()

    # channel frame: a HORIZONTAL axis along the mean (horizontal-projected)
    # momentum of the sample - real capture channels are horizontal - with
    # y vertical and the origin at the sample centroid (i.e. the channel
    # mouth sits right where the pions emerge, just outside the chamber).
    ez = np.mean(mom / np.linalg.norm(mom, axis=1)[:, None], axis=0)
    ez[1] = 0.0
    ez /= np.linalg.norm(ez)
    ey = np.array([0.0, 1.0, 0.0])
    ex = np.cross(ey, ez)
    R = np.vstack([ex, ey, ez])           # rows are channel axes
    origin = np.mean(pos, axis=0)
    origin[1] = 0.0

    posc = (pos - origin) @ R.T
    momc = mom @ R.T
    # project each track (straight line) onto the channel entrance plane
    # z=0; this ignores the ring's fringe fields between the chamber and
    # the channel mouth (idealization)
    forward = momc[:, 2] > 0
    dz = np.where(forward, posc[:, 2] / np.where(momc[:, 2] != 0, momc[:, 2], 1), 0.0)
    posc -= momc * dz[:, None]
    inbore = forward & (np.hypot(posc[:, 0], posc[:, 1]) < args.bore)

    print(f"channel axis (global) = {ez}")
    print(f"channel origin (global) = {origin}")
    print(f"forward-going          : {int(np.sum(forward))}")
    print(f"within bore r<{args.bore:.0f} mm  : {int(np.sum(inbore))} "
          f"(capture-channel geometric acceptance "
          f"{100.0*np.sum(inbore)/max(len(sel),1):.0f}% of all pi-/mu-)")
    if not np.any(inbore):
        sys.exit("nothing accepted into the channel")

    with open(args.output, "w") as f:
        f.write("#BLTrackFile pions/muons from FFA ring enclosure, channel frame\n")
        f.write("#x y z Px Py Pz t PDGid EventID TrackID ParentID Weight\n")
        for i in np.where(inbore)[0]:
            x, y, _ = posc[i]
            px, py, pz = momc[i]
            f.write(f"{x:.3f} {y:.3f} 0.0 {px:.3f} {py:.3f} {pz:.3f} "
                    f"{inwin[i,6]:.3f} {int(inwin[i,7])} {int(inwin[i,8])} "
                    f"{int(inwin[i,9])} {int(inwin[i,10])} 1\n")
    print(f"wrote {args.output} ({int(np.sum(inbore))} tracks)")


if __name__ == "__main__":
    main()
