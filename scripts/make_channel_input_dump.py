#!/usr/bin/env python3
"""
Build a decay-channel input file from a CONVENTIONAL single-pass target run
(target_yield.g4bl), applying the same idealized capture model as the ring
version (make_channel_input.py): horizontal channel axis along the mean
pi- direction, entrance plane placed at the waist of the emerging fan,
bore-radius acceptance filter, timestamps zeroed.

This makes the ring vs single-pass comparison apples-to-apples through the
identical capture solenoid + decay channel + D-T stopping stage.

Usage:
  python3 scripts/make_channel_input_dump.py out/dump_C600 \
      [--bore 600] [-o out/pions_for_channel.txt]
"""

import argparse
import glob
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from analyze_ring import read_bltrack  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", help="directory with the target box VD*.txt files")
    ap.add_argument("--bore", type=float, default=600.0)
    ap.add_argument("-o", "--output", default="out/pions_for_channel.txt")
    args = ap.parse_args()

    rows = [read_bltrack(f) for f in glob.glob(f"{args.dir}/VD*.txt")]
    rows = [r for r in rows if len(r)]
    allb = np.vstack(rows)
    sel = allb[np.isin(allb[:, 7], (-211.0, 13.0))]
    print(f"pi-/mu- leaving the target: {len(sel)}")
    if len(sel) == 0:
        sys.exit("nothing to write")

    pos = sel[:, 0:3].copy()
    mom = sel[:, 3:6].copy()

    # horizontal channel axis along the mean direction (the capture channel
    # of a conventional source looks at the target from downstream)
    ez = np.mean(mom / np.linalg.norm(mom, axis=1)[:, None], axis=0)
    ez[1] = 0.0
    ez /= np.linalg.norm(ez)
    ey = np.array([0.0, 1.0, 0.0])
    ex = np.cross(ey, ez)
    R = np.vstack([ex, ey, ez])
    origin = np.mean(pos, axis=0)
    origin[1] = 0.0

    posc = (pos - origin) @ R.T
    momc = mom @ R.T
    forward = momc[:, 2] > 0
    uz = momc[:, 2].copy()
    uz[uz == 0] = 1.0
    best = (-1, 0.0)
    for s in np.arange(-4000.0, 8000.0 + 1, 250.0):
        dzs = (posc[:, 2] - s) / uz
        xs = posc[:, 0] - momc[:, 0] * dzs
        ys = posc[:, 1] - momc[:, 1] * dzs
        n = int(np.sum(forward & (np.hypot(xs, ys) < args.bore)))
        if n > best[0]:
            best = (n, s)
    s = best[1]
    dzs = (posc[:, 2] - s) / uz
    posc = posc - momc * dzs[:, None]
    inbore = forward & (np.hypot(posc[:, 0], posc[:, 1]) < args.bore)

    print(f"channel axis = {ez}, waist offset s={s:.0f} mm")
    print(f"forward-going: {int(np.sum(forward))}")
    print(f"within bore r<{args.bore:.0f}: {int(np.sum(inbore))} "
          f"({100.0*np.sum(inbore)/len(sel):.0f}% of all pi-/mu-)")

    with open(args.output, "w") as f:
        f.write("#BLTrackFile pi-/mu- from single-pass target, channel frame\n")
        f.write("#x y z Px Py Pz t PDGid EventID TrackID ParentID Weight\n")
        for i in np.where(inbore)[0]:
            x, y, _ = posc[i]
            px, py, pz = momc[i]
            f.write(f"{x:.3f} {y:.3f} 0.0 {px:.3f} {py:.3f} {pz:.3f} "
                    f"0.0 {int(sel[i,7])} {int(sel[i,8])} "
                    f"{int(sel[i,9])} {int(sel[i,10])} 1\n")
    print(f"wrote {args.output} ({int(np.sum(inbore))} tracks)")


if __name__ == "__main__":
    main()
