#!/usr/bin/env python3
"""
Analyze the output of ffa_ring.g4bl.

Reads the BLTrackFiles written by the enclosure detectors (out/VDouter,
VDinner, VDtop, VDbot) and the turn counter (out/VDturn) and reports:
  - injected protons, turns circulated (survival of the recirculation)
  - energy balance at the turn counter (is the RF restoring the target loss?)
  - particle yields per injected proton on the enclosure (pi-, mu-, ...)
  - pi- kinematics (momentum spectrum, exit azimuth relative to the target)
Writes a machine-readable summary to out/ring_summary.txt
"""

import math
import os
import sys
import numpy as np

MP = 938.272
NAMES = {2212: "proton", 2112: "neutron", -211: "pi-", 211: "pi+",
         13: "mu-", -13: "mu+", -321: "K-", 321: "K+", 130: "K0L",
         310: "K0S", 1000010020: "deuteron", 22: "gamma", 11: "e-", -11: "e+"}


def read_bltrack(fn):
    if not os.path.exists(fn):
        return np.zeros((0, 12))
    rows = []
    with open(fn) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            p = line.split()
            if len(p) >= 12:
                rows.append([float(v) for v in p[:12]])
    return np.array(rows) if rows else np.zeros((0, 12))


def read_dirs(dirs, name):
    """concatenate one detector's files from several run dirs, offsetting
    EventID per directory so events stay distinct"""
    parts = []
    for i, d in enumerate(dirs):
        a = read_bltrack(f"{d}/{name}.txt")
        if len(a):
            a[:, 8] += i * 1000000
            parts.append(a)
    return np.vstack(parts) if parts else np.zeros((0, 12))


def main():
    dirs = sys.argv[1:] if len(sys.argv) > 1 else ["out"]
    out = dirs[0]
    turn = read_dirs(dirs, "VDturn")
    bands = {n: read_dirs(dirs, n)
             for n in ("VDouter", "VDinner", "VDtop", "VDbot")}

    # ---------------- recirculation bookkeeping ----------------
    prim = turn[(turn[:, 7] == 2212) & (turn[:, 9] == 1)] if len(turn) else turn
    nev = len(np.unique(prim[:, 8])) if len(prim) else 0
    print(f"injected protons seen at turn counter : {nev}")
    if nev == 0:
        print("no events found - did the run produce out/VDturn.txt?")
        return

    turns_per_ev = {}
    for ev in np.unique(prim[:, 8]):
        turns_per_ev[int(ev)] = int(np.sum(prim[:, 8] == ev))
    tarr = np.array(sorted(turns_per_ev.values()))
    print(f"turns per proton: mean={tarr.mean():.1f}  median={np.median(tarr):.0f} "
          f" max={tarr.max()}")

    # energy of primaries vs turn (averaged over events)
    ke_by_turn = {}
    for ev in np.unique(prim[:, 8]):
        sel = prim[prim[:, 8] == ev]
        sel = sel[np.argsort(sel[:, 6])]
        ke = np.sqrt(np.sum(sel[:, 3:6] ** 2, axis=1) + MP * MP) - MP
        for i, k in enumerate(ke):
            ke_by_turn.setdefault(i, []).append(k)
    nt = max(ke_by_turn)
    sample = sorted(set([0, 1, 2, 5, 10, 20, 40, 80, 160, nt]))
    print("mean primary KE vs turn (RF compensation check):")
    for i in sample:
        if i in ke_by_turn:
            v = ke_by_turn[i]
            print(f"   turn {i:4d}: {np.mean(v):8.2f} MeV  (n={len(v)})")

    # ---------------- yields on the enclosure ----------------
    print("\nparticles reaching the enclosure, per injected proton:")
    allb = np.vstack([b for b in bands.values() if len(b)])
    summary = {}
    for pid in np.unique(allb[:, 7]):
        n = int(np.sum(allb[:, 7] == pid))
        name = NAMES.get(int(pid), str(int(pid)))
        summary[name] = n / nev
    for name, v in sorted(summary.items(), key=lambda kv: -kv[1]):
        print(f"   {name:10s} {v:8.4f}")

    # ---------------- pi- / mu- detail ----------------
    lines = [f"nev {nev}", f"turns_mean {tarr.mean():.2f}",
             f"turns_median {np.median(tarr):.1f}"]
    for label, pid in (("pi-", -211), ("mu-", 13)):
        sel = allb[allb[:, 7] == pid]
        n = len(sel)
        lines.append(f"{label}_per_proton {n/nev:.5f}")
        if n == 0:
            print(f"\n{label}: none")
            continue
        p = np.sqrt(np.sum(sel[:, 3:6] ** 2, axis=1))
        m = 139.570 if pid == -211 else 105.658
        ke = np.sqrt(p * p + m * m) - m
        # exit azimuth relative to the target (target at +15 deg)
        phi = (np.degrees(np.arctan2(sel[:, 2], sel[:, 0])) - 15.0) % 360.0
        # how many leave through the outer band (the natural extraction side)
        outer = bands["VDouter"]
        nouter = int(np.sum(outer[:, 7] == pid)) if len(outer) else 0
        print(f"\n{label}: {n} total -> {n/nev:.4f} per proton "
              f"({nouter/max(n,1)*100:.0f}% on outer wall)")
        q = np.percentile(p, [10, 50, 90])
        print(f"   momentum spectrum: mean={p.mean():.0f} MeV/c   "
              f"10/50/90% = {q[0]:.0f}/{q[1]:.0f}/{q[2]:.0f} MeV/c")
        print(f"   kinetic energy:    mean={ke.mean():.0f} MeV")
        hist, edges = np.histogram(phi, bins=12, range=(0, 360))
        print("   exit azimuth rel. to target [deg]: "
              + " ".join(f"{int(e)}:{h}" for e, h in zip(edges[:-1], hist)))
        lines.append(f"{label}_mean_p_MeV {p.mean():.1f}")

    with open(f"{out}/ring_summary.txt", "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nwrote {out}/ring_summary.txt")


if __name__ == "__main__":
    main()
