#!/usr/bin/env python3
"""
Summarize the target_yield.g4bl energy scan (out/scan/KE*/VD*.txt).

For each beam energy: yields per proton of pi-, pi+, mu-, neutrons leaving
the target, the pi- spectrum, and the figures of merit for the FFA
internal-target source:
   pi-/proton            (thin-target yield per pass)
   pi-/interaction       (what the energy-recovery ring delivers per proton,
                          since every recirculating proton eventually
                          interacts at full energy)
   pi-/proton/GeV        (production efficiency per unit beam energy ~ per
                          watt of beam power)
Usage: python3 scripts/analyze_yield.py [--nev 2000] [--thick 20]
"""

import argparse
import glob
import math
import os
import re
import numpy as np

MP = 938.272
MPI = 139.570


def read_bltrack(fn):
    rows = []
    with open(fn) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            p = line.split()
            if len(p) >= 12:
                rows.append([float(v) for v in p[:12]])
    return np.array(rows) if rows else np.zeros((0, 12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nev", type=int, default=2000)
    ap.add_argument("--thick", type=float, default=20.0)
    ap.add_argument("--scandir", default="out/scan")
    args = ap.parse_args()

    dirs = sorted(glob.glob(f"{args.scandir}/KE*"),
                  key=lambda d: float(re.findall(r"KE([\d.]+)", d)[0]))
    if not dirs:
        print("no out/scan/KE* directories found - run scripts/run_energy_scan.sh")
        return

    print(f"target: graphite {args.thick} mm, {args.nev} protons/point")
    hdr = (f"{'KE[MeV]':>8} {'pi-/p':>9} {'pi+/p':>9} {'pi-/pi+':>8} "
           f"{'n/p':>7} {'survive':>8} {'pi-/int':>9} {'pi-/p/GeV':>10} "
           f"{'<p_pi->':>8}")
    print(hdr)
    print("-" * len(hdr))
    rows = []
    for d in dirs:
        ke = float(re.findall(r"KE([\d.]+)", d)[0])
        dat = [read_bltrack(fn) for fn in glob.glob(f"{d}/VD*.txt")]
        dat = [x for x in dat if len(x)]
        if not dat:
            continue
        allp = np.vstack(dat)
        npim = int(np.sum(allp[:, 7] == -211))
        npip = int(np.sum(allp[:, 7] == 211))
        nmum = int(np.sum(allp[:, 7] == 13))
        nn = int(np.sum(allp[:, 7] == 2112))
        # surviving primaries: primary-track protons with >90% of beam KE
        prim = allp[(allp[:, 7] == 2212) & (allp[:, 9] == 1) & (allp[:, 10] == 0)]
        if len(prim):
            kep = np.sqrt(np.sum(prim[:, 3:6] ** 2, axis=1) + MP * MP) - MP
            nsurv = int(np.sum(kep > 0.90 * ke))
        else:
            nsurv = 0
        nint = max(args.nev - nsurv, 1)
        pim = allp[allp[:, 7] == -211]
        pmean = float(np.mean(np.sqrt(np.sum(pim[:, 3:6] ** 2, axis=1)))) if npim else 0.0
        # mu- from in-flight decay count toward the pi- total
        y_pim = (npim + nmum) / args.nev
        rows.append((ke, y_pim, npip / args.nev, npim / max(npip, 1),
                     nn / args.nev, nsurv / args.nev, (npim + nmum) / nint,
                     y_pim / (ke / 1000.0), pmean))
        print(f"{ke:8.0f} {rows[-1][1]:9.5f} {rows[-1][2]:9.5f} "
              f"{rows[-1][3]:8.3f} {rows[-1][4]:7.3f} {rows[-1][5]:8.3f} "
              f"{rows[-1][6]:9.4f} {rows[-1][7]:10.5f} {rows[-1][8]:8.0f}")

    with open("out/scan_summary.txt", "w") as f:
        f.write("# KE_MeV pim_per_p pip_per_p ratio n_per_p surv pim_per_int "
                "pim_per_p_per_GeV mean_p_pim\n")
        for r in rows:
            f.write(" ".join(f"{v:.6g}" for v in r) + "\n")
    print("\nwrote out/scan_summary.txt")
    if rows:
        best_eff = max(rows, key=lambda r: r[7])
        best_int = max(rows, key=lambda r: r[6])
        print(f"best pi- per proton per GeV (beam-power FOM): {best_eff[0]:.0f} MeV")
        print(f"best pi- per interaction (recirculating-ring FOM): {best_int[0]:.0f} MeV")


if __name__ == "__main__":
    main()
