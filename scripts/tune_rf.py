#!/usr/bin/env python3
"""
Calibrate the energy balance of the FFA internal-target ring:

 1. target-in / RF-off run  -> mean energy loss per turn in the target
 2. four no-target / RF-on runs at PHID = 0/90/180/270 with a test
    gradient -> effective cavity voltage V_eff and the phase offset
    delta between the nominal timing and the actual crest
 3. prints the recommended GRAD (and PHID correction) so that
    V_eff(GRAD) * cos(PHID) = <dE_target> at the requested synchronous
    phase, and writes them to out/rf_setting.txt

Usage: python3 scripts/tune_rf.py [--tgt 5.0] [--phis 30] [--gtest 10]
"""

import argparse
import math
import subprocess
import numpy as np

MP = 938.272


def run(args_list, timeout=1800):
    cmd = ["./scripts/g4bl-docker.sh", "ffa_ring.g4bl"] + args_list
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if "simulation complete" not in r.stdout + r.stderr:
        print(r.stdout[-3000:], r.stderr[-1500:])
        raise RuntimeError(f"g4bl failed: {args_list}")


def turn_energies():
    """mean proton kinetic energy at the turn counter vs turn number"""
    rec = {}
    with open("out/VDturn.txt") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            p = line.split()
            if int(float(p[7])) != 2212 or int(float(p[9])) != 1:
                continue                      # primary protons only
            ev = int(float(p[8]))
            px, py, pz = float(p[3]), float(p[4]), float(p[5])
            ke = math.sqrt(px * px + py * py + pz * pz + MP * MP) - MP
            rec.setdefault(ev, []).append((float(p[6]), ke))
    by_turn = {}
    for ev, lst in rec.items():
        lst.sort()
        for i, (t, ke) in enumerate(lst):
            by_turn.setdefault(i, []).append(ke)
    turns = sorted(by_turn)
    return np.array(turns), np.array([np.mean(by_turn[i]) for i in turns])


def de_per_turn(nturns):
    t, e = turn_energies()
    if len(t) < 3:
        raise RuntimeError("too few turns recorded")
    n = min(len(t), nturns)
    return np.polyfit(t[:n], e[:n], 1)[0]     # MeV/turn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tgt", type=float, default=5.0)
    ap.add_argument("--phis", type=float, default=30.0)
    ap.add_argument("--gtest", type=float, default=10.0)
    args = ap.parse_args()

    print(f"=== 1) target {args.tgt} mm, RF off: energy loss per turn ===")
    run([f"TGT={args.tgt}", "GRAD=0", "NEV=40", "TMAX=4500", "SEED=7"])
    dE_t = -de_per_turn(20)
    print(f"    <dE_target> = {dE_t:.4f} MeV/turn")

    print(f"=== 2) no target, RF at {args.gtest} MV/m: voltage & phase ===")
    gains = {}
    # fit only ~8 turns: with RF on, the proton's energy and hence arrival
    # phase drift (synchrotron motion), which would bend a longer fit
    for phid in (0.0, 90.0, 180.0, 270.0):
        run(["TGT=0", f"GRAD={args.gtest}", f"PHID={phid}",
             "NEV=1", "TMAX=2500", "SEED=7"])
        g = de_per_turn(8)
        gains[phid] = g
        print(f"    PHID={phid:5.1f} deg  ->  dE = {g:+.4f} MeV/turn")

    # dE(PHID) = V cos(PHID*pi/180 + delta)
    A = gains[0.0] - gains[180.0]      # 2 V cos(delta)
    B = gains[270.0] - gains[90.0]     # 2 V sin(delta)
    V = 0.5 * math.hypot(A, B)
    delta = math.degrees(math.atan2(B, A))
    print(f"    V_eff = {V:.4f} MeV at {args.gtest} MV/m "
          f"({V/args.gtest:.4f} MeV per MV/m), phase offset delta = {delta:+.2f} deg")

    phid_set = args.phis - delta
    grad = args.gtest * dE_t / (V * math.cos(math.radians(args.phis)))
    print(f"=== recommended settings for phis = {args.phis} deg ===")
    print(f"    PHID={phid_set:.2f}  GRAD={grad:.3f}")

    with open("out/rf_setting.txt", "w") as f:
        f.write(f"dE_target_MeV_per_turn {dE_t:.4f}\n")
        f.write(f"Veff_MeV_per_MVm {V/args.gtest:.5f}\n")
        f.write(f"phase_offset_deg {delta:.3f}\n")
        f.write(f"PHID {phid_set:.3f}\nGRAD {grad:.4f}\n")
    print("wrote out/rf_setting.txt")


if __name__ == "__main__":
    main()
