#!/usr/bin/env python3
"""
Find the closed orbit of the FFA ring by scanning the launch radius X0 of
ring_test.g4bl, then measure the ring properties on the closed orbit:
revolution period, orbit length, radius in the straights (for placing the
target and RF cavity), and the betatron tunes.

The launch point (azimuth 0, center of the F magnet of cell 0) lies on a
mirror plane of the lattice, so the closed orbit there has px=0 and the
search is one-dimensional in X0.

Usage: python3 scripts/find_closed_orbit.py [--coarse lo hi step]
Writes the results to out/closed_orbit.txt
"""

import argparse
import math
import subprocess
import sys
import numpy as np

G4BL = ["./scripts/g4bl-docker.sh", "ring_test.g4bl"]
TREV_GUESS = 216.0          # ns (overridden by --trev)
EXTRA = []                  # extra deck params, e.g. MAP=..., SCALE=...


def run_g4bl(X0, Y0=0.0, turns=3.0, P0=1696.04):
    tmax = turns * TREV_GUESS
    cmd = G4BL + [f"X0={X0}", f"Y0={Y0}", f"TMAX={tmax}", f"P0={P0}"] + EXTRA
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    if "simulation complete" not in r.stdout + r.stderr:
        print(r.stdout[-3000:], r.stderr[-2000:])
        raise RuntimeError(f"g4bl failed for X0={X0}")
    dat = []
    with open("out/trace.txt") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            p = line.split()
            dat.append([float(v) for v in p[:7]])
    return np.array(dat)        # x y z Px Py Pz t


def orbit_samples(dat):
    """r and t interpolated at every cell-center azimuth (0, 30, 60, ... deg),
    plus y samples there; azimuth unwrapped so turn counting works."""
    x, y, z, t = dat[:, 0], dat[:, 1], dat[:, 2], dat[:, 6]
    phi = np.unwrap(np.arctan2(z, x))
    r = np.hypot(x, z)
    # the proton starts at phi=0 moving toward +z (phi increasing)
    n_cells = int(phi[-1] // (math.pi / 6.0))
    rs, ts, ys = [], [], []
    for k in range(1, n_cells + 1):
        target = k * math.pi / 6.0
        i = np.searchsorted(phi, target)
        if i == 0 or i >= len(phi):
            break
        f = (target - phi[i - 1]) / (phi[i] - phi[i - 1])
        rs.append(r[i - 1] + f * (r[i] - r[i - 1]))
        ts.append(t[i - 1] + f * (t[i] - t[i - 1]))
        ys.append(y[i - 1] + f * (y[i] - y[i - 1]))
    return np.array(rs), np.array(ts), np.array(ys), phi, r, t


def co_metric(X0, p0):
    dat = run_g4bl(X0, P0=p0)
    rs, ts, ys, phi, r, t = orbit_samples(dat)
    if len(rs) < 24:
        return 1e9, rs
    return float(np.std(rs)), rs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coarse", nargs=3, type=float, default=[8700, 9050, 50])
    ap.add_argument("--p0", type=float, default=1696.04,
                    help="proton momentum [MeV/c] (1696.04 = 1 GeV kinetic)")
    ap.add_argument("--trev", type=float, default=216.0,
                    help="revolution period guess [ns] (sets run lengths)")
    ap.add_argument("--extra", nargs="*", default=[],
                    help="extra deck params, e.g. MAP=out/x.txt SCALE=2.255")
    args = ap.parse_args()
    global TREV_GUESS, EXTRA
    TREV_GUESS = args.trev
    EXTRA = list(args.extra)

    print("=== coarse scan ===")
    best = (1e9, None)
    lo, hi, st = args.coarse
    for X0 in np.arange(lo, hi + st / 2, st):
        m, rs = co_metric(X0, args.p0)
        print(f"X0={X0:8.1f}   std(r@cells)={m:9.3f} mm   mean={np.mean(rs):9.2f}")
        if m < best[0]:
            best = (m, X0)

    print("=== refine (parabolic on metric^2) ===")
    X = best[1]
    step = st / 2.0
    for _ in range(6):
        xs = [X - step, X, X + step]
        ms = []
        for x0 in xs:
            m, rs = co_metric(x0, args.p0)
            ms.append(m * m)
            print(f"X0={x0:9.2f}   std={math.sqrt(ms[-1]):8.4f} mm")
        # parabola vertex
        denom = (ms[0] - 2 * ms[1] + ms[2])
        if denom <= 0:
            X = xs[int(np.argmin(ms))]
        else:
            X = X + 0.5 * step * (ms[0] - ms[2]) / denom
        step /= 3.0
        if step < 0.05:
            break

    print(f"closed orbit launch radius X0 = {X:.2f} mm")

    # ---- final characterization run: 12 turns on the closed orbit ----
    dat = run_g4bl(X, turns=12.2, P0=args.p0)
    rs, ts, ys, phi, r, t = orbit_samples(dat)
    n_turn = len(rs) // 12
    # revolution period from cell-crossing times (lstsq slope)
    k = np.arange(1, len(ts) + 1)
    Trev = 12.0 * np.polyfit(k, ts, 1)[0]
    p = np.linalg.norm(dat[0, 3:6])
    E = math.hypot(p, 938.272)
    beta = p / E
    v = beta * 299.792458
    L = Trev * v
    # orbit radius at the straight centers (azimuth 15+30k deg)
    phis = np.unwrap(np.arctan2(dat[:, 2], dat[:, 0]))
    rr = np.hypot(dat[:, 0], dat[:, 2])
    rstr = []
    for kk in range(len(rs)):
        targ = (15.0 + 30.0 * kk) * math.pi / 180.0
        i = np.searchsorted(phis, targ)
        if 0 < i < len(phis):
            f = (targ - phis[i - 1]) / (phis[i] - phis[i - 1])
            rstr.append(rr[i - 1] + f * (rr[i] - rr[i - 1]))
    rstr = np.array(rstr)

    print(f"residual std(r@cells) = {np.std(rs):.4f} mm")
    print(f"Trev = {Trev:.4f} ns   v = {v:.4f} mm/ns   orbit length = {L:.2f} mm")
    print(f"r at straight centers = {np.mean(rstr):.2f} +- {np.std(rstr):.3f} mm")

    # ---- tunes: betatron oscillations about the closed orbit ----
    datx = run_g4bl(X + 30.0, turns=16.2, P0=args.p0)
    rsx, _, _, _, _, _ = orbit_samples(datx)
    daty = run_g4bl(X, Y0=20.0, turns=16.2, P0=args.p0)
    _, _, ysy, _, _, _ = orbit_samples(daty)

    def tune_from(seq):
        seq = np.asarray(seq) - np.mean(seq)
        n = len(seq)
        ft = np.abs(np.fft.rfft(seq * np.hanning(n)))
        i = np.argmax(ft[1:]) + 1
        # 3-point interpolation
        if 1 <= i < len(ft) - 1:
            a, b, c = ft[i - 1], ft[i], ft[i + 1]
            d = 0.5 * (a - c) / (a - 2 * b + c)
        else:
            d = 0
        return (i + d) / n        # cell tune (per 30-deg cell)

    nux_cell = tune_from(rsx)
    nuy_cell = tune_from(ysy)
    print(f"cell tunes: nux={nux_cell:.4f}  nuy={nuy_cell:.4f}")
    print(f"ring tunes (x12, modulo aliasing): Qx={12*nux_cell:.3f}  Qy={12*nuy_cell:.3f}")
    ymax = np.max(np.abs(ysy))
    print(f"vertical oscillation max |y| for 20mm launch = {ymax:.2f} mm "
          f"({'STABLE' if ymax < 100 else 'POSSIBLY UNSTABLE'})")

    with open("out/closed_orbit.txt", "w") as f:
        f.write(f"X0_closed_orbit_mm {X:.3f}\n")
        f.write(f"Trev_ns {Trev:.4f}\n")
        f.write(f"orbit_length_mm {L:.2f}\n")
        f.write(f"r_straight_mm {np.mean(rstr):.2f}\n")
        f.write(f"nux_cell {nux_cell:.4f}\nnuy_cell {nuy_cell:.4f}\n")
    print("wrote out/closed_orbit.txt")


if __name__ == "__main__":
    main()
