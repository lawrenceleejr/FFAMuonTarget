#!/usr/bin/env python3
"""
pi- kinematics at 1 GeV (single-pion / Delta regime) vs 3 GeV (multi-pion).
Shows the momentum-vs-lab-angle correlation and quantifies how much a
narrow angular selection narrows the pi- momentum band -- the only route
to a 'narrow peak' at fixed (recirculation-locked) proton energy.
Writes plots/pi_kinematics.png.
"""
import glob
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MPI = 139.570


def read(path):
    rows = []
    with open(path) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            p = line.split()
            if len(p) >= 12:
                rows.append([float(v) for v in p[:12]])
    return np.array(rows) if rows else np.zeros((0, 12))


def pim_pang(tag):
    a = [read(f) for f in glob.glob(f"out/{tag}/VD*.txt")]
    a = [x for x in a if len(x)]
    allb = np.vstack(a)
    s = allb[allb[:, 7] == -211]
    p = np.sqrt(np.sum(s[:, 3:6] ** 2, axis=1))
    ang = np.degrees(np.arctan2(np.hypot(s[:, 3], s[:, 4]), s[:, 5]))
    return p, ang


def fr(p):
    return p.std() / p.mean() if len(p) > 2 else float("nan")


def main():
    p1, a1 = pim_pang("corr_1gev") if glob.glob("out/corr_1gev/VD*.txt") \
        else pim_pang("leadp_1gev")
    p3, a3 = pim_pang("leadp_3gev")

    fig, axs = plt.subplots(1, 2, figsize=(13, 5.4))
    for ax, p, a, lab, cap in ((axs[0], p1, a1, "1 GeV (single-pion / Δ)", 450),
                               (axs[1], p3, a3, "3 GeV (multi-pion)", 1400)):
        ax.scatter(a, p, s=8, alpha=0.4)
        ax.set_xlabel(r"$\pi^-$ lab angle [deg]")
        ax.set_ylabel(r"$\pi^-$ momentum [MeV/c]")
        ax.set_xlim(0, 180)
        ax.set_ylim(0, cap)
        # narrow forward-ish slice
        m = (a > 60) & (a < 90)
        ax.axvspan(60, 90, color="orange", alpha=0.12)
        s_all, s_bin = fr(p), fr(p[m])
        ax.set_title(f"{lab}\nσ/p all-angle={s_all:.2f}, "
                     f"in 60-90° slice={s_bin:.2f} (n={int(m.sum())})")
    fig.suptitle("π⁻ momentum–angle correlation: angle-selection is the only "
                 "narrowing knob (proton energy already fixed by the ring)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig("plots/pi_kinematics.png", dpi=150)
    print("wrote plots/pi_kinematics.png")
    for lab, p, a in (("1GeV", p1, a1), ("3GeV", p3, a3)):
        print(f"{lab}: n={len(p)}  sigma/p(all)={fr(p):.2f}")
        for lo, hi in ((0, 30), (60, 90), (120, 180)):
            m = (a > lo) & (a < hi)
            if m.sum() > 3:
                print(f"   {lo}-{hi} deg: n={int(m.sum())}  "
                      f"med_p={np.median(p[m]):.0f}  sigma/p={fr(p[m]):.2f}")


if __name__ == "__main__":
    main()
