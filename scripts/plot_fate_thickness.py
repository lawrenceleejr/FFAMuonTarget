#!/usr/bin/env python3
"""
Why the proton population decays (it does NOT recirculate forever) and how
target thickness trades turns for nothing.  Panel (a): fraction still
circulating vs turn for 1/5/20 mm targets (thin = many more turns, but
same fate).  Panel (b): yield/proton (flat) and mean turns (~1/thickness)
vs thickness.  Writes plots/thickness_fate.png.
"""
import glob
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from analyze_ring import read_bltrack  # noqa: E402

MP = 938.272
PER_INT = 0.061  # pi- per inelastic interaction at 1 GeV


def load(dirs):
    band, turn = [], []
    for i, d in enumerate(dirs):
        for n in ("VDouter", "VDinner", "VDtop", "VDbot"):
            a = read_bltrack(f"{d}/{n}.txt")
            if len(a):
                a = a.copy(); a[:, 8] += i * 100000; band.append(a)
        t = read_bltrack(f"{d}/VDturn.txt")
        if len(t):
            t = t.copy(); t[:, 8] += i * 100000; turn.append(t)
    band = np.vstack(band) if band else np.zeros((0, 12))
    turn = np.vstack(turn)
    prim = turn[(turn[:, 7] == 2212) & (turn[:, 9] == 1)]
    tpe = np.array([np.sum(prim[:, 8] == ev) for ev in np.unique(prim[:, 8])])
    nev = len(tpe)
    npim = int(np.sum(np.isin(band[:, 7], (-211.0, 13.0))))
    return tpe, nev, npim / nev


CASES = [
    (1, sorted(glob.glob("runs/job4[0-3]/out")), "tab:cyan"),
    (5, sorted(glob.glob("runs/job[1-9]/out")) +
        sorted(glob.glob("runs/job1[0-6]/out")), "tab:green"),
    (20, sorted(glob.glob("runs/job4[4-7]/out")), "tab:red"),
]


def main():
    fig, axs = plt.subplots(1, 2, figsize=(13, 5.2))
    summ = []
    ax = axs[0]
    for t, dirs, c in CASES:
        tpe, nev, y = load(dirs)
        summ.append((t, nev, y, tpe.mean()))
        maxt = int(tpe.max())
        ts = np.arange(0, min(maxt, 1000))
        surv = np.array([np.mean(tpe > i) for i in ts])
        ax.semilogy(ts, surv, lw=2, color=c,
                    label=f"{t} mm: mean {tpe.mean():.0f} turns")
    ax.set_xlabel("turn number")
    ax.set_ylabel("fraction of protons still circulating")
    ax.set_xlim(0, 600)
    ax.set_ylim(3e-3, 1.2)
    ax.set_title("(a) the population decays — it does NOT go forever\n"
                 "(each proton ends at its 1 inelastic interaction or the wall)")
    ax.legend(fontsize=9)

    ax = axs[1]
    ts = [s[0] for s in summ]
    ys = [s[2] for s in summ]
    yerr = [np.sqrt(s[2] * s[1]) / s[1] for s in summ]
    turns = [s[3] for s in summ]
    ax.errorbar(ts, ys, yerr=yerr, fmt="o-", color="tab:purple", capsize=4,
                label=r"$\pi^-(+\mu^-)$ per proton")
    ax.axhline(PER_INT, color="0.5", ls="--",
               label="ideal (every proton interacts): 0.061")
    ax.set_xscale("log")
    ax.set_xlabel("target thickness [mm]")
    ax.set_ylabel(r"$\pi^-(+\mu^-)$ per proton", color="tab:purple")
    ax.set_ylim(0, 0.075)
    ax2 = ax.twinx()
    ax2.plot(ts, turns, "s:", color="tab:brown")
    ax2.set_ylabel("mean turns to interact (~1/thickness)", color="tab:brown")
    ax2.set_yscale("log")
    ax.set_title("(b) thinner target = many more turns,\nSAME yield per proton")
    ax.legend(fontsize=9, loc="center right")

    fig.suptitle("Proton fate in the recirculating ring (1 GeV): population "
                 "decays, thickness only trades turns", fontsize=13)
    fig.tight_layout()
    fig.savefig("plots/thickness_fate.png", dpi=150)
    print("wrote plots/thickness_fate.png")
    for t, nev, y, mt in summ:
        print(f"  {t:2d} mm: nev={nev:4d}  yield={y:.4f}  mean_turns={mt:.0f}  "
              f"useful_frac={y/PER_INT:.2f}")


if __name__ == "__main__":
    main()
