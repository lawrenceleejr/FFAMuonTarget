#!/usr/bin/env python3
"""
Does recirculation at 1 GeV beat a single-shot 3 GeV target on pi- yield?

Pulls pi-(+in-flight mu-) escaping per proton for, at 1 and 3 GeV:
  - thin single pass (20 mm)         -> the "one pass, full energy" baseline
  - thick dump (~1.2 lambda)         -> "every proton interacts" for free
  - recirculating FFA ring           -> "every proton interacts at full E"
Decomposes the two independent gain factors (interaction probability vs
per-interaction yield) and writes plots/recycle_vs_energy.png.
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


def yield_per_p(paths, nev):
    a = [read_bltrack(p) for p in paths]
    a = [x for x in a if len(x)]
    if not a:
        return None
    allb = np.vstack(a)
    n = int(np.sum(np.isin(allb[:, 7], (-211.0, 13.0))))
    return n / nev, np.sqrt(n) / nev


CASES = [
    # label, energy, kind, paths, nev
    ("thin 20 mm\n(single pass)", 1.0, "thin",
     glob.glob("out/scan/KE1000/VD*.txt"), 8000),
    ("thick 45 cm C\n(dump)", 1.0, "dump",
     glob.glob("out/scan_thick/T450/VD*.txt"), 8000),
    ("5 mm, recirculating\n(FFA ring)", 1.0, "ring",
     [f"{d}/{n}.txt" for d in (sorted(glob.glob("runs/job[1-9]/out")) +
      sorted(glob.glob("runs/job1[0-6]/out")))
      for n in ("VDouter", "VDinner", "VDtop", "VDbot")], 1200),
    ("thin 20 mm\n(single pass)", 3.0, "thin",
     glob.glob("out/scan/KE3000/VD*.txt"), 8000),
    ("thick 60 cm C\n(dump)", 3.0, "dump",
     glob.glob("out/dump_C600/VD*.txt"), 8000),
    ("10 mm Be, recirc.\n(FFA ring)", 3.0, "ring",
     [f"{d}/{n}.txt" for d in (sorted(glob.glob("runs/job2[1-9]/out")) +
      sorted(glob.glob("runs/job3[0-6]/out")))
      for n in ("VDouter", "VDinner", "VDtop", "VDbot")], 800),
]

COLOR = {"thin": "0.7", "dump": "tab:orange", "ring": "tab:purple"}


def main():
    print(f"{'case':34s} {'E[GeV]':>6} {'pi-+mu-/p':>11} {'+/-':>8} {'/GeV':>8}")
    rows = []
    for lab, e, kind, paths, nev in CASES:
        r = yield_per_p(paths, nev)
        if r is None:
            print(f"{lab.replace(chr(10),' '):34s}  MISSING")
            continue
        y, dy = r
        rows.append((lab, e, kind, y, dy))
        print(f"{lab.replace(chr(10),' '):34s} {e:6.0f} {y:11.4f} "
              f"{dy:8.4f} {y/e:8.4f}")

    fig, axs = plt.subplots(1, 2, figsize=(13, 5.2))
    ax = axs[0]
    xpos = {1.0: 0, 3.0: 1}
    off = {"thin": -0.27, "dump": 0.0, "ring": 0.27}
    for lab, e, kind, y, dy in rows:
        x = xpos[e] + off[kind]
        ax.bar(x, y, width=0.25, color=COLOR[kind], yerr=dy, capsize=3)
        ax.annotate(f"{y:.3f}", xy=(x, y + dy), ha="center", va="bottom",
                    fontsize=8)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["1 GeV protons", "3 GeV protons"])
    ax.set_ylabel(r"$\pi^-(+\mu^-)$ escaping per proton")
    ax.set_title("recirculation rescues 1 GeV vs a thin pass,\n"
                 "but cannot reach 3 GeV")
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLOR[k]) for k in
               ("thin", "dump", "ring")]
    ax.legend(handles, ["thin single pass", "thick dump",
                        "recirculating ring"], fontsize=9)

    # decomposition: yield = P(interact) x yield-per-interaction
    ax = axs[1]
    scan = np.loadtxt("out/scan_summary.txt")
    ke = scan[:, 0] / 1000.0
    per_int = scan[:, 6]            # pi- per inelastic interaction
    ax.plot(ke, per_int, "o-", color="k", label=r"$\pi^-$ per interaction")
    ax.axvline(1.0, color="0.6", ls=":")
    ax.axvline(3.0, color="0.6", ls=":")
    ax.annotate("1 GeV\n0.061", xy=(1.0, 0.061), xytext=(1.15, 0.18),
                fontsize=9, arrowprops=dict(arrowstyle="->"))
    ax.annotate("3 GeV\n0.31", xy=(3.0, 0.31), xytext=(3.0, 0.45),
                fontsize=9, ha="center", arrowprops=dict(arrowstyle="->"))
    ax.set_xlabel("proton kinetic energy [GeV]")
    ax.set_ylabel(r"$\pi^-$ per inelastic interaction")
    ax.set_title("the factor recirculation CANNOT buy:\n"
                 "per-interaction yield (beam energy only)")
    ax.legend(fontsize=9)

    fig.suptitle("Recirculation (interaction probability) vs beam energy "
                 "(per-interaction yield)", fontsize=13)
    fig.tight_layout()
    fig.savefig("plots/recycle_vs_energy.png", dpi=150)
    print("wrote plots/recycle_vs_energy.png")


if __name__ == "__main__":
    main()
