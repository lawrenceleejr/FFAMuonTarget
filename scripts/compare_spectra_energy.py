#!/usr/bin/env python3
"""
How the pi- ENERGY SPECTRA differ: 1 GeV ring vs 3 GeV ring vs 3 GeV dump.
Left: normalized shape (3 GeV is harder).  Right: absolute per proton
(3 GeV is harder AND ~10x more).  Writes plots/spectra_energy.png.
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

MPI = 139.570


def pis(paths, nev):
    a = [read_bltrack(p) for p in paths]
    a = [x for x in a if len(x)]
    allb = np.vstack(a)
    s = allb[allb[:, 7] == -211]
    p = np.sqrt(np.sum(s[:, 3:6] ** 2, axis=1))
    ke = np.sqrt(p * p + MPI * MPI) - MPI
    return p, ke, nev


SETS = [
    ("1 GeV ring (5 mm C)", "tab:green",
     [f"{d}/{n}.txt" for d in (sorted(glob.glob("runs/job[1-9]/out")) +
      sorted(glob.glob("runs/job1[0-6]/out")))
      for n in ("VDouter", "VDinner", "VDtop", "VDbot")], 1200),
    ("3 GeV ring (10 mm Be)", "tab:purple",
     [f"{d}/{n}.txt" for d in (sorted(glob.glob("runs/job2[1-9]/out")) +
      sorted(glob.glob("runs/job3[0-6]/out")))
      for n in ("VDouter", "VDinner", "VDtop", "VDbot")], 800),
    ("3 GeV dump (60 cm C)", "tab:orange",
     glob.glob("out/dump_C600/VD*.txt"), 8000),
]


def main():
    fig, axs = plt.subplots(1, 2, figsize=(13, 5.2))
    pb = np.linspace(0, 1400, 29)
    for lab, c, paths, nev in SETS:
        p, ke, n = pis(paths, nev)
        axs[0].hist(p, bins=pb, density=True, histtype="step", lw=2,
                    color=c, label=f"{lab}: med {np.median(p):.0f} MeV/c")
        w = np.full(len(p), 1.0 / n)
        axs[1].hist(p, bins=pb, weights=w, histtype="step", lw=2, color=c,
                    label=f"{lab}: {len(p)/n:.3f}/p")
    axs[0].set_xlabel(r"$\pi^-$ momentum [MeV/c]")
    axs[0].set_ylabel("normalized")
    axs[0].set_title("(a) spectral SHAPE (3 GeV is harder)")
    axs[0].legend(fontsize=9)
    axs[1].set_xlabel(r"$\pi^-$ momentum [MeV/c]")
    axs[1].set_ylabel(r"$\pi^-$ per proton per bin")
    axs[1].set_title("(b) ABSOLUTE per proton (3 GeV ~10x more)")
    axs[1].legend(fontsize=9)
    fig.suptitle(r"$\pi^-$ energy spectra: low-energy recirculation vs "
                 "high-energy single shot", fontsize=13)
    fig.tight_layout()
    fig.savefig("plots/spectra_energy.png", dpi=150)
    print("wrote plots/spectra_energy.png")
    for lab, c, paths, nev in SETS:
        p, ke, n = pis(paths, nev)
        print(f"{lab:24s}  pi-/p={len(p)/n:.3f}  med_p={np.median(p):.0f} "
              f"med_KE={np.median(ke):.0f} MeV  <KE>={ke.mean():.0f}")


if __name__ == "__main__":
    main()
