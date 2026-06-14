#!/usr/bin/env python3
"""
Proton vs deuteron pi- production (thin 20 mm graphite, single pass).
Compares matched ENERGY PER NUCLEON (a deuteron = bound p+n, so compare a
2 GeV deuteron to a 1 GeV proton).  Reports pi- per beam particle, per
nucleon, per GeV, and the pi+/pi- charge ratio; writes plots/deuteron.png.
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

NEV = 8000


def counts(paths):
    a = [read_bltrack(p) for p in paths]
    a = [x for x in a if len(x)]
    allb = np.vstack(a)
    npim = int(np.sum(allb[:, 7] == -211)) + int(np.sum(allb[:, 7] == 13))
    npip = int(np.sum(allb[:, 7] == 211))
    return npim, npip


# (label, glob, total KE [GeV], nucleons)
SETS = [
    ("p 1 GeV", "out/scan/KE1000/VD*.txt", 1.0, 1),
    ("p 2 GeV", "out/scan/KE2000/VD*.txt", 2.0, 1),
    ("p 3 GeV", "out/scan/KE3000/VD*.txt", 3.0, 1),
    ("d 1 GeV", "out/d_1000/VD*.txt", 1.0, 2),
    ("d 2 GeV", "out/d_2000/VD*.txt", 2.0, 2),
    ("d 4 GeV", "out/d_4000/VD*.txt", 4.0, 2),
    ("d 6 GeV", "out/d_6000/VD*.txt", 6.0, 2),
]


def main():
    rows = []
    print(f"{'beam':10s} {'GeV/nuc':>8} {'pi-/beam':>9} {'pi-/nucleon':>12} "
          f"{'pi-/GeV':>9} {'pi+/pi-':>8}")
    for lab, g, ke, nuc in SETS:
        paths = glob.glob(g)
        if not paths:
            print(f"{lab}: MISSING"); continue
        npim, npip = counts(paths)
        ypb = npim / NEV
        rows.append(dict(lab=lab, ke=ke, nuc=nuc, epn=ke / nuc, ypb=ypb,
                         ypn=ypb / nuc, ypg=ypb / ke,
                         ratio=npip / max(npim, 1), kind=lab[0]))
        print(f"{lab:10s} {ke/nuc:8.1f} {ypb:9.4f} {ypb/nuc:12.4f} "
              f"{ypb/ke:9.4f} {npip/max(npim,1):8.2f}")

    fig, axs = plt.subplots(1, 3, figsize=(15, 4.8))
    pr = [r for r in rows if r["kind"] == "p"]
    de = [r for r in rows if r["kind"] == "d"]

    # (a) pi- per GeV vs energy per nucleon
    ax = axs[0]
    ax.plot([r["epn"] for r in pr], [r["ypg"] for r in pr], "o-",
            color="tab:blue", label="proton")
    ax.plot([r["epn"] for r in de], [r["ypg"] for r in de], "s-",
            color="tab:red", label="deuteron")
    ax.set_xlabel("energy per nucleon [GeV]")
    ax.set_ylabel(r"$\pi^-$ per GeV of beam energy")
    ax.set_title("(a) yield per unit beam energy")
    ax.legend()

    # (b) pi- per nucleon vs energy per nucleon
    ax = axs[1]
    ax.plot([r["epn"] for r in pr], [r["ypn"] for r in pr], "o-",
            color="tab:blue", label="proton")
    ax.plot([r["epn"] for r in de], [r["ypn"] for r in de], "s-",
            color="tab:red", label="deuteron")
    ax.set_xlabel("energy per nucleon [GeV]")
    ax.set_ylabel(r"$\pi^-$ per nucleon")
    ax.set_title("(b) yield per nucleon")
    ax.legend()

    # (c) charge ratio
    ax = axs[2]
    ax.plot([r["epn"] for r in pr], [r["ratio"] for r in pr], "o-",
            color="tab:blue", label="proton")
    ax.plot([r["epn"] for r in de], [r["ratio"] for r in de], "s-",
            color="tab:red", label="deuteron")
    ax.axhline(1.0, color="0.5", ls=":")
    ax.set_xlabel("energy per nucleon [GeV]")
    ax.set_ylabel(r"$\pi^+/\pi^-$ ratio (lower = better for $\pi^-$)")
    ax.set_title("(c) charge ratio: deuteron makes it symmetric")
    ax.legend()

    fig.suptitle("Deuteron vs proton for $\\pi^-$ production (thin graphite, "
                 "matched energy per nucleon)", fontsize=13)
    fig.tight_layout()
    fig.savefig("plots/deuteron.png", dpi=150)
    print("wrote plots/deuteron.png")


if __name__ == "__main__":
    main()
