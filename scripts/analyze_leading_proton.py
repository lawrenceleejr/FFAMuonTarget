#!/usr/bin/env python3
"""
After a proton makes a pi-, does the leading proton stay in the ring
acceptance (so it can recirculate and make MORE pi-)?  Selects single-pass
events that produced a pi-, finds the leading (max-p) outgoing proton, and
plots its momentum loss and scattering angle vs the ring acceptance, at
1 and 3 GeV.  Writes plots/leading_proton.png.
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
DPACC = 0.20          # ring momentum acceptance (+-20%)
ANGACC = 3.0          # generous angular acceptance [deg]


def leading_after_pi(tag, p0):
    a = np.vstack([read_bltrack(f) for f in glob.glob(f"out/leadp_{tag}/VD*.txt")])
    pi_ev, lead = set(), {}
    for r in a:
        ev, pid = int(r[8]), int(r[7])
        p = np.hypot(np.hypot(r[3], r[4]), r[5])
        if pid == -211:
            pi_ev.add(ev)
        elif pid == 2212:
            ang = np.degrees(np.arctan2(np.hypot(r[3], r[4]), r[5]))
            if ev not in lead or p > lead[ev][0]:
                lead[ev] = (p, ang)
    rows = [lead[e] for e in pi_ev if e in lead]
    pp = np.array([r[0] for r in rows])
    aa = np.array([r[1] for r in rows])
    return pp / p0 - 1.0, aa, len(pi_ev)


def main():
    fig, axs = plt.subplots(1, 2, figsize=(13, 5.4), sharey=True)
    for ax, (tag, p0, lab) in zip(axs, (("1gev", 1696.04, "1 GeV"),
                                        ("3gev", 3824.87, "3 GeV"))):
        dp, ang, npi = leading_after_pi(tag, p0)
        inacc = (np.abs(dp) < DPACC) & (ang < ANGACC)
        ax.scatter(dp * 100, ang, s=10, alpha=0.4,
                   color="tab:red" if tag == "1gev" else "tab:purple")
        # acceptance box
        ax.axvspan(-DPACC * 100, DPACC * 100, ymin=0, ymax=ANGACC / 60.0,
                   color="green", alpha=0.12)
        ax.axvline(-DPACC * 100, color="green", ls="--", lw=1)
        ax.axvline(DPACC * 100, color="green", ls="--", lw=1)
        ax.axhline(ANGACC, color="green", ls="--", lw=1)
        ax.set_xlim(-100, 30)
        ax.set_ylim(0, 60)
        ax.set_xlabel("leading-proton momentum change $\\Delta p/p$ [%]")
        if tag == "1gev":
            ax.set_ylabel("leading-proton scattering angle [deg]")
        ax.set_title(f"{lab}: {len(dp)} $\\pi^-$ events\n"
                     f"in ring acceptance (green): {100*np.mean(inacc):.0f}%")
        ax.annotate(f"median $\\Delta p/p$ = {np.median(dp)*100:.0f}%\n"
                    f"median angle = {np.median(ang):.0f}$\\degree$",
                    xy=(0.04, 0.96), xycoords="axes fraction", va="top",
                    fontsize=10, bbox=dict(boxstyle="round", fc="white",
                                           alpha=0.8))
    fig.suptitle("The leading proton survives the $\\pi^-$ interaction — but is "
                 "kicked far outside the ring acceptance", fontsize=13)
    fig.tight_layout()
    fig.savefig("plots/leading_proton.png", dpi=150)
    print("wrote plots/leading_proton.png")
    for tag, p0 in (("1gev", 1696.04), ("3gev", 3824.87)):
        dp, ang, npi = leading_after_pi(tag, p0)
        inacc = np.mean((np.abs(dp) < DPACC) & (ang < ANGACC))
        print(f"{tag}: {len(dp)} pi- events; leading proton median dp/p="
              f"{np.median(dp)*100:.0f}% angle={np.median(ang):.0f} deg; "
              f"in acceptance={inacc:.2f}")


if __name__ == "__main__":
    main()
