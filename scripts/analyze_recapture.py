#!/usr/bin/env python3
"""
Could a VERY inclusive FFA recapture the leading proton after a pi-
interaction and re-use it?  Scans the recapture fraction vs angular and
momentum acceptance, at 1 and 3 GeV, from the thin-target leading-proton
sample.  Marks where a realistic scaling-FFA sits and a very-inclusive
design.  Writes plots/recapture_vs_acceptance.png.
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
KEMIN_PI = 700.0   # below ~0.7 GeV KE, pi- production collapses (see scan)


def leading(tag, p0):
    a = np.vstack([read_bltrack(f) for f in glob.glob(f"out/leadp_{tag}/VD*.txt")])
    pi_ev, lead = set(), {}
    for r in a:
        ev, pid = int(r[8]), int(r[7])
        p = np.hypot(np.hypot(r[3], r[4]), r[5])
        if pid == -211:
            pi_ev.add(ev)
        elif pid == 2212:
            ang = np.degrees(np.arctan2(np.hypot(r[3], r[4]), r[5]))
            ke = np.sqrt(p * p + MP * MP) - MP
            if ev not in lead or p > lead[ev][0]:
                lead[ev] = (p, ang, ke)
    rows = np.array([lead[e] for e in pi_ev if e in lead])
    return rows[:, 0] / p0 - 1.0, rows[:, 1], rows[:, 2]   # dp/p, angle, KE


def main():
    fig, axs = plt.subplots(1, 2, figsize=(13, 5.4), sharey=True)
    angs = np.linspace(0, 25, 60)
    for ax, (tag, p0, lab) in zip(axs, (("1gev", 1696.04, "1 GeV"),
                                        ("3gev", 3824.87, "3 GeV"))):
        dp, ang, ke = leading(tag, p0)
        n = len(dp)
        for dpmax, c in ((0.30, "tab:blue"), (0.50, "tab:orange"),
                         (0.70, "tab:red")):
            frac = [np.mean((dp > -dpmax) & (ang < am)) for am in angs]
            ax.plot(angs, frac, color=c, lw=2,
                    label=f"momentum accept. -{int(dpmax*100)}%")
        # and the fraction that is BOTH recaptured AND still hard enough to
        # make pi- again (KE > 0.7 GeV) -- the only ones that would help
        useful = [np.mean((dp > -0.70) & (ang < am) & (ke > KEMIN_PI))
                  for am in angs]
        ax.plot(angs, useful, "k--", lw=2,
                label="recaptured AND KE>0.7 GeV")
        ax.axvspan(8, 13, color="green", alpha=0.10)
        ax.annotate("aggressive FFA\nangular accept.\n(~8-13 deg)",
                    xy=(10.5, 0.6), ha="center", fontsize=9, color="green")
        ax.set_xlabel("angular acceptance [deg]")
        if tag == "1gev":
            ax.set_ylabel("fraction of leading protons recaptured")
        ax.set_ylim(0, 1)
        ax.set_title(f"{lab} ({n} $\\pi^-$ events)")
        ax.legend(fontsize=8, loc="upper left")
    fig.suptitle("Even a very inclusive FFA recaptures few post-$\\pi^-$ "
                 "protons — and almost none still able to make $\\pi^-$",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig("plots/recapture_vs_acceptance.png", dpi=150)
    print("wrote plots/recapture_vs_acceptance.png")
    for tag, p0 in (("1gev", 1696.04), ("3gev", 3824.87)):
        dp, ang, ke = leading(tag, p0)
        for dpmax, am in ((0.30, 8.0), (0.50, 13.0), (0.70, 20.0)):
            r = np.mean((dp > -dpmax) & (ang < am))
            ru = np.mean((dp > -dpmax) & (ang < am) & (ke > KEMIN_PI))
            print(f"{tag}: accept -{int(dpmax*100)}% & <{am:.0f}deg -> "
                  f"recapture {r:.2f}, of which KE>0.7GeV {ru:.2f}")


if __name__ == "__main__":
    main()
