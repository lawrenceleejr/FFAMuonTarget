#!/usr/bin/env python3
"""
Head-to-head: the FFA energy-recovery internal-target ring vs a
conventional single-pass thick target, both at 3 GeV and both followed by
the IDENTICAL capture solenoid (0.6 m bore, waist-placed) + 40 m, 5 T decay
channel + 3 m liquid D-T stopping target.

Reads the preserved channel end-point files
(out/chan_ends_{ring,dump}.txt, out/VDmuEnd_{ring,dump}.txt) and the
chain-stage counts, writes plots/ring_vs_dump.png and prints the table.
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from analyze_ring import read_bltrack  # noqa: E402

# chain-stage counts from the production runs (see README for commands)
CASES = {
    "FFA ring (3 GeV, 10 mm Be internal)": dict(
        np_=800, escaped=216, captured=90, ends="out/chan_ends_ring.txt",
        vdend="out/VDmuEnd_ring.txt", color="tab:purple"),
    "single-pass dump (3 GeV, 60 cm C)": dict(
        np_=8000, escaped=2290, captured=1661, ends="out/chan_ends_dump.txt",
        vdend="out/VDmuEnd_dump.txt", color="tab:orange"),
}
LCH, DEG, CELL, RCH = 40000.0, 1.0, 3000.0, 600.0


def stops_and_depth(fn):
    ends = read_bltrack(fn)
    p = np.sqrt(np.sum(ends[:, 3:6] ** 2, axis=1))
    z0 = LCH + 100.0 + DEG
    ok = (ends[:, 7] == 13) & (p < 1.0) & (ends[:, 2] >= z0) \
        & (ends[:, 2] <= z0 + CELL) & (np.hypot(ends[:, 0], ends[:, 1]) < RCH)
    return int(ok.sum()), (ends[ok, 2] - z0) / 10.0


def main():
    stages = ["$\\pi^-+\\mu^-$\nout of target/ring", "captured into\nchannel",
              "$\\mu^-$ at\nchannel end", "$\\mu^-$ STOPPED\nin D-T"]
    fig, axs = plt.subplots(1, 2, figsize=(13, 5.2))

    ax = axs[0]
    width = 0.38
    print(f"{'stage':34s} {'ring/p':>12} {'dump/p':>12}")
    tablerows = []
    for j, (label, c) in enumerate(CASES.items()):
        endvd = read_bltrack(c["vdend"])
        nmu_end = int(np.sum(endvd[:, 7] == 13)) if len(endvd) else 0
        nstop, depth = stops_and_depth(c["ends"])
        counts = [c["escaped"], c["captured"], nmu_end, nstop]
        vals = [v / c["np_"] for v in counts]
        errs = [np.sqrt(v) / c["np_"] for v in counts]
        tablerows.append(vals)
        x = np.arange(4) + (j - 0.5) * width
        ax.bar(x, vals, width=width, yerr=errs, capsize=3,
               color=c["color"], label=f"{label} ({c['np_']} p)")
        for xi, v in zip(x, vals):
            ax.annotate(f"{v:.3f}", xy=(xi, v), ha="center", va="bottom",
                        fontsize=8)
        c["depth"] = depth
    for i, s in enumerate(stages):
        print(f"{s.replace(chr(10),' '):34s} {tablerows[0][i]:12.4f} "
              f"{tablerows[1][i]:12.4f}")
    ax.set_xticks(np.arange(4))
    ax.set_xticklabels(stages, fontsize=9)
    ax.set_ylabel("per injected 3 GeV proton")
    ax.set_title("identical capture + 40 m channel + D-T stopping stage")
    ax.legend(fontsize=9)

    ax = axs[1]
    for label, c in CASES.items():
        w = np.full(len(c["depth"]), 1.0 / c["np_"])
        ax.hist(c["depth"], bins=12, range=(0, CELL / 10.0), weights=w,
                histtype="step", lw=2, color=c["color"], label=label)
    ax.set_xlabel("Bragg stop depth in liquid D-T [cm]")
    ax.set_ylabel("stopped $\\mu^-$ per proton per bin")
    ax.set_title("stop-depth profiles (normalized per proton)")
    ax.legend(fontsize=9)

    fig.suptitle("Recirculating internal-target FFA vs conventional "
                 "single-pass target (same beam, same downstream chain)",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig("plots/ring_vs_dump.png", dpi=150)
    print("wrote plots/ring_vs_dump.png")


if __name__ == "__main__":
    main()
