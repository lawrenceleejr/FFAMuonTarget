#!/usr/bin/env python3
"""
Test the hypothesis: does the FFA recirculating internal target give a
NARROWER outgoing pi-/mu- energy spread than a conventional single-pass
thick target?  Compares, at the same 3 GeV beam:
  - escaping pi- momentum spectrum   (ring runs vs dump VD box)
  - escaping mu- momentum spectrum
  - delivered mu- at the 40 m channel end
  - Bragg stop-depth spread in the D-T target
Prints spread metrics and writes plots/energy_spread.png.
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

MMU = 105.658


def stack(paths):
    a = [read_bltrack(p) for p in paths]
    a = [x for x in a if len(x)]
    return np.vstack(a) if a else np.zeros((0, 12))


def pofsel(arr, pid):
    s = arr[arr[:, 7] == pid]
    if len(s) == 0:
        return np.array([])
    return np.sqrt(np.sum(s[:, 3:6] ** 2, axis=1))


def metrics(p):
    if len(p) < 3:
        return None
    q = np.percentile(p, [10, 50, 90])
    return dict(n=len(p), mean=p.mean(), std=p.std(),
                frac=p.std() / p.mean(), med=q[1],
                p10=q[0], p90=q[2], w1090=(q[2] - q[0]) / q[1])


def show(name, p):
    m = metrics(p)
    if not m:
        print(f"  {name:26s}  (n<3)")
        return
    print(f"  {name:26s}  n={m['n']:4d}  mean={m['mean']:6.0f}  "
          f"sigma={m['std']:6.0f}  sigma/p={m['frac']:.2f}  "
          f"med={m['med']:6.0f}  10-90/med={m['w1090']:.2f}")


def main():
    ring_dirs = sorted(glob.glob("runs/job2[1-9]/out")) + \
        sorted(glob.glob("runs/job3[0-6]/out"))
    ring = stack([f"{d}/{n}.txt" for d in ring_dirs
                  for n in ("VDouter", "VDinner", "VDtop", "VDbot")])
    dump = stack(glob.glob("out/dump_C600/VD*.txt"))

    print("ESCAPING pi- momentum [MeV/c]:")
    show("FFA ring (10 mm Be)", pofsel(ring, -211))
    show("dump (60 cm C)", pofsel(dump, -211))
    print("ESCAPING mu- momentum [MeV/c]:")
    show("FFA ring", pofsel(ring, 13))
    show("dump", pofsel(dump, 13))

    end_r = read_bltrack("out/VDmuEnd_ring.txt")
    end_d = read_bltrack("out/VDmuEnd_dump.txt")
    print("DELIVERED mu- at 40 m channel end [MeV/c]:")
    show("FFA ring", pofsel(end_r, 13))
    show("dump", pofsel(end_d, 13))

    # Bragg stop depths
    def depths(fn):
        e = read_bltrack(fn)
        p = np.sqrt(np.sum(e[:, 3:6] ** 2, axis=1))
        z0 = 40000 + 100 + 1
        ok = (e[:, 7] == 13) & (p < 1) & (e[:, 2] >= z0) & (e[:, 2] <= z0 + 3000)
        return (e[ok, 2] - z0) / 10.0
    dr, dd = depths("out/chan_ends_ring.txt"), depths("out/chan_ends_dump.txt")
    print("STOP DEPTH in D-T [cm]:")
    for nm, d in (("FFA ring", dr), ("dump", dd)):
        if len(d):
            print(f"  {nm:26s}  n={len(d):4d}  mean={d.mean():5.1f}  "
                  f"sigma={d.std():5.1f}  sigma/d={d.std()/d.mean():.2f}")

    # ---- plot ----
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.6))
    pb = np.linspace(0, 1600, 33)
    ax = axs[0]
    for arr, c, lab in ((ring, "tab:purple", "FFA ring"),
                        (dump, "tab:orange", "dump 60 cm C")):
        p = pofsel(arr, -211)
        if len(p):
            ax.hist(p, bins=pb, density=True, histtype="step", lw=2, color=c,
                    label=f"{lab} (sigma/p={metrics(p)['frac']:.2f})")
    ax.set_xlabel("escaping $\\pi^-$ momentum [MeV/c]")
    ax.set_ylabel("normalized")
    ax.set_title("(a) escaping $\\pi^-$ spectrum")
    ax.legend(fontsize=9)

    ax = axs[1]
    for p, c, lab in ((pofsel(end_r, 13), "tab:purple", "FFA ring"),
                      (pofsel(end_d, 13), "tab:orange", "dump")):
        if len(p):
            ax.hist(p, bins=pb, density=True, histtype="step", lw=2, color=c,
                    label=f"{lab} (sigma/p={metrics(p)['frac']:.2f})")
    ax.set_xlabel("delivered $\\mu^-$ momentum [MeV/c]")
    ax.set_title("(b) $\\mu^-$ at channel end")
    ax.legend(fontsize=9)

    ax = axs[2]
    db = np.linspace(0, 300, 16)
    for d, c, lab in ((dr, "tab:purple", "FFA ring"),
                      (dd, "tab:orange", "dump")):
        if len(d):
            ax.hist(d, bins=db, density=True, histtype="step", lw=2, color=c,
                    label=f"{lab} (sigma/d={d.std()/d.mean():.2f})")
    ax.set_xlabel("Bragg stop depth in D-T [cm]")
    ax.set_title("(c) where the $\\mu^-$ stop")
    ax.legend(fontsize=9)

    fig.suptitle("Energy/range spread: FFA recirculating target vs single-pass "
                 "thick target (3 GeV)", fontsize=13)
    fig.tight_layout()
    fig.savefig("plots/energy_spread.png", dpi=150)
    print("wrote plots/energy_spread.png")


if __name__ == "__main__":
    main()
