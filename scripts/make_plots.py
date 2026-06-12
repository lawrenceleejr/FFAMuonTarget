#!/usr/bin/env python3
"""
Make the standard plots/visualizations for the FFA internal-target setup.

Reads (whatever exists is plotted, the rest is skipped):
  out/probe_radial.txt, out/probe_chord.txt      <- probe_field.g4bl
  out/trace_co.txt, out/trace_betax.txt,
  out/trace_betay.txt                            <- ring_test.g4bl traces
  runs/job*/out/VD*.txt                          <- ffa_ring.g4bl production
  out/scan_summary.txt                           <- energy scan
  out/chan_trace.txt, out/VDch*.txt,
  out/VDmuEnd.txt, out/pions_for_channel.txt     <- decay channel

Writes plots/*.png

Usage: python3 scripts/make_plots.py [rundirs ...]
  (default run dirs: runs/job*/out)
"""

import glob
import math
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrow
from matplotlib.colors import TwoSlopeNorm

sys.path.insert(0, os.path.dirname(__file__))
from analyze_ring import read_bltrack                       # noqa: E402
from make_ffa_fieldmap import b0_midplane                   # noqa: E402

MP = 938.272
MPI = 139.570
MMU = 105.658

# ---- design constants (must match make_ffa_fieldmap.py defaults + k=3.6) ----
# defaults = the 1 GeV ring; the "3gev" preset rescales them (the lattice is
# scale-invariant: same fields/angles at R0 x 2.255)
N, R0, K = 12, 9000.0, 3.6
PC = 1696.04
BENDF, THF, THD, GAPFD = 48.0, 14.0, 3.5, 1.0
FRINGE = 100.0
RCO, RSTR, TREV = 8920.95, 8628.70, 212.3519
TARGET_AZ = 15.0
CAV_AZ = 195.0
KE0 = 1000.0
LAMTURNS = 78          # target interaction length in turns (lambda/TGT)
TLABEL = "1 GeV, 5 mm graphite"
PREFIX = "plots/"
TRACE_CO = "out/trace_co.txt"
RING_TRACE = "out/ring_trace.txt"

PRESETS = {
    "1gev": {},
    "3gev": dict(R0=20296.6, PC=3824.87, RCO=20119.41, RSTR=19459.11,
                 TREV=431.4673, FRINGE=225.5, KE0=3000.0, LAMTURNS=42,
                 TLABEL="3 GeV, 10 mm beryllium", PREFIX="plots/3gev_",
                 TRACE_CO="out/trace_co_3gev.txt",
                 RING_TRACE="out/ring_trace_3gev.txt"),
}


def design_P():
    cell = 360.0 / N
    bendD = BENDF - cell
    thD_lo = THF / 2.0 + GAPFD
    Brho = PC / 299.792458
    R0m = R0 / 1000.0
    BF = Brho * math.radians(BENDF) / (R0m * math.radians(THF))
    BD = Brho * math.radians(bendD) / (R0m * math.radians(2 * THD))
    return dict(R0=R0, r0=R0, k=K, BF=BF, BD=BD, thF_az=THF,
                thD_lo=thD_lo, thD_hi=thD_lo + THD, fringe=FRINGE)


def read_trace(fn):
    rows = []
    with open(fn) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            p = line.split()
            rows.append([float(v) for v in p[:11]])
    return np.array(rows)


def load_band_hits(dirs):
    out = {}
    for det in ("VDouter", "VDinner", "VDtop", "VDbot", "VDturn"):
        parts = []
        for i, d in enumerate(dirs):
            a = read_bltrack(f"{d}/{det}.txt")
            if len(a):
                a[:, 8] += i * 1000000
                parts.append(a)
        out[det] = np.vstack(parts) if parts else np.zeros((0, 12))
    return out


def ke_of(rows, mass=MP):
    p2 = np.sum(rows[:, 3:6] ** 2, axis=1)
    return np.sqrt(p2 + mass * mass) - mass


# ====================================================================
def fig_ring_layout(dirs):
    P = design_P()
    SCL = R0 / 9000.0
    n = 901
    xs = np.linspace(-10300 * SCL, 10300 * SCL, n)
    X, Z = np.meshgrid(xs, xs, indexing="ij")
    R = np.hypot(X, Z)
    PHI = np.degrees(np.arctan2(Z, X))
    TH = np.radians((PHI + 15.0) % 30.0 - 15.0)        # fold to one cell
    By = b0_midplane(R, TH, P)
    By[(R < 7950 * SCL) | (R > 9850 * SCL)] = np.nan

    fig, ax = plt.subplots(figsize=(11.5, 11))
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-2.0, vmax=3.0)
    im = ax.pcolormesh(xs / 1000, xs / 1000, By.T, cmap="RdBu_r", norm=norm,
                       rasterized=True)
    fig.colorbar(im, ax=ax, shrink=0.75, pad=0.02,
                 label=r"midplane $B_y$ [T]   (red = F, blue = reverse D)")

    # chamber / score walls
    th = np.linspace(0, 2 * np.pi, 400)
    for r in (9.65 * SCL, 8.25 * SCL):
        ax.plot(r * np.cos(th), r * np.sin(th), color="0.35", lw=1.2, ls="--")
    ax.annotate("score/kill walls", xy=(0.0, -9.7 * SCL), ha="center",
                color="0.35")

    # a recirculating proton (with target scattering), drawn faint
    if os.path.exists(RING_TRACE):
        tr = read_trace(RING_TRACE)
        prim = tr[tr[:, 9] == 1]
        if len(prim):
            best = max(np.unique(prim[:, 8]),
                       key=lambda ev: prim[prim[:, 8] == ev, 6].max())
            s = prim[prim[:, 8] == best]
            nt = s[:, 6].max() / TREV
            ax.plot(s[:, 0] / 1000, s[:, 2] / 1000, color="teal", lw=0.4,
                    alpha=0.55,
                    label=f"one proton, {nt:.0f} turns with target scattering")

    # closed orbit
    if os.path.exists(TRACE_CO):
        tr = read_trace(TRACE_CO)
        ax.plot(tr[:, 0] / 1000, tr[:, 2] / 1000, "k-", lw=1.4,
                label="closed orbit")

    # target
    a = math.radians(TARGET_AZ)
    ur = np.array([math.cos(a), math.sin(a)])
    c = RSTR / 1000 * ur
    ax.plot([c[0] - 0.15 * SCL * ur[0], c[0] + 0.15 * SCL * ur[0]],
            [c[1] - 0.15 * SCL * ur[1], c[1] + 0.15 * SCL * ur[1]],
            color="k", lw=5, solid_capstyle="butt")
    ax.annotate("internal target", xy=(c[0], c[1]),
                xytext=(c[0] - 3.6 * SCL, c[1] + 0.5 * SCL), fontsize=11,
                arrowprops=dict(arrowstyle="->", lw=1.2))

    # RF cavity
    a = math.radians(CAV_AZ)
    ur = np.array([math.cos(a), math.sin(a)])
    ut = np.array([-ur[1], ur[0]])
    c = RSTR / 1000 * ur
    corners = [c + sx * 0.288 * ur + sz * 0.15 * ut
               for sx, sz in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    corners = np.array(corners + [corners[0]])
    ax.plot(corners[:, 0], corners[:, 1], color="purple", lw=2)
    ax.annotate("RF cavity", xy=(c[0], c[1]),
                xytext=(c[0] + 1.1 * SCL, c[1] - 1.7 * SCL), fontsize=11,
                color="purple",
                arrowprops=dict(arrowstyle="->", color="purple", lw=1.2))

    # injection arrow at azimuth 0
    ax.add_patch(FancyArrow(RCO / 1000, -1.45 * SCL, 0, 1.0 * SCL,
                            width=0.05 * SCL, head_width=0.22 * SCL,
                            head_length=0.3 * SCL, color="green"))
    ax.annotate("p injection", xy=(6.3 * SCL, -2.5 * SCL),
                fontsize=11, color="green", ha="center")

    # pi-/mu- exits from production data
    hits = load_band_hits(dirs)
    allb = np.vstack([hits[d] for d in ("VDouter", "VDinner", "VDtop", "VDbot")
                      if len(hits[d])])
    if len(allb):
        pim = allb[np.isin(allb[:, 7], (-211.0, 13.0))]
        ax.plot(pim[:, 0] / 1000, pim[:, 2] / 1000, "o", ms=7,
                mfc="orange", mec="k", mew=0.6, ls="none",
                label=r"$\pi^-/\mu^-$ chamber exits (sim)")
        # capture channel direction (outward-going mean)
        ur = pim[:, 0:3].copy()
        ur[:, 1] = 0
        ur /= np.linalg.norm(ur, axis=1)[:, None]
        sel = np.sum(pim[:, 3:6] * ur, axis=1) > 0
        if np.any(sel):
            ez = np.mean(pim[sel, 3:6] / np.linalg.norm(pim[sel, 3:6],
                         axis=1)[:, None], axis=0)
            ez[1] = 0
            ez /= np.linalg.norm(ez)
            o = np.mean(pim[sel, 0:3], axis=0) / 1000
            ax.add_patch(FancyArrow(o[0], o[2], 1.6 * SCL * ez[0],
                                    1.6 * SCL * ez[2], width=0.07 * SCL,
                                    head_width=0.3 * SCL,
                                    head_length=0.4 * SCL, color="orange",
                                    alpha=0.9))
            ax.annotate("to capture solenoid,\ndecay channel + DT target",
                        xy=(o[0] + 1.4 * SCL * ez[0], o[2] + 1.4 * SCL * ez[2]),
                        xytext=(0.8 * SCL, 5.9 * SCL), fontsize=11,
                        color="darkorange", ha="center",
                        arrowprops=dict(arrowstyle="-", color="darkorange",
                                        lw=0.8, alpha=0.6))

    # cell boundaries (straights)
    for i in range(12):
        a = math.radians(15 + 30 * i)
        ax.plot([8.1 * SCL * math.cos(a), 9.8 * SCL * math.cos(a)],
                [8.1 * SCL * math.sin(a), 9.8 * SCL * math.sin(a)],
                color="0.6", lw=0.5, ls=":")

    ax.set_xlabel("x [m]")
    ax.set_ylabel("z [m]")
    ax.set_title(f"FFA internal-target ring (k=3.6, R0={R0/1000:.1f} m), "
                 f"{TLABEL}\nERIT-style energy-recovery pion source",
                 fontsize=13)
    ax.set_aspect("equal")
    ax.set_xlim(-10.4 * SCL, 10.4 * SCL)
    ax.set_ylim(-10.4 * SCL, 10.4 * SCL)
    ax.legend(loc="upper left", fontsize=11, framealpha=0.95)
    fig.tight_layout()
    fig.savefig(PREFIX + "ring_layout.png", dpi=150)
    plt.close(fig)
    print("wrote " + PREFIX + "ring_layout.png")


# ====================================================================
def fig_lattice_optics():
    from find_closed_orbit import orbit_samples
    fig, axs = plt.subplots(2, 2, figsize=(12.5, 9))

    # (a) radial field law
    ax = axs[0, 0]
    if os.path.exists("out/probe_radial.txt"):
        d = np.loadtxt("out/probe_radial.txt")
        ax.plot(d[:-1, 0] / 1000, d[:-1, 5], "o", ms=4, label="g4bl field map")
        rr = np.linspace(8.0, 9.79, 100)
        P = design_P()
        ax.plot(rr, P["BF"] * (rr * 1000 / R0) ** K, "r-", lw=1,
                label=rf"$B_F\,(r/r_0)^k$,  k={K}")
        ax.axvline(RCO / 1000, color="g", ls="--", lw=1)
        ax.annotate("closed orbit", xy=(RCO / 1000, 1.45), rotation=90,
                    fontsize=9, color="g", ha="right")
        ax.set_xlabel("radius r [m]")
        ax.set_ylabel(r"$B_y$ at F-magnet center [T]")
        ax.legend()
        ax.set_title("scaling field law (cell center, midplane)")

    # (b) azimuthal profile
    ax = axs[0, 1]
    if os.path.exists("out/probe_chord.txt"):
        d = np.loadtxt("out/probe_chord.txt")
        thd = np.degrees(np.arctan2(d[:, 2], d[:, 0]))
        ax.plot(thd, d[:, 5], "o-", ms=3, lw=0.8)
        ax.axhline(0, color="0.5", lw=0.5)
        ax.set_ylim(-2.3, 2.75)
        for x0, x1, c, lab in ((-7, 7, "red", "F (+48°)"),
                               (8, 11.5, "blue", "D (−9°)"),
                               (-11.5, -8, "blue", None)):
            ax.axvspan(x0, x1, color=c, alpha=0.12)
            if lab:
                ax.annotate(lab, xy=((x0 + x1) / 2, 2.35), ha="center",
                            fontsize=10)
        ax.set_xlabel("azimuth within cell [deg]")
        ax.set_ylabel(r"$B_y$ [T] at x = 8.9 m chord")
        ax.set_title("cell structure: DFD triplet with reverse-bending D")

    # (c) closed-orbit scallop
    ax = axs[1, 0]
    if os.path.exists("out/trace_co.txt"):
        tr = read_trace("out/trace_co.txt")
        phi = np.degrees(np.unwrap(np.arctan2(tr[:, 2], tr[:, 0])))
        r = np.hypot(tr[:, 0], tr[:, 2])
        s = phi <= 362
        ax.plot(phi[s], r[s] / 1000, "k-", lw=1.2)
        ax.axhline(RSTR / 1000, color="b", ls=":", lw=1.2,
                   label="straights: 8.629 m (target/RF)")
        ax.axhline(RCO / 1000, color="r", ls=":", lw=1.2,
                   label="F centers: 8.921 m")
        ax.axvline(TARGET_AZ, color="k", lw=4, alpha=0.35, label="target azimuth")
        ax.legend(loc="center", fontsize=8, framealpha=0.92)
        ax.set_xlabel("ring azimuth [deg]")
        ax.set_ylabel("orbit radius [m]")
        ax.set_title("closed orbit over one turn (scalloped)")

    # (d) betatron oscillations
    ax = axs[1, 1]
    for fn, col, lab0 in (("out/trace_betax.txt", "tab:red", "x"),
                          ("out/trace_betay.txt", "tab:blue", "y")):
        if not os.path.exists(fn):
            continue
        tr = read_trace(fn)
        dat = np.zeros((len(tr), 7))
        dat[:, 0], dat[:, 1], dat[:, 2], dat[:, 6] = tr[:, 0], tr[:, 1], tr[:, 2], tr[:, 6]
        rs, ts, ys, _, _, _ = orbit_samples(dat)
        seq = rs - np.mean(rs) if lab0 == "x" else ys
        nu = measured_tune(seq)
        cell = np.arange(1, len(seq) + 1)
        ax.plot(cell / 12.0, seq, color=col, lw=1,
                label=f"{lab0}: Q={12*nu:.2f}")
    ax.set_xlabel("turn number")
    ax.set_ylabel("displacement at cell samples [mm]")
    ax.set_title("betatron oscillations about the closed orbit")
    ax.legend()

    fig.suptitle("FFA lattice optics (tracked with G4beamline)", fontsize=13)
    fig.tight_layout()
    fig.savefig("plots/lattice_optics.png", dpi=150)
    plt.close(fig)
    print("wrote plots/lattice_optics.png")


def measured_tune(seq):
    seq = np.asarray(seq) - np.mean(seq)
    n = len(seq)
    ft = np.abs(np.fft.rfft(seq * np.hanning(n)))
    i = int(np.argmax(ft[1:])) + 1
    if 1 <= i < len(ft) - 1:
        a, b, c = ft[i - 1], ft[i], ft[i + 1]
        d = 0.5 * (a - c) / (a - 2 * b + c)
    else:
        d = 0.0
    return (i + d) / n


# ====================================================================
def fig_energy_recovery(dirs):
    hits = load_band_hits(dirs)
    turn = hits["VDturn"]
    prim = turn[(turn[:, 7] == 2212) & (turn[:, 9] == 1)]
    nev = len(np.unique(prim[:, 8]))
    if nev == 0:
        print("no VDturn data, skipping energy_recovery")
        return

    # group by event, order crossings by time
    ke_by_turn, x_by_turn = {}, {}
    nturns = []
    for ev in np.unique(prim[:, 8]):
        sel = prim[prim[:, 8] == ev]
        sel = sel[np.argsort(sel[:, 6])]
        ke = ke_of(sel)
        nturns.append(len(sel))
        for i in range(len(sel)):
            ke_by_turn.setdefault(i, []).append(ke[i])
            x_by_turn.setdefault(i, []).append(sel[i, 0])
    nturns = np.array(nturns)

    fig, axs = plt.subplots(2, 2, figsize=(12.5, 9))

    # (a) mean KE vs turn
    ax = axs[0, 0]
    tmax = max(k for k, v in ke_by_turn.items() if len(v) >= 5)
    ts = np.arange(tmax + 1)
    mean = np.array([np.mean(ke_by_turn[i]) for i in ts])
    std = np.array([np.std(ke_by_turn[i]) for i in ts])
    nsur = np.array([len(ke_by_turn[i]) for i in ts])
    ax.fill_between(ts, mean - std, mean + std, alpha=0.3,
                    label=r"$\pm 1\sigma$ (straggling)")
    ax.plot(ts, mean, lw=1.5, label="mean kinetic energy")
    ax.axhline(KE0, color="0.4", ls=":", lw=1)
    ax.set_xlabel("turn")
    ax.set_ylabel("primary proton KE [MeV]")
    ax.set_title("energy recovery: the cavity restores the per-turn "
                 "target loss")
    ax2 = ax.twinx()
    ax2.plot(ts, nsur, color="0.45", lw=1, ls="--")
    ax2.set_yscale("log")
    ax2.set_ylabel("protons still circulating (gray dashed)", color="0.45")
    ax2.tick_params(axis="y", colors="0.45")
    ax.annotate("late drift = a few out-of-bucket\nsurvivors, still held by the\nFFA's huge momentum acceptance",
                xy=(190, 940), fontsize=8.5, color="0.25")
    ax.legend(loc="lower left")

    # (b) survival
    ax = axs[0, 1]
    surv = np.array([np.sum(nturns > i) for i in ts]) / nev
    ax.semilogy(ts, surv, lw=1.5, label="simulated")
    m = (ts > 5) & (ts < 150) & (surv > 0)
    lam = -1.0 / np.polyfit(ts[m], np.log(surv[m] + 1e-12), 1)[0]
    ax.semilogy(ts, np.exp(-ts / lam) * np.exp(np.polyfit(ts[m],
                np.log(surv[m]), 1)[1]), "r--", lw=1,
                label=f"exp fit: 1/e = {lam:.0f} turns")
    ax.semilogy(ts, np.exp(-ts / LAMTURNS), color="0.5", ls=":",
                label=rf"nuclear $\lambda_{{inel}}$ alone ({LAMTURNS} turns)")
    ax.set_xlabel("turn")
    ax.set_ylabel("fraction still circulating")
    ax.set_title(f"recirculation survival ({nev} protons)")
    ax.legend()

    # (c) transverse blow-up at the turn counter
    ax = axs[1, 0]
    xs = np.array([np.std(x_by_turn[i]) for i in ts])
    ax.plot(ts, xs, lw=1.5)
    ax.set_xlabel("turn")
    ax.set_ylabel("rms beam size at turn counter [mm]")
    ax.set_title("multiple-scattering emittance growth\n(FFA acceptance "
                 "absorbs it: chamber half-width ~600 mm)")

    # (d) KE distributions at sample turns
    ax = axs[1, 1]
    for tn, col in ((1, "C0"), (20, "C1"), (50, "C2"), (100, "C3")):
        if tn in ke_by_turn and len(ke_by_turn[tn]) > 3:
            ax.hist(ke_by_turn[tn], bins=np.linspace(KE0 - 40, KE0 + 40, 41),
                    histtype="step", lw=1.5, color=col, density=True,
                    label=f"turn {tn} (n={len(ke_by_turn[tn])})")
    ax.set_xlabel("primary proton KE [MeV]")
    ax.set_ylabel("normalized")
    ax.set_title("straggling spreads the bucket; mean stays pinned")
    ax.legend(fontsize=9)

    fig.suptitle("Beam dynamics with the internal target + RF "
                 f"({nev} protons, {TLABEL})", fontsize=13)
    fig.tight_layout()
    fig.savefig(PREFIX + "energy_recovery.png", dpi=150)
    plt.close(fig)
    print("wrote " + PREFIX + "energy_recovery.png")


# ====================================================================
def fig_pion_yield(dirs):
    hits = load_band_hits(dirs)
    allb = np.vstack([hits[d] for d in ("VDouter", "VDinner", "VDtop", "VDbot")
                      if len(hits[d])])
    turn = hits["VDturn"]
    prim = turn[(turn[:, 7] == 2212) & (turn[:, 9] == 1)]
    nev = max(len(np.unique(prim[:, 8])), 1)

    fig, axs = plt.subplots(2, 2, figsize=(12.5, 9))

    # (a) momentum spectra
    ax = axs[0, 0]
    bins = np.linspace(0, 800 if KE0 < 1500 else 1600, 21)
    for pid, mass, col, lab in ((-211, MPI, "tab:red", r"$\pi^-$"),
                                (13, MMU, "tab:blue", r"$\mu^-$"),
                                (211, MPI, "0.6", r"$\pi^+$ (for comparison)")):
        sel = allb[allb[:, 7] == pid]
        if len(sel) == 0:
            continue
        p = np.sqrt(np.sum(sel[:, 3:6] ** 2, axis=1))
        w = np.full(len(p), 1.0 / nev)
        ax.hist(p, bins=bins, weights=w, histtype="step", lw=1.8,
                color=col, label=f"{lab}  ({len(sel)/nev:.3f}/p)")
    ax.set_xlabel("momentum at chamber wall [MeV/c]")
    ax.set_ylabel("per injected proton per bin")
    ax.set_title("spectra of particles leaving the ring")
    ax.legend()

    # (b) exit azimuth
    ax = axs[0, 1]
    bins = np.linspace(-180, 180, 37)
    for pid, col, lab in ((-211, "tab:red", r"$\pi^-$"),
                          (211, "0.6", r"$\pi^+$"),
                          (2112, "tab:green", "n")):
        sel = allb[allb[:, 7] == pid]
        if len(sel) == 0:
            continue
        phi = (np.degrees(np.arctan2(sel[:, 2], sel[:, 0])) - TARGET_AZ
               + 180.0) % 360.0 - 180.0
        ax.hist(phi, bins=bins, histtype="step", lw=1.8, color=col,
                density=True, label=lab)
    ax.axvline(0, color="k", lw=3, alpha=0.3)
    ax.annotate("target", xy=(4, ax.get_ylim()[1] * 0.9), fontsize=10)
    ax.set_xlabel("exit azimuth relative to target [deg]")
    ax.set_ylabel("normalized")
    ax.set_title("everything escapes near the target")
    ax.legend()

    # (c) species budget
    ax = axs[1, 0]
    names = {2212: "p", 2112: "n", 211: r"$\pi^+$", -211: r"$\pi^-$",
             -13: r"$\mu^+$", 13: r"$\mu^-$", 1000010020: "d"}
    vals, labs = [], []
    for pid, lab in names.items():
        cnt = int(np.sum(allb[:, 7] == pid))
        if cnt:
            vals.append(cnt / nev)
            labs.append(lab)
    order = np.argsort(vals)[::-1]
    ax.bar(range(len(vals)), [vals[i] for i in order],
           color=["tab:red" if labs[i] in (r"$\pi^-$", r"$\mu^-$")
                  else "0.7" for i in order])
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels([labs[i] for i in order])
    ax.set_yscale("log")
    ax.set_ylabel("per injected proton")
    ax.set_title("flux on the chamber walls by species")
    for j, i in enumerate(order):
        ax.annotate(f"{vals[i]:.3g}", xy=(j, vals[i]), ha="center",
                    va="bottom", fontsize=9)

    # (d) exit map near target
    ax = axs[1, 1]
    SCL = R0 / 9000.0
    th = np.linspace(np.radians(-30), np.radians(75), 200)
    ax.plot(9.65 * SCL * np.cos(th), 9.65 * SCL * np.sin(th), "0.5", ls="--", lw=1)
    ax.plot(8.25 * SCL * np.cos(th), 8.25 * SCL * np.sin(th), "0.5", ls="--", lw=1)
    if os.path.exists(TRACE_CO):
        tr = read_trace(TRACE_CO)
        ax.plot(tr[:, 0] / 1000, tr[:, 2] / 1000, "k-", lw=1)
    for pid, col, ms, lab in ((2112, "tab:green", 3, "n"),
                              (211, "0.6", 4, r"$\pi^+$"),
                              (-211, "tab:red", 7, r"$\pi^-$"),
                              (13, "tab:blue", 7, r"$\mu^-$")):
        sel = allb[allb[:, 7] == pid]
        if len(sel):
            ax.plot(sel[:, 0] / 1000, sel[:, 2] / 1000, "o", ms=ms,
                    color=col, ls="none", label=lab,
                    alpha=0.5 if pid in (2112, 211) else 1.0)
    a = math.radians(TARGET_AZ)
    ax.plot(RSTR / 1000 * math.cos(a), RSTR / 1000 * math.sin(a), "k*",
            ms=16, label="target")
    ax.set_aspect("equal")
    ax.set_xlim(5.4 * SCL, 10.2 * SCL)
    ax.set_ylim(-1.5 * SCL, 7.2 * SCL)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("z [m]")
    ax.set_title("exit positions on the chamber (top view)")
    ax.legend(loc="lower left", fontsize=9)

    fig.suptitle(f"Pion production in the ring ({nev} injected protons, "
                 f"{TLABEL})", fontsize=13)
    fig.tight_layout()
    fig.savefig(PREFIX + "pion_yield.png", dpi=150)
    plt.close(fig)
    print("wrote " + PREFIX + "pion_yield.png")


# ====================================================================
def fig_energy_scan(ring_yield=None):
    if not os.path.exists("out/scan_summary.txt"):
        print("no scan summary, skipping energy_scan")
        return
    d = np.loadtxt("out/scan_summary.txt")
    ke, pim, pip = d[:, 0] / 1000, d[:, 1], d[:, 2]
    pint, pgev = d[:, 6], d[:, 7]

    fig, axs = plt.subplots(1, 2, figsize=(12.5, 5))
    ax = axs[0]
    ax.plot(ke, pim, "o-", color="tab:red", label=r"$\pi^-$/proton (20 mm, single pass)")
    ax.plot(ke, pip, "s-", color="0.6", label=r"$\pi^+$/proton")
    ax.plot([1.0], [0.027], "*", ms=18, color="tab:red", mec="k",
            label="FFA ring @1 GeV (5 mm C): 0.027/p")
    ax.plot([3.0], [0.270], "*", ms=20, color="tab:purple", mec="k",
            label="FFA ring @3 GeV (10 mm Be): 0.270/p")
    ax.plot([1.0], [0.029], "D", ms=9, color="tab:orange", mec="k",
            label="45 cm dump @1 GeV: 0.029/p")
    ax.set_xlabel("proton kinetic energy [GeV]")
    ax.set_ylabel("yield per proton")
    ax.set_yscale("log")
    ax.set_ylim(5e-4, 0.5)
    ax.set_title("pion yield from graphite vs beam energy")
    ax.legend(fontsize=9, loc="upper left")

    ax = axs[1]
    ax.plot(ke, pint, "o-", color="tab:red",
            label=r"$\pi^-$ per inelastic interaction")
    ax.plot(ke, pgev * 10, "s-", color="tab:purple",
            label=r"$\pi^-$/proton/GeV $\times$10 (beam-power FOM)")
    ax.axvline(1.0, color="g", ls="--", lw=1)
    ax.annotate("injector\n(1 GeV)", xy=(1.03, 0.45), color="g", fontsize=10)
    ax.set_xlabel("proton kinetic energy [GeV]")
    ax.set_title("figures of merit: the ring makes every proton\n"
                 "interact at full energy (right-hand curve)")
    ax.legend(fontsize=9, loc="upper left")

    fig.tight_layout()
    fig.savefig("plots/energy_scan.png", dpi=150)
    plt.close(fig)
    print("wrote plots/energy_scan.png")


# ====================================================================
def fig_decay_channel():
    have_trace = os.path.exists("out/chan_trace.txt")
    fig, axs = plt.subplots(2, 1, figsize=(12.5, 8.5),
                            gridspec_kw={"height_ratios": [2, 1]})

    ax = axs[0]
    if have_trace:
        tr = read_trace("out/chan_trace.txt")
        # columns: x y z Px Py Pz t PDGid EventID TrackID
        for (ev, tid) in {(int(r[8]), int(r[9])) for r in tr}:
            s = tr[(tr[:, 8] == ev) & (tr[:, 9] == tid)]
            pid = int(s[0, 7])
            if pid == -211:
                col, lw, lab = "tab:red", 1.0, r"$\pi^-$"
            elif pid == 13:
                col, lw, lab = "tab:blue", 1.0, r"$\mu^-$"
            else:
                continue
            rho = np.hypot(s[:, 0], s[:, 1]) / 1000.0
            ax.plot(s[:, 2] / 1000, np.sign(s[:, 0]) * rho, color=col, lw=lw)
        ax.plot([], [], color="tab:red", label=r"$\pi^-$")
        ax.plot([], [], color="tab:blue", label=r"$\mu^-$ (after decay kink)")
        ax.annotate("tracks with $p_T \\gtrsim$ 350 MeV/c hit the wall\n"
                    "within the first turn at the entrance",
                    xy=(0.4, -0.32), fontsize=9, color="0.3")
    for ysgn in (1, -1):
        ax.axhline(ysgn * 0.5, color="k", lw=2)
    ax.annotate("solenoid bore wall (r = 0.5 m), $B_z$ = 5 T", xy=(8, 0.53),
                fontsize=10)
    ax.set_xlabel("z along channel [m]")
    ax.set_ylabel("signed radius [m]")
    ax.set_xlim(0, 30.5)
    ax.set_ylim(-0.75, 0.75)
    ax.set_title(r"captured $\pi^-$ spiralling in the decay solenoid "
                 "(traced tracks)")
    ax.legend(loc="lower right")

    # populations vs z
    ax = axs[1]
    zs, npi, nmu = [0], [None], [None]
    nin = 0
    if os.path.exists("out/pions_for_channel.txt"):
        with open("out/pions_for_channel.txt") as f:
            rows = [l.split() for l in f if not l.startswith("#") and l.strip()]
        nin = len(rows)
        npi[0] = sum(1 for r in rows if int(r[7]) == -211)
        nmu[0] = sum(1 for r in rows if int(r[7]) == 13)
    for det, z in (("VDch1", 5), ("VDch2", 10), ("VDch3", 15), ("VDch4", 20),
                   ("VDmuEnd", 30)):
        a = read_bltrack(f"out/{det}.txt")
        zs.append(z)
        npi.append(int(np.sum(a[:, 7] == -211)) if len(a) else 0)
        nmu.append(int(np.sum(a[:, 7] == 13)) if len(a) else 0)
    ax.step(zs, npi, where="post", color="tab:red", lw=2, label=r"$\pi^-$")
    ax.step(zs, nmu, where="post", color="tab:blue", lw=2, label=r"$\mu^-$")
    ax.set_xlabel("z along channel [m]")
    ax.set_ylabel("count")
    ax.set_xlim(0, 30.5)
    ax.set_title(rf"$\pi^- \to \mu^-$ conversion along the channel "
                 f"({nin} tracks captured from the ring)")
    ax.legend()

    fig.tight_layout()
    fig.savefig("plots/decay_channel.png", dpi=150)
    plt.close(fig)
    print("wrote plots/decay_channel.png")




# ====================================================================
def fig_stopping(nprotons, dirs=None, lch=40000.0, deg=1.0, cell=3000.0, rch=600.0):
    """mu- stopping (Bragg) in the DT target downstream of the channel"""
    if not os.path.exists("out/chan_ends.txt"):
        print("no out/chan_ends.txt, skipping stopping figure")
        return
    ends = read_bltrack("out/chan_ends.txt")
    p = np.sqrt(np.sum(ends[:, 3:6] ** 2, axis=1))
    mu = ends[:, 7] == 13
    stopped = mu & (p < 1.0)
    z0 = lch + 100.0 + deg
    in_cell = stopped & (ends[:, 2] >= z0) & (ends[:, 2] <= z0 + cell)

    fig, axs = plt.subplots(2, 2, figsize=(12.5, 9))

    # (a) mu- momentum at channel end vs captured pi- momentum
    ax = axs[0, 0]
    pin = []
    if os.path.exists("out/pions_for_channel.txt"):
        for line in open("out/pions_for_channel.txt"):
            if line.startswith("#") or not line.strip():
                continue
            q = line.split()
            pin.append(math.sqrt(float(q[3])**2 + float(q[4])**2 + float(q[5])**2))
    endvd = read_bltrack("out/VDmuEnd.txt")
    if len(pin):
        ax.hist(pin, bins=np.linspace(0, 1600, 33), histtype="step", lw=1.8,
                color="tab:red", label=rf"$\pi^-$ captured ({len(pin)})")
    if len(endvd):
        pmu = np.sqrt(np.sum(endvd[endvd[:, 7] == 13][:, 3:6] ** 2, axis=1))
        ax.hist(pmu, bins=np.linspace(0, 1600, 33), histtype="step", lw=1.8,
                color="tab:blue", label=rf"$\mu^-$ at channel end ({len(pmu)})")
    ax.set_xlabel("momentum [MeV/c]")
    ax.set_ylabel("count")
    ax.set_title("decay channel: spectrum entering vs delivered")
    ax.legend()

    # (b) Bragg stop-depth distribution in the DT target
    ax = axs[0, 1]
    depth = (ends[in_cell, 2] - z0) / 10.0
    ax.hist(depth, bins=24, range=(0, cell / 10.0), color="tab:blue",
            alpha=0.75)
    ax.set_xlabel("stop depth in liquid D-T [cm]")
    ax.set_ylabel(r"$\mu^-$ stops")
    med = np.median(depth) if len(depth) else 0
    ax.set_title(f"Bragg stop profile in the D-T target "
                 f"({int(in_cell.sum())} stops, median {med:.0f} cm)")
    ax2 = ax.twinx()
    if len(depth):
        ds = np.sort(depth)
        ax2.plot(ds, np.arange(1, len(ds) + 1) / len(ds), "k--", lw=1)
    ax2.set_ylabel("cumulative fraction")

    # (c) transverse stop positions in the cell
    ax = axs[1, 0]
    if np.any(in_cell):
        ax.plot(ends[in_cell, 0], ends[in_cell, 1], "o", color="tab:blue",
                ms=6)
    th = np.linspace(0, 2 * np.pi, 100)
    ax.plot(rch * np.cos(th), rch * np.sin(th), "k-", lw=2)
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_aspect("equal")
    ax.set_title("stop positions (transverse), D-T vessel wall")

    # (d) the chain budget per proton
    ax = axs[1, 1]
    npim = 0
    dirs = dirs or []
    labels, vals = [], []
    if nprotons:
        # escaped pi-+mu- comes from the analysis (passed via file count)
        try:
            hits = load_band_hits(dirs)
            allb = np.vstack([hits[d] for d in
                              ("VDouter", "VDinner", "VDtop", "VDbot")
                              if len(hits[d])])
            npim = int(np.sum(np.isin(allb[:, 7], (-211.0, 13.0))))
        except Exception:
            npim = 0
        nmu_end = int(np.sum(endvd[:, 7] == 13)) if len(endvd) else 0
        for lab, v in ((r"$\pi^-+\mu^-$ out of ring", npim),
                       (r"captured into channel", len(pin)),
                       (r"$\mu^-$ at channel end", nmu_end),
                       (r"$\mu^-$ STOPPED in D-T", int(in_cell.sum()))):
            labels.append(lab)
            vals.append(v / nprotons)
        ax.bar(range(len(vals)), vals, color=["0.6", "0.6", "0.6", "tab:blue"])
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(labels, fontsize=9, rotation=12)
        ax.set_ylabel("per injected proton")
        for j, v in enumerate(vals):
            ax.annotate(f"{v:.3f}", xy=(j, v), ha="center", va="bottom")
        ax.set_title(f"chain budget ({nprotons} protons)")

    fig.suptitle("Muon stopping stage: capture solenoid -> 40 m decay "
                 "channel -> liquid D-T target (no degrader)", fontsize=13)
    fig.tight_layout()
    fig.savefig("plots/stopping_dt.png", dpi=150)
    plt.close(fig)
    print("wrote plots/stopping_dt.png")


# ====================================================================
def main():
    ap = __import__("argparse").ArgumentParser()
    ap.add_argument("dirs", nargs="*", default=None)
    ap.add_argument("--preset", choices=list(PRESETS), default="1gev")
    ap.add_argument("--nprotons", type=int, default=0,
                    help="protons behind the channel/stopping data")
    args = ap.parse_args()
    globals().update(PRESETS[args.preset])
    dirs = args.dirs or sorted(glob.glob("runs/job*/out"))
    os.makedirs("plots", exist_ok=True)
    print(f"preset {args.preset}, run dirs: {len(dirs)}")

    # ring pi- yield for the scan figure
    ring_yield = None
    hits = load_band_hits(dirs)
    allb = np.vstack([hits[d] for d in ("VDouter", "VDinner", "VDtop", "VDbot")
                      if len(hits[d])])
    prim = hits["VDturn"]
    prim = prim[(prim[:, 7] == 2212) & (prim[:, 9] == 1)] if len(prim) else prim
    nev = len(np.unique(prim[:, 8])) if len(prim) else 0
    if nev and len(allb):
        ring_yield = float(np.sum(np.isin(allb[:, 7], (-211.0, 13.0)))) / nev

    fig_ring_layout(dirs)
    if args.preset == "1gev":
        fig_lattice_optics()       # probe/trace files are for the 1 GeV ring
        fig_decay_channel()
    fig_energy_recovery(dirs)
    fig_pion_yield(dirs)
    fig_energy_scan(ring_yield)
    if args.nprotons:
        fig_stopping(args.nprotons, dirs=dirs)


if __name__ == "__main__":
    main()
