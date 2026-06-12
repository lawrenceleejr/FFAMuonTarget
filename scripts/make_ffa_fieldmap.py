#!/usr/bin/env python3
"""
Generate a BLFieldMap (G4beamline 'fieldmap' element) for ONE cell of a
radial-sector scaling FFA, to be placed N times (rotated about the vertical
axis through the ring center = global origin) to form the full ring.

Lattice concept (ERIT-style internal-target ring):
  - N identical DFD triplet cells, ring center at the global origin,
    median plane y=0, beam circulates in the x-z plane.
  - Scaling-FFA field law on the midplane:
        By(r, theta) = (r/r0)^k * [ BF*gF(theta) - BD*gD(theta) ]
    where gF/gD are tanh-edged azimuthal profiles of the F and (reverse
    bending) D sectors.  The (r/r0)^k dependence makes orbits of every
    momentum geometrically similar (huge momentum acceptance), which is what
    lets the ring keep protons circulating while they straggle in the target.
  - Off the midplane the field is expanded to 2nd/3rd order so that it
    satisfies Maxwell's equations (this provides the correct vertical
    focusing from the edge/flutter terms):
        By = b0 - (y^2/2)*Lap(b0)
        Bx = y*db0/dx - (y^3/6)*d(Lap b0)/dx
        Bz = y*db0/dz - (y^3/6)*d(Lap b0)/dz
    with b0(x,z) the midplane field and Lap the 2-D transverse Laplacian.

Geometry bookkeeping (hard-edge equivalent, exact tanh integrals):
  cell azimuth     = 360/N degrees
  F sector         : centered on the cell axis, azimuthal width thF_az,
                     bends the beam by +bendF
  D sectors (x2)   : at +/- azimuth, width thD_az each, bend -bendD/2 each
  net bend / cell  = bendF - bendD = 360/N
  field strengths  : BF = Brho*bendF/(R0*thF_az),  BD = Brho*bendD/(R0*thD_az_total)

The map is written for cell 0, whose axis is the global +x direction; the
grid covers the wedge |azimuth| <= half-cell (plus margins).  Because each
cell's map contains only that cell's magnets (with their fringe tails), and
G4beamline sums overlapping field maps, placing N rotated copies reproduces
the full ring field including cell-to-cell fringe overlap.

Output units: mm and Tesla (BLFieldMap grid format).
"""

import argparse
import math
import os
import numpy as np


def b0_midplane(r, th, P):
    """Midplane By [T]; r in mm, th (azimuth within cell) in rad."""
    w = P["fringe"] / P["R0"]          # tanh fringe width in azimuth [rad]
    thF2 = math.radians(P["thF_az"] / 2.0)
    dlo = math.radians(P["thD_lo"])
    dhi = math.radians(P["thD_hi"])

    gF = 0.5 * (np.tanh((th + thF2) / w) - np.tanh((th - thF2) / w))
    gDp = 0.5 * (np.tanh((th - dlo) / w) - np.tanh((th - dhi) / w))
    gDm = 0.5 * (np.tanh((-th - dlo) / w) - np.tanh((-th - dhi) / w))

    return (r / P["r0"]) ** P["k"] * (P["BF"] * gF - P["BD"] * (gDp + gDm))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-o", "--output", default="ffa_cell_map.txt")
    ap.add_argument("--N", type=int, default=12, help="number of cells")
    ap.add_argument("--R0", type=float, default=9000.0, help="machine radius [mm]")
    ap.add_argument("--k", type=float, default=4.3, help="scaling field index")
    ap.add_argument("--pc", type=float, default=1696.04, help="reference momentum [MeV/c]")
    ap.add_argument("--bendF", type=float, default=48.0, help="F bend angle [deg]")
    ap.add_argument("--thF", type=float, default=14.0, help="F azimuthal width [deg]")
    ap.add_argument("--thD", type=float, default=3.5, help="azimuthal width of EACH D [deg]")
    ap.add_argument("--gapFD", type=float, default=1.0, help="azimuthal F-D gap [deg]")
    ap.add_argument("--fringe", type=float, default=None,
                    help="tanh fringe length [mm] (default: 100*R0/9000, scale-invariant)")
    ap.add_argument("--scale", type=float, default=1.0, help="overall field scale factor")
    args = ap.parse_args()

    N, R0, k = args.N, args.R0, args.k
    s = R0 / 9000.0                           # scale factor vs the 1 GeV ring
    if args.fringe is None:
        args.fringe = 100.0 * s
    cell = 360.0 / N
    bendF = args.bendF
    bendD = bendF - cell                      # total reverse bend of the two Ds
    thF_az = args.thF
    thD_lo = thF_az / 2.0 + args.gapFD        # inner azimuth of each D
    thD_hi = thD_lo + args.thD                # outer azimuth of each D
    thD_az_tot = 2.0 * args.thD

    Brho = args.pc / 299.792458               # T*m  (pc in MeV/c)
    R0_m = R0 / 1000.0
    BF = args.scale * Brho * math.radians(bendF) / (R0_m * math.radians(thF_az))
    BD = args.scale * Brho * math.radians(bendD) / (R0_m * math.radians(thD_az_tot))

    P = dict(R0=R0, r0=R0, k=k, BF=BF, BD=BD, thF_az=thF_az,
             thD_lo=thD_lo, thD_hi=thD_hi, fringe=args.fringe)

    print(f"cells N={N}  cell={cell:.3f} deg  R0={R0_m:.3f} m  k={k}")
    print(f"Brho={Brho:.4f} T.m   bendF=+{bendF} deg  bendD=-{bendD} deg")
    print(f"BF(r0)={BF:.4f} T   BD(r0)=-{BD:.4f} T (reverse)")
    print(f"D sectors span {thD_lo:.2f}..{thD_hi:.2f} deg on each side")
    print(f"drift azimuth per half-cell: {cell/2.0 - thD_hi:.2f} deg "
          f"({R0_m*math.radians(cell/2.0-thD_hi):.3f} m arc)")

    # ---- fine midplane grid (for accurate numerical derivatives) ----
    fine = 5.0 * s                             # mm
    x_f = np.arange(7900.0 * s, (9900.0 + 1e-6) * s + fine / 2, fine)
    z_f = np.arange(-2700.0 * s, (2700.0 + 1e-6) * s + fine / 2, fine)
    X, Z = np.meshgrid(x_f, z_f, indexing="ij")
    Rg = np.hypot(X, Z)
    TH = np.arctan2(Z, X)
    b0 = b0_midplane(Rg, TH, P)

    b0x, b0z = np.gradient(b0, fine, fine, edge_order=2)
    b0xx = np.gradient(b0x, fine, axis=0, edge_order=2)
    b0zz = np.gradient(b0z, fine, axis=1, edge_order=2)
    lap = b0xx + b0zz
    lapx, lapz = np.gradient(lap, fine, fine, edge_order=2)

    # ---- output grid (must be a sub-lattice of the fine grid) ----
    step = 20.0 * s                            # mm
    ix = slice(20, len(x_f) - 20, int(round(step / fine)))   # 8000..9800 (x s)
    iz = slice(20, len(z_f) - 20, int(round(step / fine)))   # -2600..2600 (x s)
    xs = x_f[ix]
    zs = z_f[iz]
    ys = np.arange(-120.0 * s, 120.0 * s + step / 2, step)

    b0c, b0xc, b0zc = b0[ix, iz], b0x[ix, iz], b0z[ix, iz]
    lapc, lapxc, lapzc = lap[ix, iz], lapx[ix, iz], lapz[ix, iz]

    nX, nY, nZ = len(xs), len(ys), len(zs)
    print(f"grid: nX={nX} nY={nY} nZ={nZ}  ({nX*nY*nZ} points)")

    outdir = os.path.dirname(args.output)
    if outdir:
        os.makedirs(outdir, exist_ok=True)
    with open(args.output, "w") as f:
        f.write(f"# scaling-FFA cell map: N={N} R0={R0}mm k={k} pc={args.pc}MeV/c\n")
        f.write(f"# BF={BF:.5f}T BD=-{BD:.5f}T thF={thF_az}deg "
                f"D:{thD_lo}-{thD_hi}deg fringe={args.fringe}mm scale={args.scale}\n")
        f.write(f"grid X0={xs[0]:.4f} Y0={ys[0]:.4f} Z0={zs[0]:.4f} "
                f"nX={nX} nY={nY} nZ={nZ} dX={step:.4f} dY={step:.4f} "
                f"dZ={step:.4f} tolerance=2.0\n")
        f.write("data\n")
        for iy, y in enumerate(ys):
            By = b0c - 0.5 * y * y * lapc
            Bx = y * b0xc - (y ** 3 / 6.0) * lapxc
            Bz = y * b0zc - (y ** 3 / 6.0) * lapzc
            for i in range(nX):
                for j in range(nZ):
                    f.write(f"{xs[i]:.4f} {y:.4f} {zs[j]:.4f} "
                            f"{Bx[i, j]:.6g} {By[i, j]:.6g} {Bz[i, j]:.6g}\n")
        print(f"wrote {args.output}")

    # quick sanity number: bend integral along the r=r0 arc over one cell
    th_probe = np.linspace(-math.radians(cell / 2), math.radians(cell / 2), 4001)
    prof = b0_midplane(np.full_like(th_probe, R0), th_probe, P)
    integ = np.trapezoid(prof, th_probe) * R0_m       # T*m
    print(f"integral B.dl along r=r0 arc over one cell: {integ:.4f} T.m "
          f"(hard-edge design: {Brho*math.radians(cell):.4f})")


if __name__ == "__main__":
    main()
