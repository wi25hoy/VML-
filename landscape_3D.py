#!/usr/bin/env python3
import argparse, os, numpy as np
import matplotlib
matplotlib.use("Agg")          # remove if you want an interactive window
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D)
from matplotlib import cm

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="indir", required=True, help="Path to the landscape out dir (the one that contains npy/)")
    ap.add_argument("--downsample", type=int, default=1, help="Plot every Nth grid point (use 2 or 3 if your grid is large)")
    ap.add_argument("--elev", type=float, default=35.0, help="Elevation angle")
    ap.add_argument("--azim", type=float, default=-60.0, help="Azimuth angle")
    ap.add_argument("--outfile", type=str, default="figs/surface_3d.png")
    args = ap.parse_args()

    npy_dir = os.path.join(args.indir, "npy")
    alphas = np.load(os.path.join(npy_dir, "alphas.npy"))
    betas  = np.load(os.path.join(npy_dir, "betas.npy"))
    Z      = np.load(os.path.join(npy_dir, "surface.npy"))

    # Mesh
    A, B = np.meshgrid(alphas, betas, indexing="xy")  # A.shape == B.shape == Z.T.shape

    # Optional downsample to speed rendering
    s = max(1, int(args.downsample))
    A_ = A[::s, ::s]
    B_ = B[::s, ::s]
    Z_ = Z.T[::s, ::s]   # note the .T to align with XY indexing

    # 3D surface
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(A_, B_, Z_, rstride=1, cstride=1,
                           cmap=cm.viridis, linewidth=0, antialiased=True)

    ax.set_xlabel("alpha (dir-1)")
    ax.set_ylabel("beta (dir-2)")
    ax.set_zlabel("Cross-Entropy")
    ax.set_title("3D CE Loss Surface")
    ax.view_init(elev=args.elev, azim=args.azim)
    fig.colorbar(surf, shrink=0.6, aspect=12, pad=0.08, label="Cross-Entropy")
    os.makedirs(os.path.join(args.indir, "figs"), exist_ok=True)
    outpath = os.path.join(args.indir, args.outfile) if not os.path.isabs(args.outfile) else args.outfile
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    print(f"[ok] wrote {outpath}")

if __name__ == "__main__":
    main()
