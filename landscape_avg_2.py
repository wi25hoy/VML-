#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Aggregate & visualize loss-landscape slices across seeds (EXTENDED, FIXED).

Adds:
- Mean+/-SD a- and ß-slices across seeds with quadratic curvature fits overlaid.
- Mean 3D surfaces per condition (BASE/SL/VPL/VML).
- Radial ?CE(r) profiles averaged across angles, overlaid across conditions.

Inputs (per run, produced by your landscape script):
  <run_dir>/npy/surface.npy   (GxG array for alphas x betas)
  <run_dir>/npy/alphas.npy    (G,)
  <run_dir>/npy/betas.npy     (G,)
  [optional] profile_steps.npy, profile_vals.npy

Usage:
  python landscape_avg_2.py \
    --root analysis/landscape_all \
    --dataset cifar10 --arch resnet14 \
    --out analysis/landscape_avg2_c10_r14 \
    --fit_window 0.5 --radial_angles 72 --radial_points 200
"""

import argparse, os, re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib import cm
import pandas as pd

COND_ORDER = ["base", "sl", "vpl", "vml"]

# ---------------- basic helpers ----------------

def infer_condition_from_path(p: Path) -> Optional[str]:
    s = p.as_posix().lower()
    if "vml" in s: return "vml"
    if "vpl" in s: return "vpl"
    if re.search(r"(^|[_\-/])sl([_\-/]|$)", s): return "sl"
    if "base" in s or "baseline" in s: return "base"
    return None

def find_run_dirs_with_surface(root: Path) -> List[Path]:
    out = []
    for surf in root.rglob("npy/surface.npy"):
        d = surf.parent.parent  # .../npy -> run_dir
        out.append(d)
    return sorted(set(out))

def load_surface(run_dir: Path):
    npy = run_dir / "npy"
    A = np.load(npy / "alphas.npy").astype(float)  # (G,)
    B = np.load(npy / "betas.npy").astype(float)   # (G,)
    Z = np.load(npy / "surface.npy").astype(float) # (G,G) with Z[i,j] at (A[i],B[j])
    return A, B, Z

def nearest_index(arr: np.ndarray, value: float) -> int:
    return int(np.argmin(np.abs(arr - value)))

def extract_slices(A: np.ndarray, B: np.ndarray, Z: np.ndarray) -> Tuple[Tuple[np.ndarray,np.ndarray], Tuple[np.ndarray,np.ndarray]]:
    ia = nearest_index(A, 0.0)  # alpha˜0 slice ? vary beta
    ib = nearest_index(B, 0.0)  # beta˜0 slice  ? vary alpha
    return (B.copy(), Z[ia, :].copy()), (A.copy(), Z[:, ib].copy())

# ---------------- interpolation (SciPy-free) ----------------

def interp_1d(x_src: np.ndarray, y_src: np.ndarray, x_tgt: np.ndarray) -> np.ndarray:
    m = np.isfinite(x_src) & np.isfinite(y_src)
    if m.sum() < 2:
        return np.full_like(x_tgt, np.nan, dtype=float)
    xs, ys = x_src[m], y_src[m]
    idx = np.argsort(xs)
    xs, ys = xs[idx], ys[idx]
    # Return NaN outside bounds to avoid extrap artifacts in averages
    out = np.interp(x_tgt, xs, ys, left=np.nan, right=np.nan)
    return out

def interp2d_separable(A_src: np.ndarray, B_src: np.ndarray, Z_src: np.ndarray,
                       A_tgt: np.ndarray, B_tgt: np.ndarray) -> np.ndarray:
    """
    Simple separable linear interpolation on rect grids using np.interp.
    1) interp along A to A_tgt for each B row, 2) interp along B to B_tgt for each A col.
    """
    # Step 1: along A (alpha)
    Z_step1 = np.full((len(A_tgt), len(B_src)), np.nan, dtype=float)
    for j in range(len(B_src)):
        Z_step1[:, j] = interp_1d(A_src, Z_src[:, j], A_tgt)

    # Step 2: along B (beta)
    Z_out = np.full((len(A_tgt), len(B_tgt)), np.nan, dtype=float)
    for i in range(len(A_tgt)):
        Z_out[i, :] = interp_1d(B_src, Z_step1[i, :], B_tgt)

    return Z_out

# ---------------- stats helpers ----------------

def stack_mean_std(arrs: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    if not arrs: return np.array([]), np.array([])
    X = np.stack(arrs, axis=0)
    mean = np.nanmean(X, axis=0)
    std  = np.nanstd(X, axis=0, ddof=1) if X.shape[0] > 1 else np.zeros_like(mean)
    return mean, std

# ---------------- curvature fit ----------------

def quad_fit_curvature(x: np.ndarray, y: np.ndarray, use_window: Optional[float] = None):
    """
    Fit y ˜ a*x^2 + b*x + c (least squares) on finite points.
    If use_window is set (e.g., 0.5), only use |x| <= use_window.
    Returns (a,b,c) and curvature ? = d²y/dx²|0 = 2a.
    """
    m = np.isfinite(x) & np.isfinite(y)
    xs, ys = x[m], y[m]
    if use_window is not None:
        m2 = np.abs(xs) <= float(use_window)
        xs, ys = xs[m2], ys[m2]
    if xs.size < 3:
        return np.nan, np.nan, np.nan, np.nan
    a, b, c = np.polyfit(xs, ys, deg=2)
    kappa = 2.0 * a
    return float(a), float(b), float(c), float(kappa)

# ---------------- plotting ----------------

def plot_slice_mean_with_fit(ax, x, mean_by_cond, std_by_cond, title, xlabel, fit_window=None):
    for cond in COND_ORDER:
        m = mean_by_cond.get(cond); s = std_by_cond.get(cond)
        if m is None or m.size == 0: continue
        ax.plot(x, m, label=f"{cond.upper()} (mean)")
        ax.fill_between(x, m - s, m + s, alpha=0.15)

        # fit and overlay
        a,b,c,k = quad_fit_curvature(x, m, use_window=fit_window)
        if np.isfinite(k):
            yfit = a*x*x + b*x + c
            ax.plot(x, yfit, linestyle="--", linewidth=1.2,
                    label=f"{cond.upper()} fit ?={k:.3g}")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Cross-Entropy")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2)

def save_surface_3d_fig(A, B, Z, out_png, elev=35, azim=-60, title="3D CE Loss Surface (mean)"):
    Amesh, Bmesh = np.meshgrid(A, B, indexing="xy")
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(Amesh, Bmesh, Z.T, rstride=1, cstride=1,
                           cmap=cm.viridis, linewidth=0, antialiased=True)
    ax.set_xlabel("alpha (dir-1)")
    ax.set_ylabel("beta (dir-2)")
    ax.set_zlabel("Cross-Entropy")
    ax.set_title(title)
    ax.view_init(elev=elev, azim=azim)
    fig.colorbar(surf, shrink=0.6, aspect=12, pad=0.08, label="Cross-Entropy")
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

# ---------------- robust bilinear sampling + radial ?CE ----------------

def _bilinear_sample_grid(A: np.ndarray, B: np.ndarray, Z: np.ndarray,
                          aq: np.ndarray, bq: np.ndarray) -> np.ndarray:
    """
    Bilinear sample Z on the rect grid (A,B) at query points (aq, bq).
    Returns a 1-D array of shape (len(aq),).
    Assumes A and B are ascending.
    """
    aq = np.asarray(aq, dtype=float)
    bq = np.asarray(bq, dtype=float)

    # clip queries to the valid interior (so i+1, j+1 exist)
    a0, a1 = float(A[0]), float(A[-1])
    b0, b1 = float(B[0]), float(B[-1])
    eps = 1e-12
    aqc = np.clip(aq, a0, a1 - eps)
    bqc = np.clip(bq, b0, b1 - eps)

    # find left indices
    i = np.searchsorted(A, aqc, side="right") - 1
    j = np.searchsorted(B, bqc, side="right") - 1
    i = np.clip(i, 0, len(A) - 2)
    j = np.clip(j, 0, len(B) - 2)

    # linear weights
    Ai = A[i]; Ai1 = A[i+1]
    Bj = B[j]; Bj1 = B[j+1]
    tx = (aqc - Ai) / (Ai1 - Ai + eps)
    ty = (bqc - Bj) / (Bj1 - Bj + eps)

    # gather corners
    Z00 = Z[i,   j  ]
    Z10 = Z[i+1, j  ]
    Z01 = Z[i,   j+1]
    Z11 = Z[i+1, j+1]

    # bilinear interpolation
    Zq = ((1-tx)*(1-ty)*Z00 +
          tx*(1-ty)*Z10 +
          (1-tx)*ty*Z01 +
          tx*ty*Z11)
    return Zq

def radial_profile_from_surface(A: np.ndarray, B: np.ndarray, Z: np.ndarray,
                                n_angles: int = 72,
                                n_r: int = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute ?CE(r) = CE(r,?) - CE(0,0), then mean+/-SD over ? for each radius r.
    Returns:
      rs: (n_r,)
      mean: (n_r,)
      std: (n_r,)
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    Z = np.asarray(Z, dtype=float)

    # conservative radius so all directions stay inside the grid
    r_max = min(A.max(), -A.min(), B.max(), -B.min())
    rs = np.linspace(0.0, float(r_max), int(n_r))
    thetas = np.linspace(0.0, 2*np.pi, int(n_angles), endpoint=False)

    # CE at origin (nearest grid point)
    ia0 = int(np.argmin(np.abs(A - 0.0)))
    ib0 = int(np.argmin(np.abs(B - 0.0)))
    ce0 = float(Z[ia0, ib0])

    # accumulate angle × radius matrix
    D = np.zeros((len(thetas), len(rs)), dtype=float)
    for t_idx, theta in enumerate(thetas):
        a_line = rs * np.cos(theta)
        b_line = rs * np.sin(theta)
        z_line = _bilinear_sample_grid(A, B, Z, a_line, b_line)  # (n_r,)
        D[t_idx, :] = z_line - ce0

    mean = np.nanmean(D, axis=0)
    std  = np.nanstd(D,  axis=0, ddof=1) if D.shape[0] > 1 else np.zeros_like(mean)
    return rs, mean, std

# ---------------- main ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Root folder that contains per-run landscape outputs.")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--arch", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fit_window", type=float, default=None,
                    help="If set (e.g., 0.5), fit curvature only on |x|<=fit_window.")
    ap.add_argument("--radial_angles", type=int, default=72)
    ap.add_argument("--radial_points", type=int, default=100)
    args = ap.parse_args()

    root = Path(args.root)
    out  = Path(args.out)
    (out / "figs").mkdir(parents=True, exist_ok=True)
    (out / "curves").mkdir(parents=True, exist_ok=True)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    figs = out / "figs"; curves = out / "curves"

    # Discover runs
    runs = find_run_dirs_with_surface(root)
    rows = []
    grouped: Dict[str, List[Path]] = {c: [] for c in COND_ORDER}
    for rd in runs:
        cond = infer_condition_from_path(rd)
        if cond is None: continue
        grouped[cond].append(rd)
        rows.append({"condition": cond, "run_dir": str(rd)})
    if not rows:
        raise SystemExit(f"No landscape runs found under {root} (need npy/surface.npy).")
    pd.DataFrame(rows).to_csv(out / "tables" / "runs_found.csv", index=False)

    # Storage
    mean_surface_by_cond: Dict[str, np.ndarray] = {}
    std_surface_by_cond:  Dict[str, np.ndarray] = {}
    A_ref_by_cond: Dict[str, np.ndarray] = {}
    B_ref_by_cond: Dict[str, np.ndarray] = {}

    # Also store slices
    beta_x_ref_by_cond: Dict[str, np.ndarray] = {}
    alpha_x_ref_by_cond: Dict[str, np.ndarray] = {}
    beta_y_list_by_cond: Dict[str, List[np.ndarray]] = {c: [] for c in COND_ORDER}
    alpha_y_list_by_cond: Dict[str, List[np.ndarray]] = {c: [] for c in COND_ORDER}

    # Aggregate per condition
    for cond in COND_ORDER:
        if not grouped.get(cond): continue
        Zs_aligned = []
        for k, rd in enumerate(grouped[cond]):
            try:
                A, B, Z = load_surface(rd)
                # reference grid for this condition = first valid run
                if cond not in A_ref_by_cond:
                    A_ref_by_cond[cond] = A
                    B_ref_by_cond[cond] = B
                # align to reference grid
                if (not np.allclose(A, A_ref_by_cond[cond])) or (not np.allclose(B, B_ref_by_cond[cond])):
                    Z = interp2d_separable(A, B, Z, A_ref_by_cond[cond], B_ref_by_cond[cond])
                    A = A_ref_by_cond[cond]; B = B_ref_by_cond[cond]
                Zs_aligned.append(Z)

                # slices for curvature
                (bx, by), (ax, ay) = extract_slices(A, B, Z)
                if cond not in beta_x_ref_by_cond: beta_x_ref_by_cond[cond] = bx
                if cond not in alpha_x_ref_by_cond: alpha_x_ref_by_cond[cond] = ax
                if not np.allclose(bx, beta_x_ref_by_cond[cond]):
                    by = interp_1d(bx, by, beta_x_ref_by_cond[cond])
                if not np.allclose(ax, alpha_x_ref_by_cond[cond]):
                    ay = interp_1d(ax, ay, alpha_x_ref_by_cond[cond])
                beta_y_list_by_cond[cond].append(by)
                alpha_y_list_by_cond[cond].append(ay)
            except Exception as e:
                print(f"[warn] skipping {rd}: {e}")

        if Zs_aligned:
            m, s = stack_mean_std(Zs_aligned)
            mean_surface_by_cond[cond] = m
            std_surface_by_cond[cond]  = s

            # save mean surface arrays
            np.save(curves / f"{cond}_A.npy", A_ref_by_cond[cond])
            np.save(curves / f"{cond}_B.npy", B_ref_by_cond[cond])
            np.save(curves / f"{cond}_surface_mean.npy", m)
            np.save(curves / f"{cond}_surface_std.npy",  s)

            # per-condition 3D plot
            save_surface_3d_fig(A_ref_by_cond[cond], B_ref_by_cond[cond], m,
                                figs / f"surface_3d_mean_{cond}.png",
                                title=f"{args.dataset.upper()} / {args.arch} - {cond.upper()} 3D mean surface")

    # Alpha / Beta slice plots with curvature fits (overlaid across conditions)
    mean_beta_by_cond, std_beta_by_cond = {}, {}
    mean_alpha_by_cond, std_alpha_by_cond = {}, {}

    for cond in COND_ORDER:
        if beta_y_list_by_cond.get(cond):
            m, s = stack_mean_std(beta_y_list_by_cond[cond])
            mean_beta_by_cond[cond] = m; std_beta_by_cond[cond] = s
            x = beta_x_ref_by_cond[cond]
            pd.DataFrame({"beta": x, "mean": m, "std": s}).to_csv(curves / f"{cond}_slice_beta.csv", index=False)
        if alpha_y_list_by_cond.get(cond):
            m, s = stack_mean_std(alpha_y_list_by_cond[cond])
            mean_alpha_by_cond[cond] = m; std_alpha_by_cond[cond] = s
            x = alpha_x_ref_by_cond[cond]
            pd.DataFrame({"alpha": x, "mean": m, "std": s}).to_csv(curves / f"{cond}_slice_alpha.csv", index=False)

    # ß slice (a˜0)
    if mean_beta_by_cond:
        # choose an x from any cond present
        some_cond = next(iter(mean_beta_by_cond))
        x = beta_x_ref_by_cond[some_cond]
        fig, ax = plt.subplots(figsize=(8,5))
        plot_slice_mean_with_fit(
            ax, x, mean_beta_by_cond, std_beta_by_cond,
            f"{args.dataset.upper()} / {args.arch} - ß-slice at a˜0 (mean +/- SD + curvature fit)",
            xlabel="beta (dir-2)", fit_window=args.fit_window
        )
        fig.tight_layout(); fig.savefig(figs / "slice_beta_mean_fit.png", dpi=200); plt.close(fig)

    # a slice (ß˜0)
    if mean_alpha_by_cond:
        some_cond = next(iter(mean_alpha_by_cond))
        x = alpha_x_ref_by_cond[some_cond]
        fig, ax = plt.subplots(figsize=(8,5))
        plot_slice_mean_with_fit(
            ax, x, mean_alpha_by_cond, std_alpha_by_cond,
            f"{args.dataset.upper()} / {args.arch} - a-slice at ß˜0 (mean +/- SD + curvature fit)",
            xlabel="alpha (dir-1)", fit_window=args.fit_window
        )
        fig.tight_layout(); fig.savefig(figs / "slice_alpha_mean_fit.png", dpi=200); plt.close(fig)

    # Radial ?CE(r): overlay conditions (means with bands are per-condition)
    fig, ax = plt.subplots(figsize=(8,5))
    for cond in COND_ORDER:
        if cond not in mean_surface_by_cond: continue
        A = A_ref_by_cond[cond]; B = B_ref_by_cond[cond]; Zm = mean_surface_by_cond[cond]
        rs, dm, ds = radial_profile_from_surface(A, B, Zm,
                                                 n_angles=int(args.radial_angles),
                                                 n_r=int(args.radial_points))
        # ensure 1-D shapes
        rs = np.ravel(rs); dm = np.ravel(dm); ds = np.ravel(ds)
        assert rs.shape == dm.shape == ds.shape, f"shape mismatch: {rs.shape}, {dm.shape}, {ds.shape}"

        ax.plot(rs, dm, label=cond.upper())
        ax.fill_between(rs, dm - ds, dm + ds, alpha=0.15)
        pd.DataFrame({"r": rs, "mean_delta": dm, "std_delta": ds}).to_csv(curves / f"{cond}_radial_delta.csv", index=False)

    ax.set_title(f"{args.dataset.upper()} / {args.arch} - Radial ?CE(r) from origin (mean +/- SD)")
    ax.set_xlabel("radius r in (alpha,beta) plane")
    ax.set_ylabel("?CE(r) = CE(r) - CE(0)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout(); fig.savefig(figs / "radial_delta_overlay.png", dpi=200); plt.close(fig)

    print("Done.")
    print(f"- Figures: {figs}")
    print(f"- Curves : {curves}")
    print(f"- Runs   : {out/'tables'/'runs_found.csv'}")

if __name__ == "__main__":
    main()
