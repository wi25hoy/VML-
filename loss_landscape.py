#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
loss_landscape.py - Filter-normalized 2D/1D loss landscape around a trained solution
with robust checkpointing, resume, 3D figure, and stability diagnostics.

Outputs (under --out):
- npy/surface.npy, npy/alphas.npy, npy/betas.npy
- figs/surface_2d.png, figs/surface_3d.png, figs/profile_1d.png
- figs/curvature_fit_alpha.png, figs/curvature_fit_beta.png
- figs/radial_deltas.png
- meta.json (arguments snapshot), metrics.json (diagnostics)

Diagnostics in metrics.json:
  Z0, curv_lam1, curv_lam2, curv_mean, curv_anisotropy,
  grad_norm_center, basin_frac_+0.5, basin_frac_+1.0,
  edge_blowup_p90, area_1d, radial_deltas (list of [r, ?Loss])
"""

import argparse, os, sys, json, math, pathlib, numpy as np
import torch, torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import data
import models_clf

# ---------------------------
# Repro helpers
# ---------------------------

def set_global_seeds(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ---------------------------
# Param utils
# ---------------------------

def _param_iterator(model):
    for n, p in model.named_parameters():
        if p.requires_grad:
            yield n, p

def _filterwise_norm_like(weight, direction):
    # Conv2d: [out,in,kh,kw] -> normalize per out-channel
    if weight.ndim == 4:
        d = direction.clone()
        w = weight.data
        out = w.shape[0]
        d2 = d.reshape(out, -1)
        w2 = w.reshape(out, -1)
        d_norm = d2.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
        w_norm = w2.norm(p=2, dim=1, keepdim=True)
        scaled = (d2 / d_norm) * w_norm
        return scaled.view_as(direction)
    # Linear: [out,in] -> normalize per out-row
    elif weight.ndim == 2:
        d = direction.clone()
        w = weight.data
        d_norm = d.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
        w_norm = w.norm(p=2, dim=1, keepdim=True)
        return (d / d_norm) * w_norm
    # Bias/BN scale: scalar norm match
    elif weight.ndim == 1:
        d = direction.clone()
        d_norm = d.norm(p=2).clamp_min(1e-12)
        w_norm = weight.data.norm(p=2)
        return (d / d_norm) * w_norm
    # Fallback
    d = direction.clone()
    n = d.norm().clamp_min(1e-12)
    return d / n

def make_filter_normalized_direction(model, seed=123):
    """
    Build a direction {name -> tensor} aligned with model params, filter-normalized.
    Works on the param device (CPU/GPU); uses torch.Generator for determinism.
    """
    device = next(model.parameters()).device
    g = torch.Generator(device=device.type)
    g.manual_seed(int(seed))

    direction = {}
    for n, p in _param_iterator(model):
        # Use torch.randn with explicit device + generator for old PyTorch compat
        r = torch.randn(tuple(p.shape), device=p.device, dtype=p.dtype, generator=g)
        direction[n] = _filterwise_norm_like(p, r)
    return direction

def add_direction(model, base_state, a, d1, b=0.0, d2=None):
    with torch.no_grad():
        for n, p in _param_iterator(model):
            w0 = base_state[n]
            if d2 is None:
                p.copy_(w0 + a * d1[n])
            else:
                p.copy_(w0 + a * d1[n] + b * d2[n])

def snapshot_state(model):
    state = {}
    for n, p in _param_iterator(model):
        state[n] = p.detach().clone()
    return state

# ---------------------------
# BN re-estimation
# ---------------------------

@torch.no_grad()
def _reestimate_bn_stats(model, loader, device, batch_limit_batches: int = 20):
    """
    Recompute BatchNorm running stats on (a subset of) training data.
    Uses model.train() but without gradients.
    """
    model.train()
    seen = 0
    for x, _ in loader:
        x = x.to(device)
        _ = model(x)
        seen += 1
        if seen >= batch_limit_batches:
            break
    model.eval()

# ---------------------------
# Loss eval
# ---------------------------

@torch.no_grad()
def ce_on_loader(model, loader, device, batch_limit=None):
    model.eval()
    ce = nn.CrossEntropyLoss(reduction='sum')
    tot_loss = 0.0
    tot_n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = ce(logits, y)
        tot_loss += float(loss.item())
        tot_n += int(y.numel())
        if batch_limit is not None and tot_n >= batch_limit:
            break
    return tot_loss / max(1, tot_n)

# ---------------------------
# Plot helpers
# ---------------------------

def _ensure(outdir, sub="figs"):
    d = os.path.join(outdir, sub)
    os.makedirs(d, exist_ok=True)
    return d

def save_surface_fig(alphas, betas, Z, outdir):
    fig_dir = _ensure(outdir, "figs")
    fig, ax = plt.subplots(figsize=(6,5))
    cs = ax.contourf(alphas, betas, Z.T, levels=30)
    cbar = fig.colorbar(cs, ax=ax)
    cbar.set_label("Cross-Entropy")
    ax.set_xlabel("alpha (dir-1)")
    ax.set_ylabel("beta (dir-2)")
    ax.set_title("2D CE Loss Surface")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "surface_2d.png"), dpi=200)
    plt.close(fig)

def save_surface_3d_fig(alphas, betas, Z, outdir, elev=35, azim=-60):
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    from matplotlib import cm
    fig_dir = _ensure(outdir, "figs")
    A, B = np.meshgrid(alphas, betas, indexing="xy")
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(A, B, Z.T, rstride=1, cstride=1,
                           cmap=cm.viridis, linewidth=0, antialiased=True)
    ax.set_xlabel("alpha (dir-1)")
    ax.set_ylabel("beta (dir-2)")
    ax.set_zlabel("Cross-Entropy")
    ax.set_title("3D CE Loss Surface")
    ax.view_init(elev=35, azim=-60)
    fig.colorbar(surf, shrink=0.6, aspect=12, pad=0.08, label="Cross-Entropy")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "surface_3d.png"), dpi=200)
    plt.close(fig)

def save_profile_fig(steps, vals, outdir):
    fig_dir = _ensure(outdir, "figs")
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(steps, vals, marker='o')
    ax.set_xlabel("alpha (dir-1)")
    ax.set_ylabel("Cross-Entropy")
    ax.set_title("1D CE Profile")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "profile_1d.png"), dpi=200)
    plt.close(fig)

# ---------------------------
# Checkpoint helpers
# ---------------------------

def _init_or_resume_arrays(outdir, g, span_x, span_y, resume: bool):
    npy_dir = os.path.join(outdir, "npy")
    os.makedirs(npy_dir, exist_ok=True)

    alpha_path = os.path.join(npy_dir, "alphas.npy")
    beta_path  = os.path.join(npy_dir, "betas.npy")
    surf_path  = os.path.join(npy_dir, "surface.npy")
    meta_path  = os.path.join(outdir, "meta.json")

    if resume and os.path.isfile(surf_path) and os.path.isfile(alpha_path) and os.path.isfile(beta_path):
        alphas = np.load(alpha_path)
        betas  = np.load(beta_path)
        Z      = np.load(surf_path)
        if alphas.shape[0] != g or betas.shape[0] != g or Z.shape != (g, g):
            print("[resume] grid dims changed; starting fresh.")
            resume = False
        else:
            print(f"[resume] loaded existing surface with {np.sum(~np.isfinite(Z))} unfinished cells.")
    if not resume:
        alphas = np.linspace(-span_x, span_x, g).astype(np.float32)
        betas  = np.linspace(-span_y, span_y, g).astype(np.float32)
        Z = np.full((g, g), np.nan, dtype=np.float32)
        np.save(alpha_path, alphas)
        np.save(beta_path, betas)
        np.save(surf_path, Z)
        print("[init] created new surface arrays.")

    return alphas, betas, Z, (alpha_path, beta_path, surf_path, meta_path)

def _save_meta(meta_path, args):
    meta = {k: (v if isinstance(v, (int, float, str, bool)) else str(v))
            for k, v in vars(args).items()}
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

def _save_surface(surf_path, Z):
    os.makedirs(os.path.dirname(surf_path), exist_ok=True)
    tmp = surf_path + ".tmp.npy"
    with open(tmp, "wb") as f:
        np.save(f, Z); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, surf_path)

# ---------------------------
# Diagnostics (numbers + extra plots)
# ---------------------------

def _center_idx(axis):
    return int(np.argmin(np.abs(axis - 0.0)))

def _finite_diff_center(Z, alphas, betas, step_idx=1):
    i0 = _center_idx(alphas); j0 = _center_idx(betas)
    da = alphas[1] - alphas[0]; db = betas[1] - betas[0]
    gx = (Z[i0+step_idx, j0] - Z[i0-step_idx, j0]) / (2*da)
    gy = (Z[i0, j0+step_idx] - Z[i0, j0-step_idx]) / (2*db)
    return float(np.sqrt(gx*gx + gy*gy))

def _quad_curvature_fit(xs, ys, window_abs=0.2, out_png=None, title="", xlabel="x"):
    xs = np.asarray(xs); ys = np.asarray(ys)
    mask = np.abs(xs) <= window_abs
    x = xs[mask]; y = ys[mask]
    # fit y ˜ a x^2 + b x + c  -> curvature ? ˜ 2a (since y ˜ c0 + 0.5 ? x^2)
    A = np.stack([x*x, x, np.ones_like(x)], axis=1)
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    a, b, c = coef
    lam = 2*a
    if out_png is not None:
        xx = np.linspace(xs.min(), xs.max(), 401)
        yy = a*xx*xx + b*xx + c
        fig, ax = plt.subplots(figsize=(6,4))
        ax.plot(xs, ys, marker='o', label="slice")
        ax.plot(xx, yy, linestyle='--', label=f"quad fit (?˜{lam:.3g})")
        ax.set_title(title)
        ax.set_xlabel(xlabel); ax.set_ylabel("Cross-Entropy")
        ax.grid(True, alpha=0.3); ax.legend()
        fig.tight_layout(); fig.savefig(out_png, dpi=200); plt.close(fig)
    return float(lam), float(c)

def _ring_stats(Z, alphas, betas, radii):
    A, B = np.meshgrid(alphas, betas, indexing='ij')
    R = np.sqrt(A*A + B*B)
    i0 = _center_idx(alphas); j0 = _center_idx(betas)
    Z0 = float(Z[i0, j0])
    tol = min(alphas[1]-alphas[0], betas[1]-betas[0]) * 0.75
    out = []
    for r in radii:
        m = np.abs(R - r) <= tol
        if np.any(m):
            out.append((float(r), float(np.nanmean(Z[m]) - Z0)))
    return Z0, out

def _plot_radial_deltas(radials, out_png):
    r = [t[0] for t in radials]
    d = [t[1] for t in radials]
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(r, d, marker='o')
    ax.set_xlabel("radius r (in plane)"); ax.set_ylabel("?Loss(r)")
    ax.set_title("Radial robustness (mean ?Loss on ring)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(out_png, dpi=200); plt.close(fig)

def _basin_fraction(Z, Z0, delta):
    return float(np.mean(Z <= (Z0 + delta)))

def _edge_blowup_p90(Z, Z0):
    edges = np.r_[Z[0,:], Z[-1,:], Z[1:-1,0], Z[1:-1,-1]]
    return float(np.nanpercentile(edges, 90) - Z0)

def _area_1d(xs, ys, Z0):
    return float(np.trapz(np.abs(ys - Z0), xs))

# ---------------------------
# Main
# ---------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["cifar10", "cifar100"])
    ap.add_argument("--arch", required=True, help="e.g., resnet14")
    ap.add_argument("--ckpt", required=True, help="Path to model.ckpt from training")
    ap.add_argument("--grid", type=int, default=41, help="Grid resolution per axis (odd recommended)")
    ap.add_argument("--span_x", type=float, default=1.0, help="Max |alpha| (dir-1)")
    ap.add_argument("--span_y", type=float, default=1.0, help="Max |beta| (dir-2)")
    ap.add_argument("--profile_span", type=float, default=1.0, help="Span for 1D profile along dir-1")
    ap.add_argument("--batch_limit", type=int, default=5000, help="Max samples to evaluate (speed/accuracy)")
    ap.add_argument("--seed", type=int, default=1, help="Direction RNG seed (and global seeds)")
    ap.add_argument("--out", required=True, help="Output directory for figs/npy")

    # BN options
    ap.add_argument("--bn_reestimate", type=int, default=0, help="1 = recompute BN stats per grid point")
    ap.add_argument("--bn_recalc_batches", type=int, default=20, help="# of train batches to re-estimate BN")
    ap.add_argument("--bn_batchsize", type=int, default=512, help="Batch size for BN re-estimation")

    # Checkpoint / resume options
    ap.add_argument("--resume", type=int, default=0, help="1 = resume from existing npy/surface.npy")
    ap.add_argument("--save_mode", type=str, default="row", choices=["row", "cell"],
                    help="Save after each row or after every N cells (see --save_every_n)")
    ap.add_argument("--save_every_n", type=int, default=100, help="If save_mode=cell, save every N cells")

    # Diagnostics knobs
    ap.add_argument("--curv_window_frac", type=float, default=0.25,
                    help="Fraction of span to use for local quadratic fit window (e.g., 0.25)")
    ap.add_argument("--ring_points", type=int, default=9, help="# radii for radial ?Loss curve")
    ap.add_argument("--basin_deltas", type=str, default="0.5,1.0",
                    help="Comma list of ? thresholds for basin width metrics")

    args = ap.parse_args()

    # Repro
    set_global_seeds(args.seed)

    # Device
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Load data
    testloader = data.get_testloader(args.dataset, batch_size=512)
    trainloader_bn = None
    if int(args.bn_reestimate) == 1:
        trainloader_bn = data.get_trainloader(args.dataset, batch_size=args.bn_batchsize)

    # Build model skeleton
    num_classes = 10 if args.dataset == "cifar10" else 100
    net = getattr(models_clf, args.arch)(num_classes=num_classes)
    if use_cuda:
        net = net.to(device)

    # Load checkpoint weights
    ckpt = torch.load(args.ckpt, map_location=device)
    net.load_state_dict(ckpt["model_state_dict"])
    net.eval()

    # Base snapshot & directions
    base_state = snapshot_state(net)
    d1 = make_filter_normalized_direction(net, seed=args.seed)
    d2 = make_filter_normalized_direction(net, seed=args.seed + 1)

    # Init or resume arrays
    alphas, betas, Z, (alpha_path, beta_path, surf_path, meta_path) = _init_or_resume_arrays(
        args.out, args.grid, args.span_x, args.span_y, resume=bool(args.resume)
    )
    _save_meta(meta_path, args)

    # 2D grid compute (fill NaNs only)
    total_cells = args.grid * args.grid
    done_before = int(np.sum(np.isfinite(Z)))
    print(f"[surface] grid={args.grid}x{args.grid} -> {total_cells} cells; "
          f"completed={done_before}, remaining={total_cells - done_before}")

    cell_counter_since_save = 0
    for i, a in enumerate(alphas):
        row_has_nan = np.isnan(Z[i, :]).any()
        if not row_has_nan:
            continue  # row already done (resume)
        for j, b in enumerate(betas):
            if np.isfinite(Z[i, j]):
                continue  # already computed
            # Set weights
            add_direction(net, base_state, float(a), d1, float(b), d2)

            # Optional BN re-estimation
            if int(args.bn_reestimate) == 1 and trainloader_bn is not None:
                _reestimate_bn_stats(net, trainloader_bn, device, batch_limit_batches=int(args.bn_recalc_batches))

            # Evaluate CE
            val = ce_on_loader(net, testloader, device, batch_limit=args.batch_limit)
            Z[i, j] = float(val)
            cell_counter_since_save += 1

            # Periodic cell-based save
            if args.save_mode == "cell" and cell_counter_since_save >= max(1, int(args.save_every_n)):
                _save_surface(surf_path, Z)
                cell_counter_since_save = 0
                print(f"[surface] saved (cell mode) @ i={i}, j={j}, val={val:.4f}")

        # Row-based save
        if args.save_mode == "row":
            _save_surface(surf_path, Z)
            print(f"[surface] saved row {i+1}/{args.grid}")

    # Restore base weights
    add_direction(net, base_state, 0.0, d1, 0.0, d2)

    # Save final surface + figures
    _save_surface(surf_path, Z)
    save_surface_fig(alphas, betas, Z, args.out)
    save_surface_3d_fig(alphas, betas, Z, args.out, elev=35, azim=-60)

    # 1D profile along d1
    g = args.grid
    steps = np.linspace(-args.profile_span, args.profile_span, g).astype(np.float32)
    vals = np.zeros_like(steps, dtype=np.float32)
    with torch.no_grad():
        for k, a in enumerate(steps):
            add_direction(net, base_state, float(a), d1)
            vals[k] = ce_on_loader(net, testloader, device, batch_limit=args.batch_limit)
        add_direction(net, base_state, 0.0, d1)

    # Save profile arrays + fig
    npy_dir = os.path.join(args.out, "npy")
    os.makedirs(npy_dir, exist_ok=True)
    np.save(os.path.join(npy_dir, "profile_steps.npy"), steps)
    np.save(os.path.join(npy_dir, "profile_vals.npy"), vals)
    save_profile_fig(steps, vals, args.out)

    # ---------- Diagnostics from surface ----------
    figs_dir = _ensure(args.out, "figs")
    i0 = int(np.argmin(np.abs(alphas)))
    j0 = int(np.argmin(np.abs(betas)))
    Z0 = float(Z[i0, j0])
    Za = Z[:, j0]
    Zb = Z[i0, :]

    # curvature window (absolute span fraction)
    win_a = float(args.curv_window_frac) * (alphas.max() - alphas.min()) / 2.0
    win_b = float(args.curv_window_frac) * (betas.max() - betas.min()) / 2.0

    lam1, _ = _quad_curvature_fit(alphas, Za, window_abs=win_a,
                                  out_png=os.path.join(figs_dir, "curvature_fit_alpha.png"),
                                  title="Curvature fit along alpha (beta=0)", xlabel="alpha")
    lam2, _ = _quad_curvature_fit(betas, Zb, window_abs=win_b,
                                  out_png=os.path.join(figs_dir, "curvature_fit_beta.png"),
                                  title="Curvature fit along beta (alpha=0)", xlabel="beta")

    # radial robustness
    radii = np.linspace(0.1, 0.9, int(args.ring_points))
    Z0_chk, radials = _ring_stats(Z, alphas, betas, radii)
    _plot_radial_deltas(radials, os.path.join(figs_dir, "radial_deltas.png"))

    # basin width
    basin_deltas = []
    for tok in str(args.basin_deltas).split(","):
        tok = tok.strip()
        if tok:
            try: basin_deltas.append(float(tok))
            except: pass
    basin_deltas = basin_deltas or [0.5, 1.0]
    basin_map = {}
    for dlt in basin_deltas:
        basin_map[f"basin_frac_+{dlt}"] = _basin_fraction(Z, Z0, dlt)

    # edge blow-up (p90) and grad norm at center
    ebi = _edge_blowup_p90(Z, Z0)
    grad0 = _finite_diff_center(Z, alphas, betas)
    area_1d = _area_1d(alphas, Za, Z0)

    # Save metrics
    metrics = {
        "Z0": Z0,
        "curv_lam1": lam1, "curv_lam2": lam2,
        "curv_mean": (lam1 + lam2) / 2.0,
        "curv_anisotropy": (lam1 / lam2) if lam2 != 0 else float("nan"),
        "grad_norm_center": grad0,
        "edge_blowup_p90": ebi,
        "area_1d": area_1d,
        "radial_deltas": radials
    }
    metrics.update(basin_map)
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print("Done:")
    print(f" - 2D surface fig : {os.path.join(args.out, 'figs/surface_2d.png')}")
    print(f" - 3D surface fig : {os.path.join(args.out, 'figs/surface_3d.png')}")
    print(f" - 1D profile fig : {os.path.join(args.out, 'figs/profile_1d.png')}")
    print(f" - Curvature fits  : {os.path.join(args.out, 'figs/curvature_fit_alpha.png')}, "
          f"{os.path.join(args.out, 'figs/curvature_fit_beta.png')}")
    print(f" - Radial plot     : {os.path.join(args.out, 'figs/radial_deltas.png')}")
    print(f" - Arrays in       : {os.path.join(args.out, 'npy')}")
    print(f" - Meta            : {os.path.join(args.out, 'meta.json')}")
    print(f" - Metrics         : {os.path.join(args.out, 'metrics.json')}")

if __name__ == "__main__":
    main()



