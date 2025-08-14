#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
landscape.py — Filter-normalized 2D/1D loss landscape around a trained solution.

Usage (example):
  python landscape.py \
    --dataset cifar10 --arch resnet14 \
    --ckpt cifar10_models/R14_C10_BASE_SEED01/model.ckpt \
    --grid 41 --span 1.0 --batch_limit 5000 \
    --out analysis/landscape/c10-resnet14-base

Then repeat for SL / VPL / VML; compare the figures.

What it computes:
- 2D grid: CE loss on a plane theta + a*d1 + b*d2 (filter-normalized directions)
- 1D profile: CE vs step along d1
Saves:
- figs/surface_2d.png, figs/profile_1d.png
- npy/surface.npy (the grid), npy/alphas.npy, npy/betas.npy
"""

import argparse, os, math, sys, pathlib, numpy as np
import torch, torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import data
import models_clf

# ---------------------------
# Param utils
# ---------------------------

def _param_iterator(model):
    for n, p in model.named_parameters():
        if (not p.requires_grad): 
            continue
        yield n, p

def _filterwise_norm_like(weight, direction):
    """
    Scale 'direction' tensor so each output filter/row has the same norm scale as weight's filter.
    Handles Linear [out,in], Conv2d [out,in,kh,kw], and 1D biases.
    """
    if weight.ndim == 4:
        # Conv2d: [out,in,kh,kw] normalize per out-channel
        d = direction.clone()
        w = weight.data
        out = w.shape[0]
        d = d.reshape(out, -1)
        w = w.reshape(out, -1)
        d_norm = d.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
        w_norm = w.norm(p=2, dim=1, keepdim=True)
        scaled = (d / d_norm) * w_norm
        return scaled.view_as(direction)
    elif weight.ndim == 2:
        # Linear: [out,in] normalize per out-row
        d = direction.clone()
        w = weight.data
        d_norm = d.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
        w_norm = w.norm(p=2, dim=1, keepdim=True)
        return (d / d_norm) * w_norm
    elif weight.ndim == 1:
        # Bias/BN scale: scale to same l2 as weight
        d = direction.clone()
        d_norm = d.norm(p=2).clamp_min(1e-12)
        w_norm = weight.data.norm(p=2)
        return (d / d_norm) * w_norm
    else:
        # Fallback: make unit and keep small
        d = direction.clone()
        n = d.norm().clamp_min(1e-12)
        return d / n

def make_filter_normalized_direction(model, seed=123):
    """
    Build a direction {name -> tensor} aligned with model params, filter-normalized.
    """
    g = torch.Generator(device='cpu')
    g.manual_seed(seed)
    direction = {}
    for n, p in _param_iterator(model):
        r = torch.randn_like(p, generator=g)
        direction[n] = _filterwise_norm_like(p, r)
    return direction

def add_direction(model, base_state, a, d1, b=0.0, d2=None):
    """
    Set model params to base + a*d1 (+ b*d2).
    """
    with torch.no_grad():
        for n, p in _param_iterator(model):
            w0 = base_state[n]
            p.copy_(w0 + a * d1[n] + (b * d2[n] if d2 is not None else 0.0))

def snapshot_state(model):
    """
    Return a dict {name: tensor} of current parameters (detached copy).
    """
    state = {}
    for n, p in _param_iterator(model):
        state[n] = p.detach().clone()
    return state

# ---------------------------
# Loss eval
# ---------------------------

@torch.no_grad()
def ce_on_loader(model, loader, device, batch_limit=None):
    """
    Compute mean CE (classification) on 'loader'. Uses eval() and no BN re-estimation.
    If batch_limit is set, only that many samples are evaluated.
    """
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

def save_surface(alphas, betas, Z, outdir):
    os.makedirs(outdir, exist_ok=True)
    # NPY
    npy_dir = os.path.join(outdir, "npy")
    fig_dir = os.path.join(outdir, "figs")
    os.makedirs(npy_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)
    np.save(os.path.join(npy_dir, "surface.npy"), Z)
    np.save(os.path.join(npy_dir, "alphas.npy"), alphas)
    np.save(os.path.join(npy_dir, "betas.npy"), betas)

    # Figure
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

def save_profile(steps, vals, outdir):
    os.makedirs(outdir, exist_ok=True)
    fig_dir = os.path.join(outdir, "figs")
    os.makedirs(fig_dir, exist_ok=True)
    np.save(os.path.join(outdir, "npy/profile_steps.npy"), steps)
    np.save(os.path.join(outdir, "npy/profile_vals.npy"), vals)

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
# Main
# ---------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["cifar10","cifar100"])
    ap.add_argument("--arch", required=True, help="e.g., resnet14, vgg16, shufflenet1")
    ap.add_argument("--ckpt", required=True, help="Path to model.ckpt from training")
    ap.add_argument("--grid", type=int, default=41, help="Grid resolution per axis (odd recommended)")
    ap.add_argument("--span", type=float, default=1.0, help="Max |alpha| and |beta| (relative scale)")
    ap.add_argument("--profile_span", type=float, default=1.0, help="Span for 1D profile")
    ap.add_argument("--batch_limit", type=int, default=5000, help="Max samples to evaluate (speed)")
    ap.add_argument("--seed", type=int, default=123, help="Direction RNG seed")
    ap.add_argument("--out", required=True, help="Output directory for figs/npy")
    args = ap.parse_args()

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Load data (eval only; reuse your data.get_* helpers)
    testloader = data.get_testloader(args.dataset, batch_size=512)

    # Build model skeleton to match training
    num_classes = 10 if args.dataset=="cifar10" else 100
    net = getattr(models_clf, args.arch)(num_classes=num_classes)
    if use_cuda: net = net.to(device)

    # Load checkpoint weights only (no optimizer/scheduler needed)
    ckpt = torch.load(args.ckpt, map_location=device)
    net.load_state_dict(ckpt["model_state_dict"])
    net.eval()

    # Base param snapshot
    base_state = snapshot_state(net)

    # Directions (filter-normalized)
    d1 = make_filter_normalized_direction(net, seed=args.seed)
    d2 = make_filter_normalized_direction(net, seed=args.seed+1)

    # 2D grid
    g = args.grid
    alphas = np.linspace(-args.span, args.span, g).astype(np.float32)
    betas  = np.linspace(-args.span, args.span, g).astype(np.float32)
    Z = np.zeros((g, g), dtype=np.float32)

    with torch.no_grad():
        for i, a in enumerate(alphas):
            for j, b in enumerate(betas):
                add_direction(net, base_state, a, d1, b, d2)
                Z[i, j] = ce_on_loader(net, testloader, device, batch_limit=args.batch_limit)

        # restore weights
        add_direction(net, base_state, 0.0, d1, 0.0, d2)

    # Save surface
    save_surface(alphas, betas, Z, os.path.join(args.out))

    # 1D profile along d1
    steps = np.linspace(-args.profile_span, args.profile_span, g).astype(np.float32)
    vals = np.zeros_like(steps)
    with torch.no_grad():
        for k, a in enumerate(steps):
            add_direction(net, base_state, a, d1)
            vals[k] = ce_on_loader(net, testloader, device, batch_limit=args.batch_limit)
        add_direction(net, base_state, 0.0, d1)

    # Save profile
    npy_dir = os.path.join(args.out, "npy")
    os.makedirs(npy_dir, exist_ok=True)
    save_profile(steps, vals, os.path.join(args.out))

    print("Done:")
    print(" - 2D surface:   {}/figs/surface_2d.png".format(args.out))
    print(" - 1D profile:   {}/figs/profile_1d.png".format(args.out))
    print(" - NPY grids in: {}/npy".format(args.out))

if __name__ == "__main__":
    main()
