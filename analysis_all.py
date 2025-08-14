# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
analysis_D.py - one-stop analysis for VML experiments (classification & forecasting)

What it does (per dataset x arch experiment root):
- Scans run folders, infers condition (BASE/SL/VPL/VML) and seed
- Aggregates final metrics (acc or mae) across seeds
- Runs paired tests (Base vs SL/VPL/VML) and effect sizes
- Bootstraps SD ratio CIs (variance reduction evidence)
- Computes training-loss volatility (rolling std) per run and averages by condition
- Exports tables (CSV) and paper-grade figures (PNG) with clear filenames
- Produces a short Markdown report linking outputs
- (Patch D) Plots internals from internals_seed_*.csv (SL lambda/sigma, VPL lambda/v_ema)
- (Patch D) Optional step-level preview if batches_seed_*.csv exists
- (New) Three zoomed step previews (early/middle/late phases), each showing batches from K epochs
"""

import argparse
import os
import re
import csv
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import sys
import subprocess  # NEW
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # headless
import matplotlib.pyplot as plt

# Optional SciPy (for robust stats). We fall back to simple approximations if missing.
try:
    from scipy import stats as sps
except Exception:
    sps = None


# ----------------------------
# Helpers: file discovery + parsing
# ----------------------------

COND_ORDER = ["base", "sl", "vpl", "vml"]

def infer_condition_from_name(name: str) -> Optional[str]:
    """Infer condition from a folder name, checking for unique tokens in order of specificity."""
    lower = name.lower()
    if "vml" in lower: return "vml"
    if "vpl" in lower: return "vpl"
    if "sl"  in lower: return "sl"
    if "base" in lower or "baseline" in lower: return "base"
    return None


def find_seed_in_training_csv(run_dir: Path) -> Optional[int]:
    """Look for training_loss_seed_<SEED>.csv and return SEED if found."""
    for p in run_dir.glob("training_loss_seed_*.csv"):
        m = re.search(r"training_loss_seed_(\d+)\.csv$", p.name)
        if m:
            return int(m.group(1))
    return None


def read_log_txt(log_path: Path) -> pd.DataFrame:
    """Read the trainer's log.txt as a dataframe."""
    df = pd.read_csv(log_path, header=0)
    # Normalize column names: 'tr_c 0 acc' -> 'tr_c0_acc'
    df.columns = [c.strip().replace(" ", "_").replace("__", "_") for c in df.columns]
    return df


def last_row_metric(df: pd.DataFrame, higher_is_better: bool) -> Tuple[float, float]:
    """Return (final_metric, final_loss) from the last epoch row."""
    last = df.iloc[-1]
    te_loss = float(last["te_loss"]) if "te_loss" in df.columns else np.nan
    if "te_acc" in df.columns and higher_is_better:
        metric = float(last["te_acc"])
    else:
        metric = float(last["te_loss"]) if "te_loss" in df.columns else te_loss
    return metric, te_loss


def rolling_std(x: np.ndarray, window: int) -> np.ndarray:
    """Rolling standard deviation with a centered window (pad with NaN at edges)."""
    if window <= 1 or len(x) < window:
        return np.full_like(x, np.nan, dtype=float)
    out = np.full(len(x), np.nan, dtype=float)
    half = window // 2
    for i in range(len(x)):
        lo = max(0, i - half)
        hi = min(len(x), i + half + 1)
        seg = x[lo:hi]
        if len(seg) >= 2:
            out[i] = float(np.std(seg, ddof=1))
    return out


# ----------------------------
# Statistics helpers
# ----------------------------

def ci_mean(values: np.ndarray, alpha: float = 0.05) -> Tuple[float, float]:
    """95% CI for the mean via t-interval if scipy is available; else normal approx."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    mean = float(np.mean(values)) if n else np.nan
    sd = float(np.std(values, ddof=1)) if n > 1 else 0.0
    if n < 2 or sd == 0.0:
        return mean, mean
    se = sd / math.sqrt(n)
    if sps is not None:
        tcrit = sps.t.ppf(1 - alpha/2, df=n-1)
    else:
        tcrit = 1.96
    return mean - tcrit * se, mean + tcrit * se


def paired_t_wilcoxon_and_d(diff: np.ndarray) -> Dict[str, float]:
    """Paired t-test, Wilcoxon signed-rank (if SciPy), and Cohen's d for paired data."""
    diff = np.asarray(diff, dtype=float)
    res = {"t_p": np.nan, "wilcoxon_p": np.nan, "cohens_d_paired": np.nan}
    if len(diff) < 2:
        return res
    mean = diff.mean()
    sd = diff.std(ddof=1) if len(diff) > 1 else 0.0
    res["cohens_d_paired"] = float(mean / sd) if sd > 0 else np.nan
    if sps is not None:
        tstat, tp = sps.ttest_1samp(diff, 0.0)
        res["t_p"] = float(tp)
        try:
            nz = diff[diff != 0]
            if len(nz) >= 1:
                wstat, wp = sps.wilcoxon(nz)
                res["wilcoxon_p"] = float(wp)
        except Exception:
            pass
    return res


def bootstrap_sd_ratio_ci(base_vals: np.ndarray, alt_vals: np.ndarray,
                          n_boot: int = 5000, seed: int = 123) -> Tuple[float, float, float]:
    """Bootstrap CI for SD ratio = sd(alt)/sd(base)."""
    rng = np.random.default_rng(seed)
    base_vals = np.asarray(base_vals, dtype=float)
    alt_vals  = np.asarray(alt_vals, dtype=float)
    n = min(len(base_vals), len(alt_vals))
    base_vals, alt_vals = base_vals[:n], alt_vals[:n]
    obs = (np.std(alt_vals, ddof=1) / np.std(base_vals, ddof=1)) if np.std(base_vals, ddof=1) > 0 else np.nan
    if n < 3 or np.isnan(obs):
        return obs, np.nan, np.nan
    boot = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        b = base_vals[idx]
        a = alt_vals[idx]
        sd_b = np.std(b, ddof=1)
        sd_a = np.std(a, ddof=1)
        boot.append(sd_a / sd_b if sd_b > 0 else np.nan)
    boot = [x for x in boot if not np.isnan(x)]
    q = np.quantile(boot, [0.025, 0.975])
    return obs, float(q[0]), float(q[1])


# ----------------------------
# Loading runs into a tidy DataFrame
# ----------------------------

def load_runs(root: Path, dataset: str, arch: str, higher_is_better: bool) -> Tuple[pd.DataFrame, Dict[str, List[pd.Series]]]:
    """Scan root for run directories and build a dataframe of final metrics.
       Also return per-run training curves (dict[condition] -> list of pandas Series of train loss).
    """
    rows = []
    curves: Dict[str, List[pd.Series]] = {"base": [], "sl": [], "vpl": [], "vml": []}
    for run_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        cond = infer_condition_from_name(run_dir.name)
        if cond is None:
            continue
        log = run_dir / "log.txt"
        if not log.exists():
            continue
        # seed
        seed = find_seed_in_training_csv(run_dir)
        # read log
        df = read_log_txt(log)
        metric, te_loss = last_row_metric(df, higher_is_better=higher_is_better)
        rows.append({
            "run_dir": str(run_dir),
            "dataset": dataset,
            "arch": arch,
            "condition": cond,
            "seed": seed,
            "final_metric": float(metric),
            "final_test_loss": float(te_loss),
            "epochs": int(len(df))  # store length for total-epochs inference
        })
        # training curve (use per-epoch tr_loss from log.txt)
        if "tr_loss" in df.columns:
            s = pd.Series(df["tr_loss"].astype(float).to_numpy(), name=f"{cond}_seed{seed}")
            curves[cond].append(s)

    res = pd.DataFrame(rows).dropna(subset=["final_metric"])
    if "seed" in res.columns:
        try:
            res["seed"] = res["seed"].astype("Int64")
        except Exception:
            pass
    return res, curves


# ----------------------------
# Additional loaders (Patch D)
# ----------------------------

def load_internals_csvs(root: Path):
    by_cond = {c: [] for c in COND_ORDER}
    for run_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        cond = infer_condition_from_name(run_dir.name)
        if cond is None: continue
        for p in run_dir.glob("internals_seed_*.csv"):
            try:
                df = pd.read_csv(p).sort_values("epoch")
                by_cond[cond].append(df)
            except Exception:
                pass
    return by_cond

def load_batch_csvs(root: Path):
    by_cond = {c: [] for c in COND_ORDER}
    for run_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        cond = infer_condition_from_name(run_dir.name)
        if cond is None: continue
        for p in run_dir.glob("batches_seed_*.csv"):
            try:
                df = pd.read_csv(p)
                by_cond[cond].append(df)
            except Exception:
                pass
    return by_cond


# ----------------------------
# Zoom helpers (NEW)
# ----------------------------

def pick_zoom_centers(total_epochs: int, zoom_epochs_arg: str) -> Tuple[int, int, int]:
    """Pick center epochs for early/middle/late."""
    if isinstance(zoom_epochs_arg, str) and zoom_epochs_arg.strip().lower() != "auto":
        try:
            E, M, L = [int(x) for x in zoom_epochs_arg.split(",")]
            E = min(max(1, E), total_epochs)
            M = min(max(1, M), total_epochs)
            L = min(max(1, L), total_epochs)
            return E, M, L
        except Exception:
            pass
    # auto centers at 5%, 50%, 95%
    E = max(1, int(round(0.05 * total_epochs)))
    M = max(1, int(round(0.50 * total_epochs)))
    L = max(1, int(round(0.95 * total_epochs)))
    E = min(max(1, E), total_epochs)
    M = min(max(1, M), total_epochs)
    L = min(max(1, L), total_epochs)
    return E, M, L

def pick_epoch_list_around(center: int, total_epochs: int, k: int) -> List[int]:
    """Return k epochs around a center within a small window."""
    if k <= 1:
        return [min(max(1, center), total_epochs)]
    # window size ~4% of total epochs, at least 2
    window = max(2, int(round(0.04 * total_epochs)))
    left = max(1, center - window // 2)
    right = min(total_epochs, left + window)
    left = max(1, min(left, center))
    right = min(total_epochs, max(right, center))
    if right == left:
        return [left]
    vals = np.linspace(left, right, num=k)
    picks = sorted({int(round(v)) for v in vals})
    step = max(1, (right - left) // max(1, k - 1))
    while len(picks) < k:
        cand = picks[-1] + step
        if cand <= right:
            picks.append(cand)
        else:
            cand2 = picks[0] - step
            if cand2 >= left:
                picks = [cand2] + picks
            else:
                break
        picks = sorted(set(picks))
    return [min(max(1, p), total_epochs) for p in picks]

def load_batches_for_epoch(run_dir: Path, seed: int, epoch: int, k_batches: int) -> Optional[pd.DataFrame]:
    """
    Expects per-batch CSV: batches_seed_<SEED>.csv with columns including:
    epoch, batch_idx, ce_loss, total_loss, sl_penalty, vpl_penalty, ...
    Returns first k_batches for that epoch ordered by batch_idx.
    """
    try:
        cand = sorted(run_dir.glob(f"batches_seed_{seed}.csv"))
        if not cand:
            return None
        dfb = pd.read_csv(cand[0])
        if "epoch" not in dfb.columns or "batch_idx" not in dfb.columns:
            return None
        dfb = dfb[dfb["epoch"] == epoch].sort_values("batch_idx")
        if dfb.empty:
            return None
        if k_batches and k_batches > 0:
            return dfb.head(k_batches).copy()
        else:
            return dfb.copy()  # all batches

    except Exception:
        return None

def plot_zoom_phase(figs_dir: Path, dataset: str, arch: str,
                    tag: str, run_name: str, seed: int,
                    epoch_list: List[int], per_epoch_data: List[pd.DataFrame]) -> Path:
    """Draw multiple epochs from one phase in one figure."""
    fig, ax = plt.subplots(figsize=(9, 5))
    added_legend = False

    for ep, dfb in zip(epoch_list, per_epoch_data):
        if dfb is None or dfb.empty:
            continue
        x = dfb["batch_idx"].to_numpy()
        cols = dfb.columns.tolist()

        if "total_loss" in cols:
            ax.plot(x, dfb["total_loss"].to_numpy(), marker="o", label=f"total@ep{ep}")
            added_legend = True
        if "ce_loss" in cols:
            ax.plot(x, dfb["ce_loss"].to_numpy(), marker="x", label=f"ce@ep{ep}")
            added_legend = True
        if "sl_penalty" in cols:
            ax.plot(x, dfb["sl_penalty"].to_numpy(), marker="^", label=f"sl@ep{ep}")
            added_legend = True
        if "vpl_penalty" in cols:
            ax.plot(x, dfb["vpl_penalty"].to_numpy(), marker="s", label=f"vpl@ep{ep}")
            added_legend = True

    ax.set_xlabel("batch index")
    ax.set_ylabel("loss")
    ax.set_title(f"{dataset.upper()} / {arch} - {tag} zoom - {run_name} - seed {seed}")
    ax.grid(True, alpha=0.3)
    if added_legend:
        ax.legend(ncol=2)

    out = figs_dir / f"figA_zoom_{tag}_{dataset}_{arch}_{run_name}_seed{seed}.png"
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out

def save_zoom_step_previews(figs_dir: Path, runs_df: pd.DataFrame,
                            dataset: str, arch: str, total_epochs: int,
                            zoom_epochs_arg: str, zoom_batches: int,
                            zoom_ep_per_phase: int) -> List[Path]:
    """
    Make three figures: early, middle, late.
    Each overlays batches from K epochs within that phase window.
    We use the first run that has per-batch data for the selected epochs.
    """
    E, M, L = pick_zoom_centers(total_epochs, zoom_epochs_arg)
    epoch_lists = {
        "early":  pick_epoch_list_around(E, total_epochs, zoom_ep_per_phase),
        "middle": pick_epoch_list_around(M, total_epochs, zoom_ep_per_phase),
        "late":   pick_epoch_list_around(L, total_epochs, zoom_ep_per_phase),
    }

    outs = []
    # Iterate phases; for each, search runs until we find one with data
    for tag, elist in epoch_lists.items():
        done = False
        for _, row in runs_df.iterrows():
            run_dir = Path(row["run_dir"])
            seed = int(row["seed"]) if not pd.isna(row["seed"]) else 1
            per_epoch = []
            any_data = False
            for ep in elist:
                dfb = load_batches_for_epoch(run_dir, seed, ep, zoom_batches)
                per_epoch.append(dfb)
                if dfb is not None and not dfb.empty:
                    any_data = True
            if not any_data:
                continue
            out = plot_zoom_phase(figs_dir, dataset, arch, tag, run_dir.name, seed, elist, per_epoch)
            outs.append(out)
            done = True
            break
        if not done:
            # placeholder if no run has batch data for that phase
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.text(0.5, 0.5, f"No per-batch data for {tag} phase.", ha="center", va="center")
            ax.axis("off")
            out = figs_dir / f"figA_zoom_{tag}_{dataset}_{arch}_placeholder.png"
            fig.tight_layout(); fig.savefig(out, dpi=200); plt.close(fig)
            outs.append(out)
    return outs


# ----------------------------
# Plotting utilities
# ----------------------------

def ensure_outdirs(out_root: Path) -> Tuple[Path, Path]:
    figs = out_root / "figs"
    tables = out_root / "tables"
    figs.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    return figs, tables


def save_violin(figs_dir: Path, df: pd.DataFrame, higher_is_better: bool, dataset: str, arch: str) -> Path:
    """
    Figure 1: Distribution of final performance across seeds.
    Robust against missing/empty/NaN groups. Prints group sizes for debugging.
    """
    metric_name = "Accuracy (%)" if higher_is_better else "MAE (lower better)"
    figs_dir.mkdir(parents=True, exist_ok=True)

    # Basic guards
    if df is None or df.empty or "final_metric" not in df.columns:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5, "No results found (df empty or final_metric missing).",
                ha="center", va="center", fontsize=12)
        ax.axis("off")
        out = figs_dir / f"fig1_distribution_{dataset}_{arch}.png"
        fig.tight_layout(); fig.savefig(out, dpi=200); plt.close(fig)
        return out

    # Clean metric and condition
    tmp = df.copy()
    tmp = tmp.dropna(subset=["final_metric"])
    if "condition" not in tmp.columns:
        tmp["condition"] = "base"
    tmp["condition"] = tmp["condition"].astype(str).str.lower()

    # Build groups: only include conditions that exist and have >=1 value
    existing_conds = list(tmp["condition"].unique())
    order = [c for c in COND_ORDER if c in existing_conds] or sorted(existing_conds)

    groups = []
    labels = []
    debug_sizes = []
    for c in order:
        arr = pd.to_numeric(tmp.loc[tmp["condition"] == c, "final_metric"], errors="coerce").dropna().to_numpy()
        if arr.size > 0:
            groups.append(arr)
            labels.append(c.upper())
            debug_sizes.append((c, int(arr.size)))

    print("[fig1] groups to plot:", debug_sizes)

    fig, ax = plt.subplots(figsize=(8, 5))

    if len(groups) == 0:
        ax.text(0.5, 0.5, "No completed runs to plot yet.", ha="center", va="center", fontsize=12)
        ax.axis("off")
    else:
        ax.violinplot(groups, showmeans=True, showextrema=True, widths=0.85)
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels)
        ax.set_ylabel(metric_name)
        ax.set_title(f"{dataset.upper()} / {arch} - Final performance across seeds")
        ax.grid(True, axis='y', alpha=0.3)

    out = figs_dir / f"fig1_distribution_{dataset}_{arch}.png"
    fig.tight_layout(); fig.savefig(out, dpi=200); plt.close(fig)
    return out


def save_training_stability(figs_dir: Path, curves: Dict[str, List[pd.Series]],
                            rolling_window: int, dataset: str, arch: str) -> Tuple[Path, Path]:
    """
    Figure 2: Stability over training (loss trajectories and volatility).
    Left: Mean Â± SD band of total train loss per epoch (across seeds).
    Right: Mean of rolling-std (window=W) of train loss per epoch.
    """
    # Align lengths by padding with NaN to max length
    def stack_and_mean_std(series_list: List[pd.Series]) -> Tuple[np.ndarray, np.ndarray]:
        if not series_list: return np.array([]), np.array([])
        maxlen = max(len(s) for s in series_list)
        arr = np.full((len(series_list), maxlen), np.nan)
        for i, s in enumerate(series_list):
            v = s.to_numpy(dtype=float)
            arr[i, :len(v)] = v
        mean = np.nanmean(arr, axis=0)
        std = np.nanstd(arr, axis=0, ddof=1)
        return mean, std

    # Panel A: curves
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    for cond in COND_ORDER:
        mean, std = stack_and_mean_std(curves.get(cond, []))
        if mean.size == 0: continue
        x = np.arange(1, len(mean)+1)
        ax1.plot(x, mean, label=cond.upper())
        ax1.fill_between(x, mean-std, mean+std, alpha=0.15)
    ax1.set_title(f"{dataset.upper()} / {arch} - Train loss (mean Â± SD)")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Total train loss")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    out_a = figs_dir / f"fig2a_train_loss_{dataset}_{arch}.png"
    fig1.tight_layout(); fig1.savefig(out_a, dpi=200); plt.close(fig1)

    # Panel B: rolling std
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    for cond in COND_ORDER:
        series_list = curves.get(cond, [])
        if not series_list: continue
        rstd_mat = []
        for s in series_list:
            rstd = rolling_std(s.to_numpy(dtype=float), rolling_window)
            rstd_mat.append(rstd)
        maxlen = max(len(r) for r in rstd_mat)
        arr = np.full((len(rstd_mat), maxlen), np.nan)
        for i, r in enumerate(rstd_mat):
            arr[i, :len(r)] = r
        mean = np.nanmean(arr, axis=0)
        ax2.plot(np.arange(1, len(mean)+1), mean, label=cond.UPPER() if hasattr(cond, "UPPER") else cond.upper())
    ax2.set_title(f"{dataset.upper()} / {arch} - Training volatility (rolling std, W={rolling_window})")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Rolling std of train loss")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    out_b = figs_dir / f"fig2b_volatility_{dataset}_{arch}.png"
    fig2.tight_layout(); fig2.savefig(out_b, dpi=200); plt.close(fig2)
    return out_a, out_b


def save_pareto(figs_dir: Path, df: pd.DataFrame, higher_is_better: bool, dataset: str, arch: str) -> Path:
    """
    Figure 3: Pareto of variability vs performance (one point per condition).
    X = SD across seeds; Y = mean performance (Accuracy or negative MAE).
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    for cond in COND_ORDER:
        vals = df[df["condition"]==cond]["final_metric"].to_numpy()
        if len(vals) == 0: continue
        sd = np.std(vals, ddof=1) if len(vals) > 1 else 0.0
        mean = np.mean(vals)
        y = mean if higher_is_better else -mean
        ax.scatter(sd, y, label=cond.upper(), s=70)
        ax.annotate(cond.upper(), (sd, y), xytext=(5,5), textcoords="offset points")
    ax.set_xlabel("Across-seed SD")
    ax.set_ylabel("Mean performance" + (" (higher)" if higher_is_better else " (-MAE; higher is better)"))
    ax.set_title(f"{dataset.upper()} / {arch} - Variability vs performance")
    ax.grid(True, alpha=0.3)
    out = figs_dir / f"fig3_pareto_{dataset}_{arch}.png"
    fig.tight_layout(); fig.savefig(out, dpi=200); plt.close(fig)
    return out


# ----------------------------
# Patch D: Internals plots
# ----------------------------

def _stack_mean_std(list_of_arrays: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    if not list_of_arrays:
        return np.array([]), np.array([])
    maxlen = max(len(a) for a in list_of_arrays)
    arr = np.full((len(list_of_arrays), maxlen), np.nan)
    for i, a in enumerate(list_of_arrays):
        arr[i, :len(a)] = a
    mean = np.nanmean(arr, axis=0)
    std = np.nanstd(arr, axis=0, ddof=1)
    return mean, std

def save_internals_plots(figs_dir: Path, internals_by_cond, dataset: str, arch: str):
    # SL: lambda and sigma_ema
    fig1, ax1 = plt.subplots(figsize=(8,5))
    fig2, ax2 = plt.subplots(figsize=(8,5))
    for cond in COND_ORDER:
        lst = internals_by_cond.get(cond, [])
        if not lst: continue
        lam_list = []
        sig_list = []
        for df in lst:
            if "sl_lambda" in df.columns:
                lam_list.append(df["sl_lambda"].astype(float).to_numpy())
            if "sl_sigma_ema" in df.columns:
                sig_list.append(df["sl_sigma_ema"].astype(float).to_numpy())
        lam_mean, lam_std = _stack_mean_std(lam_list)
        sig_mean, sig_std = _stack_mean_std(sig_list)
        if lam_mean.size:
            x = np.arange(len(lam_mean))
            ax1.plot(x, lam_mean, label=cond.upper())
            ax1.fill_between(x, lam_mean-lam_std, lam_mean+lam_std, alpha=0.15)
        if sig_mean.size:
            x = np.arange(len(sig_mean))
            ax2.plot(x, sig_mean, label=cond.upper())
            ax2.fill_between(x, sig_mean-sig_std, sig_mean+sig_std, alpha=0.15)
    ax1.set_title(f"{dataset.upper()} / {arch} - SL gain lambda_t (mean Â± SD)")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("lambda_t"); ax1.grid(True, alpha=0.3); ax1.legend()
    ax2.set_title(f"{dataset.upper()} / {arch} - SL volatility sigma_ema (mean Â± SD)")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("sigma_ema"); ax2.grid(True, alpha=0.3); ax2.legend()
    out1 = figs_dir / f"figA_sl_lambda_{dataset}_{arch}.png"
    out2 = figs_dir / f"figA_sl_sigma_{dataset}_{arch}.png"
    fig1.tight_layout(); fig1.savefig(out1, dpi=200); plt.close(fig1)
    fig2.tight_layout(); fig2.savefig(out2, dpi=200); plt.close(fig2)

    # VPL: lambda and v_ema
    fig3, ax3 = plt.subplots(figsize=(8,5))
    fig4, ax4 = plt.subplots(figsize=(8,5))
    for cond in COND_ORDER:
        lst = internals_by_cond.get(cond, [])
        if not lst: continue
        lam_list = []
        vema_list = []
        for df in lst:
            if "vpl_lambda" in df.columns:
                lam_list.append(df["vpl_lambda"].astype(float).to_numpy())
            if "vpl_v_ema" in df.columns:
                vema_list.append(df["vpl_v_ema"].astype(float).to_numpy())
        lam_mean, lam_std = _stack_mean_std(lam_list)
        vema_mean, vema_std = _stack_mean_std(vema_list)
        if lam_mean.size:
            x = np.arange(len(lam_mean))
            ax3.plot(x, lam_mean, label=cond.upper())
            ax3.fill_between(x, lam_mean-lam_std, lam_mean+lam_std, alpha=0.15)
        if vema_mean.size:
            x = np.arange(len(vema_mean))
            ax4.plot(x, vema_mean, label=cond.upper())
            ax4.fill_between(x, vema_mean-vema_std, vema_mean+vema_std, alpha=0.15)
    ax3.set_title(f"{dataset.upper()} / {arch} - VPL gain lambda_t (mean Â± SD)")
    ax3.set_xlabel("Epoch"); ax3.set_ylabel("lambda_t"); ax3.grid(True, alpha=0.3); ax3.legend()
    ax4.set_title(f"{dataset.upper()} / {arch} - VPL v_ema (mean Â± SD)")
    ax4.set_xlabel("Epoch"); ax4.set_ylabel("v_ema"); ax4.grid(True, alpha=0.3); ax4.legend()
    out3 = figs_dir / f"figA_vpl_lambda_{dataset}_{arch}.png"
    out4 = figs_dir / f"figA_vpl_vema_{dataset}_{arch}.png"
    fig3.tight_layout(); fig3.savefig(out3, dpi=200); plt.close(fig3)
    fig4.tight_layout(); fig4.savefig(out4, dpi=200); plt.close(fig4)

    return [out1, out2, out3, out4]

def save_batch_preview(figs_dir: Path, batches_by_cond, dataset: str, arch: str):
    # Plot step-level total loss for first available run per condition (sanity peek)
    any_plotted = False
    fig, ax = plt.subplots(figsize=(8,5))
    for cond in COND_ORDER:
        runs = batches_by_cond.get(cond, [])
        if not runs: continue
        df = runs[0].sort_values(["epoch","batch_idx"])
        if "total_loss" not in df.columns: continue
        y = df["total_loss"].astype(float).to_numpy()
        ax.plot(np.arange(len(y)), y, label=cond.UPPER() if hasattr(cond, "UPPER") else cond.upper(), alpha=0.85)
        any_plotted = True
    if not any_plotted:
        plt.close(fig)
        return None
    ax.set_title(f"{dataset.upper()} / {arch} - Step-level total loss (preview)")
    ax.set_xlabel("Batch step"); ax.set_ylabel("total_loss"); ax.grid(True, alpha=0.3); ax.legend()
    out = figs_dir / f"figA_step_preview_{dataset}_{arch}.png"
    fig.tight_layout(); fig.savefig(out, dpi=200); plt.close(fig)
    return out


# ----------------------------
# Tables
# ----------------------------

def make_summary_tables(tables_dir: Path, df: pd.DataFrame, higher_is_better: bool) -> Tuple[Path, Path]:
    """
    Summary by condition + paired tests vs base.
    """
    conds_present = [c for c in COND_ORDER if c in df["condition"].unique()]
    pivot = df.pivot_table(index="seed", columns="condition", values="final_metric", aggfunc="first")
    pivot = pivot[conds_present].sort_index()
    wide_csv = tables_dir / "per_seed_metrics.csv"
    pivot.to_csv(wide_csv)

    rows = []
    base_vals = df[df["condition"]=="base"]["final_metric"].to_numpy()
    for cond in conds_present:
        vals = df[df["condition"]==cond]["final_metric"].to_numpy()
        n = len(vals)
        mean = float(np.mean(vals)) if n else np.nan
        sd = float(np.std(vals, ddof=1)) if n > 1 else np.nan
        cv = float(sd / mean) if n > 1 and mean != 0 else np.nan
        lo, hi = ci_mean(vals) if n else (np.nan, np.nan)
        rows.append({
            "condition": cond,
            "n": n,
            "mean": mean,
            "sd": sd,
            "cv": cv,
            "ci95_low": lo,
            "ci95_high": hi,
        })
    summary = pd.DataFrame(rows)

    if len(base_vals) >= 2 and np.std(base_vals, ddof=1) > 0:
        base_sd = float(np.std(base_vals, ddof=1))
        sd_red = []
        sd_ratio_ci = []
        for cond in conds_present:
            vals = df[df["condition"]==cond]["final_metric"].to_numpy()
            if len(vals) >= 2:
                sd = float(np.std(vals, ddof=1))
                pct = 100.0 * (1.0 - sd/base_sd)
                ratio, lo, hi = bootstrap_sd_ratio_ci(base_vals, vals)
            else:
                pct, ratio, lo, hi = (np.nan, np.nan, np.nan, np.nan)
            sd_red.append(pct)
            sd_ratio_ci.append((ratio, lo, hi))
        summary["sd_reduction_vs_base_%"] = sd_red
        summary["sd_ratio_vs_base"] = [r[0] for r in sd_ratio_ci]
        summary["sd_ratio_ci95_low"] = [r[1] for r in sd_ratio_ci]
        summary["sd_ratio_ci95_high"] = [r[2] for r in sd_ratio_ci]

    def paired_diff(cond):
        if "base" not in pivot.columns or cond not in pivot.columns:
            return None
        joined = pd.concat([pivot["base"], pivot[cond]], axis=1, join="inner").dropna()
        if joined.shape[0] < 2: return None
        diff = joined[cond] - joined["base"]
        if not higher_is_better:
            diff = -diff
        return diff.to_numpy()

    paired_rows = []
    for cond in conds_present:
        if cond == "base": continue
        d = paired_diff(cond)
        if d is None:
            paired_rows.append({"vs": f"{cond} vs base", "paired_n": 0,
                                "t_p": np.nan, "wilcoxon_p": np.nan, "cohens_d_paired": np.nan})
            continue
        stats = paired_t_wilcoxon_and_d(d)
        paired_rows.append({"vs": f"{cond} vs base", "paired_n": len(d), **stats})

    tests = pd.DataFrame(paired_rows)

    summary_csv = tables_dir / "summary_by_condition.csv"
    tests_csv = tables_dir / "paired_tests_vs_base.csv"
    summary.to_csv(summary_csv, index=False)
    tests.to_csv(tests_csv, index=False)
    return summary_csv, tests_csv


# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True, help="Root folder containing run subdirs (each with log.txt).")
    ap.add_argument("--dataset", type=str, required=True, help="Dataset name (e.g., cifar10).")
    ap.add_argument("--arch", type=str, required=True, help="Architecture name (e.g., resnet14).")
    ap.add_argument("--out", type=str, required=True, help="Output directory for figures/tables/report.")
    ap.add_argument("--rolling_window", type=int, default=5, help="Window for rolling std (training volatility).")
    ap.add_argument("--higher_is_better", type=int, default=1, help="1=accuracy-like; 0=MAE-like.")
    # NEW: zoom controls
    ap.add_argument("--zoom_batches", type=int, default=0,
                    help="Batches per epoch to plot inside each zoom figure. 0 = all batches.")
    ap.add_argument("--zoom_epochs", type=str, default="auto",
                    help="Centers for early,middle,late. 'auto' or 'E,M,L' like '5,100,190'.")
    ap.add_argument("--zoom_ep_per_phase", type=int, default=3,
                    help="How many epochs to include per phase figure. Set 1 for a single epoch per phase.")
    ap.add_argument("--auto_push_wandb", type=int, default=1,
                    help="If 1, run wandb_push_analysis.py on --out after analysis.")
    ap.add_argument("--wandb_project", type=str, default="vml-analysis",
                    help="W&B project to push analysis artifacts to.")
    ap.add_argument("--wandb_entity", type=str, default=None,
                    help="Optional W&B entity (team/user).")

    args = ap.parse_args()
    
    root = Path(args.root)
    out_root = Path(args.out)
    figs_dir, tables_dir = ensure_outdirs(out_root)
    hib = bool(args.higher_is_better)

    print(f"[load] scanning runs in: {root}")
    df, curves = load_runs(root, args.dataset, args.arch, higher_is_better=hib)
    if df.empty:
        raise SystemExit("No runs found with log.txt (and identifiable condition).")

    # Save the raw run table
    df_out = tables_dir / "runs_raw.csv"
    df.to_csv(df_out, index=False)

    # Infer total epochs from runs (max length of logs); fallback 200
    total_epochs = 200
    if "epochs" in df.columns and not df["epochs"].isna().all():
        try:
            total_epochs = int(df["epochs"].max())
        except Exception:
            pass

    # Figures
    print("[plot] fig1 distribution")
    f1 = save_violin(figs_dir, df, higher_is_better=hib, dataset=args.dataset, arch=args.arch)

    print("[plot] fig2 stability panels")
    f2a, f2b = save_training_stability(figs_dir, curves, rolling_window=args.rolling_window,
                                       dataset=args.dataset, arch=args.arch)

    print("[plot] fig3 pareto")
    f3 = save_pareto(figs_dir, df, higher_is_better=hib, dataset=args.dataset, arch=args.arch)

    # Patch D: internals and optional batch preview
    print("[plot] internals (SL/VPL) and optional step preview")
    internals_by_cond = load_internals_csvs(root)
    internals_figs = save_internals_plots(figs_dir, internals_by_cond, args.dataset, args.arch)
    batches_by_cond = load_batch_csvs(root)
    batch_preview = save_batch_preview(figs_dir, batches_by_cond, args.dataset, args.arch)

    # NEW: zoomed step previews (early/middle/late)
    print("[plot] zoomed step previews (early/middle/late)")
    zoom_figs = save_zoom_step_previews(
        figs_dir=figs_dir,
        runs_df=df[["run_dir", "seed"]].drop_duplicates(),
        dataset=args.dataset,
        arch=args.arch,
        total_epochs=total_epochs,
        zoom_epochs_arg=args.zoom_epochs,
        zoom_batches=args.zoom_batches,
        zoom_ep_per_phase=args.zoom_ep_per_phase
    )

    # Tables
    print("[table] summaries + paired tests")
    t1, t2 = make_summary_tables(tables_dir, df, higher_is_better=hib)

    # Tiny markdown report
    

    report_path = out_root / "report.md"
    with open(report_path, "w") as out_f:
        out_f.write(f"# VML Analysis - {args.dataset.upper()} / {args.arch}\n\n")

        out_f.write("## Figures\n")
        out_f.write(f"- Fig 1: Final performance across seeds — `{Path(f1).name}`\n")
        out_f.write(f"- Fig 2a: Train loss (mean ± SD) — `{Path(f2a).name}`\n")
        out_f.write(f"- Fig 2b: Training volatility (rolling std) — `{Path(f2b).name}`\n")
        out_f.write(f"- Fig 3: Variability vs performance (Pareto) — `{Path(f3).name}`\n\n")

        out_f.write("## Figures (internals / appendix)\n")
        for p in internals_figs:
            if p:  # guard in case any are None
                out_f.write(f"- `{Path(p).name}`\n")

        if batch_preview:
            out_f.write(f"- Step-level preview: `{Path(batch_preview).name}`\n")

    # If you have zoomed previews variables, list them too (if your script defines them):
        try:
            if zoom_figs:  # e.g., list of paths from your zoom function
                out_f.write("\n## Zoomed step previews\n")
                for zp in zoom_figs:
                    if zp:
                        out_f.write(f"- `{Path(zp).name}`\n")
        except NameError:
            pass

        out_f.write("\n## Tables\n")
        out_f.write(f"- Runs (raw): `{Path(df_out).name}`\n")
        out_f.write(f"- Summary by condition: `{Path(t1).name}`\n")
        out_f.write(f"- Paired tests vs BASE: `{Path(t2).name}`\n\n")

        out_f.write("## How to read\n")
        out_f.write("- Goal: VML reduces across-seed variability without hurting mean performance.\n")
        out_f.write("- Fig 1: Narrower spread and similar or better mean for VML vs BASE.\n")
        out_f.write("- Fig 2a/2b: Smoother training and lower volatility under SL/VPL/VML.\n")
        out_f.write("- Fig 3: VML near the best corner (low SD; high mean).\n")
        out_f.write("- Internals: SL/VPL lambda_t should stabilize; sigma_ema/v_ema show volatility tracking.\n")

    print("\nDone!")
    print(f"- Figures: {figs_dir}")
    print(f"- Tables:  {tables_dir}")
    print(f"- Report:  {report_path}")
        # --- Auto push to W&B (optional) ---
    if int(args.auto_push_wandb) == 1:
        push_cmd = [  
            # use same interpreter
            sys.executable if 'sys' in globals() else 'python',
            "wandb_push_analysis.py",
            "--project", args.wandb_project,
            "--analysis_dir", str(out_root),
        ]
        if args.wandb_entity:
            push_cmd += ["--entity", args.wandb_entity]
        try:
            print(f"[wandb] pushing analysis via: {' '.join(push_cmd)}")
            subprocess.run(push_cmd, check=False)
        except Exception as e:
            print(f"[wandb] push skipped (error invoking wandb_push_analysis.py): {e}")

if __name__ == "__main__":
    main()

