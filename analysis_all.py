# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
analysis_multi.py - unified analysis for VML experiments (classification)

New in this version
-------------------
- Supports hyperparameter sweep comparison via --hp_tag_prefix (e.g., VW ? VW002/VW005/...).
- Dynamic condition discovery: when sweeping, conditions are the tag tokens found in run dir names.
- Tables now compare each condition against a flexible "ref" (first condition in order, or 'base' if present).
- Backwards compatible: without --hp_tag_prefix, it falls back to base/sl/vpl/vml comparison.
- Fig 1b: Box plots of final metrics across seeds (variance-focused).
- Fig 1c: KDE density plots of final metrics across seeds (density on y, metric on x).
- NEW: Reactivity overlays (SL/VPL internals vs test loss) + lag-correlation table.

Other features (as before)
-------------------------
- Fig 2 uses TEST loss (mean +/- SD + volatility).
- Internals plots mask leading zeros (pre-activation/warmup) as NaN.
- Zoomed step previews overlay ALL conditions.
"""

import argparse
import os
import re
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict
import sys
import subprocess

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    from scipy import stats as sps
except Exception:
    sps = None

# Default condition order (used when NOT doing a sweep)
COND_ORDER: List[str] = ["base", "sl", "vpl", "vml"]

# When set (e.g., "VW"), runs are grouped by the tag token (VW002, VW005, ...)
# and only those runs are analyzed.
HP_TAG_PREFIX: Optional[str] = None


# ---------------- Basic helpers ----------------

def infer_condition_from_name(name: str) -> Optional[str]:
    """
    - If HP_TAG_PREFIX is set (e.g., 'VW'), try to find a token like 'VW002' in the directory name.
      If found, return it uppercased (e.g., 'VW002'). If not found, ignore this run (return None).
    - Otherwise, fall back to canonical conditions: base/sl/vpl/vml derived from folder name.
    """
    lower = name.lower()

    if HP_TAG_PREFIX:
        pref = HP_TAG_PREFIX.lower()
        toks = re.split(r"[\/_\-]+", lower)
        for t in toks:
            if t.startswith(pref):
                return t.upper()
        return None  # not part of this sweep

    if "vml" in lower: return "vml"
    if "vpl" in lower: return "vpl"
    if "sl"  in lower: return "sl"
    if "base" in lower or "baseline" in lower: return "base"
    return None


def find_seed_in_training_csv(run_dir: Path) -> Optional[int]:
    for p in run_dir.glob("training_loss_seed_*.csv"):
        m = re.search(r"training_loss_seed_(\d+)\.csv$", p.name)
        if m:
            return int(m.group(1))
    return None


def read_log_txt(log_path: Path) -> pd.DataFrame:
    df = pd.read_csv(log_path, header=0)
    df.columns = [c.strip().replace(" ", "_").replace("__", "_") for c in df.columns]
    return df


def last_row_metric(df: pd.DataFrame, higher_is_better: bool) -> Tuple[float, float]:
    last = df.iloc[-1]
    te_loss = float(last["te_loss"]) if "te_loss" in df.columns else np.nan
    if "te_acc" in df.columns and higher_is_better:
        metric = float(last["te_acc"])
    else:
        metric = float(last["te_loss"]) if "te_loss" in df.columns else te_loss
    return metric, te_loss


def rolling_std(x: np.ndarray, window: int) -> np.ndarray:
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


# ---------------- Stats helpers ----------------

def ci_mean(values: np.ndarray, alpha: float = 0.05) -> Tuple[float, float]:
    values = np.asarray(values, dtype=float)
    n = len(values)
    mean = float(np.mean(values)) if n else np.nan
    sd = float(np.std(values, ddof=1)) if n > 1 else 0.0
    if n < 2 or sd == 0.0:
        return mean, mean
    se = sd / math.sqrt(n)
    tcrit = sps.t.ppf(1 - alpha/2, df=n-1) if sps is not None else 1.96
    return mean - tcrit * se, mean + tcrit * se


def paired_t_wilcoxon_and_d(diff: np.ndarray):
    diff = np.asarray(diff, dtype=float)
    res = {"t_p": np.nan, "wilcoxon_p": np.nan, "cohens_d_paired": np.nan}
    if len(diff) < 2:
        return res
    mean = diff.mean()
    sd = diff.std(ddof=1) if len(diff) > 1 else 0.0
    res["cohens_d_paired"] = float(mean / sd) if sd > 0 else np.nan
    if sps is not None:
        _, tp = sps.ttest_1samp(diff, 0.0)
        res["t_p"] = float(tp)
        try:
            nz = diff[diff != 0]
            if len(nz) >= 1:
                _, wp = sps.wilcoxon(nz)
                res["wilcoxon_p"] = float(wp)
        except Exception:
            pass
    return res


def bootstrap_sd_ratio_ci(base_vals: np.ndarray, alt_vals: np.ndarray,
                          n_boot: int = 5000, seed: int = 123) -> Tuple[float, float, float]:
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
        b = base_vals[idx]; a = alt_vals[idx]
        sd_b = np.std(b, ddof=1)
        sd_a = np.std(a, ddof=1)
        boot.append(sd_a / sd_b if sd_b > 0 else np.nan)
    boot = [x for x in boot if not np.isnan(x)]
    q = np.quantile(boot, [0.025, 0.975])
    return obs, float(q[0]), float(q[1])


def best_lag_corr(a: np.ndarray, b: np.ndarray, max_lag: int = 10):
    """
    Pearson/Spearman at best lag on first-differences (reactivity).
    Positive lag means 'a' leads 'b' by 'lag' epochs: a_t -> b_{t+lag}.
    """
    def _corr(x, y):
        if len(x) < 3 or len(y) < 3:
            return np.nan, np.nan
        if sps is None:
            r = np.corrcoef(x, y)[0, 1]
            return float(r), np.nan
        pr = float(sps.pearsonr(x, y)[0]) if np.std(x) > 0 and np.std(y) > 0 else np.nan
        sr = float(sps.spearmanr(x, y)[0]) if (len(np.unique(x)) > 2 and len(np.unique(y)) > 2) else np.nan
        return pr, sr

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = np.diff(a); b = np.diff(b)

    na, nb = len(a), len(b)
    if na < 3 or nb < 3:
        return 0, np.nan, np.nan

    max_lag_eff = min(max_lag, na - 1, nb - 1)
    best = (0, np.nan, np.nan, -np.inf)  # lag, pearson, spearman, |pearson|

    for lag in range(-max_lag_eff, max_lag_eff + 1):
        if lag < 0:
            k = -lag
            x = a[k:]
            y = b[:nb - k]
        elif lag > 0:
            k = lag
            x = a[:na - k]
            y = b[k:]
        else:
            x = a
            y = b

        # Equalize lengths before masking
        n = min(len(x), len(y))
        if n < 3:
            continue
        x = x[:n]; y = y[:n]

        mask = np.isfinite(x) & np.isfinite(y)
        x = x[mask]; y = y[mask]
        if len(x) < 3:
            continue

        pr, sr = _corr(x, y)
        score = abs(pr) if np.isfinite(pr) else -np.inf
        if score > best[3]:
            best = (lag, pr, sr, score)

    return best[0], best[1], best[2]



# ---------------- Loading runs and curves ----------------

def load_runs(root: Path, dataset: str, arch: str, higher_is_better: bool
              ) -> Tuple[pd.DataFrame,
                         Dict[str, List[pd.Series]],
                         Dict[str, List[pd.Series]],
                         Dict[str, List[pd.Series]]]:
    rows = []
    curves_tr:  Dict[str, List[pd.Series]] = defaultdict(list)
    curves_acc: Dict[str, List[pd.Series]] = defaultdict(list)
    curves_teloss: Dict[str, List[pd.Series]] = defaultdict(list)

    for run_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        cond = infer_condition_from_name(run_dir.name)
        if cond is None:
            continue
        log = run_dir / "log.txt"
        if not log.exists():
            continue

        seed = find_seed_in_training_csv(run_dir)
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
            "epochs": int(len(df))
        })

        tag = f"{cond}_seed{seed if seed is not None else 'NA'}"
        if "tr_loss" in df.columns:
            curves_tr[cond].append(pd.Series(df["tr_loss"].astype(float).to_numpy(), name=tag))
        if "te_acc" in df.columns:
            curves_acc[cond].append(pd.Series(df["te_acc"].astype(float).to_numpy(), name=tag))
        if "te_loss" in df.columns:
            curves_teloss[cond].append(pd.Series(df["te_loss"].astype(float).to_numpy(), name=tag))

    res = pd.DataFrame(rows).dropna(subset=["final_metric"])
    if "seed" in res.columns:
        try:
            res["seed"] = res["seed"].astype("Int64")
        except Exception:
            pass
    return res, curves_tr, curves_acc, curves_teloss


# ---------------- Seed filtering utilities ----------------

def _parse_series_seed(name: str) -> Optional[int]:
    m = re.search(r"_seed([0-9]+)$", name or "")
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None
    return None


def filter_df_by_seed_set(df: pd.DataFrame, seed_set: Set[int]) -> pd.DataFrame:
    if not seed_set:
        return df.copy()
    return df[df["seed"].isin(list(seed_set))].copy()


def filter_curves_by_seed_set(curves: Dict[str, List[pd.Series]], seed_set: Set[int]) -> Dict[str, List[pd.Series]]:
    if not seed_set:
        return {c: list(lst) for c, lst in curves.items()}
    out: Dict[str, List[pd.Series]] = defaultdict(list)
    for cond, lst in curves.items():
        for s in lst:
            s_seed = _parse_series_seed(s.name)
            if s_seed in seed_set:
                out[cond].append(s)
    return out


def limit_first_n_seeds_per_condition(df: pd.DataFrame,
                                      curves_tr: Dict[str, List[pd.Series]],
                                      curves_acc: Dict[str, List[pd.Series]],
                                      curves_teloss: Dict[str, List[pd.Series]],
                                      n: int):
    if n <= 0:
        return df, curves_tr, curves_acc, curves_teloss
    keep: Dict[str, List[Optional[int]]] = {}
    for cond in COND_ORDER:
        sub = df[df["condition"] == cond]
        uniq = sorted([int(s) for s in sub["seed"].dropna().unique()])
        keep[cond] = uniq[:n]

    def seed_in_keep(cond: str, seed_val: Optional[float]) -> bool:
        if cond not in keep: return False
        if pd.isna(seed_val): return False
        return int(seed_val) in keep[cond]

    df2 = df[df.apply(lambda r: seed_in_keep(str(r["condition"]), r["seed"]), axis=1)].copy()

    def filt_curves(curves: Dict[str, List[pd.Series]]) -> Dict[str, List[pd.Series]]:
        out: Dict[str, List[pd.Series]] = defaultdict(list)
        for cond, lst in curves.items():
            keep_set = set(keep.get(cond, []))
            for s in lst:
                s_seed = _parse_series_seed(s.name)
                if s_seed in keep_set:
                    out[cond].append(s)
        return out

    return df2, filt_curves(curves_tr), filt_curves(curves_acc), filt_curves(curves_teloss)


# ---------------- Internals & batches loaders ----------------

def load_internals_csvs(root: Path):
    by_cond: Dict[str, List[pd.DataFrame]] = defaultdict(list)
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


def load_batch_csvs_recursive(root: Path):
    """Find batches_seed_*.csv anywhere under each run dir (handles nested or renamed subdirs)."""
    by_cond: Dict[str, List[pd.DataFrame]] = defaultdict(list)
    for run_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        cond = infer_condition_from_name(run_dir.name)
        if cond is None:
            continue
        for p in sorted(run_dir.rglob("batches_seed_*.csv")):
            try:
                df = pd.read_csv(p)
                df["__run_dir__"] = str(run_dir)
                by_cond[cond].append(df)
            except Exception:
                pass
    return by_cond


# ---------------- Zoom helpers ----------------

def pick_zoom_centers(total_epochs: int, zoom_epochs_arg: str) -> Tuple[int, int, int]:
    """
    Return 0-based epoch indices for early/middle/late.
    """
    if isinstance(zoom_epochs_arg, str) and zoom_epochs_arg.strip().lower() != "auto":
        try:
            E, M, L = [int(x) for x in zoom_epochs_arg.split(",")]
            E = min(max(0, E), total_epochs-1)
            M = min(max(0, M), total_epochs-1)
            L = min(max(0, L), total_epochs-1)
            return E, M, L
        except Exception:
            pass
    # defaults (0-based)
    E = max(0, int(round(0.05 * (total_epochs-1))))
    M = max(0, int(round(0.50 * (total_epochs-1))))
    L = max(0, int(round(0.95 * (total_epochs-1))))
    E = min(max(0, E), total_epochs-1)
    M = min(max(0, M), total_epochs-1)
    L = min(max(0, L), total_epochs-1)
    return E, M, L


def _leading_zeros_to_nan(arr: np.ndarray) -> np.ndarray:
    out = arr.astype(float).copy()
    i = 0
    while i < len(out) and out[i] == 0.0:
        out[i] = np.nan
        i += 1
    return out


# ---------------- Plotting ----------------

def ensure_outdirs(out_root: Path) -> Tuple[Path, Path]:
    figs = out_root / "figs"
    tables = out_root / "tables"
    figs.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    return figs, tables


def _collect_groups_by_condition(df: pd.DataFrame) -> Tuple[List[np.ndarray], List[str]]:
    tmp = df.dropna(subset=["final_metric"]).copy()
    if tmp.empty:
        return [], []
    tmp["condition"] = tmp.get("condition", "base").astype(str)
    existing_conds = list(tmp["condition"].unique())
    order = [c for c in COND_ORDER if c in existing_conds] or sorted(existing_conds)

    groups, labels = [], []
    for c in order:
        arr = pd.to_numeric(tmp.loc[tmp["condition"] == c, "final_metric"], errors="coerce").dropna().to_numpy()
        if arr.size > 0:
            groups.append(arr)
            labels.append(c)
    return groups, labels


def save_violin(figs_dir: Path, df: pd.DataFrame, higher_is_better: bool, dataset: str, arch: str) -> Path:
    metric_name = "Accuracy (%)" if higher_is_better else "Metric (lower is better)"
    groups, labels = _collect_groups_by_condition(df)

    fig, ax = plt.subplots(figsize=(8, 5))
    if not groups:
        ax.text(0.5, 0.5, "No results found.", ha="center", va="center", fontsize=12)
        ax.axis("off")
    else:
        ax.violinplot(groups, showmeans=True, showextrema=True, widths=0.85)
        ax.set_xticks(range(1, len(labels) + 1)); ax.set_xticklabels(labels, rotation=0)
        ax.set_ylabel(metric_name)
        ax.set_title(f"{dataset.upper()} / {arch} - Final performance across seeds (Violin)")
        ax.grid(True, axis='y', alpha=0.3)
    out = figs_dir / f"fig1a_violin_{dataset}_{arch}.png"
    fig.tight_layout(); fig.savefig(out, dpi=200); plt.close(fig)
    return out


def save_boxplot(figs_dir: Path, df: pd.DataFrame, higher_is_better: bool, dataset: str, arch: str) -> Path:
    metric_name = "Accuracy (%)" if higher_is_better else "Metric (lower is better)"
    groups, labels = _collect_groups_by_condition(df)

    fig, ax = plt.subplots(figsize=(8, 5))
    if not groups:
        ax.text(0.5, 0.5, "No results found.", ha="center", va="center", fontsize=12)
        ax.axis("off")
    else:
        ax.boxplot(groups, showmeans=True, whis=1.5, widths=0.65)
        ax.set_xticks(range(1, len(labels) + 1)); ax.set_xticklabels(labels, rotation=0)
        ax.set_ylabel(metric_name)
        ax.set_title(f"{dataset.upper()} / {arch} - Final performance across seeds (Box plot)")
        ax.grid(True, axis='y', alpha=0.3)
    out = figs_dir / f"fig1b_box_{dataset}_{arch}.png"
    fig.tight_layout(); fig.savefig(out, dpi=200); plt.close(fig)
    return out


def save_kde(figs_dir: Path, df: pd.DataFrame, higher_is_better: bool, dataset: str, arch: str) -> Path:
    """
    Overlay KDE curves of per-seed final metrics for each condition.
    y-axis: density, x-axis: metric (accuracy if higher_is_better else generic metric).
    Falls back to simple hist-outline or vertical line if KDE is not available or data is too sparse.
    """
    metric_name = "Accuracy (%)" if higher_is_better else "Metric (lower is better)"
    groups, labels = _collect_groups_by_condition(df)

    fig, ax = plt.subplots(figsize=(8, 5))
    if not groups:
        ax.text(0.5, 0.5, "No results found.", ha="center", va="center", fontsize=12)
        ax.axis("off")
        out = figs_dir / f"fig1c_kde_{dataset}_{arch}.png"
        fig.tight_layout(); fig.savefig(out, dpi=200); plt.close(fig)
        return out

    all_vals = np.concatenate(groups) if groups else np.array([])
    if all_vals.size == 0:
        ax.text(0.5, 0.5, "No numeric metrics found.", ha="center", va="center", fontsize=12)
        ax.axis("off")
        out = figs_dir / f"fig1c_kde_{dataset}_{arch}.png"
        fig.tight_layout(); fig.savefig(out, dpi=200); plt.close(fig)
        return out

    xmin, xmax = np.nanmin(all_vals), np.nanmax(all_vals)
    if not np.isfinite(xmin) or not np.isfinite(xmax):
        xmin, xmax = 0.0, 1.0
    if xmin == xmax:
        xmin -= 1e-6; xmax += 1e-6
    pad = 0.02 * (xmax - xmin) if (xmax - xmin) > 0 else 0.01
    xs = np.linspace(xmin - pad, xmax + pad, 512)

    for arr, lab in zip(groups, labels):
        n = arr.size
        this_label = f"{lab} (n={n})"
        plotted = False
        if sps is not None and hasattr(sps, "gaussian_kde") and n >= 2 and np.std(arr) > 0:
            try:
                kde = sps.gaussian_kde(arr)
                ys = kde(xs)
                ax.plot(xs, ys, label=this_label, linewidth=2)
                plotted = True
            except Exception:
                plotted = False
        if not plotted:
            if n >= 2:
                hist, edges = np.histogram(arr, bins=min(10, n), density=True)
                centers = 0.5 * (edges[:-1] + edges[1:])
                ax.plot(centers, hist, label=this_label, linestyle='--')
            else:
                ax.axvline(arr[0], linestyle=':', label=this_label)

    ax.set_xlabel(metric_name); ax.set_ylabel("Density")
    ax.set_title(f"{dataset.upper()} / {arch} - Final performance across seeds (KDE)")
    ax.grid(True, axis='y', alpha=0.3); ax.legend()
    out = figs_dir / f"fig1c_kde_{dataset}_{arch}.png"
    fig.tight_layout(); fig.savefig(out, dpi=200); plt.close(fig)
    return out


def _stack_and_mean_std(series_list: List[pd.Series]) -> Tuple[np.ndarray, np.ndarray]:
    if not series_list: return np.array([]), np.array([])
    maxlen = max(len(s) for s in series_list)
    arr = np.full((len(series_list), maxlen), np.nan)
    for i, s in enumerate(series_list):
        v = s.to_numpy(dtype=float)
        arr[i, :len(v)] = v
    mean = np.nanmean(arr, axis=0)
    std = np.nanstd(arr, axis=0, ddof=1)
    return mean, std


def save_test_stability(figs_dir: Path,
                        curves_teloss: Dict[str, List[pd.Series]],
                        rolling_window: int, dataset: str, arch: str) -> Tuple[Path, Path]:
    """Panel A: mean+/-sd of TEST loss across seeds.
       Panel B: rolling std (volatility) of TEST loss."""
    def stack_and_mean_std(series_list: List[pd.Series]) -> Tuple[np.ndarray, np.ndarray]:
        if not series_list: return np.array([]), np.array([])
        maxlen = max(len(s) for s in series_list)
        arr = np.full((len(series_list), maxlen), np.nan)
        for i, s in enumerate(series_list):
            v = s.to_numpy(dtype=float)
            arr[i, :len(v)] = v
        mean = np.nanmean(arr, axis=0)
        std  = np.nanstd(arr,  axis=0, ddof=1)
        return mean, std

    # Panel A
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    for cond in COND_ORDER:
        mean, std = stack_and_mean_std(curves_teloss.get(cond, []))
        if mean.size == 0: continue
        x = np.arange(1, len(mean)+1)
        ax1.plot(x, mean, label=cond)
        ax1.fill_between(x, mean-std, mean+std, alpha=0.15)
    ax1.set_title(f"{dataset.upper()} / {arch} - Test loss (mean +/- SD)")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Test loss")
    ax1.grid(True, alpha=0.3); ax1.legend()
    out_a = figs_dir / f"fig2a_test_loss_{dataset}_{arch}.png"
    fig1.tight_layout(); fig1.savefig(out_a, dpi=200); plt.close(fig1)

    # Panel B
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    for cond in COND_ORDER:
        series_list = curves_teloss.get(cond, [])
        if not series_list: continue
        rstd_mat = []
        for s in series_list:
            rstd_mat.append(rolling_std(s.to_numpy(dtype=float), rolling_window))
        if not rstd_mat: continue
        maxlen = max(len(r) for r in rstd_mat)
        arr = np.full((len(rstd_mat), maxlen), np.nan)
        for i, r in enumerate(rstd_mat):
            arr[i, :len(r)] = r
        mean = np.nanmean(arr, axis=0)
        ax2.plot(np.arange(1, len(mean)+1), mean, label=cond)
    ax2.set_title(f"{dataset.upper()} / {arch} - Test-loss volatility (rolling std, W={rolling_window})")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Rolling std of test loss")
    ax2.grid(True, alpha=0.3); ax2.legend()
    out_b = figs_dir / f"fig2b_test_volatility_{dataset}_{arch}.png"
    fig2.tight_layout(); fig2.savefig(out_b, dpi=200); plt.close(fig2)
    return out_a, out_b


def save_pareto(figs_dir: Path, df: pd.DataFrame, higher_is_better: bool, dataset: str, arch: str) -> Path:
    fig, ax = plt.subplots(figsize=(6, 5))
    for cond in COND_ORDER:
        vals = df[df["condition"]==cond]["final_metric"].to_numpy()
        if len(vals) == 0: continue
        sd = np.std(vals, ddof=1) if len(vals) > 1 else 0.0
        mean = np.mean(vals)
        y = mean if higher_is_better else -mean
        ax.scatter(sd, y, label=cond, s=70)
        ax.annotate(cond, (sd, y), xytext=(5,5), textcoords="offset points")
    ax.set_xlabel("Across-seed SD")
    ax.set_ylabel("Mean performance" + (" (higher)" if higher_is_better else " (-MAE; higher is better)"))
    ax.set_title(f"{dataset.upper()} / {arch} - Variability vs performance")
    ax.grid(True, alpha=0.3)
    out = figs_dir / f"fig3_pareto_{dataset}_{arch}.png"
    fig.tight_layout(); fig.savefig(out, dpi=200); plt.close(fig)
    return out


# ---------------- Internals plots (means across seeds) ----------------

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


def _mask_leading_zeros(a: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    a = np.asarray(a, dtype=float).copy()
    k = 0
    for v in a:
        if abs(v) <= tol: k += 1
        else: break
    if k > 0:
        a[:k] = np.nan
    return a


def save_internals_plots(figs_dir: Path, internals_by_cond, dataset: str, arch: str):
    def prep_series(list_of_np):
        lst = [_mask_leading_zeros(v) for v in list_of_np]
        return _stack_mean_std(lst)

    # --- SL ---
    fig1, ax1 = plt.subplots(figsize=(8,5))  # lambda_t
    fig2, ax2 = plt.subplots(figsize=(8,5))  # sigma_ema, sigma_ref, l_ema

    for cond in COND_ORDER:
        lst = internals_by_cond.get(cond, [])
        if not lst: continue

        sl_lam, sl_sig_ema, sl_sig_ref, sl_lbar = [], [], [], []
        for df in lst:
            if "sl_lambda"     in df.columns: sl_lam.append(df["sl_lambda"].astype(float).to_numpy())
            if "sl_sigma_ema"  in df.columns: sl_sig_ema.append(df["sl_sigma_ema"].astype(float).to_numpy())
            if "sl_sigma_ref"  in df.columns: sl_sig_ref.append(df["sl_sigma_ref"].astype(float).to_numpy())
            if "sl_l_ema"      in df.columns: sl_lbar.append(df["sl_l_ema"].astype(float).to_numpy())

        lam_mean, lam_std     = prep_series(sl_lam)     if sl_lam     else (np.array([]), np.array([]))
        sig_mean, sig_std     = prep_series(sl_sig_ema) if sl_sig_ema else (np.array([]), np.array([]))
        sref_mean, sref_std   = prep_series(sl_sig_ref) if sl_sig_ref else (np.array([]), np.array([]))
        lbar_mean, lbar_std   = prep_series(sl_lbar)    if sl_lbar    else (np.array([]), np.array([]))

        if lam_mean.size:
            x = np.arange(1, len(lam_mean)+1)
            ax1.plot(x, lam_mean, label=cond); ax1.fill_between(x, lam_mean-lam_std, lam_mean+lam_std, alpha=0.15)

        def maybe(ax, m, s, lab):
            if m.size:
                x = np.arange(1, len(m)+1)
                ax.plot(x, m, label=lab); ax.fill_between(x, m-s, m+s, alpha=0.12)

        maybe(ax2, sig_mean,  sig_std,  f"{cond} s^_t (EMA)")
        maybe(ax2, sref_mean, sref_std, f"{cond} s_ref")
        maybe(ax2, lbar_mean, lbar_std, f"{cond} L¯_t (EMA)")

    ax1.set_title(f"{dataset.upper()} / {arch} - SL $\\lambda_t$ (mean +/- SD)")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("$\\lambda_t$"); ax1.grid(True, alpha=0.3); ax1.legend()
    ax2.set_title(f"{dataset.upper()} / {arch} - SL $\\hat\\sigma_t$, $\\sigma_\\mathrm{{ref}}$, $\\bar L_t$ (mean +/- SD)")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Value"); ax2.grid(True, alpha=0.3); ax2.legend()

    out1 = figs_dir / f"figA_sl_lambda_{dataset}_{arch}.png"
    out2 = figs_dir / f"figA_sl_sigma_ref_lbar_{dataset}_{arch}.png"
    fig1.tight_layout(); fig1.savefig(out1, dpi=200); plt.close(fig1)
    fig2.tight_layout(); fig2.savefig(out2, dpi=200); plt.close(fig2)

    # --- VPL ---
    fig3, ax3 = plt.subplots(figsize=(8,5))  # lambda variants
    fig4, ax4 = plt.subplots(figsize=(8,5))  # v_ema, v_ref, v_batch
    fig5, ax5 = plt.subplots(figsize=(8,5))  # entropy

    for cond in COND_ORDER:
        lst = internals_by_cond.get(cond, [])
        if not lst: continue
        lam, pre, post, vema, vref, vbatch, ent = [], [], [], [], [], [], []
        for df in lst:
            if "vpl_lambda"            in df.columns: lam.append(df["vpl_lambda"].astype(float).to_numpy())
            if "vpl_lambda_preclip"    in df.columns: pre.append(df["vpl_lambda_preclip"].astype(float).to_numpy())
            if "vpl_lambda_postclip"   in df.columns: post.append(df["vpl_lambda_postclip"].astype(float).to_numpy())
            if "vpl_v_ema"             in df.columns: vema.append(df["vpl_v_ema"].astype(float).to_numpy())
            if "vpl_v_ref"             in df.columns: vref.append(df["vpl_v_ref"].astype(float).to_numpy())
            if "vpl_v_batch_mean"      in df.columns: vbatch.append(df["vpl_v_batch_mean"].astype(float).to_numpy())
            if "vpl_entropy_mean"      in df.columns: ent.append(df["vpl_entropy_mean"].astype(float).to_numpy())

        def ms(x): return prep_series(x) if x else (np.array([]), np.array([]))
        lam_m, lam_s     = ms(lam)
        pre_m, pre_s     = ms(pre)
        post_m, post_s   = ms(post)
        vema_m, vema_s   = ms(vema)
        vref_m, vref_s   = ms(vref)
        vbatch_m, vbatch_s = ms(vbatch)
        ent_m, ent_s     = ms(ent)

        def plot_band(ax, m, s, lab):
            if m.size:
                x = np.arange(1, len(m)+1)
                ax.plot(x, m, label=lab); ax.fill_between(x, m-s, m+s, alpha=0.12)

        plot_band(ax3, lam_m,   lam_s,   f"{cond} $\\lambda_t$")
        plot_band(ax3, pre_m,   pre_s,   f"{cond} $\\lambda_\\mathrm{{pre}}$")
        plot_band(ax3, post_m,  post_s,  f"{cond} $\\lambda_\\mathrm{{post}}$")
        plot_band(ax4, vema_m,  vema_s,  f"{cond} v_ema")
        plot_band(ax4, vref_m,  vref_s,  f"{cond} v_ref")
        plot_band(ax4, vbatch_m,vbatch_s,f"{cond} v_batch")
        plot_band(ax5, ent_m,   ent_s,   f"{cond} entropy")

    ax3.set_title(f"{dataset.upper()} / {arch} - VPL $\\lambda$ variants (mean +/- SD)")
    ax3.set_xlabel("Epoch"); ax3.set_ylabel("$\\lambda$"); ax3.grid(True, alpha=0.3); ax3.legend()
    ax4.set_title(f"{dataset.upper()} / {arch} - VPL $v_\\mathrm{{ema}}$, $v_\\mathrm{{ref}}$, $v_\\mathrm{{batch}}$ (mean +/- SD)")
    ax4.set_xlabel("Epoch"); ax4.set_ylabel("v"); ax4.grid(True, alpha=0.3); ax4.legend()
    ax5.set_title(f"{dataset.upper()} / {arch} - VPL entropy (mean +/- SD)")
    ax5.set_xlabel("Epoch"); ax5.set_ylabel("entropy"); ax5.grid(True, alpha=0.3); ax5.legend()

    out3 = figs_dir / f"figA_vpl_lambda_variants_{dataset}_{arch}.png"
    out4 = figs_dir / f"figA_vpl_v_stats_{dataset}_{arch}.png"
    out5 = figs_dir / f"figA_vpl_entropy_{dataset}_{arch}.png"
    for fig, out in [(fig3, out3), (fig4, out4), (fig5, out5)]:
        fig.tight_layout(); fig.savefig(out, dpi=200); plt.close(fig)

    return [out1, out2, out3, out4, out5]



# ---------------- Reactivity overlays (internals vs loss) ----------------

def _mean_across_seeds(df_list: List[pd.DataFrame], col: str) -> np.ndarray:
    """Return per-epoch mean across runs for a given column, masking leading zeros."""
    arrs = []
    for df in df_list:
        if col in df.columns:
            v = df[col].astype(float).to_numpy()
            arrs.append(_leading_zeros_to_nan(v))
    if not arrs:
        return np.array([])
    m, _ = _stack_mean_std(arrs)
    return m

def _mean_test_loss(curves_teloss: Dict[str, List[pd.Series]], cond: str) -> np.ndarray:
    mean, _ = _stack_and_mean_std(curves_teloss.get(cond, []))
    return mean

def save_reactivity_overlays(
    figs_dir: Path,
    internals_by_cond,
    curves_teloss: Dict[str, List[pd.Series]],
    dataset: str, arch: str
) -> Tuple[List[Path], pd.DataFrame]:
    """
    For each condition create:
      - SL_vs_Loss: overlays SL internals (?_t, s^_t, s_ref, L¯_t) on left axis
                    with Test loss AND Train total loss (epoch mean) on right axis.
      - VPL_vs_Loss: overlays VPL internals (?_t, pre/post, v_ema, v_ref, v_batch, entropy)
                     with Test loss AND Train total loss (epoch mean) on right axis.
    Also compute lag-correlation table (still vs TEST loss).
    """
    outs: List[Path] = []
    table_rows = []

    for cond in COND_ORDER:
        df_list = internals_by_cond.get(cond, [])
        if not df_list:
            continue

        # Means across seeds (leading zeros masked)
        sl_lambda    = _mean_across_seeds(df_list, "sl_lambda")
        sl_sigma     = _mean_across_seeds(df_list, "sl_sigma_ema")
        sl_sigma_ref = _mean_across_seeds(df_list, "sl_sigma_ref")
        sl_l_ema     = _mean_across_seeds(df_list, "sl_l_ema")

        vpl_lambda   = _mean_across_seeds(df_list, "vpl_lambda")
        vpl_pre      = _mean_across_seeds(df_list, "vpl_lambda_preclip")
        vpl_post     = _mean_across_seeds(df_list, "vpl_lambda_postclip")
        vpl_vema     = _mean_across_seeds(df_list, "vpl_v_ema")
        vpl_vref     = _mean_across_seeds(df_list, "vpl_v_ref")
        vpl_vbatch   = _mean_across_seeds(df_list, "vpl_v_batch_mean")
        vpl_entropy  = _mean_across_seeds(df_list, "vpl_entropy_mean")

        # Loss series
        loss_test_mean  = _mean_test_loss(curves_teloss, cond)               # from log.txt
        loss_train_mean = _mean_across_seeds(df_list, "tr_loss_epoch")       # from internals_seed_*.csv

        # X for test loss (ok if others differ in length)
        x_test = np.arange(1, len(loss_test_mean) + 1) if loss_test_mean.size else None

        # ------------ SL overlay ------------
        if (sl_lambda.size or sl_sigma.size or sl_sigma_ref.size or sl_l_ema.size) and (loss_test_mean.size or loss_train_mean.size):
            fig, ax1 = plt.subplots(figsize=(9, 5))

            def _plot_left(arr: np.ndarray, lab: str, style: Optional[str] = None, lw: float = 2.0):
                if arr.size:
                    x = np.arange(1, len(arr) + 1)
                    ax1.plot(x, arr, label=lab, linestyle=style if style else "-", linewidth=lw)

            _plot_left(sl_lambda,    r"SL $\lambda_t$")
            _plot_left(sl_sigma,     r"SL $\hat{\sigma}_t$ (EMA)", style="--", lw=1.8)
            _plot_left(sl_sigma_ref, r"SL $\sigma_\mathrm{ref}$",  style=":",  lw=1.8)
            _plot_left(sl_l_ema,     r"SL $\bar L_t$ (EMA)",       style="-.", lw=1.8)

            ax1.set_xlabel("Epoch")
            ax1.set_ylabel("SL internals")
            ax1.grid(True, alpha=0.3)

            ax2 = ax1.twinx()
            lines_r, labels_r = [], []

            if loss_test_mean.size:
                l = ax2.plot(x_test, loss_test_mean, label="Test loss", alpha=0.9, linewidth=2.2, color="tab:red")[0]
                lines_r.append(l); labels_r.append("Test loss")
            if loss_train_mean.size:
                x_tr = np.arange(1, len(loss_train_mean) + 1)
                l = ax2.plot(x_tr, loss_train_mean, label="Train total loss (epoch mean)", alpha=0.9, linestyle="--")[0]
                lines_r.append(l); labels_r.append("Train total loss (epoch mean)")

            ax2.set_ylabel("Loss")

            # unified legend (left + right)
            lines_l, labels_l = ax1.get_legend_handles_labels()
            ax1.legend(lines_l + lines_r, labels_l + labels_r, loc="upper left", ncol=2)

            plt.title(f"{dataset.upper()} / {arch} - {cond} : SL vs Loss")
            out = figs_dir / f"fig4_sl_vs_loss_{cond}_{dataset}_{arch}.png"
            fig.tight_layout(); fig.savefig(out, dpi=200); plt.close(fig)
            outs.append(out)

        # ------------ VPL overlay ------------
        if any(s.size for s in [vpl_lambda, vpl_pre, vpl_post, vpl_vema, vpl_vref, vpl_vbatch, vpl_entropy]) and (loss_test_mean.size or loss_train_mean.size):
            fig, ax1 = plt.subplots(figsize=(9, 5))

            def _plot_left(arr: np.ndarray, lab: str, style: Optional[str] = None, lw: float = 2.0):
                if arr.size:
                    x = np.arange(1, len(arr) + 1)
                    ax1.plot(x, arr, label=lab, linestyle=style if style else "-", linewidth=lw)

            _plot_left(vpl_lambda,         r"VPL $\lambda_t$")
            _plot_left(vpl_pre,            r"$\lambda_\mathrm{pre}$",  style="--", lw=1.8)
            _plot_left(vpl_post,           r"$\lambda_\mathrm{post}$", style=":",  lw=1.8)
            _plot_left(vpl_vema,           "v_ema",   style="-.", lw=1.6)
            _plot_left(vpl_vref,           "v_ref",   style="-.", lw=1.6)
            _plot_left(vpl_vbatch,         "v_batch", style="-.", lw=1.6)
            _plot_left(vpl_entropy,        "entropy", style=(0, (3, 1, 1, 1)), lw=1.6)

            ax1.set_xlabel("Epoch")
            ax1.set_ylabel("VPL internals / entropy")
            ax1.grid(True, alpha=0.3)

            ax2 = ax1.twinx()
            lines_r, labels_r = [], []

            if loss_test_mean.size:
                l = ax2.plot(x_test, loss_test_mean, label="Test loss", alpha=0.9, linewidth=2.2, color="tab:red")[0]
                lines_r.append(l); labels_r.append("Test loss")
            if loss_train_mean.size:
                x_tr = np.arange(1, len(loss_train_mean) + 1)
                l = ax2.plot(x_tr, loss_train_mean, label="Train total loss (epoch mean)", alpha=0.9, linestyle="--")[0]
                lines_r.append(l); labels_r.append("Train total loss (epoch mean)")

            ax2.set_ylabel("Loss")

            # unified legend
            lines_l, labels_l = ax1.get_legend_handles_labels()
            ax1.legend(lines_l + lines_r, labels_l + labels_r, loc="upper left", ncol=2)

            plt.title(f"{dataset.upper()} / {arch} - {cond} : VPL vs Loss")
            out = figs_dir / f"fig5_vpl_vs_loss_{cond}_{dataset}_{arch}.png"
            fig.tight_layout(); fig.savefig(out, dpi=200); plt.close(fig)
            outs.append(out)

        # --- Reactivity metrics (still vs TEST loss) ---
        def add_metric(name: str, series: np.ndarray, loss_series: np.ndarray):
            if not (series.size and loss_series.size): 
                return
            L = min(len(series), len(loss_series))
            if L < 3:
                return
            s = series[:L]; l = loss_series[:L]
            dz = np.diff(s); dl = np.diff(l)
            mask = np.isfinite(dz) & np.isfinite(dl)
            if mask.sum() >= 3:
                if sps is not None:
                    p0 = float(sps.pearsonr(dz[mask], dl[mask])[0]) if np.std(dz[mask])>0 and np.std(dl[mask])>0 else np.nan
                    s0 = float(sps.spearmanr(dz[mask], dl[mask])[0]) if (len(np.unique(dz[mask]))>2 and len(np.unique(dl[mask]))>2) else np.nan
                else:
                    p0 = float(np.corrcoef(dz[mask], dl[mask])[0,1]); s0 = np.nan
            else:
                p0, s0 = np.nan, np.nan
            lag, pbest, sbest = best_lag_corr(s, l, max_lag=10)
            dirn = "leads" if lag > 0 else ("lags" if lag < 0 else "simultaneous")
            table_rows.append({
                "condition": cond, "signal": name,
                "pearson_zero_lag": p0, "spearman_zero_lag": s0,
                "best_lag_epochs": int(lag), "pearson_best": pbest, "spearman_best": sbest,
                "direction": dirn
            })

        # Keep correlation vs TEST loss (as before)
        add_metric("SL_lambda", sl_lambda, loss_test_mean)
        add_metric("SL_sigma_ema", sl_sigma, loss_test_mean)
        add_metric("SL_sigma_ref", sl_sigma_ref, loss_test_mean)
        add_metric("SL_l_ema", sl_l_ema, loss_test_mean)
        add_metric("VPL_lambda", vpl_lambda, loss_test_mean)
        add_metric("VPL_lambda_preclip", vpl_pre, loss_test_mean)
        add_metric("VPL_lambda_postclip", vpl_post, loss_test_mean)
        add_metric("VPL_v_ema", vpl_vema, loss_test_mean)
        add_metric("VPL_v_ref", vpl_vref, loss_test_mean)
        add_metric("VPL_v_batch_mean", vpl_vbatch, loss_test_mean)
        add_metric("VPL_entropy_mean", vpl_entropy, loss_test_mean)

    reactivity_df = pd.DataFrame(table_rows)
    return outs, reactivity_df




# ---------------- Step preview (all conditions) ----------------

def save_batch_preview(figs_dir: Path, batches_by_cond, dataset: str, arch: str):
    fig, ax = plt.subplots(figsize=(9,5))
    any_plotted = False
    missing = []
    for cond in COND_ORDER:
        runs = batches_by_cond.get(cond, [])
        if not runs:
            missing.append(cond); continue
        df = runs[0].sort_values(["epoch","batch_idx"])
        if "total_loss" not in df.columns:
            missing.append(cond); continue
        y = df["total_loss"].astype(float).to_numpy()
        ax.plot(np.arange(len(y)), y, label=cond, alpha=0.9)
        any_plotted = True
    if not any_plotted:
        plt.close(fig)
        return None
    ax.set_title(f"{dataset.upper()} / {arch} - Step-level loss preview (all conditions)")
    ax.set_xlabel("Batch step"); ax.set_ylabel("loss"); ax.grid(True, alpha=0.3); ax.legend(ncol=2)
    out = figs_dir / f"figA_step_preview_{dataset}_{arch}.png"
    fig.tight_layout(); fig.savefig(out, dpi=200); plt.close(fig)
    if missing:
        print("[step-preview] no per-batch CSV for:", ", ".join(m for m in missing))
    return out


# ---------------- Zoomed previews (early/middle/late) ----------------

def _pick_one_run_per_condition(df_runs: pd.DataFrame, batches_by_cond: Dict[str, List[pd.DataFrame]]) -> Dict[str, pd.DataFrame]:
    picks = {}
    for cond in COND_ORDER:
        lst = batches_by_cond.get(cond, [])
        if lst:
            picks[cond] = lst[0]
    return picks


def _nearest_epoch_available(df: pd.DataFrame, target_epoch_0based: int) -> Optional[int]:
    if "epoch" not in df.columns: return None
    ep_vals = sorted(df["epoch"].dropna().unique().astype(int).tolist())
    if not ep_vals: return None
    return min(ep_vals, key=lambda e: abs(e - target_epoch_0based))


def save_zoom_step_previews_allconds(figs_dir: Path, df_runs: pd.DataFrame,
                                     dataset: str, arch: str,
                                     total_epochs: int, zoom_epochs_arg: str,
                                     zoom_batches: int,
                                     batches_by_cond: Dict[str, List[pd.DataFrame]]) -> List[Path]:
    E0, M0, L0 = pick_zoom_centers(total_epochs, zoom_epochs_arg)
    centers = {"early": E0, "middle": M0, "late": L0}
    outs: List[Path] = []

    picks = _pick_one_run_per_condition(df_runs, batches_by_cond)

    for tag, cen in centers.items():
        fig, ax = plt.subplots(figsize=(10,5))
        plotted = False
        for cond in COND_ORDER:
            df = picks.get(cond, None)
            if df is None:
                continue
            ep = _nearest_epoch_available(df, cen)
            if ep is None:
                continue
            use = df[df["epoch"].astype(int) == int(ep)].sort_values("batch_idx").copy()
            if use.empty or "total_loss" not in use.columns:
                continue
            x = use["batch_idx"].to_numpy()
            y = use["total_loss"].astype(float).to_numpy()
            if zoom_batches and zoom_batches > 0:
                x = x[:zoom_batches]; y = y[:zoom_batches]
            ax.plot(x, y, label=cond)
            plotted = True
        if not plotted:
            plt.close(fig)
            ph = figs_dir / f"figA_zoom_{tag}_{dataset}_{arch}_placeholder.png"
            fig2, ax2 = plt.subplots(figsize=(6,3))
            ax2.text(0.5, 0.5, f"No per-batch data for {tag} phase.", ha="center", va="center")
            ax2.axis("off")
            fig2.tight_layout(); fig2.savefig(ph, dpi=200); plt.close(fig2)
            outs.append(ph)
            continue
        ax.set_title(f"{dataset.upper()} / {arch} - {tag} zoom (loss vs steps, all conditions)")
        ax.set_xlabel("batch index"); ax.set_ylabel("loss"); ax.grid(True, alpha=0.3); ax.legend()
        out = figs_dir / f"figA_zoom_{tag}_{dataset}_{arch}.png"
        fig.tight_layout(); fig.savefig(out, dpi=200); plt.close(fig)
        outs.append(out)
    return outs


# ---------------- Per-epoch mean exports ----------------

def _stack(series_list: List[pd.Series]) -> np.ndarray:
    if not series_list: return np.empty((0,0))
    maxlen = max(len(s) for s in series_list)
    arr = np.full((len(series_list), maxlen), np.nan)
    for i, s in enumerate(series_list):
        v = s.to_numpy(dtype=float)
        arr[i, :len(v)] = v
    return arr


def export_per_epoch_means(out_root: Path,
                           curves_tr: Dict[str, List[pd.Series]],
                           curves_acc: Dict[str, List[pd.Series]],
                           curves_teloss: Dict[str, List[pd.Series]]) -> Path:
    curves_dir = out_root / "curves"
    curves_dir.mkdir(parents=True, exist_ok=True)

    for cond in COND_ORDER:
        arr_tr = _stack(curves_tr.get(cond, []))
        arr_acc = _stack(curves_acc.get(cond, []))
        arr_tl = _stack(curves_teloss.get(cond, []))

        maxlen = 0
        for a in (arr_tr, arr_acc, arr_tl):
            maxlen = max(maxlen, a.shape[1] if a.size else 0)
        if maxlen == 0:
            continue

        def mean_std(a: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            if a.size == 0:
                return np.full(maxlen, np.nan), np.full(maxlen, np.nan)
            m = np.nanmean(a, axis=0)
            s = np.nanstd(a, axis=0, ddof=1)
            if a.shape[1] < maxlen:
                pad_m = np.full(maxlen, np.nan); pad_m[:len(m)] = m
                pad_s = np.full(maxlen, np.nan); pad_s[:len(s)] = s
                return pad_m, pad_s
            return m, s

        m_tr, s_tr = mean_std(arr_tr)
        m_acc, s_acc = mean_std(arr_acc)
        m_tl, s_tl = mean_std(arr_tl)

        rows = []
        for ep in range(1, maxlen + 1):
            i = ep - 1
            rows.append({
                "epoch": ep,
                "mean_tr_loss": float(m_tr[i]) if not np.isnan(m_tr[i]) else np.nan,
                "std_tr_loss":  float(s_tr[i]) if not np.isnan(s_tr[i]) else np.nan,
                "mean_te_acc":  float(m_acc[i]) if not np.isnan(m_acc[i]) else np.nan,
                "std_te_acc":   float(s_acc[i]) if not np.isnan(s_acc[i]) else np.nan,
                "mean_te_loss": float(m_tl[i]) if not np.isnan(m_tl[i]) else np.nan,
                "std_te_loss":  float(s_tl[i]) if not np.isnan(s_tl[i]) else np.nan,
            })
        out_csv = curves_dir / f"{cond}.csv"
        pd.DataFrame(rows).to_csv(out_csv, index=False)

    return curves_dir


def _crop_window(y: np.ndarray, center_ep0: int, halfw: int):
    """Return (x_epoch_numbers, y_window) cropped around 0-based center with +/-halfw (inclusive)."""
    if y.size == 0: return np.array([]), np.array([])
    lo = max(0, center_ep0 - halfw)
    hi = min(len(y)-1, center_ep0 + halfw)
    idx = np.arange(lo, hi+1)
    # 1-based epoch numbers for x-axis
    x = idx + 1
    return x, y[lo:hi+1]

def save_reactivity_overlays_zoom(figs_dir: Path,
                                  internals_by_cond,
                                  curves_teloss: Dict[str, List[pd.Series]],
                                  dataset: str, arch: str,
                                  total_epochs: int,
                                  zoom_epochs_arg: str,
                                  zoom_ep_per_phase: int) -> List[Path]:
    outs = []
    E0, M0, L0 = pick_zoom_centers(total_epochs, zoom_epochs_arg)
    centers = {"early": E0, "middle": M0, "late": L0}
    halfw = max(1, int(zoom_ep_per_phase))

    for cond in COND_ORDER:
        df_list = internals_by_cond.get(cond, [])
        if not df_list: continue

        # Assemble series means like in the non-zoom path
        sl_lambda   = _mean_across_seeds(df_list, "sl_lambda")
        sl_sigma    = _mean_across_seeds(df_list, "sl_sigma_ema")
        sl_lbar     = _mean_across_seeds(df_list, "sl_l_ema")
        sl_sref     = _mean_across_seeds(df_list, "sl_sigma_ref")

        vpl_lambda  = _mean_across_seeds(df_list, "vpl_lambda")
        vpl_pre     = _mean_across_seeds(df_list, "vpl_lambda_preclip")
        vpl_post    = _mean_across_seeds(df_list, "vpl_lambda_postclip")
        vpl_entropy = _mean_across_seeds(df_list, "vpl_entropy_mean")
        vpl_vema    = _mean_across_seeds(df_list, "vpl_v_ema")
        vpl_vref    = _mean_across_seeds(df_list, "vpl_v_ref")
        vpl_vbatch  = _mean_across_seeds(df_list, "vpl_v_batch_mean")

        loss_mean, _ = _stack_and_mean_std(curves_teloss.get(cond, []))

        for tag, cen in centers.items():
            if loss_mean.size == 0: continue
            # ---- SL zoom
            if sl_lambda.size or sl_sigma.size or sl_lbar.size or sl_sref.size:
                fig, ax1 = plt.subplots(figsize=(8.5,4.8))
                def plot_zoom(arr, lab, style=None, lw=2):
                    if arr.size:
                        x, y = _crop_window(arr, cen, halfw)
                        if y.size: ax1.plot(x, y, label=lab, linestyle=(style or '-'), linewidth=lw)
                plot_zoom(sl_lambda, r"SL $\lambda_t$")
                plot_zoom(sl_sigma,  r"SL $\hat{\sigma}_t$ (EMA)", style="--", lw=1.6)
                plot_zoom(sl_sref,   r"SL $\sigma_\mathrm{ref}$", style=":",  lw=1.6)
                plot_zoom(sl_lbar,   r"SL $\bar L_t$ (EMA)",      style="-.", lw=1.6)

                ax2 = ax1.twinx()
                x_loss, y_loss = _crop_window(loss_mean, cen, halfw)
                if y_loss.size: ax2.plot(x_loss, y_loss, alpha=0.85)

                ax1.set_title(f"{dataset.upper()} / {arch} - {cond} : SL vs loss (zoom {tag})")
                ax1.set_xlabel("Epoch"); ax1.set_ylabel("SL internals")
                ax2.set_ylabel("Test loss"); ax1.grid(True, alpha=0.3); ax1.legend(loc="upper left")
                out = figs_dir / f"fig4z_sl_vs_loss_{cond}_{tag}_{dataset}_{arch}.png"
                fig.tight_layout(); fig.savefig(out, dpi=200); plt.close(fig)
                outs.append(out)

            # ---- VPL zoom
            if any(s.size for s in [vpl_lambda, vpl_pre, vpl_post, vpl_entropy, vpl_vema, vpl_vref, vpl_vbatch]):
                fig, ax1 = plt.subplots(figsize=(8.5,4.8))
                def plot_zoom(arr, lab, style=None, lw=2):
                    if arr.size:
                        x, y = _crop_window(arr, cen, halfw)
                        if y.size: ax1.plot(x, y, label=lab, linestyle=(style or '-'), linewidth=lw)
                plot_zoom(vpl_lambda, r"VPL $\lambda_t$")
                plot_zoom(vpl_pre,    r"$\lambda_\mathrm{pre}$",  style="--", lw=1.6)
                plot_zoom(vpl_post,   r"$\lambda_\mathrm{post}$", style=":",  lw=1.6)
                plot_zoom(vpl_vema,   "v_ema",   style="-.", lw=1.4)
                plot_zoom(vpl_vref,   "v_ref",   style="-.", lw=1.4)
                plot_zoom(vpl_vbatch, "v_batch", style="-.", lw=1.4)
                plot_zoom(vpl_entropy,"entropy", style=(0,(3,1,1,1)), lw=1.4)

                ax2 = ax1.twinx()
                x_loss, y_loss = _crop_window(loss_mean, cen, halfw)
                if y_loss.size: ax2.plot(x_loss, y_loss, alpha=0.85)

                ax1.set_title(f"{dataset.upper()} / {arch} - {cond} : VPL vs loss (zoom {tag})")
                ax1.set_xlabel("Epoch"); ax1.set_ylabel("VPL internals / entropy")
                ax2.set_ylabel("Test loss"); ax1.grid(True, alpha=0.3); ax1.legend(loc="upper left")
                out = figs_dir / f"fig5z_vpl_vs_loss_{cond}_{tag}_{dataset}_{arch}.png"
                fig.tight_layout(); fig.savefig(out, dpi=200); plt.close(fig)
                outs.append(out)

    return outs





# ---------------- Tables ----------------

def make_summary_tables(tables_dir: Path, df: pd.DataFrame, higher_is_better: bool) -> Tuple[Path, Path]:
    conds_present = [c for c in COND_ORDER if c in df["condition"].unique()]
    if not conds_present:
        raise SystemExit("No conditions present to summarize.")

    # Pick reference: prefer 'base' if present, else the first condition
    ref = "base" if "base" in conds_present else conds_present[0]

    pivot = df.pivot_table(index="seed", columns="condition", values="final_metric", aggfunc="first")
    pivot = pivot[[c for c in conds_present if c in pivot.columns]].sort_index()
    wide_csv = tables_dir / "per_seed_metrics.csv"
    pivot.to_csv(wide_csv)

    rows = []
    ref_vals = df[df["condition"]==ref]["final_metric"].to_numpy()
    for cond in conds_present:
        vals = df[df["condition"]==cond]["final_metric"].to_numpy()
        n = len(vals)
        mean = float(np.mean(vals)) if n else np.nan
        sd = float(np.std(vals, ddof=1)) if n > 1 else np.nan
        cv = float(sd / mean) if n > 1 and mean != 0 else np.nan
        lo, hi = ci_mean(vals) if n else (np.nan, np.nan)
        rows.append({"condition": cond, "n": n, "mean": mean, "sd": sd, "cv": cv,
                     "ci95_low": lo, "ci95_high": hi})
    summary = pd.DataFrame(rows)

    if len(ref_vals) >= 2 and np.std(ref_vals, ddof=1) > 0:
        ref_sd = float(np.std(ref_vals, ddof=1))
        sd_red, sd_ratio_ci = [], []
        for cond in conds_present:
            vals = df[df["condition"]==cond]["final_metric"].to_numpy()
            if len(vals) >= 2:
                sd = float(np.std(vals, ddof=1))
                pct = 100.0 * (1.0 - sd/ref_sd)
                ratio, lo, hi = bootstrap_sd_ratio_ci(ref_vals, vals)
            else:
                pct, ratio, lo, hi = (np.nan, np.nan, np.nan, np.nan)
            sd_red.append(pct); sd_ratio_ci.append((ratio, lo, hi))
        summary["sd_reduction_vs_ref_%"] = sd_red
        summary["sd_ratio_vs_ref"] = [r[0] for r in sd_ratio_ci]
        summary["sd_ratio_ci95_low"] = [r[1] for r in sd_ratio_ci]
        summary["sd_ratio_ci95_high"] = [r[2] for r in sd_ratio_ci]

    def paired_diff(cond):
        if ref not in pivot.columns or cond not in pivot.columns:
            return None
        joined = pd.concat([pivot[ref], pivot[cond]], axis=1, join="inner").dropna()
        if joined.shape[0] < 2: return None
        diff = joined[cond] - joined[ref]
        if not higher_is_better:
            diff = -diff
        return diff.to_numpy()

    paired_rows = []
    for cond in conds_present:
        if cond == ref: continue
        d = paired_diff(cond)
        if d is None:
            paired_rows.append({"vs": f"{cond} vs {ref}", "paired_n": 0,
                                "t_p": np.nan, "wilcoxon_p": np.nan, "cohens_d_paired": np.nan})
            continue
        stats = paired_t_wilcoxon_and_d(d)
        paired_rows.append({"vs": f"{cond} vs {ref}", "paired_n": len(d), **stats})
    tests = pd.DataFrame(paired_rows)

    summary_csv = tables_dir / "summary_by_condition.csv"
    tests_csv = tables_dir / "paired_tests_vs_ref.csv"
    summary.to_csv(summary_csv, index=False)
    tests.to_csv(tests_csv, index=False)
    return summary_csv, tests_csv


# ---------------- One analysis run (reusable) ----------------

def run_one_analysis(
    out_root: Path,
    df: pd.DataFrame,
    curves_tr: Dict[str, List[pd.Series]],
    curves_acc: Dict[str, List[pd.Series]],
    curves_teloss: Dict[str, List[pd.Series]],
    dataset: str, arch: str,
    higher_is_better: bool,
    root_for_optionals: Path,
    rolling_window: int,
    zoom_batches: int, zoom_epochs: str, zoom_ep_per_phase: int,
    auto_push_wandb: int, wandb_project: str, wandb_entity: Optional[str]
):
    """
    One full analysis pass (overall + optional per-seed).
    Includes zoomed SL/VPL vs loss overlays (early/middle/late) that show
    Test loss and Train total loss (epoch mean) together.
    """
    figs_dir, tables_dir = ensure_outdirs(out_root)

    if df.empty:
        raise SystemExit(f"No runs found for analysis at {out_root}")

    # Save raw run table
    df_out = tables_dir / "runs_raw.csv"
    df.to_csv(df_out, index=False)

    # Infer total epochs from runs (fallback 200)
    total_epochs = 200
    if "epochs" in df.columns and not df["epochs"].isna().all():
        try:
            total_epochs = int(df["epochs"].max())
        except Exception:
            pass

    # ---------- Figures: performance + stability ----------
    print(f"[plot] fig1 violin/box/kde -> {out_root}")
    f1a = save_violin(figs_dir, df, higher_is_better=higher_is_better, dataset=dataset, arch=arch)
    f1b = save_boxplot(figs_dir, df, higher_is_better=higher_is_better, dataset=dataset, arch=arch)
    f1c = save_kde(figs_dir, df, higher_is_better=higher_is_better, dataset=dataset, arch=arch)

    print(f"[plot] fig2 stability (TEST loss) -> {out_root}")
    f2a, f2b = save_test_stability(figs_dir, curves_teloss, rolling_window=rolling_window,
                                   dataset=dataset, arch=arch)

    print(f"[plot] fig3 pareto -> {out_root}")
    f3 = save_pareto(figs_dir, df, higher_is_better=higher_is_better, dataset=dataset, arch=arch)

    # ---------- Internals + Reactivity ----------
    print(f"[plot] internals and step previews -> {out_root}")
    internals_by_cond = load_internals_csvs(root_for_optionals)
    internals_figs = save_internals_plots(figs_dir, internals_by_cond, dataset, arch)

    print(f"[plot] reactivity overlays + table -> {out_root}")
    overlay_figs, reactivity_df = save_reactivity_overlays(
        figs_dir, internals_by_cond, curves_teloss, dataset, arch
    )
    react_csv = tables_dir / "reactivity_by_condition.csv"
    if not reactivity_df.empty:
        reactivity_df.to_csv(react_csv, index=False)

    # ---------- Reactivity ZOOM overlays (early/middle/late) ----------
    # Helper (updated): overlays Train total loss (epoch mean) alongside Test loss.
    def _save_reactivity_zoom_overlays(
        figs_dir: Path,
        internals_by_cond,
        curves_teloss: Dict[str, List[pd.Series]],
        dataset: str, arch: str,
        total_epochs: int,
        zoom_epochs_arg: str,
        half_window: int
    ) -> List[Path]:
        outs: List[Path] = []

        # determine zoom centers (0-based)
        E0, M0, L0 = pick_zoom_centers(total_epochs, zoom_epochs_arg)
        centers = [("early", E0), ("middle", M0), ("late", L0)]

        def _slice_window(arr: np.ndarray, center: int, halfw: int) -> Tuple[np.ndarray, np.ndarray]:
            if arr.size == 0:
                return np.array([]), np.array([])
            start = max(0, center - halfw)
            end = min(len(arr), center + halfw + 1)
            x = np.arange(start + 1, end + 1)  # 1-based epoch labels
            return arr[start:end], x

        for cond in COND_ORDER:
            df_list = internals_by_cond.get(cond, [])
            if not df_list:
                continue

            # Internals (means across seeds; leading zeros masked by _mean_across_seeds)
            sl_lambda    = _mean_across_seeds(df_list, "sl_lambda")
            sl_sigma_ema = _mean_across_seeds(df_list, "sl_sigma_ema")
            sl_sigma_ref = _mean_across_seeds(df_list, "sl_sigma_ref")
            sl_l_ema     = _mean_across_seeds(df_list, "sl_l_ema")

            vpl_lambda          = _mean_across_seeds(df_list, "vpl_lambda")
            vpl_lambda_preclip  = _mean_across_seeds(df_list, "vpl_lambda_preclip")
            vpl_lambda_postclip = _mean_across_seeds(df_list, "vpl_lambda_postclip")
            vpl_v_ema           = _mean_across_seeds(df_list, "vpl_v_ema")
            vpl_v_ref           = _mean_across_seeds(df_list, "vpl_v_ref")
            vpl_v_batch_mean    = _mean_across_seeds(df_list, "vpl_v_batch_mean")
            vpl_entropy_mean    = _mean_across_seeds(df_list, "vpl_entropy_mean")

            # Loss series
            loss_test_mean  = _mean_test_loss(curves_teloss, cond)
            loss_train_mean = _mean_across_seeds(df_list, "tr_loss_epoch")  # epoch-average total loss from training script

            for tag, cen in centers:
                # ---- SL zoom ----
                if sl_lambda.size or sl_sigma_ema.size or sl_sigma_ref.size or sl_l_ema.size:
                    fig, ax1 = plt.subplots(figsize=(9, 5))

                    def _plot(arr: np.ndarray, lab: str, style: Optional[str] = None, lw: float = 2.0):
                        a, x = _slice_window(arr, cen, half_window)
                        if a.size:
                            ax1.plot(x, a, label=lab, linestyle=style if style else "-", linewidth=lw)

                    _plot(sl_lambda,    r"SL $\lambda_t$")
                    _plot(sl_sigma_ema, r"SL $\hat{\sigma}_t$ (EMA)", style="--", lw=1.8)
                    _plot(sl_sigma_ref, r"SL $\sigma_\mathrm{ref}$",  style=":",  lw=1.8)
                    _plot(sl_l_ema,     r"SL $\bar L_t$ (EMA)",       style="-.", lw=1.8)

                    ax1.set_xlabel("Epoch")
                    ax1.set_ylabel("SL internals")
                    ax1.grid(True, alpha=0.3)

                    ax2 = ax1.twinx()
                    lines_r, labels_r = [], []
                    lt, xt = _slice_window(loss_test_mean, cen, half_window)
                    if lt.size:
                        l = ax2.plot(xt, lt, label="Test loss", alpha=0.9, linewidth=2.2, color="tab:red")[0]
                        lines_r.append(l); labels_r.append("Test loss")
                    ltr, xtr = _slice_window(loss_train_mean, cen, half_window)
                    if ltr.size:
                        l = ax2.plot(xtr, ltr, label="Train total loss (epoch mean)", alpha=0.9, linestyle="--", color="tab:purple")[0]
                        lines_r.append(l); labels_r.append("Train total loss (epoch mean)")
                    ax2.set_ylabel("Loss")

                    # unified legend
                    lines_l, labels_l = ax1.get_legend_handles_labels()
                    ax1.legend(lines_l + lines_r, labels_l + labels_r, loc="upper left", ncol=2)

                    plt.title(f"{dataset.upper()} / {arch} - {cond} : SL vs Loss ({tag} zoom)")
                    out = figs_dir / f"fig4z_sl_vs_loss_{cond}_{tag}_{dataset}_{arch}.png"
                    fig.tight_layout(); fig.savefig(out, dpi=200); plt.close(fig)
                    outs.append(out)

                # ---- VPL zoom ----
                if any(s.size for s in [
                    vpl_lambda, vpl_lambda_preclip, vpl_lambda_postclip,
                    vpl_v_ema, vpl_v_ref, vpl_v_batch_mean, vpl_entropy_mean
                ]):
                    fig, ax1 = plt.subplots(figsize=(9, 5))

                    def _plot(arr: np.ndarray, lab: str, style: Optional[str] = None, lw: float = 2.0):
                        a, x = _slice_window(arr, cen, half_window)
                        if a.size:
                            ax1.plot(x, a, label=lab, linestyle=style if style else "-", linewidth=lw)

                    _plot(vpl_lambda,          r"VPL $\lambda_t$")
                    _plot(vpl_lambda_preclip,  r"$\lambda_\mathrm{pre}$",  style="--", lw=1.8)
                    _plot(vpl_lambda_postclip, r"$\lambda_\mathrm{post}$", style=":",  lw=1.8)
                    _plot(vpl_v_ema,           "v_ema",   style="-.", lw=1.6)
                    _plot(vpl_v_ref,           "v_ref",   style="-.", lw=1.6)
                    _plot(vpl_v_batch_mean,    "v_batch", style="-.", lw=1.6)
                    _plot(vpl_entropy_mean,    "entropy", style=(0, (3, 1, 1, 1)), lw=1.6)

                    ax1.set_xlabel("Epoch")
                    ax1.set_ylabel("VPL internals / entropy")
                    ax1.grid(True, alpha=0.3)

                    ax2 = ax1.twinx()
                    lines_r, labels_r = [], []
                    lt, xt = _slice_window(loss_test_mean, cen, half_window)
                    if lt.size:
                        l = ax2.plot(xt, lt, label="Test loss", alpha=0.9, linewidth=2.2, color="tab:red")[0]
                        lines_r.append(l); labels_r.append("Test loss")
                    ltr, xtr = _slice_window(loss_train_mean, cen, half_window)
                    if ltr.size:
                        l = ax2.plot(xtr, ltr, label="Train total loss (epoch mean)", alpha=0.9, linestyle="--", color="tab:purple")[0]
                        lines_r.append(l); labels_r.append("Train total loss (epoch mean)")
                    ax2.set_ylabel("Loss")

                    lines_l, labels_l = ax1.get_legend_handles_labels()
                    ax1.legend(lines_l + lines_r, labels_l + labels_r, loc="upper left", ncol=2)

                    plt.title(f"{dataset.upper()} / {arch} - {cond} : VPL vs Loss ({tag} zoom)")
                    out = figs_dir / f"fig5z_vpl_vs_loss_{cond}_{tag}_{dataset}_{arch}.png"
                    fig.tight_layout(); fig.savefig(out, dpi=200); plt.close(fig)
                    outs.append(out)

        return outs

    print(f"[plot] reactivity ZOOM overlays -> {out_root}")
    overlay_zoom_figs = _save_reactivity_zoom_overlays(
        figs_dir=figs_dir,
        internals_by_cond=internals_by_cond,
        curves_teloss=curves_teloss,
        dataset=dataset, arch=arch,
        total_epochs=total_epochs,
        zoom_epochs_arg=zoom_epochs,
        half_window=int(max(1, zoom_ep_per_phase))  # CLI arg used as half-window
    )

    # ---------- Per-batch previews ----------
    batches_by_cond = load_batch_csvs_recursive(root_for_optionals)
    batch_preview = save_batch_preview(figs_dir, batches_by_cond, dataset, arch)

    print(f"[plot] zoomed step previews (all conditions) -> {out_root}")
    zoom_figs = save_zoom_step_previews_allconds(
        figs_dir=figs_dir,
        df_runs=df[["run_dir", "seed"]].drop_duplicates(),
        dataset=dataset,
        arch=arch,
        total_epochs=total_epochs,
        zoom_epochs_arg=zoom_epochs,
        zoom_batches=zoom_batches,
        batches_by_cond=batches_by_cond
    )

    # ---------- Curves exports + tables ----------
    print(f"[export] per-epoch mean curves -> {out_root}")
    curves_dir = export_per_epoch_means(out_root, curves_tr, curves_acc, curves_teloss)

    print(f"[table] summaries and paired tests -> {out_root}")
    t1, t2 = make_summary_tables(tables_dir, df, higher_is_better=higher_is_better)

    # ---------- Simple report ----------
    report_path = out_root / "report.md"
    with open(report_path, "w", encoding="utf-8") as out_f:
        out_f.write(f"# VML Analysis - {dataset.upper()} / {arch}\n\n")
        out_f.write("## Figures\n")
        out_f.write(f"- Fig 1a: Final performance across seeds (Violin) - `{Path(f1a).name}`\n")
        out_f.write(f"- Fig 1b: Final performance across seeds (Box) - `{Path(f1b).name}`\n")
        out_f.write(f"- Fig 1c: Final performance across seeds (KDE) - `{Path(f1c).name}`\n")
        out_f.write(f"- Fig 2a: Test loss (mean +/- SD) - `{Path(f2a).name}`\n")
        out_f.write(f"- Fig 2b: Test loss volatility (rolling std) - `{Path(f2b).name}`\n")
        out_f.write(f"- Fig 3: Variability vs performance (Pareto) - `{Path(f3).name}`\n")

        if overlay_figs:
            out_f.write("\n## Reactivity overlays\n")
            for p in overlay_figs:
                out_f.write(f"- `{Path(p).name}`\n")

        if overlay_zoom_figs:
            out_f.write("\n## Reactivity overlays (ZOOMED: early/middle/late)\n")
            for p in overlay_zoom_figs:
                out_f.write(f"- `{Path(p).name}`\n")

        out_f.write("\n## Figures (internals / appendix)\n")
        for p in internals_figs:
            if p:
                out_f.write(f"- `{Path(p).name}`\n")
        if batch_preview:
            out_f.write(f"- Step-level preview (all conditions): `{Path(batch_preview).name}`\n")
        if zoom_figs:
            out_f.write("\n## Zoomed step previews (all conditions)\n")
            for zp in zoom_figs:
                if zp:
                    out_f.write(f"- `{Path(zp).name}`\n")

        out_f.write("\n## Curves (CSV exports)\n")
        out_f.write(f"- Per-epoch means directory: `./{Path(curves_dir).relative_to(out_root)}`\n")
        out_f.write(f"  (files for: {', '.join(COND_ORDER)})\n")

        out_f.write("\n## Tables\n")
        out_f.write(f"- Runs (raw): `{Path(df_out).name}`\n")
        out_f.write(f"- Summary by condition: `{Path(t1).name}`\n")
        out_f.write(f"- Paired tests vs ref: `{Path(t2).name}`\n")
        if not reactivity_df.empty:
            out_f.write(f"- Reactivity by condition: `{Path(react_csv).name}`\n")

        out_f.write("\n## Notes\n")
        out_f.write("- When --hp_tag_prefix is used, conditions are the tag tokens (e.g., VW002, VW005, ...).\n")
        out_f.write("- SL/VPL internals: early zeros are masked; modules start when their start_epoch/warmup allows.\n")
        out_f.write("- Reactivity table uses first-difference signals and scans lags in [-10,+10] epochs.\n")
        out_f.write("- Zoomed overlays use centers from --zoom_epochs (or auto) and a half-window of --zoom_ep_per_phase.\n")
        out_f.write("- Batch CSVs must exist for each condition to appear in step/zoom plots.\n")

    print(f"Done -> {out_root}")

    # Optional push to W&B
    if int(auto_push_wandb) == 1:
        push_cmd = [sys.executable, "wandb_push_analysis.py",
                    "--project", wandb_project,
                    "--analysis_dir", str(out_root)]
        if wandb_entity:
            push_cmd += ["--entity", wandb_entity]
        try:
            print(f"[wandb] pushing analysis via: {' '.join(push_cmd)}")
            subprocess.run(push_cmd, check=False)
        except Exception as e:
            print(f"[wandb] push skipped: {e}")




# ---------------- CLI and main ----------------

def parse_seed_list(s: Optional[str]) -> Set[int]:
    if not s: return set()
    toks = re.split(r"[,\s]+", s.strip())
    vals = set()
    for t in toks:
        if not t: continue
        try:
            vals.add(int(t))
        except Exception:
            pass
    return vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True, help="Root folder with run subdirs (each has log.txt).")
    ap.add_argument("--dataset", type=str, required=True)
    ap.add_argument("--arch", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--rolling_window", type=int, default=5)
    ap.add_argument("--higher_is_better", type=int, default=1)

    # Zoom controls
    ap.add_argument("--zoom_batches", type=int, default=0, help="Batches per epoch in zoom figs. 0=all.")
    ap.add_argument("--zoom_epochs", type=str, default="auto", help="'auto' or 'E,M,L' like '5,100,190'.")
    ap.add_argument("--zoom_ep_per_phase", type=int, default=3)

    # Overall averaging controls
    ap.add_argument("--avg_seeds", type=int, default=0,
                    help="Use first N seeds per condition (by seed id). 0=use all.")
    ap.add_argument("--avg_seed_list", type=str, default="",
                    help="Comma/space list of seeds to include across conditions (overrides --avg_seeds).")

    # Per-seed analyses
    ap.add_argument("--per_seed_list", type=str, default="",
                    help="Comma/space list of seeds to generate single-seed analyses for.")
    ap.add_argument("--per_seed_all", type=int, default=0,
                    help="If 1, generate per-seed analyses for all discovered seeds.")

    # W&B push
    ap.add_argument("--auto_push_wandb", type=int, default=1)
    ap.add_argument("--wandb_project", type=str, default="vml-analysis")
    ap.add_argument("--wandb_entity", type=str, default=None)

    # sweep grouping by tag prefix
    ap.add_argument("--hp_tag_prefix", type=str, default=None,
                    help="Group runs by a hyperparameter tag prefix in folder names (e.g., 'VW' groups VW002, VW005, ...).")

    args = ap.parse_args()

    root = Path(args.root)
    hib = bool(args.higher_is_better)

    # Configure sweep behavior and condition order
    global HP_TAG_PREFIX, COND_ORDER
    HP_TAG_PREFIX = str(args.hp_tag_prefix).strip() if args.hp_tag_prefix else None

    if HP_TAG_PREFIX:
        # Discover tag conditions under root
        conds = []
        for run_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
            c = infer_condition_from_name(run_dir.name)
            if c:
                conds.append(c)
        conds = sorted(set(conds), key=lambda s: (
            (re.match(r"([A-Za-z]+)", s).group(1).upper() if re.match(r"([A-Za-z]+)", s) else s.upper()),
            int(re.search(r"(\d+)$", s).group(1)) if re.search(r"(\d+)$", s) else 0
        ))
        if not conds:
            raise SystemExit(f"No runs found for sweep prefix '{HP_TAG_PREFIX}' under {root}")
        COND_ORDER = conds
        print(f"[sweep] prefix={HP_TAG_PREFIX} -> conditions: {COND_ORDER}")
    else:
        COND_ORDER = ["base", "sl", "vpl", "vml"]
        print(f"[default] conditions: {COND_ORDER}")

    print(f"[load] scanning runs in: {root}")
    df_all, curves_tr_all, curves_acc_all, curves_teloss_all = load_runs(
        root, args.dataset, args.arch, higher_is_better=hib
    )
    if df_all.empty:
        raise SystemExit("No runs found with log.txt (and identifiable condition).")

    # Discover seeds
    all_seeds = sorted([int(s) for s in df_all["seed"].dropna().unique()])
    print(f"[info] discovered seeds: {all_seeds}")

    # ---------- Overall averaged analysis ----------
    out_overall = Path(args.out) / "overall"
    out_overall.mkdir(parents=True, exist_ok=True)

    # Decide averaging filter
    seed_list = parse_seed_list(args.avg_seed_list)
    if seed_list:
        print(f"[filter overall] using explicit seed list for averaging: {sorted(seed_list)}")
        df_over = filter_df_by_seed_set(df_all, seed_list)
        curves_tr_over = filter_curves_by_seed_set(curves_tr_all, seed_list)
        curves_acc_over = filter_curves_by_seed_set(curves_acc_all, seed_list)
        curves_teloss_over = filter_curves_by_seed_set(curves_teloss_all, seed_list)
    elif args.avg_seeds > 0:
        print(f"[filter overall] limiting to first {args.avg_seeds} seed(s) per condition")
        df_over, curves_tr_over, curves_acc_over, curves_teloss_over = limit_first_n_seeds_per_condition(
            df_all, curves_tr_all, curves_acc_all, curves_teloss_all, int(args.avg_seeds)
        )
    else:
        print("[filter overall] using all seeds")
        df_over = df_all.copy()
        curves_tr_over = {k: list(v) for k, v in curves_tr_all.items()}
        curves_acc_over = {k: list(v) for k, v in curves_acc_all.items()}
        curves_teloss_over = {k: list(v) for k, v in curves_teloss_all.items()}

    run_one_analysis(
        out_root=out_overall,
        df=df_over,
        curves_tr=curves_tr_over,
        curves_acc=curves_acc_over,
        curves_teloss=curves_teloss_over,
        dataset=args.dataset,
        arch=args.arch,
        higher_is_better=hib,
        root_for_optionals=root,
        rolling_window=args.rolling_window,
        zoom_batches=args.zoom_batches,
        zoom_epochs=args.zoom_epochs,
        zoom_ep_per_phase=args.zoom_ep_per_phase,
        auto_push_wandb=args.auto_push_wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity
    )

    # ---------- Per-seed analyses ----------
    seeds_for_single: Set[int] = parse_seed_list(args.per_seed_list)
    if int(args.per_seed_all) == 1:
        seeds_for_single.update(all_seeds)

    if seeds_for_single:
        print(f"[per-seed] generating analyses for seeds: {sorted(seeds_for_single)}")
        base_out = Path(args.out) / "per_seed"
        base_out.mkdir(parents=True, exist_ok=True)

        for s in sorted(seeds_for_single):
            seed_set = {int(s)}
            df_s = filter_df_by_seed_set(df_all, seed_set)
            if df_s.empty:
                print(f"[per-seed] seed {s}: no runs found, skipping.")
                continue
            curves_tr_s = filter_curves_by_seed_set(curves_tr_all, seed_set)
            curves_acc_s = filter_curves_by_seed_set(curves_acc_all, seed_set)
            curves_teloss_s = filter_curves_by_seed_set(curves_teloss_all, seed_set)

            out_seed = base_out / f"seed_{s}"
            out_seed.mkdir(parents=True, exist_ok=True)

            run_one_analysis(
                out_root=out_seed,
                df=df_s,
                curves_tr=curves_tr_s,
                curves_acc=curves_acc_s,
                curves_teloss=curves_teloss_s,
                dataset=args.dataset,
                arch=args.arch,
                higher_is_better=hib,
                root_for_optionals=root,
                rolling_window=args.rolling_window,
                zoom_batches=args.zoom_batches,
                zoom_epochs=args.zoom_epochs,
                zoom_ep_per_phase=args.zoom_ep_per_phase,
                auto_push_wandb=args.auto_push_wandb,
                wandb_project=args.wandb_project,
                wandb_entity=args.wandb_entity
            )

    print("\nAll analyses complete.")
    print(f"- Root output: {Path(args.out)}")


if __name__ == "__main__":
    main()
