# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
analysis_multi.py - unified analysis for VML experiments (classification)

Fixes in this version
---------------------
- Fig 2 uses TEST loss (mean+/-SD + volatility) instead of train loss.
- Internals plots mask leading zeros (pre-activation/warmup) as NaN.
- figA_step_preview overlays ALL conditions (recursive CSV search).
- Zoomed step previews (early/middle/late) overlay ALL conditions and
  correctly handle 0-based epochs in batches CSV (no epoch text in legend).
"""

import argparse
import os
import re
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
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

COND_ORDER = ["base", "base_rnd", "sl", "sl_rnd", "vpl", "vpl_rnd", "vml", "vml_rnd"]

# ---------------- Basic helpers ----------------

def infer_condition_from_run_dir(run_dir: Path) -> Optional[str]:
    name = run_dir.name
    lower = name.lower()
    # detect the base condition
    if   "vml" in lower: cond = "vml"
    elif "vpl" in lower: cond = "vpl"
    elif "sl"  in lower: cond = "sl"
    elif "base" in lower or "baseline" in lower: cond = "base"
    else:
        return None
    # detect non-deterministic runs by naming convention
    is_rnd = ("_rnd" in lower) or ("-rnd" in lower) or lower.endswith("rnd") or ("seedrnd" in lower)
    return f"{cond}_rnd" if is_rnd else cond


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

# ---------------- Loading runs and curves ----------------

def load_runs(root: Path, dataset: str, arch: str, higher_is_better: bool
              ) -> Tuple[pd.DataFrame,
                         Dict[str, List[pd.Series]],
                         Dict[str, List[pd.Series]],
                         Dict[str, List[pd.Series]]]:
    rows = []
    curves_tr: Dict[str, List[pd.Series]] = {c: [] for c in COND_ORDER}
    curves_acc: Dict[str, List[pd.Series]] = {c: [] for c in COND_ORDER}
    curves_teloss: Dict[str, List[pd.Series]] = {c: [] for c in COND_ORDER}

    for run_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        cond = infer_condition_from_run_dir(run_dir)
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
    out = {c: [] for c in COND_ORDER}
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
        out = {c: [] for c in COND_ORDER}
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
    by_cond = {c: [] for c in COND_ORDER}
    for run_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        cond = infer_condition_from_run_dir(run_dir)
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
    by_cond = {c: [] for c in COND_ORDER}
    for run_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        cond = infer_condition_from_run_dir(run_dir)
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

def save_violin(figs_dir: Path, df: pd.DataFrame, higher_is_better: bool, dataset: str, arch: str) -> Path:
    metric_name = "Accuracy (%)" if higher_is_better else "MAE (lower better)"
    if df is None or df.empty or "final_metric" not in df.columns:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5, "No results found.", ha="center", va="center", fontsize=12)
        ax.axis("off")
        out = figs_dir / f"fig1_distribution_{dataset}_{arch}.png"
        fig.tight_layout(); fig.savefig(out, dpi=200); plt.close(fig)
        return out

    tmp = df.dropna(subset=["final_metric"]).copy()
    tmp["condition"] = tmp.get("condition", "base").astype(str).str.lower()
    existing_conds = list(tmp["condition"].unique())
    order = [c for c in COND_ORDER if c in existing_conds] or sorted(existing_conds)

    groups, labels = [], []
    for c in order:
        arr = pd.to_numeric(tmp.loc[tmp["condition"] == c, "final_metric"], errors="coerce").dropna().to_numpy()
        if arr.size > 0:
            groups.append(arr); labels.append(c.upper())

    fig, ax = plt.subplots(figsize=(8, 5))
    if len(groups) == 0:
        ax.text(0.5, 0.5, "No completed runs to plot yet.", ha="center", va="center", fontsize=12)
        ax.axis("off")
    else:
        ax.violinplot(groups, showmeans=True, showextrema=True, widths=0.85)
        ax.set_xticks(range(1, len(labels) + 1)); ax.set_xticklabels(labels)
        ax.set_ylabel(metric_name)
        ax.set_title(f"{dataset.upper()} / {arch} - Final performance across seeds")
        ax.grid(True, axis='y', alpha=0.3)
    out = figs_dir / f"fig1_distribution_{dataset}_{arch}.png"
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

    # Panel A: mean+/-sd test loss
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    for cond in COND_ORDER:
        mean, std = stack_and_mean_std(curves_teloss.get(cond, []))
        if mean.size == 0: continue
        x = np.arange(1, len(mean)+1)
        ax1.plot(x, mean, label=cond.upper())
        ax1.fill_between(x, mean-std, mean+std, alpha=0.15)
    ax1.set_title(f"{dataset.upper()} / {arch} - Test loss (mean +/- SD)")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Test loss")
    ax1.grid(True, alpha=0.3); ax1.legend()
    out_a = figs_dir / f"fig2a_test_loss_{dataset}_{arch}.png"
    fig1.tight_layout(); fig1.savefig(out_a, dpi=200); plt.close(fig1)

    # Panel B: volatility (rolling std of test loss)
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    for cond in COND_ORDER:
        series_list = curves_teloss.get(cond, [])
        if not series_list: continue
        rstd_mat = []
        for s in series_list:
            rstd_mat.append(rolling_std(s.to_numpy(dtype=float), rolling_window))
        maxlen = max(len(r) for r in rstd_mat)
        arr = np.full((len(rstd_mat), maxlen), np.nan)
        for i, r in enumerate(rstd_mat):
            arr[i, :len(r)] = r
        mean = np.nanmean(arr, axis=0)
        ax2.plot(np.arange(1, len(mean)+1), mean, label=cond.upper())
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
        ax.scatter(sd, y, label=cond.upper(), s=70)
        ax.annotate(cond.upper(), (sd, y), xytext=(5,5), textcoords="offset points")
    ax.set_xlabel("Across-seed SD")
    ax.set_ylabel("Mean performance" + (" (higher)" if higher_is_better else " (-MAE; higher is better)"))
    ax.set_title(f"{dataset.upper()} / {arch} - Variability vs performance")
    ax.grid(True, alpha=0.3)
    out = figs_dir / f"fig3_pareto_{dataset}_{arch}.png"
    fig.tight_layout(); fig.savefig(out, dpi=200); plt.close(fig)
    return out

# ---------------- Internals plots and preview ----------------

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

    # SL
    fig1, ax1 = plt.subplots(figsize=(8,5))
    fig2, ax2 = plt.subplots(figsize=(8,5))
    for cond in COND_ORDER:
        lst = internals_by_cond.get(cond, [])
        if not lst: continue
        lam_list, sig_list = [], []
        for df in lst:
            if "sl_lambda" in df.columns:     lam_list.append(df["sl_lambda"].astype(float).to_numpy())
            if "sl_sigma_ema" in df.columns:  sig_list.append(df["sl_sigma_ema"].astype(float).to_numpy())
        lam_mean, lam_std = prep_series(lam_list) if lam_list else (np.array([]), np.array([]))
        sig_mean, sig_std = prep_series(sig_list) if sig_list else (np.array([]), np.array([]))
        if lam_mean.size:
            x = np.arange(len(lam_mean))
            ax1.plot(x, lam_mean, label=cond.upper()); ax1.fill_between(x, lam_mean-lam_std, lam_mean+lam_std, alpha=0.15)
        if sig_mean.size:
            x = np.arange(len(sig_mean))
            ax2.plot(x, sig_mean, label=cond.upper()); ax2.fill_between(x, sig_mean-sig_std, sig_mean+sig_std, alpha=0.15)
    ax1.set_title(f"{dataset.upper()} / {arch} - SL lambda_t (mean +/- SD)")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("lambda_t"); ax1.grid(True, alpha=0.3); ax1.legend()
    ax2.set_title(f"{dataset.upper()} / {arch} - SL sigma_ema (mean +/- SD)")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("sigma_ema"); ax2.grid(True, alpha=0.3); ax2.legend()
    out1 = figs_dir / f"figA_sl_lambda_{dataset}_{arch}.png"
    out2 = figs_dir / f"figA_sl_sigma_{dataset}_{arch}.png"
    fig1.tight_layout(); fig1.savefig(out1, dpi=200); plt.close(fig1)
    fig2.tight_layout(); fig2.savefig(out2, dpi=200); plt.close(fig2)

    # VPL
    fig3, ax3 = plt.subplots(figsize=(8,5))
    fig4, ax4 = plt.subplots(figsize=(8,5))
    for cond in COND_ORDER:
        lst = internals_by_cond.get(cond, [])
        if not lst: continue
        lam_list, vema_list = [], []
        for df in lst:
            if "vpl_lambda" in df.columns: lam_list.append(df["vpl_lambda"].astype(float).to_numpy())
            if "vpl_v_ema"  in df.columns: vema_list.append(df["vpl_v_ema"].astype(float).to_numpy())
        lam_mean, lam_std = prep_series(lam_list) if lam_list else (np.array([]), np.array([]))
        vema_mean, vema_std = prep_series(vema_list) if vema_list else (np.array([]), np.array([]))
        if lam_mean.size:
            x = np.arange(len(lam_mean))
            ax3.plot(x, lam_mean, label=cond.upper()); ax3.fill_between(x, lam_mean-lam_std, lam_mean+lam_std, alpha=0.15)
        if vema_mean.size:
            x = np.arange(len(vema_mean))
            ax4.plot(x, vema_mean, label=cond.upper()); ax4.fill_between(x, vema_mean-vema_std, vema_mean+vema_std, alpha=0.15)
    ax3.set_title(f"{dataset.upper()} / {arch} - VPL lambda_t (mean +/- SD)")
    ax3.set_xlabel("Epoch"); ax3.set_ylabel("lambda_t"); ax3.grid(True, alpha=0.3); ax3.legend()
    ax4.set_title(f"{dataset.upper()} / {arch} - VPL v_ema (mean +/- SD)")
    ax4.set_xlabel("Epoch"); ax4.set_ylabel("v_ema"); ax4.grid(True, alpha=0.3); ax4.legend()
    out3 = figs_dir / f"figA_vpl_lambda_{dataset}_{arch}.png"
    out4 = figs_dir / f"figA_vpl_vema_{dataset}_{arch}.png"
    fig3.tight_layout(); fig3.savefig(out3, dpi=200); plt.close(fig3)
    fig4.tight_layout(); fig4.savefig(out4, dpi=200); plt.close(fig4)
    return [out1, out2, out3, out4]

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
        ax.plot(np.arange(len(y)), y, label=cond.upper(), alpha=0.9)
        any_plotted = True
    if not any_plotted:
        plt.close(fig)
        return None
    ax.set_title(f"{dataset.upper()} / {arch} - Step-level loss preview (all conditions)")
    ax.set_xlabel("Batch step"); ax.set_ylabel("loss"); ax.grid(True, alpha=0.3); ax.legend(ncol=2)
    out = figs_dir / f"figA_step_preview_{dataset}_{arch}.png"
    fig.tight_layout(); fig.savefig(out, dpi=200); plt.close(fig)
    if missing:
        print("[step-preview] no per-batch CSV for:", ", ".join(m.upper() for m in missing))
    return out

# ---------------- Zoomed previews (early/middle/late) ----------------

def _pick_one_run_per_condition(df_runs: pd.DataFrame, batches_by_cond: Dict[str, List[pd.DataFrame]]) -> Dict[str, pd.DataFrame]:
    """Prefer a common seed across conditions; otherwise take first available per condition."""
    cond_seed = {c:{} for c in COND_ORDER}
    for cond in COND_ORDER:
        for df in batches_by_cond.get(cond, []):
            cond_seed[cond].setdefault("any", []).append(df)
    picks = {}
    for cond in COND_ORDER:
        lst = []
        for k in cond_seed[cond]:
            lst.extend(cond_seed[cond][k])
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
    # pick_zoom_centers already returns 0-based centers
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
            ax.plot(x, y, label=cond.upper())
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

# ---------------- Tables ----------------

def make_summary_tables(tables_dir: Path, df: pd.DataFrame, higher_is_better: bool) -> Tuple[Path, Path]:
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
        rows.append({"condition": cond, "n": n, "mean": mean, "sd": sd, "cv": cv,
                     "ci95_low": lo, "ci95_high": hi})
    summary = pd.DataFrame(rows)

    if len(base_vals) >= 2 and np.std(base_vals, ddof=1) > 0:
        base_sd = float(np.std(base_vals, ddof=1))
        sd_red, sd_ratio_ci = [], []
        for cond in conds_present:
            vals = df[df["condition"]==cond]["final_metric"].to_numpy()
            if len(vals) >= 2:
                sd = float(np.std(vals, ddof=1))
                pct = 100.0 * (1.0 - sd/base_sd)
                ratio, lo, hi = bootstrap_sd_ratio_ci(base_vals, vals)
            else:
                pct, ratio, lo, hi = (np.nan, np.nan, np.nan, np.nan)
            sd_red.append(pct); sd_ratio_ci.append((ratio, lo, hi))
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

# ---------------- One analysis run (reusable) ----------------

def run_one_analysis(out_root: Path,
                     df: pd.DataFrame,
                     curves_tr: Dict[str, List[pd.Series]],
                     curves_acc: Dict[str, List[pd.Series]],
                     curves_teloss: Dict[str, List[pd.Series]],
                     dataset: str, arch: str,
                     higher_is_better: bool,
                     root_for_optionals: Path,
                     rolling_window: int,
                     zoom_batches: int, zoom_epochs: str, zoom_ep_per_phase: int,  # zoom_ep_per_phase unused
                     auto_push_wandb: int, wandb_project: str, wandb_entity: Optional[str]):
    figs_dir, tables_dir = ensure_outdirs(out_root)

    if df.empty:
        raise SystemExit(f"No runs found for analysis at {out_root}")

    # Save raw run table
    df_out = tables_dir / "runs_raw.csv"
    df.to_csv(df_out, index=False)

    # total epochs inference
    total_epochs = 200
    if "epochs" in df.columns and not df["epochs"].isna().all():
        try:
            total_epochs = int(df["epochs"].max())
        except Exception:
            pass

    # Figures
    print(f"[plot] fig1 distribution -> {out_root}")
    f1 = save_violin(figs_dir, df, higher_is_better=higher_is_better, dataset=dataset, arch=arch)

    print(f"[plot] fig2 stability (TEST loss) -> {out_root}")
    f2a, f2b = save_test_stability(figs_dir, curves_teloss, rolling_window=rolling_window,
                                   dataset=dataset, arch=arch)

    print(f"[plot] fig3 pareto -> {out_root}")
    f3 = save_pareto(figs_dir, df, higher_is_better=higher_is_better, dataset=dataset, arch=arch)

    # Internals and step previews
    print(f"[plot] internals and step previews -> {out_root}")
    internals_by_cond = load_internals_csvs(root_for_optionals)
    internals_figs = save_internals_plots(figs_dir, internals_by_cond, dataset, arch)

    batches_by_cond = load_batch_csvs_recursive(root_for_optionals)
    batch_preview = save_batch_preview(figs_dir, batches_by_cond, dataset, arch)

    # Zoomed step previews (overlay all conditions)
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

    # Curves exports
    print(f"[export] per-epoch mean curves -> {out_root}")
    curves_dir = export_per_epoch_means(out_root, curves_tr, curves_acc, curves_teloss)

    # Tables
    print(f"[table] summaries and paired tests -> {out_root}")
    t1, t2 = make_summary_tables(tables_dir, df, higher_is_better=higher_is_better)

    # Simple report
    report_path = out_root / "report.md"
    with open(report_path, "w", encoding="utf-8") as out_f:
        out_f.write(f"# VML Analysis - {dataset.upper()} / {arch}\n\n")
        out_f.write("## Figures\n")
        out_f.write(f"- Fig 1: Final performance across seeds - `{Path(f1).name}`\n")
        out_f.write(f"- Fig 2a: Test loss (mean +/- SD) - `{Path(f2a).name}`\n")
        out_f.write(f"- Fig 2b: Test loss volatility (rolling std) - `{Path(f2b).name}`\n")
        out_f.write(f"- Fig 3: Variability vs performance (Pareto) - `{Path(f3).name}`\n\n")

        out_f.write("## Figures (internals / appendix)\n")
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
        out_f.write("  (files: base.csv, sl.csv, vpl.csv, vml.csv if present)\n")

        out_f.write("\n## Tables\n")
        out_f.write(f"- Runs (raw): `{Path(df_out).name}`\n")
        out_f.write(f"- Summary by condition: `{Path(t1).name}`\n")
        out_f.write(f"- Paired tests vs BASE: `{Path(t2).name}`\n\n")

        out_f.write("## Notes\n")
        out_f.write("- SL/VPL internals: early zeros are masked; modules start when their start_epoch/warmup allows.\n")
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

    args = ap.parse_args()

    root = Path(args.root)
    hib = bool(args.higher_is_better)

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


