#!/usr/bin/env python3
# coding: utf-8
"""
Generate publication-style plots from SMORE result CSVs.

Reads CSVs produced by parse_smore_results.py and produces PNGs:
  1. fig_rate_sensitivity.png   - dropout rate vs Recall@10 (from rate search CSV)
  2. fig_robustness.png         - Recall@10 under 6 perturbation modes (bar chart)
  3. fig_noise_level.png        - noise std vs Recall@10 (from noise level CSV)

Usage:
    python scripts/plot_results.py --rate results_mdr_rate.csv \
        --robust results_robustness.csv --noise results_noise_level.csv \
        --outdir figs

Each input is optional; missing CSVs are skipped.
Requires: pandas, matplotlib.
"""

import os
import argparse
import csv
from collections import defaultdict

try:
    import pandas as pd
except ImportError:
    pd = None

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------- helpers ----------
def load_csv(path):
    if not path or not os.path.exists(path):
        return None
    if pd is not None:
        return pd.read_csv(path)
    # fallback: csv module -> list of dict
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def to_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def mean_std(rows, value_col):
    """rows: list of dict-like; return (mean, std) of value_col."""
    vals = [to_float(r[value_col]) for r in rows if to_float(r[value_col]) is not None]
    if not vals:
        return None, None
    if pd is not None:
        import statistics
        m = sum(vals) / len(vals)
        s = statistics.stdev(vals) if len(vals) > 1 else 0.0
        return m, s
    m = sum(vals) / len(vals)
    s = (sum((v - m) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5 if len(vals) > 1 else 0.0
    return m, s


def group_mean(df, group_cols, value_col):
    """df: pandas DF or list-of-dict. Returns dict {group_key: (mean, std)}."""
    out = {}
    if df is None:
        return out
    if pd is not None and isinstance(df, pd.DataFrame):
        for key, grp in df.groupby(group_cols):
            vals = grp[value_col].dropna().astype(float)
            if len(vals) == 0:
                continue
            m = vals.mean()
            s = vals.std() if len(vals) > 1 else 0.0
            out[key if isinstance(key, tuple) else (key,)] = (m, s)
    else:
        buckets = defaultdict(list)
        for r in df:
            try:
                key = tuple(r[c] for c in group_cols)
            except KeyError:
                continue
            v = to_float(r[value_col])
            if v is not None:
                buckets[key].append(v)
        for key, vals in buckets.items():
            m = sum(vals) / len(vals)
            s = (sum((v - m) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5 if len(vals) > 1 else 0.0
            out[key] = (m, s)
    return out


# ---------- plot 1: rate sensitivity ----------
def plot_rate_sensitivity(rate_csv, outdir):
    df = load_csv(rate_csv)
    if df is None:
        print("[skip] rate sensitivity (no CSV)")
        return
    datasets = sorted(set(r["dataset"] for r in _iter(df)))
    methods = ["MDR", "FBG+MDR"]
    fig, axes = plt.subplots(1, len(datasets), figsize=(5 * len(datasets), 4), sharey=True)
    if len(datasets) == 1:
        axes = [axes]
    for ax, ds in zip(axes, datasets):
        for method in methods:
            # group by (dataset, method) -> but we want per-rate, so group by (dataset, method, rate)
            g = group_mean(_filter(df, dataset=ds, method=method), ["dropout_rate"], "recall@10")
            rates = sorted(g.keys(), key=lambda k: to_float(k[0]) or 0)
            xs = [to_float(k[0]) for k in rates]
            ys = [g[k][0] for k in rates]
            es = [g[k][1] for k in rates]
            ax.errorbar(xs, ys, yerr=es, marker="o", label=method, capsize=3)
        ax.set_title(ds)
        ax.set_xlabel("modality_dropout_rate")
        ax.set_ylabel("Recall@10")
        ax.legend()
        ax.grid(alpha=0.3)
    fig.suptitle("Dropout Rate Sensitivity (3-seed mean ± std)")
    fig.tight_layout()
    path = os.path.join(outdir, "fig_rate_sensitivity.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[ok] {path}")


# ---------- plot 2: robustness bar chart ----------
def plot_robustness(robust_csv, outdir):
    df = load_csv(robust_csv)
    if df is None:
        print("[skip] robustness (no CSV)")
        return
    datasets = sorted(set(r["dataset"] for r in _iter(df)))
    methods = ["baseline", "MDR", "FBG+MDR"]
    modes = ["normal", "drop_image", "drop_text", "noise_image", "noise_text", "noise_both"]
    fig, axes = plt.subplots(1, len(datasets), figsize=(5 * len(datasets), 4), sharey=True)
    if len(datasets) == 1:
        axes = [axes]
    width = 0.25
    for ax, ds in zip(axes, datasets):
        x = range(len(modes))
        for i, method in enumerate(methods):
            g = group_mean(_filter(df, dataset=ds, method=method), ["robust_mode"], "recall@10")
            ys = [g.get((m,), (None, None))[0] for m in modes]
            ax.bar([xi + i * width for xi in x], ys, width=width, label=method)
        ax.set_xticks([xi + width for xi in x])
        ax.set_xticklabels(modes, rotation=30, ha="right")
        ax.set_title(ds)
        ax.set_ylabel("Recall@10")
        ax.legend()
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle("Robustness under Inference-time Perturbation (3-seed mean)")
    fig.tight_layout()
    path = os.path.join(outdir, "fig_robustness.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[ok] {path}")


# ---------- plot 3: noise level ----------
def plot_noise_level(noise_csv, outdir):
    df = load_csv(noise_csv)
    if df is None:
        print("[skip] noise level (no CSV)")
        return
    datasets = sorted(set(r["dataset"] for r in _iter(df)))
    modes = sorted(set(r["robust_mode"] for r in _iter(df)))
    methods = ["baseline", "MDR"]
    fig, axes = plt.subplots(len(datasets), len(modes),
                             figsize=(5 * len(modes), 4 * len(datasets)), sharey="row")
    if len(datasets) == 1:
        axes = [axes]
    for ri, ds in enumerate(datasets):
        for ci, mode in enumerate(modes):
            ax = axes[ri][ci] if len(datasets) > 1 else axes[ci]
            for method in methods:
                g = group_mean(_filter(df, dataset=ds, method=method, robust_mode=mode),
                               ["robust_noise_std"], "recall@10")
                stds = sorted(g.keys(), key=lambda k: to_float(k[0]) or 0)
                xs = [to_float(k[0]) for k in stds]
                ys = [g[k][0] for k in stds]
                es = [g[k][1] for k in stds]
                ax.errorbar(xs, ys, yerr=es, marker="o", label=method, capsize=3)
            ax.set_title(f"{ds} / {mode}")
            ax.set_xlabel("noise std")
            if ci == 0:
                ax.set_ylabel("Recall@10")
            ax.legend()
            ax.grid(alpha=0.3)
    fig.suptitle("Robustness vs Noise Level (3-seed mean ± std)")
    fig.tight_layout()
    path = os.path.join(outdir, "fig_noise_level.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[ok] {path}")


# ---------- tiny helpers for list-of-dict / df agnosticism ----------
def _iter(df):
    if df is None:
        return []
    if pd is not None and isinstance(df, pd.DataFrame):
        return df.to_dict("records")
    return df


def _filter(df, **kwargs):
    rows = _iter(df)
    out = []
    for r in rows:
        ok = True
        for k, v in kwargs.items():
            if str(r.get(k, "")) != str(v):
                ok = False
                break
        if ok:
            out.append(r)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rate", default="results_mdr_rate.csv",
                    help="rate search CSV (default: results_mdr_rate.csv)")
    ap.add_argument("--robust", default="results_robustness.csv",
                    help="robustness CSV (default: results_robustness.csv)")
    ap.add_argument("--noise", default="results_noise_level.csv",
                    help="noise level CSV (default: results_noise_level.csv)")
    ap.add_argument("--outdir", default="figs", help="output dir for PNGs")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    plot_rate_sensitivity(args.rate, args.outdir)
    plot_robustness(args.robust, args.outdir)
    plot_noise_level(args.noise, args.outdir)
    print(f"\nDone. Figures in {args.outdir}/")


if __name__ == "__main__":
    main()
