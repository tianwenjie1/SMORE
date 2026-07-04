#!/usr/bin/env python3
# coding: utf-8
"""
Parse SMORE experiment logs and extract Test metrics into a CSV.

Log filename convention (any of these tokens may appear):
    SMORE_{dataset}_{method}_seed{seed}[_rate{rate}][_{robust_mode}].log
    e.g. SMORE_sports_MDR_seed42_drop_image.log
         SMORE_clothing_FBG+MDR_rate0.2_seed999.log
         SMORE_baby_baseline_seed999.log

The script extracts the LAST "Test:" line from each log and pulls out all
metrics. It also infers dataset / method / seed / robust_mode / dropout_rate
from the filename so downstream analysis is easy.

Usage:
    python scripts/parse_smore_results.py <log_dir> [-o output.csv]
    python scripts/parse_smore_results.py logs_robustness_20260702_120000 -o results_robustness.csv
"""

import os
import re
import csv
import argparse
import glob


# metrics to extract, in output order
METRIC_KEYS = [
    "recall@5", "recall@10", "recall@20", "recall@50",
    "ndcg@5", "ndcg@10", "ndcg@20", "ndcg@50",
    "precision@5", "precision@10", "precision@20", "precision@50",
    "map@5", "map@10", "map@20", "map@50",
]

OUTPUT_FIELDS = [
    "dataset", "method", "seed", "robust_mode", "dropout_rate",
    "recall@10", "recall@20", "recall@50",
    "ndcg@10", "ndcg@20", "ndcg@50",
    "precision@10", "map@10",
    "log_file",
]


def parse_metrics_from_line(line):
    """Extract metric: value pairs from a Test:/Valid: line."""
    out = {}
    # pattern like  recall@10: 0.0660    ndcg@5: 0.0280
    for m in re.finditer(r"([a-zA-Z]+@\d+):\s*([0-9.]+)", line):
        key = m.group(1).lower()
        try:
            out[key] = float(m.group(2))
        except ValueError:
            pass
    return out


def parse_filename(fname):
    """Infer dataset, method, seed, robust_mode, dropout_rate from filename."""
    base = os.path.basename(fname)
    base = base[:-4] if base.endswith(".log") else base  # strip .log
    if base.startswith("SMORE_"):
        base = base[len("SMORE_"):]

    info = {
        "dataset": "", "method": "", "seed": "",
        "robust_mode": "normal", "dropout_rate": "",
    }

    # 1. robust mode SUFFIX (modes contain underscores, so check suffix first)
    #    check longer/multi-word modes before the bare 'normal'
    mode_set = ["drop_image", "drop_text", "noise_image", "noise_text",
                "noise_both", "normal"]
    for mode in mode_set:
        if base.endswith("_" + mode):
            info["robust_mode"] = mode
            base = base[:-(len(mode) + 1)]  # strip '_<mode>'
            break

    # 2. seed  _seed999
    m = re.search(r"_seed(\d+)", base)
    if m:
        info["seed"] = m.group(1)
        base = base[:m.start()] + base[m.end():]

    # 3. dropout rate  _rate0.2
    m = re.search(r"_rate([0-9.]+)", base)
    if m:
        info["dropout_rate"] = m.group(1)
        base = base[:m.start()] + base[m.end():]

    # 4. remaining: dataset_method (method names like baseline/FBG/MDR/FBG+MDR
    #    contain no underscore, so split on '_')
    tokens = [t for t in base.split("_") if t]
    if tokens:
        info["dataset"] = tokens[0]
        info["method"] = "_".join(tokens[1:]) if len(tokens) > 1 else ""

    for k in ("dataset", "method"):
        info[k] = info[k].strip("_")
    return info


def find_last_test_line(log_path):
    """Return the last line containing 'Test:' (the final best test result)."""
    last = None
    try:
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if "Test:" in line or "test result" in line.lower():
                    last = line
    except OSError:
        return None
    return last


def main():
    ap = argparse.ArgumentParser(description="Parse SMORE logs to CSV.")
    ap.add_argument("log_dirs", nargs="+", help="one or more dirs/globs containing SMORE_*.log")
    ap.add_argument("-o", "--output", default="smore_results.csv",
                    help="output CSV path (default: smore_results.csv)")
    args = ap.parse_args()

    # collect files from all given dirs/globs, dedupe by basename
    files = []
    seen = set()
    for pat in args.log_dirs:
        if os.path.isdir(pat):
            pat = os.path.join(pat, "SMORE_*.log")
        for fp in sorted(glob.glob(pat)):
            bn = os.path.basename(fp)
            if bn not in seen:
                seen.add(bn)
                files.append(fp)

    if not files:
        print(f"[WARN] no SMORE_*.log found under {args.log_dirs}")
        return

    rows = []
    for fp in files:
        info = parse_filename(fp)
        line = find_last_test_line(fp)
        if line is None:
            info.update({k: "" for k in METRIC_KEYS})
            info["log_file"] = os.path.basename(fp)
            info["_status"] = "no_test"
            rows.append(info)
            continue
        metrics = parse_metrics_from_line(line)
        info.update({k: metrics.get(k, "") for k in METRIC_KEYS})
        info["log_file"] = os.path.basename(fp)
        info["_status"] = "ok"
        rows.append(info)

    # write CSV
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS + ["status"])
        writer.writeheader()
        for r in rows:
            r["status"] = r.pop("_status", "")
            # only keep output fields + status
            row = {k: r.get(k, "") for k in OUTPUT_FIELDS}
            row["status"] = r["status"]
            writer.writerow(row)

    ok = sum(1 for r in rows if r.get("status") == "ok")
    print(f"[OK] parsed {ok}/{len(rows)} logs with test results -> {args.output}")


if __name__ == "__main__":
    main()
