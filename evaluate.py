#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate.py

Read all *.parquet in results/, compute for each model
  - RMSE | SMAPE | MASE | R2
  - Back-test stats: TotalReturn | CAGR | SuccessRate

Print them in a fixed order.
"""

import sys
import math
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

# ─────────────────────────────────────────────────────────────
# 1. metric helpers
# ─────────────────────────────────────────────────────────────
def smape(y_t, y_p, eps=1e-8):
    denom = np.abs(y_t) + np.abs(y_p) + eps
    return np.mean(2.0 * np.abs(y_t - y_p) / denom) * 100

def mase(y_t, y_p, groups, m=1):
    mae = np.abs(y_t - y_p).mean()
    denom = []
    for g in np.unique(groups):
        v = y_t[groups == g]
        if len(v) > m:
            denom.append(np.abs(np.diff(v, n=m)).mean())
    return mae / (np.mean(denom) + 1e-8) if denom else np.nan

def gather_metrics(df):
    """
    Compute RMSE, SMAPE, MASE, R2 for the raw sliding-window DataFrame.
    """
    df = df.dropna(subset=["y_true", "y_pred", "lsoa"])
    if df.empty:
        return {"RMSE": np.nan, "SMAPE": np.nan, "MASE": np.nan, "R2": np.nan}
    y_t = df["y_true"].values
    y_p = df["y_pred"].values
    g   = df["lsoa"].values
    return {
        "RMSE":  math.sqrt(mean_squared_error(y_t, y_p)),
        "SMAPE": smape(y_t, y_p),
        "MASE":  mase(y_t, y_p, g),
        "R2":    r2_score(y_t, y_p),
    }

# ─────────────────────────────────────────────────────────────
# 2. panel builder
# ─────────────────────────────────────────────────────────────
def build_panel(y_true, y_pred, years, lsoa_ids):
    """
    Aggregate sliding-window outputs into one row per (lsoa, year),
    taking the median to de-duplicate overlapping windows.
    """
    df = pd.DataFrame({
        "lsoa":   lsoa_ids,
        "year":   years,
        "y_true": y_true,
        "y_pred": y_pred,
    })
    return df.groupby(["lsoa", "year"], as_index=False).median()

# ─────────────────────────────────────────────────────────────
# 3. back-test – exactly the notebook logic
# ─────────────────────────────────────────────────────────────
def backtest_from_panel(
    df_panel: pd.DataFrame,
    top_k: int = 4000,
    start_cap: float = 1_000_000,
    val_start: int = 2020,
    val_end:   int = 2023,
):
    df = df_panel.sort_values(["lsoa", "year"]).copy()
    df["y_true_next"] = df.groupby("lsoa")["y_true"].shift(-1)
    df["y_pred_next"] = df.groupby("lsoa")["y_pred"].shift(-1)
    df = df[(df["year"] > val_start) & (df["year"] <= val_end)]
    df = df.dropna(subset=["y_true_next"])
    df["pred_pct"] = (df["y_pred_next"] - df["y_true"]) / df["y_true"]
    df["real_pct"] = (df["y_true_next"] - df["y_true"]) / df["y_true"]

    cap = start_cap
    equity = []
    success_rate = np.nan
    for yr, grp in df.groupby("year"):
        top = grp.nlargest(top_k, "pred_pct")
        top = top[top["pred_pct"] > 0.0]
        if not top.empty:
            top_pos = top[top["real_pct"] > 0.0]
            success_rate = len(top_pos) / len(top)
            rate = top["real_pct"].mean()
            cap *= (1 + rate)
            equity.append((yr, rate, cap))

    if not equity:
        return {"TotalReturn": "N/A", "CAGR": "N/A", "SuccessRate": "N/A"}

    first_cap = equity[0][2]
    last_cap  = equity[-1][2]
    total_ret = last_cap / first_cap - 1
    nyrs = len(equity)
    cagr = (1 + total_ret) ** (1 / nyrs) - 1
    return {
        "TotalReturn": f"{total_ret*100:.2f}%",
        "CAGR":        f"{cagr*100:.2f}%",
        "SuccessRate": f"{success_rate*100:.2f}%"
    }

# ─────────────────────────────────────────────────────────────
# 4. CLI & startup banner
# ─────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Model evaluation & back-test")
parser.add_argument("--results_dir", default="results",
                    help="Folder with model .parquet files")
args = parser.parse_args()

print(f"🔍 Evaluating models\n")

root = Path(args.results_dir)
if not root.is_dir():
    sys.exit(f"[Error] '{root}' is not a directory.")
parquets = {p.stem: p for p in root.glob("*.parquet")}
if not parquets:
    sys.exit(f"[Error] No .parquet files found in '{root}'.")

# ─────────────────────────────────────────────────────────────
# 5. compute metrics & back-test
# ─────────────────────────────────────────────────────────────
order = [
    "GPT-4o", "Qwen", "Gemini-2",
    "RNN", "LSTM", "ST-RNN", "ST-LSTM",
    "Win-LSTR", "Win-LSTL", "WinFM"
]

eval_m, bt_m = {}, {}
for m in order:
    fp = parquets.get(m)
    if fp:
        df = pd.read_parquet(fp)
        eval_m[m] = gather_metrics(df)
        panel = build_panel(
            df["y_true"].values,
            df["y_pred"].values,
            df["year"].values,
            df["lsoa"].values,
        )
        bt_m[m] = backtest_from_panel(panel)
    else:
        eval_m[m] = {"RMSE": np.nan, "SMAPE": np.nan, "MASE": np.nan, "R2": np.nan}
        bt_m[m]   = {"TotalReturn": "N/A", "CAGR": "N/A", "SuccessRate": "N/A"}

eval_df = pd.DataFrame(eval_m).T
bt_df   = pd.DataFrame(bt_m).T

# ─────────────────────────────────────────────────────────────
# 6. print results
# ─────────────────────────────────────────────────────────────
print("=== Evaluation Metrics ===")
print("Model\tRMSE\tSMAPE\tMASE\tR2")
for m, row in eval_df.iterrows():
    print(f"{m}\t{row.RMSE:.1f}\t{row.SMAPE:.2f}%\t{row.MASE:.3f}\t{row.R2:.3f}")

print("\n=== Back-test Results ===")
print("Model\tTotalReturn\tCAGR\tSuccessRate")
for m, row in bt_df.iterrows():
    print(f"{m}\t{row.TotalReturn}\t{row.CAGR}\t{row.SuccessRate}")
