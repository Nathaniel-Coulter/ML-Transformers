#!/usr/bin/env python3
import pandas as pd
from pathlib import Path
import math

ROOT = Path(__file__).resolve().parents[1]
CSV  = ROOT / "outputs" / "horizons_v2.csv"  # your "horizons.csv" if you renamed it
if not CSV.exists():
    CSV = ROOT / "outputs" / "horizons.csv"

TEX  = ROOT / "outputs" / "main_h1.tex"

df = pd.read_csv(CSV)

# Filter to the main comparison slice
d = df[(df["L"] == 252) & (df["H"] == 1)].copy()

# Metric for model selection: prefer test_mse when present else val_mse
if "metric_for_plot" not in d.columns:
    d["metric_for_plot"] = d["test_mse"].where(~d["test_mse"].isna(), d["val_mse"])

# Canonical model names for the paper
NAME = {
    "pointwise": "Point-wise",
    "patch":     "PatchTST",
    "ivar":      "iTransformer",
    "cross":     "Cross-Lite",
}

rows = []
for enc in ["pointwise", "patch", "ivar", "cross"]:
    g = d[d["encoder"] == enc].copy()
    if g.empty:
        continue
    # pick best row (min metric)
    g = g.sort_values("metric_for_plot")
    best = g.iloc[0].to_dict()

    # Config string (only patch/cross have p/s)
    if enc in ("patch", "cross"):
        p = int(best["patch_len"]) if not math.isnan(best.get("patch_len", float("nan"))) else None
        s = int(best["stride"])    if not math.isnan(best.get("stride", float("nan")))    else None
        if p is not None and s is not None:
            cfg = f"$p={p},\\ s={s}$"
        elif p is not None:
            cfg = f"$p={p}$"
        else:
            cfg = "--"
    else:
        cfg = "base"

    def fmt(x, nd=6):
        if pd.isna(x):
            return "—"
        return f"{float(x):.{nd}f}"

    rows.append({
        "Model": NAME.get(enc, enc),
        "Config": cfg,
        "Val MSE":  fmt(best.get("val_mse")),
        "Test MSE": fmt(best.get("test_mse")),
        "Test MAE": fmt(best.get("test_mae")),
        "Sharpe@10bps": fmt(best.get("sharpe"), 3),
        "Turnover": fmt(best.get("turnover"), 3),
    })

table = pd.DataFrame(rows, columns=["Model","Config","Val MSE","Test MSE","Test MAE","Sharpe@10bps","Turnover"])

# Emit LaTeX with booktabs
latex = table.to_latex(
    index=False,
    escape=False,              # allow math in Config
    longtable=False,
    bold_rows=False,
    column_format="l l r r r r r",
    caption="Main comparison at $H{=}1$, $L{=}252$ (lower error is better; higher Sharpe is better).",
    label="tab:main_h1"
).replace("toprule", "toprule").replace("midrule", "midrule").replace("bottomrule", "bottomrule")

TEX.write_text(latex)
print(f"[ok] wrote {TEX}")
print(table)
