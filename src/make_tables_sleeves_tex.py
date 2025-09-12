#!/usr/bin/env python3
"""
Build per-sleeve LaTeX tables from outputs/sleeves_h1.csv.

For each sleeve (equities, rates_credit, commod_alt):
- select the best row per encoder (prefer lowest Test MSE if present, else Val MSE)
- format a compact 1-column table with Model/Config and metrics
- write to outputs/<sleeve>_h1.tex

Usage:
    python scripts/make_tables_sleeves_tex.py
"""
from pathlib import Path
import math
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CSV  = ROOT / "outputs" / "sleeves_h1.csv"
OUTD = ROOT / "outputs"
OUTD.mkdir(parents=True, exist_ok=True)

SLEEVES = ["equities", "rates_credit", "commod_alt"]
MODEL_NAME = {
    "pointwise": "Point-wise",
    "patch":     "PatchTST",
    "ivar":      "iTransformer",
    "cross":     "Cross-Lite",
}

def cfg_str(row):
    p = row.get("patch_len", None)
    s = row.get("stride", None)
    # base encoders (pointwise/ivar) have no patches
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return "base"
    if s is None or (isinstance(s, float) and math.isnan(s)):
        return f"$p={int(p)}$"
    return f"$p={int(p)},\\ s={int(s)}$"

def pick_best(g: pd.DataFrame) -> pd.Series:
    if g["test_mse"].notna().any():
        g = g.sort_values("test_mse", kind="mergesort")
    else:
        g = g.sort_values("val_mse", kind="mergesort")
    return g.iloc[0]

def build_table(df: pd.DataFrame, sleeve: str, tex_path: Path):
    d = df[df["sleeve"] == sleeve].copy()
    if d.empty:
        tex_path.write_text("% (no rows for this sleeve)\n")
        print(f"[warn] no rows for sleeve={sleeve}; wrote stub {tex_path}")
        return

    rows = (d.groupby("encoder", as_index=False)
              .apply(pick_best)
              .reset_index(drop=True))

    rows["Model"]  = rows["encoder"].map(MODEL_NAME).fillna(rows["encoder"])
    rows["Config"] = rows.apply(cfg_str, axis=1)

    # order & rename columns
    rows = rows[["Model","Config","val_mse","test_mse","test_mae","sharpe","turnover"]]
    rows = rows.rename(columns={
        "val_mse":  "Val MSE",
        "test_mse": "Test MSE",
        "test_mae": "Test MAE",
        "sharpe":   "Sharpe@10bps",
        "turnover": "Turnover",
    })

    # nice numeric formatting (keep None as em-dash)
    def fmt_num(x, prec=6):
        if pd.isna(x):
            return "—"
        # short formats for readability
        if isinstance(x, float):
            return f"{x:.6f}" if abs(x) < 1 else f"{x:.3f}"
        return str(x)

    for col in ["Val MSE","Test MSE","Test MAE","Sharpe@10bps","Turnover"]:
        rows[col] = rows[col].map(fmt_num)

    # consistent ordering: Point-wise, PatchTST, iTransformer, Cross-Lite
    order = ["Point-wise", "PatchTST", "iTransformer", "Cross-Lite"]
    rows["__ord"] = rows["Model"].map({m:i for i,m in enumerate(order)})
    rows = rows.sort_values(["__ord","Config"]).drop(columns="__ord")

    caption = {
        "equities":     "Equities sleeve (SPY, QQQ, IWM).",
        "rates_credit": "Rates \\& Credit sleeve (TLT, IEF, LQD, HYG).",
        "commod_alt":   "Commodities \\& Alternatives sleeve (GLD, DBC, VNQ, EFA, EEM).",
    }[sleeve]
    label = f"tab:{sleeve}_h1"

    latex = rows.to_latex(
        index=False,
        escape=False,                       # allow math in Config ($...$)
        column_format="l l r r r r r",      # two left, five right
        caption=caption,
        label=label
    )
    tex_path.write_text(latex)
    print(f"[ok] wrote {tex_path}")

def main():
    if not CSV.exists():
        raise SystemExit(f"CSV not found: {CSV}")
    df = pd.read_csv(CSV)
    for sleeve in SLEEVES:
        build_table(df, sleeve, OUTD / f"{sleeve}_h1.tex")

if __name__ == "__main__":
    main()
