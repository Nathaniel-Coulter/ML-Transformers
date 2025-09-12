#!/usr/bin/env python3
import argparse, json
from pathlib import Path
from typing import Optional, Tuple, Dict
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = ROOT / "outputs"

# -----------------------
# Utility feature funcs
# -----------------------
def pct_change_log(s, n):
    return np.log(s / s.shift(n))

def realized_vol(returns, window=21):
    return returns.rolling(window).std() * np.sqrt(252)

# -----------------------
# Macro integration helpers
# -----------------------
def load_macro_features(path: str) -> pd.DataFrame:
    """
    Load weekly macro features (Level/Slope/Curvature) from CSV.
    Robust to variant column names like:
      - 'level', 'lvl', 'YC_Level_10Y'
      - 'slope', 'slp', 'YC_Slope_10Y_3M_or_2Y'
      - 'curvature', 'curv', 'YC_Curv_2*5Y-2Y-10Y'
    Returns a DataFrame indexed by DatetimeIndex with columns ['lvl','slope','curv'] as float.
    """
    df = pd.read_csv(path)

    # --- find date column robustly ---
    date_col = None
    for c in df.columns:
        cl = c.lower().strip()
        if cl in ("date", "dt", "timestamp"):
            date_col = c
            break
    if date_col is None:
        # If not found, try the first column if it parses to datetime
        try:
            test = pd.to_datetime(df.iloc[:, 0], errors="coerce")
            if test.notna().any():
                date_col = df.columns[0]
        except Exception:
            pass
    if date_col is None:
        raise ValueError("Macro CSV must contain a date column (e.g., 'Date').")

    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).set_index(date_col)

    # --- map headers to lvl/slope/curv by substring ---
    def pick(colnames, needles):
        # return the first column whose lowercase name contains ANY needle
        clmap = {c: c.lower() for c in colnames}
        for c, cl in clmap.items():
            if any(n in cl for n in needles):
                return c
        return None

    col_level = pick(df.columns, ["lvl", "level", "yc_level"])
    col_slope = pick(df.columns, ["slope", "slp", "yc_slope"])
    col_curv  = pick(df.columns, ["curv", "curve", "curvature", "yc_curv"])

    # Fallback: try exact names you reported
    if col_level is None and "YC_Level_10Y" in df.columns: col_level = "YC_Level_10Y"
    if col_slope is None and "YC_Slope_10Y_3M_or_2Y" in df.columns: col_slope = "YC_Slope_10Y_3M_or_2Y"
    if col_curv  is None and "YC_Curv_2*5Y-2Y-10Y" in df.columns: col_curv  = "YC_Curv_2*5Y-2Y-10Y"

    keep = []
    rename = {}
    if col_level is not None: keep.append(col_level); rename[col_level] = "lvl"
    if col_slope is not None: keep.append(col_slope); rename[col_slope] = "slope"
    if col_curv  is not None: keep.append(col_curv);  rename[col_curv]  = "curv"

    if len(keep) < 3:
        raise ValueError(f"Macro CSV must have lvl/slope/curv; found={keep}.")

    out = df[keep].rename(columns=rename).astype(float)
    return out

def merge_asof_macro(
    df_with_date: pd.DataFrame,
    macro_df: pd.DataFrame,
    date_col: str = "Date",
    forward_fill: bool = True,
    scale_in_train: bool = True,
    train_end: Optional[pd.Timestamp] = None,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    """
    As-of merge 3 macro columns onto df_with_date using date_col, then (optionally) z-score using train-only stats.
    df_with_date must be sorted by date_col and contain that column as datetime (not index).
    macro_df must be indexed by DatetimeIndex with columns ['lvl','slope','curv'].
    Returns (merged_df, stats) where stats has per-column mean/std used for scaling.
    """
    if date_col not in df_with_date.columns:
        raise ValueError(f"`{date_col}` must be a column in df_with_date.")

    pdf = df_with_date.copy()
    pdf[date_col] = pd.to_datetime(pdf[date_col])
    pdf = pdf.sort_values(date_col)

    m = macro_df.copy().sort_index().reset_index().rename(columns={"index": date_col})

    merged = pd.merge_asof(pdf, m, on=date_col, direction="backward")

    if forward_fill:
        merged[["lvl", "slope", "curv"]] = merged[["lvl", "slope", "curv"]].ffill()

    stats: Dict[str, Dict[str, float]] = {}
    if scale_in_train:
        if train_end is None:
            split_idx = int(0.7 * len(merged))
            train_mask = np.zeros(len(merged), dtype=bool)
            train_mask[:split_idx] = True
        else:
            train_end = pd.to_datetime(train_end)
            train_mask = merged[date_col] <= train_end

        for col in ["lvl", "slope", "curv"]:
            mu = merged.loc[train_mask, col].mean()
            sd = merged.loc[train_mask, col].std(ddof=0)
            if not np.isfinite(sd) or sd == 0:
                sd = 1.0
            merged[col] = (merged[col] - mu) / sd
            stats[col] = {"mean": float(mu), "std": float(sd)}

    return merged, stats

# -----------------------
# Main feature builder
# -----------------------
def build_features(cfg_path: Path):
    cfg = json.loads(Path(cfg_path).read_text())
    prices = pd.read_parquet(ROOT / "outputs" / "prices.parquet")

    # prices MultiIndex columns: (ticker, field) with fields like ["Adj Close","Volume"]
    px = prices.xs("Adj Close", level=1, axis=1)
    vol = prices.xs("Volume", level=1, axis=1) if ("Volume" in prices.columns.get_level_values(1)) else None

    # ---- HOTFIX: ensure numeric dtypes and clean rows ----
    px = px.apply(lambda s: pd.to_numeric(s, errors="coerce"))
    if vol is not None:
        vol = vol.apply(lambda s: pd.to_numeric(s, errors="coerce"))

    # drop dates where all prices are NaN (after coercion)
    px = px.dropna(how="all")
    # align volume to price index
    if vol is not None:
        vol = vol.reindex(px.index)

    rets = np.log(px / px.shift(1))
    feats: Dict[str, pd.DataFrame] = {}

    # Build per-asset feature frames
    for t in px.columns:
        df = pd.DataFrame(index=px.index)
        df["ret_1"] = rets[t]
        for n in [5, 21, 63, 126]:
            df[f"ret_{n}"] = pct_change_log(px[t], n)
        for w in [21, 63]:
            df[f"rv_{w}"] = realized_vol(rets[t], w)
        df["mom_252_21"] = pct_change_log(px[t], 252) - pct_change_log(px[t], 21)  # 12-1
        if vol is not None and t in vol.columns:
            df["vol_z"] = (vol[t] - vol[t].rolling(63).mean()) / (vol[t].rolling(63).std() + 1e-9)
        # carry a materialized Date column for macro merge
        df["Date"] = df.index
        feats[t] = df

    # Optional cross-asset correlations to SPY/QQQ/TLT
    base = {"SPY", "QQQ", "TLT"} & set(px.columns)
    for b in base:
        corr = rets.rolling(63).corr(rets[b]).rename(columns=lambda c: f"corr63_{b}_{c}")
        for t in px.columns:
            colname = f"corr63_{b}_{t}"
            if colname in corr.columns:
                feats[t][f"corr63_{b}"] = corr[colname]

    # === Section 5.5: Merge macro features (once per asset) ===
    macro_cols = ["lvl", "slope", "curv"]
    used_train_stats: Dict[str, Dict[str, float]] = {}
    if cfg.get("use_macro", False):
        macro_df = load_macro_features(cfg.get("macro_path", "figures/yield_features_weekly.csv"))
        train_end = cfg.get("train_end")  # e.g., "2020-12-31" or None

        for t in list(feats.keys()):
            df_t = feats[t].copy()

            # Ensure 'Date' is ONLY a column (not also the index name)
            if df_t.index.name == "Date":
                # If there's already a 'Date' column, keep it and drop index to RangeIndex
                if "Date" in df_t.columns:
                    df_t = df_t.reset_index(drop=True)
                else:
                    df_t = df_t.reset_index()  # makes index into a 'Date' column
            elif "Date" not in df_t.columns:
                # No 'Date' anywhere -> create it from the index
                df_t = df_t.reset_index().rename(columns={"index": "Date"})

            merged, stats = merge_asof_macro(
                df_t,
                macro_df,
                date_col="Date",
                forward_fill=True,
                scale_in_train=True,
                train_end=train_end,
            )

            # Restore original datetime index from the 'Date' column
            merged = merged.set_index("Date")
            merged.index.name = px.index.name  # usually 'Date'
            # Align to original index (in case of any asof edge effects)
            merged = merged.reindex(feats[t].index)

            feats[t] = merged
            used_train_stats = stats  # same stats across tickers since dates align

    # Combine to panel (MultiIndex columns: (ticker, feature))
    panel_raw = pd.concat({k: v for k, v in feats.items()}, axis=1)

    # Normalization:
    # - Apply expanding z-score to NON-macro columns per asset
    # - Leave macro columns as-is (already train-only z-scored)
    def expanding_z(x):
        mu = x.expanding().mean()
        sd = x.expanding().std().replace(0, np.nan)
        return (x - mu) / sd

    panel_norm_parts = []
    for t in px.columns:
        df_t = panel_raw[t].copy()
        if cfg.get("use_macro", False):
            non_macro = [c for c in df_t.columns if c not in macro_cols and c != "Date"]
            if non_macro:
                df_t[non_macro] = df_t[non_macro].transform(expanding_z)
            df_t = df_t.drop(columns=["Date"], errors="ignore")
            # macro cols (if present) are already scaled; left untouched
        else:
            non_macro = [c for c in df_t.columns if c != "Date"]
            if non_macro:
                df_t[non_macro] = df_t[non_macro].transform(expanding_z)
            df_t = df_t.drop(columns=["Date"], errors="ignore")
        panel_norm_parts.append((t, df_t))

    panel_norm = pd.concat({t: df for t, df in panel_norm_parts}, axis=1)

    out_path = Path(cfg["panel_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    panel_norm.to_parquet(out_path)
    print(f"[ok] wrote normalized feature panel -> {out_path}, shape={panel_norm.shape}")
    if cfg.get("use_macro", False):
        print("[macro] train-only z-score stats:", used_train_stats)
    return out_path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=str(ROOT / "config.json"))
    args = ap.parse_args()
    build_features(Path(args.config))

if __name__ == "__main__":
    main()
