#!/usr/bin/env python3
# scripts/eval_regimes_costs.py
import json
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------
# Paths / constants
# --------------------
ROOT = Path(__file__).resolve().parents[1]
PRED_DIR = ROOT / "outputs" / "preds"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True, parents=True)

COSTS = [10.0, 25.0, 50.0]  # bps

# --------------------
# Robust VIX (or proxy) loader
# --------------------
def _read_csv_robust(fp: Path) -> pd.DataFrame:
    with fp.open("r", encoding="utf-8") as f:
        first = f.readline().strip()
    if first.startswith("Ticker"):
        return pd.read_csv(fp, skiprows=1)
    return pd.read_csv(fp)

def _normalize_date_index(df: pd.DataFrame) -> pd.DataFrame:
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce", utc=True).dt.tz_localize(None)
        df = df.dropna(subset=["Date"]).sort_values("Date").set_index("Date")
    elif not isinstance(df.index, pd.DatetimeIndex):
        first = df.columns[0]
        df[first] = pd.to_datetime(df[first], errors="coerce", utc=True).dt.tz_localize(None)
        df = df.dropna(subset=[first]).sort_values(first).set_index(first)
        df.index.name = "Date"
    return df

def _load_vix_local() -> pd.Series | None:
    """
    Try local files that might contain a VIX column.
    """
    candidates = [
        ROOT / "data" / "VIX.csv",
        ROOT / "data" / "^VIX.csv",
        ROOT / "figures" / "yield_features_weekly.csv",  # if you baked VIX there
    ]
    for p in candidates:
        if not p.exists():
            continue
        df = _read_csv_robust(p) if p.suffix.lower() == ".csv" else pd.read_parquet(p)
        df = _normalize_date_index(df)
        # scan for any column that looks like VIX (works with flat or MultiIndex)
        for col in df.columns:
            top = str(col[0]).upper() if isinstance(col, tuple) else str(col).upper()
            sub = str(col[1]).upper() if isinstance(col, tuple) and len(col) > 1 else ""
            if "VIX" in top or "VIX" in sub or str(col).upper() == "VIX":
                s = pd.to_numeric(df[col], errors="coerce").rename("VIX")
                s = s.replace([np.inf, -np.inf], np.nan).ffill().dropna()
                if not s.empty:
                    return s
    return None

def _fetch_vix_yfinance(start="1990-01-01", end=None) -> pd.Series | None:
    """
    Optional: only works if yfinance + internet are available.
    """
    try:
        import yfinance as yf  # type: ignore
    except Exception:
        return None
    try:
        df = yf.download("^VIX", start=start, end=end, auto_adjust=False, progress=False)
        if df is None or df.empty:
            return None
        df.index = pd.to_datetime(df.index, utc=True).tz_localize(None)
        col = "Adj Close" if "Adj Close" in df.columns else "Close"
        s = pd.to_numeric(df[col], errors="coerce").rename("VIX")
        return s.replace([np.inf, -np.inf], np.nan).ffill().dropna()
    except Exception:
        return None

def _spy_realized_vol_proxy() -> pd.Series:
    """
    Offline fallback: build a VIX-like proxy from SPY log-return vol.
    Uses outputs/prices.parquet if it contains SPY; otherwise data/SPY.csv.
    Annualized 21d rolling std * 100 to be on a VIX-like scale.
    """
    # 1) Try outputs/prices.parquet
    px = None
    pp = ROOT / "outputs" / "prices.parquet"
    if pp.exists():
        dfp = pd.read_parquet(pp)
        if ("SPY", "Adj Close") in dfp.columns:
            px = pd.to_numeric(dfp[("SPY", "Adj Close")], errors="coerce").dropna()
            px.index = pd.to_datetime(px.index, utc=True).tz_localize(None)

    # 2) Local CSV
    if px is None:
        for cand in [ROOT / "data" / "SPY.csv", ROOT / "data" / "^SPY.csv"]:
            if cand.exists():
                df = _read_csv_robust(cand)
                df = _normalize_date_index(df)
                found = None
                for col in df.columns:
                    name = str(col[0]).upper() if isinstance(col, tuple) else str(col).upper()
                    sub = str(col[1]).upper() if isinstance(col, tuple) and len(col) > 1 else ""
                    if "ADJ" in name or "ADJ" in sub or str(col).lower() == "adj close":
                        found = pd.to_numeric(df[col], errors="coerce").dropna()
                        break
                if found is not None:
                    px = found
                    break

    if px is None or px.empty:
        raise RuntimeError("Could not find SPY prices for realized-vol proxy. Add data/SPY.csv or include SPY in outputs/prices.parquet.")

    logret = np.log(px).diff()
    rv21 = logret.rolling(21).std() * np.sqrt(252.0) * 100.0
    return rv21.rename("VIX").ffill().dropna()

def load_vix() -> pd.Series:
    """
    1) local VIX file(s)  2) yfinance ^VIX  3) SPY realized-vol proxy
    Returns a Series named 'VIX' with DatetimeIndex.
    """
    s = _load_vix_local()
    if s is not None:
        return s
    s = _fetch_vix_yfinance()
    if s is not None:
        return s
    return _spy_realized_vol_proxy()

# --------------------
# Metrics / helpers
# --------------------
def allocator_weights(sig, mode="dollar_neutral", clip=3.0, eps=1e-12):
    # sig: (T, N)
    if mode == "long_only":
        s = np.maximum(sig, 0.0)
        denom = s.sum(axis=1, keepdims=True) + eps
        return s / denom
    # dollar-neutral
    mu = np.nanmean(sig, axis=1, keepdims=True)
    sd = np.nanstd(sig, axis=1, keepdims=True) + 1e-8
    z = (sig - mu) / sd
    z = np.clip(z, -clip, clip)
    denom = np.sum(np.abs(z), axis=1, keepdims=True) + eps
    w = z / denom
    w = w - w.mean(axis=1, keepdims=True)
    denom = np.sum(np.abs(w), axis=1, keepdims=True) + eps
    return w / denom

def sharpe_turnover(y_true, y_pred, costs_bps, mode="dollar_neutral"):
    w = allocator_weights(y_pred, mode=mode)         # (T, N)
    pnl_gross = (w * y_true).sum(axis=1)             # (T,)
    dw = np.diff(w, axis=0)                          # (T-1, N)
    turnover_t = 0.5 * np.abs(dw).sum(axis=1)
    turnover_t = np.concatenate([[0.0], turnover_t]) # (T,)
    pnl_net = pnl_gross - (costs_bps / 10_000.0) * turnover_t
    mu = float(np.mean(pnl_net))
    sd = float(np.std(pnl_net, ddof=1) + 1e-12)
    sharpe_ann = (mu / sd) * np.sqrt(252.0)
    return sharpe_ann, float(np.mean(turnover_t))

def mse_mae(y_true, y_pred):
    err = y_pred - y_true
    return float(np.mean(err**2)), float(np.mean(np.abs(err)))

def terciles(x):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return np.nan, np.nan
    q1 = np.nanpercentile(x, 33.333)
    q2 = np.nanpercentile(x, 66.667)
    return q1, q2

# --------------------
# Main
# --------------------
def main():
    vix = load_vix()  # Series, index=Date, name="VIX"

    rows = []
    for f in sorted(PRED_DIR.glob("*_y_true.npy")):
        tag = f.name.replace("_y_true.npy", "")
        y_true = np.load(f)
        y_pred = np.load(PRED_DIR / f"{tag}_y_pred.npy")

        dates_f = PRED_DIR / f"{tag}_dates.npy"
        if not dates_f.exists():
            print(f"[warn] missing dates for {tag}; skipping")
            continue

        dates_raw = np.load(dates_f, allow_pickle=True)
        # handle datetime64 or object-strings uniformly
        idx = pd.to_datetime(dates_raw, errors="coerce", utc=True).tz_localize(None)
        mask_ok = ~pd.isna(idx)
        y_true = y_true[mask_ok]
        y_pred = y_pred[mask_ok]
        idx = idx[mask_ok]

        if len(idx) == 0:
            print(f"[warn] empty aligned dates for {tag}; skipping")
            continue

        # Align VIX to these dates (ffill to handle non-trading days)
        v = vix.reindex(idx, method="ffill")
        v = v.astype(float)
        # compute terciles on this test window (drop NaNs)
        q1, q2 = terciles(v.values)
        if np.isnan(q1) or np.isnan(q2):
            print(f"[warn] could not compute terciles for {tag}; skipping")
            continue

        reg_labels = np.where(v.values <= q1, "Low", np.where(v.values <= q2, "Mid", "High"))

        # Evaluate per regime, per cost
        for regime in ("Low", "Mid", "High"):
            reg_mask = (reg_labels == regime)
            # Guard: need enough points
            if reg_mask.sum() < 10:
                continue

            yt = y_true[reg_mask]
            yp = y_pred[reg_mask]
            test_mse, test_mae = mse_mae(yt, yp)

            for c in COSTS:
                sharpe, to = sharpe_turnover(yt, yp, costs_bps=c, mode="dollar_neutral")
                rows.append({
                    "tag": tag,
                    "regime": regime,
                    "cost_bps": c,
                    "test_mse": test_mse,
                    "test_mae": test_mae,
                    "sharpe": sharpe,
                    "turnover": to,
                    "n_obs": int(reg_mask.sum()),
                })

    df = pd.DataFrame(rows)
    out = OUT / "regimes_costs.csv"
    df.to_csv(out, index=False)
    print(f"[ok] wrote {out} with {len(df)} rows")

if __name__ == "__main__":
    main()
