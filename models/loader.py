#!/usr/bin/env python3
import argparse, json, sys
from pathlib import Path
from typing import Tuple, Dict, Any, Optional, List

import numpy as np
import pandas as pd

# Optional torch imports (datasets work even if torch not installed until training time)
try:
    import torch
    from torch.utils.data import Dataset
except Exception:
    torch = None
    Dataset = object  # type: ignore

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "data"
OUT = ROOT / "outputs"

# ---------------------------
# IO helpers (kept from your version, lightly hardened)
# ---------------------------

def _read_csv_robust(fp: Path) -> pd.DataFrame:
    """
    Robust CSV reader:
      - If the first row starts with 'Ticker', skip it (yfinance sometimes writes a label row)
      - Otherwise read normally.
    """
    with fp.open("r", encoding="utf-8") as f:
        first = f.readline().strip()
    if first.startswith("Ticker"):
        df = pd.read_csv(fp, skiprows=1)
    else:
        df = pd.read_csv(fp)
    return df

def load_from_csv(ticker: str) -> pd.DataFrame:
    """
    Load a single-ticker CSV from data/ with columns: Date, Adj Close, Volume (others ignored).
    Returns a DataFrame indexed by Date with columns [Adj Close, Volume?].
    """
    fp = DATA / f"{ticker}.csv"
    if not fp.exists():
        raise FileNotFoundError(f"Missing CSV for {ticker} at {fp}. Provide CSVs or run --fetch.")

    df = _read_csv_robust(fp)

    # Normalize date/index
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce", utc=True).dt.tz_localize(None)
        df = df.dropna(subset=["Date"]).sort_values("Date").set_index("Date")
    else:
        first_col = df.columns[0]
        df[first_col] = pd.to_datetime(df[first_col], errors="coerce", utc=True).dt.tz_localize(None)
        df = df.dropna(subset=[first_col]).sort_values(first_col).set_index(first_col)
        df.index.name = "Date"

    # Drop stray columns
    for junk in ["Ticker", "Symbols"]:
        if junk in df.columns:
            df = df.drop(columns=[junk])

    # Keep only needed columns
    cols_keep = [c for c in ["Adj Close", "Volume"] if c in df.columns]
    if "Adj Close" not in cols_keep:
        raise ValueError(f"{fp} must include an 'Adj Close' column; got {df.columns.tolist()}")
    return df[cols_keep]

def fetch_with_yfinance(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Optional helper that requires internet + yfinance."""
    import yfinance as yf  # type: ignore
    df = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
    # Ensure columns exist
    if "Adj Close" not in df.columns:
        # Auto-adjust as fallback if adj close missing
        df["Adj Close"] = df["Close"] * (df["Adj Close"] / df["Close"] if "Adj Close" in df.columns else 1.0)
    df = df.rename(columns={"Adj Close": "Adj Close", "Volume": "Volume"})
    take = ["Adj Close"] + (["Volume"] if "Volume" in df.columns else [])
    df = df[take]
    df.index.name = "Date"
    return df

# ---------------------------
# Panel building and returns
# ---------------------------

def build_panel_from_tickers(tickers: List[str], fetch: bool, start: str, end: str) -> pd.DataFrame:
    """
    Returns a MultiIndex column DataFrame: columns = (ticker, field), index = Date.
    """
    DATA.mkdir(exist_ok=True, parents=True)
    OUT.mkdir(exist_ok=True, parents=True)

    panel: Dict[str, pd.DataFrame] = {}
    for t in tickers:
        if fetch:
            try:
                df = fetch_with_yfinance(t, start, end)
                df.reset_index().to_csv(DATA / f"{t}.csv", index=False)
                print(f"[fetch] saved {t} -> {DATA/f'{t}.csv'}")
            except Exception as e:
                print(f"[warn] fetch failed for {t}: {e}")
        try:
            df = load_from_csv(t)
            panel[t] = df
        except Exception as e:
            print(f"[warn] skipping {t}: {e}")

    if not panel:
        raise RuntimeError("[error] No tickers loaded. Provide CSVs in data/ or use --fetch.")

    all_df = pd.concat(panel, axis=1)  # {(ticker)->cols}
    all_df.sort_index(inplace=True)

    out_path = OUT / "prices.parquet"
    all_df.to_parquet(out_path)
    print(f"[ok] wrote {out_path} with shape {all_df.shape}")
    return all_df

def to_log_returns(price_panel: pd.DataFrame) -> pd.DataFrame:
    """
    Convert (ticker, 'Adj Close') price panel to log returns per ticker.
    Output columns: (ticker, 'ret')
    Robust to stray strings / bad types / non-positive prices.
    """
    # 1) grab only Adj Close into a 2D wide df [Date x tickers]
    adj = price_panel.xs('Adj Close', axis=1, level=1, drop_level=False)
    adj_wide = pd.concat({t: adj[(t, 'Adj Close')] for t, _ in adj.columns}, axis=1)

    # 2) coerce to numeric and clean
    adj_wide = adj_wide.apply(pd.to_numeric, errors='coerce')

    # drop non-positive (cannot log) → set NaN first
    adj_wide = adj_wide.mask(adj_wide <= 0, np.nan)

    # forward-fill within each ticker to handle sporadic NaNs, then drop remaining NaNs
    adj_wide = adj_wide.ffill().dropna(how='any')

    # 3) compute log returns
    logp = np.log(adj_wide)
    ret_wide = logp.diff().dropna()

    # 4) return as MultiIndex columns (ticker, 'ret')
    ret_wide.columns = pd.MultiIndex.from_tuples([(t, 'ret') for t in ret_wide.columns],
                                                 names=price_panel.columns.names)
    return ret_wide

# ---------------------------
# Windowing and splits
# ---------------------------

def time_series_split(df: pd.DataFrame, train_frac=0.8, val_frac=0.1) -> Tuple[pd.DatetimeIndex, pd.DatetimeIndex, pd.DatetimeIndex]:
    """
    Deterministic chronological split indices.
    """
    n = len(df.index)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    n_test = n - n_train - n_val
    if n_test <= 0:
        raise ValueError("Not enough data for test split. Adjust fractions.")
    idx_train = df.index[:n_train]
    idx_val = df.index[n_train:n_train+n_val]
    idx_test = df.index[n_train+n_val:]
    return idx_train, idx_val, idx_test

def windowize_returns(
    ret_panel: pd.DataFrame,
    lookback: int = 252,
    horizon: int = 1,
    standardize: bool = True,
    fit_range: Optional[pd.DatetimeIndex] = None
) -> Tuple[np.ndarray, np.ndarray, List[str], pd.DatetimeIndex]:
    """
    Build X, y from log-returns panel.
    - X: [N, L, V] with L=lookback, V=#tickers
    - y: [N, V] = next-period (or next-horizon) returns per asset
    - Standardize per-asset using train stats only (if standardize=True)
    Returns: (X, y, tickers, target_dates)
    """
    # Expect columns: (ticker, 'ret')
    # Reindex to single wide df: columns = tickers
    tickers = sorted({t for t, inner in ret_panel.columns})
    wide = pd.concat({t: ret_panel[(t, 'ret')] for t in tickers}, axis=1)
    wide = wide.dropna(how='any')

    # Train mean/std for standardization
    if standardize:
        if fit_range is None:
            raise ValueError("fit_range (train index) required for standardization to avoid leakage.")
        train_wide = wide.loc[wide.index.intersection(fit_range)]
        mu = train_wide.mean(axis=0)
        sd = train_wide.std(axis=0).replace(0.0, 1.0)
        wide_z = (wide - mu) / sd
    else:
        wide_z = wide

    # Build rolling windows
    L = lookback
    H = horizon
    dates = wide_z.index
    N = len(dates) - L - H + 1
    if N <= 0:
        raise ValueError(f"Not enough rows ({len(dates)}) for lookback={L} and horizon={H}")

    X = np.zeros((N, L, len(tickers)), dtype=np.float32)
    y = np.zeros((N, len(tickers)), dtype=np.float32)
    tgt_dates = []
    for i in range(N):
        start = i
        end = i + L
        tgt = end + H - 1
        X[i] = wide_z.iloc[start:end, :].values
        # Use *raw* (unstandardized) returns for y to keep target in native units
        y[i] = wide.iloc[tgt, :].values
        tgt_dates.append(dates[tgt])

    tgt_index = pd.DatetimeIndex(tgt_dates, name="TargetDate")
    return X, y, tickers, tgt_index

# ---------------------------
# PyTorch Dataset views (tokenization-aware)
# ---------------------------

class PointwiseDataset(Dataset):
    """
    Tokens = time steps. Expects X: [N, L, V], y: [N, V]
    """
    def __init__(self, X: np.ndarray, y: np.ndarray):
        if torch is None:
            raise ImportError("PyTorch is required for PointwiseDataset.")
        self.X = torch.from_numpy(X)  # [N, L, V]
        self.y = torch.from_numpy(y)  # [N, V]

    def __len__(self): return self.X.shape[0]
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class PatchDataset(Dataset):
    """
    PatchTST view: returns [V, P, P_len] per sample.
    """
    def __init__(self, X: np.ndarray, y: np.ndarray, patch_len: int = 16, stride: int = 8):
        if torch is None:
            raise ImportError("PyTorch is required for PatchDataset.")
        self.X = torch.from_numpy(X)  # [N, L, V]
        self.y = torch.from_numpy(y)  # [N, V]
        self.patch_len = patch_len
        self.stride = stride

    def __len__(self): return self.X.shape[0]

    def _make_patches(self, seq_1d: torch.Tensor) -> torch.Tensor:
        # seq_1d: [L]
        L = seq_1d.shape[0]
        idxs = list(range(0, max(L - self.patch_len, 0) + 1, self.stride))
        patches = [seq_1d[i:i+self.patch_len] for i in idxs if i + self.patch_len <= L]
        if len(patches) == 0:
            pad = torch.zeros(self.patch_len - L, dtype=seq_1d.dtype, device=seq_1d.device)
            patches = [torch.cat([pad, seq_1d], dim=0)]
        return torch.stack(patches, dim=0)  # [P, P_len]

    def __getitem__(self, idx):
        x = self.X[idx]  # [L, V]
        V = x.shape[1]
        per_var = []
        for v in range(V):
            per_var.append(self._make_patches(x[:, v]))
        x_out = torch.stack(per_var, dim=0)  # [V, P, P_len]
        return x_out, self.y[idx]

class VarTokenDataset(Dataset):
    """
    iTransformer view: tokens are variates. Returns [V, L] per sample.
    """
    def __init__(self, X: np.ndarray, y: np.ndarray):
        if torch is None:
            raise ImportError("PyTorch is required for VarTokenDataset.")
        self.X = torch.from_numpy(X)  # [N, L, V]
        self.y = torch.from_numpy(y)  # [N, V]

    def __len__(self): return self.X.shape[0]

    def __getitem__(self, idx):
        x = self.X[idx].transpose(0, 1)  # [V, L]
        return x, self.y[idx]

# ---------------------------
# CLI: build parquet, then (optionally) tensors for training
# ---------------------------

def build_and_optionally_make_tensors(
    cfg_path: Path,
    fetch: bool,
    make_tensors: bool,
    lookback: int,
    horizon: int,
    train_frac: float,
    val_frac: float,
    standardize: bool
) -> Dict[str, Any]:
    cfg = json.loads(Path(cfg_path).read_text())
    tickers = cfg["tickers"]
    start, end = cfg["start_date"], cfg["end_date"]

    price_panel = build_panel_from_tickers(tickers, fetch=fetch, start=start, end=end)

    if not make_tensors:
        return {"parquet": str(OUT / "prices.parquet"), "made_tensors": False}

    # Returns panel
    ret_panel = to_log_returns(price_panel)

    # Splits (on returns index)
    idx_train, idx_val, idx_test = time_series_split(ret_panel, train_frac=train_frac, val_frac=val_frac)

    # Build tensors
    X_all, y_all, tickers_sorted, tgt_idx = windowize_returns(
        ret_panel, lookback=lookback, horizon=horizon, standardize=standardize, fit_range=idx_train
    )

    # Map windows into splits by target date
    def mask_for(idx_slice: pd.DatetimeIndex) -> np.ndarray:
        return tgt_idx.isin(idx_slice)

    m_train = mask_for(idx_train)
    m_val   = mask_for(idx_val)
    m_test  = mask_for(idx_test)

    def take(m: np.ndarray):
        return X_all[m], y_all[m], tgt_idx[m]

    X_train, y_train, d_train = take(m_train)
    X_val,   y_val,   d_val   = take(m_val)
    X_test,  y_test,  d_test  = take(m_test)

    # Save tensors
    npz_path = OUT / f"tensors_L{lookback}_H{horizon}.npz"
    np.savez_compressed(
        npz_path,
        X_train=X_train, y_train=y_train, dates_train=d_train.astype("datetime64[ns]"),
        X_val=X_val, y_val=y_val, dates_val=d_val.astype("datetime64[ns]"),
        X_test=X_test, y_test=y_test, dates_test=d_test.astype("datetime64[ns]"),
        tickers=np.array(tickers_sorted)
    )
    print(f"[ok] wrote {npz_path} | X_train {X_train.shape} X_val {X_val.shape} X_test {X_test.shape}")

    return {
        "parquet": str(OUT / "prices.parquet"),
        "npz": str(npz_path),
        "shapes": {
            "X_train": X_train.shape, "X_val": X_val.shape, "X_test": X_test.shape,
            "y_train": y_train.shape, "y_val": y_val.shape, "y_test": y_test.shape
        },
        "tickers": tickers_sorted
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true", help="Fetch data via yfinance into data/")
    ap.add_argument("--config", type=str, default=str(ROOT / "config.json"))
    ap.add_argument("--make-tensors", action="store_true", help="Also build windowed tensors and save to outputs/")
    ap.add_argument("--lookback", type=int, default=252, help="Lookback window length L")
    ap.add_argument("--horizon", type=int, default=1, help="Prediction horizon (steps ahead)")
    ap.add_argument("--train-frac", type=float, default=0.8, help="Train fraction (chronological)")
    ap.add_argument("--val-frac", type=float, default=0.1, help="Val fraction (chronological)")
    ap.add_argument("--no-standardize", action="store_true", help="Disable per-asset standardization (default on)")
    args = ap.parse_args()

    OUT.mkdir(exist_ok=True, parents=True)

    try:
        res = build_and_optionally_make_tensors(
            cfg_path=Path(args.config),
            fetch=args.fetch,
            make_tensors=args.make_tensors,
            lookback=args.lookback,
            horizon=args.horizon,
            train_frac=args.train_frac,
            val_frac=args.val_frac,
            standardize=not args.no_standardize
        )
    except Exception as e:
        print(f"[error] {e}")
        sys.exit(1)

    # Friendly summary
    print(json.dumps(res, indent=2, default=str))

if __name__ == "__main__":
    main()
