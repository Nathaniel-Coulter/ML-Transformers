#!/usr/bin/env python3
import argparse, os
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "synth"
OUT.mkdir(parents=True, exist_ok=True)

def gen_series(T, gamma, alpha, seed=0, sigma=1.0):
    rng = np.random.default_rng(seed)
    e1 = rng.normal(0, sigma, size=T)
    e2 = rng.normal(0, sigma, size=T)
    x1 = np.zeros(T, dtype=np.float32)
    x2 = np.zeros(T, dtype=np.float32)
    for t in range(1, T):
        x1[t] = gamma * x1[t-1] + e1[t]
        x2[t] = gamma * x2[t-1] + alpha * x1[t-1] + e2[t]
    return np.vstack([x1, x2]).T  # [T,2]

def windowize(ret_wide, L, H, standardize=True, fit_idx=None):
    # ret_wide: pd.DataFrame index=Date, cols=['X1','X2']
    if standardize:
        if fit_idx is None:
            raise ValueError("fit_idx required for standardization")
        mu = ret_wide.loc[fit_idx].mean(axis=0)
        sd = ret_wide.loc[fit_idx].std(axis=0).replace(0.0, 1.0)
        z = (ret_wide - mu) / sd
    else:
        z = ret_wide
    dates = ret_wide.index
    N = len(dates) - L - H + 1
    X = np.zeros((N, L, 2), dtype=np.float32)
    y = np.zeros((N, 2), dtype=np.float32)
    tgt_dates = []
    for i in range(N):
        s, e = i, i+L
        t = e+H-1
        X[i] = z.iloc[s:e, :].values
        y[i] = ret_wide.iloc[t, :].values  # raw target
        tgt_dates.append(dates[t])
    return X, y, np.array(tgt_dates, dtype="datetime64[ns]")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gamma", type=float, required=True)
    ap.add_argument("--alpha", type=float, required=True)
    ap.add_argument("--T", type=int, default=5000)
    ap.add_argument("--L", type=int, default=252)
    ap.add_argument("--H", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # Generate synthetic returns
    R = gen_series(args.T, args.gamma, args.alpha, seed=args.seed)  # [T,2]
    idx = pd.date_range("2000-01-01", periods=args.T, freq="B")
    df = pd.DataFrame(R, index=idx, columns=["X1","X2"])

    # Splits on the returns index
    n = len(df)
    ntr = int(0.8 * n)
    nva = int(0.1 * n)
    itrain = df.index[:ntr]
    ival = df.index[ntr:ntr+nva]
    itest = df.index[ntr+nva:]

    # Windowize (z-norm inputs using train stats)
    X_all, y_all, tgt_all = windowize(df, args.L, args.H, standardize=True, fit_idx=itrain)

    # Map windows to splits by target date
    m_tr = np.isin(tgt_all, itrain)
    m_va = np.isin(tgt_all, ival)
    m_te = np.isin(tgt_all, itest)

    def take(m):
        return X_all[m], y_all[m], tgt_all[m]

    Xtr, ytr, dtr = take(m_tr)
    Xva, yva, dva = take(m_va)
    Xte, yte, dte = take(m_te)

    tag = f"synth_a{args.alpha}_g{args.gamma}_L{args.L}_H{args.H}"
    npz = OUT / f"{tag}.npz"
    tickers = np.array(["X1","X2"])
    np.savez_compressed(
        npz,
        X_train=Xtr, y_train=ytr, dates_train=dtr,
        X_val=Xva, y_val=yva, dates_val=dva,
        X_test=Xte, y_test=yte, dates_test=dte,
        tickers=tickers
    )
    print(f"[ok] wrote {npz} | Xtr {Xtr.shape} Xva {Xva.shape} Xte {Xte.shape}")

if __name__ == "__main__":
    main()
