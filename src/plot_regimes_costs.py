#!/usr/bin/env python3
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "outputs" / "regimes_costs.csv"
OUTD = ROOT / "figs"
OUTD.mkdir(parents=True, exist_ok=True)

# map from tag prefix to pretty encoder name (adjust to match your tags)
NAME_MAP = {
    "pointwise": "Point-wise",
    "patch":     "PatchTST",
    "cross":     "Cross-Lite",
    "ivar":      "iTransformer",
}
def enc_from_tag(tag: str) -> str:
    # tags look like: equities_L252_H1_patch_2  OR  rates_credit_L252_H1_pointwise_1
    for k,v in NAME_MAP.items():
        if f"_{k}_" in tag:
            return v
    # fallback
    return tag

def plot_one_regime(df, regime: str, fname: Path):
    d = df[df["regime"] == regime].copy()
    if d.empty: return
    d["encoder"] = d["tag"].map(enc_from_tag)
    encoders = ["Point-wise","PatchTST","Cross-Lite","iTransformer"]
    costs = [10.0,25.0,50.0]

    fig, axes = plt.subplots(2, 3, figsize=(12, 6), constrained_layout=True)
    metrics = [("sharpe","Sharpe@bps"), ("turnover","Turnover")]
    for r,(m,label) in enumerate(metrics):
        for c,co in enumerate(costs):
            ax = axes[r,c]
            dd = d[d["cost_bps"] == co]
            means = []
            for enc in encoders:
                vals = dd[dd["encoder"] == enc][m].values
                means.append(np.nanmean(vals) if len(vals) else np.nan)
            x = np.arange(len(encoders))
            ax.bar(x, means)
            ax.set_xticks(x)
            ax.set_xticklabels(encoders, rotation=30, ha="right")
            ax.set_title(f"{label} @ {int(co)}bps")
            if r == 0:
                ax.axhline(0, lw=0.8)
    fig.suptitle(f"{regime} VIX regime: Sharpe & Turnover vs Costs")
    fig.savefig(fname, bbox_inches="tight")
    print(f"[ok] wrote {fname}")

def main():
    df = pd.read_csv(CSV)
    for regime in ["Low","Mid","High"]:
        plot_one_regime(df, regime, OUTD / f"regimes_costs_{regime.lower()}.pdf")

if __name__ == "__main__":
    main()
