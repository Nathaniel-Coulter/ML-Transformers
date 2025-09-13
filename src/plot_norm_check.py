#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "outputs" / "norm_check_summary.csv"
FIG = ROOT / "figs" / "norm_check_panel.pdf"
FIG.parent.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(CSV)
# Keep L=252, H=1 to match the runs (robust if you expand later)
df = df[(df["L"]==252)&(df["H"]==1)]

# Compute Δ (Z − NoZ) per (sleeve, encoder)
pivot_mse = df.pivot_table(index=["sleeve","encoder"], columns="norm", values="test_mse")
pivot_mae = df.pivot_table(index=["sleeve","encoder"], columns="norm", values="test_mae")
pivot_shp = df.pivot_table(index=["sleeve","encoder"], columns="norm", values="sharpe")
pivot_to  = df.pivot_table(index=["sleeve","encoder"], columns="norm", values="turnover")

d_mse = (pivot_mse["Z"] - pivot_mse["NoZ"]).reset_index(name="delta_mse")
d_mae = (pivot_mae["Z"] - pivot_mae["NoZ"]).reset_index(name="delta_mae")
d_shp = (pivot_shp["Z"] - pivot_shp["NoZ"]).reset_index(name="delta_sharpe")
d_to  = (pivot_to["Z"]  - pivot_to["NoZ"]).reset_index(name="delta_turnover")

m = d_mse.merge(d_mae, on=["sleeve","encoder"]).merge(d_shp, on=["sleeve","encoder"]).merge(d_to, on=["sleeve","encoder"])

enc_order = ["pointwise","patch","ivar","cross"]
sleeve_order = ["equities","rates_credit","commod_alt"]
m["encoder"] = pd.Categorical(m["encoder"], categories=enc_order, ordered=True)
m["sleeve"]  = pd.Categorical(m["sleeve"],  categories=sleeve_order, ordered=True)
m = m.sort_values(["sleeve","encoder"])

# Plot
fig, axes = plt.subplots(2, 2, figsize=(10,6), constrained_layout=True)
ax = axes.ravel()

def _bar(ax, ycol, title, ylabel):
    X = range(len(m))
    ax.bar(X, m[ycol])
    ax.set_title(title)
    ax.set_xticks(X)
    ax.set_xticklabels([f"{s}\n{e}" for s,e in zip(m["sleeve"], m["encoder"])], rotation=45, ha="right")
    ax.axhline(0, linewidth=1)
    ax.set_ylabel(ylabel)

_bar(ax[0], "delta_mse",   "Δ Test MSE (Z − NoZ)", "Δ MSE")
_bar(ax[1], "delta_mae",   "Δ Test MAE (Z − NoZ)", "Δ MAE")
_bar(ax[2], "delta_sharpe","Δ Sharpe (Z − NoZ)",   "Δ Sharpe")
_bar(ax[3], "delta_turnover","Δ Turnover (Z − NoZ)","Δ Turnover")

fig.suptitle("Normalization Check: Effect of Z-Norm (Inputs Only) vs No-Z", fontsize=12)
fig.savefig(FIG, bbox_inches="tight")
print(f"[ok] wrote {FIG}")
