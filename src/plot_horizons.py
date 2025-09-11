#!/usr/bin/env python3
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
CSV  = ROOT / "outputs" / "horizons_v2.csv"
FIGS = ROOT / "figs"
FIGS.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(CSV)

# Use test_mse when available; otherwise val_mse
if "metric_for_plot" not in df.columns:
    df["metric_for_plot"] = df["test_mse"].where(~df["test_mse"].isna(), df["val_mse"])

ENCODER_NAMES = {
    "pointwise": "Point-wise",
    "patch":     "PatchTST",
    "ivar":      "iTransformer",
    "cross":     "Cross-Lite",
}

for L in sorted(df["L"].dropna().unique()):
    dL = df[df["L"] == L].copy()

    def pick_best(g: pd.DataFrame) -> pd.DataFrame:
        # g is a sub-DF for one (H, encoder)
        H_val = g["H"].iloc[0]
        enc_key = g["encoder"].iloc[0]
        if enc_key in ("patch", "cross"):
            return g.nsmallest(1, "metric_for_plot")
        return g.tail(1)  # single-config encoders

    # Important: reset_index(drop=True) to avoid H being both index & column
    picked = (
        dL.groupby(["H", "encoder"], group_keys=False)
          .apply(pick_best)
          .reset_index(drop=True)
    )

    # Drop groups lacking metrics
    picked = picked.dropna(subset=["metric_for_plot"])

    fig, ax = plt.subplots(figsize=(7, 4.2))
    for enc_key, enc_name in ENCODER_NAMES.items():
        dE = picked[picked["encoder"] == enc_key].sort_values("H").copy()
        if dE.empty:
            continue
        ax.plot(dE["H"], dE["metric_for_plot"], marker="o", label=enc_name)

    ax.set_title(f"Error vs Horizon (L={L})")
    ax.set_xlabel("Horizon H")
    ax.set_ylabel("Error (Test MSE or Val MSE)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    out = FIGS / f"horizons_mse_L{L}.pdf"
    fig.tight_layout()
    fig.savefig(out)
    print(f"[ok] wrote {out}")
