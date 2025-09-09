#!/usr/bin/env python3
import json, glob, os
from pathlib import Path
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True, parents=True)

def load_history(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)  # list of {epoch, train_mse, val_mse, val_mae}
    # Ensure sorted by epoch (just in case)
    data = sorted(data, key=lambda d: d.get("epoch", 0))
    epochs = [d["epoch"] for d in data]
    val_mse = [d["val_mse"] for d in data]
    return epochs, val_mse

def try_plot(label, pattern, ax, style_kwargs=None):
    files = sorted(glob.glob(str(OUT / pattern)))
    if not files:
        print(f"[skip] No files for {label} ({pattern})")
        return False
    # Pick the first match (these are per-config histories)
    epochs, val_mse = load_history(files[0])
    ax.plot(epochs, val_mse, label=label, **(style_kwargs or {}))
    return True

def plot_main_curves():
    fig, ax = plt.subplots(figsize=(6.0, 3.4))

    plotted = []
    plotted.append(try_plot("Pointwise",                "pointwise_history_*.json", ax))
    plotted.append(try_plot("PatchTST p8/s4",           "patch_history_p8s4.json", ax))
    plotted.append(try_plot("PatchTST p16/s8",          "patch_history_p16s8.json", ax))
    plotted.append(try_plot("PatchTST p32/s16",         "patch_history_p32s16.json", ax))
    plotted.append(try_plot("iTransformer",             "ivar_history_*.json", ax))
    plotted.append(try_plot("Cross-Lite p8",            "cross_history_cross_p8.json", ax))
    plotted.append(try_plot("Cross-Lite p16",           "cross_history_cross_p16.json", ax))

    if any(plotted):
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Validation MSE")
        ax.set_title("Validation MSE vs Epoch (Main Configs)")
        ax.grid(True, alpha=0.3, linestyle=":")
        ax.legend(loc="best", frameon=False)
        out = FIG / "val_curves_all.pdf"
        fig.tight_layout()
        fig.savefig(out, bbox_inches="tight")
        print(f"[ok] wrote {out}")
    else:
        print("[warn] No curves were plotted in main figure.")

def plot_patch_p16_seeds():
    # Gather all seed histories for PatchTST p16/s8
    seed_files = sorted(glob.glob(str(OUT / "patch_history_p16s8_seed*.json")))
    if not seed_files:
        print("[skip] No seed files found for PatchTST p16/s8.")
        return

    # Load all
    seed_curves = []
    for fp in seed_files:
        epochs, val_mse = load_history(fp)
        seed = os.path.basename(fp).split("seed")[-1].split(".json")[0]
        seed_curves.append((seed, epochs, val_mse))

    # Ensure all have the same epoch grid (they should—3 epochs)
    epoch_grid = seed_curves[0][1]

    # Compute mean/std across seeds at each epoch
    import numpy as np
    vals = np.array([c[2] for c in seed_curves])  # [n_seeds, n_epochs]
    mean = vals.mean(axis=0)
    std  = vals.std(axis=0, ddof=0)

    # Plot
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    # individual seeds (thin lines)
    for seed, epochs, val_mse in seed_curves:
        ax.plot(epochs, val_mse, alpha=0.5, linewidth=1.0, label=f"seed {seed}")

    # mean ± std band
    ax.plot(epoch_grid, mean, linewidth=2.0, label="mean")
    ax.fill_between(epoch_grid, mean-std, mean+std, alpha=0.15, label="±1 std")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation MSE")
    ax.set_title("PatchTST (p16/s8): Seeds (mean ± std)")
    ax.grid(True, alpha=0.3, linestyle=":")
    ax.legend(loc="best", frameon=False, ncol=2)

    out = FIG / "val_curves_patch_p16s8_seeds.pdf"
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    print(f"[ok] wrote {out}")

if __name__ == "__main__":
    plot_main_curves()
    plot_patch_p16_seeds()
