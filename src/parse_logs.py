#!/usr/bin/env python3
import re, glob, os, json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True, parents=True)

# regexes
re_test = re.compile(r"^\[test\]\s+mse=([0-9.eE-]+)\s+mae=([0-9.eE-]+)")
re_save = re.compile(r"^\[save\]\s+checkpoint\s+->\s+(.+?_best_(.+?)\.pt)")
re_info = re.compile(r"^\[info\].*encoder:\s*(\w+)", re.IGNORECASE)  # not always present

def parse_chain_log(fp: Path):
    rows = []
    current = {"encoder": None, "tag": None}
    with fp.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            mtest = re_test.match(line)
            if mtest:
                mse, mae = float(mtest.group(1)), float(mtest.group(2))
                rows.append({
                    "log": fp.name,
                    "encoder": current.get("encoder"),
                    "tag": current.get("tag"),
                    "test_mse": mse,
                    "test_mae": mae
                })
                continue
            msave = re_save.match(line)
            if msave:
                # parse the save path to infer encoder + tag
                save_path = msave.group(1)
                tag = msave.group(2)
                # examples:
                # outputs/pointwise_L252_V12_d256_best_exp1.pt
                # outputs/patch_L252_V12_d256_best_p16s8.pt
                # outputs/cross_L252_V12_d256_best_cross_p16.pt
                base = os.path.basename(save_path)
                enc = None
                if base.startswith("pointwise_"):
                    enc = "pointwise"
                elif base.startswith("patch_"):
                    enc = "patch"
                elif base.startswith("ivar_"):
                    enc = "ivar"
                elif base.startswith("cross_"):
                    enc = "cross"
                current["encoder"] = enc
                current["tag"] = tag
                continue
            minfo = re_info.match(line)
            if minfo:
                current["encoder"] = minfo.group(1).lower()
    return rows

def main():
    logs = sorted(glob.glob(str(OUT / "*chain_*.log")))
    all_rows = []
    for fp in logs:
        rows = parse_chain_log(Path(fp))
        all_rows.extend(rows)

    if not all_rows:
        print("[warn] no results parsed from logs.")
        return

    df = pd.DataFrame(all_rows)
    # make a friendly label
    def labeler(r):
        enc = r["encoder"] or "unknown"
        tag = r["tag"] or ""
        if tag:
            return f"{enc}:{tag}"
        return enc
    df["label"] = df.apply(labeler, axis=1)

    # Save CSV
    csv_path = OUT / "results_from_logs.csv"
    df.to_csv(csv_path, index=False)
    print(f"[ok] wrote {csv_path} with {len(df)} rows")

    # Bar chart: Test MSE (one bar per label), sorted
    df_sorted = df.sort_values("test_mse")
    plt.figure(figsize=(8, 4))
    plt.bar(df_sorted["label"], df_sorted["test_mse"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Test MSE")
    plt.title("Overview: Test MSE by model/config")
    plt.tight_layout()
    fig_path = FIG / "overview_results.pdf"
    plt.savefig(fig_path, bbox_inches="tight")
    print(f"[ok] wrote {fig_path}")

if __name__ == "__main__":
    main()
