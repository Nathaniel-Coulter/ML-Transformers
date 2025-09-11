#!/usr/bin/env python3
import re
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "outputs" / "logs"
OUT  = ROOT / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

TAG_RE = re.compile(r"L(?P<L>\d+)_H(?P<H>\d+)_(?P<enc>\w+)_\d+")

# Be liberal about log phrasing / capitalization / separators
PAT = {
    "train_mse": re.compile(r"(?:train[_\s-]*mse)\s*[:=]\s*([0-9.eE+-]+)", re.I),
    "val_mse":   re.compile(r"(?:val(?:idation)?[_\s-]*mse)\s*[:=]\s*([0-9.eE+-]+)", re.I),
    "test_mse":  re.compile(r"(?:test[_\s-]*mse)\s*[:=]\s*([0-9.eE+-]+)", re.I),
    "test_mae":  re.compile(r"(?:test[_\s-]*mae)\s*[:=]\s*([0-9.eE+-]+)", re.I),
    "sharpe":    re.compile(r"(?:sharpe)\s*[:=]\s*([0-9.eE+-]+)", re.I),
    "turnover":  re.compile(r"(?:turnover)\s*[:=]\s*([0-9.eE+-]+)", re.I),
    "epochs":    re.compile(r"(?:epochs?)\s*[:=]\s*(\d+)", re.I),
    "elapsed_s": re.compile(r"(?:elapsed|time)\s*[:=]\s*([0-9.eE+-]+)\s*(?:s|sec|seconds)?", re.I),
    "patch_len": re.compile(r"(?:patch[_-]?len|p)\s*[:=]\s*(\d+)", re.I),
    "stride":    re.compile(r"(?:stride|s)\s*[:=]\s*(\d+)", re.I),
}

def last_match_float(p, text):
    m = None
    for m in p.finditer(text):
        pass
    return float(m.group(1)) if m else None

def last_match_int(p, text):
    m = None
    for m in p.finditer(text):
        pass
    return int(m.group(1)) if m else None

rows = []
for log in sorted(LOGS.glob("L*_H*_*_*.log")):
    tag = log.stem
    tm = TAG_RE.match(tag)
    if not tm:
        continue
    L = int(tm.group("L"))
    H = int(tm.group("H"))
    enc = tm.group("enc")

    text = log.read_text(errors="ignore")

    row = {
        "tag": tag, "L": L, "H": H, "encoder": enc,
        "train_mse": last_match_float(PAT["train_mse"], text),
        "val_mse":   last_match_float(PAT["val_mse"],   text),
        "test_mse":  last_match_float(PAT["test_mse"],  text),
        "test_mae":  last_match_float(PAT["test_mae"],  text),
        "sharpe":    last_match_float(PAT["sharpe"],    text),
        "turnover":  last_match_float(PAT["turnover"],  text),
        "epochs":    last_match_int(PAT["epochs"],      text),
        "elapsed_s": last_match_float(PAT["elapsed_s"], text),
        "patch_len": last_match_int(PAT["patch_len"],   text),
        "stride":    last_match_int(PAT["stride"],      text),
    }

    # Fallback metric for plotting if test_mse is missing
    row["metric_for_plot"] = row["test_mse"] if row["test_mse"] is not None else row["val_mse"]

    rows.append(row)

df = pd.DataFrame(rows).sort_values(["L","H","encoder","patch_len","stride"], na_position="last")
csv_path = OUT / "horizons_v2.csv"
df.to_csv(csv_path, index=False)
print(f"[ok] wrote {csv_path} with {len(df)} rows")

# Quick per-(L,H,encoder) median on the chosen metric
with pd.option_context("display.max_rows", None):
    print(df.groupby(["L","H","encoder"])["metric_for_plot"].median())
