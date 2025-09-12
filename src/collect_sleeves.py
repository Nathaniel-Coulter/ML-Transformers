#!/usr/bin/env python3
"""
Collect metrics from outputs/logs_sleeves/*.log and write outputs/sleeves_h1.csv

Expected log lines (examples):
  [epoch 01] train_mse=...  val_mse=...  val_mae=...
  [test] mse=...  mae=...
  [RESULT] tag=<tag> test_mse=... test_mae=... sharpe=... turnover=...

We also parse CLI echoes in the first line to capture patch_len/stride if present, e.g.:
  ... --patch-len 16 --stride 8 ...
"""

from pathlib import Path
import re
import math
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "outputs" / "logs_sleeves"
OUT  = ROOT / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

# tag format from the runner: <sleeve>_L252_H1_<encoder>_<idx>.log
TAG = re.compile(
    r"(?P<sleeve>equities|rates_credit|commod_alt)_L(?P<L>\d+)_H(?P<H>\d+)_(?P<encoder>\w+)_\d+",
    re.I
)

# Flexible patterns (handle "val mse", "val_mse", "validation mse", etc.)
PAT = {
    "val_mse":   re.compile(r"\bval(?:idation)?[\s_-]*mse\s*[:=]\s*([0-9.eE+-]+)", re.I),
    "test_mse":  re.compile(r"\btest(?:[\s_-]*mse|[\s_-]*loss)?\s*[:=]\s*([0-9.eE+-]+)", re.I),
    "test_mae":  re.compile(r"\btest[\s_-]*mae\s*[:=]\s*([0-9.eE+-]+)", re.I),
    # [RESULT] key=val form, most reliable
    "res_block": re.compile(
        r"\[RESULT\][^\n]*?\btag\s*=\s*([^\s]+)[^\n]*?\btest_mse\s*=\s*([0-9.eE+-]+)"
        r"[^\n]*?\btest_mae\s*=\s*([0-9.eE+-]+)[^\n]*?\bsharpe\s*=\s*([0-9.eE+-]+)"
        r"[^\n]*?\bturnover\s*=\s*([0-9.eE+-]+)", re.I),
    # fallbacks (if [RESULT] missing)
    "test_mse_any": re.compile(r"\bmse\s*=\s*([0-9.eE+-]+)", re.I),
    "test_mae_any": re.compile(r"\bmae\s*=\s*([0-9.eE+-]+)", re.I),
    # flags in the echoed command line at the top of the log
    "patch_len": re.compile(r"(?:\bpatch[_-]?len\b|\bp\b)\s*[:=\s]+\s*(\d+)", re.I),
    "stride":    re.compile(r"(?:\bstride\b|\bs\b)\s*[:=\s]+\s*(\d+)", re.I),
}

def last_match_float(pat, text, group=1):
    m = None
    for m in pat.finditer(text):
        pass
    if not m:
        return None
    try:
        return float(m.group(group))
    except Exception:
        return None

def first_block_result(text):
    """
    Return dict from [RESULT] line if present.
    Keys: tag, test_mse, test_mae, sharpe, turnover
    """
    m = PAT["res_block"].search(text)
    if not m:
        return None
    return {
        "tag":      m.group(1),
        "test_mse": float(m.group(2)),
        "test_mae": float(m.group(3)),
        "sharpe":   float(m.group(4)),
        "turnover": float(m.group(5)),
    }

def safe_float(x):
    try:
        return float(x)
    except Exception:
        return None

rows = []
logs = sorted(LOGS.glob("*.log"))
if not logs:
    print(f"[warn] no logs found in {LOGS}")
for log in logs:
    name = log.stem
    m = TAG.match(name)
    if not m:
        # skip non-standard names
        continue
    sleeve = m.group("sleeve")
    L = int(m.group("L"))
    H = int(m.group("H"))
    enc = m.group("encoder")

    text = log.read_text(errors="ignore")

    # Grab config flags if present
    p = last_match_float(PAT["patch_len"], text)
    s = last_match_float(PAT["stride"], text)

    # Validation mse
    val_mse = last_match_float(PAT["val_mse"], text)

    # Prefer the standardized [RESULT] line
    res = first_block_result(text)
    if res:
        test_mse = res["test_mse"]
        test_mae = res["test_mae"]
        sharpe   = res["sharpe"]
        turnover = res["turnover"]
        tag      = res["tag"]
    else:
        # Fallbacks if [RESULT] is missing (older logs)
        tag = name
        # Try specific [test] lines first
        test_mse = last_match_float(PAT["test_mse"], text)
        test_mae = last_match_float(PAT["test_mae"], text)
        # Very last resort: any "mse=" / "mae=" in the file (least specific)
        if test_mse is None:
            test_mse = last_match_float(PAT["test_mse_any"], text)
        if test_mae is None:
            test_mae = last_match_float(PAT["test_mae_any"], text)
        sharpe = None
        turnover = None

    rows.append({
        "tag": tag,
        "sleeve": sleeve,
        "L": L,
        "H": H,
        "encoder": enc,
        "patch_len": int(p) if p is not None and not math.isnan(p) else None,
        "stride": int(s) if s is not None and not math.isnan(s) else None,
        "val_mse": val_mse,
        "test_mse": test_mse,
        "test_mae": test_mae,
        "sharpe": sharpe,
        "turnover": turnover,
    })

df = pd.DataFrame(rows).sort_values(["sleeve","encoder","patch_len","stride"], na_position="last")
out_csv = OUT / "sleeves_h1.csv"
df.to_csv(out_csv, index=False)
print(f"[ok] wrote {out_csv} with {len(df)} rows")

# Quick: best-per-encoder within each sleeve (prioritize Test MSE if present, else Val MSE)
def pick_best(g):
    if g["test_mse"].notna().any():
        g = g.sort_values("test_mse")
    else:
        g = g.sort_values("val_mse")
    return g.iloc[0]

if len(df):
    best = (df.groupby(["sleeve","encoder"], as_index=False)
              .apply(pick_best)
              .reset_index(drop=True))
    cols = ["sleeve","encoder","patch_len","stride","val_mse","test_mse","test_mae","sharpe","turnover"]
    print("\n[best-by-encoder within each sleeve]")
    print(best[cols].to_string(index=False))
