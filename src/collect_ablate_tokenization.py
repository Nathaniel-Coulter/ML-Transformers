#!/usr/bin/env python3
import json, re
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "outputs" / "logs_ablate"
METRICS = ROOT / "outputs" / "metrics"
OUTCSV = ROOT / "outputs" / "ablations.csv"

rows = []

# Prefer metrics JSON (created when you passed --dump-preds)
for f in METRICS.glob("*.json"):
    tag = f.stem
    d = json.loads(f.read_text())
    rows.append({
        "tag": tag,
        "val_mse": d.get("val_mse"),
        "test_mse": d.get("test_mse"),
        "test_mae": d.get("test_mae"),
        "sharpe": d.get("sharpe"),
        "turnover": d.get("turnover"),
        "costs_bps": d.get("costs_bps"),
        "alloc_mode": d.get("alloc_mode"),
    })

# Fallback: scrape logs if any run didn’t emit metrics (optional)
pat = re.compile(r"\[RESULT\]\s+tag=(\S+)\s+test_mse=([\d.]+)\s+test_mae=([\d.]+)\s+sharpe=([-\d.]+)\s+turnover=([\d.]+)")
for f in LOGS.glob("*.log"):
    txt = f.read_text()
    m = pat.search(txt)
    if m:
        tag, tmse, tmae, shp, to = m.groups()
        if not any(r["tag"] == tag for r in rows):
            rows.append({
                "tag": tag,
                "val_mse": None,
                "test_mse": float(tmse),
                "test_mae": float(tmae),
                "sharpe": float(shp),
                "turnover": float(to),
                "costs_bps": None,
                "alloc_mode": None,
            })

df = pd.DataFrame(rows)
# Nice ordering by family if you want
if not df.empty:
    df = df.sort_values("tag")
df.to_csv(OUTCSV, index=False)
print(f"[ok] wrote {OUTCSV} with {len(df)} rows")
