#!/usr/bin/env python3
import json
from pathlib import Path
import re
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MET = ROOT / "outputs" / "metrics"
OUTCSV = ROOT / "outputs" / "synth_summary.csv"

rowz = []
pat = re.compile(r"synth_a(?P<a>[\d.]+)_g(?P<g>[\d.]+)_(?P<enc>\w+).*")

for js in MET.glob("*.json"):
    m = pat.match(js.stem)
    if not m:
        continue
    d = json.loads(js.read_text())
    rowz.append({
        "alpha": float(m.group("a")),
        "gamma": float(m.group("g")),
        "encoder": m.group("enc"),
        "val_mse": d.get("val_mse", None),
        "test_mse": d.get("test_mse", None),
        "test_mae": d.get("test_mae", None),
        "sharpe": d.get("sharpe", None),
        "turnover": d.get("turnover", None),
    })

df = pd.DataFrame(rowz).sort_values(["gamma","alpha","encoder"])
df.to_csv(OUTCSV, index=False)
print(f"[ok] wrote {OUTCSV} with {len(df)} rows")
