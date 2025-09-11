#!/usr/bin/env python3
import re, json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "outputs" / "logs"
METR = ROOT / "outputs" / "metrics"   # optional per-run jsons here (if present)
OUT  = ROOT / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

TAG_RE = re.compile(r"L(?P<L>\d+)_H(?P<H>\d+)_(?P<enc>\w+)_\d+")

# Liberal patterns for various print styles
PAT = {
    # train/val
    "train_mse":  re.compile(r"\btrain(?:[_\s-]*mse|[_\s-]*loss)\b\s*[:=]\s*([0-9.eE+-]+)", re.I),
    "val_mse":    re.compile(r"\bval(?:idation)?(?:[_\s-]*mse|[_\s-]*loss)\b\s*[:=]\s*([0-9.eE+-]+)", re.I),

    # common test variants
    "test_mse":   re.compile(r"\btest(?:[_\s-]*mse|[_\s-]*loss)\b\s*[:=]\s*([0-9.eE+-]+)", re.I),
    "test_mae":   re.compile(r"\btest(?:[_\s-]*mae)\b\s*[:=]\s*([0-9.eE+-]+)", re.I),
    "sharpe":     re.compile(r"\bsharpe(?:@10bps)?\b\s*[:=]\s*([0-9.eE+-]+)", re.I),
    "turnover":   re.compile(r"\bturnover\b\s*[:=]\s*([0-9.eE+-]+)", re.I),

    # compact “Test -> mse=..., mae=...”
    "test_mse_kv": re.compile(r"test[^\n]*?\bmse\s*=\s*([0-9.eE+-]+)", re.I),
    "test_mae_kv": re.compile(r"test[^\n]*?\bmae\s*=\s*([0-9.eE+-]+)", re.I),

    # misc
    "epochs":     re.compile(r"\bepochs?\b\s*[:=]\s*(\d+)", re.I),
    "elapsed_s":  re.compile(r"\b(elapsed|time)\b\s*[:=]\s*([0-9.eE+-]+)\s*(?:s|sec|seconds)?", re.I),
    "patch_len":  re.compile(r"(?:\bpatch[_-]?len\b|\bp\b)\s*[:=\s]+\s*(\d+)", re.I),
    "stride":     re.compile(r"(?:\bstride\b|\bs\b)\s*[:=\s]+\s*(\d+)", re.I),

}

def last_num(p: re.Pattern, text: str, group: int = 1, cast=float):
    m = None
    for m in p.finditer(text):
        pass
    if not m:
        return None
    return cast(m.group(group))

def try_metrics_json(tag: str):
    """Optionally merge from outputs/metrics/{tag}.json if it exists."""
    f = METR / f"{tag}.json"
    if not f.exists():
        return {}
    try:
        j = json.loads(f.read_text())
    except Exception:
        return {}
    # Normalize potential keys
    out = {}
    for k in ("test_mse","test_mae","sharpe","turnover","val_mse","train_mse"):
        if k in j and isinstance(j[k], (int,float)):
            out[k] = float(j[k])
    # common alternates
    if "test_loss" in j and "test_mse" not in out:
        out["test_mse"] = float(j["test_loss"])
    if "Sharpe" in j and "sharpe" not in out:
        out["sharpe"] = float(j["Sharpe"])
    return out

rows = []
for log in sorted(LOGS.glob("L*_H*_*_*.log")):
    tag = log.stem
    tm = TAG_RE.match(tag)
    if not tm:
        continue
    L = int(tm.group("L")); H = int(tm.group("H")); enc = tm.group("enc")

    text = log.read_text(errors="ignore")

    row = {"tag": tag, "L": L, "H": H, "encoder": enc}

    # parse from log (liberal)
    row["train_mse"] = last_num(PAT["train_mse"], text)
    row["val_mse"]   = last_num(PAT["val_mse"], text)
    row["test_mse"]  = last_num(PAT["test_mse"], text)
    row["test_mae"]  = last_num(PAT["test_mae"], text)
    row["sharpe"]    = last_num(PAT["sharpe"], text)
    row["turnover"]  = last_num(PAT["turnover"], text)

    # fallbacks from “Test -> mse=..., mae=...”
    if row["test_mse"] is None:
        row["test_mse"] = last_num(PAT["test_mse_kv"], text)
    if row["test_mae"] is None:
        row["test_mae"] = last_num(PAT["test_mae_kv"], text)

    row["epochs"]    = last_num(PAT["epochs"], text, cast=int)
    row["elapsed_s"] = last_num(PAT["elapsed_s"], text, group=2)
    row["patch_len"] = last_num(PAT["patch_len"], text, cast=int)
    row["stride"]    = last_num(PAT["stride"], text, cast=int)

    # Optional: merge metrics JSON if present
    row.update({k:v for k,v in try_metrics_json(tag).items() if v is not None})

    # plotting metric: prefer test_mse if present
    row["metric_for_plot"] = row["test_mse"] if row["test_mse"] is not None else row["val_mse"]

    rows.append(row)

df = pd.DataFrame(rows).sort_values(["L","H","encoder","patch_len","stride"], na_position="last")
csv_path = OUT / "horizons_v2.csv"
df.to_csv(csv_path, index=False)
print(f"[ok] wrote {csv_path} with {len(df)} rows")

# Tiny summary
with pd.option_context("display.max_rows", None):
    print(df[(df["L"]==252)&(df["H"]==1)][["encoder","patch_len","stride","val_mse","test_mse","test_mae","sharpe","turnover"]])
