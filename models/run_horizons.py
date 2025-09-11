#!/usr/bin/env python3
# run_horizons.py (repo root)
import itertools, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CFG  = ROOT / "config.json"

L_grid = [252, 128]
H_grid = [1, 5, 20]

encoders = [
    ("pointwise", {}),
    ("patch", {"--patch-len":"8",  "--stride":"4"}),
    ("patch", {"--patch-len":"16", "--stride":"8"}),
    ("patch", {"--patch-len":"32", "--stride":"16"}),
    ("ivar",   {}),
    ("cross",  {"--patch-len":"8",  "--stride":"4"}),
    ("cross",  {"--patch-len":"16", "--stride":"8"}),
]

def run(cmd):
    print("[cmd]", " ".join(map(str, cmd)), flush=True)
    subprocess.run(cmd, check=True)

for L in L_grid:
    for H in H_grid:
        # Make tensors
        run([sys.executable, "-m", "src.loader",
             "--config", str(CFG),
             "--make-tensors",
             "--lookback", str(L),
             "--horizon",  str(H)])
        npz = ROOT / f"outputs/tensors_L{L}_H{H}.npz"

        # Train models
        for i, (enc, extra) in enumerate(encoders, 1):
            cmd = [sys.executable, "-m", "src.train_baselines",
                   "--npz", str(npz),
                   "--encoder", enc,
                   "--d-model", "256",
                   "--heads", "8",
                   "--depth", "3",
                   "--dropout", "0.1",
                   "--epochs", "3",
                   "--eval-test",
                   "--tag", f"L{L}_H{H}_{enc}_{i}"]
            for k, v in extra.items():
                cmd += [k, v]
            run(cmd)
