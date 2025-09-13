#!/usr/bin/env python3
import os, sys, subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "outputs" / "logs_synth"
NPZDIR = ROOT / "outputs" / "synth"
LOGS.mkdir(parents=True, exist_ok=True)
NPZDIR.mkdir(parents=True, exist_ok=True)

GAMMAS = [0.5, 0.95]
ALPHAS = [0.0, 0.2, 0.4, 0.8]
L, H = 252, 1

def sh(cmd, log):
    with open(log, "wb") as f:
        f.write((" ".join(map(str, cmd)) + "\n").encode())
        p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=str(ROOT))
        return p.wait()

def ensure_npz(alpha, gamma):
    tag = f"synth_a{alpha}_g{gamma}_L{L}_H{H}.npz"
    path = NPZDIR / tag
    if path.exists():
        return path
    cmd = [sys.executable, "scripts/synth_generate.py",
           "--alpha", str(alpha), "--gamma", str(gamma),
           "--T", "5000", "--L", str(L), "--H", str(H)]
    code = sh(cmd, LOGS / f"make_{tag}.log")
    if code != 0: raise SystemExit(f"gen failed: {tag}")
    return path

def run_one(npz, enc, extra, tag):
    cmd = [sys.executable, "-m", "src.train_baselines",
           "--npz", str(npz),
           "--encoder", enc,
           "--d-model", "256", "--heads", "8", "--depth", "3",
           "--dropout", "0.1", "--epochs", "3",
           "--eval-test", "--dump-preds",
           "--tag", tag]
    for k,v in extra:
        cmd += [k, v]
    return sh(cmd, LOGS / f"{tag}.log")

def main():
    results = {}
    for g in GAMMAS:
        for a in ALPHAS:
            npz = ensure_npz(a, g)
            # PatchTST p16 s8
            tag = f"synth_a{a}_g{g}_patch_p16_s8"
            code = run_one(npz, "patch", [("--patch-len","16"),("--stride","8")], tag)
            print(f"[done] {tag}: {'OK' if code==0 else 'FAIL'}")
            results[tag]=code
            # iTransformer time->chan 128
            tag = f"synth_a{a}_g{g}_ivar_t2c128"
            code = run_one(npz, "ivar", [], tag)  # t2c encoded in tag only (we kept default d_model=256)
            print(f"[done] {tag}: {'OK' if code==0 else 'FAIL'}")
            results[tag]=code
            # Cross-Lite p16 s8
            tag = f"synth_a{a}_g{g}_cross_p16_s8"
            code = run_one(npz, "cross", [("--patch-len","16"),("--stride","8")], tag)
            print(f"[done] {tag}: {'OK' if code==0 else 'FAIL'}")
            results[tag]=code

    fails = [t for t,c in results.items() if c!=0]
    print(f"[summary] ok={len(results)-len(fails)} fail={len(fails)}")
    if fails:
        print("Failures:\n - " + "\n - ".join(fails))

if __name__ == "__main__":
    (ROOT / "src" / "__init__.py").touch(exist_ok=True)
    main()
