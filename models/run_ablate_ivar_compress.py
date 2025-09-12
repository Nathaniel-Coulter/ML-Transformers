#!/usr/bin/env python3
import os, sys, json, subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE_CFG = ROOT / "config.json"
LOGS = ROOT / "outputs" / "logs_ablate"
TMP = ROOT / "outputs" / "tmp_configs"
LOGS.mkdir(parents=True, exist_ok=True)
TMP.mkdir(parents=True, exist_ok=True)

# Sleeves and T->C sweep
SLEEVES = {
    "equities":     ["SPY","QQQ","IWM"],
    "rates_credit": ["TLT","IEF","LQD","HYG"],
    "commod_alt":   ["GLD","DBC","VNQ","EFA","EEM"],
}
L = 252
H = 1
T2C = [64, 96, 128, 160, 192]

# Thread env (match your other runners)
MAX_PROCS = int(os.environ.get("SLEEVES_MAX_PROCS", "4"))
CPU = os.cpu_count() or 8
threads_per_job = max(1, CPU // MAX_PROCS)
BASE_ENV = os.environ.copy()
for var in ("OMP_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    BASE_ENV[var] = str(threads_per_job)

def sh(cmd, log_file):
    with open(log_file, "wb") as log:
        log.write((" ".join(map(str, cmd)) + "\n").encode())
        p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=BASE_ENV, cwd=str(ROOT))
        return p.wait()

def make_temp_config(sleeve, tickers):
    if not BASE_CFG.exists():
        raise SystemExit(f"Base config not found: {BASE_CFG}")
    base = json.loads(BASE_CFG.read_text())
    base["tickers"] = tickers
    tmp = TMP / f"config_{sleeve}.json"
    tmp.write_text(json.dumps(base, indent=2))
    return tmp

def ensure_tensors(sleeve, tickers):
    npz = ROOT / f"outputs/tensors_{sleeve}_L{L}_H{H}.npz"
    if npz.exists():
        return npz
    cfg = make_temp_config(sleeve, tickers)
    log_path = LOGS / f"make_{sleeve}_L{L}_H{H}.log"
    cmd = [sys.executable, "-m", "src.loader",
           "--config", str(cfg),
           "--make-tensors", "--lookback", str(L), "--horizon", str(H)]
    code = sh(cmd, log_path)
    if code != 0:
        print(f"[error] tensor build failed for {sleeve}. See {log_path}")
        raise SystemExit(1)
    generic = ROOT / f"outputs/tensors_L{L}_H{H}.npz"
    if generic.exists() and not npz.exists():
        generic.rename(npz)
    return npz

def main():
    results = {}
    for sleeve, tickers in SLEEVES.items():
        npz = ensure_tensors(sleeve, tickers)
        for k in T2C:
            tag = f"{sleeve}_L{L}_H{H}_ivar_t2c{k}"
            cmd = [sys.executable, "-m", "src.train_baselines",
                   "--npz", str(npz),
                   "--encoder", "ivar",
                   "--d-model", "256", "--heads", "8", "--depth", "3",
                   "--dropout", "0.1", "--epochs", "3",
                   "--eval-test", "--dump-preds",
                   "--ivar-time2chan", str(k),
                   "--tag", tag]
            code = sh(cmd, LOGS / f"{tag}.log")
            print(f"[done] {tag}: {'OK' if code==0 else 'FAIL'}")
            results[tag] = code

    fails = [t for t,c in results.items() if c!=0]
    print(f"[summary] ok={len(results)-len(fails)} fail={len(fails)}")
    if fails:
        print("Failures:\n - " + "\n - ".join(fails))

if __name__ == "__main__":
    (ROOT / "src" / "__init__.py").touch(exist_ok=True)
    main()
