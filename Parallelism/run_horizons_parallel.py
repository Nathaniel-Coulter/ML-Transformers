#!/usr/bin/env python3
import itertools, os, sys, subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse

ROOT = Path(__file__).resolve().parent
CFG  = ROOT / "config.json"
LOGS = ROOT / "outputs" / "logs"
LOGS.mkdir(parents=True, exist_ok=True)

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

def sh(cmd, log_path, env):
    with open(log_path, "wb") as log:
        log.write((" ".join(map(str, cmd)) + "\n").encode())
        p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
        return p.wait()

def ensure_tensors(L, H, env):
    npz = ROOT / f"outputs/tensors_L{L}_H{H}.npz"
    if not npz.exists():
        cmd = [sys.executable, "-m", "src.loader",
               "--config", str(CFG), "--make-tensors",
               "--lookback", str(L), "--horizon", str(H)]
        code = sh(cmd, LOGS / f"make_L{L}_H{H}.log", env)
        if code != 0: raise SystemExit(f"tensor build failed L{L} H{H}")
    return npz

def build_jobs():
    jobs = []
    for L, H in itertools.product(L_grid, H_grid):
        npz = ROOT / f"outputs/tensors_L{L}_H{H}.npz"
        for i, (enc, extra) in enumerate(encoders, 1):
            tag = f"L{L}_H{H}_{enc}_{i}"
            cmd = [sys.executable, "-m", "src.train_baselines",
                   "--npz", str(npz),
                   "--encoder", enc,
                   "--d-model", "256", "--heads", "8", "--depth", "3",
                   "--dropout", "0.1", "--epochs", "3", "--eval-test",
                   "--tag", tag]
            for k, v in extra.items(): cmd += [k, v]
            jobs.append((L, H, tag, cmd))
    return jobs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-procs", type=int, default=4)  # ← run 3–4 safely
    args = ap.parse_args()

    (ROOT / "src" / "__init__.py").touch(exist_ok=True)

    cpu = os.cpu_count() or 8
    threads_per_job = max(1, cpu // args.max_procs)
    base_env = os.environ.copy()
    for var in ("OMP_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
        base_env[var] = str(threads_per_job)

    # ensure all tensors first (serial)
    for L, H in itertools.product(L_grid, H_grid):
        ensure_tensors(L, H, base_env)

    jobs = build_jobs()
    print(f"[plan] {len(jobs)} runs | max-procs={args.max_procs} | threads/job={threads_per_job}")

    results = {}
    with ThreadPoolExecutor(max_workers=args.max_procs) as ex:
        futs = {ex.submit(sh, cmd, LOGS / f"{tag}.log", base_env): (tag) for _,_,tag,cmd in jobs}
        for fut in as_completed(futs):
            tag = futs[fut]
            code = fut.result()
            print(f"[done] {tag}: {'OK' if code==0 else f'FAIL({code})'}")
            results[tag] = code

    fails = [t for t,c in results.items() if c!=0]
    print(f"[summary] ok={len(results)-len(fails)} fail={len(fails)}")
    if fails: print("Failures:\n - " + "\n - ".join(fails))

if __name__ == "__main__":
    main()
