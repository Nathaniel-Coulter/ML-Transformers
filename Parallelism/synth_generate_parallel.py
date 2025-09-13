#!/usr/bin/env python3
import os, sys, json, subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "outputs" / "logs_synth"
NPZ_DIR = ROOT / "outputs" / "synth"
LOGS.mkdir(parents=True, exist_ok=True)
NPZ_DIR.mkdir(parents=True, exist_ok=True)

# ========= Experiment grid =========
ALPHAS = [0.0, 0.2, 0.4, 0.8]   # dependency strength (lead–lag)
GAMMAS = [0.5, 0.95]            # AR(1) autocorr levels

# Models to run
MODELS = [
    ("patch", {"--patch-len": "16", "--stride": "8"},   "patch_p16_s8"),
    ("ivar",  {},                                      "ivar"),
    ("cross", {"--patch-len": "16", "--stride": "8"},  "cross_p16_s8"),
]

# ========= Parallelism/threading control =========
MAX_PROCS = int(os.environ.get("SYN_MAX_PROCS", "4"))
CPU = os.cpu_count() or 8
threads_per_job = max(1, CPU // max(1, MAX_PROCS))
BASE_ENV = os.environ.copy()
for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    BASE_ENV[var] = str(threads_per_job)

def f2s(x: float) -> str:
    """filename-safe float: 0.95 -> 0p95"""
    s = f"{x}".replace(".", "p")
    if s.endswith("0") and "p" in s:
        s = s.rstrip("0")
    return s

def synth_npz_path(alpha: float, gamma: float) -> Path:
    return NPZ_DIR / f"synth_a{f2s(alpha)}_g{f2s(gamma)}.npz"

def sh(cmd, log_path: Path):
    """Run a command, tee stdout/stderr to a log file, return exit code."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "wb") as log:
        log.write((" ".join(map(str, cmd)) + "\n").encode())
        p = subprocess.Popen(
            cmd, cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT, env=BASE_ENV
        )
        return p.wait()

def ensure_dataset(alpha: float, gamma: float) -> Path:
    """Create the synthetic dataset if missing, return its path."""
    out = synth_npz_path(alpha, gamma)
    if out.exists():
        return out
    log = LOGS / f"gen_a{f2s(alpha)}_g{f2s(gamma)}.log"
    cmd = [
        sys.executable, "scripts/synth_generate.py",
        "--alpha", str(alpha),
        "--gamma", str(gamma),
        "--n", "4000",            # tweak if you want longer series
        "--seed", "42",
        "--out", str(out),
    ]
    code = sh(cmd, log)
    if code != 0 or not out.exists():
        raise SystemExit(f"[error] synth generation failed for a={alpha}, g={gamma}. See {log}")
    return out

def jobs():
    """Yield (tag, cmd, log) tuples for all runs."""
    for a in ALPHAS:
        for g in GAMMAS:
            npz = ensure_dataset(a, g)  # blocking ensure, once per combo
            for enc, extra, suffix in MODELS:
                tag = f"synth_a{f2s(a)}_g{f2s(g)}_{suffix}"
                cmd = [
                    sys.executable, "-m", "src.train_baselines",
                    "--npz", str(npz),
                    "--encoder", enc,
                    "--d-model", "256", "--heads", "8", "--depth", "3",
                    "--dropout", "0.1", "--epochs", "3",
                    "--eval-test", "--dump-preds",
                    "--tag", tag,
                ]
                for k, v in extra.items():
                    cmd += [k, v]
                log = LOGS / f"{tag}.log"
                yield tag, cmd, log

def main():
    print(f"[plan] alphas={ALPHAS} gammas={GAMMAS} models={len(MODELS)} "
          f"max-procs={MAX_PROCS} threads/job={threads_per_job}")
    results = {}
    # Build all jobs (dataset generation happens inside jobs() in a blocking way per (a,g))
    job_list = list(jobs())
    # Now run training in parallel
    with ThreadPoolExecutor(max_workers=MAX_PROCS) as ex:
        fut2tag = {}
        for tag, cmd, log in job_list:
            fut = ex.submit(sh, cmd, log)
            fut2tag[fut] = tag
        for fut in as_completed(fut2tag):
            tag = fut2tag[fut]
            code = fut.result()
            print(f"[done] {tag}: {'OK' if code==0 else f'FAIL({code})'}")
            results[tag] = code

    fails = [t for t, c in results.items() if c != 0]
    print(f"[summary] ok={len(results)-len(fails)} fail={len(fails)}")
    if fails:
        print("Failures:\n - " + "\n - ".join(fails))

if __name__ == "__main__":
    (ROOT / "src" / "__init__.py").touch(exist_ok=True)
    main()
