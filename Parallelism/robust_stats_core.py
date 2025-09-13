#!/usr/bin/env python3
import os, sys, json, math, subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd

# -------------------------
# Paths / constants
# -------------------------
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
PRED_DIR = OUT / "preds"
FIGS = ROOT / "figs"
(OUT).mkdir(exist_ok=True, parents=True)
(PRED_DIR).mkdir(exist_ok=True, parents=True)
(FIGS).mkdir(exist_ok=True, parents=True)

# Parallel controls (like your sleeves runner)
MAX_PROCS = int(os.environ.get("ROBUST_MAX_PROCS", "4"))
CPU = os.cpu_count() or 8
threads_per_job = max(1, CPU // max(1, MAX_PROCS))
BASE_ENV = os.environ.copy()
for var in ("OMP_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    BASE_ENV[var] = str(threads_per_job)

# Sleeves + NPZ sources (created by your earlier runners)
SLEEVES = {
    "equities":     ["SPY","QQQ","IWM"],
    "rates_credit": ["TLT","IEF","LQD","HYG"],
    "commod_alt":   ["GLD","DBC","VNQ","EFA","EEM"],
}
L, H = 252, 1
COSTS_BPS = 10.0

# Core models/configs we’ll standardize on
CORE = [
    ("pointwise", {"tag_sfx": "pointwise_base"}),
    ("patch",     {"tag_sfx": "patch_p16_s8", "patch_len": 16, "stride": 8}),
    ("ivar",      {"tag_sfx": "ivar_base"}),  # time->chan default from your enc
    ("cross",     {"tag_sfx": "cross_p16_s8", "patch_len": 16, "stride": 8}),
]

# -------------------------
# Small helpers: allocator + stats
# -------------------------
def _zscore_np(x, axis=1, eps=1e-8):
    m = np.nanmean(x, axis=axis, keepdims=True)
    s = np.nanstd(x, axis=axis, keepdims=True) + eps
    return (x - m) / s

def weights_from_signals(sig, mode="dollar_neutral", clip=3.0, eps=1e-12):
    if mode == "long_only":
        s = np.maximum(sig, 0.0)
        denom = s.sum(axis=1, keepdims=True) + eps
        return s / denom
    z = _zscore_np(sig, axis=1)
    z = np.clip(z, -clip, clip)
    denom = np.sum(np.abs(z), axis=1, keepdims=True) + eps
    w = z / denom
    w = w - np.mean(w, axis=1, keepdims=True)
    denom = np.sum(np.abs(w), axis=1, keepdims=True) + eps
    return w / denom

def pnl_series(y_true, y_pred, costs_bps=COSTS_BPS, mode="dollar_neutral"):
    w = weights_from_signals(y_pred, mode=mode)       # (T, N)
    pnl_gross = np.sum(w * y_true, axis=1)            # (T,)
    dw = np.diff(w, axis=0)
    turnover_t = 0.5 * np.sum(np.abs(dw), axis=1)
    turnover_t = np.concatenate([[0.0], turnover_t])
    cost_rate = costs_bps / 10_000.0
    pnl_net = pnl_gross - cost_rate * turnover_t
    return pnl_net

def newey_west_se(x, bandwidth=None):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    T = len(x)
    if T < 5: return float("nan")
    mu = np.mean(x); e = x - mu
    if bandwidth is None:
        bandwidth = int(np.floor(4 * (T/100.0)**(2/9))); bandwidth = max(1, bandwidth)
    gamma0 = np.dot(e, e) / T
    var = gamma0
    for k in range(1, min(bandwidth, T-1)+1):
        w = 1.0 - k/(bandwidth+1.0)
        cov = np.dot(e[:-k], e[k:]) / T
        var += 2.0 * w * cov
    se_mean = math.sqrt(var / T)
    return se_mean

def dm_test_mse(e1, e2, bandwidth=None):
    e1 = np.asarray(e1); e2 = np.asarray(e2)
    if e1.ndim == 2:
        l1_t = np.mean((e1)**2, axis=1)
        l2_t = np.mean((e2)**2, axis=1)
    else:
        l1_t = (e1)**2; l2_t = (e2)**2
    d = l1_t - l2_t
    d = d[np.isfinite(d)]
    if len(d) < 10: return float("nan"), float("nan")
    mu = np.mean(d)
    se = newey_west_se(d, bandwidth=bandwidth)
    if not np.isfinite(se) or se == 0.0: return float("nan"), float("nan")
    stat = mu / se
    from math import erf
    Phi = lambda z: 0.5*(1.0 + erf(z/math.sqrt(2.0)))
    p = 2.0 * (1.0 - Phi(abs(stat)))
    return float(stat), float(p)

def moving_block_bootstrap_idx(T, block=10, B=1000, rng=None):
    if rng is None: rng = np.random.default_rng(42)
    starts = rng.integers(0, T, size=(B, math.ceil(T/block)))
    idx = []
    for b in range(B):
        seq = []
        for s in starts[b]:
            seq.extend([(s + j) % T for j in range(block)])
            if len(seq) >= T: break
        idx.append(np.array(seq[:T], dtype=int))
    return idx

def bootstrap_ci_mean(x, block=10, B=1000, alpha=0.05, rng=None):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    T = len(x)
    if T < 10: return (float("nan"), float("nan"))
    idxs = moving_block_bootstrap_idx(T, block=block, B=B, rng=rng)
    means = np.array([np.mean(x[i]) for i in idxs])
    lo, hi = np.quantile(means, [alpha/2, 1-alpha/2])
    return float(lo), float(hi)

def sharpe_from_pnl(pnl):
    pnl = np.asarray(pnl, float)
    mu = np.mean(pnl); sd = np.std(pnl, ddof=1) + 1e-12
    return float((mu/sd) * np.sqrt(252.0))

def bootstrap_ci_sharpe(pnl, block=10, B=1000, alpha=0.05, rng=None):
    pnl = np.asarray(pnl, float); pnl = pnl[np.isfinite(pnl)]
    T = len(pnl)
    if T < 10: return (float("nan"), float("nan"))
    idxs = moving_block_bootstrap_idx(T, block=block, B=B, rng=rng)
    s = np.array([sharpe_from_pnl(pnl[i]) for i in idxs])
    lo, hi = np.quantile(s, [alpha/2, 1-alpha/2])
    return float(lo), float(hi)

# -------------------------
# Preds availability + training task builder
# -------------------------
def pred_paths_for(tag):
    return (PRED_DIR / f"{tag}_y_true.npy",
            PRED_DIR / f"{tag}_y_pred.npy",
            PRED_DIR / f"{tag}_dates.npy")

def needs_training(tag):
    ytrue, ypred, ydates = pred_paths_for(tag)
    return not (ytrue.exists() and ypred.exists() and ydates.exists())

def train_once(sleeve: str, enc: str, cfg: dict):
    tag = f"{sleeve}_L{L}_H{H}_{cfg['tag_sfx']}"
    if not needs_training(tag):
        return (tag, 0)
    npz = ROOT / f"outputs/tensors_{sleeve}_L{L}_H{H}.npz"
    if not npz.exists():
        raise SystemExit(f"Missing NPZ for {sleeve}: {npz}")

    cmd = [sys.executable, "-m", "src.train_baselines",
           "--npz", str(npz),
           "--encoder", enc,
           "--d-model", "256", "--heads", "8", "--depth", "3",
           "--dropout", "0.1", "--epochs", "3",
           "--eval-test", "--dump-preds",
           "--tag", tag]
    if enc in ("patch", "cross"):
        if "patch_len" in cfg: cmd += ["--patch-len", str(cfg["patch_len"])]
        if "stride"    in cfg: cmd += ["--stride",    str(cfg["stride"])]

    log = OUT / "logs_ablate" / f"robust_{tag}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "wb") as f:
        f.write((" ".join(map(str, cmd))+"\n").encode())
        ret = subprocess.Popen(cmd, cwd=str(ROOT), stdout=f, stderr=subprocess.STDOUT, env=BASE_ENV).wait()
    # confirm
    ytrue, ypred, ydates = pred_paths_for(tag)
    if ret != 0 or not (ytrue.exists() and ypred.exists() and ydates.exists()):
        raise SystemExit(f"Pred dump failed for {tag}. See {log}")
    return (tag, ret)

# -------------------------
# Main
# -------------------------
def main():
    # 1) Build task list for missing preds
    tasks = []
    for sleeve in SLEEVES.keys():
        for enc, cfg in CORE:
            tag = f"{sleeve}_L{L}_H{H}_{cfg['tag_sfx']}"
            if needs_training(tag):
                tasks.append((sleeve, enc, cfg))

    # 2) Run trainings in parallel (only what’s missing)
    if tasks:
        fut2desc = {}
        print(f"[plan] training tasks={len(tasks)}  max-procs={MAX_PROCS}  threads/job={threads_per_job}")
        with ThreadPoolExecutor(max_workers=MAX_PROCS) as ex:
            for sleeve, enc, cfg in tasks:
                fut = ex.submit(train_once, sleeve, enc, cfg)
                fut2desc[fut] = (sleeve, enc, cfg)
            for fut in as_completed(fut2desc):
                sleeve, enc, cfg = fut2desc[fut]
                try:
                    tag, code = fut.result()
                    print(f"[done] {tag}: {'OK' if code==0 else 'FAIL'}")
                except Exception as e:
                    print(f"[fail] {sleeve}/{enc}: {e}")
                    raise

    # 3) Stats computation (fast, single-threaded)
    rows = []
    dm_rows = []
    for sleeve in SLEEVES.keys():
        per_model_series = {}
        for enc, cfg in CORE:
            tag = f"{sleeve}_L{L}_H{H}_{cfg['tag_sfx']}"
            y_true = np.load(PRED_DIR / f"{tag}_y_true.npy")
            y_pred = np.load(PRED_DIR / f"{tag}_y_pred.npy")
            dates  = np.load(PRED_DIR / f"{tag}_dates.npy", allow_pickle=True)

            mse_t = np.mean((y_pred - y_true)**2, axis=1)
            mse_mean = float(np.mean(mse_t))
            mse_hac  = float(newey_west_se(mse_t))
            mse_lo, mse_hi = bootstrap_ci_mean(mse_t, block=10, B=1000, alpha=0.05)

            pnl = pnl_series(y_true, y_pred, costs_bps=COSTS_BPS, mode="dollar_neutral")
            sharpe = float(sharpe_from_pnl(pnl))
            pnl_hac = float(newey_west_se(pnl))
            s_lo, s_hi = bootstrap_ci_sharpe(pnl, block=10, B=1000, alpha=0.05)

            rows.append({
                "sleeve": sleeve, "encoder": enc, "tag": tag,
                "test_mse": mse_mean, "test_mse_hac_se": mse_hac,
                "test_mse_ci_lo": mse_lo, "test_mse_ci_hi": mse_hi,
                "sharpe": sharpe, "mean_pnl_hac_se": pnl_hac,
                "sharpe_ci_lo": s_lo, "sharpe_ci_hi": s_hi
            })
            per_model_series[enc] = {"err": (y_pred - y_true)}

        base_enc = "patch"
        for enc, _ in CORE:
            if enc == base_enc: continue
            stat, p = dm_test_mse(per_model_series[enc]["err"], per_model_series[base_enc]["err"])
            dm_rows.append({
                "sleeve": sleeve,
                "compared_to": base_enc,
                "encoder": enc,
                "dm_stat": stat,
                "dm_pvalue": p
            })

    df = pd.DataFrame(rows)
    out_csv = OUT / "robust_stats_core.csv"
    df.to_csv(out_csv, index=False)
    print(f"[ok] wrote {out_csv} with {len(df)} rows")

    df_dm = pd.DataFrame(dm_rows)
    out_dm = OUT / "robust_stats_core_dm.csv"
    df_dm.to_csv(out_dm, index=False)
    print(f"[ok] wrote {out_dm} with {len(df_dm)} rows")

    # 4) Emit LaTeX table (same format)
    def fmt_pm(x, se):
        if not (np.isfinite(x) and np.isfinite(se)): return r"--"
        return f"{x:.6f} $\\pm$ {se:.6f}"

    def fmt_ci(lo, hi, nd=6):
        if not (np.isfinite(lo) and np.isfinite(hi)): return r"--"
        return f"[{lo:.{nd}f}, {hi:.{nd}f}]"

    enc_order = ["pointwise","patch","ivar","cross"]
    enc_label = {
        "pointwise":"Point-wise",
        "patch":"PatchTST (p16,s8)",
        "ivar":"iTransformer",
        "cross":"Cross-Lite (p16,s8)"
    }

    lines = []
    lines.append(r"% auto-generated by scripts/robust_stats_core.py")
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Robustness statistics on the core $H{=}1$ comparison. We report HAC (Newey--West) standard errors for per-date Test MSE means, Diebold--Mariano (DM) tests versus PatchTST (p16,s8) on squared-error losses, and 95\% block-bootstrap confidence intervals for Test MSE and Sharpe (dollar-neutral, 10 bps).}")
    lines.append(r"\label{tab:robust-core}")
    lines.append(r"\resizebox{0.98\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{l l c c c c c}")
    lines.append(r"\toprule")
    lines.append(r"Sleeve & Model & Test MSE $\pm$ HAC SE & MSE 95\% CI & Sharpe & Sharpe 95\% CI & DM $p$ (vs PatchTST) \\")
    lines.append(r"\midrule")

    for sleeve in SLEEVES.keys():
        dsl = df[df.sleeve==sleeve]
        dmd = df_dm[df_dm.sleeve==sleeve].set_index("encoder") if len(df_dm)>0 else pd.DataFrame()
        first = True
        for enc in enc_order:
            row = dsl[dsl.encoder==enc]
            if row.empty: continue
            r = row.iloc[0]
            dm_p = "--" if enc=="patch" or dmd.empty or enc not in dmd.index else f"{float(dmd.loc[enc,'dm_pvalue']):.3f}"
            prefix = sleeve if first else ""
            first = False
            lines.append(
                f"{prefix} & {enc_label[enc]} & "
                f"{fmt_pm(r.test_mse, r.test_mse_hac_se)} & "
                f"{fmt_ci(r.test_mse_ci_lo, r.test_mse_ci_hi)} & "
                f"{r.sharpe:.3f} & {fmt_ci(r.sharpe_ci_lo, r.sharpe_ci_hi, nd=3)} & "
                f"{dm_p} \\"
            )
        lines.append(r"\midrule")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}}")
    lines.append(r"\end{table*}")

    tex_path = OUT / "robustness_main.tex"
    tex_path.write_text("\n".join(lines))
    print(f"[ok] wrote {tex_path}")

if __name__ == "__main__":
    main()
