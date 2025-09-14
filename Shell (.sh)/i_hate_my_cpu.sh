#!/usr/bin/env bash
set -euo pipefail

# ---------- config knobs you can tweak ----------
export SLEEVES_MAX_PROCS="${SLEEVES_MAX_PROCS:-4}"   # internal parallelism for the runners
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-3}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-3}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-3}"
PY="${PYTHON:-python}"                                # or python3 if you prefer
# -------------------------------------------------

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
OUT="${ROOT}/outputs"
FIGS="${ROOT}/figs"
LOGROOT="${OUT}/logs_grand"
RUNSTAMP="$(date +%Y%m%d_%H%M%S)"
RUNROOT="${OUT}/experiment_${RUNSTAMP}"
mkdir -p "${LOGROOT}" "${RUNROOT}"/{preds,metrics,figs,logs}

logrun () {
  local name="$1"; shift
  echo "[run] ${name} -> ${LOGROOT}/${name}.log"
  ( set -x; "$@" ) >"${LOGROOT}/${name}.log" 2>&1
  echo "[done] ${name}"
}

# ---- 1) Tokenization ablations (stride, ivar compress, cross spillover) ----
logrun "ablate_patch_stride"     "${PY}" "${ROOT}/scripts/run_ablate_patch_stride.py"
logrun "ablate_ivar_compress"    "${PY}" "${ROOT}/scripts/run_ablate_ivar_compress.py"
logrun "ablate_cross_spillover"  "${PY}" "${ROOT}/scripts/run_ablate_crosslite_spillover.py"

# ---- 2) Normalization check (z vs no-z) + collect + plot ----
logrun "norm_check"              "${PY}" "${ROOT}/scripts/run_norm_check.py"
logrun "collect_norm_check"      "${PY}" "${ROOT}/scripts/collect_norm_check.py"
logrun "plot_norm_check"         "${PY}" "${ROOT}/scripts/plot_norm_check.py"

# ---- 3) Synthetic grid + collect + plot ----
logrun "synth_grid"              "${PY}" "${ROOT}/scripts/run_synth_grid.py"
logrun "collect_synth"           "${PY}" "${ROOT}/scripts/collect_synth.py"
logrun "plot_synth"              "${PY}" "${ROOT}/scripts/plot_synth.py"

# ---- 4) Robustness stats (HAC/DM/bootstrap) ----
logrun "robust_stats_core"       "${PY}" "${ROOT}/scripts/robust_stats_core.py"

# ---------- Collect artifacts into a neat package ----------
# preds
shopt -s nullglob
for f in "${OUT}/preds/"*_y_true.npy "${OUT}/preds/"*_y_pred.npy "${OUT}/preds/"*_dates.npy; do
  cp -n "$f" "${RUNROOT}/preds/"
done

# metrics (csv/json/tex we generated)
for f in \
  "${OUT}/"regimes_costs.csv \
  "${OUT}/"ablations.csv \
  "${OUT}/"norm_check_summary.csv \
  "${OUT}/"synth_summary.csv \
  "${OUT}/"robust_stats_core.csv \
  "${OUT}/"robust_stats_core_dm.csv \
  "${OUT}/"robustness_main.tex \
  "${OUT}/metrics/"*.json 2>/dev/null
do
  [ -e "$f" ] && cp -n "$f" "${RUNROOT}/metrics/"
done

# figures
for f in "${FIGS}/"*.pdf "${FIGS}/"*.png 2>/dev/null; do
  [ -e "$f" ] && cp -n "$f" "${RUNROOT}/figs/"
done

# logs
cp -r "${LOGROOT}"/* "${RUNROOT}/logs/" || true
cp -r "${OUT}/logs_ablate"/* "${RUNROOT}/logs/" 2>/dev/null || true
cp -r "${OUT}/logs_sleeves"/* "${RUNROOT}/logs/" 2>/dev/null || true

# manifest
cat > "${RUNROOT}/MANIFEST.json" <<EOF
{
  "run_stamp": "${RUNSTAMP}",
  "env": {
    "SLEEVES_MAX_PROCS": "${SLEEVES_MAX_PROCS}",
    "OMP_NUM_THREADS": "${OMP_NUM_THREADS}",
    "MKL_NUM_THREADS": "${MKL_NUM_THREADS}",
    "NUMEXPR_NUM_THREADS": "${NUMEXPR_NUM_THREADS}"
  },
  "folders": ["preds", "metrics", "figs", "logs"],
  "notes": "All overnight experiments (ablations, norm check, synthetic, robustness) collected."
}
EOF

echo ""
echo "[OK] Overnight bundle at: ${RUNROOT}"
echo "     - preds/: npy arrays"
echo "     - metrics/: csv/json/tex"
echo "     - figs/: pdf/png"
echo "     - logs/: full logs"
