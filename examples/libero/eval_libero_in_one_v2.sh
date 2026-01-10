#!/usr/bin/env bash
set -euo pipefail

# -------- Defaults (override via flags or env vars) --------
PYTHON_BIN="${PYTHON_BIN:-python}"
EVAL_SCRIPT="${EVAL_SCRIPT:-examples/libero/eval_libero_in_one.py}"

POLICY_CONFIG_DEFAULT="pi05_ours_low_mem_finetune_openvla_libero_pt_v2"
POLICY_ROOT_DEFAULT="/scratch2/whwjdqls99/pi/pi05_ours_low_mem_finetune_openvla_libero_pt_v2/pt_v2_1.0"

# Space-separated lists
CHECKPOINTS_DEFAULT="25000 24000 23000 22000 21000 20000"
SUITES_DEFAULT="libero_goal libero_spatial libero_object libero_10"

NO_SAVE_VIDEOS_DEFAULT=1
JOBS_DEFAULT=1
SEED_DEFAULT=7
usage() {
  cat <<'EOF'
Usage:
  eval_libero_in_one_v2.sh [options]

Options:
  --config NAME            policy.config (default: pi05_ours_low_mem_finetune_openvla_libero)
  --root DIR               root directory containing checkpoint subdirs (default: /scratch2/.../debug)
  --checkpoints "A B C"    checkpoints to evaluate (default: "29999 29000 28000 27000 26000")
  --suites "S1 S2"         task suites (default: "libero_goal libero_spatial libero_object libero_10")
  --out-subdir NAME        output dir name inside each checkpoint dir (default: eval_outputs)
  --save-videos            do not pass --args.no-save-videos
  --jobs N                 run up to N evals in parallel (default: 1)
  -h, --help               show help

Environment overrides:
  PYTHON_BIN, EVAL_SCRIPT

Examples:
  ./eval_libero_in_one_v2.sh
  ./eval_libero_in_one_v2.sh --checkpoints "29999 30000" --jobs 2
  ./eval_libero_in_one_v2.sh --root /scratch2/.../debug --config my_cfg --suites "libero_goal"
EOF
}

# -------- Parse args --------
POLICY_CONFIG="${POLICY_CONFIG_DEFAULT}"
POLICY_ROOT="${POLICY_ROOT_DEFAULT}"
CHECKPOINTS="${CHECKPOINTS_DEFAULT}"
SUITES="${SUITES_DEFAULT}"
SEED="${SEED_DEFAULT}"
OUT_SUBDIR="eval_outputs_${SEED_DEFAULT}"
NO_SAVE_VIDEOS="${NO_SAVE_VIDEOS_DEFAULT}"
JOBS="${JOBS_DEFAULT}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) POLICY_CONFIG="$2"; shift 2;;
    --root) POLICY_ROOT="$2"; shift 2;;
    --checkpoints) CHECKPOINTS="$2"; shift 2;;
    --suites) SUITES="$2"; shift 2;;
    --out-subdir) OUT_SUBDIR="$2"; shift 2;;
    --save-videos) NO_SAVE_VIDEOS=0; shift 1;;
    --jobs) JOBS="$2"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown arg: $1"; echo; usage; exit 2;;
  esac
done

# -------- Build commands --------
no_save_flag=()
if [[ "${NO_SAVE_VIDEOS}" == "1" ]]; then
  no_save_flag=(--args.no-save-videos)
fi

run_one() {
  local ckpt="$1"
  local suite="$2"
  local policy_dir="${POLICY_ROOT}/${ckpt}"
  local out_dir="${policy_dir}/${OUT_SUBDIR}"

  if [[ ! -d "${policy_dir}" ]]; then
    echo "[SKIP] Missing policy dir: ${policy_dir}" >&2
    return 0
  fi

  mkdir -p "${out_dir}"

  echo "[RUN] ckpt=${ckpt} suite=${suite}"
  "${PYTHON_BIN}" "${EVAL_SCRIPT}" \
    --policy.config="${POLICY_CONFIG}" \
    --policy.dir="${policy_dir}" \
    "${no_save_flag[@]}" \
    --args.out_dir "${out_dir}" \
    --args.task_suite_name "${suite}" \
    --args.seed "${SEED}"
}

export -f run_one
export PYTHON_BIN EVAL_SCRIPT POLICY_CONFIG POLICY_ROOT OUT_SUBDIR
export NO_SAVE_VIDEOS
# no_save_flag can't be exported as array; recompute inside subshell calls if parallelized via bash -lc.

# -------- Execute (serial or parallel) --------
if [[ "${JOBS}" -le 1 ]]; then
  for ckpt in ${CHECKPOINTS}; do
    for suite in ${SUITES}; do
      run_one "${ckpt}" "${suite}"
    done
  done
else
  # Parallel via xargs; uses bash -lc to preserve function + env.
  # Note: recompute no-save flag inside the subshell.
  printf "%s\n" ${CHECKPOINTS} | while read -r ckpt; do
    for suite in ${SUITES}; do
      printf "%s\t%s\n" "${ckpt}" "${suite}"
    done
  done | xargs -P "${JOBS}" -n 1 -I {} bash -lc '
    set -euo pipefail
    ckpt="$(printf "%s" "{}" | cut -f1)"
    suite="$(printf "%s" "{}" | cut -f2)"
    no_save_flag=()
    if [[ "${NO_SAVE_VIDEOS}" == "1" ]]; then no_save_flag=(--args.no-save-videos); fi
    policy_dir="${POLICY_ROOT}/${ckpt}"
    out_dir="${policy_dir}/${OUT_SUBDIR}"
    if [[ ! -d "${policy_dir}" ]]; then
      echo "[SKIP] Missing policy dir: ${policy_dir}" >&2
      exit 0
    fi
    mkdir -p "${out_dir}"
    echo "[RUN] ckpt=${ckpt} suite=${suite}"
    "${PYTHON_BIN}" "${EVAL_SCRIPT}" \
      --policy.config="${POLICY_CONFIG}" \
      --policy.dir="${policy_dir}" \
      "${no_save_flag[@]}" \
      --args.out_dir "${out_dir}" \
      --args.task_suite_name "${suite}"\
      --args.seed "${SEED}"
  '
fi