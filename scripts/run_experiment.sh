#!/usr/bin/env bash
# Train an Ultralytics detection model for one or more seeds, then test each best checkpoint.
# Test artifacts are written below the training run directory: <run>/test/ and <run>/test_metrics.log.

set -Eeuo pipefail

usage() {
  echo "Usage: $0 [--device DEVICE] [--seeds 0,1,2 | --seed 0] MODEL RUN_NAME [yolo key=value ...]"
  echo "Without --seed/--seeds: runs/detect/RUN_NAME (legacy layout)."
  echo "With --seed/--seeds:    runs/detect/RUN_NAME/SEED (sequential independent runs)."
  echo "RUNS_DIR and DATA_YAML override the output root and dataset, respectively."
  echo "Existing run directories are never overwritten; the first failure stops the sequence."
  echo "After successful testing, only weights/best.pt is retained; last.pt is deleted."
}

fail() {
  echo "$*" >&2
  exit 2
}

device=0
seeds_arg=""
while (( $# )); do
  case "$1" in
    --device)
      (( $# >= 2 )) && [[ -n "$2" && "$2" != --* ]] || fail "Missing value for --device"
      device="$2"
      shift 2
      ;;
    --seeds|--seed)
      (( $# >= 2 )) && [[ -n "$2" && "$2" != --* ]] || fail "Missing value for $1"
      [[ -z "${seeds_arg}" ]] || fail "Specify --seed or --seeds only once"
      [[ "$1" != --seed || "$2" != *,* ]] || fail "Use --seeds for a comma-separated seed list"
      seeds_arg="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      break
      ;;
    -*) fail "Unknown option: $1. Use --help for usage." ;;
    *) break ;;
  esac
done

if (( $# < 2 )); then
  usage >&2
  exit 2
fi

model="$1"
run_name="$2"
shift 2

[[ -n "${run_name}" && "${run_name}" != /* && "${run_name}" != *\\* \
  && "/${run_name}/" != *"/../"* && "/${run_name}/" != *"/./"* ]] \
  || fail "RUN_NAME must be a relative path without '.' or '..' components"

seeds=()
if [[ -n "${seeds_arg}" ]]; then
  [[ "${seeds_arg}" =~ ^[0-9]+(,[0-9]+)*$ ]] \
    || fail "--seeds expects comma-separated non-negative integers, e.g. 0,1,2"
  IFS=',' read -r -a seeds <<< "${seeds_arg}"
  seen_seeds=","
  for seed in "${seeds[@]}"; do
    [[ "${seed}" =~ ^(0|[1-9][0-9]{0,9})$ ]] && (( seed <= 4294967295 )) \
      || fail "Invalid seed '${seed}': use integers from 0 to 4294967295 without leading zeros"
    [[ "${seen_seeds}" != *",${seed},"* ]] || fail "Duplicate seed: ${seed}"
    seen_seeds+="${seed},"
  done
fi

train_args=("$@")
eval_args=()
for argument in "${train_args[@]}"; do
  [[ "${argument}" == *=* && "${argument}" != --* ]] \
    || fail "Additional YOLO arguments must use key=value syntax: ${argument}"
  case "${argument}" in
    project=*|name=*|model=*|exist_ok=*|resume=*)
      fail "The script manages model/output directories and fresh runs; do not pass ${argument}"
      ;;
    seed=*)
      [[ -z "${seeds_arg}" ]] || fail "Do not combine seed= with --seed/--seeds"
      ;;
    data=*|imgsz=*|batch=*|device=*|workers=*|max_det=*)
      # Keep test settings consistent with any explicit shared training settings.
      eval_args+=("${argument}")
      ;;
  esac
done

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
runs_dir="${RUNS_DIR:-${repo_dir}/runs/detect}"
data_yaml="${DATA_YAML:-/home/tkz/datasets/pingwen_yolo/data.yaml}"

common_args=(
  "data=${data_yaml}"
  imgsz=640
  epochs=300
  patience=80
  batch=16
  "device=${device}"
  workers=8
  max_det=16
)

cd "${repo_dir}"

check_run_dir() {
  if [[ -e "$1" || -L "$1" ]]; then
    echo "Refusing to overwrite existing run directory: $1" >&2
    exit 1
  fi
}

run_one() {
  local project="$1" name="$2" seed="$3"
  local run_dir="${project}/${name}"
  local best_checkpoint="${run_dir}/weights/best.pt"
  local seed_args=()
  [[ -z "${seed}" ]] || seed_args=("seed=${seed}")
  check_run_dir "${run_dir}"
  printf '\nStarting experiment: %s\n' "${run_dir}"
  yolo detect train "model=${model}" "${common_args[@]}" "${train_args[@]}" \
    "project=${project}" "name=${name}" "${seed_args[@]}"

  if [[ ! -f "${best_checkpoint}" ]]; then
    echo "Training completed but best checkpoint was not found: ${best_checkpoint}" >&2
    exit 1
  fi

  yolo detect val \
    "model=${best_checkpoint}" \
    "data=${data_yaml}" \
    split=test \
    imgsz=640 \
    batch=16 \
    "device=${device}" \
    workers=8 \
    max_det=16 \
    "${eval_args[@]}" \
    "project=${run_dir}" \
    name=test \
    exist_ok=True \
    2>&1 | tee "${run_dir}/test_metrics.log"

  find "${run_dir}/weights" -maxdepth 1 -type f -name '*.pt' ! -name 'best.pt' -print -delete \
    | tee "${run_dir}/deleted_checkpoints.log"
}

if [[ -n "${seeds_arg}" ]]; then
  # Check the complete batch before starting an expensive training job.
  for seed in "${seeds[@]}"; do
    check_run_dir "${runs_dir}/${run_name}/${seed}"
  done
  for seed in "${seeds[@]}"; do
    run_one "${runs_dir}/${run_name}" "${seed}" "${seed}"
  done
else
  run_one "${runs_dir}" "${run_name}" ""
fi
