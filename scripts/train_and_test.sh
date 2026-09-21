#!/usr/bin/env bash
# Train one Ultralytics detection model, then evaluate its best checkpoint on the test split.
# Test artifacts are written below the training run directory: <run>/test/ and <run>/test_metrics.log.

set -Eeuo pipefail

if (( $# < 2 )); then
  echo "Usage: $0 MODEL RUN_NAME [additional yolo key=value arguments]" >&2
  exit 2
fi

model="$1"
run_name="$2"
shift 2

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
runs_dir="${RUNS_DIR:-${repo_dir}/runs/detect}"
data_yaml="${DATA_YAML:-/home/tkz/datasets/pingwen_yolo/data.yaml}"
run_dir="${runs_dir}/${run_name}"

if [[ -e "${run_dir}" ]]; then
  echo "Refusing to overwrite existing run directory: ${run_dir}" >&2
  exit 1
fi

common_args=(
  "data=${data_yaml}"
  imgsz=640
  epochs=300
  patience=80
  batch=16
  device=0
  workers=8
  max_det=16
  "project=${runs_dir}"
  "name=${run_name}"
)

cd "${repo_dir}"
yolo detect train "model=${model}" "${common_args[@]}" "$@"

best_checkpoint="${run_dir}/weights/best.pt"
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
  device=0 \
  workers=8 \
  max_det=16 \
  "project=${run_dir}" \
  name=test \
  exist_ok=True \
  2>&1 | tee "${run_dir}/test_metrics.log"
