#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python_bin="${CATPOSE_PYTHON:-.venv/bin/python}"
config="${CATPOSE_CONFIG:-configs/rtx2080.yaml}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
case "${1:-all}" in
  prepare)
    "$python_bin" -m catpose.prepare --data data/AP-10K --split 1 --species cat --archive data/ap-10k.zip --out manifests
    ;;
  pilot)
    "$python_bin" -m catpose.check --config "$config" --overfit-images 4
    "$python_bin" -m catpose.profile --config "$config" --warmup 20 --steps 100
    "$python_bin" -m catpose.run_matrix --config "$config" --sizes 100 --seeds 11 --resume
    "$python_bin" -m catpose.preview
    ;;
  matrix)
    "$python_bin" -m catpose.run_matrix --config "$config" --sizes 100 25 50 all --inits imagenet coco_pose --seeds 11 22 33 --resume
    ;;
  results)
    if [[ ! -f results/test_lock.json ]]; then
      "$python_bin" -m catpose.evaluate --lock
    fi
    "$python_bin" -m catpose.evaluate --runs runs --split test --bootstrap 2000 --out results
    "$python_bin" -m catpose.demo --runs runs --size 100 --seed 11 --storyboard demo/storyboard.json --out demo/cat_pose_demo.mp4
    ;;
  all)
    "$0" prepare
    "$0" pilot
    "$0" matrix
    "$0" results
    ;;
  *)
    printf 'Usage: %s [prepare|pilot|matrix|results|all]\n' "$0" >&2
    exit 2
    ;;
esac
