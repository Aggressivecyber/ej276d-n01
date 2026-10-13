#!/usr/bin/env bash
set -euo pipefail
root=/home/rylan/projects/ej276d-n01/window008_20261013
python=/home/rylan/projects/hpge-cpunet-validation-20261003/venv/bin/python
kind=${1:?primary or replay01 is required}
case "$kind" in primary|replay01) ;; *) exit 2 ;; esac
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$root/src"
export TMPDIR="$root/tmp"
export XDG_CACHE_HOME="$root/cache"
export OMP_NUM_THREADS=12
export MKL_NUM_THREADS=12
export OPENBLAS_NUM_THREADS=12
export CUDA_VISIBLE_DEVICES=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
cd "$root"
exec "$python" -B scripts/supervise_window008_evaluation.py --kind "$kind"
