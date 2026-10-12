#!/usr/bin/env bash
set -euo pipefail
root=/home/rylan/projects/ej276d-n01/noise_origin_audit_003_20261012
python=/home/rylan/projects/hpge-cpunet-validation-20261003/venv/bin/python
mkdir -p "$root/runs/supervisor" "$root/tmp" "$root/cache"
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
exec "$python" -B scripts/supervise.py --output "$root/runs/supervisor" --rss-limit-gib 8 -- \
  "$python" -B scripts/run_noise_origin_003.py
