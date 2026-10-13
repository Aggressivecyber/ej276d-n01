#!/usr/bin/env bash
set -euo pipefail
root=/home/rylan/projects/ej276d-n01/response_007_20261013_deadline01
python=/home/rylan/projects/hpge-cpunet-validation-20261003/venv/bin/python
mkdir -p "$root/runs/verification_supervisor" "$root/tmp" "$root/cache"
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
exec "$python" -B scripts/supervise.py --output "$root/runs/verification_supervisor" --rss-limit-gib 8 -- \
 timeout --signal=TERM --kill-after=15s 300s "$python" -B scripts/replay_response007_deadline01.py
