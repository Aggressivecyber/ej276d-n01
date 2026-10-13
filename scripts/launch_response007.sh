#!/usr/bin/env bash
set -euo pipefail
root=/home/rylan/projects/ej276d-n01/response_007_20261013
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
 timeout --signal=TERM --kill-after=15s 28800s "$python" -B -m n01.train_response007 \
 --container /home/rylan/projects/ej276d-sim2real-two-papers-20261008-0305/R33_REMOTE_20261009_175239_UTC/inputs/TRAIN_PRIVATE.npz \
 --identities "$root/data/PARENT_IDENTITIES.npz" --config "$root/response007_config.json" \
 --source-manifest "$root/RESPONSE007_SOURCE_MANIFEST.json" --out "$root/runs/science" --device cuda:0
