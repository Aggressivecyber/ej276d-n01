#!/usr/bin/env bash
set -euo pipefail
root=/home/rylan/projects/ej276d-n01/repair_002_20261012
python=/home/rylan/projects/hpge-cpunet-validation-20261003/venv/bin/python
phase=${1:?noise or all}
mode=${2:?engineering or science}
case "$phase" in noise|all|responses) ;; *) exit 2 ;; esac
case "$mode" in engineering|science) ;; *) exit 2 ;; esac
output="$root/runs/$mode"
mkdir -p "$output" "$root/tmp" "$root/cache"
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
extra=()
if [[ "$mode" == engineering ]]; then extra+=(--engineering); fi
if [[ "${3:-}" == resume ]]; then extra+=(--resume); fi
exec "$python" -B scripts/supervise.py --output "$output" --rss-limit-gib 48 -- \
 "$python" -B -m n01.train_repair --phase "$phase" --device cuda \
 --container /home/rylan/projects/ej276d-sim2real-two-papers-20261008-0305/R33_REMOTE_20261009_175239_UTC/inputs/TRAIN_PRIVATE.npz \
 --identities "$root/data/PARENT_IDENTITIES.npz" --config "$root/repair_run_config.json" --output "$output" "${extra[@]}"
