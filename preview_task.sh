#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 || $# -gt 5 ]]; then
    echo "Usage: bash preview_task.sh TASK CONFIG GPU_ID [BREAD_COUNT|random] [SEED]" >&2
    exit 2
fi

task_name=$1
task_config=$2
gpu_id=$3
bread_count=${4:-random}
seed=${5:-0}

if [[ ! $gpu_id =~ ^[0-9]+$ ]]; then
    echo "GPU_ID must be a non-negative integer" >&2
    exit 2
fi
if [[ $bread_count != random && ! $bread_count =~ ^[123]$ ]]; then
    echo "BREAD_COUNT must be 1, 2, 3, or random" >&2
    exit 2
fi
if [[ ! $seed =~ ^[0-9]+$ ]]; then
    echo "SEED must be a non-negative integer" >&2
    exit 2
fi

cd "$(dirname "${BASH_SOURCE[0]}")"
export CUDA_VISIBLE_DEVICES=$gpu_id

preview_args=("$task_name" "$task_config" --seed "$seed")
if [[ $bread_count != random ]]; then
    preview_args+=(--bread-count "$bread_count")
fi

PYTHONWARNINGS=ignore::UserWarning python scripts/preview_task.py "${preview_args[@]}"
