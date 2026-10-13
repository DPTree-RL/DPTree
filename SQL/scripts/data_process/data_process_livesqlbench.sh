#!/bin/bash
# Build train.parquet (from livesqlbench-base-full) and test.parquet (from
# livesqlbench-base-lite) under $DATA_DIR, for the SQL-agent RL training scripts
# (train_livesqlbench_*.sh) to read directly.

set -e

DATA_DIR=${DATA_DIR:-/dataset/wang0702/livesqlbench}
WORK_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)

python3 "$WORK_DIR/scripts/data_process/livesqlbench_merge.py" \
    --dataset_root "$DATA_DIR/livesqlbench-base-full" \
    --local_dir "$DATA_DIR" \
    --output_filename train.parquet \
    --split train \
    --data_source livesqlbench

python3 "$WORK_DIR/scripts/data_process/livesqlbench_merge.py" \
    --dataset_root "$DATA_DIR/livesqlbench-base-lite" \
    --local_dir "$DATA_DIR" \
    --output_filename test.parquet \
    --split test \
    --data_source livesqlbench
