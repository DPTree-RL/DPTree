<h1 align="center" style="margin-top: 10px;">Tree-Based Reinforcement Learning for Credit-Precise Data Agents</h1>

## Quick Start

### Installation

#### RL Environment

```bash
conda create -n adtreenew python=3.12
conda activate adtreenew
pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu126
pip install transformers==5.5.3 ray[default]==2.56.1 vllm==0.19.1
pip install flash-attn==2.8.3.post1 --no-build-isolation
pip install flash-linear-attention
pip install causal-conv1d --no-build-isolation
pip install tilelang
# verl
pip install -e .
pip install swanlab==0.9.3
```

### DB Environment

The SQL agent's tool calls (`get_table_schema`, `execute_sql`, `submit_sql`, ...) are served by a separate FastAPI microservice (`search_r1/db_environment/`) that clones a per-task PostgreSQL database from a LiveSQLBench template for every task in the current training/validation batch. Keep its dependencies in their own env, isolated from the RL training env above:

```bash
conda create -n db_environment python=3.10
conda activate db_environment
pip install fastapi uvicorn psycopg2-binary
```

This service assumes a reachable PostgreSQL instance that already has one `<db_name>_template` database (`datistemplate=true`) per LiveSQLBench database, built ahead of time from LiveSQLBench's own `evaluation/env/init-databases_postgresql*.sh` scripts (see the [LiveSQLBench-Agent](https://livesqlbench.github.io/) project) -- this service only ever clones from those templates via `createdb --template`, it never builds one from scratch.

#### Building the Postgres templates

First, clone the LiveSQLBench repo itself (a separate checkout from this one -- it's where the `evaluation/env/init-databases_postgresql*.sh` scripts and Docker build context live):

```bash
git clone https://github.com/bird-bench/livesqlbench.git
cd livesqlbench/evaluation
```

`init-databases_postgresql.sh` (18 lite dbs) and `init-databases_postgresql_base_full.sh` (22 full dbs) are meant to run as Postgres `docker-entrypoint-initdb.d` scripts against the official prebuilt images -- the table-dump `.sql` files they import aren't in this git checkout, only baked into `docker.io/shawnxxh/bird-interact-postgresql:latest` (lite) and `docker.io/shawnxxh/bird-interact-postgresql-full:latest` (full). The db_environment service needs **one** Postgres instance serving both splits (it only takes a single `--pg_host`/`--pg_port`), and since lite's 18 db names and full's 22 db names never collide, both splits' templates can be merged into that one instance. Run the following from inside `livesqlbench/evaluation/`:

```bash
# 1. Pull the two official images and extract their baked-in table dumps
#    (docker create doesn't run the entrypoint, so this is safe)
docker pull shawnxxh/bird-interact-postgresql:latest
docker pull shawnxxh/bird-interact-postgresql-full:latest

docker create --name tmp_lite shawnxxh/bird-interact-postgresql:latest
docker cp tmp_lite:/docker-entrypoint-initdb.d/postgre_table_dumps ./postgre_table_dumps_lite
docker rm tmp_lite

docker create --name tmp_full shawnxxh/bird-interact-postgresql-full:latest
docker cp tmp_full:/docker-entrypoint-initdb.d/postgre_table_dumps ./postgre_table_dumps_full
docker rm tmp_full

# 2. Merge the dump dirs (no name collisions between splits)
mkdir -p ./postgre_table_dumps_merged
cp -r ./postgre_table_dumps_lite/* ./postgre_table_dumps_merged/
cp -r ./postgre_table_dumps_full/*  ./postgre_table_dumps_merged/

# 3. Stage both init scripts (from env/, cloned above) to run in order at first container boot
mkdir -p ./initdb
cp ./env/init-databases_postgresql.sh           ./initdb/01-init-lite.sh
cp ./env/init-databases_postgresql_base_full.sh ./initdb/02-init-full.sh

# 4. Start one fresh Postgres 14 instance with everything mounted
docker volume create livesqlbench_pg_data
docker run -d --name livesqlbench_postgres \
  -e POSTGRES_USER=root -e POSTGRES_PASSWORD=123123 \
  -p 5432:5432 \
  -v livesqlbench_pg_data:/var/lib/postgresql/data \
  -v $(pwd)/postgre_table_dumps_merged:/docker-entrypoint-initdb.d/postgre_table_dumps \
  -v $(pwd)/initdb/01-init-lite.sh:/docker-entrypoint-initdb.d/01-init-lite.sh \
  -v $(pwd)/initdb/02-init-full.sh:/docker-entrypoint-initdb.d/02-init-full.sh \
  postgres:14.12
```

Postgres's official image entrypoint runs every `/docker-entrypoint-initdb.d/*.sh` in lexicographic order, but only on first boot (when the data dir is still empty) -- the two scripts create/import/`datistemplate=true` all 40 template dbs into this one instance. Watch `docker logs -f livesqlbench_postgres` for two "Marking these template databases as 'datistemplate = true'..." lines with no import errors, then verify:

```bash
PGPASSWORD=123123 psql -h 127.0.0.1 -U root -d postgres \
  -c "SELECT count(*) FROM pg_database WHERE datistemplate=true AND datname LIKE '%_template';"
# -> 40
```

This only needs to be done once (the named volume persists across restarts). Point `local_db_environment_launch.sh`'s `pg_host`/`pg_port` at wherever this container ends up running.

### Dataset

Download the LiveSQLBench dataset (`livesqlbench-base-full` for training, `livesqlbench-base-lite` for validation) to the directory the training scripts expect (`DATA_DIR`). Each split's directory must contain `livesqlbench_data.jsonl` plus one folder per database (schema/column-meaning/knowledge-base files).

Process both splits into parquet:

_Modify `DATA_DIR` in the script first._

```bash
bash scripts/data_process/data_process_livesqlbench.sh
```

This writes `$DATA_DIR/train.parquet` (600 tasks) and `$DATA_DIR/test.parquet` (270 tasks).

### RL Training

Launch the DB environment server:

_Modify `dataset_root` in `local_db_environment_launch.sh` first._

```bash
conda activate db_environment
bash local_db_environment_launch.sh
```

Run one of the four SQL-agent training methods with Qwen3.5-9B on LiveSQLBench (each uses the same tree-search infrastructure with different m/n/l, advantage estimator, policy loss, and reward aggregation -- see `CLAUDE.md`):

```bash
conda activate adtreenew
bash train_livesqlbench_adtree.sh   # AutoDenseTree: m=8, n=8, l=7, tree reward + turn-level policy loss
bash train_livesqlbench_atpo.sh     # ATPO: same m/n/l, softmax-aggregated reward, no leaf-count downweighting
bash train_livesqlbench_grpo.sh     # GRPO baseline: m=56, n=1, l=0 (no branching)
bash train_livesqlbench_gspo.sh     # GSPO baseline: same as GRPO but with sequence-level policy loss
```

## Acknowledgement

The codebase is built upon [Tree-GRPO](https://github.com/AMAP-ML/Tree-GRPO), and based on [Search-R1](https://github.com/PeterGriffinJin/Search-R1) and [veRL](https://github.com/volcengine/verl).
We express our gratitude to these open-source projects.
