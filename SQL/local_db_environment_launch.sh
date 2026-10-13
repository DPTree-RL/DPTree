#!/bin/bash

echo "db_environment use python:"
which python
# db_utils.py shells out to psql/createdb/dropdb (see _terminate_backends/_drop_and_create_db)
# to reset each task's db -- if this comes up empty, every /init_batch call will fail with
# "[Errno 2] No such file or directory: 'psql'" instead of a clear "not on PATH" message here.
echo "db_environment use psql:"
which psql || echo "  psql NOT found on PATH -- install it (e.g. 'conda install -c conda-forge postgresql' into this env) before submitting"

host=0.0.0.0
port=6002

pg_host=127.0.0.1
pg_port=5432
pg_user=root
pg_password=123123
pg_minconn=1
pg_maxconn=5

dataset_root=/dataset/wang0702/livesqlbench

mkdir -p db_env_log
log_file=db_env_log/db_environment_$(date +%Y%m%d_%H%M%S).log

PYTHONUNBUFFERED=1 python -m search_r1.db_environment.server --host $host \
                                                              --port $port \
                                                              --pg_host $pg_host \
                                                              --pg_port $pg_port \
                                                              --pg_user $pg_user \
                                                              --pg_password $pg_password \
                                                              --pg_minconn $pg_minconn \
                                                              --pg_maxconn $pg_maxconn \
                                                              --dataset_root $dataset_root
                                                              2>&1 | tee $log_file