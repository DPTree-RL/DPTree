"""Postgres DB management and SQL grading utilities.

Ported from LiveSQLBench-Agent's ``shared/db_utils.py`` (see
``/dataset/wang0702/livesqlbench/LiveSQLBench-Agent/shared/db_utils.py``).
The grading/comparison logic (``ex_base``, ``preprocess_results``,
``remove_round``/``remove_distinct``/``remove_comments``) is kept faithful
to the original; only imports and connection-settings plumbing were adapted
to this project's module layout (``search_r1.db_environment.config``
instead of a pydantic ``Settings`` object).

All Postgres connections are lazy: nothing in this module talks to Postgres
at import time, only when a function is actually called.
"""

import hashlib
import json
import logging
import os
import re
import subprocess
import threading
from datetime import date, datetime
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional, Tuple

import psycopg2
from psycopg2 import OperationalError
from psycopg2.pool import ThreadedConnectionPool

from search_r1.db_environment.config import settings

logger = logging.getLogger(__name__)

_postgresql_pools: Dict[str, ThreadedConnectionPool] = {}
_pool_lock = threading.Lock()


def sanitize_task_id(task_id: str) -> str:
    """Sanitize a task_id for use in a Postgres db name: '-' and '.' -> '_'."""
    return task_id.replace("-", "_").replace(".", "_")


# Postgres silently truncates identifiers longer than NAMEDATALEN-1 (63 bytes by
# default). LiveSQLBench task_ids are often "{selected_database}_..." themselves, so
# f"{selected_database}__{task_id}" commonly exceeds that and gets truncated -- two
# different tasks whose names share the same first 63 bytes then collide on the exact
# same physical db, and every createdb after the first fails with "already exists".
_PG_MAX_IDENTIFIER_LEN = 63


def task_db_name(selected_database: str, task_id: str) -> str:
    name = f"{selected_database}__{sanitize_task_id(task_id)}"
    if len(name) <= _PG_MAX_IDENTIFIER_LEN:
        return name
    # Keep a readable (but now non-unique on its own) prefix and disambiguate with a
    # hash of the full name, so distinct task_ids never collide after truncation.
    digest = hashlib.sha1(name.encode()).hexdigest()[:10]
    prefix_len = _PG_MAX_IDENTIFIER_LEN - len(digest) - 1
    return f"{name[:prefix_len]}_{digest}"


def template_db_name(selected_database: str) -> str:
    return f"{selected_database}_template"


# ---------------------------------------------------------------------------
# Connection pooling
# ---------------------------------------------------------------------------

def get_or_init_pool(db_name: str) -> ThreadedConnectionPool:
    with _pool_lock:
        if db_name not in _postgresql_pools:
            _postgresql_pools[db_name] = ThreadedConnectionPool(
                settings.pg_minconn, settings.pg_maxconn,
                dbname=db_name, user=settings.pg_user,
                password=settings.pg_password, host=settings.pg_host,
                port=settings.pg_port,
            )
        return _postgresql_pools[db_name]


# Backwards-compatible alias matching the reference module's private name.
_get_or_init_pool = get_or_init_pool


def close_pool(db_name: str):
    with _pool_lock:
        if db_name in _postgresql_pools:
            pool = _postgresql_pools.pop(db_name)
            pool.closeall()


# ---------------------------------------------------------------------------
# Query execution
# ---------------------------------------------------------------------------

def perform_query(query: str, db_name: str, conn=None):
    """Run a single statement with a 60s statement_timeout.

    Returns (result_rows_or_None, conn, cursor_description). Raises on error
    (including psycopg2.errors.QueryCanceled on timeout); the connection is
    rolled back before the exception propagates.
    """
    MAX_ROWS = 10000
    pool = get_or_init_pool(db_name)
    if conn is None:
        conn = pool.getconn()
    cursor = conn.cursor()
    cursor.execute("SET statement_timeout = '60s';")
    try:
        cursor.execute(query)
        conn.commit()
        lower_q = query.strip().lower()
        if lower_q.startswith("select") or lower_q.startswith("with") or lower_q.startswith("explain"):
            rows = cursor.fetchmany(MAX_ROWS + 1)
            result = rows[:MAX_ROWS]
        else:
            try:
                result = cursor.fetchall()
            except psycopg2.ProgrammingError:
                result = None
        desc = cursor.description
        return result, conn, desc
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        cursor.close()


def execute_queries(queries, db_name: str, conn=None):
    """Execute one or more queries (list of statement strings) sequentially.

    Returns (result, error, timeout, cursor_description) — mirrors the
    reference implementation. Only the result/description of the *last*
    non-empty query is returned.
    """
    if isinstance(queries, str):
        queries = [queries]
    if not queries:
        return None, None, False, None
    result = None
    desc = None
    for query in queries:
        if not query or not query.strip():
            continue
        try:
            result, conn, desc = perform_query(query, db_name, conn=conn)
        except psycopg2.errors.QueryCanceled:
            return None, None, True, None
        except (OperationalError, psycopg2.Error) as e:
            return None, str(e), False, None
        except Exception as e:
            return None, str(e), False, None
    return result, None, False, desc


# ---------------------------------------------------------------------------
# DB (re)creation via psql/createdb/dropdb subprocesses
# ---------------------------------------------------------------------------

def _pg_env() -> Tuple[list, dict]:
    """Return (common_args, env_vars) for psql/createdb/dropdb subprocess calls."""
    env_vars = os.environ.copy()
    env_vars["PGPASSWORD"] = settings.pg_password
    args = ["-h", settings.pg_host, "-p", str(settings.pg_port), "-U", settings.pg_user]
    return args, env_vars


def _terminate_backends(db_name: str, args: list, env_vars: dict):
    subprocess.run(
        ["psql", *args, "-d", "postgres", "-c",
         f"SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
         f"WHERE datname = '{db_name}' AND pid <> pg_backend_pid();"],
        check=True, env=env_vars, timeout=60,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def _drop_and_create_db(db_name: str, template_db: str):
    """Force-terminate backends on db_name, drop it if it exists, then
    recreate it from template_db. Mirrors the reference implementation.
    """
    args, env_vars = _pg_env()
    close_pool(db_name)
    _terminate_backends(db_name, args, env_vars)
    subprocess.run(
        ["dropdb", "--if-exists", *args, db_name],
        check=True, env=env_vars, timeout=60,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    subprocess.run(
        ["createdb", *args, db_name, "--template", template_db],
        check=True, env=env_vars, timeout=60,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def create_task_db(selected_database: str, task_id: str) -> str:
    """Provision a fresh per-task db copy from its template. Returns the
    physical db name."""
    task_db = task_db_name(selected_database, task_id)
    template = template_db_name(selected_database)
    _drop_and_create_db(task_db, template)
    return task_db


def reset_task_db(task_db: str, template_db: str):
    """Reset a per-task db back to a pristine copy of its template."""
    _drop_and_create_db(task_db, template_db)


def drop_task_db(task_db: str):
    """Drop a per-task db (terminating backends first) and close its pool."""
    args, env_vars = _pg_env()
    close_pool(task_db)
    _terminate_backends(task_db, args, env_vars)
    subprocess.run(
        ["dropdb", "--if-exists", *args, task_db],
        check=True, env=env_vars, timeout=60,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


# ---------------------------------------------------------------------------
# Result post-processing / grading (ported verbatim from the reference)
# ---------------------------------------------------------------------------

def process_decimals_recursive(item, decimal_places: int):
    quantizer = Decimal(1).scaleb(-decimal_places)
    if isinstance(item, Decimal):
        return item.quantize(quantizer, rounding=ROUND_HALF_UP)
    elif isinstance(item, float):
        return round(item, decimal_places)
    elif isinstance(item, (list, tuple)):
        return type(item)(process_decimals_recursive(x, decimal_places) for x in item)
    elif isinstance(item, dict):
        return {k: process_decimals_recursive(v, decimal_places) for k, v in item.items()}
    return item


def preprocess_results(results, decimal_places: int = 2):
    if results is None:
        return []
    processed = []
    for row in results:
        processed_row = []
        for item in row:
            if isinstance(item, (date, datetime)):
                processed_row.append(item.strftime("%Y-%m-%d"))
            else:
                pi = process_decimals_recursive(item, decimal_places)
                if isinstance(pi, (dict, list)):
                    processed_row.append(json.dumps(pi, sort_keys=True))
                else:
                    processed_row.append(pi)
        processed.append(tuple(processed_row))
    return processed


def remove_comments(sql_list: List[str]) -> List[str]:
    cleaned = []
    for sql in sql_list:
        no_block = re.sub(r"/\*.*?\*/", "", sql, flags=re.DOTALL)
        no_line = re.sub(r"--.*?(\r\n|\r|\n)", r"\1", no_block)
        no_blank = re.sub(r"\n\s*\n+", "\n", no_line)
        cleaned.append(no_blank.strip())
    return cleaned


def remove_distinct(sql_list: List[str]) -> List[str]:
    return [" ".join(t for t in q.split(" ") if t.lower() != "distinct") for q in sql_list]


def _remove_round_functions(sql_string: str) -> str:
    def find_matching_paren(text, start):
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "(":
                depth += 1
            elif text[i] == ")":
                depth -= 1
                if depth == 0:
                    return i
        return -1

    def find_first_arg_end(text, start):
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "(":
                depth += 1
            elif text[i] == ")":
                if depth == 0:
                    return i
                depth -= 1
            elif text[i] == "," and depth == 0:
                return i
        return len(text)

    result = sql_string
    while True:
        match = re.search(r"ROUND\s*\(", result, re.IGNORECASE)
        if not match:
            break
        start = match.start()
        open_p = match.end() - 1
        first_end = find_first_arg_end(result, open_p + 1)
        close_p = find_matching_paren(result, open_p)
        if close_p == -1:
            break
        first_arg = result[open_p + 1: first_end].strip()
        result = result[:start] + first_arg + result[close_p + 1:]
    return result


def remove_round(sql_list: List[str]) -> List[str]:
    return [_remove_round_functions(sql) for sql in sql_list]


def ex_base(pred_sqls, sol_sqls, db_name, conn, conditions=None) -> int:
    """Soft execution-accuracy grader: 1 if pred/sol result sets match, else 0."""
    if not pred_sqls or not sol_sqls:
        return 0
    pred_res, pred_err, pred_to, _ = execute_queries(pred_sqls, db_name, conn)
    gt_res, gt_err, gt_to, _ = execute_queries(sol_sqls, db_name, conn)
    if any([pred_err, pred_to, gt_err, gt_to]):
        return 0
    pred_res = preprocess_results(pred_res)
    gt_res = preprocess_results(gt_res)
    if not pred_res or not gt_res:
        return 0
    if conditions and conditions.get("order", False):
        return 1 if pred_res == gt_res else 0
    return 1 if set(pred_res) == set(gt_res) else 0


def test_case_default(pred_sqls, sol_sqls, db_name, conn, conditions=None):
    """Default 'Query' category test case: preprocess both sides then ex_base."""
    pred_sqls = remove_round(remove_distinct(remove_comments(pred_sqls)))
    sol_sqls = remove_round(remove_distinct(remove_comments(sol_sqls)))
    result = ex_base(pred_sqls, sol_sqls, db_name, conn, conditions)
    assert result == 1, f"ex_base returned {result} but expected 1."
    return result
