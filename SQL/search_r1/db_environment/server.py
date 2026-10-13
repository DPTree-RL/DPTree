"""DB Environment microservice for the SQL-agent RL task (LiveSQLBench).

Replaces ``search_r1/search/retrieval_server.py``'s role in the training
loop: an RL rollout worker calls this service over HTTP to run/submit SQL
against per-task Postgres databases cloned from LiveSQLBench templates.

Design notes (see the task contract this was built against):
  - In-memory per-task state only: ``_tasks[task_id] -> {...}``. Nothing is
    persisted across process restarts; ``/init_batch`` must be called again
    after a restart.
  - All blocking Postgres/subprocess work runs via ``asyncio.to_thread`` so
    the FastAPI event loop is never blocked by a slow query or a
    createdb/dropdb subprocess call.
  - Each task has its own ``threading.Lock``. ``/execute`` and ``/submit``
    acquire it (blocking, inside the thread-pool thread) so that concurrent
    requests for the *same* task_id are serialized — many RL rollout
    branches can share one logical task_id and run concurrently, but must
    never race against each other on the shared physical DB, especially
    since ``/submit`` resets the whole database. Different task_ids run
    fully in parallel (separate locks, separate physical DBs).
  - Read-only metadata endpoints (/table_names, /table_schema, ...) never
    touch Postgres; they just read (and cache) files under
    ``<dataset_root>/<split>/<selected_database>/`` (``--dataset_root``,
    searched across both LiveSQLBench split dirs) for the task's db, so they
    don't need the per-task lock.

Infra prerequisites (out of scope for this service):
  - For every ``selected_database`` used, a Postgres database named
    ``<selected_database>_template`` (``datistemplate=true``) must already
    exist in the target Postgres instance, created ahead of time by
    LiveSQLBench's own ``init-databases_postgresql*.sh`` scripts. This
    service only ever clones from those templates via ``createdb
    --template``; it never builds a template from scratch.
  - ``--dataset_root`` must point to a local directory containing the
    LiveSQLBench split dirs (``livesqlbench-base-full/``,
    ``livesqlbench-base-lite/``), e.g. the same path as ``$DATA_DIR`` in the
    training scripts. This service reads schema/column-meaning/knowledge
    files straight off disk, so it needs its own local (or mounted) copy --
    it does not fetch them from the training process.
"""

import argparse
import asyncio
import json
import logging
import os
import re
import threading
from typing import Any, Dict, List, Optional

import psycopg2
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from search_r1.db_environment import db_utils
from search_r1.db_environment.config import settings

logger = logging.getLogger("db_environment")

app = FastAPI(title="AutoDenseTree DB Environment")

# ---------------------------------------------------------------------------
# In-memory state
# ---------------------------------------------------------------------------

# task_id -> {selected_database, task_db, sol_sql, test_cases,
#             category, conditions, lock}
_tasks: Dict[str, Dict[str, Any]] = {}

# LiveSQLBench split dirs to search under --dataset_root for a given
# selected_database's files. selected_database names never collide between
# splits, so trying both in order is unambiguous.
_SPLIT_DIRS = ("livesqlbench-base-full", "livesqlbench-base-lite")

# selected_database -> file contents / parsed structures
_schema_cache: Dict[str, str] = {}
_column_meaning_cache: Dict[str, Dict[str, str]] = {}
_kb_cache: Dict[str, List[Dict[str, Any]]] = {}
# selected_database -> {table_name: "CREATE TABLE ... sample rows"} (lazily split from
# _schema_cache's full text the first time any table of that db is requested)
_table_schema_cache: Dict[str, Dict[str, str]] = {}

MAX_RESULT_ROWS = 100
MAX_RESULT_WORDS = 500
MAX_RESULT_CHARS = 4000


# ---------------------------------------------------------------------------
# Pydantic request models
# ---------------------------------------------------------------------------

class TaskInit(BaseModel):
    task_id: str
    selected_database: str
    sol_sql: List[str] = Field(default_factory=list)
    test_cases: List[str] = Field(default_factory=list)
    category: str = "Query"
    conditions: Dict[str, Any] = Field(default_factory=dict)


class InitBatchRequest(BaseModel):
    tasks: List[TaskInit]


class CleanupBatchRequest(BaseModel):
    task_ids: List[str]


class ColumnMeaningRequest(BaseModel):
    task_id: str
    table_name: str
    column_name: str


class TableSchemaRequest(BaseModel):
    task_id: str
    table_name: str


class KnowledgeRequest(BaseModel):
    task_id: str
    knowledge_name: str


class ExecuteRequest(BaseModel):
    task_id: str
    sql: str


class SubmitRequest(BaseModel):
    task_id: str
    sql: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _task_not_found(task_id: str) -> JSONResponse:
    return JSONResponse(status_code=404, content={"error": f"task {task_id} not initialized"})


def _server_error(e: Exception) -> JSONResponse:
    """Log the actual exception server-side (uvicorn's access log only ever shows the status
    code, so a caught-and-converted-to-500 error is otherwise silently lost) before returning it
    to the caller."""
    logger.exception("metadata endpoint failed: %s", e)
    return JSONResponse(status_code=500, content={"error": str(e)})


def _db_dir(db: str) -> str:
    """Resolve <settings.dataset_root>/<split>/<db>/ by trying each known
    LiveSQLBench split dir in turn (selected_database names never collide
    across splits, so this is unambiguous)."""
    for split in _SPLIT_DIRS:
        candidate = os.path.join(settings.dataset_root, split, db)
        if os.path.isdir(candidate):
            return candidate
    raise FileNotFoundError(
        f"no directory for db '{db}' under dataset_root={settings.dataset_root!r} "
        f"(tried splits: {_SPLIT_DIRS})"
    )


def _load_schema(db: str) -> str:
    if db not in _schema_cache:
        path = os.path.join(_db_dir(db), f"{db}_schema.txt")
        with open(path, "r") as f:
            _schema_cache[db] = f.read()
    return _schema_cache[db]



# Most *_schema.txt files write 'CREATE TABLE "name"', but a handful of LiveSQLBench databases
# (cross_border, crypto_exchange, cybermarket_pattern, insider_trading, museum_artifact,
# polar_equipment -- confirmed by inspecting the actual files) instead write
# '"CREATE" TABLE "name"', with CREATE itself quoted. Both must match or those six dbs'
# get_table_schema silently returns nothing for those tables (and, at data-processing time,
# scripts/data_process/livesqlbench_merge.py's own copy of this regex would list none of them).
_TABLE_HEADER_RE = re.compile(r'^"?CREATE"? TABLE "([^"]+)"', re.MULTILINE)


def _load_table_schemas(db: str) -> Dict[str, str]:
    """Split a db's full schema text into one CREATE-TABLE-plus-sample-rows block per table,
    keyed by table name. Table blocks start at each 'CREATE TABLE "name"' line and run up to
    (not including) the next one, matching LiveSQLBench's own *_schema.txt layout."""
    if db not in _table_schema_cache:
        text = _load_schema(db)
        matches = list(_TABLE_HEADER_RE.finditer(text))
        blocks: Dict[str, str] = {}
        for i, m in enumerate(matches):
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            blocks[m.group(1)] = text[m.start():end].rstrip()
        _table_schema_cache[db] = blocks
    return _table_schema_cache[db]


def _load_column_meanings(db: str) -> Dict[str, str]:
    if db not in _column_meaning_cache:
        path = os.path.join(_db_dir(db), f"{db}_column_meaning_base.json")
        with open(path, "r") as f:
            raw = json.load(f)
        # Lower-case keys, matching the reference LiveSQLBench server's
        # _load_db_data loading convention.
        _column_meaning_cache[db] = {k.lower(): v for k, v in raw.items()}
    return _column_meaning_cache[db]


def _load_kb(db: str) -> List[Dict[str, Any]]:
    if db not in _kb_cache:
        path = os.path.join(_db_dir(db), f"{db}_kb.jsonl")
        entries: List[Dict[str, Any]] = []
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entries.append(json.loads(line))
        _kb_cache[db] = entries
    return _kb_cache[db]


_COMMENT_BLOCK_RE = re.compile(r"/\*.*?\*/", re.DOTALL)
_COMMENT_LINE_RE = re.compile(r"--.*?(?=\r\n|\r|\n|$)")


def _strip_sql_comments(sql: str) -> str:
    no_block = _COMMENT_BLOCK_RE.sub("", sql)
    no_line = _COMMENT_LINE_RE.sub("", no_block)
    return no_line


def _format_result(result, desc) -> str:
    if result is None:
        return "Query executed successfully."
    if not isinstance(result, list):
        return str(result)
    if not result:
        return "Query executed, empty result set."

    lines: List[str] = []
    if desc:
        cols = [d[0] for d in desc]
        header = " | ".join(cols)
        lines.append(header)
        lines.append("-" * min(len(header), 200))

    truncated = len(result) > MAX_RESULT_ROWS
    for row in result[:MAX_RESULT_ROWS]:
        cells = [str(c)[:200] for c in row]
        lines.append(" | ".join(cells))

    text = "\n".join(lines)
    words = text.split()
    if len(words) > MAX_RESULT_WORDS:
        text = " ".join(words[:MAX_RESULT_WORDS])
        truncated = True
    if len(text) > MAX_RESULT_CHARS:
        text = text[:MAX_RESULT_CHARS]
        truncated = True
    if truncated:
        text += "\n... [truncated]"
    return text


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# /init_batch, /cleanup_batch
# ---------------------------------------------------------------------------

def _provision_task_sync(task: TaskInit) -> Optional[str]:
    """Blocking provisioning for one task. Returns an error string, or None
    on success."""
    task_db = db_utils.task_db_name(task.selected_database, task.task_id)
    template = db_utils.template_db_name(task.selected_database)
    try:
        db_utils._drop_and_create_db(task_db, template)
    except Exception as e:
        return str(e)
    return None


async def _init_one_task(task: TaskInit):
    err = await asyncio.to_thread(_provision_task_sync, task)
    if err is not None:
        return task.task_id, err
    task_db = db_utils.task_db_name(task.selected_database, task.task_id)
    # Dict assignment happens back on the event loop thread (single
    # threaded), so this is safe without an extra lock even though many
    # of these coroutines are gathered concurrently.
    _tasks[task.task_id] = {
        "selected_database": task.selected_database,
        "task_db": task_db,
        "sol_sql": task.sol_sql,
        "test_cases": task.test_cases,
        "category": task.category,
        "conditions": task.conditions,
        "lock": threading.Lock(),
    }
    return task.task_id, None


@app.post("/init_batch")
async def init_batch(req: InitBatchRequest):
    results = await asyncio.gather(*[_init_one_task(t) for t in req.tasks])
    initialized: List[str] = []
    errors: Dict[str, str] = {}
    for task_id, err in results:
        if err is None:
            initialized.append(task_id)
        else:
            errors[task_id] = err
    return {"initialized": initialized, "errors": errors}


async def _cleanup_one_task(task_id: str):
    td = _tasks.get(task_id)
    if td is None:
        return None  # unknown task_id: skip silently
    task_db = td["task_db"]
    try:
        await asyncio.to_thread(db_utils.drop_task_db, task_db)
    except Exception as e:
        return task_id, str(e)
    _tasks.pop(task_id, None)
    return task_id, None


@app.post("/cleanup_batch")
async def cleanup_batch(req: CleanupBatchRequest):
    results = await asyncio.gather(*[_cleanup_one_task(tid) for tid in req.task_ids])
    cleaned: List[str] = []
    errors: Dict[str, str] = {}
    for r in results:
        if r is None:
            continue
        task_id, err = r
        if err is None:
            cleaned.append(task_id)
        else:
            errors[task_id] = err
    return {"cleaned": cleaned, "errors": errors}


# ---------------------------------------------------------------------------
# Read-only metadata endpoints
# ---------------------------------------------------------------------------

@app.post("/table_schema")
async def table_schema(req: TableSchemaRequest):
    td = _tasks.get(req.task_id)
    if td is None:
        return _task_not_found(req.task_id)
    try:
        blocks = _load_table_schemas(td["selected_database"])
    except Exception as e:
        return _server_error(e)
    if req.table_name in blocks:
        return {"schema": blocks[req.table_name]}
    return {"schema": None, "error": "not found"}


@app.post("/column_meaning")
async def column_meaning(req: ColumnMeaningRequest):
    td = _tasks.get(req.task_id)
    if td is None:
        return _task_not_found(req.task_id)
    db = td["selected_database"]
    try:
        meanings = _load_column_meanings(db)
    except Exception as e:
        return _server_error(e)
    key = f"{db}|{req.table_name}|{req.column_name}".lower()
    if key in meanings:
        return {"meaning": meanings[key]}
    return {"meaning": None, "error": "not found"}


@app.post("/knowledge")
async def knowledge(req: KnowledgeRequest):
    td = _tasks.get(req.task_id)
    if td is None:
        return _task_not_found(req.task_id)
    try:
        kb = _load_kb(td["selected_database"])
    except Exception as e:
        return _server_error(e)
    for entry in kb:
        if isinstance(entry, dict) and entry.get("knowledge") == req.knowledge_name:
            return {"knowledge": entry}
    return {"knowledge": None, "error": "not found"}


# ---------------------------------------------------------------------------
# /execute — read-only exploration tool
# ---------------------------------------------------------------------------

def _execute_sql_sync(td: Dict[str, Any], sql: str) -> Dict[str, Any]:
    lock: threading.Lock = td["lock"]
    task_db = td["task_db"]
    lock.acquire()
    try:
        pool = db_utils.get_or_init_pool(task_db)
        conn = pool.getconn()
        try:
            try:
                result, conn, desc = db_utils.perform_query(sql, task_db, conn=conn)
            except psycopg2.errors.QueryCanceled as e:
                return {"success": False, "status": "sql_timeout", "error": str(e)}
            except psycopg2.errors.SyntaxError as e:
                return {"success": False, "status": "sql_parse_error", "error": str(e)}
            except psycopg2.ProgrammingError as e:
                pgcode = getattr(e, "pgcode", None) or ""
                if pgcode.startswith("42"):
                    return {"success": False, "status": "sql_parse_error", "error": str(e)}
                return {"success": False, "status": "sql_execution_error", "error": str(e)}
            except psycopg2.Error as e:
                msg = str(e)
                if "statement timeout" in msg.lower():
                    return {"success": False, "status": "sql_timeout", "error": msg}
                return {"success": False, "status": "sql_execution_error", "error": msg}
            else:
                formatted = _format_result(result, desc)
                return {"success": True, "status": "ok", "result": formatted}
        finally:
            try:
                conn.rollback()
            except Exception:
                pass
            try:
                pool.putconn(conn)
            except Exception:
                pass
    finally:
        lock.release()


@app.post("/execute")
async def execute(req: ExecuteRequest):
    td = _tasks.get(req.task_id)
    if td is None:
        return _task_not_found(req.task_id)

    cleaned = _strip_sql_comments(req.sql).strip()
    if not cleaned.upper().startswith(("SELECT", "WITH", "EXPLAIN")):
        return {
            "success": False,
            "status": "unsafe_sql",
            "error": (
                "Only SELECT/WITH/EXPLAIN queries are allowed in execute_sql; "
                "use submit_sql for statements that modify data."
            ),
        }

    return await asyncio.to_thread(_execute_sql_sync, td, req.sql)


# ---------------------------------------------------------------------------
# /submit — one-shot final-answer tool
# ---------------------------------------------------------------------------

def _grade_submission(td: Dict[str, Any], sql: str, pred_query_result, task_db: str, conn) -> bool:
    category = td.get("category", "Query")
    test_cases = td.get("test_cases") or []
    sol_sql = td.get("sol_sql") or []
    conditions = td.get("conditions") or {}

    if category == "Query" or not test_cases:
        pred_sqls = db_utils.remove_round(db_utils.remove_distinct(db_utils.remove_comments([sql])))
        sol_sqls = db_utils.remove_round(db_utils.remove_distinct(db_utils.remove_comments(sol_sql)))
        result = db_utils.ex_base(pred_sqls, sol_sqls, task_db, conn, conditions)
        return result == 1

    def _execute_queries_compat(queries, db_name, conn=None):
        result, error, timeout, _ = db_utils.execute_queries(queries, db_name, conn)
        return result, error, timeout

    exec_globals = {
        "execute_queries": _execute_queries_compat,
        "ex_base": db_utils.ex_base,
        "remove_distinct": db_utils.remove_distinct,
        "remove_comments": db_utils.remove_comments,
        "remove_round": db_utils.remove_round,
        "pred_query_result": pred_query_result,
    }
    for tc_code in test_cases:
        if not isinstance(tc_code, str):
            continue
        try:
            exec_locals: Dict[str, Any] = {}
            exec(tc_code, exec_globals, exec_locals)
            tc_func = exec_locals.get("test_case")
            if tc_func and callable(tc_func):
                tc_func([sql], sol_sql, task_db, conn)
        except Exception:
            return False
    return True


def _submit_sql_sync(td: Dict[str, Any], sql: str) -> Dict[str, Any]:
    lock: threading.Lock = td["lock"]
    task_db = td["task_db"]
    selected_database = td["selected_database"]
    template = db_utils.template_db_name(selected_database)

    lock.acquire()
    try:
        # Step 1: reset the task db fresh from its template, so grading
        # always runs against a pristine copy regardless of prior
        # execute_sql / earlier submit_sql calls from sibling branches.
        try:
            db_utils.reset_task_db(task_db, template)
        except Exception as e:
            return {"status": "invalid_submission", "message": f"failed to reset task db: {e}"}

        pool = db_utils.get_or_init_pool(task_db)
        conn = pool.getconn()
        status_result: Dict[str, Any]
        try:
            # Step 2: execute the submitted SQL as one (possibly
            # multi-statement) script.
            try:
                cursor = conn.cursor()
                cursor.execute("SET statement_timeout = '60s';")
                cursor.execute(sql)
                conn.commit()
                try:
                    pred_query_result = cursor.fetchall()
                except psycopg2.ProgrammingError:
                    pred_query_result = None
                cursor.close()
            except Exception as e:
                try:
                    conn.rollback()
                except Exception:
                    pass
                status_result = {"status": "invalid_submission", "message": str(e)}
            else:
                # Step 3: grade.
                try:
                    passed = _grade_submission(td, sql, pred_query_result, task_db, conn)
                except Exception:
                    passed = False
                if passed:
                    status_result = {"status": "correct", "message": "Submitted SQL passed grading."}
                else:
                    status_result = {"status": "incorrect", "message": "Submitted SQL did not pass grading."}
        finally:
            try:
                pool.putconn(conn)
            except Exception:
                pass
            # Step 4: reset the task db back to template once more,
            # regardless of pass/fail, so it's pristine for whatever
            # sibling rollout branch queries it next via /execute.
            try:
                db_utils.reset_task_db(task_db, template)
            except Exception as e:
                logger.warning("post-submit reset failed for %s: %s", task_db, e)

        return status_result
    finally:
        lock.release()


@app.post("/submit")
async def submit(req: SubmitRequest):
    td = _tasks.get(req.task_id)
    if td is None:
        return _task_not_found(req.task_id)
    return await asyncio.to_thread(_submit_sql_sync, td, req.sql)


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser(description="Launch the DB environment service.")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind the server to.")
    parser.add_argument("--port", type=int, default=6002, help="Port to bind the server to.")
    parser.add_argument("--pg_host", type=str, default="127.0.0.1", help="Postgres host.")
    parser.add_argument("--pg_port", type=int, default=5432, help="Postgres port.")
    parser.add_argument("--pg_user", type=str, default="root", help="Postgres user.")
    parser.add_argument("--pg_password", type=str, default="123123", help="Postgres password.")
    parser.add_argument("--pg_minconn", type=int, default=1, help="Min connections per per-db pool.")
    parser.add_argument("--pg_maxconn", type=int, default=5, help="Max connections per per-db pool.")
    parser.add_argument(
        "--dataset_root",
        type=str,
        required=True,
        help=(
            "Local directory containing the LiveSQLBench split dirs "
            "(livesqlbench-base-full/, livesqlbench-base-lite/). Must exist "
            "on the machine running this service, independent of the "
            "training process's own dataset path."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    settings.pg_host = args.pg_host
    settings.pg_port = args.pg_port
    settings.pg_user = args.pg_user
    settings.pg_password = args.pg_password
    settings.pg_minconn = args.pg_minconn
    settings.pg_maxconn = args.pg_maxconn
    settings.dataset_root = os.path.abspath(args.dataset_root)

    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host=args.host, port=args.port)
