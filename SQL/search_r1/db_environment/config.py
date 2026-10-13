"""Postgres connection settings for the DB environment service.

Populated from CLI args in server.py's ``__main__`` block before uvicorn
starts. Kept as a mutable module-level singleton (rather than pydantic
settings / dotenv) since this package has no other config dependencies and
the training-loop caller only ever talks to this service over HTTP.
"""

from dataclasses import dataclass


@dataclass
class PGSettings:
    pg_host: str = "127.0.0.1"
    pg_port: int = 5432
    pg_user: str = "root"
    pg_password: str = "123123"
    pg_minconn: int = 1
    pg_maxconn: int = 5
    # Local filesystem root containing the LiveSQLBench split directories
    # (``livesqlbench-base-full/``, ``livesqlbench-base-lite/``), each holding
    # one ``<db>/`` folder per selected_database with that db's schema/column
    # meaning/knowledge-base files. Set from --dataset_root in server.py so
    # this service can be deployed on a different machine than the training
    # process, independent of whatever path the training side uses.
    dataset_root: str = ""


# Module-level singleton. `server.py` mutates this in-place from argparse
# before `uvicorn.run(...)` is called, so every request handler (and
# db_utils, which imports this object) sees the final values.
settings = PGSettings()
