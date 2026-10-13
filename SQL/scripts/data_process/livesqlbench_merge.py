"""
Turn a LiveSQLBench split (livesqlbench-base-full for training, livesqlbench-base-lite for
validation) into a parquet file the verl RLHFDataset (verl/utils/dataset/rl_dataset.py) can
load directly.

Each output row's 'prompt' column is a [system, user] chat-message list. The system message's
content is pre-rendered via search_r1.llm_agent.sql_tools.render_tools_system_block so it is
byte-for-byte what tokenizer.apply_chat_template(..., tools=SQL_TOOLS) would have produced
(Qwen3.5's chat template renders a tools-less system message's content verbatim), meaning
RLHFDataset's plain apply_chat_template(chat, add_generation_prompt=True) call (no tools= kwarg)
at training time reproduces the exact tool-calling prompt with zero further training-side
plumbing -- see sql_tools.py's docstring for why this works.

'reward_model.ground_truth' carries everything the DB environment server and the training loop
need at rollout time: which physical database to clone, the gold SQL / test cases to grade
against, and the dataset root (schema/knowledge files) to serve get_table_schema etc. from -- see
LLMGenerationTreeSearchManager._init_db_batch in search_r1/llm_agent/generation_ts.py.

'extra_info.index' is set to the LiveSQLBench instance_id (already globally unique within a
split) -- this becomes non_tensor_batch['uid'], which doubles as BOTH the GRPO/tree
normalization grouping key AND the DB environment task_id, so every tree-search branch of one
task shares exactly one physical DB instance for the training step (see CLAUDE.md's dataset
section / the project's "接入" design notes).
"""

import argparse
import json
import os
import re
import sys

import pandas as pd

# Make search_r1 importable regardless of cwd (this script lives at
# <repo_root>/scripts/data_process/livesqlbench_merge.py).
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from search_r1.llm_agent.sql_tools import render_tools_system_block

# The literal '__MAX_TURNS__' placeholder below is intentionally left unresolved here -- it's
# substituted with the live max_turns config value at training time by RLHFDataset
# (verl/utils/dataset/rl_dataset.py), via a plain string .replace() on the fully rendered prompt
# (not str.format(), since the rendered system block also embeds the tools' JSON schemas, which
# are full of unrelated '{'/'}' characters). This keeps the prompt text in sync with whatever
# max_turns the launch script actually passes, without needing to re-run this merge script every
# time max_turns changes.
TASK_INSTRUCTIONS = """\
You are a PostgreSQL expert database agent. Your job is to write SQL that correctly answers the \
user's question about the given database, using the tools above to explore the schema and \
domain knowledge before answering. The user message below already lists every table name and \
every knowledge-base entry name available for this database -- you don't need a tool call to \
discover those.

Rules:
- Explore first: call get_table_schema on only the tables you actually need (and \
get_column_meaning / get_knowledge_definition as needed), using the names already given to you \
in the user message. Domain-specific terms and formulas are often only defined in the knowledge \
base, not in the schema itself.
- Use execute_sql to test read-only (SELECT / WITH / EXPLAIN) queries and check your work. \
execute_sql can NEVER modify the database -- any INSERT/UPDATE/DELETE/CREATE/ALTER/DROP \
statement submitted there is rejected.
- If the task requires modifying the database, put that modifying SQL in submit_sql instead of \
execute_sql -- submit_sql may contain CRUD/DDL statements.
- You have exactly ONE submission attempt: call submit_sql only once you are confident in your \
final SQL, since the task ends immediately (pass or fail) the moment you call it.
- You have at most __MAX_TURNS__ tool calls for this task, submit_sql included -- explore \
efficiently."""

# Mirrors search_r1/db_environment/server.py's own _TABLE_HEADER_RE exactly (kept as a separate
# copy rather than an import so this offline preprocessing script doesn't need to pull in
# db_environment's FastAPI/psycopg2 dependencies just to list table names). Most *_schema.txt
# files write 'CREATE TABLE "name"', but a handful of LiveSQLBench databases (cross_border,
# crypto_exchange, cybermarket_pattern, insider_trading, museum_artifact, polar_equipment)
# instead write '"CREATE" TABLE "name"', with CREATE itself quoted -- both must match or those
# six dbs' table lists come back empty.
_TABLE_HEADER_RE = re.compile(r'^"?CREATE"? TABLE "([^"]+)"', re.MULTILINE)


def _table_names_for(dataset_root: str, db: str) -> list:
    """Every table name in db's schema.txt, in file order (same source server.py's
    get_table_schema reads at rollout time, so the names given here are guaranteed valid
    get_table_schema arguments)."""
    path = os.path.join(dataset_root, db, f"{db}_schema.txt")
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()
    return [m.group(1) for m in _TABLE_HEADER_RE.finditer(text)]


def _knowledge_names_for(dataset_root: str, db: str) -> list:
    """Every knowledge-base entry name for db, in file order (same source server.py's
    get_knowledge_definition reads at rollout time)."""
    path = os.path.join(dataset_root, db, f"{db}_kb.jsonl")
    names = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if isinstance(entry, dict) and entry.get('knowledge'):
                names.append(entry['knowledge'])
    return names


def build_prompt(query: str, table_names: list, knowledge_names: list) -> list:
    system_content = render_tools_system_block(TASK_INSTRUCTIONS)
    user_content = (
        f"User Query:\n{query}\n\n"
        f"Available tables: {json.dumps(table_names, ensure_ascii=False)}\n"
        f"Available knowledge base entries: {json.dumps(knowledge_names, ensure_ascii=False)}"
    )
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]


def load_manifest(dataset_root: str) -> list:
    manifest_path = os.path.join(dataset_root, 'livesqlbench_data.jsonl')
    rows = []
    with open(manifest_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', required=True,
                         help='Path to a LiveSQLBench split directory, e.g. '
                              '/dataset/wang0702/livesqlbench/livesqlbench-base-full '
                              '(must contain livesqlbench_data.jsonl and one folder per database).')
    parser.add_argument('--local_dir', required=True, help='Output directory for the parquet file.')
    parser.add_argument('--output_filename', required=True, help='e.g. train.parquet or test.parquet')
    parser.add_argument('--split', required=True, help="Stored in extra_info.split, e.g. 'train' or 'test'.")
    parser.add_argument('--data_source', default='livesqlbench')
    args = parser.parse_args()

    dataset_root = os.path.abspath(args.dataset_root)
    manifest = load_manifest(dataset_root)
    print(f'Loaded {len(manifest)} tasks from {dataset_root}')

    seen_ids = set()
    records = []
    table_names_cache = {}
    knowledge_names_cache = {}
    for task in manifest:
        instance_id = task['instance_id']
        assert instance_id not in seen_ids, f'duplicate instance_id: {instance_id}'
        seen_ids.add(instance_id)

        db = task['selected_database']
        if db not in table_names_cache:
            table_names_cache[db] = _table_names_for(dataset_root, db)
            knowledge_names_cache[db] = _knowledge_names_for(dataset_root, db)

        record = {
            'data_source': args.data_source,
            'prompt': build_prompt(task['query'], table_names_cache[db], knowledge_names_cache[db]),
            'ability': 'sql',
            'reward_model': {
                'style': 'rule',
                'ground_truth': {
                    'selected_database': task['selected_database'],
                    'sol_sql': task.get('sol_sql', []),
                    'test_cases': task.get('test_cases', []),
                    'category': task.get('category', 'Query'),
                    'conditions': task.get('conditions', {}),
                },
            },
            'extra_info': {
                'split': args.split,
                'index': instance_id,
                'instance_id': instance_id,
                'selected_database': task['selected_database'],
                'category': task.get('category', 'Query'),
            },
        }
        records.append(record)

    df = pd.DataFrame(records)
    os.makedirs(args.local_dir, exist_ok=True)
    out_path = os.path.join(args.local_dir, args.output_filename)
    df.to_parquet(out_path)
    print(f'Wrote {len(df)} rows to {out_path}')


if __name__ == '__main__':
    main()
