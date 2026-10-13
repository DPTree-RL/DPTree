"""
Shared definition of the LiveSQLBench agent's tool set and its XML tool-call protocol.

The protocol is Qwen3.5's native tool-calling format (the same one Spider2's spider-agent-tc
harness hand-rolls with regex): the model wraps a single call as

    <tool_call>
    <function=execute_sql>
    <parameter=sql>
    SELECT * FROM foo LIMIT 5;
    </parameter>
    </function>
    </tool_call>

and generation stops right after </tool_call> (see vllm_rollout_spmd_ts.py). Qwen's own chat
template renders this exact block (plus a <tools>...</tools> system preamble) whenever
apply_chat_template is called with a `tools=` argument -- see render_tools_system_block below,
which reproduces that same rendering by hand so scripts/data_process can bake it directly into
the parquet 'prompt' column without generation_ts.py needing to pass tools= through the chat
template at all (the template's tools-less system-message branch renders the content verbatim,
so a pre-rendered content string round-trips byte-for-byte).

This module is imported by both scripts/data_process/*.py (to build the system prompt) and
search_r1/llm_agent/generation_ts.py (to parse/dispatch tool calls during rollout), so the tool
set only needs to be defined once.
"""

import json
import re
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Tool schema (OpenAI-style function-calling JSON Schema, matching what Qwen's own chat
# template would emit for `tools=[{"type": "function", "function": t} for t in SQL_TOOLS]`).
# ---------------------------------------------------------------------------

SQL_TOOLS: List[Dict] = [
    {
        "name": "get_table_schema",
        "description": "Get one table's CREATE TABLE statement plus a few sample rows. Call this once per table you actually need, using a table name from the 'Available tables' list in the user message -- it does not return the whole database at once.",
        "parameters": {
            "type": "object",
            "properties": {
                "table_name": {"type": "string", "description": "The table name, as listed in the user message's 'Available tables'."},
            },
            "required": ["table_name"],
        },
    },
    {
        "name": "get_column_meaning",
        "description": "Get the human-readable description of one specific column.",
        "parameters": {
            "type": "object",
            "properties": {
                "table_name": {"type": "string", "description": "The table the column belongs to."},
                "column_name": {"type": "string", "description": "The column name."},
            },
            "required": ["table_name", "column_name"],
        },
    },
    {
        "name": "get_knowledge_definition",
        "description": "Get the full definition (including any formula) of one named domain-knowledge entry.",
        "parameters": {
            "type": "object",
            "properties": {
                "knowledge_name": {"type": "string", "description": "The knowledge entry's name, as listed in the user message's 'Available knowledge base entries'."},
            },
            "required": ["knowledge_name"],
        },
    },
    {
        "name": "execute_sql",
        "description": "Run a read-only SQL query (SELECT / WITH / EXPLAIN only) against the database and see its results. Use this to explore data and test queries before submitting. This does NOT submit your final answer.",
        "parameters": {
            "type": "object",
            "properties": {
                "sql": {"type": "string", "description": "The SQL query to run. Must start with SELECT, WITH, or EXPLAIN."},
            },
            "required": ["sql"],
        },
    },
    {
        "name": "submit_sql",
        "description": "Submit your final SQL answer. This may include INSERT/UPDATE/DELETE/CREATE/ALTER/DROP statements when the task requires modifying the database. You have exactly ONE submission attempt -- once you call this, the task ends immediately (pass or fail), so make sure you are confident before calling it.",
        "parameters": {
            "type": "object",
            "properties": {
                "sql": {"type": "string", "description": "The final SQL statement(s) that answer the task."},
            },
            "required": ["sql"],
        },
    },
]

TOOL_REQUIRED_ARGS: Dict[str, set] = {
    t["name"]: set(t["parameters"].get("required", [])) for t in SQL_TOOLS
}
TOOL_NAMES: set = set(TOOL_REQUIRED_ARGS.keys())

# Tools that mutate/finalize task state and are dispatched with special handling by the caller.
READ_ONLY_METADATA_TOOLS = {
    "get_table_schema", "get_column_meaning", "get_knowledge_definition",
}


def render_tools_system_block(task_instructions: str) -> str:
    """
    Build the system-message content string that reproduces exactly what Qwen3.5's chat
    template renders when apply_chat_template is called with tools=[{"type":"function",
    "function": t} for t in SQL_TOOLS] and a system message whose content is
    `task_instructions` -- see the template's tools-branch in tokenizer_config.json. Baking this
    in at data-processing time means scripts/data_process can just write
    {"role": "system", "content": render_tools_system_block(...)} into the parquet 'prompt'
    column and apply_chat_template(..., add_generation_prompt=True) (no tools= kwarg needed)
    renders it identically, since the template's no-tools branch emits system content verbatim.
    """
    lines = [
        "# Tools",
        "",
        "You have access to the following functions:",
        "",
        "<tools>",
    ]
    for t in SQL_TOOLS:
        lines.append(json.dumps({"type": "function", "function": t}, ensure_ascii=False))
    lines.append("</tools>")
    lines.append("")
    lines.append("If you choose to call a function ONLY reply in the following format with NO suffix:")
    lines.append("")
    lines.append("<tool_call>")
    lines.append("<function=example_function_name>")
    lines.append("<parameter=example_parameter_1>")
    lines.append("value_1")
    lines.append("</parameter>")
    lines.append("<parameter=example_parameter_2>")
    lines.append("This is the value for the second parameter")
    lines.append("that can span")
    lines.append("multiple lines")
    lines.append("</parameter>")
    lines.append("</function>")
    lines.append("</tool_call>")
    lines.append("")
    lines.append("<IMPORTANT>")
    lines.append("Reminder:")
    lines.append("- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags")
    lines.append("- Required parameters MUST be specified")
    lines.append("- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after")
    lines.append("- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls")
    lines.append("</IMPORTANT>")
    lines.append("")
    lines.append(task_instructions.strip())
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Parser: extract (function_name, arguments) from raw generated text. Regex-based, mirroring
# Spider2's spider-agent-tc/agent/message_processor.py -- Qwen never emits real XML entity
# escaping in parameter values (e.g. a SQL "WHERE x < 5" is embedded raw), so a proper XML
# parser would choke; this matches the format actually produced.
# ---------------------------------------------------------------------------

_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
_FUNCTION_RE = re.compile(r"<function=([^>]+)>")
_PARAMETER_RE = re.compile(r"<parameter=([^>]+)>(.*?)</parameter>", re.DOTALL)


def parse_tool_call(text: str) -> Tuple[Optional[str], Optional[Dict[str, str]]]:
    """
    Parse the first <tool_call>...</tool_call> block out of raw generated text.

    Returns (function_name, arguments_dict). Returns (None, None) if no <tool_call> block (or no
    <function=...> tag inside one) was found at all -- this is the "malformed_xml" case, i.e. no
    tool call was attempted. A tool call with a name/parameters that don't match a known tool /
    its required arguments is still returned here (name, args) -- checking that is the caller's
    job ("invalid_arguments" case), since parsing succeeded even if the call itself is invalid.
    """
    m = _TOOL_CALL_RE.search(text)
    if not m:
        return None, None
    inner = m.group(1)
    fm = _FUNCTION_RE.search(inner)
    if not fm:
        return None, None
    name = fm.group(1).strip()
    args = {pname.strip(): pval.strip() for pname, pval in _PARAMETER_RE.findall(inner)}
    return name, args
