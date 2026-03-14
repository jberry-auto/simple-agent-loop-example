"""Minimal AI agent harness — tool-call loop with interleaved thinking.

Setup:
    python -m venv .venv && source .venv/bin/activate
    pip install anthropic==0.83.0
    export ANTHROPIC_API_KEY="sk-ant-..."

Usage:
    python agent.py 'your task here'
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

import anthropic

logger = logging.getLogger(__name__)

MODEL = "claude-sonnet-4-6"
MAX_CONTEXT_TOKENS = 180_000
THINKING_BUDGET = 10_000
MEMORY_DIR = Path("agent_memory")
INTERLEAVED_THINKING_BETA = "interleaved-thinking-2025-05-14"

TOOLS: list[dict[str, Any]] = [
    {
        "name": "read_file",
        "description": "Read a file's contents.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string", "description": "File path."}},
            "required": ["path"],
        },
    },
    {
        "name": "search_files",
        "description": "Grep for a pattern across files.",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Regex pattern."},
                "path": {"type": "string", "description": "Directory.", "default": "."},
                "glob": {"type": "string", "description": "File glob.", "default": "*"},
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "write_file",
        "description": "Write content to a file, creating directories as needed.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path."},
                "content": {"type": "string", "description": "File content."},
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "plan",
        "description": "Record a step-by-step plan for reference.",
        "input_schema": {
            "type": "object",
            "properties": {"plan": {"type": "string", "description": "Your plan."}},
            "required": ["plan"],
        },
    },
    {
        "name": "memory_save",
        "description": "Save a key-value pair to persistent memory.",
        "input_schema": {
            "type": "object",
            "properties": {
                "key": {"type": "string", "description": "Memory key."},
                "value": {"type": "string", "description": "Content to store."},
            },
            "required": ["key", "value"],
        },
    },
    {
        "name": "memory_load",
        "description": "Load from memory by key. Use '_index' to list all keys.",
        "input_schema": {
            "type": "object",
            "properties": {"key": {"type": "string", "description": "Key or '_index'."}},
            "required": ["key"],
        },
    },
]


# --- Tool Execution ---

def execute_tool(name: str, args: dict[str, Any]) -> str:
    """Dispatch and execute a single tool call."""
    match name:
        case "read_file":
            return _read_file(args["path"])
        case "search_files":
            return _search_files(args["pattern"], args.get("path", "."), args.get("glob", "*"))
        case "write_file":
            return _write_file(args["path"], args["content"])
        case "plan":
            return f"Plan recorded:\n{args['plan']}"
        case "memory_save":
            return _memory_save(args["key"], args["value"])
        case "memory_load":
            return _memory_load(args["key"])
        case _:
            return f"Error: unknown tool '{name}'"


def _read_file(path: str) -> str:
    try:
        return Path(path).read_text()
    except (FileNotFoundError, IsADirectoryError) as exc:
        return f"Error: {exc}"


def _search_files(pattern: str, path: str, glob: str) -> str:
    try:
        result = subprocess.run(
            ["grep", "-rn", "--include", glob, pattern, path],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.stdout[:10_000] if result.stdout else "No matches found."
    except subprocess.TimeoutExpired:
        return "Error: search timed out."


def _write_file(path: str, content: str) -> str:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(content)
    return f"Wrote {len(content)} bytes to {path}"


def _memory_save(key: str, value: str) -> str:
    MEMORY_DIR.mkdir(exist_ok=True)
    (MEMORY_DIR / f"{key}.md").write_text(value)
    return f"Saved memory: {key}"


def _memory_load(key: str) -> str:
    if key == "_index":
        if not MEMORY_DIR.exists():
            return "No memories saved yet."
        keys = [f.stem for f in MEMORY_DIR.glob("*.md")]
        return f"Stored keys: {', '.join(keys)}" if keys else "No memories saved yet."
    path = MEMORY_DIR / f"{key}.md"
    return path.read_text() if path.exists() else f"No memory for key: {key}"


# --- Context Management ---

def load_memory_context() -> str:
    """Load all memory files into a context string."""
    if not MEMORY_DIR.exists():
        return ""
    parts = [
        f"[memory:{f.stem}]\n{f.read_text()}"
        for f in sorted(MEMORY_DIR.glob("*.md"))
    ]
    return "\n\n".join(parts)


def estimate_tokens(messages: list[dict[str, Any]]) -> int:
    """Rough estimate: ~4 chars per token."""
    return len(json.dumps(messages, default=str)) // 4


def deduplicate_tool_results(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove duplicate tool results (same tool_use_id seen twice)."""
    seen_ids: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for msg in messages:
        if msg["role"] == "user" and isinstance(msg.get("content"), list):
            unique_blocks = []
            for block in msg["content"]:
                if block.get("type") == "tool_result":
                    tid = block.get("tool_use_id", "")
                    if tid not in seen_ids:
                        seen_ids.add(tid)
                        unique_blocks.append(block)
                else:
                    unique_blocks.append(block)
            if unique_blocks:
                deduped.append({**msg, "content": unique_blocks})
        else:
            deduped.append(msg)
    return deduped


def _strip_thinking_for_summary(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Strip thinking blocks from messages so they can be sent to a fresh API call."""
    stripped = []
    for msg in messages:
        if msg["role"] == "assistant" and isinstance(msg.get("content"), list):
            filtered = [
                b for b in msg["content"]
                if not (isinstance(b, dict) and b.get("type") in ("thinking", "redacted_thinking"))
            ]
            if filtered:
                stripped.append({**msg, "content": filtered})
        else:
            stripped.append(msg)
    return stripped


def compact_context(
    client: anthropic.Anthropic,
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Summarize conversation via Claude to reduce context size."""
    logger.info("Compacting context (%d est. tokens)...", estimate_tokens(messages))
    # Thinking blocks have signatures bound to the original conversation — strip them
    stripped = _strip_thinking_for_summary(messages)
    resp = client.messages.create(
        model=MODEL,
        max_tokens=4096,
        system=(
            "Summarize this conversation concisely. Preserve: decisions made, "
            "current plan/progress, key file contents, pending work. "
            "Drop: verbose tool outputs, repeated info."
        ),
        messages=stripped + [{"role": "user", "content": "Summarize the conversation so far."}],
    )
    summary = resp.content[0].text
    return [{"role": "user", "content": f"[Conversation compacted]\n\n{summary}"}]


# --- Main Loop ---

def run_agent(
    user_task: str,
    system_prompt: str = "",
) -> str:
    """Run the agent loop until the model stops calling tools.

    Args:
        user_task: The user's task description.
        system_prompt: Optional system prompt override.

    Returns:
        The final text response from the model.
    """
    client = anthropic.Anthropic()

    default_system = (
        "You are a capable AI agent with tools to read files, search code, "
        "write files, plan your work, and save/load memory. Use tools to "
        "accomplish the user's task. When done, respond without tool calls."
    )
    system = system_prompt or default_system
    messages: list[dict[str, Any]] = [{"role": "user", "content": user_task}]

    while True:
        messages = deduplicate_tool_results(messages)

        # Compact if approaching context limit
        if estimate_tokens(messages) > MAX_CONTEXT_TOKENS:
            messages = compact_context(client, messages)
            messages.append({"role": "user", "content": f"Continue working on: {user_task}"})

        # Refresh memory into system prompt
        active_system = system
        memory = load_memory_context()
        if memory:
            active_system += f"\n\n## Persistent Memory\n{memory}"

        logger.info("Calling model (%d est. tokens)...", estimate_tokens(messages))

        response = client.messages.create(
            model=MODEL,
            max_tokens=16_000,
            system=active_system,
            tools=TOOLS,
            messages=messages,
            thinking={"type": "enabled", "budget_tokens": THINKING_BUDGET},
            extra_headers={"anthropic-beta": INTERLEAVED_THINKING_BETA},
        )

        # Preserve the full response content (thinking + text + tool_use) unmodified.
        # The SDK returns content blocks as objects; pass them directly so thinking
        # block signatures stay intact for the next turn.
        messages.append({"role": "assistant", "content": response.content})

        # Log thinking for observability
        for block in response.content:
            if getattr(block, "type", None) == "thinking":
                logger.info("Thinking: %s", block.thinking[:200])

        # Collect tool calls
        tool_calls = [b for b in response.content if getattr(b, "type", None) == "tool_use"]

        # Exit condition: no tool calls means the agent is done
        if not tool_calls:
            return "\n".join(
                b.text for b in response.content if getattr(b, "type", None) == "text"
            )

        # Execute tools and feed results back
        tool_results = []
        for tc in tool_calls:
            logger.info("Executing: %s", tc.name)
            result = execute_tool(tc.name, tc.input)
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tc.id,
                "content": result,
            })

        messages.append({"role": "user", "content": tool_results})


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if len(sys.argv) < 2:
        print("Usage: python agent.py '<task>'")
        sys.exit(1)

    task = " ".join(sys.argv[1:])
    print("\n" + run_agent(task))
