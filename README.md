# simple-agent-loop-example

Minimal AI agent harness in a single Python file. A `while` loop calls Claude with tools until the model stops making tool calls.

Uses Claude Sonnet 4.6 with interleaved extended thinking so the model reasons between tool calls.

## How it works

```
User task
    |
    v
while True:
    load memory into system prompt
    estimate full context (messages + system + tools)
    compact while over token limit
    call Claude API (with thinking + tools)
    append response (thinking blocks preserved unmodified)
    if no tool_use blocks -> return final text
    execute tool calls
    append tool results
    loop
```

## Tools

| Tool | Description |
|------|-------------|
| `read_file` | Read a file's contents |
| `search_files` | Grep for a regex pattern across files |
| `write_file` | Write content to a file |
| `plan` | Record a step-by-step plan |
| `memory_save` | Persist a key-value pair to disk |
| `memory_load` | Retrieve from persistent memory |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export ANTHROPIC_API_KEY="sk-ant-..."
```

## Usage

```bash
python agent.py 'read all python files and summarize the architecture'
python agent.py 'find all TODO comments and create a summary'
```

## Limitations

- **No streaming.** Responses are returned in full after each API call. For long-running tool loops this means no incremental output — you wait for the entire response before seeing anything.

- **Sequential tool execution.** When the model returns multiple tool calls in a single response, they are executed one at a time. There is no concurrent/parallel dispatch, so independent tool calls that could run simultaneously are bottlenecked.

- **No sandboxing or input validation on tool calls.** The `write_file` tool writes anywhere the process has permission. The `search_files` tool passes patterns directly to `grep` via `subprocess`. A model-generated path like `../../etc/passwd` or a crafted glob/pattern could read or overwrite sensitive files. In production, tool inputs must be validated and execution must be sandboxed.

- **No human-in-the-loop confirmation.** Destructive actions (file writes, overwrites) execute immediately without user approval. A production harness should gate dangerous operations behind confirmation prompts.

- **Naive token estimation.** Context size is estimated at ~4 characters per token. This is a rough heuristic — actual token counts vary. The compaction threshold may trigger too early or too late. The estimate does account for system prompt, tools, and memory overhead.

- **No rate limiting or retry logic.** API errors, rate limits, and transient failures are not handled. The loop will crash on the first API error.

## License

MIT
