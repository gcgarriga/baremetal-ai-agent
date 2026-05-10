# Baremetal Agent

A tiny framework-free agent in Python. It shows the whole tool-use loop with
raw LLM API calls, explicit tool schemas, and a small registry — no LangChain,
no CrewAI, no agent abstraction layers. Read the source; the source *is* the
framework.

## What this is

- **Tiny framework** — a small, useful structure around the raw API loop, not an opinionated platform.
- **Framework-free internals** — no LangChain, no CrewAI, no agent abstraction layers. The loop is just a loop.
- **Inspectable by default** — every API payload is logged, trajectories are exportable, and replay works offline.
- **Safe enough to learn with** — path traversal protection, secret redaction, and confirmation gates on dangerous tools.

## What this is not

- Not a production agent platform — no persistence layer, no multi-tenancy, no SLA.
- Not a general workflow engine — no DAG scheduler, no state machine, no retries per node.
- Not a wrapper around an agent framework — the source is the framework; read it directly.
- Not a benchmark harness with LLM judges — evals are deterministic rubric checks, not model-graded.

## Quickstart

Requires **Python 3.10+**. A virtual environment is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

For development (adds `pytest` and `ruff`):

```bash
pip install -e .[dev]
```

Get a token at **GitHub → Settings → Developer settings → Fine-grained tokens**
with `Models: Read` permission, then drop it in a `.env` file:

```bash
cp .env.example .env   # then edit GITHUB_TOKEN
```

Launch the interactive REPL:

```bash
python -m baremetal_agent
# or:
baremetal-agent
```

Run one prompt and exit:

```bash
baremetal-agent -p "Summarize this repo in three bullets."
```

Runtime dependencies are intentionally small: `httpx` for API calls and `rich`
for terminal rendering.

### Debugging

Set `AGENT_VERBOSE=1` (or run `verbose` in the REPL) to print every API
request and response inside a box frame, with likely secrets redacted by
`safety.redact_secrets`. For finer control, `AGENT_RENDER_VERBOSE=1` switches
to the plain renderer without enabling payload logging, and
`AGENT_LOG_PAYLOADS=1` enables payload logging while keeping the rich UI.
Most prompt failures are obvious from the payloads — start there before
reading the loop.

## The three files that matter

If you only read three files, read these. They are the whole agent.

### [`baremetal_agent/agent.py`](baremetal_agent/agent.py) — the loop

The entry point is `run_agent_turn`. It takes a user message, sends it
to the model, executes any tool calls the model returns, appends the results
back into the conversation, and repeats until the model returns a plain
assistant message or the iteration limit is reached. That's the whole agent.

### [`baremetal_agent/tools.py`](baremetal_agent/tools.py) — what the model can do

A `TOOLS` registry holds every callable the model is allowed to invoke. Each
implementation is a plain function decorated with `@tool(...)`, which
co-locates the JSON schema with the implementation. Tools accept keyword
arguments only and **always return strings** — including their errors — so the
loop never has to handle exceptions from a tool. `execute_tool` is the single
dispatch point.

### [`baremetal_agent/client.py`](baremetal_agent/client.py) — how it talks to the API

`chat_completion` is one function: build the request body, POST it to the
GitHub Models endpoint, retry on 429/5xx with backoff, and return the parsed
JSON. Every payload is logged inside a box-drawing frame when verbose mode is
on, with secrets redacted. No SDK, no abstractions — that is the point.
(A `chat_completion_stream` companion exists for the optional streaming path;
see [docs/advanced.md](docs/advanced.md#streaming).)

## Architecture

```
user prompt
   │
   ▼
┌──────────────────────────────────────────┐
│  agent.py: run_agent_turn()       │
│                                          │
│   ┌───────────────────────────────┐      │
│   │ client.chat_completion(...)   │  ←── client.py
│   └──────────────┬────────────────┘      │
│                  │                       │
│         tool_calls?                      │
│           │     │                        │
│          yes    no → return final text   │
│           │                              │
│           ▼                              │
│   ┌───────────────────────────────┐      │
│   │ tools.execute_tool(name, args)│  ←── tools.py
│   └──────────────┬────────────────┘      │
│                  │                       │
│         append result, loop              │
└──────────────────────────────────────────┘
```

Components:

| File | Role |
|------|------|
| `agent.py` | Tool-use loop — read this first |
| `tools.py` | Tool registry and implementations (path-safe, string-returning) |
| `client.py` | Raw HTTP to the model API, retry/backoff, payload logging |
| `cli.py` | REPL, one-shot, and command dispatch |
| `config.py` | Env vars + `.env` file loading into an `AgentConfig` dataclass |
| `safety.py` | Shared redaction and truncation helpers |

Streaming, trajectory export, replay, and the eval harness are layered on top
of these primitives without changing the core loop.

## Configuration

The agent reads configuration from environment variables, with a `.env`
fallback loaded from the current working directory.

| Variable | Required | Default | Description |
|----------|:--------:|---------|-------------|
| `GITHUB_TOKEN` | Yes | None | GitHub token used to call GitHub Models. Use a token with `Models: Read` permission. |
| `AGENT_API_URL` | No | `https://models.github.ai/inference/chat/completions` | Chat completions endpoint used for model requests. Assumes an OpenAI-compatible chat completions endpoint. |
| `AGENT_MODEL` | No | `openai/gpt-4.1` | Model identifier sent to the GitHub Models API. You can also change this during a session with `model <name>`. |
| `AGENT_MAX_ITERATIONS` | No | `10` | Maximum number of tool-use loop iterations for one user prompt. Must be an integer. |
| `AGENT_WORKING_DIR` | No | `.` | Directory that file and shell tools are allowed to access. It is resolved to an absolute path at startup. |
| `AGENT_VERBOSE` | No | `false` | Legacy one-knob verbose flag. When set to `1`, `true`, or `yes` it enables both the plain renderer (raw API payloads) and payload logging. Granular env vars below override it when set. |
| `AGENT_RENDER_VERBOSE` | No | inherits `AGENT_VERBOSE` | When set, controls renderer choice independently: truthy values use the plain renderer so raw API payloads are not interleaved with Rich panels. |
| `AGENT_LOG_PAYLOADS` | No | inherits `AGENT_VERBOSE` | When set, controls API request/response payload logging independently of the renderer. |
| `AGENT_STREAM` | No | `false` | Opts model calls into streaming responses when set to `1`, `true`, or `yes`. Non-streaming remains the default. |
| `AGENT_DOTENV` | No | `<current directory>/.env` | Path to the dotenv file to load. Set this in your shell before launch if you want to use a non-default dotenv path. |

## REPL commands

| Command | Description |
|---------|-------------|
| `help` | Show available commands |
| `tools` | List registered tools |
| `history` | Show conversation history |
| `trajectory [path]` | Export redacted, bounded conversation history as ATIF-v1.4 JSON |
| `verbose` | Toggle verbose mode (raw API payloads) |
| `stream` | Toggle streaming responses on/off |
| `model <name>` | Switch model mid-conversation |
| `clear` | Reset conversation |
| `exit` | Quit |

## Tools

| Tool | Confirmation |
|------|:---:|
| `read_file`, `list_directory`, `search_code` | No |
| `git_status`, `git_diff`, `git_log` | No |
| `write_file`, `shell_exec` | ⚠️ Yes |

In one-shot mode, confirmation-required tools such as `write_file` and
`shell_exec` are denied by default so scripts never block waiting for input.
Use `--allow-dangerous-tools` to explicitly permit them, or `--read-only` to
expose only tools that do not require confirmation.

High-risk tool outputs and saved trajectories are redacted for likely secrets
and capped with visible truncation markers before they are replayed or
persisted.

## Example prompts

```text
Help me understand this repo: summarize the architecture, required environment variables, and the current test surface.
```

```text
I need to add a new tool to this agent. Show me where tools are registered, how their schemas are defined, how confirmation is enforced for dangerous tools, and what tests I should update.
```

## What to read next

The features above are the whole core. Streaming, trajectory replay, and the
eval harness are optional and documented separately:

➡ **[docs/advanced.md](docs/advanced.md)** — streaming SSE, ATIF-v1.4
trajectories and offline replay, and the deterministic eval harness.

## Project structure

```
baremetal_agent/
├── __init__.py     — Package version
├── __main__.py     — python -m entry point
├── agent.py        — The agentic loop (read this first)
├── cli.py          — REPL with commands and confirmation prompts
├── client.py       — Raw HTTP to GitHub Models API, payload logging with secret redaction
├── config.py       — Env vars + .env file loading into AgentConfig
├── eval/           — Eval task loader/runner, rubric checks, reports, cache reuse
├── replay.py       — Offline ATIF-v1.4 trajectory replay and diff inspection
├── safety.py       — Shared redaction and truncation helpers
├── streaming.py    — Tiny SSE parser and streamed tool-call assembler
├── tools.py        — Tool registry + 8 implementations with path traversal protection
├── trajectory.py   — Redacted, bounded ATIF-v1.4 trajectory export
└── visualizer.py   — Live rich terminal visualization of agent steps
evals/
└── tasks/          — JSON eval task definitions consumed by `baremetal-agent eval`
tests/              — Unit and CLI coverage for client, tools, trajectory, replay, eval, streaming, and safety behavior
pyproject.toml      — Project metadata, dependencies, tool config
```

Every verbose API request and response is printed with likely secrets
redacted — that's the point.
