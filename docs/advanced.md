# Advanced topics

The [README](../README.md) covers the core teaching surface (the agent loop, the
tool registry, the API client). This document covers the optional features
layered on top: streaming, trajectory replay, and the eval harness.

Each section is self-contained. Read in any order.

- [Streaming](#streaming)
- [Trajectories and replay](#trajectories-and-replay)
- [Eval harness](#eval-harness)

---

## Streaming

By default, an agent turn makes a single non-streaming HTTP request and waits
for the complete response. Streaming is opt-in and **purely cosmetic** — it
exists so you can watch tokens arrive in real time. The final assistant message
shape is identical to the non-streaming path, so trajectory export, replay, and
eval all behave the same way.

Enable per-invocation:

```bash
baremetal-agent --stream -p "Reply with exactly: stream-ok"
```

Or set `AGENT_STREAM=1` to make it the default. Inside the REPL, the `stream`
command toggles it on and off mid-session.

How it works:

- `chat_completion_stream` in [`baremetal_agent/client.py`](../baremetal_agent/client.py)
  opens an SSE connection and yields chunks.
- [`baremetal_agent/streaming.py`](../baremetal_agent/streaming.py) is a small
  SSE parser plus an assembler that stitches content and tool-call fragments
  back into a single OpenAI-shaped response dict.
- Tool calls can stream in fragments, but a confirmation-required tool always
  waits for the complete JSON arguments before it runs.

**Where to read more:** start with `chat_completion_stream` in
`baremetal_agent/client.py`, then follow it into `baremetal_agent/streaming.py`.

---

## Trajectories and replay

A **trajectory** is a structured, redacted recording of one agent run, written
as ATIF-v1.4 JSON. It includes the user prompt, every assistant message, every
tool call and result, and per-step metrics. It is inspectable, diffable, and
replayable offline without a model call.

Record a trajectory from a one-shot run:

```bash
baremetal-agent -p "Inspect the current git status." --trajectory-out trajectory.json
```

Replay it:

```bash
baremetal-agent replay trajectory.json
baremetal-agent replay trajectory.json --step 3
baremetal-agent replay trajectory.json --diff baseline-trajectory.json
```

The REPL `trajectory [path]` command exports the current session's history the
same way.

How it works:

- [`baremetal_agent/trajectory.py`](../baremetal_agent/trajectory.py) converts
  the in-memory message history plus captured API responses into an ATIF-v1.4
  document. Strings are run through `safety.redact_secrets` and bounded to
  prevent runaway sizes.
- [`baremetal_agent/replay.py`](../baremetal_agent/replay.py) reads a saved
  trajectory and re-renders it through the same visualizer used by live runs,
  with optional step seeking and diff against a baseline.

**Where to read more:** `baremetal_agent/trajectory.py` for the export shape,
`baremetal_agent/replay.py` for the offline render and diff logic.

---

## Eval harness

The eval harness runs a directory of JSON task files against the agent and
checks each result with deterministic rubrics. It is **not** an LLM-judged
benchmark — every check is a concrete predicate over the trajectory or the
post-run workspace.

Run the bundled eval suite:

```bash
baremetal-agent eval --tasks evals/tasks --out report.md --json-out report.json
```

`baremetal-agent eval` with no flags uses those same defaults
(`evals/tasks`, `report.md`, `report.json`).

### Parallel execution

Pass `--workers N` to run tasks concurrently on a `ThreadPoolExecutor`:

```bash
baremetal-agent eval --workers 4
```

The default is `1` (serial) so reproducing a run only requires the same task
set, model, and cache state — no extra timing-dependent surprises. With `N>1`
each task still gets its own isolated sandbox and `ToolContext`, so trajectories
and the `.baremetal-eval-cache/` artifacts are unaffected by the worker count.

The order of results in `report.md` and `report.json` follows the task load
order regardless of which worker finishes first; aggregate counts and per-task
statuses are identical to a serial run modulo execution timing. The one
visible difference is on stdout — log lines from concurrently-running tasks
may interleave. Per-task ATIF trajectories on disk remain isolated.

Threads (not processes) are used because the in-tree tools are I/O-bound
(subprocess + HTTP); a `ProcessPoolExecutor` would buy nothing here and would
make sandbox cleanup harder to reason about.

### Task file format

Tasks are JSON objects with keys: `id`, `description`, `prompt`,
`requires_writes`, `setup`, and `rubric`.

```json
{
  "id": "write-exact-result-file",
  "description": "Write an exact result file without using shell_exec.",
  "prompt": "Create result.txt with exactly: eval harness ok",
  "requires_writes": true,
  "setup": {
    "directories": ["docs"],
    "files": [{"path": "docs/input.txt", "content": "seed"}],
    "fixture_dir": "fixtures/task-write",
    "copy_repo": false
  },
  "rubric": [
    {"type": "tool_called_with", "name": "write_file", "arguments": {"path": "result.txt"}},
    {"type": "file_contains", "path": "result.txt", "contains": "eval harness ok"}
  ]
}
```

### Rubric primitives

`final_response_contains`, `trajectory_atif`, `tool_called`,
`tool_called_with`, `tool_not_called`, `file_exists`, `file_contains`,
`max_agent_steps`.

Use `trajectory_atif` for deterministic checks against the ATIF trajectory
written by the harness. This validates the harness-produced artifact, not
whether the model's prose correctly describes trajectory internals; pair it
with `final_response_contains` when the answer text itself is part of the task.
Example: `{"type": "trajectory_atif", "schema_version": "ATIF-v1.4", "require_final_metrics": true}`.

Prefer behavior/output checks for bundled evals; reserve exact tool-call checks
for tasks where tool choice is the behavior under test.

### Safety defaults

- Tasks are read-only by default (`requires_writes: false`).
- Write-enabled tasks (`requires_writes: true`) can use `write_file` but still
  cannot use `shell_exec`.
- Setup supports only declarative directories/files/fixture copy and explicit
  controlled repo copy; no arbitrary setup shell commands.
- `setup.copy_repo: true` copies only tracked files from the task file's Git
  repository into the sandbox and seeds a clean local Git repo there. Tracked
  files named `.env` are excluded from the model workspace.
- Eval runs non-interactively (no confirmation prompts).
- Each task runs in an isolated temporary sandbox before snapshotting outputs.

### Results, cache, and report

Task statuses are: `pass`, `fail`, `error`, `skipped`.

The harness caches per-task artifacts under `.baremetal-eval-cache/`:

- `trajectory.json` (ATIF trajectory)
- `workspace.json` (post-run workspace snapshot)
- `metadata.json` (execution hash inputs)

Markdown and JSON report string fields are sanitized with the same
redaction/truncation helpers used for tool outputs and trajectories.

If you change only rubric logic, cached trajectory/workspace artifacts are
reused and checks are re-evaluated. Delete `.baremetal-eval-cache/` to force a
full rerun.

Compact Markdown report snippet:

```md
# Baremetal Agent Eval Report

| Task | Status | Checks | Cache |
| --- | --- | --- | --- |
| sample-write-file | pass | 2/2 | miss |
| git-status-without-shell | pass | 2/2 | hit |
```

**Where to read more:** the [`baremetal_agent/eval/`](../baremetal_agent/eval/)
subpackage. Start with `runner.py` for the per-task loop, `loader.py` for task
parsing, `rubric.py` for the check dispatcher, `cache.py` for cache reuse, and
`report.py` for the Markdown/JSON output.
