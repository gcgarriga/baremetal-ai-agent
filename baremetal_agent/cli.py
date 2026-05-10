"""Command-line interface for the baremetal agent."""

import argparse
import json
import sys
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path

from baremetal_agent import agent, replay, tools, trajectory
from baremetal_agent import eval as eval_harness
from baremetal_agent.config import AgentConfig, load_config
from baremetal_agent.visualizer import NullRenderer, make_renderer


def interactive_confirmer(tool_name: str, arguments: dict) -> bool:
    """Prompt the user before executing a confirmation-required tool.

    Lives in the CLI layer because it's the only place where reading from
    stdin is appropriate. Returns ``False`` on EOF/Ctrl-C so the agent loop
    treats those signals as a denial rather than an exception.
    """
    args_str = json.dumps(arguments, indent=2)
    print(f"\n⚠️  Tool '{tool_name}' requires confirmation.")
    print(f"   Arguments: {args_str}")
    try:
        answer = input("   Execute? [y/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return False
    return answer in ("y", "yes")


def _print_banner(cfg: AgentConfig) -> None:
    """Print the startup banner with configuration and system prompt."""
    tool_names = ", ".join(tools.get_tool_names())
    print()
    print("═" * 62)
    print("  🔧  Baremetal Tool-Use Agent")
    print("═" * 62)
    print(f"  Model:    {cfg.model}")
    print(f"  API:      {cfg.api_url}")
    print(f"  Tools:    {len(tools.TOOLS)} registered ({tool_names})")
    print(f"  Max iter: {cfg.max_iterations}")
    print(f"  Work dir: {cfg.working_dir}")
    render_state = "plain" if cfg.render_verbose else "rich"
    payload_state = "on" if cfg.log_payloads else "off"
    print(f"  Renderer: {render_state}    Payload logging: {payload_state}")
    print("─" * 62)
    print("  System Prompt:")
    for line in cfg.system_prompt.splitlines():
        print(f"    {line}")
    print("─" * 62)
    print("  Type 'help' for commands, or just ask a question.")
    print("═" * 62)
    print()


def _cmd_help() -> None:
    """Print available commands."""
    print()
    print("Commands:")
    print("  help           Show this help message")
    print("  tools          List registered tools with descriptions")
    print("  history        Show conversation history")
    print("  trajectory     Export conversation as ATIF trajectory JSON")
    print("  clear          Reset conversation history")
    print("  model <name>   Switch to a different model")
    print("  verbose        Toggle verbose mode (plain renderer + raw API payloads)")
    print("  stream         Toggle streaming responses on/off")
    print("  exit / quit    Exit the agent")
    print()


def _cmd_tools() -> None:
    """List all registered tools with descriptions."""
    print()
    for name, tool in tools.TOOLS.items():
        desc = tool["definition"]["function"]["description"]
        confirm = " ⚠️  (requires confirmation)" if tool["requires_confirmation"] else ""
        print(f"  {name}{confirm}")
        print(f"    {desc}")
        params = tool["definition"]["function"]["parameters"]["properties"]
        if params:
            for pname, pdef in params.items():
                req = pname in tool["definition"]["function"]["parameters"].get("required", [])
                req_tag = " (required)" if req else ""
                print(f"      - {pname}: {pdef.get('type', '?')}{req_tag} — {pdef.get('description', '')}")
        print()


def _cmd_history(history: list[dict]) -> None:
    """Show conversation history with role and content summaries."""
    print()
    if len(history) <= 1:  # only system prompt
        print("  (no conversation history)")
        print()
        return

    for i, msg in enumerate(history):
        role = msg["role"]
        if role == "system":
            continue

        if role == "tool":
            content = msg.get("content", "")
            preview = content[:100].replace("\n", "\\n")
            call_id = msg.get("tool_call_id", "?")
            print(f"  [{i}] tool (id={call_id}): {preview}...")
        elif role == "assistant" and msg.get("tool_calls"):
            calls = msg["tool_calls"]
            names = [c["function"]["name"] for c in calls]
            print(f"  [{i}] assistant → tool_calls: {', '.join(names)}")
        else:
            content = msg.get("content", "")
            preview = content[:100].replace("\n", "\\n") if content else "(empty)"
            print(f"  [{i}] {role}: {preview}")

    print()


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level CLI parser."""
    parser = argparse.ArgumentParser(
        prog="baremetal-agent",
        description="Run the Baremetal Agent REPL or execute a single prompt.",
    )
    prompt_group = parser.add_mutually_exclusive_group()
    prompt_group.add_argument("-p", "--prompt", help="Run one prompt and exit.")
    prompt_group.add_argument("--prompt-file", metavar="PATH", help="Read one prompt from a UTF-8 text file.")
    parser.add_argument("--trajectory-out", metavar="PATH", help="Write the one-shot run as ATIF-v1.4 JSON.")
    parser.add_argument("--stream", action="store_true", help="Stream one-shot assistant text as it arrives.")

    safety_group = parser.add_mutually_exclusive_group()
    safety_group.add_argument(
        "--allow-dangerous-tools",
        action="store_true",
        help="Allow confirmation-required tools in one-shot mode.",
    )
    safety_group.add_argument(
        "--read-only",
        action="store_true",
        help="Expose only tools that do not require confirmation in one-shot mode.",
    )
    subparsers = parser.add_subparsers(dest="command")
    replay_parser = subparsers.add_parser("replay", help="Inspect an ATIF-v1.4 trajectory offline.")
    replay_parser.add_argument("path", help="Path to the ATIF-v1.4 trajectory JSON file.")
    replay_mode = replay_parser.add_mutually_exclusive_group()
    replay_mode.add_argument("--step", type=int, help="Render only the matching ATIF step_id.")
    replay_mode.add_argument("--diff", metavar="PATH", help="Compare this trajectory with another ATIF JSON file.")
    eval_parser = subparsers.add_parser("eval", help="Run eval tasks and write reports.")
    eval_parser.add_argument("--tasks", default="evals/tasks", help="Directory containing eval task files.")
    eval_parser.add_argument("--out", default="report.md", help="Output path for Markdown summary report.")
    eval_parser.add_argument("--json-out", default="report.json", help="Output path for JSON report.")
    eval_parser.add_argument(
        "--workers",
        type=int,
        default=1,
        metavar="N",
        help=(
            "Number of worker threads for parallel task execution (default: 1). "
            "With N>1, tasks run concurrently via ThreadPoolExecutor. Each task "
            "still runs in its own isolated sandbox, but stdout log lines from "
            "different tasks may interleave."
        ),
    )
    return parser


def _read_prompt_file(path: str) -> str:
    """Read prompt text from a UTF-8 file."""
    return Path(path).read_text(encoding="utf-8")


def _run_one_shot(prompt: str, args: argparse.Namespace, cfg: AgentConfig) -> int:
    """Run a single prompt and return a process exit code."""
    history: list[agent.Message] = [{"role": "system", "content": cfg.system_prompt}]
    api_responses: list[dict] = []
    confirmer = agent.auto_approve_confirmer if args.allow_dangerous_tools else agent.auto_deny_confirmer
    tool_names = tools.get_read_only_tool_names() if args.read_only else None
    if args.stream and not cfg.stream:
        cfg = replace(cfg, stream=True)
    streamed_chars: list[str] = []
    stdout = sys.stdout

    def on_stream_delta(text: str) -> None:
        streamed_chars.append(text)
        print(text, end="", file=stdout, flush=True)

    with redirect_stdout(sys.stderr):
        result = agent.run_agent_turn(
            prompt,
            history,
            api_responses,
            cfg=cfg,
            renderer=NullRenderer(),
            confirmer=confirmer,
            tool_names=tool_names,
            on_stream_delta=on_stream_delta if cfg.stream else None,
        )

    if result.status == agent.STATUS_OK:
        if cfg.stream and streamed_chars:
            if not "".join(streamed_chars).endswith("\n"):
                print(file=stdout)
        else:
            print(result.content)
    else:
        print(result.content, file=sys.stderr)

    if args.trajectory_out:
        atif = trajectory.history_to_atif(history, api_responses, cfg.model)
        try:
            trajectory.save_trajectory(atif, args.trajectory_out)
        except OSError as exc:
            print(f"Failed to write trajectory: {exc}", file=sys.stderr)
            return 1

    return 0 if result.status == agent.STATUS_OK else 1


def _run_replay(args: argparse.Namespace) -> int:
    """Run the offline trajectory replay command."""
    try:
        atif = replay.load_trajectory(args.path)
        if args.diff:
            other = replay.load_trajectory(args.diff)
            print(replay.diff(atif, other))
        else:
            replay.render(atif, step_id=args.step)
    except ValueError as exc:
        print(f"replay: {exc}", file=sys.stderr)
        return 1
    return 0


def _run_eval(args: argparse.Namespace, cfg: AgentConfig) -> int:
    """Run the eval suite command."""
    return eval_harness.run_eval_suite(args.tasks, args.out, args.json_out, cfg=cfg, workers=args.workers)


def main(argv: list[str] | None = None) -> int:
    """Route command-line args to the REPL or one-shot mode."""
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        cfg = load_config()
    except ValueError as exc:
        parser.error(str(exc))

    if args.command == "replay":
        if (
            args.prompt is not None
            or args.prompt_file is not None
            or args.trajectory_out is not None
            or args.stream
            or args.allow_dangerous_tools
            or args.read_only
        ):
            parser.error("replay cannot be combined with one-shot prompt options")
        return _run_replay(args)

    if args.command == "eval":
        if (
            args.prompt is not None
            or args.prompt_file is not None
            or args.trajectory_out is not None
            or args.stream
            or args.allow_dangerous_tools
            or args.read_only
        ):
            parser.error("eval cannot be combined with one-shot prompt options")
        if args.workers < 1:
            parser.error(f"--workers must be >= 1 (got {args.workers})")
        return _run_eval(args, cfg)

    if args.prompt is None and args.prompt_file is None:
        if args.trajectory_out:
            parser.error("--trajectory-out requires --prompt or --prompt-file")
        if args.stream:
            parser.error("--stream requires --prompt or --prompt-file")
        if args.allow_dangerous_tools or args.read_only:
            parser.error("--allow-dangerous-tools and --read-only require --prompt or --prompt-file")
        return run(cfg=cfg)

    try:
        prompt = args.prompt if args.prompt is not None else _read_prompt_file(args.prompt_file)
    except (OSError, UnicodeError) as exc:
        parser.exit(2, f"{parser.prog}: error: could not read prompt file {args.prompt_file!r}: {exc}\n")

    return _run_one_shot(prompt, args, cfg)


def run(cfg: AgentConfig | None = None) -> int:
    """Run the interactive REPL."""
    if cfg is None:
        cfg = load_config()
    _print_banner(cfg)
    renderer = make_renderer(cfg)

    # Initialize conversation history and response log — owned here, passed down
    history: list[dict] = [{"role": "system", "content": cfg.system_prompt}]
    api_responses: list[dict] = []

    while True:
        try:
            user_input = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            return 0

        if not user_input:
            continue

        # Handle special commands
        cmd = user_input.lower()

        if cmd in ("exit", "quit"):
            print("Goodbye!")
            return 0

        if cmd == "help":
            _cmd_help()
            continue

        if cmd == "tools":
            _cmd_tools()
            continue

        if cmd == "history":
            _cmd_history(history)
            continue

        if cmd == "clear":
            history.clear()
            history.append({"role": "system", "content": cfg.system_prompt})
            api_responses.clear()
            print("\n  Conversation history cleared.\n")
            continue

        if cmd == "trajectory" or cmd.startswith("trajectory "):
            parts = user_input.split(maxsplit=1)
            path = parts[1] if len(parts) > 1 else "trajectory.json"
            atif = trajectory.history_to_atif(history, api_responses, cfg.model)
            try:
                trajectory.save_trajectory(atif, path)
            except OSError as exc:
                print(f"\n  ❌ Failed to write trajectory: {exc}\n")
                continue
            n_steps = atif["final_metrics"]["total_steps"]
            tokens = atif["final_metrics"]["total_prompt_tokens"] + atif["final_metrics"]["total_completion_tokens"]
            print(f"\n  ✅ Trajectory exported: {path}")
            print(f"     {n_steps} steps, {tokens} total tokens")
            print("     Format: ATIF-v1.4\n")
            continue

        if cmd.startswith("model"):
            new_model = user_input[5:].strip()
            if new_model:
                old = cfg.model
                cfg = replace(cfg, model=new_model)
                print(f"\n  Model changed: {old} → {new_model}\n")
            else:
                print(f"\n  Current model: {cfg.model}\n")
            continue

        if cmd == "verbose":
            # One-knob backward-compat: toggle both flags together. They are
            # always either both on (raw payloads + plain renderer) or both
            # off (rich UI, no payload logs); if they have drifted apart via
            # env vars we collapse to the inverse of `render_verbose`.
            new_value = not cfg.render_verbose
            cfg = replace(cfg, render_verbose=new_value, log_payloads=new_value)
            renderer = make_renderer(cfg)
            state = "on (raw API payloads)" if new_value else "off (rich visualization)"
            print(f"\n  Verbose: {state}\n")
            continue

        if cmd == "stream":
            cfg = replace(cfg, stream=not cfg.stream)
            state = "on" if cfg.stream else "off"
            print(f"\n  Streaming: {state}\n")
            continue

        # Send to the agent
        print()
        result = agent.run_agent_turn(
            user_input,
            history,
            api_responses,
            cfg=cfg,
            renderer=renderer,
            confirmer=interactive_confirmer,
        )
        if cfg.log_payloads:
            print("─" * 62)
            print(result.content)
            print("─" * 62)
        print()
