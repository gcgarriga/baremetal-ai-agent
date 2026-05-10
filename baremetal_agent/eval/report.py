"""Markdown and JSON eval report rendering."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from baremetal_agent.safety import sanitize_text

if TYPE_CHECKING:
    from baremetal_agent.config import AgentConfig

    from .runner import EvalResult


def _result_summary_counts(results: list[EvalResult]) -> dict[str, int]:
    from .runner import STATUS_ERROR, STATUS_FAIL, STATUS_PASS, STATUS_SKIPPED

    counts = {STATUS_PASS: 0, STATUS_FAIL: 0, STATUS_ERROR: 0, STATUS_SKIPPED: 0}
    for result in results:
        if result.status in counts:
            counts[result.status] += 1
    return counts


def _sanitize_report_string(value: str) -> str:
    return sanitize_text(value, label="eval_report")


def _sanitize_optional_report_string(value: str | None) -> str | None:
    if value is None:
        return None
    return _sanitize_report_string(value)


def render_markdown_report(results: list[EvalResult], *, cfg: AgentConfig) -> str:
    from .runner import STATUS_ERROR, STATUS_FAIL, STATUS_PASS, STATUS_SKIPPED

    counts = _result_summary_counts(results)
    lines = [
        "# Baremetal Agent Eval Report",
        "",
        f"- Model: `{_sanitize_report_string(cfg.model)}`",
        f"- Total tasks: {len(results)}",
        f"- {STATUS_PASS}: {counts[STATUS_PASS]}",
        f"- {STATUS_FAIL}: {counts[STATUS_FAIL]}",
        f"- {STATUS_ERROR}: {counts[STATUS_ERROR]}",
        f"- {STATUS_SKIPPED}: {counts[STATUS_SKIPPED]}",
        "",
        "| Task | Status | Checks | Cache |",
        "| --- | --- | --- | --- |",
    ]

    for result in results:
        total_checks = len(result.checks)
        passed_checks = sum(1 for check in result.checks if check.passed)
        checks_summary = f"{passed_checks}/{total_checks}" if total_checks > 0 else "-"
        cache_status = "hit" if result.cached else "miss"
        if result.status == STATUS_SKIPPED:
            cache_status = "-"
        lines.append(f"| {result.task_id} | {result.status} | {checks_summary} | {cache_status} |")

    for result in results:
        description = _sanitize_report_string(result.description)
        lines.extend(
            [
                "",
                f"## {result.task_id} — {description}",
                f"- Status: {result.status}",
            ]
        )
        if result.trajectory_path:
            lines.append(f"- Trajectory: `{result.trajectory_path}`")
        if result.error:
            lines.append(f"- Error: {_sanitize_report_string(result.error)}")
        if result.checks:
            lines.append("- Checks:")
            for check in result.checks:
                symbol = "✅" if check.passed else "❌"
                message = _sanitize_report_string(check.message)
                lines.append(f"  - {symbol} {check.type}: {message}")
        else:
            lines.append("- Checks: none")

    return "\n".join(lines) + "\n"


def render_json_report(results: list[EvalResult], *, cfg: AgentConfig) -> dict:
    from .runner import STATUS_ERROR, STATUS_FAIL, STATUS_PASS, STATUS_SKIPPED

    counts = _result_summary_counts(results)
    return {
        "model": _sanitize_report_string(cfg.model),
        "summary": {
            "total": len(results),
            STATUS_PASS: counts[STATUS_PASS],
            STATUS_FAIL: counts[STATUS_FAIL],
            STATUS_ERROR: counts[STATUS_ERROR],
            STATUS_SKIPPED: counts[STATUS_SKIPPED],
        },
        "results": [
            {
                "task_id": result.task_id,
                "description": _sanitize_report_string(result.description),
                "status": result.status,
                "cached": result.cached,
                "trajectory_path": result.trajectory_path,
                "checks": [
                    {
                        "type": check.type,
                        "passed": check.passed,
                        "message": _sanitize_report_string(check.message),
                    }
                    for check in result.checks
                ],
                "error": _sanitize_optional_report_string(result.error),
            }
            for result in results
        ],
    }


def _write_report_outputs(
    markdown_report: str,
    json_report: dict,
    out_path: str | Path,
    json_out_path: str | Path,
) -> None:
    markdown_output_path = Path(out_path)
    json_output_path = Path(json_out_path)
    try:
        markdown_output_path.parent.mkdir(parents=True, exist_ok=True)
        json_output_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_output_path.write_text(markdown_report, encoding="utf-8")
        json_output_path.write_text(json.dumps(json_report, indent=2, ensure_ascii=False), encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"could not write eval report output: {exc}") from exc
