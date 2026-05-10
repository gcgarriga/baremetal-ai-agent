"""Convert conversation history to ATIF (Agent Trajectory Interchange Format).

Transforms the raw OpenAI-format message history and captured API responses
into the standardized ATIF-v1.4 JSON structure for debugging, replay, and
training data pipelines.

Spec: https://github.com/laude-institute/harbor/blob/main/docs/rfcs/0001-trajectory-format.md
"""

import json
import uuid
from datetime import datetime, timezone

from baremetal_agent import __version__, safety

SCHEMA_VERSION = "ATIF-v1.4"
FINAL_PROMPT_TOKENS_FIELD = "total_prompt_tokens"
FINAL_COMPLETION_TOKENS_FIELD = "total_completion_tokens"
FINAL_CACHED_TOKENS_FIELD = "total_cached_tokens"
FINAL_STEPS_FIELD = "total_steps"
FINAL_METRIC_FIELDS = (
    FINAL_PROMPT_TOKENS_FIELD,
    FINAL_COMPLETION_TOKENS_FIELD,
    FINAL_CACHED_TOKENS_FIELD,
    FINAL_STEPS_FIELD,
)


def _extract_metrics(response: dict) -> dict:
    """Pull token usage from an API response into ATIF metrics."""
    usage = response.get("usage", {})
    metrics = {
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "cached_tokens": usage.get("prompt_tokens_details", {}).get("cached_tokens", 0),
    }
    reasoning = usage.get("completion_tokens_details", {}).get("reasoning_tokens", 0)
    if reasoning:
        metrics["reasoning_tokens"] = reasoning
    return metrics


def _timestamp_from_response(response: dict) -> str:
    """Convert the API response's Unix 'created' field to ISO 8601."""
    created = response.get("created")
    if created is not None:
        return datetime.fromtimestamp(created, tz=timezone.utc).isoformat()
    return datetime.now(timezone.utc).isoformat()


def history_to_atif(
    history: list[dict],
    api_responses: list[dict],
    model: str,
    session_id: str | None = None,
) -> dict:
    """Convert raw conversation history and API responses into ATIF format.

    Walks the history list (system/user/assistant/tool messages) and groups
    them into ATIF steps, pairing each assistant message with the corresponding
    API response for metrics and timestamps.
    """
    if session_id is None:
        session_id = str(uuid.uuid4())

    steps: list[dict] = []
    step_id = 1
    resp_idx = 0  # tracks which API response corresponds to the current assistant msg
    now = datetime.now(timezone.utc).isoformat()
    # Track the model used by the most recent assistant turn so the top-level
    # model reflects the latest turn while per-step ``model_name`` records each
    # individual turn's model.
    latest_turn_model: str | None = None

    i = 0
    while i < len(history):
        msg = history[i]
        role = msg["role"]

        if role == "system":
            i += 1
            continue

        if role == "user":
            steps.append(
                {
                    "step_id": step_id,
                    "timestamp": now,
                    "source": "user",
                    "message": safety.sanitize_text(msg["content"], label="trajectory"),
                }
            )
            step_id += 1
            i += 1
            continue

        if role == "assistant":
            # Match this assistant message to its API response
            resp = api_responses[resp_idx] if resp_idx < len(api_responses) else {}
            ts = _timestamp_from_response(resp) if resp else now

            # Per-turn model: prefer the model recorded on the assistant message
            # itself (stamped by the agent loop at dispatch time), then the API
            # response's model echo, then the trajectory-level fallback.
            turn_model = msg.get("_model") or resp.get("model") or model
            latest_turn_model = turn_model

            step: dict = {
                "step_id": step_id,
                "timestamp": ts,
                "source": "agent",
                "model_name": turn_model,
            }

            if resp:
                step["metrics"] = _extract_metrics(resp)

            if msg.get("tool_calls"):
                # Agent step with tool calls
                tool_calls = []
                for tc in msg["tool_calls"]:
                    args = tc["function"]["arguments"]
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            args = {"_raw": args}
                    safe_args = safety.sanitize_json_value(args, label="trajectory")
                    tool_calls.append(
                        {
                            "tool_call_id": tc["id"],
                            "function_name": tc["function"]["name"],
                            "arguments": safe_args,
                        }
                    )
                step["tool_calls"] = tool_calls

                # Collect observation results from following tool messages
                results = []
                j = i + 1
                while j < len(history) and history[j]["role"] == "tool":
                    tool_msg = history[j]
                    results.append(
                        {
                            "source_call_id": tool_msg.get("tool_call_id", ""),
                            "content": safety.sanitize_text(tool_msg.get("content", ""), label="trajectory"),
                        }
                    )
                    j += 1
                if results:
                    step["observation"] = {"results": results}

                i = j  # skip past tool messages
            else:
                # Final text response
                step["message"] = safety.sanitize_text(msg.get("content", ""), label="trajectory")
                i += 1

            resp_idx += 1
            steps.append(step)
            step_id += 1
            continue

        # Skip standalone tool messages (already consumed above)
        i += 1

    # Aggregate final metrics
    total_prompt = sum(r.get("usage", {}).get("prompt_tokens", 0) for r in api_responses)
    total_completion = sum(r.get("usage", {}).get("completion_tokens", 0) for r in api_responses)
    total_cached = sum(
        r.get("usage", {}).get("prompt_tokens_details", {}).get("cached_tokens", 0) for r in api_responses
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "session_id": session_id,
        "agent": {
            "name": "baremetal-agent",
            "version": __version__,
            "model_name": latest_turn_model or model,
        },
        "steps": steps,
        "final_metrics": {
            FINAL_PROMPT_TOKENS_FIELD: total_prompt,
            FINAL_COMPLETION_TOKENS_FIELD: total_completion,
            FINAL_CACHED_TOKENS_FIELD: total_cached,
            FINAL_STEPS_FIELD: len(steps),
        },
    }


def save_trajectory(trajectory: dict, path: str) -> str:
    """Write an ATIF trajectory to a JSON file. Returns the path written."""
    safe_trajectory = safety.sanitize_json_value(trajectory, label="trajectory")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(safe_trajectory, f, indent=2, ensure_ascii=False)
    return path
