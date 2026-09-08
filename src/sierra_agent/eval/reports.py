"""Markdown and JSON reports for simulated-eval runs."""

import json
from pathlib import Path
from typing import Any

from sierra_agent.eval.harness import SCORE_KEYS, EvalRun
from sierra_agent.eval.scenarios import Scenario


def tool_outcome(result: object) -> str:
    """Summarize a tool result the same way the CLI report does."""

    if not isinstance(result, dict):
        return "unknown"
    if not result.get("ok"):
        return "rejected"
    if "found" in result:
        return "found" if result["found"] else "not_found"
    if "eligible" in result:
        return "eligible" if result["eligible"] else "not_eligible"
    return "completed"


def serialize_run(run: EvalRun) -> dict[str, Any]:
    """JSON-friendly transcript, scores, and tool outcomes."""

    turns: list[dict[str, Any]] = []
    for turn in run.turns:
        item: dict[str, Any] = {
            "customer": turn.customer,
            "agent": turn.agent,
        }
        if turn.error is not None:
            item["error"] = turn.error
        turns.append(item)
    return {
        "scenario_id": run.scenario.id,
        "title": run.scenario.title,
        "interview_note": run.scenario.interview_note,
        "stop_reason": run.stop_reason,
        "pass": bool(run.judgment.get("pass")),
        "scores": run.judgment.get("scores") or {},
        "notes": str(run.judgment.get("notes", "")),
        "turns": turns,
        "tools": [
            {
                "name": str(call.get("name", "")),
                "outcome": tool_outcome(call.get("result", {})),
            }
            for call in run.tool_calls
        ],
    }


def format_report(run: EvalRun) -> str:
    """Readable Markdown transcript for one scenario."""

    scores = run.judgment.get("scores") or {}
    score_line = " / ".join(
        f"{key} {scores.get(key, '-')}" for key in SCORE_KEYS
    )
    verdict = "PASS" if run.judgment.get("pass") else "FAIL"
    lines = [
        f"# {run.scenario.id}: {run.scenario.title}",
        "",
        f"**Result:** {verdict}",
        f"**Scores:** {score_line}",
        f"**Interview note:** {run.scenario.interview_note}",
        f"**Stop reason:** {run.stop_reason}",
        "",
        "## Transcript",
        "",
    ]
    if not run.turns:
        lines.append("(no customer turns)")
        lines.append("")
    for turn in run.turns:
        lines.append(f"**You:** {turn.customer}")
        lines.append("")
        if turn.error is not None:
            lines.append(f"**Agent error:** {turn.error}")
        else:
            lines.append(f"**Agent:** {turn.agent}")
        lines.append("")
    lines.append("## Tools")
    lines.append("")
    if run.tool_calls:
        for call in run.tool_calls:
            outcome = tool_outcome(call.get("result", {}))
            lines.append(f"- `{call['name']}` → {outcome}")
    else:
        lines.append("(none)")
    lines.append("")
    lines.append("## Judge")
    lines.append("")
    lines.append(str(run.judgment.get("notes", "")))
    return "\n".join(lines).rstrip()


def format_summary_markdown(runs: list[EvalRun]) -> str:
    """Index table linking to each scenario report."""

    passed = sum(1 for run in runs if run.judgment.get("pass"))
    failed = len(runs) - passed
    lines = [
        "# Eval summary",
        "",
        f"{passed} passed, {failed} failed",
        "",
        "| Scenario | Result | task | grounding | guardrails | recovery |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for run in runs:
        scores = run.judgment.get("scores") or {}
        verdict = "PASS" if run.judgment.get("pass") else "FAIL"
        cells = " | ".join(str(scores.get(key, "-")) for key in SCORE_KEYS)
        lines.append(
            f"| [{run.scenario.id}]({run.scenario.id}.md) | {verdict} | {cells} |"
        )
    return "\n".join(lines)


def write_reports(runs: list[EvalRun], directory: Path) -> list[Path]:
    """Write Markdown, JSON, and a summary index for the given runs."""

    directory.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for run in runs:
        markdown_path = directory / f"{run.scenario.id}.md"
        json_path = directory / f"{run.scenario.id}.json"
        markdown_path.write_text(format_report(run) + "\n", encoding="utf-8")
        json_path.write_text(
            json.dumps(serialize_run(run), indent=2) + "\n",
            encoding="utf-8",
        )
        written.extend([markdown_path, json_path])
    summary_path = directory / "summary.md"
    summary_path.write_text(format_summary_markdown(runs) + "\n", encoding="utf-8")
    written.append(summary_path)
    return written


def load_result(directory: Path, scenario_id: str) -> dict[str, Any] | None:
    """Load a saved run from JSON, or fall back to parsing Markdown."""

    json_path = directory / f"{scenario_id}.json"
    if json_path.is_file():
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            return payload
    markdown_path = directory / f"{scenario_id}.md"
    if markdown_path.is_file():
        return parse_report(markdown_path.read_text(encoding="utf-8"))
    return None


def parse_report(text: str) -> dict[str, Any]:
    """Parse a report produced by format_report."""

    stripped = text.strip()
    if not stripped.startswith("# "):
        raise ValueError("Report is missing a heading.")
    lines = stripped.splitlines()
    heading = lines[0][2:]
    scenario_id, _, title = heading.partition(": ")
    fields: dict[str, str] = {}
    index = 1
    while index < len(lines) and not lines[index].startswith("## "):
        line = lines[index]
        if line.startswith("**") and ":**" in line:
            key, _, value = line[2:].partition(":**")
            fields[key.strip().lower().replace(" ", "_")] = value.strip()
        index += 1

    turns: list[dict[str, Any]] = []
    tools: list[dict[str, str]] = []
    notes_lines: list[str] = []
    section = ""
    pending_customer: str | None = None

    def flush_agent(body: str, error: bool) -> None:
        nonlocal pending_customer
        if pending_customer is None:
            return
        item: dict[str, Any] = {"customer": pending_customer, "agent": ""}
        if error:
            item["error"] = body
        else:
            item["agent"] = body
        turns.append(item)
        pending_customer = None

    current_kind = ""
    current_body: list[str] = []

    def flush_body() -> None:
        nonlocal current_kind
        if current_kind == "you":
            pass
        elif current_kind == "agent":
            flush_agent("\n".join(current_body).strip(), error=False)
        elif current_kind == "error":
            flush_agent("\n".join(current_body).strip(), error=True)
        current_kind = ""
        current_body.clear()

    while index < len(lines):
        line = lines[index]
        if line.startswith("## "):
            flush_body()
            section = line[3:].strip().lower()
            index += 1
            continue
        if section == "transcript":
            if line.startswith("**You:**"):
                flush_body()
                pending_customer = line[len("**You:**") :].strip()
                current_kind = "you"
            elif current_kind == "you" and line.strip() and not line.startswith("**"):
                pending_customer = (
                    f"{pending_customer}\n{line}" if pending_customer else line
                )
            elif line.startswith("**Agent error:**"):
                flush_body()
                current_kind = "error"
                current_body.append(line[len("**Agent error:**") :].strip())
            elif line.startswith("**Agent:**"):
                flush_body()
                current_kind = "agent"
                current_body.append(line[len("**Agent:**") :].strip())
            elif current_kind in {"agent", "error"} and line.strip():
                current_body.append(line)
        elif section == "tools":
            stripped_line = line.strip()
            if stripped_line.startswith("- `") and "` → " in stripped_line:
                name = stripped_line[3:].split("`", 1)[0]
                outcome = stripped_line.split("` → ", 1)[1]
                tools.append({"name": name, "outcome": outcome})
        elif section == "judge":
            notes_lines.append(line)
        index += 1
    flush_body()

    scores: dict[str, int] = {}
    for part in fields.get("scores", "").split("/"):
        part = part.strip()
        if not part:
            continue
        name, _, value = part.partition(" ")
        if name in SCORE_KEYS and value.isdigit():
            scores[name] = int(value)

    return {
        "scenario_id": scenario_id.strip(),
        "title": title.strip(),
        "interview_note": fields.get("interview_note", ""),
        "stop_reason": fields.get("stop_reason", ""),
        "pass": fields.get("result") == "PASS",
        "scores": scores,
        "notes": "\n".join(notes_lines).strip(),
        "turns": turns,
        "tools": tools,
    }


def scenario_listing(scenarios: tuple[Scenario, ...], directory: Path) -> list[dict[str, Any]]:
    """Describe each scenario and attach a saved verdict when one exists."""

    listing: list[dict[str, Any]] = []
    for scenario in scenarios:
        item: dict[str, Any] = {
            "id": scenario.id,
            "title": scenario.title,
            "interview_note": scenario.interview_note,
        }
        saved = load_result(directory, scenario.id)
        if saved is not None:
            item["pass"] = saved.get("pass")
            item["scores"] = saved.get("scores") or {}
        listing.append(item)
    return listing
