"""CLI for the on-demand adversarial eval suite."""

import argparse
from pathlib import Path
import sys

from openai import OpenAI, OpenAIError

from sierra_agent.config import PROJECT_ROOT, ConfigurationError, Settings
from sierra_agent.eval.harness import SCORE_KEYS, EvalRun, execute_scenario
from sierra_agent.eval.reports import format_report, write_reports
from sierra_agent.eval.scenarios import SCENARIOS, select_scenarios
from sierra_agent.tools import DataError


def main(argv: list[str] | None = None) -> int:
    """List scenarios or run the eval suite. Spends OpenAI credits."""

    parser = argparse.ArgumentParser(
        prog="python -m sierra_agent.eval",
        description=(
            "Run simulated-customer conversations against the Sierra agent "
            "and score them with a judge model. Uses OPENAI_API_KEY and "
            "OPENAI_MODEL; OPENAI_EVAL_MODEL is optional. This is not pytest "
            "and is not free."
        ),
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print scenario ids and titles, then exit.",
    )
    parser.add_argument(
        "--scenario",
        metavar="ID",
        help="Run one scenario instead of the full suite.",
    )
    parser.add_argument(
        "--save",
        dest="save_dir",
        type=Path,
        default=PROJECT_ROOT / "eval-results",
        help="Directory for Markdown reports (default: eval-results/).",
    )
    args = parser.parse_args(argv)

    if args.list:
        _print_scenario_list()
        return 0

    try:
        scenarios = select_scenarios(args.scenario)
    except ValueError as error:
        print(error, file=sys.stderr)
        return 1

    try:
        settings = Settings.from_env()
    except ConfigurationError as error:
        print(f"Setup error: {error}", file=sys.stderr)
        return 1

    try:
        client = OpenAI(api_key=settings.openai_api_key)
        total = len(scenarios)
        print(f"Running {total} scenario{'s' if total != 1 else ''}...", file=sys.stderr)
        runs: list[EvalRun] = []
        for index, scenario in enumerate(scenarios, start=1):
            progress = _CliProgress(index, total, scenario.id)
            run = execute_scenario(
                client,
                settings.openai_model,
                settings.openai_eval_model,
                scenario,
                settings.data_dir,
                settings.promotion_secret,
                on_progress=progress,
            )
            progress.finish(
                "PASS" if run.judgment.get("pass") else "FAIL"
            )
            runs.append(run)
    except (OpenAIError, DataError, ValueError) as error:
        print(f"Eval error: {error}", file=sys.stderr)
        return 1

    for run in runs:
        print(format_report(run))
        print()
    print(_format_summary(runs))

    saved = write_reports(runs, args.save_dir)
    print()
    print("Saved:")
    for path in saved:
        print(f"  {path}")

    return 0 if all(run.judgment.get("pass") for run in runs) else 1


def _print_scenario_list() -> None:
    width = max(len(scenario.id) for scenario in SCENARIOS)
    for scenario in SCENARIOS:
        print(f"{scenario.id.ljust(width)}  {scenario.title}")


def _progress_bar(completed: int, total: int, width: int = 18) -> str:
    """ASCII bar. `completed` is how many scenarios are finished."""

    if total <= 0:
        filled = width
    else:
        filled = min(width, max(0, round(width * completed / total)))
    return f"[{'#' * filled}{'-' * (width - filled)}]"


class _CliProgress:
    """Rewrite one stderr line as a scenario runs."""

    def __init__(self, index: int, total: int, scenario_id: str) -> None:
        self.index = index
        self.total = total
        self.scenario_id = scenario_id
        self._draw("starting")

    def __call__(self, event: str, info: dict[str, object]) -> None:
        turn = info.get("turn")
        max_turns = info.get("max_turns")
        if event == "customer":
            self._draw(f"turn {turn}/{max_turns}  customer")
        elif event == "agent":
            self._draw(f"turn {turn}/{max_turns}  agent")
        elif event == "judge":
            self._draw("judging")

    def finish(self, verdict: str) -> None:
        self._draw(verdict, completed=True)
        print(file=sys.stderr)

    def _draw(self, status: str, completed: bool = False) -> None:
        filled = self.index if completed else self.index - 1
        bar = _progress_bar(filled, self.total)
        line = (
            f"\r{bar} {self.index}/{self.total}  "
            f"{self.scenario_id}  {status}"
        )
        padded = line.ljust(72)
        print(padded, end="", file=sys.stderr, flush=True)


def _format_summary(runs: list[EvalRun]) -> str:
    width = max(len(run.scenario.id) for run in runs)
    lines = ["----"]
    passed = 0
    for run in runs:
        lines.append(
            f"{run.scenario.id.ljust(width)}  {_format_judgment_line(run)}"
        )
        if run.judgment.get("pass"):
            passed += 1
    failed = len(runs) - passed
    lines.append("----")
    lines.append(f"{passed} passed, {failed} failed")
    return "\n".join(lines)


def _format_judgment_line(run: EvalRun) -> str:
    verdict = "PASS" if run.judgment.get("pass") else "FAIL"
    scores = run.judgment.get("scores") or {}
    parts = [str(scores.get(key, "-")) for key in SCORE_KEYS]
    return f"{verdict}  task/grounding/guardrails/recovery {'/'.join(parts)}"


if __name__ == "__main__":
    raise SystemExit(main())
