"""A tiny local web chat for the Trailhead Hotel agent."""

from dataclasses import dataclass, field
from secrets import token_urlsafe
from threading import Lock

from flask import Flask, jsonify, render_template, request, session
from openai import OpenAI, OpenAIError

from sierra_agent.__main__ import build_agent
from sierra_agent.agent import AgentLoopError, SierraAgent
from sierra_agent.config import PROJECT_ROOT, ConfigurationError, Settings
from sierra_agent.eval.harness import execute_scenario
from sierra_agent.eval.reports import load_result, scenario_listing, serialize_run, write_reports
from sierra_agent.eval.scenarios import SCENARIOS, select_scenarios
from sierra_agent.tools import DataError

app = Flask(__name__)
app.secret_key = token_urlsafe(32)

EVAL_RESULTS_DIR = PROJECT_ROOT / "eval-results"


@dataclass
class ChatSession:
    """One browser's agent and a lock for its conversation state."""

    agent: SierraAgent
    lock: Lock = field(default_factory=Lock)


_sessions: dict[str, ChatSession] = {}
_sessions_lock = Lock()
_eval_lock = Lock()


def _session_id() -> str:
    browser_session_id = session.get("browser_session_id")
    if not isinstance(browser_session_id, str):
        browser_session_id = token_urlsafe(32)
        session["browser_session_id"] = browser_session_id
    return browser_session_id


def get_agent() -> ChatSession:
    """Return the agent dedicated to the current browser session."""

    browser_session_id = _session_id()
    with _sessions_lock:
        chat_session = _sessions.get(browser_session_id)
        if chat_session is None:
            chat_session = ChatSession(agent=build_agent())
            _sessions[browser_session_id] = chat_session
        return chat_session


def _setup_error():
    return jsonify({"reply": "The guest desk needs a quick setup check."}), 500


def _chat_response(message: str, chat_session: ChatSession, status: int = 200):
    return (
        jsonify({"reply": message, "trace": chat_session.agent.last_trace}),
        status,
    )


@app.get("/")
def home():
    return render_template("index.html")


@app.post("/chat")
def chat():
    payload = request.get_json(silent=True) or {}
    message = str(payload.get("message", "")).strip()
    if not message:
        return jsonify({"error": "empty"}), 400

    chat_session: ChatSession | None = None
    try:
        chat_session = get_agent()
        with chat_session.lock:
            reply = chat_session.agent.reply(message)
    except (ConfigurationError, DataError):
        return _setup_error()
    except OpenAIError:
        if chat_session is not None:
            return _chat_response(
                "I couldn't reach the support service. Please try again in a moment.",
                chat_session,
                502,
            )
        return _setup_error()
    except (AgentLoopError, ValueError):
        if chat_session is not None:
            return _chat_response(
                "I hit an unexpected issue. Please try your request again.",
                chat_session,
                500,
            )
        return _setup_error()

    return _chat_response(reply, chat_session)


@app.post("/reset")
def reset():
    try:
        new_agent = build_agent()
    except (ConfigurationError, DataError):
        return _setup_error()

    with _sessions_lock:
        _sessions[_session_id()] = ChatSession(agent=new_agent)
    return jsonify({"ok": True})


@app.get("/eval/scenarios")
def eval_scenarios():
    """List eval cases and any saved pass/fail marks."""

    return jsonify({"scenarios": scenario_listing(SCENARIOS, EVAL_RESULTS_DIR)})


@app.get("/eval/results/<scenario_id>")
def eval_result(scenario_id: str):
    """Return one saved transcript and score, if present."""

    try:
        select_scenarios(scenario_id)
    except ValueError:
        return jsonify({"error": "unknown scenario"}), 404
    saved = load_result(EVAL_RESULTS_DIR, scenario_id)
    if saved is None:
        return jsonify({"error": "no saved result"}), 404
    return jsonify(saved)


@app.post("/eval/run")
def eval_run():
    """Run one simulated-eval scenario, save it, and return the result."""

    payload = request.get_json(silent=True) or {}
    scenario_id = str(payload.get("scenario", "")).strip()
    try:
        scenarios = select_scenarios(scenario_id)
    except ValueError:
        return jsonify({"error": "unknown scenario"}), 404

    try:
        settings = Settings.from_env()
        client = OpenAI(api_key=settings.openai_api_key)
        with _eval_lock:
            run = execute_scenario(
                client,
                settings.openai_model,
                scenarios[0],
                settings.data_dir,
                settings.promotion_secret,
            )
            write_reports([run], EVAL_RESULTS_DIR)
    except ConfigurationError:
        return _setup_error()
    except (OpenAIError, DataError, ValueError):
        return jsonify({"error": "eval failed"}), 502

    return jsonify(serialize_run(run))


def main() -> None:
    app.run(host="127.0.0.1", port=5050, debug=False)


if __name__ == "__main__":
    main()
