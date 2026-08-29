"""A tiny local web chat for the Trailhead Hotel agent."""

from dataclasses import dataclass, field
from secrets import token_urlsafe
from threading import Lock

from flask import Flask, jsonify, render_template, request, session
from openai import OpenAIError

from sierra_agent.__main__ import build_agent
from sierra_agent.agent import AgentLoopError, SierraAgent
from sierra_agent.config import ConfigurationError
from sierra_agent.tools import DataError

app = Flask(__name__)
app.secret_key = token_urlsafe(32)


@dataclass
class ChatSession:
    """One browser's agent and a lock for its conversation state."""

    agent: SierraAgent
    lock: Lock = field(default_factory=Lock)


_sessions: dict[str, ChatSession] = {}
_sessions_lock = Lock()


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


@app.get("/")
def home():
    return render_template("index.html")


@app.post("/chat")
def chat():
    payload = request.get_json(silent=True) or {}
    message = str(payload.get("message", "")).strip()
    if not message:
        return jsonify({"error": "empty"}), 400

    try:
        chat_session = get_agent()
        with chat_session.lock:
            reply = chat_session.agent.reply(message)
    except (ConfigurationError, DataError):
        return _setup_error()
    except OpenAIError:
        return jsonify(
            {
                "reply": (
                    "I couldn't reach the support service. "
                    "Please try again in a moment."
                )
            }
        ), 502
    except (AgentLoopError, ValueError):
        return jsonify(
            {
                "reply": (
                    "I hit an unexpected issue. "
                    "Please try your request again."
                )
            }
        ), 500

    return jsonify({"reply": reply})


@app.post("/reset")
def reset():
    try:
        new_agent = build_agent()
    except (ConfigurationError, DataError):
        return _setup_error()

    with _sessions_lock:
        _sessions[_session_id()] = ChatSession(agent=new_agent)
    return jsonify({"ok": True})


def main() -> None:
    app.run(host="127.0.0.1", port=5050, debug=False)


if __name__ == "__main__":
    main()
