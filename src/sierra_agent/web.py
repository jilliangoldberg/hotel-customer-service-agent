"""A tiny local web chat for the Trailhead Hotel agent."""

from flask import Flask, jsonify, render_template, request
from openai import OpenAIError

from sierra_agent.__main__ import build_agent
from sierra_agent.agent import AgentLoopError, SierraAgent
from sierra_agent.tools import DataError

app = Flask(__name__)
_agent: SierraAgent | None = None


def get_agent() -> SierraAgent:
    global _agent
    if _agent is None:
        _agent = build_agent()
    return _agent


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
        reply = get_agent().reply(message)
    except OpenAIError:
        return jsonify(
            {
                "reply": (
                    "I couldn't reach the support service. "
                    "Please try again in a moment."
                )
            }
        ), 502
    except (AgentLoopError, DataError, ValueError):
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
    global _agent
    _agent = build_agent()
    return jsonify({"ok": True})


def main() -> None:
    get_agent()
    app.run(host="127.0.0.1", port=5050, debug=False)


if __name__ == "__main__":
    main()
