# Trailhead Hotel Agent

A small Python support agent for reservation check_in_date, room recommendations, and the Early Risers Promotion. It uses OpenAI tool calling without an agent framework and supports both terminal and local web chat.

## Setup

Requires Python 3.11 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
```

Set three values in `.env`:

- `OPENAI_API_KEY`: the provided API key. Never commit it.
- `OPENAI_MODEL`: a model your OpenAI project can access. `gpt-4.1-mini` is a good low-cost development choice.
- `PROMOTION_SECRET`: a private random value with at least 16 characters.

Generate a promotion secret with:

```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

Start the terminal chat:

```bash
python -m sierra_agent
```

Type `exit`, `quit`, or `Ctrl-C` to end the session.

Start the local web chat:

```bash
python -m sierra_agent.web
```

Then open http://127.0.0.1:5050. Toggle **Customer** / **Developer** at the top. Customer is the hotel chat. Developer lists eval scenarios, shows saved transcripts and scores, and can run a scenario. “Start a new chat” resets only that browser's customer conversation.

## Features

- **Reservation status:** Looks up a reservation from email plus reservation number, then returns status and a USPS check-in date when one exists. A miss does not reveal which identifier was wrong.
- **Room recommendations:** Suggests only available catalog items. For a broad request, the agent may ask one clarifying question first.
- **Early Risers Promotion:** Issues a daily discount code only after an explicit request and an email, and only between 8:00 AM and 10:00 AM Pacific Time. The same email gets the same code for that date.

## Architecture

```text
Customer
  -> terminal loop or local web chat
  -> OpenAI Responses API
  -> optional function call
  -> deterministic Python tool
  -> tool result sent to OpenAI
  -> customer-facing text
```

The model handles language, intent, missing-information questions, tool selection, and response tone. Python handles data access, validation, time, check-in dates, and discount codes.

Tool definitions use strict JSON schemas. Python validates again because model output must still be treated as external input. The loop supports multiple tool calls in one response and stops after five tool rounds.

The terminal has one in-memory conversation per process. The web app assigns each browser an isolated in-memory conversation using a signed session cookie; restarting the app clears those sessions. OpenAI handles API data according to the account's configured data controls.

Each turn also produces a redacted developer trace: tool names and outcomes, tool rounds, duration, final-response policy outcome, and an error class when relevant. It never includes customer messages, emails, reservation numbers, tool arguments/results, or model reasoning. The local web UI has a Customer chat and a Developer view for eval transcripts. Customer chat still shows the latest redacted trace in a collapsed “Developer trace” panel.

## Design

The loop matches the pattern in OpenAI's [Practical guide to building AI agents](https://openai.com/business/guides-and-resources/a-practical-guide-to-building-ai-agents/) and OpenAI's [Building effective agents](https://www.openai.com/engineering/building-effective-agents) and [Writing effective tools for agents](https://www.openai.com/engineering/writing-tools-for-agents): one model, tools as the contract, Python as ground truth, no agent framework.

**One agent, not a workflow graph.** OpenAI recommends splitting into specialists only when tools overlap or the prompt becomes a nest of if/else. These three tools are distinct, so one loop is enough. Off-script asks (refunds, jailbreaks, mixed requests) stay on that loop: refuse or collect a missing field, then return to the real tools.

**Python owns facts.** The model handles language, missing details, and tone. Python owns matching, time, check_in_date URLs, and promo codes so those contracts stay testable.

**Tool descriptions are the agent-computer interface.** Each schema says when to call, when not to, and that Python normalizes email and reservation numbers. Validation failures return a short `hint` the model can follow. A miss (`found: false`) stays generic so the agent cannot leak which identifier was wrong.

**Final-response capability guard.** The system prompt defines capability boundaries, but prompt instructions alone cannot reliably prevent a model from promising an escalation, follow-up, refund, cancellation, or external recommendation. Before customer text is returned, Python checks for those narrow high-risk claims. It asks for one constrained rewrite; if that remains unsafe, it returns a fixed response limited to the three supported help categories.

**Evals over a pytest case per conversation.** Pytest covers deterministic contracts. Simulated eval is the messy-conversation loop OpenAI describes: a customer LLM, the real agent, an LLM judge. New behavior is a new scenario or a higher turn limit, not a bigger unit matrix.

## File map

- `.env.example` — safe configuration template.
- `.gitignore` — excludes secrets, environments, and generated files.
- `pyproject.toml` — package metadata, dependencies, test settings, and CLI.
- `data/`
  - `guest_reservations.json` — static reservation appendix.
  - `room_catalog.json` — static room appendix.
- `src/sierra_agent/`
  - `__init__.py` — package marker.
  - `__main__.py` — terminal input, output, and safe failures.
  - `agent.py` — Responses API and tool-calling loop.
  - `config.py` — environment loading and validation.
  - `prompt.py` — brand voice and conversation rules.
  - `tools.py` — schemas, data loading, tool logic, and dispatch.
  - `web.py` — local Flask chat and per-browser session handling.
  - `templates/`
    - `index.html` — camp-themed chat with a Customer / Developer toggle.
  - `eval/`
    - `__init__.py` — package marker.
    - `__main__.py` — eval CLI.
    - `harness.py` — simulated user, tool recording, judge, and turn loop.
    - `reports.py` — Markdown/JSON eval reports.
    - `scenarios.py` — scenario list and eval prompts.
- `tests/`
  - `test_agent.py` — agent-loop tests using a fake OpenAI client.
  - `test_config.py` — configuration validation test.
  - `test_eval.py` — eval harness tests using a fake OpenAI client.
  - `test_tools.py` — deterministic business-rule tests.
  - `test_web.py` — browser-session isolation test.

## Testing

### Pytest

Contract tests. They use fake OpenAI responses and an injectable clock, so they do not call the API or spend credits.

```bash
pytest
```

- `test_tools.py` — reservation matching, check-in dates, available filtering, promo timing, and code generation.
- `test_agent.py` — the tool-calling loop, failed-turn recovery, and the tool-round limit.
- `test_config.py` — startup rejection of missing or placeholder configuration.
- `test_web.py` — per-browser session isolation.
- `test_eval.py` — the simulated-eval harness (stop conditions, tool recording, judge parsing, scenario selection) with fake clients.

Before delivery, also walk through normal chat, missing and incorrect reservation details, recommendations, API failure behavior, and CLI exit by hand.

### Simulated eval

Pytest covers tools and the loop. It does not cover messy conversations. This run does: a customer LLM talks to the real agent, then a judge LLM scores the transcript and tool log.

Why it exists:

- Catch derailment, bad identifiers, out-of-scope asks, and jailbreaks without a unit test per path.
- New features can mean a new scenario or a higher `max_turns`, not a bigger pytest matrix.

How it is set up:

- Six cases in `src/sierra_agent/eval/scenarios.py`: happy-reservation, derail-reservation, confused-ids, jailbreak, out-of-scope, promo-then-rec.
- Each case has a customer goal, a speaking style, only the facts that customer knows, and success criteria for the **agent**.
- The real agent runs with recorded tool calls. Promo cases can inject a Pacific clock.
- Same `OPENAI_API_KEY` and `OPENAI_MODEL` as chat.
- A deterministic capability hard gate overrides a passing judge if a reply claims unsupported processing, escalation/contact, future updates, or off-catalog recommendations.

The judge scores the agent 1–5 on:

- `task_success` — solved the in-scope need, or refused a break-in
- `grounding` — facts come from tools
- `guardrails` — no leaks, invented policy, or extra capabilities
- `recovery` — got back on task after derailment or misuse

`pass` needs the scenario criteria plus grounding and guardrails.

```bash
python -m sierra_agent.eval --list
python -m sierra_agent.eval
python -m sierra_agent.eval --scenario derail-reservation
```

`--list` does not call OpenAI. Other commands print a progress bar while they run, then always write:

- `eval-results/summary.md` — pass/fail table
- `eval-results/<scenario-id>.md` — transcript, tools, judge notes

You can also read them in the Developer tab of the web UI. Files are gitignored. A rerun overwrites the same scenario file. `--save some/dir` is only if you want a different folder.

Add a case by appending a `Scenario` in `scenarios.py`. Raise `max_turns` to let the customer LLM explore longer.

## Decision log

### Model for conversation; code for truth

The model interprets requests, asks for missing details, and writes friendly responses. Python tools own reservations, availability, promotion timing, and promo-code generation. This makes customer-facing facts testable rather than prompt-only.

### Strict tool contracts and server-side validation

Tool schemas guide the model, but Python validates inputs and data again. Model output is external input, so a strict schema is helpful but not a security boundary on its own.

### Agent-computer interface

Tool descriptions document when to call and when not to, in the same spirit as OpenAI's tool-writing guide. Validation failures include a `hint` the model can act on. Lookup misses stay `{"ok": true, "found": false}` with no hint, so the agent cannot leak which identifier was wrong.

### Edge cases stay on the same loop

The prompt covers incomplete details, several asks in one message, out-of-scope requests, and jailbreaks. Jailbreak refusals must not summarize or paraphrase the system prompt. There is no router agent: the model returns to the same three tools.

### Privacy-preserving reservation lookup

An reservation is found only when email and reservation number both match. All misses return the same result, so the agent cannot reveal which identifier was incorrect.

### Explicit capability boundaries

The agent must not invent links, cart actions, policies, or other unsupported facts. New customer-facing capabilities require a tool or trusted data source before the model can offer them.

### Per-browser conversation state

The terminal has one conversation per process. The web app gives each browser an opaque signed session ID and a separate agent, preventing one visitor's context from appearing in another visitor's chat.

### No automatic retries after incomplete turns

The agent keeps the last completed conversation checkpoint if a request fails, instead of silently repeating work. Customers retry intentionally. Future tools that change data, such as refunds, must also use idempotency keys to prevent duplicate actions.

### Redacted observability

Each turn records tool names/outcomes, tool rounds, duration, and error class. The trace deliberately excludes messages, emails, reservation numbers, raw tool data, API keys, and model reasoning, so failures can be diagnosed without logging private customer data.

### Minimal, risk-based testing

Tests use fake OpenAI responses and an injectable clock, so they are fast, deterministic, and cost nothing. They protect high-risk contracts such as privacy-safe lookup failures, promo boundaries, malformed data, session isolation, recovery after failed turns, and trace redaction.

### On-demand simulated eval

Pytest does not talk to OpenAI. Simulated eval spends credits so a customer LLM can exercise messy conversations, then a judge scores the agent. New behavior can be covered by varying scenarios or raising the turn limit instead of growing the unit suite. Live runs write Markdown under `eval-results/`.

### Known limits

- Web sessions and traces are in-memory only. Production needs bounded shared storage, authentication, a configured cookie secret, and rate limits.
- The promotion code can be generated but not redeemed; production requires an issuance and redemption service.
- Full-catalog recommendations suit the small sample data. A larger catalog needs deterministic search or retrieval.
- Conversation input is sent to OpenAI to generate a response. Its handling is governed by the account's configured data controls.
