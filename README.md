# Trailhead Hotel Agent

A small Python support agent for reservation check_in_date, room recommendations, and
the Early Risers Promotion. It uses OpenAI tool calling without an agent
framework and supports both terminal and local web chat.

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
- `OPENAI_MODEL`: a model your OpenAI project can access. `gpt-4.1-mini` is a
  good low-cost development choice.
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

Then open http://127.0.0.1:5050. “Start a new chat” resets only that browser's
conversation.

## Behavior

### Reservation status and check-in date

The agent asks for both email and reservation number. The tool matches both values,
then returns the reservation status and a USPS check-in date. A failed match does not
reveal which identifier was incorrect.

### Room recommendations

The tool returns every room with positive availability. The model can ask one
clarifying question, then recommends only rooms and details in that result.
Returning the full catalog is deliberate while the supplied catalog is small.

### Early Risers Promotion

The agent creates a code only after an explicit request and email collection.
Python—not the model—checks whether the current time is from 8:00 AM inclusive
to 10:00 AM exclusive in `America/Los_Angeles`.

The code is an HMAC of the normalized email and Pacific date. It is stable for
one user on one date without storing the email or issued code.

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

The model handles language, intent, missing-information questions, tool
selection, and response tone. Python handles data access, validation, time,
check-in dates, and discount codes.

Tool definitions use strict JSON schemas. Python validates again because model
output must still be treated as external input. The loop supports multiple tool
calls in one response and stops after five tool rounds.

The terminal has one in-memory conversation per process. The web app assigns
each browser an isolated in-memory conversation using a signed session cookie;
restarting the app clears those sessions. OpenAI handles API data according to
the account's configured data controls.

Each turn also produces a redacted developer trace: tool names and outcomes,
tool rounds, duration, and an error class when relevant. It never includes
customer messages, emails, reservation numbers, tool arguments/results, or model
reasoning. The local web UI shows the latest trace in a collapsed “Developer
trace” panel.

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
    - `index.html` — camp-themed web chat interface.
- `tests/`
  - `test_agent.py` — agent-loop tests using a fake OpenAI client.
  - `test_config.py` — configuration validation test.
  - `test_tools.py` — deterministic business-rule tests.
  - `test_web.py` — browser-session isolation test.

## Test

Tests do not call OpenAI or spend API credits:

```bash
pytest
```

They cover:

- Combined reservation matching and privacy-preserving failures.
- USPS check-in date construction.
- Out-of-stock room filtering.
- Promotion boundaries, normalization, and stable code generation.
- Startup validation for malformed catalog data and invalid configuration.
- Plain responses, single and multiple tool calls, failed-turn recovery, and
  the tool-round limit.
- Browser-session isolation and redacted execution traces.

Before delivery, manually check normal chat, missing and incorrect reservation
details, broad and specific recommendations, API failure behavior, and CLI
exit. Promotion success and failure paths can be demonstrated at any hour by
the injected-clock unit tests.

## Decision log

### Model for conversation; code for truth
The model interprets requests, asks for missing details, and writes friendly
responses. Python tools own reservations, availability, promotion timing, and promo-code
generation. This makes customer-facing facts testable rather than prompt-only.

### Strict tool contracts and server-side validation
Tool schemas guide the model, but Python validates inputs and data again.
Model output is external input, so a strict schema is helpful but not a
security boundary on its own.

### Privacy-preserving reservation lookup
An reservation is found only when email and reservation number both match. All misses return
the same result, so the agent cannot reveal which identifier was incorrect.

### Explicit capability boundaries
The agent must not invent links, cart actions, policies, or other unsupported
facts. New customer-facing capabilities require a tool or trusted data source
before the model can offer them.

### Per-browser conversation state
The terminal has one conversation per process. The web app gives each browser
an opaque signed session ID and a separate agent, preventing one visitor's
context from appearing in another visitor's chat.

### No automatic retries after incomplete turns
The agent keeps the last completed conversation checkpoint if a request fails,
instead of silently repeating work. Customers retry intentionally. Future tools
that change data, such as refunds, must also use idempotency keys to prevent
duplicate actions.

### Redacted observability
Each turn records tool names/outcomes, tool rounds, duration, and error class.
The trace deliberately excludes messages, emails, reservation numbers, raw tool
data, API keys, and model reasoning, so failures can be diagnosed without
logging private customer data.

### Minimal, risk-based testing
Tests use fake OpenAI responses and an injectable clock, so they are fast,
deterministic, and cost nothing. They protect high-risk contracts such as
privacy-safe lookup failures, promo boundaries, malformed data, session
isolation, recovery after failed turns, and trace redaction.

### Known limits
- Web sessions and traces are in-memory only. Production needs bounded shared
  storage, authentication, a configured cookie secret, and rate limits.
- The promotion code can be generated but not redeemed; production requires an
  issuance and redemption service.
- Full-catalog recommendations suit the small sample data. A larger catalog
  needs deterministic search or retrieval.
- Conversation input is sent to OpenAI to generate a response. Its handling is
  governed by the account's configured data controls.
