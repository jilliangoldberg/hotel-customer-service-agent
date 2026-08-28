# Trailhead Hotel Agent

A small Python chat agent for reservation check_in_date, room recommendations, and the
Early Risers Promotion. It uses OpenAI tool calling without an agent framework.

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
- `OPENAI_MODEL`: defaults to `gpt-4o-mini`; `gpt-4o` is also supported.
- `PROMOTION_SECRET`: a private random value with at least 16 characters.

Generate a promotion secret with:

```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

Start the chat:

```bash
python -m sierra_agent
```

Type `exit`, `quit`, or `Ctrl-C` to end the session.

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
  -> terminal loop
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

The app keeps only the last OpenAI response ID during the process lifetime.
Exiting clears local session state. OpenAI handles API data according to the
account's configured data controls.

## File map

- `.env.example` — safe configuration template.
- `.gitignore` — excludes secrets, environments, and generated files.
- `pyproject.toml` — package metadata, dependencies, test settings, and CLI.
- `data/guest_reservations.json` — static reservation appendix.
- `data/room_catalog.json` — static room appendix.
- `src/sierra_agent/__init__.py` — package marker.
- `src/sierra_agent/__main__.py` — terminal input, output, and safe failures.
- `src/sierra_agent/config.py` — environment loading and validation.
- `src/sierra_agent/prompt.py` — brand voice and conversation rules.
- `src/sierra_agent/tools.py` — schemas, data loading, tool logic, and dispatch.
- `src/sierra_agent/agent.py` — Responses API and tool-calling loop.
- `tests/test_tools.py` — deterministic business-rule tests.
- `tests/test_agent.py` — agent-loop tests using a fake OpenAI client.

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
- Plain responses, single and multiple tool calls, session continuity, and the
  tool-round limit.

Before delivery, manually check normal chat, missing and incorrect reservation
details, broad and specific recommendations, API failure behavior, and CLI
exit. Promotion success and failure paths can be demonstrated at any hour by
the injected-clock unit tests.

## Design decisions and limits

- `gpt-4o-mini` keeps early testing fast and inexpensive. The model is an
  environment setting so no code change is needed to move to `gpt-4o`.
- The prompt is separate from orchestration so conversation design can evolve
  without changing the agent loop.
- Tool logic is deterministic and receives an injectable clock for reliable
  boundary tests.
- Customer emails, reservation details, API keys, and secrets are never logged.
- Codes are generated but cannot be redeemed because no commerce backend is
  provided. A production promotion service would issue, audit, and redeem them.
- A production catalog should replace the full-catalog tool with search or
  retrieval as it grows.
- The data files currently contain only the complete sample records supplied in
  the brief. Replace them unchanged when the full appendices are available.
