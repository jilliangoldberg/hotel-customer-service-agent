# Trailhead Hotel Agent

Welcome to your next adventure! 🏨

The Trailhead Hotel Agent is your trail guide for:

- Tracking down a reservation
- Finding available rooms for your next outing
- Claiming an Early Risers Promotion code

Try it out in your terminal or local web browser. Happy camping! 🛎️

## Setup

Requires Python 3.11 or newer.

### 1. Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
```

### 2. Configure

Update the copied `.env` file:

```dotenv
OPENAI_API_KEY=your-api-key
OPENAI_MODEL=gpt-4.1-mini
OPENAI_EVAL_MODEL=
PROMOTION_SECRET=paste-generated-secret-here
```

`OPENAI_EVAL_MODEL` is optional and defaults to `OPENAI_MODEL`. Use any models your OpenAI project can access.

Generate a promotion secret with:

```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

Paste the generated value into `PROMOTION_SECRET`.

Never commit your `.env` file.

### 3. Run

Terminal chat:

```bash
python -m sierra_agent
```

Type `exit`, `quit`, or `Ctrl-C` to stop.

Local web chat:

```bash
python -m sierra_agent.web
```

Then open [http://127.0.0.1:5050](http://127.0.0.1:5050).

The web app has two views:

- **Customer** — Chat with the support agent
- **Developer** — Run eval scenarios and evaluate results from agent testing

## Current Features

- 🗓️ **Reservation status:** Looks up a reservation using an email and reservation number. Returns the status and a USPS check-in date when available. Failed lookups do not reveal which identifier was incorrect.
- 🛏️ **Room recommendations:** Suggests only available catalog items. The agent may ask one clarifying question for broad requests.
- 🌅 **Early Risers Promotion:** Helps customers claim one daily code per email after an explicit request and a verified 8:00 AM to 10:00 AM Pacific-time window.

## How It Works

When a customer sends a message:

1. The terminal or web app passes it to the model.
2. The model decides whether it needs information from a reservation, room, or promotion tool.
3. If so, Python validates the request, runs the tool, and returns the result to the model.
4. The model writes a customer-friendly response.
5. A final policy check catches known unsupported promises or actions before the response is shown.

Customer messages are sent to OpenAI to generate responses and are handled according to the account's configured data controls.

## Project Guide

The main code lives in `src/sierra_agent/`:

- `__main__.py` — terminal chat entry point
- `agent.py` — OpenAI tool-calling loop
- `config.py` — environment configuration and validation
- `tools.py` — reservation, catalog, and promotion tools
- `prompt.py` — agent instructions and tone
- `policy.py` — customer-response safety checks
- `factory.py` — builds the shared agent used by every interface
- `web.py` — local Flask app
- `eval/` — simulated conversation evals and reports
  - `scenarios.py` — customer scenarios and success criteria
  - `harness.py` — runs the simulated conversations and judging
  - `reports.py` — creates Markdown and JSON results
  - `__main__.py` — command-line interface for running evals

Supporting files:

- `data/` — sample reservations and room catalog
- `tests/` — deterministic unit and integration tests
- `eval-results/` — generated eval reports

## Testing

Run the test suite:

```bash
pytest
```

Tests use fake OpenAI responses, so they do not call the API or spend credits.

## Simulated Evals

Simulated evals create a full agent-to-agent conversation. One LLM role plays a customer with a specific goal, facts, and speaking style; the real support agent responds; then a separate judge request scores the completed interaction. This tests realistic, messy conversations that are difficult to cover comprehensively with deterministic or manual tests.

The customer and judge use `OPENAI_EVAL_MODEL`, which defaults to `OPENAI_MODEL`. Choosing a different eval model from the support agent can reduce correlated self-evaluation bias. Running scenarios calls the OpenAI API and may spend credits; `--list` does not.

List available scenarios:

```bash
python -m sierra_agent.eval --list
```

Run all scenarios:

```bash
python -m sierra_agent.eval
```

Run one scenario:

```bash
python -m sierra_agent.eval --scenario derail-reservation
```

Results are saved in `eval-results/` and can also be viewed from the web app's Developer view.

## Adding a Capability

1. Add and validate the tool handler in `tools.py`.
2. Register its model-facing schema.
3. Add customer-facing instructions in `prompt.py`.
4. Update `policy.py` if the capability changes a safety boundary.
5. Add tests and, when useful, an eval scenario.

State-changing tools such as refunds need authentication, confirmation, authorization, and idempotency before they are safe to add.

## Technical Decisions

**Simple, shared architecture**

A single OpenAI Responses API loop handles every request. The capabilities are distinct and small enough that a workflow graph or agent framework would add unnecessary complexity. The terminal, web app, and eval harness all build this same agent through `factory.py`, reducing drift between development, testing, and the customer experience.

**Deterministic code for customer-facing facts**

The model handles intent, missing-information questions, tool selection, and tone. Anything that must be exact—reservations, availability, time checks, check-in dates, and promotion codes—lives in deterministic Python so it remains predictable and testable. Each tool keeps its schema beside its handler, and Python validates generated arguments as untrusted input.

**Safety and privacy in code**

The prompt defines capability boundaries, while narrow Python rules catch unsupported promises and request a rewrite or return a safe fallback. Failed reservation lookups stay generic, browser sessions are isolated, and developer traces omit customer messages, identifiers, tool data, and model reasoning.

**LLM customer simulation and judging**

Behavioral evals use separate customer and judge requests, while code-based gates enforce selected hard requirements. This combines realistic multi-turn testing with repeatable checks.

### Other Technical Decisions

- Conversations stay in memory to keep local setup simple. The terminal has one conversation per process, while the web app separates browsers with signed session cookies.
- Failed turns return to the last completed conversation checkpoint instead of retrying automatically.
- Local JSON data and full-catalog recommendations keep the sample app self-contained; larger production data would need databases and retrieval.

## 🚀 Future Improvements

**Production integrations**

Replace sample JSON with database and commerce APIs for live reservations, availability, promotion redemption, returns, and purchases. State-changing actions would require authentication, confirmation, authorization, and idempotency.

**Personalized recommendations**

Use purchase history, preferences, sizing, and past conversations—with clear privacy controls—to make suggestions more relevant.

**Visual room search**

Let customers upload an image to find similar rooms or describe the style and features they want.

**Agent infrastructure**

As the tool set grows, evaluate the OpenAI Agents SDK for built-in tracing, guardrails, handoffs, and tool orchestration.

**Stronger evaluation**

Add more scenarios, repeated trials, pass-rate check_in_date, regression thresholds, and cost and latency measurements.
