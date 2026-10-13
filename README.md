# agent-langgraph

A small, **offline** slice to answer one question: **can a LangGraph agent drop in for our
hand-rolled one?** You get a hand-rolled document-validator agent behind a clean port. The
task is to build a **LangGraph** version of the *same* agent behind the *same* port, then run
both through identical scenarios (with a fake model) and show they produce **identical
verdicts**. No key, no database, no network — everything runs with `make test`.

## What the agent does

It validates one uploaded PDF against what we expected it to be (document type, tax year,
owner) and returns a `Verdict`: does it match, with what confidence, what the model detected,
and a bilingual (EN/EL) note. The real app runs this when a buyer uploads a document.

The flow has real branches — which is why a graph can earn its place:

1. **readability gate** — a non-PDF / locked file returns a **terminal** verdict, no model call;
2. **call the model** through the `LLMClient` port;
3. **parse** the reply into a `Verdict` (coercing the control-flow fields to safe values);
4. **repair** — a non-JSON reply is nudged once; still bad → it raises (the queue redelivers).

## Where this starts (the "before")

```
core/
├── ports.py         # LLMClient protocol + LLMRequest/LLMResponse + error taxonomy
├── validation.py    # Verdict, build_expectations, parse_verdict, the readability gate
├── custom_agent.py  # the HAND-ROLLED validate(doc, *, llm) -> Verdict  ← the before
└── fakes.py         # FakeLLM: scriptable canned replies / errors (offline model)
tests/test_custom.py # 4 passing tests pinning the behaviour
```

`core/` imports no SDK, no AWS, no DB. The model is a `FakeLLM` you script.

## The task — build the LangGraph version and prove it's a drop-in

1. **`core/langgraph_agent.py`** — the same `validate(doc, *, llm) -> Verdict`, built as a
   LangGraph `StateGraph`: nodes for read → call model → parse → repair → verdict, with
   conditional edges for the terminal and repair branches. Same port in, same `Verdict` out.
2. **Preserve the behaviour, not just the 4 tests.** The full contract is below; the tests
   only sample it. A drop-in that changes untested behaviour is not a drop-in.
3. **`compare.py` + `make compare`** — run a scenario table through *both* agents with the
   same `FakeLLM` and print a side-by-side verdict diff.
4. **`tests/test_equivalence.py`** — the headline: for every scenario, assert
   `custom.validate(...) == langgraph.validate(...)`. Plus: terminal short-circuits before any
   model call on both; `RateLimited` propagates on both; the repair branch fires on
   non-JSON on both.

### The behaviour contract to preserve
- an unreadable file (`not_pdf` / `password_protected`) → terminal verdict, **no model call**;
- `ContentError` from the model → terminal `bad_request` verdict (never retried);
- `RateLimited` / `TransientError` → **propagate** (the queue redelivers), not swallowed;
- a non-JSON reply → **one** repair retry, then raise if still non-JSON;
- `matches` / `name_match` coerced to `bool | None`, `confidence` to `{high,medium,low} | None`
  (an unexpected token becomes `None`, never passes through);
- the note is enriched bilingually with what the model detected.

### Then show what LangGraph actually *buys* (the payoff)
The hand-rolled version can't do these cleanly — demonstrate them on the LangGraph side:
- **Persistence / checkpoints** — attach a checkpointer keyed by a `thread_id`, so a run
  resumes from the last node instead of re-running from the top.
- **Crash survival** — back the checkpointer with a SQLite file
  (`langgraph-checkpoint-sqlite` is already a dependency). Prove it with a **real** kill, not
  a mock: one process dies hard mid-graph (`os._exit`), a **fresh process** opens the same
  file and finishes the run, and the nodes that already completed do **not** run again.
- **Human-in-the-loop (interrupt)** — when confidence is `low`, `interrupt()` for a human
  (a broker) to confirm/override, then resume with `Command(resume=...)`.

Keep these additive — the equivalence above must still hold on the straight-through path.

## Tests you must write (at least 30 cases, parametrize freely)

The 4 starting tests are a sample, not the bar. Cover, for **both** agents unless noted:

| Area | What to pin |
|---|---|
| Equivalence | a scenario table of 10+ inputs (happy path, wrong year, wrong type, name mismatch, low / medium / high confidence, missing fields, extra fields, empty notes, Greek notes, `null` year) → identical `Verdict` from both |
| Coercion | `matches` / `name_match` given `"true"`, `1`, `"yes"`, `null`, a list → `None` unless a real bool; `confidence` given `"HIGH"`, `"certain"`, `5`, `null` |
| Terminal | `not_pdf`, `password_protected`, `ContentError` → terminal verdict, **zero** model calls |
| Propagation | `RateLimited` and `TransientError` raise out, on first call **and** on the repair call |
| Repair | non-JSON → exactly one retry; JSON array / bare string / empty string → same; still bad → raises; the 2nd request carries the nudge |
| Graph (LangGraph only) | the node and edge set matches the flow described above; a failed node retries alone; completed nodes don't re-run |
| Human review | low confidence pauses; `Command(resume=...)` finishes; high confidence never pauses |
| Crash | a real subprocess is hard-killed mid-graph, a fresh process finishes from the SQLite file, completed nodes ran once |

A test that can't fail doesn't count: if you break one rule on purpose, a test should go red.

## How to see it working (build these commands)

```bash
make test          # 30+ cases, all green
make compare       # one row per scenario, both agents, ✓ / ✗ per row, and a total
make demo-crash    # prints: process 1 killed → process 2 resumed → how many times each node ran
make demo-review   # prints: paused on low confidence → human answer → final verdict
make graph         # prints the LangGraph nodes and edges so the topology can be checked
```

## Run (all offline, no key, no DB)

```bash
make setup     # uv sync (pulls langgraph from PyPI once)
make test      # the suite — 4 "before" tests to start; you add the rest
make lint      # ruff + mypy
make compare   # once built: both agents, same scenarios, side-by-side
```

Python 3.11+ via [uv](https://docs.astral.sh/uv/). LangGraph runs fully in-process; only the
model nodes would need a key in production, and here they call the `FakeLLM`.

## Out of scope / notes

This is a slice of a bigger private repo (a Greek mortgage/accounting SaaS; the real agent
runs on SQS + a provider LLM + Postgres). `core/validation.py` is trimmed faithfully from the
real validator — foreign-document prompts, apostille, periodic-obligation periods and the
gov.gr-signature signal are left out to keep it bounded. The UI is out of scope. The only
"personal" field is a buyer name, deliberately a fake (`Alex Example`).
