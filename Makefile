.PHONY: setup test lint compare demo-crash demo-review graph

setup:  ## install deps (langgraph + pytest/ruff/mypy) — pulls from PyPI once, then offline
	uv sync

# Deterministic, offline, key-free — the model is a FakeLLM, no DB, no network.
test:
	uv run pytest -q

lint:
	uv run ruff check core tests && uv run ruff format --check core tests && uv run mypy core

# Once you've built core/langgraph_agent.py + compare.py: run the same scenarios through
# BOTH agents and print a side-by-side verdict diff.
compare:
	uv run python -m compare

# These three are yours to build (see README): a real kill + resume, the human-review pause,
# and a printout of the graph's nodes and edges.
demo-crash:
	uv run python -m demo_crash

demo-review:
	uv run python -m demo_review

graph:
	uv run python -m graph_topology
