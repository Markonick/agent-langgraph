"""An offline, scriptable stand-in for the real LLM adapter.

Program it with the exact text the model should return, or an exception to raise, so every
branch of the agent is reachable with no key and no network — the whole point of depending
on the ``LLMClient`` port rather than a concrete SDK.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator

from .ports import LLMRequest, LLMResponse


class FakeLLM:
    """A deterministic ``LLMClient``.

    ``script`` is either:
      * a single ``str`` (or ``Exception``) reused for every call, or
      * an iterable of ``str`` / ``Exception`` consumed one per call (to simulate a repair
        round-trip, or a transient error followed by a success).

    A ``str`` item is returned as the model's raw text; an ``Exception`` item is raised (to
    simulate ``RateLimited`` / ``TransientError`` / ``ContentError``). Every request is
    recorded on ``.calls`` so tests can assert how many times — and with what — the model
    was called.
    """

    def __init__(self, script: str | Exception | Iterable[str | Exception]) -> None:
        if isinstance(script, (str, Exception)):
            self._single: str | Exception | None = script
            self._queue: Iterator[str | Exception] | None = None
        else:
            self._single = None
            self._queue = iter(script)
        self.calls: list[LLMRequest] = []

    async def generate(self, req: LLMRequest) -> LLMResponse:
        self.calls.append(req)
        item = self._single if self._single is not None else next(self._queue)  # type: ignore[arg-type]
        if isinstance(item, Exception):
            raise item
        return LLMResponse(text=item)
