"""The LLM port + a normalised error taxonomy — lifted from the real agent's llm_client.

The agent core depends ONLY on the ``LLMClient`` protocol, never on a concrete SDK, so the
same core runs against a real provider adapter in production and against a ``FakeLLM`` in
tests — offline, keyless, deterministic.

The error taxonomy is the important part: callers branch on *why* a call failed, not on a
provider's wire error. ``ContentError`` is terminal (another attempt won't help);
``RateLimited`` / ``TransientError`` are retryable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class LLMRequest:
    system: str
    user: str
    max_tokens: int = 1024
    temperature: float = 0.0


@dataclass(frozen=True)
class LLMResponse:
    text: str


class LLMError(Exception):
    """Base for all adapter errors."""


class ContentError(LLMError):
    """Permanent, terminal failure (unreadable input, context-window, auth). Never
    succeeds on retry — a different attempt or model won't fix it."""


class RateLimited(LLMError):
    """The provider throttled us — back off and retry."""


class TransientError(LLMError):
    """Transient network/server error — retry, or let the queue redeliver."""


@runtime_checkable
class LLMClient(Protocol):
    async def generate(self, req: LLMRequest) -> LLMResponse: ...
