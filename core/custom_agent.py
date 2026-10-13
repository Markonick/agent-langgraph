"""The hand-rolled document-validator agent — the "before" a LangGraph version must match.

A plain async function with explicit branches:

  1. readability gate      — a non-PDF / locked file returns a TERMINAL verdict, no model call;
  2. one model call        — through the ``LLMClient`` port;
  3. parse                 — ``parse_verdict`` turns the reply into a ``Verdict``;
  4. one repair retry      — a non-JSON reply is nudged once; still bad → the ``ValueError``
                             propagates (the caller/queue redelivers).

Retryable provider errors (``RateLimited`` / ``TransientError``) are deliberately NOT caught
— they propagate so the queue redelivers. Only ``ContentError`` (terminal) is turned into a
verdict.

Call source in the real app: the SQS handler for ``document.uploaded`` invokes ``validate``,
then persists the ``Verdict`` and flips the requirement status. There is NO intermediate
persistence — if the process dies mid-validation, SQS redelivers and it reruns from the top.
"""

from __future__ import annotations

from dataclasses import dataclass

from .ports import ContentError, LLMClient, LLMRequest
from .validation import (
    Verdict,
    build_expectations,
    parse_verdict,
    unreadable_reason,
    unreadable_verdict,
)


@dataclass(frozen=True)
class Document:
    """One uploaded file plus what we expected it to be."""

    pdf: bytes
    document_type: str
    label: str
    tax_year: int | None = None
    buyer_name: str | None = None


_SYSTEM = (
    "You validate a Greek mortgage document against a list of expectations. "
    'Reply with ONLY a JSON object: {"matches": bool, "confidence": "high|medium|low", '
    '"detected_type": string, "detected_tax_year": integer|null, "name_match": bool, '
    '"notes_en": string, "notes_el": string}.'
)
_REPAIR_CLAUSE = " Your previous reply was not valid JSON. Reply with ONLY the JSON object, nothing else."


def build_request(doc: Document, *, repair: bool = False) -> LLMRequest:
    """Assemble the model request from a document's expectations. ``repair`` appends the
    nudge used for the one retry after a non-JSON reply."""
    system = _SYSTEM + (_REPAIR_CLAUSE if repair else "")
    user = "Validate this document against the following expectations:\n" + "\n".join(
        build_expectations(doc.document_type, doc.label, doc.tax_year, doc.buyer_name)
    )
    return LLMRequest(system=system, user=user)


async def validate(doc: Document, *, llm: LLMClient) -> Verdict:
    reason = unreadable_reason(doc.pdf)
    if reason is not None:
        return unreadable_verdict(reason)  # terminal — never calls the model

    try:
        resp = await llm.generate(build_request(doc))
    except ContentError:
        return unreadable_verdict("bad_request")  # terminal — do not retry

    try:
        return parse_verdict(resp.text)
    except ValueError:
        resp = await llm.generate(build_request(doc, repair=True))  # one repair retry
        return parse_verdict(resp.text)  # still non-JSON → ValueError propagates (retryable)
