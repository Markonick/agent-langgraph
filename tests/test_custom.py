"""The 3 "before" tests pinning the hand-rolled agent's behaviour.

These are the contract a LangGraph re-implementation must reproduce. (They do NOT cover the
whole surface — see README; preserving the untested behaviour too is part of the task.)
"""

import json

import pytest

from core.custom_agent import Document, validate
from core.fakes import FakeLLM
from core.ports import RateLimited

GOOD_JSON = json.dumps(
    {
        "matches": True,
        "confidence": "high",
        "detected_type": "e1_form",
        "detected_tax_year": 2024,
        "name_match": True,
        "notes_en": "Looks correct.",
        "notes_el": "Φαίνεται σωστό.",
    }
)

DOC = Document(
    pdf=b"%PDF-1.7 ...bytes...",
    document_type="e1_form",
    label="E1 income tax return",
    tax_year=2024,
    buyer_name="Alex Example",
)


async def test_happy_path_parses_and_enriches():
    llm = FakeLLM(GOOD_JSON)
    verdict = await validate(DOC, llm=llm)

    assert verdict.matches is True
    assert verdict.confidence == "high"
    assert verdict.detected_type == "e1_form"
    assert verdict.terminal_reason is None
    # the note is enriched with what the model detected, in both languages
    assert "detected: e1_form 2024" in verdict.notes_en
    assert "name match: yes" in verdict.notes_en
    assert "αντιστοίχιση ονόματος: ναι" in verdict.notes_el
    assert len(llm.calls) == 1


async def test_unreadable_is_terminal_and_never_calls_the_model():
    llm = FakeLLM(GOOD_JSON)
    verdict = await validate(Document(pdf=b"not-a-pdf", document_type="e1_form", label="E1"), llm=llm)

    assert verdict.terminal_reason == "not_pdf"
    assert verdict.matches is None
    assert llm.calls == []  # short-circuited before any model call


async def test_non_json_triggers_one_repair_retry():
    llm = FakeLLM(["sorry, here is the answer: it matches", GOOD_JSON])
    verdict = await validate(DOC, llm=llm)

    assert verdict.matches is True  # recovered on the repair round-trip
    assert len(llm.calls) == 2
    assert "not valid JSON" in llm.calls[1].system  # the second call carried the repair nudge


async def test_rate_limited_propagates_for_redelivery():
    llm = FakeLLM(RateLimited("429"))
    with pytest.raises(RateLimited):
        await validate(DOC, llm=llm)
