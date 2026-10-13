"""Pure validation pieces — no model, no transport, no I/O.

Shared by BOTH the hand-rolled agent and the LangGraph agent, so a ``Verdict`` means the
same thing whoever produced it. This is the behaviour surface a re-implementation must
preserve: the readability gate, the expectation lines, the safe coercion of model-provided
control-flow fields, the bilingual note, and — critically — that a non-JSON reply RAISES
rather than silently passing.

Trimmed from the production validator (``agents/validation.py``): foreign-document prompts,
apostille, periodic-obligation periods and the gov.gr-signature signal are intentionally out
of scope here. What remains is faithful.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

_CONFIDENCE = {"high", "medium", "low"}


@dataclass(frozen=True)
class Verdict:
    """The outcome of validating one PDF against its expectations.

    A normal model verdict has ``terminal_reason is None`` and a bool-or-None ``matches``.
    A terminal (permanent) content failure has ``matches is None`` and a ``terminal_reason``
    naming the cause — the caller records the note and does not retry.
    """

    matches: bool | None
    confidence: str | None
    detected_type: str | None
    detected_tax_year: int | None
    name_match: bool | None
    notes_en: str
    notes_el: str
    terminal_reason: str | None = None


# Terminal, permanent content failures: the file can never be read by the model, so these
# are actionable verdicts rather than errors to retry.
_UNREADABLE_NOTES: dict[str, tuple[str, str]] = {
    "not_pdf": (
        "Couldn't read this file — it does not appear to be a valid PDF. Please upload a PDF.",
        "Δεν ήταν δυνατή η ανάγνωση του αρχείου — δεν φαίνεται έγκυρο PDF. Ανεβάστε ένα PDF.",
    ),
    "password_protected": (
        "Couldn't read this PDF — it appears to be password-protected. Upload an unlocked copy.",
        "Δεν ήταν δυνατή η ανάγνωση — φαίνεται κλειδωμένο με κωδικό. Ανεβάστε αντίγραφο χωρίς κλείδωμα.",
    ),
    "bad_request": (
        "Couldn't read this PDF automatically. Please check the file and re-upload.",
        "Δεν ήταν δυνατή η αυτόματη ανάγνωση. Ελέγξτε το αρχείο και ανεβάστε το ξανά.",
    ),
}


def unreadable_reason(data: bytes) -> str | None:
    """A real readability gate. Valid PDFs start with ``%PDF-``; a ``LOCKED`` sentinel
    stands in for the production password-protected check. ``None`` means readable."""
    if data[:6] == b"LOCKED":
        return "password_protected"
    if data[:5] != b"%PDF-":
        return "not_pdf"
    return None


def unreadable_verdict(reason: str) -> Verdict:
    """A terminal verdict for a PDF the model can never read (bad bytes, locked, rejected)."""
    note_en, note_el = _UNREADABLE_NOTES[reason]
    return Verdict(
        matches=None,
        confidence=None,
        detected_type=None,
        detected_tax_year=None,
        name_match=None,
        notes_en=note_en,
        notes_el=note_el,
        terminal_reason=reason,
    )


def build_expectations(
    document_type: str,
    label: str,
    tax_year: int | None,
    buyer_name: str | None,
) -> list[str]:
    """The human-readable expectation lines the model checks the PDF against. Type is
    always present; year and owner are added only when supplied."""
    lines = [
        f"- Document type: it should be a **{label}** (key `{document_type}`). "
        "Confirm the content/context actually matches this type.",
    ]
    if tax_year is not None:
        lines.append(
            f"- Tax year: it should be for tax year **{tax_year}** — the fiscal year the "
            "document reports on, NOT when it was printed or downloaded."
        )
    if buyer_name:
        lines.append(
            f"- Owner: it should belong to **{buyer_name}**. Check this name appears as the "
            "taxpayer/holder."
        )
    return lines


def _coerce_bool(value: object) -> bool | None:
    """Constrain a model-provided bool to its domain: anything that is not a real bool
    becomes the safe ``None`` (unknown), never a truthy/falsy pass-through."""
    return value if isinstance(value, bool) else None


def _clean_confidence(value: object) -> str | None:
    """Constrain ``confidence`` to {high, medium, low}; an unexpected token → ``None``."""
    if isinstance(value, str) and value.lower() in _CONFIDENCE:
        return value.lower()
    return None


def parse_verdict(raw: str) -> Verdict:
    """Parse the model's raw text into a typed ``Verdict``.

    Control-flow fields (``matches`` / ``confidence`` / ``name_match``) are coerced to their
    known domains so an unexpected token becomes the safe ``None`` rather than passing
    through. The note is enriched (bilingual) with what the model detected.

    Raises ``ValueError`` if the model did not return a JSON object — an unparseable reply is
    a retryable upstream condition, never a silent pass.
    """
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError("model did not return JSON") from exc
    if not isinstance(data, dict):
        raise ValueError("model did not return a JSON object")

    base_en = str(data.get("notes_en") or data.get("notes") or "").strip()
    base_el = str(data.get("notes_el") or data.get("notes") or "").strip()
    detected = data.get("detected_type")
    detected_year = data.get("detected_tax_year")
    name_match = _coerce_bool(data.get("name_match"))
    matches = _coerce_bool(data.get("matches"))
    confidence = _clean_confidence(data.get("confidence"))

    bits_en: list[str] = []
    bits_el: list[str] = []
    if detected:
        suffix = f" {detected_year}" if detected_year else ""
        bits_en.append(f"detected: {detected}{suffix}")
        bits_el.append(f"ανιχνεύθηκε: {detected}{suffix}")
    if name_match is not None:
        bits_en.append(f"name match: {'yes' if name_match else 'no'}")
        bits_el.append(f"αντιστοίχιση ονόματος: {'ναι' if name_match else 'όχι'}")

    notes_en = f"{base_en} ({'; '.join(bits_en)})".strip() if bits_en else base_en
    notes_el = f"{base_el} ({'; '.join(bits_el)})".strip() if bits_el else base_el

    return Verdict(
        matches=matches,
        confidence=confidence,
        detected_type=detected,
        detected_tax_year=detected_year,
        name_match=name_match,
        notes_en=notes_en,
        notes_el=notes_el,
    )
