"""Parse all-at-once attribution responses into structured fields.

The all-at-once protocol asks the model for exactly four labelled fields::

    Agent Name: ...
    Step Number: ...
    Error Mode: ...
    Reason: ...

Real-world outputs are messier than the spec — models sometimes:

- prepend prose ("Sure, let me analyse..."),
- bold the labels (``**Agent Name:** ...``),
- wrap fields in code fences,
- spread Reason across multiple lines or add trailing prose,
- emit ``Error Mode: A.3 (Premature Termination)`` instead of just ``A.3``,
- forget the colon, the space, or a field entirely.

We extract leniently and validate the error mode against the codes read
from the dataset's own ``taxonomy.yaml`` (see
:func:`whowhen_eval.prompts.load_taxonomy`) — the same codes the prompt
enumerated and the same codes ``ground_truth.mode`` uses. Predictions that
don't yield a recognised mode get ``error_mode=None`` and a note in
``parse_warnings``, so the runner can still write a record and downstream
analysis can decide whether to count it as wrong or unparseable.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Collection, Optional


# Match a taxonomy code anywhere in a string: one or more uppercase
# letters, dot, one or more digits. Bounded by non-alnum / start / end so
# we don't match e.g. ``CR.7`` substrings inside a longer token.
_CODE_RE = re.compile(r"(?<![A-Za-z0-9])([A-Z]+\.\d+)(?![A-Za-z0-9])")

# Field-extraction regex: tolerate optional bold (``**Field:**``) on
# either side of the colon, missing space after the colon, and a value
# that runs until the next field label or end-of-text. We use ``[ \t]*``
# (horizontal whitespace only) around the colon so a missing/empty value
# field doesn't gobble the newline before the next field — that would
# break the lookahead and steal the next field's content.
_FIELD_NAMES = ("Agent Name", "Step Number", "Error Mode", "Reason")
_FIELD_LOOKAHEAD = "|".join(re.escape(n) for n in _FIELD_NAMES)
_FIELD_RE = re.compile(
    rf"(?:\*\*)?(?P<key>{_FIELD_LOOKAHEAD})(?:\*\*)?[ \t]*:[ \t]*"
    rf"(?:\*\*)?[ \t]*(?P<val>.*?)"
    rf"(?=\n[ \t]*(?:\*\*)?(?:{_FIELD_LOOKAHEAD})(?:\*\*)?[ \t]*:|\Z)",
    flags=re.DOTALL | re.IGNORECASE,
)


@dataclass
class ParsedPrediction:
    """Structured form of an all-at-once response.

    All fields are optional; downstream code should treat ``None`` as
    "not extracted from the model output". ``error_mode`` is a code from
    the dataset taxonomy, directly comparable to ``ground_truth.mode``.
    """
    agent_name: Optional[str] = None
    step_coord: Optional[str] = None
    error_mode: Optional[str] = None
    reason: Optional[str] = None
    parse_warnings: list[str] = field(default_factory=list)


def parse_all_at_once(
    text: Optional[str],
    valid_codes: Collection[str],
) -> ParsedPrediction:
    """Lenient parser for the four-field all-at-once response.

    ``valid_codes`` is the taxonomy's code list (``Taxonomy.codes``); a
    mode is only accepted if it is one of them.

    Returns a fully-populated ``ParsedPrediction``; missing or unparseable
    fields stay ``None`` and accumulate a note in ``parse_warnings``.
    Always succeeds — never raises — so a bad model response degrades the
    record but doesn't kill the eval.
    """
    pred = ParsedPrediction()
    if not text:
        pred.parse_warnings.append("empty response")
        return pred

    matches = {m.group("key").title(): m.group("val").strip()
               for m in _FIELD_RE.finditer(text)}
    # ``Agent Name``/``Step Number`` etc. — title-casing normalises any
    # accidental ``**agent name:**`` from the model.

    pred.agent_name = _clean_value(matches.get("Agent Name"))
    pred.step_coord = _clean_value(matches.get("Step Number"))
    pred.reason = _clean_reason(matches.get("Reason"))

    raw_mode_field = matches.get("Error Mode")
    pred.error_mode, mode_warn = _extract_error_mode(
        raw_mode_field, text, valid_codes
    )
    if mode_warn:
        pred.parse_warnings.append(mode_warn)

    for field_name, value in (
        ("Agent Name", pred.agent_name),
        ("Step Number", pred.step_coord),
        ("Reason", pred.reason),
    ):
        if not value:
            pred.parse_warnings.append(f"missing field: {field_name}")

    return pred


def _clean_value(raw: Optional[str]) -> Optional[str]:
    """Strip whitespace, surrounding quotes/backticks/bold markers, and
    parenthetical placeholders the model may have copied verbatim from
    the format spec.
    """
    if raw is None:
        return None
    val = raw.strip()
    # Drop trailing bold-close markers and surrounding quote pairs that
    # may have leaked past the field-extraction regex.
    while val.endswith("**"):
        val = val[:-2].rstrip()
    while val.startswith("**"):
        val = val[2:].lstrip()
    for q in ("`", '"', "'"):
        if len(val) >= 2 and val.startswith(q) and val.endswith(q):
            val = val[1:-1].strip()
    # Spec placeholder leaks like "(the agent ID whose turn ...)" mean the
    # model didn't actually fill the field.
    if val.startswith("(") and val.endswith(")") and len(val) > 4:
        return None
    return val or None


def _clean_reason(raw: Optional[str]) -> Optional[str]:
    """Reason can span multiple lines; collapse whitespace and trim."""
    if raw is None:
        return None
    val = re.sub(r"\s+", " ", raw).strip()
    return val or None


def _extract_error_mode(
    raw: Optional[str],
    full_text: str,
    valid_codes: Collection[str],
) -> tuple[Optional[str], Optional[str]]:
    """Return ``(mode_code, warning_or_None)``.

    Strategy: pull the first taxonomy-shaped code (``[A-Z]+\\.\\d+``) from
    the ``Error Mode:`` value. If absent (or unrecognised), fall back to
    scanning the whole response for the first such code — sometimes the
    model puts it in the Reason line ("This is an R.4 because...").
    """
    known = set(valid_codes)
    candidates: list[str] = []
    if raw:
        candidates.extend(_CODE_RE.findall(raw))
    # Fallback: any taxonomy-shaped code anywhere in the response.
    if not candidates:
        candidates.extend(_CODE_RE.findall(full_text or ""))

    for code in candidates:
        if code in known:
            return code, None

    if raw:
        return None, f"unrecognised error mode: {raw[:80]!r}"
    return None, "missing field: Error Mode"
