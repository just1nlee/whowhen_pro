"""The all-at-once failure-attribution prompt.

One single-user-turn prompt (no system message — keeps every backend on
the same footing). Adapted from the Who&When prompts in arXiv:2505.01001
with two changes:

1. Add error-mode prediction, so the judge answers *what* went wrong as
   well as *who* and *when*.
2. Use the framework-specific ``step_format_hint`` carried on
   ``RenderResult`` so each framework's native step coordinate is described
   to the judge (flat ``step N`` for smolagents, hierarchical ``step R.P``
   for round-grouped systems).

The builder returns a **list of OpenAI content parts** — interleaved
``{type: "text", ...}`` and ``{type: "image_url", ...}`` entries — so a
multimodal trace renders end-to-end without a separate assembly pass.
Wrap the list with :func:`user_msg` to call a model. For text-only
inspection, :func:`parts_to_text` flattens the list back into a single
string with ``[image]`` placeholders.

Taxonomy
--------
The taxonomy block is built from ``taxonomy.yaml`` **at the root of the
dataset checkout**, read at runtime via :func:`load_taxonomy`. The dataset
and the prompt therefore share one namespace by construction: every code
the prompt enumerates is a code the data can carry, and vice versa. There
is no second, "internal" numbering and no allowlist filtering.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Union

import yaml

from .render.base import TASK_ANCHOR, RenderResult, text_part


TAXONOMY_FILENAME = "taxonomy.yaml"


# Words kept lowercase inside title-cased mode names (the leading word of a
# name is always capitalised regardless).
_TITLE_CASE_LOWER = {"or", "and", "of", "in", "to", "for", "from", "by", "vs",
                     "with", "the", "on", "at"}


def _title_case_mode_name(name: str) -> str:
    """Title-case a taxonomy mode name while preserving punctuation breaks
    (slashes, hyphens). Short joining words stay lowercase unless they're
    the very first token."""

    def cap(word: str, *, first: bool) -> str:
        if not word:
            return word
        if not first and word.lower() in _TITLE_CASE_LOWER:
            return word.lower()
        return word[:1].upper() + word[1:].lower()

    out: list[str] = []
    seen_word = False
    for chunk in re.split(r"(\s+)", name):
        if chunk.isspace() or not chunk:
            out.append(chunk)
            continue
        sub_parts = re.split(r"([/\-])", chunk)
        out.append("".join(
            sp if sp in {"/", "-"} else cap(sp, first=not seen_word and i == 0)
            for i, sp in enumerate(sub_parts)
        ))
        seen_word = True
    return "".join(out)


@dataclass(frozen=True)
class Taxonomy:
    """The error-mode vocabulary shared by the prompt, the parser and the data.

    ``codes`` is the ordered list of mode codes (``"P.1"``, ``"PL.2"``, …)
    exactly as they appear in ``ground_truth.mode`` and in trace filenames.
    ``block`` is the markdown bullet list spliced into the prompt.
    """

    codes: tuple[str, ...]
    block: str

    def __contains__(self, code: object) -> bool:
        return code in self.codes


def load_taxonomy(data_root: Union[str, Path]) -> Taxonomy:
    """Read ``<data_root>/taxonomy.yaml`` into a :class:`Taxonomy`.

    The YAML maps each mode code to ``{name, description}``. Codes are
    emitted in file order; names are title-cased and descriptions are
    collapsed onto one line so each mode is a single markdown bullet:

        ``- P.1: **Visual Misidentification** - Wrong object, entity, ...``
    """
    path = Path(data_root) / TAXONOMY_FILENAME
    if not path.is_file():
        raise FileNotFoundError(
            f"no {TAXONOMY_FILENAME} at {path} — --data-root must point at the "
            "root of the dataset checkout (the directory holding data/ and "
            f"{TAXONOMY_FILENAME})"
        )
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    codes: list[str] = []
    lines: list[str] = []
    for code, body in raw.items():
        if not isinstance(body, dict):
            continue
        name = _title_case_mode_name((body.get("name") or "").strip())
        desc = re.sub(r"\s+", " ", (body.get("description") or "").strip())
        codes.append(str(code))
        lines.append(f"- {code}: **{name}** - {desc}")

    if not codes:
        raise ValueError(f"{path} declares no error modes")

    return Taxonomy(codes=tuple(codes), block="\n".join(lines))


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------

_OUTPUT_FORMAT_ALL_AT_ONCE = (
    "Please answer in the following format, exactly:\n"
    "Agent Name: (the agent ID whose turn first introduces the error)\n"
    "Step Number: (the step coordinate, exactly as used in the conversation above)\n"
    "Error Mode: (one of the error modes listed)\n"
    "Reason: (one or two sentences explaining the error)"
)


# ---------------------------------------------------------------------------
# Content-part assembly helpers
# ---------------------------------------------------------------------------


def _coalesce(parts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge adjacent text parts into one. Reduces the number of content
    entries the SDK has to ship and keeps the wire payload tidy.

    Adjacent text parts can each carry their own leading/trailing newlines
    (text part A ends in ``\\n\\n`` to give a following image margin; text
    part B starts with ``\\n\\n`` to give a preceding image margin). When
    no image sits between them, those margins compound into 3-4 blank
    lines on merge. Collapse any run of 3+ newlines back to ``\\n\\n``
    only at the seam, so single text parts keep their internal layout
    intact (e.g. transcript turns separated by single newlines)."""
    out: list[dict[str, Any]] = []
    for p in parts:
        if (
            p.get("type") == "text"
            and out
            and out[-1].get("type") == "text"
        ):
            prev = out[-1]["text"]
            curr = p["text"]
            # Find the seam: trailing newlines of prev + leading newlines of curr.
            seam_lead = len(prev) - len(prev.rstrip("\n"))
            seam_tail = len(curr) - len(curr.lstrip("\n"))
            if seam_lead + seam_tail >= 3:
                merged = (
                    prev.rstrip("\n")
                    + "\n\n"
                    + curr.lstrip("\n")
                )
            else:
                merged = prev + curr
            out[-1] = text_part(merged)
        else:
            out.append(p)
    return out


def _transcript_parts(rr: RenderResult) -> list[dict[str, Any]]:
    """Walk ``rr.blocks`` (skipping the task block — that's spliced
    earlier) and emit interleaved text/image/body parts.

    Per-block emit order is ``text → images → body_text`` (see
    ``TranscriptBlock`` docstring for the rationale). ``body_text`` is
    empty for almost every renderer and the result matches the legacy
    text-after-images shape; pixelcraft uses it to wedge inline cropped
    images between the step header and the agent's reasoning.

    Blocks are joined with a single ``\\n`` so the flattened transcript
    reads like ``rr.chat_content``.
    """
    parts: list[dict[str, Any]] = []
    first = True
    for block in rr.blocks:
        if block.coord == TASK_ANCHOR:
            continue
        if block.text:
            sep = "" if first else "\n"
            parts.append(text_part(sep + block.text))
            first = False
        if block.images:
            parts.extend(block.images)
        if block.body_text:
            sep = "" if first else "\n"
            parts.append(text_part(sep + block.body_text))
            first = False
    return parts


def parts_to_text(parts: list[dict[str, Any]]) -> str:
    """Flatten a content-parts list to a single string for eyeball /
    text-only-backend use. Image parts become a ``[image #K]`` placeholder
    so the layout / boundaries are still legible."""
    out: list[str] = []
    img_idx = 0
    for p in parts:
        if p.get("type") == "text":
            out.append(p.get("text", ""))
        elif p.get("type") == "image_url":
            url = (p.get("image_url") or {}).get("url", "")
            if url.startswith("data:"):
                head = url.split(",", 1)[0]
                out.append(f"[image #{img_idx} {head}]")
            else:
                out.append(f"[image #{img_idx} {url[:80]}]")
            img_idx += 1
    return "".join(out)


def user_msg(parts: list[dict[str, Any]]) -> dict[str, Any]:
    """Wrap a content-parts list in an OpenAI user message envelope.

    Convenience for the runner: ``generate(model, [user_msg(parts)])``.
    """
    return {"role": "user", "content": parts}


# ---------------------------------------------------------------------------
# Protocol prompt
# ---------------------------------------------------------------------------


def all_at_once(
    rr: RenderResult,
    *,
    problem: str,
    taxonomy: Taxonomy,
) -> list[dict[str, Any]]:
    """Build the all-at-once attribution prompt as interleaved content parts.

    Layout::

        # Task ...
        ## Error Mode Taxonomy ...
        ## User Question  <problem>
        <task images>                                  ← from rr.blocks[TASK_ANCHOR]
        ## Transcript
        <step_1 obs images>  <step_1 text>             ← from rr.blocks
        <step_2 obs images>  <step_2 text>
        ...
        ## Step Coordinate Format ...   ## Response Format ...

    Question + task images sit immediately before the transcript so the
    diagnostic case (question → trace) reads as one contiguous narrative,
    with the taxonomy upstream as reference material.
    """
    # Frameworks whose step coordinate is self-explanatory (flat 1-indexed
    # sequence, single agent, etc.) signal that by returning an empty
    # ``step_format_hint``; the section is then omitted entirely so the
    # prompt doesn't carry redundant boilerplate.
    step_format_section = (
        f"## Step Coordinate Format\n\n{rr.step_format_hint}\n\n"
        if rr.step_format_hint
        else ""
    )

    # 1. Header + taxonomy. Establishes the diagnostic frame and the
    #    vocabulary the model will use, before showing any case data.
    parts: list[dict[str, Any]] = [text_part(
        "# Task\n\n"
        "You are an expert at diagnosing failures in agentic systems.\n\n"
        "You will be given the transcript of an agentic system attempting "
        "to answer a user question. The system failed because of a decisive "
        "error somewhere in the transcript. Your job is to identify the "
        "first decisive error: the step that most directly causes the "
        "system to go wrong and eventually produce an incorrect answer.\n\n"
        "Report which agent made that decisive error, the exact step "
        "coordinate where it occurred, and the best matching error mode "
        "from the taxonomy below. Then briefly explain your reasoning.\n\n"
        "## Error Mode Taxonomy\n\n"
        f"{taxonomy.block}\n\n"
        "## User Question\n\n"
        f"{problem}\n\n"
    )]

    # 2. Task images splice in right after the question so the model sees
    #    them as part of the case, not the transcript.
    task_block = next(
        (b for b in rr.blocks if b.coord == TASK_ANCHOR),
        None,
    )
    if task_block and task_block.images:
        parts.extend(task_block.images)

    # 3. Transcript header. Leading newlines give the task image (if any)
    #    breathing room in flattened text-only previews.
    parts.append(text_part("\n\n## Transcript\n\n"))

    # 4. Per-step transcript (interleaved images + text).
    parts.extend(_transcript_parts(rr))

    # 5. Footer.
    parts.append(text_part(
        f"\n\n{step_format_section}"
        "## Response Format\n\n"
        f"{_OUTPUT_FORMAT_ALL_AT_ONCE}\n"
    ))

    return _coalesce(parts)
