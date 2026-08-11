"""smolagents renderer (single-agent CodeAgent).

Step coordinate: ``step N`` (cursor-based, sequential across the trajectory).

The cursor is assigned by linear scan over non-framing trajectory entries
(skipping ``kind: user`` and ``kind: final_answer``). It starts at 0 if
the first non-framing entry is ``planning``, else 1 — so traces with no
planning step (e.g. P.1) keep the old action-only numbering, and traces
that have a planning step get coord 0 for the plan, 1 for the first
action, etc. Re-planning entries also consume an index.

Release ``ground_truth.step`` matches this same cursor scheme (the old
ActionStep-only value is preserved as ``ground_truth.step_legacy``).

Trajectory shape:
    [
        {kind: "user", content, images?},
        {kind: "planning", plan?}*,
        {kind: "action", step_number, reasoning, code, observation,
                         observation_images?, error?, is_final_answer},
        ...
    ]

We render one transcript block per *non-framing* entry. Action blocks use
the two fields the agent actually authored: ``reasoning`` (the model's
raw output, which already contains its own Thought/Code structure) and
``observation`` (the tool-execution output), wrapped in explicit
``[output]`` / ``[observation]`` markers. Planning blocks render the
``plan`` field wrapped in ``[plan]`` so the judge can address it
distinctly — this matters for PL.1, where the corruption lives in the
plan rather than in any single action's output.

Image handling
--------------
Two image streams reach the prompt:

1. **Task images** (``release["task"]["images"]``) — typically the
   user-question screenshot. Anchored under ``TASK_ANCHOR`` so the
   runner inlines them next to the question, before the transcript.
2. **Per-step observation images** (``observation_images``) — usually
   the multi-image return value of an ``image_search``-style tool. The
   tool's ``print()`` output ends up rendered as a Python list of
   ``<PIL.Image.Image image mode=RGB size=AxB at 0x...>`` reprs, which
   are meaningless to the judge. We rewrite each repr in-place to a
   ``[observation_image #N size=AxB]`` token (numbering matches the
   ``observation_images`` order), so the textual observation reads
   coherently and aligns 1:1 with the actual image parts the runner
   splices in. Any unmatched images (more parts than placeholders) get
   a sentinel ``[observation_image #N size=AxB]`` prefix at the top of
   the observation block so nothing is dropped.

Both image streams pass through ``pil_image_part`` with JPEG q80 + max-dim
1024 — the original release stores PNG, which bloats prompts without
visible quality gain at this scale (a single 768² PNG screenshot is
~300 KB base64; JPEG q80 is ~40 KB).

Reasoning post-processing
-------------------------
smolagents auto-appends its configured closing code-block tag (default
``</code>``) to the model output when the model didn't already end on it
— see ``smolagents/src/smolagents/agents.py:2117-2122``. Models that emit
markdown triple-backtick python fences end up with a dangling unpaired
``</code>`` saved in the ``reasoning`` field. We strip that trailing tag
when there is no matching opening ``<code>`` in the same reasoning blob,
so the rendered transcript reads as the model actually wrote it.
"""
from __future__ import annotations

import re
from typing import Any

from .base import (
    RenderResult,
    StepCoord,
    TASK_ANCHOR,
    TranscriptBlock,
    pil_image_part,
    task_image_parts,
)


# Compression knobs applied to all images this renderer emits. JPEG q80 at
# max-dim 1024 trades essentially-zero perceptual quality for ~85% prompt
# bandwidth on the typical 600-768² mmsearch screenshots.
_IMG_OPTS = {"max_dim": 1024, "jpeg_quality": 80}


# Matches the str() of a PIL Image: ``<PIL.Image.Image image mode=RGB size=600x600 at 0x71B...>``.
# Captures (mode, width, height) so we can preserve them in the marker.
_PIL_REPR_RE = re.compile(
    r"<PIL\.Image\.Image\s+image\s+mode=(\w+)\s+size=(\d+)x(\d+)\s+at\s+0x[0-9a-fA-F]+>"
)


def _format_step(coord: str, agent: str, body: str, kind_tag: str = "") -> str:
    suffix = f" ({kind_tag})" if kind_tag else ""
    return f"Step {coord} | Agent: {agent}{suffix}\n{body.rstrip()}"


def _strip_unpaired_close_code(text: str) -> str:
    """Remove a trailing ``</code>`` if the text has no opening ``<code>``.

    smolagents always appends its closing code-block tag to the model
    output, even when the model emitted markdown triple-backtick fences
    instead of ``<code>``. The result is an unpaired ``</code>`` saved
    verbatim in ``reasoning``. Strip exactly one trailing copy here when
    the opening tag is absent.
    """
    if not text:
        return text
    if "<code>" in text:
        return text
    stripped = text.rstrip()
    if stripped.endswith("</code>"):
        return stripped[: -len("</code>")].rstrip()
    return text


def _rewrite_pil_placeholders(
    observation: str, n_images: int
) -> tuple[str, int]:
    """Replace ``<PIL.Image.Image ...>`` reprs with numbered markers.

    Returns ``(new_text, n_substituted)`` where ``n_substituted`` is the
    count of placeholders that map to an actual image (capped at
    ``n_images``). Extra placeholders beyond ``n_images`` are left as-is
    (so the text still flags that an image was involved, even though no
    matching part is anchored).
    """
    if n_images <= 0 or not observation:
        return observation, 0
    counter = {"i": 0}

    def _sub(m: re.Match) -> str:
        counter["i"] += 1
        if counter["i"] > n_images:
            return m.group(0)
        _mode, w, h = m.group(1), m.group(2), m.group(3)
        return f"[observation_image #{counter['i']} size={w}x{h}]"

    new_text = _PIL_REPR_RE.sub(_sub, observation)
    return new_text, min(counter["i"], n_images)


def render(release: dict) -> RenderResult:
    blocks: list[TranscriptBlock] = []
    step_index: list[tuple[str, StepCoord]] = []

    # Task images (the user's question image, e.g. mmsearch question
    # screenshot) — emitted as a TASK_ANCHOR block so the assembler
    # splices them next to the question, before the transcript.
    task_imgs = task_image_parts(release, **_IMG_OPTS)
    if task_imgs:
        blocks.append(TranscriptBlock(coord=TASK_ANCHOR, text="", images=task_imgs))

    # browsecomp_vl-style trick: smolagents re-passes the task image to
    # the agent on every action step (it lives in the vision context),
    # and the converter dumped that copy into ``observation_images``.
    # The model already has the task image at the top of the prompt, so
    # we drop bytes-identical duplicates here to save tokens and avoid
    # surfacing meaningless ``[observation_image #N source="<inline>"]``
    # markers that the user can't make sense of.
    task_image_blobs: set[str] = {
        img.get("data") for img in ((release.get("task") or {}).get("images") or [])
        if isinstance(img, dict) and img.get("data")
    }

    framework_agent = "agent"  # smolagents traces only carry one agent

    # Cursor-based coord. Walk non-framing entries and assign sequential
    # integers. Cursor starts at 0 if the first non-framing entry is a
    # planning step, else 1 — so a P.1-style trace with no planning entry
    # keeps the old action-only numbering (action #1 -> coord 1).
    cursor: int | None = None

    for entry in release.get("trajectory") or []:
        kind = entry.get("kind")
        if kind in ("user", "final_answer", None):
            # Framing entries don't get a coord. final_answer is harvested
            # separately below.
            continue
        if cursor is None:
            cursor = 0 if kind == "planning" else 1
        coord = str(cursor)

        if kind == "planning":
            plan_text = (entry.get("plan") or "").strip()
            body = f"[plan]\n{plan_text}\n[/plan]" if plan_text else "(empty plan)"
            blocks.append(TranscriptBlock(
                coord=coord,
                text=_format_step(coord, framework_agent, body, kind_tag="planning"),
            ))
            # Native coord for a planning block is just the cursor itself
            # — there's no upstream ActionStep.step_number to record.
            step_index.append((coord, (cursor,)))
            cursor += 1
            continue

        if kind != "action":
            # Unknown entry kind — emit a placeholder block so coord
            # accounting stays consistent with the migration script.
            blocks.append(TranscriptBlock(
                coord=coord,
                text=_format_step(coord, framework_agent, f"(unknown kind={kind!r})"),
            ))
            step_index.append((coord, (cursor,)))
            cursor += 1
            continue

        # ----- action step -----
        step_number = entry.get("step_number")  # original ActionStep number, kept for traceability
        reasoning = _strip_unpaired_close_code((entry.get("reasoning") or "").strip())
        observation = (entry.get("observation") or "").strip()

        # Per-step observation images (e.g. mmsearch image-search results),
        # minus any byte-identical re-attachment of the task image.
        valid_imgs = [
            img for img in (entry.get("observation_images") or [])
            if isinstance(img, dict)
            and img.get("data")
            and img.get("data") not in task_image_blobs
        ]
        step_image_parts = [pil_image_part(img, **_IMG_OPTS) for img in valid_imgs]

        # Rewrite the inline PIL repr placeholders in the observation
        # text so they read as `[observation_image #N size=AxB]`.
        observation_rewritten, n_subbed = _rewrite_pil_placeholders(
            observation, len(valid_imgs)
        )
        # If the observation didn't contain repr placeholders (or fewer
        # than we have images), prepend markers for the unmatched tail
        # so every anchored image is announced in the text.
        leftover_markers: list[str] = []
        for k in range(n_subbed, len(valid_imgs)):
            img = valid_imgs[k]
            # Best-effort size hint: we can't introspect bytes here cheaply,
            # so omit dimensions for prefix-only markers.
            src = img.get("source") or "<inline>"
            leftover_markers.append(
                f'[observation_image #{k + 1} source="{src}"]'
            )

        body_parts: list[str] = []
        if reasoning:
            body_parts.append(f"[output]\n{reasoning}\n[/output]")
        if leftover_markers or observation_rewritten:
            inner = "\n".join(
                leftover_markers
                + ([observation_rewritten] if observation_rewritten else [])
            )
            body_parts.append(f"[observation]\n{inner}\n[/observation]")
        body = "\n".join(body_parts) if body_parts else "(empty step)"

        native = (step_number,) if isinstance(step_number, int) else (cursor,)
        blocks.append(TranscriptBlock(
            coord=coord,
            text=_format_step(coord, framework_agent, body),
            images=step_image_parts,
        ))
        step_index.append((coord, native))
        cursor += 1

    final_answer = None
    for entry in release.get("trajectory") or []:
        if entry.get("kind") == "final_answer":
            final_answer = entry.get("content")
            break

    return RenderResult(
        blocks=blocks,
        # Single-agent sequential coords. Step 0 (when present) is a
        # planning turn — the agent's strategy/facts-survey before any
        # tool use. Subsequent action steps are numbered consecutively.
        # Mid-trajectory ``(planning)`` blocks indicate a re-plan and
        # also consume an index. If no planning is present the sequence
        # starts at Step 1 (the first action).
        step_format_hint=(
            "Step coords are sequential integers across the trajectory. "
        ),
        step_index=step_index,
        trajectory_length=len(step_index),
        final_answer=final_answer,
        extras={
            "framework": release.get("framework"),
            "benchmark": release.get("benchmark"),
            "modality": release.get("modality"),
            "topology": "single",
            "agents": release.get("agents") or [],
        },
    )
