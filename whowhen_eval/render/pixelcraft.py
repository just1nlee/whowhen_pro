"""PixelCraft renderer (round-grouped, hierarchical step format).

Step coordinate: ``step R.P`` where R = round (0-indexed) and P = position
within the round (0-indexed). Matches the PixelCraft release schema's
``ground_truth.{round, position}`` exactly.

Per-turn images: a ``critic`` or ``reasoner`` turn may have viewed a
processed/cropped image (PixelCraft tools produce ``task_<id>/N.jpg`` files
that flow into downstream agents). When the release JSON includes a
turn-level ``images`` field, this renderer lays out the step like::

    Step R.P | Agent: <name>
    [viewed image: task_<id>/N.jpg]   ← text annotation introduces the image
    <inline image>                    ← the cropped tile itself
    [output]
    <agent's reasoning about the image>
    [/output]

The annotation lives in ``TranscriptBlock.text`` (rendered upstream of
the image), the image in ``TranscriptBlock.images``, and the agent's
output in ``TranscriptBlock.body_text`` (rendered downstream of the
image). For backends that strip inline images the textual marker keeps
the trace legible — the model still sees which cropped tile was shown.
"""
from __future__ import annotations

from typing import Any

from .base import (
    RenderResult,
    StepCoord,
    TASK_ANCHOR,
    TranscriptBlock,
    coord_str_hier,
    pil_image_part,
    task_image_parts,
)


def render(release: dict) -> RenderResult:
    blocks: list[TranscriptBlock] = []
    step_index: list[tuple[str, StepCoord]] = []

    # Task images (the chart/screenshot the question is about) — pass
    # through without JPEG re-encode so chart text stays crisp.
    task_imgs = task_image_parts(release)
    if task_imgs:
        blocks.append(TranscriptBlock(coord=TASK_ANCHOR, text="", images=task_imgs))

    for entry in release.get("trajectory") or []:
        kind = entry.get("kind")
        if kind != "round":
            continue
        round_ = int(entry.get("round", 0))
        for turn in entry.get("turns") or []:
            position = int(turn.get("position", 0))
            agent = str(turn.get("agent_id") or "unknown")
            output = (turn.get("output") or "").strip()

            coord = coord_str_hier(round_, position)
            body = f"[output]\n{output}\n[/output]" if output else "(empty turn)"

            # Per-turn processed image, if any. The textual ``[viewed
            # image: ...]`` markers are emitted upstream of the inline
            # images (in ``text``); the agent's reasoning sits downstream
            # in ``body_text`` so it reads as: "agent viewed THIS image
            # → here is the image → here's what the agent concluded."
            step_imgs: list[dict[str, Any]] = []
            markers: list[str] = []
            for img in turn.get("images") or []:
                if not isinstance(img, dict) or not img.get("data"):
                    continue
                step_imgs.append(pil_image_part(img))
                src = img.get("source") or "<inline>"
                markers.append(f"[viewed image: {src}]")

            header = f"Step {coord} | Agent: {agent}"
            if markers:
                header = f"{header}\n" + "\n".join(markers)

            blocks.append(TranscriptBlock(
                coord=coord,
                text=header,
                images=step_imgs,
                body_text=body,
            ))
            step_index.append((coord, (round_, position)))

    final_answer = None
    for entry in release.get("trajectory") or []:
        if entry.get("kind") == "final_answer":
            final_answer = entry.get("content")
            break

    return RenderResult(
        blocks=blocks,
        step_format_hint="",
        step_index=step_index,
        trajectory_length=len(step_index),
        final_answer=final_answer,
        extras={
            "framework": release.get("framework"),
            "benchmark": release.get("benchmark"),
            "modality": release.get("modality"),
            "topology": "multi",
            "agents": release.get("agents") or [],
        },
    )
