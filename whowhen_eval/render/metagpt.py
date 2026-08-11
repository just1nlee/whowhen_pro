"""MetaGPT renderer (linear 3-stage SOP: architect → engineer → reviewer).

Step coordinate: ``step S`` where S is the SOP ``stage`` (0-indexed) — every
shipped trace runs the same fixed 3-stage pipeline, so the stage number
also serves as the position in the trajectory:

    S=0 → architect  (technical brief)
    S=1 → engineer   (code)
    S=2 → reviewer   (review of the code)

This matches the release schema's ``ground_truth.stage`` directly. The
``agent_id`` field already encodes the role (``architect`` / ``engineer`` /
``reviewer``), so the per-turn header shows only ``Agent: <agent_id>`` —
adding a separate role label would just duplicate the same string.

Trajectory shape::

    [
        {kind: "user",    content: <task>},
        {agent_id: "architect", role: "architect", stage: 0, output: ...},
        {agent_id: "engineer",  role: "engineer",  stage: 1, output: ...},
        {agent_id: "reviewer",  role: "reviewer",  stage: 2, output: ...},
        {kind: "final_answer", content: ...},
    ]
"""
from __future__ import annotations

from typing import Any

from .base import (
    RenderResult,
    StepCoord,
    TASK_ANCHOR,
    TranscriptBlock,
    coord_str_flat,
    task_image_parts,
)


def render(release: dict) -> RenderResult:
    blocks: list[TranscriptBlock] = []
    step_index: list[tuple[str, StepCoord]] = []

    task_imgs = task_image_parts(release)
    if task_imgs:
        blocks.append(TranscriptBlock(coord=TASK_ANCHOR, text="", images=task_imgs))

    for entry in release.get("trajectory") or []:
        if entry.get("kind") in ("user", "final_answer"):
            continue
        if entry.get("stage") is None:
            continue

        agent_id = str(entry.get("agent_id") or "unknown")
        stage = int(entry.get("stage"))
        output = (entry.get("output") or "").strip()

        coord = coord_str_flat(stage)
        body = f"[output]\n{output}\n[/output]" if output else "(empty turn)"
        blocks.append(TranscriptBlock(
            coord=coord,
            text=f"Step {coord} | Agent: {agent_id}\n{body}",
        ))
        step_index.append((coord, (stage,)))

    final_answer = None
    for entry in release.get("trajectory") or []:
        if entry.get("kind") == "final_answer":
            final_answer = entry.get("content")
            break

    return RenderResult(
        blocks=blocks,
        step_format_hint=(
            "step S where S is the SOP stage (0-indexed)"
        ),
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
