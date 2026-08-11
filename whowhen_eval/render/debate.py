"""Debate renderer (round-grouped multi-agent debate).

Step coordinate: ``step R.P`` where R = round (0-indexed) and P = the
turn's ordinal position within the round (0-indexed). Debate turns do not
carry an explicit ``position`` field, so P is the index of the turn in the
round's ``turns`` list. ``ground_truth.round`` is also 0-indexed and refers
directly to the round number stored on each round entry.

Per-turn body uses the same ``[output]`` / ``[/output]`` tag pair as the
smolagents renderer, but no ``[observation]`` block — debate agents only
emit reasoning, no tool calls. The text-only modality means we never
produce per-turn image anchors.

Only the agent_id is shown in the per-turn header — the role (always
``debater`` here, but more substantive in DyLAN) is intentionally dropped
so it can't bias the attribution model toward labelling a particular role
as the failure point.
"""
from __future__ import annotations

from typing import Any

from .base import (
    RenderResult,
    StepCoord,
    TASK_ANCHOR,
    TranscriptBlock,
    coord_str_hier,
    task_image_parts,
)


def render(release: dict) -> RenderResult:
    blocks: list[TranscriptBlock] = []
    step_index: list[tuple[str, StepCoord]] = []

    # Task images are typically absent for debate (text-only modality).
    task_imgs = task_image_parts(release)
    if task_imgs:
        blocks.append(TranscriptBlock(coord=TASK_ANCHOR, text="", images=task_imgs))

    for entry in release.get("trajectory") or []:
        if entry.get("kind") != "round":
            continue
        round_ = int(entry.get("round", 0))
        for pos, turn in enumerate(entry.get("turns") or []):
            agent_id = str(turn.get("agent_id") or "unknown")
            output = (turn.get("output") or "").strip()
            coord = coord_str_hier(round_, pos)

            body = f"[output]\n{output}\n[/output]" if output else "(empty turn)"
            blocks.append(TranscriptBlock(
                coord=coord,
                text=f"Step {coord} | Agent: {agent_id}\n{body}",
            ))
            step_index.append((coord, (round_, pos)))

    final_answer = None
    for entry in release.get("trajectory") or []:
        if entry.get("kind") == "final_answer":
            final_answer = entry.get("content")
            break

    return RenderResult(
        blocks=blocks,
        step_format_hint=(
            "step R.P where R is the round and P is the "
            "turn's position within that round. The agent cannot see the other agents' turns in the same round."
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
