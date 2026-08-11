"""DyLAN renderer (round-grouped multi-agent debate with explicit roles).

Step coordinate: ``step R.P`` where R = round (0-indexed) and P =
``turn["position"]`` (0-indexed, recorded by the framework). Position
values are NOT necessarily contiguous within a round — DyLAN's adaptive
agent-selection skips agents in later rounds (e.g. round 2 may have only
positions 0, 2, 3 because position 1 was dropped that round). The
renderer preserves whatever positions the source records so the step
coordinates we display match the source data exactly.

Per-turn body uses ``[output]`` / ``[/output]``, identical to the debate
renderer. The agent header shows only ``agent_id`` — DyLAN's role labels
(Economist, Doctor, Lawyer, Mathematician) are intentionally dropped so
they can't bias the attribution model toward picking the agent whose
role sounds most relevant to the question's domain.
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

    task_imgs = task_image_parts(release)
    if task_imgs:
        blocks.append(TranscriptBlock(coord=TASK_ANCHOR, text="", images=task_imgs))

    for entry in release.get("trajectory") or []:
        if entry.get("kind") != "round":
            continue
        round_ = int(entry.get("round", 0))
        for turn in entry.get("turns") or []:
            agent_id = str(turn.get("agent_id") or "unknown")
            position = int(turn.get("position", 0))
            output = (turn.get("output") or "").strip()
            coord = coord_str_hier(round_, position)

            body = f"[output]\n{output}\n[/output]" if output else "(empty turn)"
            blocks.append(TranscriptBlock(
                coord=coord,
                text=f"Step {coord} | Agent: {agent_id}\n{body}",
            ))
            step_index.append((coord, (round_, position)))

    final_answer = None
    for entry in release.get("trajectory") or []:
        if entry.get("kind") == "final_answer":
            final_answer = entry.get("content")
            break

    return RenderResult(
        blocks=blocks,
        step_format_hint=(
            "step R.P where R is the round and P is the "
            "agent's position within that round (0-indexed, as recorded "
            "by DyLAN — values may be non-contiguous when the framework "
            "skips agents in later rounds). The agent cannot see the other agents' turns in the same round."
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
