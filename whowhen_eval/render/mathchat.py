"""MathChat renderer (AutoGen 2-agent dialog: assistant ↔ user_proxy).

Step coordinate: ``step N`` (1-indexed flat). Since the dialog is just
two alternating speakers, the framework's hierarchical ``(round,
position)`` is over-structured for the attribution model — we collapse
it to a single ordinal ``N`` running over the post-bootstrap turns:

    flat N = 2 * round + position

so the assistant lands on odd N (1, 3, 5, ...) and the user_proxy
Python-execution turns land on even N (2, 4, 6, ...).

The framework's ``round 0 / position 0`` user_proxy turn is the AutoGen
framing prompt that bootstraps the dialog (Python-tool instructions
plus the user's task restated). It is the system's *user input* to the
agentic dialog, not an agent reasoning step. We render it with a
``User Input`` header (no step number) and exclude it from
``step_index`` so the attribution model can't misattribute the failure
to the framing prompt.

Trajectory shape::

    [
        {kind: "user",     content: <problem>},
        {agent_id: "user_proxy", round: 0, position: 0, output: <framing>},   # User Input (unnumbered)
        {agent_id: "assistant",  round: 0, position: 1, output: <reasoning>},  # step 1
        {agent_id: "user_proxy", round: 1, position: 0, output: <exec result>},# step 2
        {agent_id: "assistant",  round: 1, position: 1, output: <reasoning>},  # step 3
        ...
        {kind: "final_answer", content: <answer>},
    ]

Per-turn body uses ``[output]`` / ``[/output]`` tags identical to
debate/dylan. Header shows only ``Agent: agent_id`` (no role label) —
``assistant`` / ``user_proxy`` are themselves the role.

GT alignment: the release schema records ``ground_truth.{round,
position}``; map to flat N via the formula above. In every shipped
trace ``GT.position == 1`` (only assistant turns are injected), so GT
always lands on an odd N.
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
        # Action turns have no ``kind`` field. Defensive: skip anything
        # missing the round/position pair.
        if entry.get("round") is None or entry.get("position") is None:
            continue

        agent_id = str(entry.get("agent_id") or "unknown")
        round_ = int(entry.get("round"))
        position = int(entry.get("position"))
        output = (entry.get("output") or "").strip()
        body = f"[output]\n{output}\n[/output]" if output else "(empty turn)"

        # Bootstrap framing turn → unnumbered "User Input" header, NOT
        # added to step_index. Emitted as a non-step block (coord=None)
        # so it sits in the transcript but doesn't get a step coordinate.
        if round_ == 0 and position == 0 and agent_id == "user_proxy":
            blocks.append(TranscriptBlock(coord=None, text=f"User Input\n{body}"))
            continue

        flat_n = 2 * round_ + position
        coord = coord_str_flat(flat_n)
        blocks.append(TranscriptBlock(
            coord=coord,
            text=f"Step {coord} | Agent: {agent_id}\n{body}",
        ))
        step_index.append((coord, (flat_n,)))

    final_answer = None
    for entry in release.get("trajectory") or []:
        if entry.get("kind") == "final_answer":
            final_answer = entry.get("content")
            break

    return RenderResult(
        blocks=blocks,
        # Empty hint = flat 1-indexed sequence; the ``Step N | Agent:
        # ...`` headers carry assistant vs user_proxy distinction
        # already, and prompts.py drops the Step Coordinate Format
        # section when the hint is empty.
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
            # Scorer convenience: GT records (round, position); flat N is
            # 2 * round + position.
            "gt_to_step_formula": "2 * round + position",
        },
    )
