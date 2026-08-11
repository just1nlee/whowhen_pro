"""Magentic-One renderer (orchestrator + 4 specialist agents).

Magentic-One uses an Orchestrator that maintains a ledger and routes work
to one of four specialists (``websurfer``, ``coder``, ``filesurfer``,
``computerterminal``) per round. Trajectory shape:

    [
        {kind: "user",  content: <task>},
        {kind: "agent", agent: "orchestrator", role: "Orchestrator",
                        round: 0, position: 0, output: <routing instruction>,
                        ledger?, next_speaker?, phase?},
        {kind: "agent", agent: "<specialist>", role: ...,
                        round: 0, position: 1, output: <specialist reply>,
                        tool_actions?, phase?},
        {kind: "agent", agent: "orchestrator", round: 1, position: 0, ...},
        ...
    ]

Step coordinate: ``step R.P`` where R is the round (0-indexed) and P is
the within-round position (0 for the orchestrator's planning turn,
1 for the specialist's response). This matches the release schema's
``ground_truth.step`` field — already a ``"R.P"`` string in the data
(e.g. ``"1.0"``, ``"3.1"``).

Per-turn body uses ``[output]`` for the agent's text and a separate
``[tool_actions]`` block when the specialist made tool calls (browser
actions, code blocks, terminal commands). The orchestrator's ledger /
next_speaker routing is NOT rendered — those fields are derivable from
the orchestrator's ``output`` (which is the ``instruction_or_question``
restated) and would inflate the prompt 3-5x for marginal gain.

Per-turn header shows only ``Agent: <agent>`` (the lowercase id,
matching ``ground_truth.agent``). The capitalised ``role`` (e.g.
``"WebSurfer"``) is dropped since it duplicates the agent id.

Magentic-One has no ``kind: final_answer`` entry — the final answer
arrives as the last orchestrator turn whose ``phase`` includes
``"final_answer"``. We surface that turn's ``output`` as the rendered
``final_answer``.
"""
from __future__ import annotations

import json
from typing import Any

from .base import (
    RenderResult,
    StepCoord,
    TASK_ANCHOR,
    TranscriptBlock,
    coord_str_hier,
    task_image_parts,
)


def _format_tool_actions(actions: list[Any]) -> str:
    """Render a tool-actions list as a compact JSON block.

    Specialist tool calls vary in shape (browser ``input_text``, ``click``,
    ``visit_url``; coder ``code_block``; computer-terminal ``shell``).
    Dump them as JSON so the structure stays self-describing without us
    having to enumerate every variant.
    """
    if not actions:
        return ""
    try:
        return json.dumps(actions, ensure_ascii=False, indent=2)
    except (TypeError, ValueError):
        return repr(actions)


def render(release: dict) -> RenderResult:
    blocks: list[TranscriptBlock] = []
    step_index: list[tuple[str, StepCoord]] = []
    final_answer: str | None = None

    task_imgs = task_image_parts(release)
    if task_imgs:
        blocks.append(TranscriptBlock(coord=TASK_ANCHOR, text="", images=task_imgs))

    for entry in release.get("trajectory") or []:
        if entry.get("kind") != "agent":
            continue
        if entry.get("round") is None or entry.get("position") is None:
            continue

        agent = str(entry.get("agent") or "unknown")
        round_ = int(entry.get("round"))
        position = int(entry.get("position"))
        output = (entry.get("output") or "").strip()
        tool_actions = entry.get("tool_actions") or []

        coord = coord_str_hier(round_, position)
        body_parts: list[str] = []
        if output:
            body_parts.append(f"[output]\n{output}\n[/output]")
        if tool_actions:
            tool_block = _format_tool_actions(tool_actions)
            if tool_block:
                body_parts.append(f"[tool_actions]\n{tool_block}\n[/tool_actions]")
        body = "\n".join(body_parts) if body_parts else "(empty turn)"

        blocks.append(TranscriptBlock(
            coord=coord,
            text=f"Step {coord} | Agent: {agent}\n{body}",
        ))
        step_index.append((coord, (round_, position)))

        # Final orchestrator turn with phase=final_answer is the system's answer.
        phase = entry.get("phase") or []
        if isinstance(phase, list) and "final_answer" in phase:
            final_answer = output

    return RenderResult(
        blocks=blocks,
        step_format_hint=(
            "Magentic-One is an orchestrator + specialist framework: a "
            "central orchestrator routes work each round to a specialist agent. "
            "Step coordinate is 'step R.P' where R is the round "
            "and P is the within-round position: P=0 is the orchestrator's "
            "turn, P=1 is the specialist's response. "
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
