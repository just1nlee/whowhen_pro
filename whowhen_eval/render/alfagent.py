"""alfagent renderer (single-agent ALFWorld text agent).

Step coordinate: ``step N`` (1-indexed, matches ``ground_truth.step`` and
the source trajectory's native ``step_number``). The injected step is
literally the same integer the converter wrote into ``ground_truth.step``,
so no remapping is needed.

Trajectory shape:
    [
      {kind: "user", content},
      {kind: "action", step_number, reasoning?, action, observation,
                       is_final_answer},
      ...
    ]

We render one transcript block per ``action`` entry. The body is the union
of whatever fields are populated:

    [think]       ← model's reasoning at that step
    {reasoning}
    [/think]
    [action]      ← the env action the agent emitted
    {action}
    [/action]
    [observation] ← the env's response
    {observation}
    [/observation]

Reasoning is null for post-injection steps (the misled rollout's outputs
were not saved into the source ``history``); those blocks render with
``[action]`` + ``[observation]`` only.

ALFWorld is text-only — no per-step images, no task images. Same single
``framework_agent`` label as smolagents (no agent column in the source).
"""
from __future__ import annotations

from .base import (
    RenderResult,
    StepCoord,
    TranscriptBlock,
)


def _format_step(coord: str, agent: str, body: str) -> str:
    return f"Step {coord} | Agent: {agent}\n{body.rstrip()}"


def render(release: dict) -> RenderResult:
    blocks: list[TranscriptBlock] = []
    step_index: list[tuple[str, StepCoord]] = []

    framework_agent = "agent"

    for entry in release.get("trajectory") or []:
        kind = entry.get("kind")
        if kind in ("user", "final_answer", None):
            continue
        if kind != "action":
            # Defensive: unknown kinds still get a placeholder so coords
            # stay consistent with whatever the converter wrote.
            sn = entry.get("step_number")
            coord = str(sn) if isinstance(sn, int) else "?"
            blocks.append(TranscriptBlock(
                coord=coord,
                text=_format_step(coord, framework_agent, f"(unknown kind={kind!r})"),
            ))
            if isinstance(sn, int):
                step_index.append((coord, (sn,)))
            continue

        sn = entry.get("step_number")
        if not isinstance(sn, int):
            continue
        coord = str(sn)

        reasoning = (entry.get("reasoning") or "").strip()
        action = (entry.get("action") or "").strip()
        observation = (entry.get("observation") or "").strip()

        body_parts: list[str] = []
        if reasoning:
            body_parts.append(f"[think]\n{reasoning}\n[/think]")
        if action:
            body_parts.append(f"[action]\n{action}\n[/action]")
        if observation:
            body_parts.append(f"[observation]\n{observation}\n[/observation]")
        body = "\n".join(body_parts) if body_parts else "(empty step)"

        blocks.append(TranscriptBlock(
            coord=coord,
            text=_format_step(coord, framework_agent, body),
        ))
        step_index.append((coord, (sn,)))

    return RenderResult(
        blocks=blocks,
        # Sequential 1-indexed integers; self-explanatory once the agent
        # column is fixed.
        step_format_hint="",
        step_index=step_index,
        trajectory_length=len(step_index),
        final_answer=None,
        extras={
            "framework": release.get("framework"),
            "benchmark": release.get("benchmark"),
            "modality": release.get("modality"),
            "topology": "single",
            "agents": release.get("agents") or [],
        },
    )
