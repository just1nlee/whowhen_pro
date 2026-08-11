"""MacNet renderer (DAG pipeline with author / critic / rewriter / sink).

MacNet's release trajectory is a **flat** ordered list of turns, not the
``kind: round`` wrapping used by debate/dylan. Each shipped trace runs the
same fixed 2-layer chain: ``node_0`` (author) → ``node_1`` (critic, then
rewriter on the same node) → ``node_out`` (sink).

Step coordinate: ``step R.P`` where R = layer depth in the DAG (0 = source
layer, 1 = inner-edge layer, 2 = sink layer) and P = within-layer position.
This matches the release schema's ``ground_truth.round`` /
``ground_truth.position`` (both 0-indexed). The mapping for the shipped
2-layer chain is::

    R=0, P=0 → author    (node_0,  source_generate_code)
    R=1, P=0 → critic    (node_1,  critic stage of edge_rewrite_code)
    R=1, P=1 → rewriter  (node_1,  edge_rewrite_code)
    R=2, P=0 → sink      (node_out, sink_passthrough)

GT alignment: every shipped trace has GT.round ∈ {0, 1} with GT.role ∈
{author, critic} — the renderer's R coordinate matches GT.round directly.
For C.3 (S2_critic injection) the critic's ``output`` is the injected
``review_comment``; for the other modes (S1_S3_output) the author's code
output is overwritten.

Per-turn header shows only ``Agent: <node_id>`` (matches the release
schema's ``ground_truth.agent`` format: ``node_0``, ``node_1``,
``node_out``). The role (author/critic/rewriter/sink) is **not** in the
header — instead it is encoded into the step coordinate so the
``(agent_id, step)`` pair is unique even when the same node appears
twice (node_1 acts as both critic and rewriter on the inner edge layer):

    R=0, P=0 → author    (node_0)
    R=1, P=0 → critic    (node_1, critic sub-step)
    R=1, P=1 → rewriter  (node_1, rewriter sub-step)
    R=2, P=0 → sink      (node_out)

This keeps the model's predicted ``Agent Name`` field aligned with
``GT.agent`` (no role/agent_id confusion), and the predicted ``Step
Number`` field carries the role information.

The renderer publishes the role↔coord convention in
``extras["role_to_coord"]`` so the scorer can map ``GT.role`` to the
renderer's expected step coordinate.
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


# Map a turn's (role, ordinal-of-this-role-in-trace) to its (R, P) coord.
# R is the DAG layer depth, P is the within-layer position. Author and sink
# always sit alone in their layers; node_1 holds critic at P=0 then rewriter
# at P=1 on the inner-edge layer (R=1).
_ROLE_LAYER: dict[str, int] = {
    "author":   0,
    "critic":   1,
    "rewriter": 1,
    "sink":     2,
}
_ROLE_LAYER_POS: dict[str, int] = {
    "author":   0,
    "critic":   0,
    "rewriter": 1,
    "sink":     0,
}


def render(release: dict) -> RenderResult:
    blocks: list[TranscriptBlock] = []
    step_index: list[tuple[str, StepCoord]] = []

    task_imgs = task_image_parts(release)
    if task_imgs:
        blocks.append(TranscriptBlock(coord=TASK_ANCHOR, text="", images=task_imgs))

    for entry in release.get("trajectory") or []:
        if entry.get("kind") in ("user", "final_answer"):
            continue
        # Action turns carry no ``kind`` field in macnet — they're plain
        # dicts with agent_id/role/stage/output. Defensive: skip anything
        # that doesn't look like one.
        if not entry.get("role"):
            continue

        agent_id = str(entry.get("agent_id") or "unknown")
        role = str(entry.get("role"))
        output = (entry.get("output") or "").strip()

        round_ = _ROLE_LAYER.get(role)
        position = _ROLE_LAYER_POS.get(role)
        if round_ is None or position is None:
            # Unknown role (e.g. multi-predecessor critic flavour): fall
            # back to the inner layer with position = how many turns
            # we've already placed there.
            round_ = 1
            position = sum(1 for _, (r, _p) in step_index if r == 1)

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
            # role↔coord convention for the scorer: GT.role maps to the
            # listed (round, position) when GT.agent is the node holding
            # that role on the inner edge layer.
            "role_to_coord": {
                "author":   (0, 0),
                "critic":   (1, 0),
                "rewriter": (1, 1),
                "sink":     (2, 0),
            },
        },
    )
