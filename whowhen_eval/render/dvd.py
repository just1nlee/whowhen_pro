"""DVD (DeepVideoDiscovery) renderer — orchestrator + frame_inspect_agent + retrieval tools.

Step coordinate: ``step N`` (0-indexed flat) — the **rendered** integer is
``trajectory_idx - 2``: trajectory[0] (system) and trajectory[1] (user
question) are framing, not steps, so the first agent action lands at
``Step 0``. The ``step_index`` keeps the native coord as the underlying
trajectory index so the runner / scorer can still match a model's
predicted ``Step k`` against the release schema's ``ground_truth.step``
(which equals the native trajectory index) via the ``coord_str ↔ native``
map.

Trajectory shape (release format):

    [
        {kind: "system",  content},
        {kind: "user",    content: <question + tool list>},
        {kind: "assistant", content: <orchestrator reasoning>,
                            tool_calls?: [{id, name, arguments}, ...]},
        {kind: "tool", tool_name, tool_call_id, content: <tool return text>,
                       frames?: [{index, time_s, path}, ...]},
        {kind: "assistant", ...},  # next ReAct iteration
        ...
    ]

Agents vs tools
---------------
DVD's ``frame_inspect_tool`` internally spins up a separate VLM that
sees the frames at the requested ``time_ranges_hhmmss`` and returns a
natural-language description. The orchestrator only ever sees that
description string — frames never reach it. So the inner VLM is
materially a *sub-agent*, not a dumb tool, and a P.1 (perception) error
attributes to it, not to the orchestrator.

We render that distinction in two places:

1. Header label: ``Agent: frame_inspect_agent`` (like the orchestrator)
   for the inner VLM's reply step; ``Tool: <name>`` for the plain
   retrievers (``global_browse_tool``, ``clip_search_tool``, ``finish``).
2. Body framing: ``[output]<vlm reply>[/output]`` for the agent (same
   shape as the orchestrator's ``[output]``) vs ``[tool_result]…[/tool_result]``
   for the dumb tools. Frames the VLM consumed are interleaved between
   a leading ``[input_frames at <range>]`` marker and the ``[output]``
   block so the judge can verify the VLM's reply.

For consistency, the orchestrator's ``[tool_call]`` line that *invokes*
the inner VLM is also rebranded to ``frame_inspect_agent(...)`` — the
underlying DVD code uses ``frame_inspect_tool``, but in the rendered
prompt the orchestrator and the sub-agent share one canonical name so
the judge sees a coherent two-agent narrative.

User question
-------------
The release ``task.query`` is the bare question; ``trajectory[1].content``
is the *formatted* version actually shown to the orchestrator (question
+ MCQ options + tool list + ReAct framing). We expose that formatted
content via ``extras["user_question_text"]`` so the prompt assembler can
inject it as the ``## User Question`` section instead of recomposing
from raw ``task.query`` + options.

Per-step frame budget
---------------------
``_FRAME_CAP_PER_STEP=8`` evenly-sampled. The inner VLM may have
consumed up to 50 frames per call; reproducing all of them in a
multi-call trace would routinely blow context. Eight evenly-spaced
frames preserve the temporal range while keeping payload manageable.

The renderer MUST NOT propagate ``injected`` flags into the rendered
text — the judge has to discover the injection on its own.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from .base import (
    RenderResult,
    StepCoord,
    TranscriptBlock,
    coord_str_flat,
    path_image_part,
)


# JPEG q75 + max-dim 768 — same compression EVA uses. Frames are 720p
# JPGs on disk; this trims them to ~30-40 KB each base64.
_FRAME_OPTS = {"max_dim": 512, "jpeg_quality": 75}

# Per-step frame cap.
_FRAME_CAP_PER_STEP = 8

# Canonical sub-agent name used in both the orchestrator's [tool_call]
# line and the responder step's "Agent: ..." header. The underlying DVD
# code names it ``frame_inspect_tool``; we rebrand at the prompt edge
# so the inner VLM reads as a peer agent the orchestrator delegates to.
_FRAME_INSPECT_AGENT_NAME = "frame_inspect_agent"


def _format_tool_call(tc: dict) -> str:
    """Render one ``tool_calls`` entry as a single line.

    Strips DVD's auto-injected ``database`` argument (a 100 KB captions-DB
    blob the orchestrator never authored). Rebrands ``frame_inspect_tool``
    → ``frame_inspect_agent`` so the orchestrator's invocation matches
    the responder step's header.
    """
    name = tc.get("name", "?")
    if name == "frame_inspect_tool":
        name = _FRAME_INSPECT_AGENT_NAME
    raw_args = tc.get("arguments", "")
    if isinstance(raw_args, str):
        try:
            args = json.loads(raw_args)
        except json.JSONDecodeError:
            args = raw_args
    else:
        args = raw_args
    if isinstance(args, dict):
        args = {k: v for k, v in args.items() if k != "database"}
        args_repr = ", ".join(f"{k}={v!r}" for k, v in args.items())
    else:
        args_repr = str(args)
    return f"{name}({args_repr})"


def _evenly_sample(items: list, k: int) -> list:
    """Pick ``k`` evenly-spaced items (preserving order). Returns ``items``
    unchanged when ``len(items) <= k``."""
    n = len(items)
    if n <= k or k <= 0:
        return items
    return [items[round(i * (n - 1) / (k - 1))] for i in range(k)]


def _resolve_frames(
    frames: list[dict],
    source_dir: Path,
    cap: int,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Sample + load frame bytes. Returns (image_parts, miss_markers)."""
    sampled = _evenly_sample(list(frames or []), cap)
    parts: list[dict[str, Any]] = []
    misses: list[str] = []
    for fr in sampled:
        rel = fr.get("path", "")
        if not rel:
            continue
        p = source_dir / rel
        try:
            parts.append(path_image_part(p, **_FRAME_OPTS))
        except FileNotFoundError:
            ts = _fmt_time_s(fr.get("time_s"))
            misses.append(f"[frame {fr.get('index', '?')} at {ts} missing]")
    return parts, misses


def _fmt_time_s(t) -> str:
    if t is None:
        return "??:??"
    try:
        t = float(t)
    except (TypeError, ValueError):
        return str(t)
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    if h:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def _format_time_ranges(tr: Any) -> str:
    """Coerce a time_ranges_hhmmss arg into a compact ``HH:MM:SS-HH:MM:SS``
    string. Accepts list-of-pair, list-of-tuple, or already-string inputs."""
    if tr is None:
        return ""
    if isinstance(tr, str):
        return tr
    if isinstance(tr, list):
        out: list[str] = []
        for item in tr:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                out.append(f"{item[0]}-{item[1]}")
            else:
                out.append(str(item))
        return ", ".join(out)
    return str(tr)


def _render_assistant_body(turn: dict) -> str:
    content = (turn.get("content") or "").strip()
    tool_calls = turn.get("tool_calls") or []
    parts: list[str] = []
    if content:
        parts.append(f"[output]\n{content}\n[/output]")
    for tc in tool_calls:
        parts.append(f"[tool_call]\n{_format_tool_call(tc)}\n[/tool_call]")
    return "\n".join(parts) if parts else "(empty assistant turn)"


def _render_plain_tool_body(turn: dict) -> str:
    """Body for a non-agent tool turn (global_browse_tool, clip_search_tool,
    finish). Wraps the raw tool return in a ``[tool_result]`` block."""
    tool_name = turn.get("tool_name") or "?"
    content = (turn.get("content") or "").strip() or "(empty)"
    return f"[tool_result tool={tool_name}]\n{content}\n[/tool_result]"


def render(release: dict) -> RenderResult:
    blocks: list[TranscriptBlock] = []
    step_index: list[tuple[str, StepCoord]] = []

    source_dir = Path(release.get("__source_dir__") or ".")

    # Stash the time_ranges_hhmmss from each pending frame_inspect call
    # so the responder step (the inner-VLM reply) can label the frame
    # window. Keyed by tool_call_id.
    pending_inspect_ranges: dict[str, str] = {}

    final_answer: Optional[str] = None
    user_question_text: Optional[str] = None

    trajectory = release.get("trajectory") or []
    for i, turn in enumerate(trajectory):
        kind = turn.get("kind")

        # idx 0: orchestrator system prompt — framework boilerplate.
        # idx 1: initial user message — kept as the formatted question
        #         and surfaced via extras for the prompt assembler.
        if i == 0 and kind == "system":
            continue
        if i == 1 and kind == "user":
            user_question_text = (turn.get("content") or "").strip() or None
            continue

        # Rendered coord starts at 0 for the first agent action.
        coord = coord_str_flat(i - 2)

        if kind == "assistant":
            # Stash time_ranges for any frame_inspect calls so the next
            # tool turn can show "[input_frames at <range>]".
            for tc in turn.get("tool_calls") or []:
                if tc.get("name") != "frame_inspect_tool":
                    continue
                tcid = tc.get("id")
                raw_args = tc.get("arguments")
                args: dict = {}
                if isinstance(raw_args, str):
                    try:
                        args = json.loads(raw_args)
                    except json.JSONDecodeError:
                        args = {}
                elif isinstance(raw_args, dict):
                    args = raw_args
                tr = args.get("time_ranges_hhmmss") or args.get("time_ranges")
                if tcid and tr:
                    pending_inspect_ranges[tcid] = _format_time_ranges(tr)

            body = _render_assistant_body(turn)
            header = f"Step {coord} | Agent: orchestrator"
            blocks.append(TranscriptBlock(
                coord=coord, text=header, images=[], body_text=body,
            ))
            step_index.append((coord, (i,)))

        elif kind == "tool":
            tool_name = turn.get("tool_name") or "?"
            tcid = turn.get("tool_call_id") or ""

            if tool_name == "frame_inspect_tool":
                # Render as a peer agent: [input_frames] header note,
                # then the frames themselves, then [output]<vlm reply>[/output].
                time_range = pending_inspect_ranges.pop(tcid, None)
                step_imgs, misses = _resolve_frames(
                    turn.get("frames") or [], source_dir, _FRAME_CAP_PER_STEP,
                )
                header_lines = [f"Step {coord} | Agent: {_FRAME_INSPECT_AGENT_NAME}"]
                if step_imgs or misses:
                    n_resolved = len(step_imgs)
                    if time_range:
                        header_lines.append(
                            f"[input_frames at {time_range}, {n_resolved} frame(s)]"
                        )
                    else:
                        header_lines.append(
                            f"[input_frames, {n_resolved} frame(s)]"
                        )
                header_text = "\n".join(header_lines)

                content = (turn.get("content") or "").strip() or "(empty)"
                body_lines = []
                if misses:
                    body_lines.extend(misses)
                body_lines.append(f"[output]\n{content}\n[/output]")
                body_text = "\n".join(body_lines)

                blocks.append(TranscriptBlock(
                    coord=coord,
                    text=header_text,
                    images=step_imgs,
                    body_text=body_text,
                ))
                step_index.append((coord, (i,)))

            else:
                # Plain retrieval tool or finish.
                if tool_name == "finish":
                    final_answer = (turn.get("content") or "").strip() or final_answer
                body = _render_plain_tool_body(turn)
                header = f"Step {coord} | Tool: {tool_name}"
                blocks.append(TranscriptBlock(
                    coord=coord, text=header, images=[], body_text=body,
                ))
                step_index.append((coord, (i,)))

        else:
            # Unknown kind past idx 1 — render gracefully but keep coord.
            body = f"[unknown kind={kind!r}]\n{(turn.get('content') or '')}"
            header = f"Step {coord} | {kind}"
            blocks.append(TranscriptBlock(
                coord=coord, text=header, images=[], body_text=body,
            ))
            step_index.append((coord, (i,)))

    # Fallback for A.2 / truncated traces that never call ``finish``.
    if final_answer is None:
        final_answer = release.get("final_answer")

    return RenderResult(
        blocks=blocks,
        step_format_hint="",  # plain 0-indexed integer step — self-explanatory
        step_index=step_index,
        trajectory_length=len(step_index),
        final_answer=final_answer,
        extras={
            "framework": release.get("framework"),
            "benchmark": release.get("benchmark"),
            "modality": release.get("modality"),
            "topology": "multi",  # orchestrator + frame_inspect_agent
            "agents": release.get("agents") or [],
            # Formatted question shown to the orchestrator at trajectory[1].
            # Prompt assemblers should prefer this over recomposing from
            # raw task.query + options, since it preserves the exact
            # phrasing the agent saw (incl. tool list + ReAct framing).
            "user_question_text": user_question_text,
        },
    )
