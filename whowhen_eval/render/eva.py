"""EVA renderer (single-agent video orchestrator with frame_select tool).

Step coordinate: ``step N`` (0-indexed flat, dense) — the **rendered**
integer counts agent actions (assistant turns), starting at 0. Tool
turns fold into the preceding assistant step's body as observations
(no separate coord). ``step_index`` keeps the native trajectory index
as the underlying coord so the runner can map a predicted ``Step k``
back to the release schema's ``ground_truth.step`` (which is the native
trajectory index of the injected turn). The scoring rule is just
``rendered_coord = (gt.step - 2) // 2`` because each assistant step
covers two native indices (assistant + folded tool).

EVA's chat-history shape uses two non-obvious mechanical conventions
that we *normalise away* in the rendered transcript so the judge sees
something legible:

  1. **Frame delivery via ``role: user`` turn.** EVA's native-tool
     rollout originally emitted two trajectory turns per ``frame_select``
     call: a ``tool`` stub ("Returned N frames at ...") followed by a
     ``user`` turn carrying the actual image bytes. Mechanically it was
     a user message; semantically it's the tool result. The released
     traces have these merged into a single ``tool`` turn with
     ``frames`` attached and the descriptor + per-frame markers
     concatenated into ``content``.

  2. **Tool result folded as observation, not its own step.** Each
     ``frame_select`` tool turn is rendered inside the preceding
     assistant step's body as a ``[tool_output tool=frame_select]`` block
     — same pattern as smolagents. The tool turn does not get its own
     ``Step`` coord; the assistant turn that initiated the call owns the
     observation. This keeps the step count equal to the agent's action
     count, which is how a judge naturally thinks about the trajectory.

EVA is single-agent, so every step uses the generic agent name
``agent`` — matching the smolagents renderer convention (and what the
prompt's ``Agent Name:`` slot expects when there's no real role split).

Frames live as on-disk JPG/PNG files at paths relative to the release
JSON's parent directory. The runner threads that directory in via
``release["__source_dir__"]`` before calling ``render``; missing-file
frames render as text-only ``[frame N at HH:MM:SS missing]`` markers
without raising.

The renderer MUST NOT propagate ``injected`` flags into the rendered
text — the judge has to discover the injection on its own.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Optional

from .base import (
    RenderResult,
    StepCoord,
    TranscriptBlock,
    coord_str_flat,
    path_image_part,
)


# JPEG q75 + max-dim 768 keeps a 16-frame call under ~600 KB total
# base64 — affordable for a single attribution prompt while preserving
# legibility of timestamp overlays / on-screen captions the model needs
# to ground its judgment.
_FRAME_OPTS = {"max_dim": 512, "jpeg_quality": 75}


_IMAGE_TOKEN_RE = re.compile(r"<image>")
# EVA's user-frames turn re-appends "Question: <query>\n(A)...(D)..." at
# the end of the message. We strip that tail so the question doesn't
# get rendered twice (once in ``## User Question``, once in the body).
_TRAILING_QUESTION_RE = re.compile(
    r"\n*Question:\s.*\Z",
    re.DOTALL | re.IGNORECASE,
)
# Some traces also carry a "If more information is needed, call the
# frame selection tool again." reminder above the question — drop it
# too; it's a static framework instruction, not signal.
_TOOL_REMINDER_RE = re.compile(
    r"\n*If more information is needed, call the frame selection tool again\.?\s*",
    re.IGNORECASE,
)


def _format_tool_call(tc: dict) -> str:
    """One-line summary of an assistant ``tool_calls`` entry."""
    name = tc.get("name", "?")
    raw_args = tc.get("arguments", "")
    if isinstance(raw_args, str):
        try:
            args = json.loads(raw_args)
        except json.JSONDecodeError:
            args = raw_args
    else:
        args = raw_args
    if isinstance(args, dict):
        args_repr = ", ".join(f"{k}={v!r}" for k, v in args.items())
    else:
        args_repr = str(args)
    return f"{name}({args_repr})"


def _resolve_frames(
    frames: list[dict],
    source_dir: Path,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Load frame bytes via path_image_part. Returns (parts, miss_markers)."""
    parts: list[dict[str, Any]] = []
    misses: list[str] = []
    for fr in frames or []:
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


def _strip_redundant_tail(text: str) -> str:
    """Drop EVA's per-frame-turn boilerplate (tool-reminder + duplicated
    question) so the rendered observation only carries the per-frame
    timestamp markers."""
    text = _TOOL_REMINDER_RE.sub("\n", text)
    text = _TRAILING_QUESTION_RE.sub("", text)
    return text.rstrip()


def _rewrite_image_tokens(text: str, n_frames: int) -> str:
    """Replace literal ``<image>`` tokens with ``[frame N]`` markers so
    the text remains coherent even when the model can't render the
    inline images. Anything past ``n_frames`` falls back to ``<image>``."""
    if n_frames <= 0 or "<image>" not in text:
        return text
    counter = {"i": 0}

    def _sub(_m):
        i = counter["i"]
        counter["i"] += 1
        return f"[frame {i}]" if i < n_frames else "<image>"

    return _IMAGE_TOKEN_RE.sub(_sub, text)


# --------------------------------------------------------------------------- #
# Step renderers
# --------------------------------------------------------------------------- #

def _render_tool_observation(turn: dict, n_frames: int, misses: list[str]) -> str:
    """Render an EVA tool turn as a folded ``[tool_output tool=X]`` block
    that lives inside the preceding assistant step's body. The tool turn
    carries the descriptor plus per-frame markers in a single ``content``
    (merged at conversion time) and frames as ``turn["frames"]``."""
    tool_name = turn.get("tool_name") or "?"
    content = (turn.get("content") or "").strip()
    content = _strip_redundant_tail(content)
    content = _rewrite_image_tokens(content, n_frames)
    inner_lines: list[str] = []
    if content:
        inner_lines.append(content)
    if misses:
        inner_lines.extend(misses)
    if not inner_lines:
        inner_lines.append("(empty)")
    inner = "\n".join(inner_lines)
    return f"[tool_output tool={tool_name}]\n{inner}\n[/tool_output]"


# --------------------------------------------------------------------------- #
# Main entry
# --------------------------------------------------------------------------- #

def render(release: dict) -> RenderResult:
    blocks: list[TranscriptBlock] = []
    step_index: list[tuple[str, StepCoord]] = []

    source_dir = Path(release.get("__source_dir__") or ".")
    framework_agent = "agent"  # single-agent: generic name (matches smolagents)

    final_answer: str | None = None
    last_assistant_content: Optional[str] = None

    trajectory = release.get("trajectory") or []
    n = len(trajectory)
    user_question_text: Optional[str] = None
    step_counter = 0
    i = 0
    while i < n:
        turn = trajectory[i]
        kind = turn.get("kind")

        # Skip framing turns:
        #   idx 0: orchestrator system prompt (framework-meta)
        #   idx 1: initial user question — surfaced via prompts.problem
        if i == 0 and kind == "system":
            i += 1
            continue
        if i == 1 and kind == "user":
            user_question_text = (turn.get("content") or "").strip() or None
            i += 1
            continue

        # Rendered coord is a dense 0-indexed counter over agent actions
        # (assistant turns). Tool turns are folded into the preceding
        # assistant step as ``[tool_output]`` observations and don't get
        # their own coord. The scorer uses ``(gt.step - 2) // 2`` to map
        # the native trajectory idx of the injected turn back to this
        # dense coord.
        coord = coord_str_flat(step_counter)
        step_imgs: list[dict[str, Any]] = []

        if kind == "assistant":
            content = (turn.get("content") or "").strip()
            tool_calls = turn.get("tool_calls") or []
            body_parts: list[str] = []
            if content:
                body_parts.append(f"[output]\n{content}\n[/output]")
                last_assistant_content = content
            for tc in tool_calls:
                body_parts.append(f"[tool_call]\n{_format_tool_call(tc)}\n[/tool_call]")

            # Fold up to one tool turn per tool_call into this step.
            j = i + 1
            consumed = 0
            tool_budget = len(tool_calls) if tool_calls else 1
            while (
                j < n
                and trajectory[j].get("kind") == "tool"
                and consumed < tool_budget
            ):
                tool_turn = trajectory[j]
                frames = tool_turn.get("frames") or []
                imgs, misses = _resolve_frames(frames, source_dir)
                step_imgs.extend(imgs)
                body_parts.append(
                    _render_tool_observation(tool_turn, n_frames=len(imgs), misses=misses)
                )
                j += 1
                consumed += 1

            body = "\n".join(body_parts) if body_parts else "(empty assistant turn)"
            header = f"Step {coord} | Agent: {framework_agent}"
            blocks.append(TranscriptBlock(
                coord=coord, text=header, images=step_imgs, body_text=body,
            ))
            step_index.append((coord, (i,)))
            step_counter += 1
            i = j

        else:
            # Unexpected post-question turn (no kind="tool" or kind="user"
            # should appear here in the merged release schema). Render
            # gracefully but flag.
            body = f"[unknown kind={kind!r}]\n{turn.get('content') or ''}"
            header = f"Step {coord} | {kind}"
            blocks.append(TranscriptBlock(
                coord=coord, text=header, images=[], body_text=body,
            ))
            step_index.append((coord, (i,)))
            step_counter += 1
            i += 1

    # Final answer: prefer the explicit <answer>X</answer> tag in the
    # last assistant content; fall back to the raw content.
    if last_assistant_content:
        m = re.search(
            r"<answer>\s*([^<]+?)\s*</answer>",
            last_assistant_content,
            re.IGNORECASE,
        )
        final_answer = m.group(1) if m else last_assistant_content

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
            "topology": "single",
            "agents": release.get("agents") or [],
            "user_question_text": user_question_text,
        },
    )
