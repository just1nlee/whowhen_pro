"""Shared types + helpers for per-framework renderers.

Renderers return a ``RenderResult`` describing the rendered conversation
plus the bookkeeping needed by the runner (image content parts, step index
for scoring, framework-specific topology metadata).
"""
from __future__ import annotations

import base64
from dataclasses import dataclass, field
from io import BytesIO
from typing import Any, Optional, Union


# Sentinel coord used in ``image_anchors`` for task-level images (those
# attached to the user's question, not to a particular agent step). Picked
# so it cannot collide with any framework's real coord_str (all real coords
# start with a digit). The runner is expected to splice these images next
# to the user-question section, before the transcript.
TASK_ANCHOR = "task"


# ---------------------------------------------------------------------------
# Step coordinates
# ---------------------------------------------------------------------------

# Canonical step coordinate as used in prompts AND in the JSONL prediction:
#   * flat frameworks (smolagents, mathchat, metagpt, magentic, dvd, eva):
#     ``StepCoord = (step_number,)`` — 1-tuple, 1-indexed
#   * round-grouped frameworks (debate, dylan, pixelcraft, macnet):
#     ``StepCoord = (round, position)`` — 2-tuple, 1-indexed for debate/dylan,
#     0-indexed for pixelcraft/macnet (whichever matches the release schema's
#     ``ground_truth`` coordinate)
#
# The renderer also returns a ``coord_to_str`` callable so the runner can
# stringify both predicted and ground-truth coordinates uniformly when
# building / parsing prompts.
StepCoord = tuple[int, ...]


# ---------------------------------------------------------------------------
# Image part helper (OpenAI ``image_url`` content-part shape)
# ---------------------------------------------------------------------------

def pil_image_part(
    release_image: dict,
    *,
    max_dim: Optional[int] = None,
    jpeg_quality: Optional[int] = None,
) -> dict[str, Any]:
    """Convert a release-schema image entry to an OpenAI ``image_url`` part.

    Release entries look like
    ``{"__type__": "PIL.Image", "data": "<base64>", "source": "..."}``;
    by default the base64 PNG payload is wrapped verbatim as a
    ``data:image/png;base64,...`` URL.

    Optional in-place compression (caller opt-in):

    * ``max_dim`` — if set, downsample so the longest side ≤ ``max_dim``
      (PIL ``thumbnail``, preserves aspect ratio).
    * ``jpeg_quality`` — if set, re-encode the image as JPEG at this
      quality. Cuts payload by ~85% on PNG screenshots / chart-search
      results without visible degradation at q≥75.

    If either knob is set we decode-then-re-encode via PIL; the payload
    becomes JPEG (regardless of the input mime). When both are ``None``
    the bytes pass through untouched, preserving any lossless content
    (charts, plot text) the renderer wants to keep pristine.
    """
    if not isinstance(release_image, dict):
        raise ValueError(f"expected dict, got {type(release_image)!r}")
    data = release_image.get("data")
    if not isinstance(data, str) or not data:
        raise ValueError("release image entry has empty/non-str 'data'")
    if max_dim is None and jpeg_quality is None:
        # Pass through whatever the release encoder produced (typically PNG).
        mime = release_image.get("mime") or "image/png"
        url = f"data:{mime};base64,{data}"
        return {"type": "image_url", "image_url": {"url": url}}

    # Re-encode path: decode → optional resize → JPEG re-encode.
    try:
        from PIL import Image  # local import keeps the helper cheap when unused
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "Pillow required for image downsizing; install pillow"
        ) from e
    raw = base64.b64decode(data)
    img = Image.open(BytesIO(raw))
    if max_dim is not None:
        img.thumbnail((max_dim, max_dim))
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    buf = BytesIO()
    img.save(buf, "JPEG", quality=int(jpeg_quality or 85), optimize=True)
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    url = f"data:image/jpeg;base64,{encoded}"
    return {"type": "image_url", "image_url": {"url": url}}


def text_part(text: str) -> dict[str, Any]:
    """Convenience: build an OpenAI text content part."""
    return {"type": "text", "text": text}


def path_image_part(
    path,
    *,
    max_dim: Optional[int] = 768,
    jpeg_quality: Optional[int] = 75,
) -> dict[str, Any]:
    """Read an on-disk image and return an OpenAI ``image_url`` part.

    Mirrors ``pil_image_part`` but starts from a filesystem path instead
    of a release dict. Used by video renderers (eva, dvd) where frames
    live as PNG/JPG files in ``assets/`` rather than inline base64.

    The PIL re-encode path is always taken: video traces routinely carry
    50+ frames per tool call, so aggressive JPEG compression is the
    right default. Pass ``max_dim=None, jpeg_quality=None`` to disable
    re-encoding (rare; only useful when bit-exact frames are needed).
    """
    from pathlib import Path
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"image path does not exist: {p}")

    if max_dim is None and jpeg_quality is None:
        # Pass-through: emit a data URL with the file's native bytes.
        import mimetypes
        mime, _ = mimetypes.guess_type(p.name)
        mime = mime or "image/jpeg"
        encoded = base64.b64encode(p.read_bytes()).decode("ascii")
        return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}}

    try:
        from PIL import Image
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("Pillow required for image downsizing") from e
    img = Image.open(p)
    if max_dim is not None:
        img.thumbnail((max_dim, max_dim))
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    buf = BytesIO()
    img.save(buf, "JPEG", quality=int(jpeg_quality or 80), optimize=True)
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}}


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class TranscriptBlock:
    """One unit of the rendered transcript.

    A block is either:

    * a **task block** — ``coord == TASK_ANCHOR``, ``text == ""``,
      ``images = [task images...]``. Task images attach to the user's
      question, not to any agent turn. Emitted once at the head of
      ``RenderResult.blocks`` when the trace has task images.
    * a **step block** — ``coord`` is the step coordinate string used in
      prompts (e.g. ``"1"``, ``"1.2"``, ``"3.0"``). ``text`` is the full
      multi-line rendering of that step (header + ``[output]`` /
      ``[observation]`` body). ``images`` are the observation images
      attached to that step.
    * a **non-step text block** — ``coord is None``, ``text`` carries
      content that's part of the transcript but not associated with any
      step (e.g. the MathChat bootstrap "User Input" framing turn).

    The all-at-once prompt assembler walks blocks in order and produces
    interleaved OpenAI content parts:

        for block in blocks:
            if block.text:
                parts.append(text_part(block.text + "\n"))
            if block.images:
                parts.extend(block.images)
            if block.body_text:
                parts.append(text_part(block.body_text + "\n"))

    The ``text → images → body_text`` shape lets a block split its
    rendering around the inline images, which is how pixelcraft frames a
    cropped tile: header + ``[viewed image: ...]`` marker upstream of the
    image, then ``[output]...[/output]`` (the agent's reasoning *about*
    the image) downstream. Renderers that don't need this just leave
    ``body_text`` empty — the resulting layout matches the original
    text-after-images contract for every existing framework except
    pixelcraft.
    """

    coord: Optional[str]
    text: str
    images: list[dict[str, Any]] = field(default_factory=list)
    body_text: str = ""


@dataclass
class RenderResult:
    """Output of a per-framework ``render(release)`` call.

    Fields
    ------
    blocks
        Ordered list of ``TranscriptBlock`` objects covering the full
        transcript. The first block (if any) is the task block carrying
        the user-question images; the rest are step blocks (one per
        agent turn) plus optional non-step text blocks for framing
        content. Drives all downstream message assembly.
    step_format_hint
        Single-sentence description of how step coordinates are written
        (e.g. ``"step N (1-indexed)"`` or ``"step R.P ..."``). Empty
        string signals "self-explanatory" — prompts.py omits the section.
    step_index
        Ordered list of ``(coord_str, native_coord)`` pairs covering
        every step block. ``coord_str`` is the prompt-side string;
        ``native_coord`` is the framework's internal tuple suitable for
        direct comparison against ``ground_truth``. Excludes task and
        non-step blocks.
    trajectory_length
        Number of step blocks (excludes ``kind: user``,
        ``kind: final_answer``, framing turns).
    final_answer
        The system-produced final answer string from the release, if any.
    extras
        Free-form dict for framework-specific knobs (topology, agent
        registry, role map, etc.).

    Convenience properties
    ----------------------
    ``chat_content`` / ``images`` / ``image_anchors`` are derived from
    ``blocks`` for back-compat with eyeball tests (and as a textual fallback
    when a backend doesn't accept multimodal input).
    """

    blocks: list[TranscriptBlock] = field(default_factory=list)
    step_format_hint: str = ""
    step_index: list[tuple[str, StepCoord]] = field(default_factory=list)
    trajectory_length: int = 0
    final_answer: Optional[str] = None
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def chat_content(self) -> str:
        """Flat-text rendering of the transcript: concatenate every
        non-task block's text and body_text with single newlines. Used
        by eyeball tests and as a degraded view for text-only judges."""
        out: list[str] = []
        for b in self.blocks:
            if b.coord == TASK_ANCHOR:
                continue
            if b.text:
                out.append(b.text)
            if b.body_text:
                out.append(b.body_text)
        return "\n".join(out)

    @property
    def images(self) -> list[dict[str, Any]]:
        """All image parts in order (task first, then per-step in
        trajectory order). Index alignment with ``image_anchors``."""
        out: list[dict[str, Any]] = []
        for b in self.blocks:
            out.extend(b.images)
        return out

    @property
    def image_anchors(self) -> dict[str, list[int]]:
        """``coord_str -> [image_part_index, ...]`` — recomputed from
        ``blocks`` so a block reorder updates the map automatically."""
        anchors: dict[str, list[int]] = {}
        flat_idx = 0
        for b in self.blocks:
            if b.images:
                key = b.coord if b.coord is not None else "__nonstep__"
                anchors.setdefault(key, []).extend(
                    range(flat_idx, flat_idx + len(b.images))
                )
            flat_idx += len(b.images)
        return anchors


# ---------------------------------------------------------------------------
# Common conversation-line helpers
# ---------------------------------------------------------------------------

def fmt_step_flat(idx: int, agent: str, text: str) -> str:
    """``step 3: agent_name: <text>`` (1-indexed flat)."""
    return f"step {idx}: {agent}: {text.rstrip()}"


def fmt_step_hier(round_: int, position: int, agent: str, text: str) -> str:
    """``step 1.2: agent_name: <text>``."""
    return f"step {round_}.{position}: {agent}: {text.rstrip()}"


def coord_str_flat(idx: int) -> str:
    return str(idx)


def coord_str_hier(round_: int, position: int) -> str:
    return f"{round_}.{position}"


# ---------------------------------------------------------------------------
# Task-image extractor (shared)
# ---------------------------------------------------------------------------

def task_image_parts(
    release: dict,
    *,
    max_dim: Optional[int] = None,
    jpeg_quality: Optional[int] = None,
) -> list[dict[str, Any]]:
    """Pull the original task image(s) from ``release["task"]["images"]``.

    Returns an empty list for text-only modalities. ``max_dim`` /
    ``jpeg_quality`` flow through to ``pil_image_part`` so renderers can
    cap task-image size the same way they cap step-level images.
    """
    imgs = (release.get("task") or {}).get("images") or []
    return [
        pil_image_part(i, max_dim=max_dim, jpeg_quality=jpeg_quality)
        for i in imgs
        if isinstance(i, dict) and i.get("data")
    ]
