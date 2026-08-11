"""Atomic JSONL append store for eval results.

Layout::

    results/
    └── <model_safe>/
        └── <benchmark>.jsonl

Each line is one trace's evaluation record (raw output, parsed prediction,
token usage, ground truth). The store provides two operations:

- ``done_trace_ids()`` — set of ``trace_id`` strings already on disk, so
  the runner can skip them on resume.
- ``append(record)`` — write one JSON-serialised line, guarded by
  ``fcntl.LOCK_EX`` so concurrent workers (asyncio + threadpool, or
  multiple processes) can't tear a line in half. Linux flock is per-FD,
  so we open-lock-write-close on every append; the cost is one extra
  open per record but multi-process safety is free.

Model names get sanitised for the filesystem because many LiteLLM ids
contain ``/`` (``gemini/gemini-3-flash-preview``).
"""
from __future__ import annotations

import fcntl
import json
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


def model_to_dirname(model: str) -> str:
    """Map a model id to a filesystem-safe directory name.

    Replaces every char outside ``[A-Za-z0-9._-]`` with ``_``. LiteLLM
    provider prefixes (``gemini/gemini-3-flash-preview``) become
    underscores so each model gets its own results subdirectory.
    """
    return re.sub(r"[^A-Za-z0-9._-]", "_", model)


@dataclass
class ResultsStore:
    """File-locked JSONL append store for one ``(model, benchmark)`` cell.

    Construct one per cell; the path is created lazily on first ``append``
    so a dry run (no records) doesn't litter empty files.
    """
    path: Path
    # In-process lock so multiple coroutines from one process don't race
    # the open() before fcntl gets a chance. Cheap and orthogonal to the
    # cross-process flock.
    _lock: threading.Lock = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self._lock = threading.Lock()

    def _open_append(self):
        """Open the JSONL for appending, creating the model directory on
        first write only — so a dry run leaves no trace on disk."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        return self.path.open("a", encoding="utf-8")

    @classmethod
    def for_cell(
        cls,
        results_root: Path,
        model: str,
        benchmark: str,
    ) -> "ResultsStore":
        """Resolve the JSONL path for a result cell."""
        path = results_root / model_to_dirname(model) / f"{benchmark}.jsonl"
        return cls(path=path)

    def done_trace_ids(self) -> set[str]:
        """Return the set of ``trace_id`` strings already on disk.

        Tolerates partial writes / corrupt lines (e.g. truncated final
        line from a previous crash) — those lines are skipped silently.
        Reading is unlocked: if a writer is mid-append, fcntl ensures
        the line we read is whole-or-not-yet-written, never half.
        """
        if not self.path.exists():
            return set()
        ids: set[str] = set()
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                tid = rec.get("trace_id")
                if isinstance(tid, str):
                    ids.add(tid)
        return ids

    def append(self, record: dict[str, Any]) -> None:
        """Append one record as a single JSON line, atomically.

        ``LOCK_EX`` blocks competing writers until the line is flushed;
        the in-process ``_lock`` short-circuits the case where many
        coroutines from the same process want to write at once
        (asyncio + threadpool would otherwise contend on the OS lock
        unnecessarily).
        """
        line = json.dumps(record, ensure_ascii=False, default=_json_default) + "\n"
        with self._lock:
            with self._open_append() as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    f.write(line)
                    f.flush()
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    def extend(self, records: Iterable[dict[str, Any]]) -> int:
        """Bulk-append. Acquires the OS lock once for the whole batch."""
        encoded = [
            json.dumps(r, ensure_ascii=False, default=_json_default) + "\n"
            for r in records
        ]
        if not encoded:
            return 0
        with self._lock:
            with self._open_append() as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    f.writelines(encoded)
                    f.flush()
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        return len(encoded)

    def count(self) -> int:
        if not self.path.exists():
            return 0
        with self.path.open("r", encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())


def _json_default(obj: Any) -> Any:
    """Fallback encoder for non-JSON-native objects we routinely shove
    into records — e.g. ``Path`` from trace_path, ``datetime`` from
    timestamps, dataclasses for parsed predictions.
    """
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    if hasattr(obj, "__dict__"):
        return {k: v for k, v in obj.__dict__.items() if not k.startswith("_")}
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")
