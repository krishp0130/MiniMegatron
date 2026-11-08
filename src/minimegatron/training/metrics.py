"""Metric logging: one JSON object per line, plus a readable stdout line."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class JSONLLogger:
    """Append-only JSONL writer.

    JSONL over a binary event format because it is trivially greppable, tailable while a run is
    in flight, and parseable by anything; over CSV because records can gain fields between
    milestones without breaking old logs.
    """

    def __init__(self, path: str | Path, echo: bool = True) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.echo = echo
        self._file = self.path.open("a", encoding="utf-8")

    def log(self, record: dict[str, Any]) -> None:
        self._file.write(json.dumps(record, sort_keys=True) + "\n")
        self._file.flush()  # a killed run must not lose the lines it already produced
        if self.echo:
            print(format_record(record), flush=True)

    def close(self) -> None:
        if not self._file.closed:
            self._file.close()

    def __enter__(self) -> JSONLLogger:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def format_record(record: dict[str, Any]) -> str:
    """Render a record as a compact fixed-width line, for watching a run live."""
    parts = []
    for key, fmt in (
        ("step", "{:>6}"),
        ("loss", "{:>8.4f}"),
        ("val_loss", "{:>8.4f}"),
        ("lr", "{:>10.3e}"),
        ("grad_norm", "{:>8.3f}"),
        ("tokens_seen", "{:>12,}"),
        ("tokens_per_sec", "{:>10.1f}"),
    ):
        if key in record and record[key] is not None:
            parts.append(f"{key} " + fmt.format(record[key]))
    return " | ".join(parts)
