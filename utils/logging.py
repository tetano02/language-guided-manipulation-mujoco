"""Structured logging helpers for debugging, CI, and artifact persistence."""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def _timestamp_utc() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    return value


@dataclass
class StructuredLogger:
    """JSONL logger with simple artifact writers."""

    output_dir: Path
    run_name: str = "run"
    enabled: bool = True

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)
        self.run_dir = self.output_dir / f"{self.run_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.events_path = self.run_dir / "events.jsonl"
        if self.enabled:
            self.run_dir.mkdir(parents=True, exist_ok=True)

    def log(self, record: Mapping[str, Any]) -> None:
        if not self.enabled:
            return
        payload = dict(record)
        payload.setdefault("ts_utc", _timestamp_utc())
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(_to_jsonable(payload), ensure_ascii=True) + "\n")

    def log_event(
        self,
        event: str,
        payload: Mapping[str, Any] | None = None,
        *,
        step: int | None = None,
    ) -> None:
        body: dict[str, Any] = {"type": "event", "event": event}
        if step is not None:
            body["step"] = step
        if payload is not None:
            body["payload"] = payload
        self.log(body)

    def log_step(self, step: int, observation: Mapping[str, Any], info: Mapping[str, Any]) -> None:
        self.log(
            {
                "type": "step",
                "step": step,
                "observation": observation,
                "info": info,
            }
        )

    def save_json(self, filename: str, payload: Any) -> Path:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        path = self.run_dir / filename
        with path.open("w", encoding="utf-8") as handle:
            json.dump(_to_jsonable(payload), handle, indent=2, ensure_ascii=True)
        return path

    def save_pickle(self, filename: str, payload: Any) -> Path:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        path = self.run_dir / filename
        with path.open("wb") as handle:
            pickle.dump(payload, handle)
        return path

    def close(self) -> None:
        # No buffered state currently, kept for API symmetry.
        return

