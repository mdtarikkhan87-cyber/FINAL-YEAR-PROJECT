"""Bounded in-memory store for recent predictions and their SHAP values.

Deliberately simple: an ordered dict with a size cap, evicting oldest-first. It
lives in the process, so it empties on restart and is not shared across workers
-- run the API with a single worker or swap this for Redis before that matters.
Storing explanations is what makes ``GET /explain/{prediction_id}`` possible at
all: SHAP needs the exact feature row that produced the prediction, and
recomputing it from a re-parsed request would risk explaining something subtly
different from what was returned.
"""

from __future__ import annotations

import threading
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

DEFAULT_CAPACITY = 500


@dataclass
class StoredPrediction:
    prediction_id: str
    task: str                      # "regression" | "classification"
    created_at: str
    prediction: float
    explained_class: str | None
    base_value: float
    units: str
    contributions: list[dict[str, Any]]
    reconstruction: float
    request: dict[str, Any] = field(default_factory=dict)


class PredictionStore:
    """Thread-safe, size-capped, oldest-first eviction."""

    def __init__(self, capacity: int = DEFAULT_CAPACITY) -> None:
        self._items: OrderedDict[str, StoredPrediction] = OrderedDict()
        self._capacity = capacity
        self._lock = threading.Lock()

    @property
    def capacity(self) -> int:
        return self._capacity

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)

    def new_id(self) -> str:
        return str(uuid.uuid4())

    def put(self, record: StoredPrediction) -> None:
        with self._lock:
            self._items[record.prediction_id] = record
            self._items.move_to_end(record.prediction_id)
            while len(self._items) > self._capacity:
                self._items.popitem(last=False)

    def get(self, prediction_id: str) -> StoredPrediction | None:
        with self._lock:
            return self._items.get(prediction_id)

    def clear(self) -> None:
        with self._lock:
            self._items.clear()


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
