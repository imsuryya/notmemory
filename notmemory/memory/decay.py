# notmemory/memory/decay.py
"""
Confidence decay for notmemory.

Implements exponential half-life decay:

    c(t) = c₀ · 2^(-t / half_life_days)

Where:
  c₀             = original confidence at write time
  t               = age of the entry in days
  half_life_days  = days until confidence halves (default 30)

A memory with confidence 1.0 will decay to:
  0.5  after 30 days
  0.25 after 60 days
  0.125 after 90 days

Entries below the deprecation_threshold are considered stale.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from notmemory.memory.models import MemoryEntry


def decayed_confidence(
    entry: MemoryEntry,
    *,
    half_life_days: float = 30.0,
    now: datetime | None = None,
) -> float:
    """
    Return the current decayed confidence for a memory entry.

    Args:
        entry:          The MemoryEntry to evaluate.
        half_life_days: Days until confidence halves (default 30).
        now:            Reference time (default: UTC now). Pass explicitly
                        in tests for deterministic results.

    Returns:
        Float in [0.0, 1.0] — decayed confidence value.
    """
    if now is None:
        now = datetime.now(UTC)

    entry_time = entry.timestamp
    if entry_time.tzinfo is None:
        entry_time = entry_time.replace(tzinfo=UTC)

    age_days = (now - entry_time).total_seconds() / 86_400.0
    age_days = max(0.0, age_days)

    return entry.confidence * math.pow(2.0, -age_days / half_life_days)


def is_stale(
    entry: MemoryEntry,
    *,
    half_life_days: float = 30.0,
    deprecation_threshold: float = 0.05,
    now: datetime | None = None,
) -> bool:
    """
    Return True if the entry's decayed confidence is below the threshold.

    Args:
        entry:                  The MemoryEntry to evaluate.
        half_life_days:         Days until confidence halves.
        deprecation_threshold:  Confidence floor (default 0.05).
        now:                    Reference time (default: UTC now).
    """
    return decayed_confidence(entry, half_life_days=half_life_days, now=now) < deprecation_threshold


def filter_fresh(
    entries: list[MemoryEntry],
    *,
    half_life_days: float = 30.0,
    deprecation_threshold: float = 0.05,
    now: datetime | None = None,
) -> list[MemoryEntry]:
    """
    Filter out stale entries, return only fresh ones.

    Entries are sorted by decayed confidence descending.

    Args:
        entries:                List of MemoryEntry objects.
        half_life_days:         Days until confidence halves.
        deprecation_threshold:  Confidence floor below which entries
                                are considered stale.
        now:                    Reference time (default: UTC now).
    """
    if now is None:
        now = datetime.now(UTC)

    fresh = [
        e
        for e in entries
        if not is_stale(
            e,
            half_life_days=half_life_days,
            deprecation_threshold=deprecation_threshold,
            now=now,
        )
    ]
    return sorted(
        fresh,
        key=lambda e: decayed_confidence(e, half_life_days=half_life_days, now=now),
        reverse=True,
    )


def decay_score_entries(
    entries: list[MemoryEntry],
    *,
    half_life_days: float = 30.0,
    now: datetime | None = None,
) -> list[tuple[MemoryEntry, float]]:
    """
    Return entries paired with their current decayed confidence score.

    Sorted by score descending (highest confidence first).

    Args:
        entries:        List of MemoryEntry objects.
        half_life_days: Days until confidence halves.
        now:            Reference time (default: UTC now).

    Returns:
        List of (MemoryEntry, decayed_confidence) tuples.
    """
    if now is None:
        now = datetime.now(UTC)

    scored = [(e, decayed_confidence(e, half_life_days=half_life_days, now=now)) for e in entries]
    return sorted(scored, key=lambda x: x[1], reverse=True)


__all__ = [
    "decayed_confidence",
    "is_stale",
    "filter_fresh",
    "decay_score_entries",
]
