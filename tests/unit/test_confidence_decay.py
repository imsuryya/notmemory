# tests/unit/test_confidence_decay.py
"""Tests for confidence decay module."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

from notmemory.memory.decay import (
    decay_score_entries,
    decayed_confidence,
    filter_fresh,
    is_stale,
)
from notmemory.memory.models import MemoryEntry


def _entry(
    confidence: float = 1.0,
    age_days: float = 0.0,
) -> MemoryEntry:
    """Create a MemoryEntry with a specific age and confidence."""
    now = datetime.now(UTC)
    timestamp = now - timedelta(days=age_days)
    return MemoryEntry(
        id=str(uuid.uuid4()),
        bank_id="test",
        content={"data": "test"},
        transaction_id=str(uuid.uuid4()),
        timestamp=timestamp,
        confidence=confidence,
    )


def _now() -> datetime:
    return datetime.now(UTC)


# ── decayed_confidence ────────────────────────────────────────────────────────


def test_fresh_entry_confidence_unchanged():
    """Entry written now should have ~original confidence."""
    entry = _entry(confidence=1.0, age_days=0)
    result = decayed_confidence(entry, half_life_days=30.0, now=_now())
    assert abs(result - 1.0) < 0.01


def test_half_life_halves_confidence():
    """After exactly 30 days, confidence should be c₀/2."""
    entry = _entry(confidence=1.0, age_days=30.0)
    result = decayed_confidence(entry, half_life_days=30.0, now=_now())
    assert abs(result - 0.5) < 0.01


def test_double_half_life_quarters_confidence():
    """After 60 days, confidence should be c₀/4."""
    entry = _entry(confidence=1.0, age_days=60.0)
    result = decayed_confidence(entry, half_life_days=30.0, now=_now())
    assert abs(result - 0.25) < 0.01


def test_triple_half_life():
    """After 90 days, confidence should be c₀/8."""
    entry = _entry(confidence=1.0, age_days=90.0)
    result = decayed_confidence(entry, half_life_days=30.0, now=_now())
    assert abs(result - 0.125) < 0.01


def test_custom_initial_confidence():
    """Starting confidence of 0.8 should decay proportionally."""
    entry = _entry(confidence=0.8, age_days=30.0)
    result = decayed_confidence(entry, half_life_days=30.0, now=_now())
    assert abs(result - 0.4) < 0.01


def test_custom_half_life():
    """7-day half-life should halve confidence after 7 days."""
    entry = _entry(confidence=1.0, age_days=7.0)
    result = decayed_confidence(entry, half_life_days=7.0, now=_now())
    assert abs(result - 0.5) < 0.01


def test_result_bounded_between_zero_and_one():
    entry = _entry(confidence=1.0, age_days=365.0)
    result = decayed_confidence(entry, half_life_days=30.0, now=_now())
    assert 0.0 <= result <= 1.0


def test_naive_timestamp_treated_as_utc():
    """Entries with naive timestamps should not raise."""
    entry = _entry(confidence=1.0, age_days=0)
    entry = entry.model_copy(update={"timestamp": entry.timestamp.replace(tzinfo=None)})
    result = decayed_confidence(entry, half_life_days=30.0)
    assert 0.0 <= result <= 1.0


# ── is_stale ──────────────────────────────────────────────────────────────────


def test_fresh_entry_not_stale():
    entry = _entry(confidence=1.0, age_days=0)
    assert not is_stale(entry, half_life_days=30.0, deprecation_threshold=0.05)


def test_very_old_entry_is_stale():
    """After many half-lives, entry should be stale."""
    entry = _entry(confidence=1.0, age_days=300.0)
    assert is_stale(entry, half_life_days=30.0, deprecation_threshold=0.05)


def test_stale_threshold_respected():
    """Entry just below threshold should be stale."""
    entry = _entry(confidence=0.04, age_days=0)
    assert is_stale(
        entry,
        half_life_days=30.0,
        deprecation_threshold=0.05,
        now=_now(),
    )


# ── filter_fresh ──────────────────────────────────────────────────────────────


def test_filter_fresh_removes_stale():
    fresh = _entry(confidence=1.0, age_days=0)
    stale = _entry(confidence=1.0, age_days=300)
    result = filter_fresh(
        [fresh, stale],
        half_life_days=30.0,
        deprecation_threshold=0.05,
        now=_now(),
    )
    assert len(result) == 1
    assert result[0].id == fresh.id


def test_filter_fresh_sorted_by_confidence_desc():
    high = _entry(confidence=1.0, age_days=0)
    low = _entry(confidence=1.0, age_days=20)
    result = filter_fresh(
        [low, high],
        half_life_days=30.0,
        deprecation_threshold=0.05,
        now=_now(),
    )
    assert result[0].id == high.id


def test_filter_fresh_empty_list():
    assert filter_fresh([]) == []


# ── decay_score_entries ───────────────────────────────────────────────────────


def test_decay_score_returns_tuples():
    entries = [_entry(age_days=0), _entry(age_days=30)]
    scored = decay_score_entries(entries, half_life_days=30.0, now=_now())
    assert len(scored) == 2
    for _e, score in scored:
        assert 0.0 <= score <= 1.0


def test_decay_score_sorted_desc():
    fresh = _entry(confidence=1.0, age_days=0)
    old = _entry(confidence=1.0, age_days=60)
    scored = decay_score_entries([old, fresh], half_life_days=30.0, now=_now())
    assert scored[0][0].id == fresh.id
    assert scored[0][1] > scored[1][1]
