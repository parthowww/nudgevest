"""Synthetic transaction generation for the NudgeVest simulation."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np


CATEGORIES = ("coffee", "food", "transport", "shopping", "entertainment")
CATEGORY_RANGES = {
    "coffee": (2.50, 8.50),
    "food": (8.00, 35.00),
    "transport": (2.00, 22.00),
    "shopping": (12.00, 120.00),
    "entertainment": (8.00, 65.00),
}


def generate_transaction(
    rng: np.random.Generator, timestamp: datetime | None = None
) -> dict[str, Any]:
    """Return one realistic-looking synthetic transaction with context."""
    timestamp = timestamp or datetime.now()
    category = str(rng.choice(CATEGORIES))
    low, high = CATEGORY_RANGES[category]

    return {
        "category": category,
        "amount": round(float(rng.uniform(low, high)), 2),
        "day_of_week": timestamp.weekday(),
        "day_name": timestamp.strftime("%a"),
        "hour": timestamp.hour,
        "timestamp": timestamp,
    }