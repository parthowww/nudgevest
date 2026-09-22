"""Synthetic XGBoost propensity model and shared acceptance ground truth."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from bandit import NUDGE_TYPES, amount_bucket
from transactions import CATEGORIES


FEATURE_COLUMNS = [
    *(f"category_{category}" for category in CATEGORIES),
    "amount_low",
    "amount_medium",
    "amount_high",
    *(f"day_{day}" for day in range(7)),
    *(f"arm_{index}" for index in range(len(NUDGE_TYPES))),
]


def ground_truth_acceptance_probability(
    context: dict[str, Any], nudge_type: str
) -> float:
    """Hand-coded behavior model used for both training and live rewards."""
    amount = float(context["amount"])
    category = str(context["category"])
    day_of_week = int(context["day_of_week"])

    if nudge_type == "No nudge":
        return 0.0

    base_rates = {
        "Round-up-and-invest": 0.53,
        "Weekly-savings-invest": 0.44,
        "Skip-this-purchase-invest": 0.35,
        "Streak-bonus-invest": 0.41,
    }
    probability = base_rates[nudge_type]

    if nudge_type == "Round-up-and-invest":
        probability += 0.15 if amount < 8 else -0.08 if amount >= 25 else 0
        probability += 0.04 if category == "coffee" else 0
    elif nudge_type == "Weekly-savings-invest":
        probability += 0.14 if day_of_week >= 5 else 0
        probability += 0.05 if category in {"food", "entertainment"} else 0
    elif nudge_type == "Skip-this-purchase-invest":
        probability += 0.09 if category in {"shopping", "entertainment"} else 0
        probability -= 0.10 if amount >= 50 else 0
    elif nudge_type == "Streak-bonus-invest":
        probability += 0.11 if day_of_week in {0, 4} else 0
        probability += 0.04 if category == "transport" else 0

    return float(np.clip(probability, 0.02, 0.9))


def sample_acceptance(
    context: dict[str, Any], nudge_type: str, rng: np.random.Generator
) -> int:
    """Sample a noisy observed reward from the same behavior function."""
    probability = ground_truth_acceptance_probability(context, nudge_type)
    noisy_probability = float(np.clip(probability + rng.normal(0, 0.035), 0, 1))
    return int(rng.random() < noisy_probability)


def _feature_row(context: dict[str, Any], nudge_type: str) -> dict[str, float]:
    row = {column: 0.0 for column in FEATURE_COLUMNS}
    row[f"category_{context['category']}"] = 1.0
    row[f"amount_{amount_bucket(float(context['amount']))}"] = 1.0
    row[f"day_{int(context['day_of_week'])}"] = 1.0
    row[f"arm_{NUDGE_TYPES.index(nudge_type)}"] = 1.0
    return row


class PropensityModel:
    """XGBoost classifier trained once on synthetic contextual interactions."""

    def __init__(self, model: XGBClassifier) -> None:
        self.model = model

    @classmethod
    def train(cls, rows: int = 2000, seed: int = 2026) -> "PropensityModel":
        rng = np.random.default_rng(seed)
        records: list[dict[str, float]] = []
        labels: list[int] = []

        for _ in range(rows):
            category = str(rng.choice(CATEGORIES))
            amount_low, amount_high = {
                "coffee": (2.5, 8.5),
                "food": (8, 35),
                "transport": (2, 22),
                "shopping": (12, 120),
                "entertainment": (8, 65),
            }[category]
            context = {
                "category": category,
                "amount": float(rng.uniform(amount_low, amount_high)),
                "day_of_week": int(rng.integers(0, 7)),
            }
            nudge_type = str(rng.choice(NUDGE_TYPES))
            records.append(_feature_row(context, nudge_type))
            labels.append(
                int(
                    rng.random()
                    < ground_truth_acceptance_probability(context, nudge_type)
                )
            )

        training_frame = pd.DataFrame(records, columns=FEATURE_COLUMNS)
        classifier = XGBClassifier(
            n_estimators=90,
            max_depth=3,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=seed,
            n_jobs=1,
        )
        classifier.fit(training_frame, np.array(labels))
        return cls(classifier)

    def predict_propensity(
        self, context: dict[str, Any], nudge_type: str
    ) -> float:
        """Return the model-estimated acceptance probability for an arm."""
        features = pd.DataFrame([_feature_row(context, nudge_type)], columns=FEATURE_COLUMNS)
        return float(np.clip(self.model.predict_proba(features)[0, 1], 0.0, 1.0))

    def all_propensities(self, context: dict[str, Any]) -> dict[str, float]:
        return {
            nudge_type: self.predict_propensity(context, nudge_type)
            for nudge_type in NUDGE_TYPES
        }