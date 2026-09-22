"""A contextual Thompson Sampling bandit for nudge selection."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np


NUDGE_TYPES = (
    "Round-up-and-invest",
    "Weekly-savings-invest",
    "Skip-this-purchase-invest",
    "Streak-bonus-invest",
    "No nudge",
)


def amount_bucket(amount: float) -> str:
    """Bucket an amount so the posterior stays interpretable."""
    if amount < 8:
        return "low"
    if amount < 25:
        return "medium"
    return "high"


def context_key(context: dict[str, Any]) -> str:
    """Discretize the context into the posterior lookup bucket."""
    return "|".join(
        (
            str(context["category"]),
            amount_bucket(float(context["amount"])),
            str(int(context["day_of_week"])),
        )
    )


class ContextualThompsonSamplingBandit:
    """Per-context Beta posteriors with propensity-weighted arm selection."""

    def __init__(self, seed: int = 2026) -> None:
        self.rng = np.random.default_rng(seed)
        self.alpha: dict[str, np.ndarray] = defaultdict(
            lambda: np.ones(len(NUDGE_TYPES), dtype=float)
        )
        self.beta: dict[str, np.ndarray] = defaultdict(
            lambda: np.ones(len(NUDGE_TYPES), dtype=float)
        )

    def _ensure_context(self, context: dict[str, Any]) -> str:
        key = context_key(context)
        _ = self.alpha[key]
        _ = self.beta[key]
        return key

    def select(
        self, context: dict[str, Any], propensities: dict[str, float] | None = None
    ) -> str:
        """Sample a posterior for every arm and return the highest score."""
        key = self._ensure_context(context)
        samples = self.rng.beta(self.alpha[key], self.beta[key])

        if propensities:
            propensity_vector = np.array(
                [max(0.01, float(propensities[arm])) for arm in NUDGE_TYPES]
            )
            # Hybrid mechanism: contextual Thompson samples are weighted by the
            # XGBoost propensity score before choosing the highest-scoring arm.
            samples = samples * propensity_vector

        return NUDGE_TYPES[int(np.argmax(samples))]

    def update(self, context: dict[str, Any], arm: str, reward: int) -> None:
        """Update the selected context-arm posterior from an observed reward."""
        if arm not in NUDGE_TYPES:
            raise ValueError(f"Unknown nudge type: {arm}")
        key = self._ensure_context(context)
        arm_index = NUDGE_TYPES.index(arm)
        if int(reward):
            self.alpha[key][arm_index] += 1
        else:
            self.beta[key][arm_index] += 1

    def arm_stats(self) -> list[dict[str, float | str]]:
        """Aggregate posterior statistics across all visited contexts."""
        alpha_totals = np.ones(len(NUDGE_TYPES), dtype=float)
        beta_totals = np.ones(len(NUDGE_TYPES), dtype=float)
        for values in self.alpha.values():
            alpha_totals += values - 1
        for values in self.beta.values():
            beta_totals += values - 1

        return [
            {
                "Nudge type": arm,
                "Alpha": round(float(alpha_totals[index]), 1),
                "Beta": round(float(beta_totals[index]), 1),
                "Posterior mean": round(
                    float(
                        alpha_totals[index]
                        / (alpha_totals[index] + beta_totals[index])
                    ),
                    3,
                ),
                "Observations": round(
                    float(alpha_totals[index] + beta_totals[index] - 2), 0
                ),
            }
            for index, arm in enumerate(NUDGE_TYPES)
        ]