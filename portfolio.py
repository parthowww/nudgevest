"""Small simulated portfolio used by the NudgeVest demo."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def investment_amount(amount: float, nudge_type: str) -> float:
    """Map each accepted nudge to the simulated contribution it creates."""
    if nudge_type == "Round-up-and-invest":
        round_up = np.ceil(amount) - amount
        return round(float(max(0.05, round_up)), 2)
    if nudge_type == "Weekly-savings-invest":
        return round(float(min(5.0, max(0.5, amount * 0.02))), 2)
    if nudge_type == "Skip-this-purchase-invest":
        return round(float(min(8.0, max(0.5, amount * 0.05))), 2)
    if nudge_type == "Streak-bonus-invest":
        return 0.50
    return 0.0


@dataclass
class Portfolio:
    balance: float = 0.0
    total_contributed: float = 0.0
    total_return: float = 0.0
    history: list[float] = field(default_factory=list)

    def tick(
        self,
        rng: np.random.Generator,
        contribution: float = 0.0,
    ) -> float:
        """Apply a daily return, then add the contribution from this tick."""
        starting_balance = self.balance
        daily_return = float(rng.normal(0.0003, 0.0015))
        self.balance *= 1 + daily_return
        self.total_return += self.balance - starting_balance
        if contribution > 0:
            self.balance += contribution
            self.total_contributed += contribution
        self.balance = round(max(0.0, self.balance), 2)
        self.history.append(self.balance)
        return self.balance