"""
portfolio.py
Simulated micro-investment portfolio tracker for NudgeVest.

Maintains running investment balance, logs micro-deposits upon accepted nudges,
and simulates market returns (e.g., diversified ETF growth with daily drift & volatility).
"""

import numpy as np
from typing import Dict, Any, List, Optional
from propensity_model import ARM_NAMES


def get_nudge_investment_amount(nudge_type: int, context: Dict[str, Any]) -> float:
    """
    Computes the dollar amount micro-invested when a nudge is accepted.
    
      - Arm 0 (Round-up): spare change to next dollar (e.g., $0.25 - $1.00).
      - Arm 1 (Weekly-savings): fixed micro-deposit of $5.00.
      - Arm 2 (Skip-purchase): invest 50% of purchase value (capped $3 - $20).
      - Arm 3 (Streak-bonus): gamified micro-deposit of $2.50.
      - Arm 4 (No nudge): $0.00.
    """
    if nudge_type == 0:
        return float(context.get("round_up_amount", 0.75))
    elif nudge_type == 1:
        return 5.00
    elif nudge_type == 2:
        amt = float(context.get("amount", 10.0))
        return float(round(np.clip(amt * 0.50, 3.00, 20.00), 2))
    elif nudge_type == 3:
        return 2.50
    else:
        return 0.00


class SimulatedPortfolio:
    """
    Tracks portfolio balance, contributions, returns, and historical performance.
    """

    def __init__(self, initial_balance: float = 0.0, mean_daily_return: float = 0.0003, volatility: float = 0.0018):
        """
        initial_balance: Starting portfolio value in USD.
        mean_daily_return: Daily market drift (default 0.03% ~ 8% annualized).
        volatility: Daily standard deviation (default 0.18%).
        """
        self.balance: float = float(initial_balance)
        self.total_contributions: float = float(initial_balance)
        self.total_gains: float = 0.0
        self.mean_daily_return: float = mean_daily_return
        self.volatility: float = volatility
        self.history: List[Dict[str, Any]] = []

    def step(
        self,
        tick: int,
        context: Dict[str, Any],
        nudge_type: int,
        accepted: bool,
    ) -> Dict[str, Any]:
        """
        Executes one simulation tick:
          1. Adds micro-investment if accepted.
          2. Applies randomized market return to existing balance.
          3. Logs state to history.
        """
        invested_amount = 0.0
        if accepted and nudge_type != 4:
            invested_amount = get_nudge_investment_amount(nudge_type, context)
            self.balance += invested_amount
            self.total_contributions += invested_amount

        # Randomized market return (e.g. S&P 500 / Total World ETF daily fluctuations)
        daily_return = float(np.random.normal(self.mean_daily_return, self.volatility))
        return_dollars = self.balance * daily_return
        self.balance = max(0.0, self.balance + return_dollars)
        self.total_gains += return_dollars

        record = {
            "tick": tick,
            "category": context.get("category", ""),
            "amount": context.get("amount", 0.0),
            "merchant": context.get("merchant", ""),
            "day_name": context.get("day_name", ""),
            "hour": context.get("hour", 0),
            "nudge_type": nudge_type,
            "nudge_name": ARM_NAMES[nudge_type],
            "accepted": bool(accepted),
            "invested_amount": round(invested_amount, 2),
            "daily_return_pct": round(daily_return * 100, 3),
            "market_growth": round(return_dollars, 2),
            "portfolio_balance": round(self.balance, 2),
            "total_contributions": round(self.total_contributions, 2),
            "total_gains": round(self.total_gains, 2),
        }
        self.history.append(record)
        return record

    def reset(self, initial_balance: float = 0.0):
        """Resets the portfolio state."""
        self.balance = float(initial_balance)
        self.total_contributions = float(initial_balance)
        self.total_gains = 0.0
        self.history.clear()
