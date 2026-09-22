"""
propensity_model.py
Behavioral propensity model for NudgeVest using XGBoost.

Trains an XGBoost binary classifier on synthetic interaction data generated from a
ground-truth behavioral acceptance function (e.g., round-ups excel on small purchases,
weekly savings surge on weekends). Exposes predict_propensity(context, nudge_type)
to guide the hybrid contextual bandit.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from typing import Dict, Any, Tuple, Optional
from transactions import CATEGORIES, get_amount_bucket

# 5 Arms (Nudge Types)
ARM_NAMES = [
    "Round-up-and-invest",
    "Weekly-savings-invest",
    "Skip-this-purchase-invest",
    "Streak-bonus-invest",
    "No nudge (control)",
]
NUM_ARMS = len(ARM_NAMES)

CATEGORY_MAP = {cat: i for i, cat in enumerate(CATEGORIES)}
BUCKET_MAP = {"low": 0, "med": 1, "high": 2}


def ground_truth_acceptance_prob(context: Dict[str, Any], nudge_type: int) -> float:
    """
    Hand-coded behavioral ground-truth acceptance probability function.
    
    Behavioral heuristics:
      - Arm 0 (Round-up): Highly accepted for small, frequent purchases (coffee/transit).
      - Arm 1 (Weekly-savings): Peaks on weekends (Fri-Sun) during weekly budgeting.
      - Arm 2 (Skip-purchase): Higher for discretionary shopping/entertainment impulse buys.
      - Arm 3 (Streak-bonus): Peaks during daily morning/routine hours for coffee/food.
      - Arm 4 (No nudge): Control arm with 0.0 acceptance.
    """
    if nudge_type == 4:
        return 0.0  # Control arm: no nudge displayed, zero prompt acceptance

    category = context.get("category", "coffee")
    amount = float(context.get("amount", 10.0))
    amount_bucket = context.get("amount_bucket", get_amount_bucket(amount))
    day_of_week = int(context.get("day_of_week", 0))
    hour = int(context.get("hour", 12))

    p = 0.25  # default base

    if nudge_type == 0:
        # Arm 0: Round-up-and-invest
        p = 0.42
        if amount_bucket == "low" or amount < 15.0:
            p += 0.36  # Micro-pennies feel effortless on low-value items
        elif amount_bucket == "med":
            p += 0.08
        else:
            p -= 0.24  # High tickets make spare change feel irrelevant
        if category in ["coffee", "transport"]:
            p += 0.12

    elif nudge_type == 1:
        # Arm 1: Weekly-savings-invest
        p = 0.28
        # Weekend mindset boost (Fri evening through Sunday)
        is_weekend = (day_of_week in [5, 6]) or (day_of_week == 4 and hour >= 17)
        if is_weekend:
            p += 0.38  # Payday/weekend goal setting
        else:
            p -= 0.10
        if category in ["food", "shopping"]:
            p += 0.08

    elif nudge_type == 2:
        # Arm 2: Skip-this-purchase-invest
        p = 0.16
        # Impulse reflection works best for discretionary shopping/entertainment
        if category in ["shopping", "entertainment"]:
            p += 0.34
            if 25.0 <= amount <= 120.0:
                p += 0.18
        elif category in ["transport", "coffee"]:
            p -= 0.18  # Cannot skip mandatory daily transit or morning routine

    elif nudge_type == 3:
        # Arm 3: Streak-bonus-invest
        p = 0.32
        # Daily habit routine boosts (morning 7-11 or evening 17-20)
        if 7 <= hour <= 11 or 17 <= hour <= 20:
            p += 0.22
        if category in ["coffee", "food"]:
            p += 0.14
        if day_of_week in [5, 6]:
            p -= 0.06

    # Clip to realistic probability range
    return float(np.clip(p, 0.03, 0.95))


def simulate_user_response(context: Dict[str, Any], nudge_type: int, noise_scale: float = 0.05) -> Tuple[int, float]:
    """
    Simulates real user interaction with behavioral noise.
    Returns (accepted: 0 or 1, true_probability: float).
    """
    if nudge_type == 4:
        return 0, 0.0

    base_p = ground_truth_acceptance_prob(context, nudge_type)
    # Add minor behavioral variance / noise
    noisy_p = float(np.clip(base_p + np.random.normal(0, noise_scale), 0.01, 0.99))
    accepted = 1 if np.random.rand() < noisy_p else 0
    return accepted, base_p


def extract_features(context: Dict[str, Any], nudge_type: int) -> np.ndarray:
    """Encodes context and nudge_type into a numerical feature vector for XGBoost."""
    category = context.get("category", "coffee")
    cat_idx = CATEGORY_MAP.get(category, 0)
    
    amount = float(context.get("amount", 10.0))
    bucket = context.get("amount_bucket", get_amount_bucket(amount))
    bucket_idx = BUCKET_MAP.get(bucket, 0)
    
    day_of_week = int(context.get("day_of_week", 0))
    hour = int(context.get("hour", 12))
    is_weekend = 1.0 if day_of_week in [5, 6] else 0.0

    # One-hot encode category (5 cols)
    cat_one_hot = [1.0 if cat_idx == i else 0.0 for i in range(5)]
    
    # One-hot encode nudge_type (5 cols)
    nudge_one_hot = [1.0 if nudge_type == i else 0.0 for i in range(NUM_ARMS)]

    # Feature vector: [amount, bucket_idx, day_of_week, hour, is_weekend] + cat_one_hot + nudge_one_hot
    features = [
        amount,
        float(bucket_idx),
        float(day_of_week),
        float(hour),
        is_weekend,
    ] + cat_one_hot + nudge_one_hot

    return np.array(features, dtype=np.float32)


FEATURE_NAMES = [
    "amount", "bucket_idx", "day_of_week", "hour", "is_weekend",
    "cat_coffee", "cat_food", "cat_transport", "cat_shopping", "cat_entertainment",
    "nudge_roundup", "nudge_weekly", "nudge_skip", "nudge_streak", "nudge_control"
]


class PropensityModel:
    """
    XGBoost propensity scoring model for NudgeVest.
    Trained on synthetic offline transaction records to estimate P(accept | context, nudge_type).
    """

    def __init__(self):
        self.model: Optional[xgb.XGBClassifier] = None
        self.is_trained: bool = False

    def generate_training_data(self, n_samples: int = 2500, random_seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
        """Generates synthetic training dataset using the ground-truth behavioral logic."""
        np.random.seed(random_seed)
        X = []
        y = []

        for _ in range(n_samples):
            cat = np.random.choice(CATEGORIES)
            if cat == "coffee":
                amt = np.random.uniform(2.75, 9.50)
            elif cat == "food":
                amt = np.random.uniform(8.00, 52.00)
            elif cat == "transport":
                amt = np.random.uniform(2.75, 38.00)
            elif cat == "shopping":
                amt = np.random.uniform(18.00, 175.00)
            else:
                amt = np.random.uniform(12.00, 95.00)
            
            amt = round(amt, 2)
            day = int(np.random.randint(0, 7))
            hour = int(np.random.randint(6, 24))
            nudge = int(np.random.randint(0, NUM_ARMS))

            ctx = {
                "category": cat,
                "amount": amt,
                "amount_bucket": get_amount_bucket(amt),
                "day_of_week": day,
                "hour": hour,
            }

            features = extract_features(ctx, nudge)
            accepted, _ = simulate_user_response(ctx, nudge, noise_scale=0.06)

            X.append(features)
            y.append(accepted)

        return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)

    def train(self, n_samples: int = 2500, random_seed: int = 42) -> Dict[str, Any]:
        """Trains the XGBoost classifier on synthetic data."""
        X, y = self.generate_training_data(n_samples=n_samples, random_seed=random_seed)
        
        self.model = xgb.XGBClassifier(
            n_estimators=75,
            max_depth=4,
            learning_rate=0.08,
            subsample=0.85,
            colsample_bytree=0.85,
            eval_metric="logloss",
            random_state=random_seed,
            n_jobs=1,
        )
        self.model.fit(X, y)
        self.is_trained = True

        preds = self.model.predict_proba(X)[:, 1]
        acc = float(np.mean((preds >= 0.5) == y))

        return {
            "n_samples": n_samples,
            "train_accuracy": round(acc, 4),
            "positive_rate": round(float(np.mean(y)), 4),
        }

    def predict_propensity(self, context: Dict[str, Any], nudge_type: int) -> float:
        """
        Estimates the probability that the user will accept the given nudge type.
        Returns a float in [0.0, 1.0].
        """
        if nudge_type == 4:
            return 0.001  # Control arm propensity is near zero

        if not self.is_trained or self.model is None:
            # Fallback to ground truth heuristic if not yet fitted
            return ground_truth_acceptance_prob(context, nudge_type)

        feats = extract_features(context, nudge_type).reshape(1, -1)
        # Return predicted class 1 probability
        prob = float(self.model.predict_proba(feats)[0, 1])
        return float(np.clip(prob, 0.01, 0.99))

    def predict_all_arms(self, context: Dict[str, Any]) -> np.ndarray:
        """Computes propensity score vector across all 5 arms for a given context."""
        scores = [self.predict_propensity(context, arm_idx) for arm_idx in range(NUM_ARMS)]
        return np.array(scores, dtype=np.float32)
