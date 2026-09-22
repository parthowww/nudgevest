"""
bandit.py
Contextual Thompson Sampling multi-armed bandit (hand-rolled, NumPy only).

Implements a contextual Beta-Bernoulli Thompson Sampling bandit across 5 nudge arms.
Discretizes context = [category, amount_bucket, day_of_week] into discrete buckets
with localized Beta(alpha, beta) posteriors. Incorporates a hybrid mechanism
weighting Thompson posterior samples with XGBoost propensity scores.
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional, List
from transactions import CATEGORIES, get_amount_bucket
from propensity_model import ARM_NAMES, NUM_ARMS, CATEGORY_MAP, BUCKET_MAP


class ContextualThompsonBandit:
    """
    Contextual Thompson Sampling Bandit with discretized context space.
    
    Context elements:
      - category: 5 discrete values
      - amount_bucket: 3 discrete values ('low', 'med', 'high')
      - day_of_week: 7 discrete values (0..6)
    Total buckets = 5 * 3 * 7 = 105.
    
    Each bucket maintains an independent Beta(alpha, beta) posterior for all 5 arms.
    """

    def __init__(self, hybrid_weighting: bool = True):
        self.num_arms = NUM_ARMS
        self.arm_names = ARM_NAMES
        self.hybrid_weighting = hybrid_weighting

        # Dimensions: 5 categories, 3 amount buckets, 7 days of week
        self.n_categories = len(CATEGORIES)
        self.n_buckets_amount = len(BUCKET_MAP)
        self.n_days = 7
        self.total_buckets = self.n_categories * self.n_buckets_amount * self.n_days

        # Per-context-bucket Beta parameters: shape (105, 5)
        # Initialized to Beta(1.0, 1.0) uniform prior
        self.alpha_table = np.ones((self.total_buckets, self.num_arms), dtype=np.float64)
        self.beta_table = np.ones((self.total_buckets, self.num_arms), dtype=np.float64)

        # Global aggregate tracking across all interactions
        self.global_pulls = np.zeros(self.num_arms, dtype=np.int64)
        self.global_rewards = np.zeros(self.num_arms, dtype=np.int64)
        self.global_alpha = np.ones(self.num_arms, dtype=np.float64)
        self.global_beta = np.ones(self.num_arms, dtype=np.float64)

    def get_bucket_index(self, context: Dict[str, Any]) -> int:
        """Maps continuous/categorical context dictionary into a discrete integer bucket index in [0, 104]."""
        cat = context.get("category", "coffee")
        cat_idx = CATEGORY_MAP.get(cat, 0)

        amt = float(context.get("amount", 10.0))
        bucket = context.get("amount_bucket", get_amount_bucket(amt))
        bucket_idx = BUCKET_MAP.get(bucket, 0)

        day = int(context.get("day_of_week", 0)) % 7

        # Compact index calculation
        idx = (cat_idx * self.n_buckets_amount * self.n_days) + (bucket_idx * self.n_days) + day
        return int(idx)

    def select_arm(
        self,
        context: Dict[str, Any],
        propensity_model: Optional[Any] = None,
    ) -> Tuple[int, np.ndarray, np.ndarray]:
        """
        Selects an arm using Contextual Thompson Sampling, optionally weighted by propensity score.
        
        Returns:
          - chosen_arm: int in [0, 4]
          - raw_samples: np.ndarray of shape (5,) drawn from Beta posteriors
          - final_scores: np.ndarray of shape (5,) after hybrid weighting
        """
        bucket_idx = self.get_bucket_index(context)
        alphas = self.alpha_table[bucket_idx]
        betas = self.beta_table[bucket_idx]

        # Sample from each arm's Beta posterior
        raw_samples = np.random.beta(alphas, betas)

        # -------------------------------------------------------------------------
        # HYBRID MECHANISM:
        # Weight the Thompson posterior sample by the XGBoost propensity score.
        # This blends offline statistical pre-training with live online Bayesian learning.
        # final_score[a] = theta_a * propensity_score[a]
        # -------------------------------------------------------------------------
        if self.hybrid_weighting and propensity_model is not None:
            propensities = propensity_model.predict_all_arms(context)
            final_scores = raw_samples * propensities
        else:
            final_scores = raw_samples.copy()

        # Handle control arm: control arm has 0 propensity in hybrid mode,
        # but can still occasionally be chosen if non-hybrid or slight exploration.
        # Break any exact ties randomly:
        max_val = np.max(final_scores)
        best_arms = np.where(final_scores == max_val)[0]
        chosen_arm = int(np.random.choice(best_arms))

        return chosen_arm, raw_samples, final_scores

    def update(self, context: Dict[str, Any], arm: int, reward: int):
        """
        Updates the Beta posterior for the selected arm under the observed context bucket.
        reward: 1 (accepted) or 0 (declined).
        """
        bucket_idx = self.get_bucket_index(context)
        
        # Update localized bucket posterior
        if reward == 1:
            self.alpha_table[bucket_idx, arm] += 1.0
            self.global_alpha[arm] += 1.0
        else:
            self.beta_table[bucket_idx, arm] += 1.0
            self.global_beta[arm] += 1.0

        self.global_pulls[arm] += 1
        self.global_rewards[arm] += reward

    def get_arm_stats(self) -> List[Dict[str, Any]]:
        """Returns summary statistics for each arm across all contexts for UI display."""
        stats = []
        for arm_idx in range(self.num_arms):
            pulls = int(self.global_pulls[arm_idx])
            rewards = int(self.global_rewards[arm_idx])
            alpha = float(self.global_alpha[arm_idx])
            beta = float(self.global_beta[arm_idx])
            
            # Posterior mean E[theta] = alpha / (alpha + beta)
            post_mean = alpha / (alpha + beta)
            emp_rate = (rewards / pulls) if pulls > 0 else 0.0

            stats.append({
                "arm_index": arm_idx,
                "arm_name": self.arm_names[arm_idx],
                "pulls": pulls,
                "acceptances": rewards,
                "empirical_rate": emp_rate,
                "global_alpha": alpha,
                "global_beta": beta,
                "posterior_mean": post_mean,
            })
        return stats


class RandomArmBaseline:
    """
    Baseline agent that selects nudge arms uniformly at random.
    Runs in parallel to benchmark bandit performance and compute cumulative regret.
    """

    def __init__(self):
        self.num_arms = NUM_ARMS
        self.total_pulls = np.zeros(self.num_arms, dtype=np.int64)
        self.total_rewards = np.zeros(self.num_arms, dtype=np.int64)

    def select_arm(self) -> int:
        """Selects an arm uniformly at random in [0, 4]."""
        return int(np.random.randint(0, self.num_arms))

    def update(self, arm: int, reward: int):
        self.total_pulls[arm] += 1
        self.total_rewards[arm] += reward
