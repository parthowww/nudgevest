"""NudgeVest: a behavioral micro-investing MVP demo."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from bandit import ContextualThompsonSamplingBandit, NUDGE_TYPES
from portfolio import Portfolio, investment_amount
from propensity_model import PropensityModel, sample_acceptance
from transactions import generate_transaction


st.set_page_config(
    page_title="NudgeVest",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner="Training the synthetic propensity model…")
def get_propensity_model() -> PropensityModel:
    return PropensityModel.train()


def reset_simulation() -> None:
    st.session_state.bandit = ContextualThompsonSamplingBandit(seed=2026)
    st.session_state.portfolio = Portfolio()
    st.session_state.records = []
    st.session_state.rng = np.random.default_rng(2026)
    st.session_state.clock = datetime(2026, 9, 21, 8, 0)


def ensure_state() -> None:
    if "records" not in st.session_state:
        reset_simulation()
    if "propensity_model" not in st.session_state:
        st.session_state.propensity_model = get_propensity_model()


def simulate_ticks(ticks: int) -> None:
    bandit: ContextualThompsonSamplingBandit = st.session_state.bandit
    portfolio: Portfolio = st.session_state.portfolio
    model: PropensityModel = st.session_state.propensity_model
    rng: np.random.Generator = st.session_state.rng
    timestamp = st.session_state.clock
    records: list[dict[str, Any]] = st.session_state.records

    bandit_cumulative = float(records[-1]["bandit_cumulative"]) if records else 0.0
    random_cumulative = float(records[-1]["random_cumulative"]) if records else 0.0

    for _ in range(ticks):
        timestamp += timedelta(
            hours=int(rng.choice([1, 2, 3, 4, 6, 8])),
            minutes=int(rng.integers(0, 60)),
        )
        transaction = generate_transaction(rng, timestamp)
        propensities = model.all_propensities(transaction)
        chosen_arm = bandit.select(transaction, propensities)
        bandit_accepted = sample_acceptance(transaction, chosen_arm, rng)
        bandit.update(transaction, chosen_arm, bandit_accepted)

        baseline_arm = str(rng.choice(NUDGE_TYPES))
        baseline_accepted = sample_acceptance(transaction, baseline_arm, rng)
        contribution = (
            investment_amount(transaction["amount"], chosen_arm)
            if bandit_accepted
            else 0.0
        )
        balance = portfolio.tick(rng, contribution)

        bandit_cumulative += bandit_accepted
        random_cumulative += baseline_accepted
        records.append(
            {
                "timestamp": transaction["timestamp"],
                "category": transaction["category"].title(),
                "amount": transaction["amount"],
                "day": transaction["day_name"],
                "hour": transaction["hour"],
                "nudge": chosen_arm,
                "accepted": bool(bandit_accepted),
                "invested": contribution,
                "portfolio_balance": balance,
                "bandit_cumulative": bandit_cumulative,
                "random_cumulative": random_cumulative,
                "regret_gap": bandit_cumulative - random_cumulative,
                "baseline_arm": baseline_arm,
                "baseline_accepted": bool(baseline_accepted),
                "propensity": propensities[chosen_arm],
            }
        )

    st.session_state.clock = timestamp


def make_performance_chart(frame: pd.DataFrame) -> go.Figure:
    chart = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.12,
        subplot_titles=(
            "Portfolio balance",
            "Rolling acceptance rate · 10 interactions",
            "Cumulative reward vs. random-arm baseline",
        ),
    )
    chart.add_trace(
        go.Scatter(
            x=frame["step"],
            y=frame["portfolio_balance"],
            mode="lines",
            name="Portfolio",
            line={"color": "#168aad", "width": 3},
            hovertemplate="Tick %{x}<br>$%{y:.2f}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    rolling_acceptance = frame["accepted"].astype(float).rolling(10, min_periods=1).mean() * 100
    chart.add_trace(
        go.Scatter(
            x=frame["step"],
            y=rolling_acceptance,
            mode="lines",
            name="Acceptance rate",
            line={"color": "#f08c46", "width": 3},
            hovertemplate="Tick %{x}<br>%{y:.1f}%<extra></extra>",
        ),
        row=2,
        col=1,
    )
    chart.add_trace(
        go.Scatter(
            x=frame["step"],
            y=frame["bandit_cumulative"],
            mode="lines",
            name="Bandit reward",
            line={"color": "#2f9e44", "width": 3},
            hovertemplate="Tick %{x}<br>%{y:.0f} accepted<extra></extra>",
        ),
        row=3,
        col=1,
    )
    chart.add_trace(
        go.Scatter(
            x=frame["step"],
            y=frame["random_cumulative"],
            mode="lines",
            name="Random baseline",
            line={"color": "#868e96", "dash": "dash", "width": 2},
            hovertemplate="Tick %{x}<br>%{y:.0f} accepted<extra></extra>",
        ),
        row=3,
        col=1,
    )
    chart.update_yaxes(title_text="$", row=1, col=1)
    chart.update_yaxes(title_text="%", row=2, col=1)
    chart.update_yaxes(title_text="Reward", row=3, col=1)
    chart.update_xaxes(title_text="Simulation tick", row=3, col=1)
    chart.update_layout(
        height=760,
        margin={"l": 20, "r": 20, "t": 70, "b": 20},
        legend={"orientation": "h", "y": 1.08, "x": 0},
        hovermode="x unified",
    )
    return chart


def render_live_feed(frame: pd.DataFrame) -> None:
    st.subheader("Live feed")
    recent = frame.tail(10).copy()
    recent["timestamp"] = recent["timestamp"].dt.strftime("%b %d · %H:%M")
    recent["amount"] = recent["amount"].map(lambda value: f"${value:,.2f}")
    recent["invested"] = recent["invested"].map(
        lambda value: f"${value:,.2f}" if value else "—"
    )
    recent["accepted"] = recent["accepted"].map(lambda value: "Accepted" if value else "Declined")
    recent = recent[
        ["timestamp", "category", "amount", "nudge", "accepted", "invested"]
    ].rename(
        columns={
            "timestamp": "Time",
            "category": "Category",
            "amount": "Purchase",
            "nudge": "Nudge shown",
            "accepted": "Outcome",
            "invested": "Invested",
        }
    )
    st.dataframe(recent, use_container_width=True, hide_index=True)


ensure_state()

with st.sidebar:
    st.title("NudgeVest")
    st.caption("Behavioral micro-investing lab")
    st.divider()
    ticks = st.slider(
        "Ticks to simulate",
        min_value=10,
        max_value=500,
        value=50,
        step=10,
        help="Each tick generates a transaction, chooses a nudge, and updates the portfolio.",
    )
    run = st.button("Run simulation", type="primary", use_container_width=True)
    if st.button("Reset simulation", use_container_width=True):
        reset_simulation()
        st.rerun()
    st.divider()
    st.caption("Hybrid selector")
    st.write(
        "Thompson Sampling explores each nudge while the XGBoost propensity model "
        "weights choices toward contextually likely acceptances."
    )

if run:
    simulate_ticks(ticks)

records = st.session_state.records
st.title("NudgeVest")
st.markdown(
    "A behavioral micro-investing simulation that learns which small prompt is most "
    "likely to turn everyday spending into invested capital."
)

if records:
    data = pd.DataFrame(records)
    data["step"] = range(1, len(data) + 1)
    accepted_count = int(data["accepted"].sum())
    latest_balance = float(data.iloc[-1]["portfolio_balance"])
    advantage = float(data.iloc[-1]["regret_gap"])
    acceptance_rate = accepted_count / len(data)
else:
    data = pd.DataFrame(
        columns=[
            "timestamp",
            "category",
            "amount",
            "nudge",
            "accepted",
            "invested",
            "portfolio_balance",
            "bandit_cumulative",
            "random_cumulative",
            "regret_gap",
        ]
    )
    accepted_count = 0
    latest_balance = 0.0
    advantage = 0.0
    acceptance_rate = 0.0

metric_a, metric_b, metric_c, metric_d = st.columns(4)
metric_a.metric("Portfolio balance", f"${latest_balance:,.2f}")
metric_b.metric("Accepted nudges", f"{accepted_count:,}")
metric_c.metric("Acceptance rate", f"{acceptance_rate:.1%}")
metric_d.metric(
    "Bandit advantage",
    f"{advantage:+.0f}",
    help="Cumulative accepted nudges from the learned bandit minus a random-arm baseline run on the same transactions.",
)

if records:
    render_live_feed(data)
    st.subheader("Performance")
    st.plotly_chart(make_performance_chart(data), use_container_width=True)
else:
    st.info("Choose the number of ticks, then run the simulation to populate the live feed and charts.")

st.subheader("Arm stats")
st.caption("Aggregated Beta posteriors across visited context buckets. Higher posterior mean means the bandit currently expects more acceptances.")
stats = pd.DataFrame(st.session_state.bandit.arm_stats())
stats["Posterior mean"] = stats["Posterior mean"].map(lambda value: f"{value:.1%}")
st.dataframe(stats, use_container_width=True, hide_index=True)

with st.expander("How this demo works"):
    st.markdown(
        """
        1. A synthetic purchase provides category, amount, day, and time context.
        2. The XGBoost model estimates the acceptance propensity of each nudge.
        3. A contextual Thompson Sampling bandit samples each nudge's Beta posterior,
           weighted by those propensities, and chooses an arm.
        4. A noisy hand-coded behavior function produces the observed acceptance.
        5. Accepted nudges add money to the simulated portfolio, which also receives a
           small randomized daily market return.
        6. The random-arm baseline is run in parallel on the same transaction contexts,
           so the reward comparison stays visible as the bandit learns.
        """
    )