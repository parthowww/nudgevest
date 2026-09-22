# 🌱 NudgeVest — Behavioral Micro-Investing MVP Demo

A modern behavioral micro-investing web application running on `localhost:8000` without Streamlit.

## Features
- **Contextual Thompson Sampling Bandit:** 5 behavioral nudge arms across 105 discrete context buckets with Beta(α, β) posteriors.
- **Hybrid XGBoost Propensity Weighting:** Blends offline behavioral machine learning with online Bayesian exploration.
- **Simulated Investment Portfolio:** Micro-deposits on accepted nudges with randomized daily ETF market drift and volatility.
- **Live Feed & Real-Time Analytics:**
  1. **Live Transaction Feed:** Real-time stream of synthetic user transactions, showing category, amount, nudge selected, and acceptance outcome.
  2. **Regret & Cumulative Reward Charts:** Interactive Chart.js graphs proving the bandit's outperformance vs. a parallel random baseline and sub-linear regret vs. optimal oracle.
  3. **Arm Stats Table:** Live Bayesian inspection of $\alpha$, $\beta$, $\mathbb{E}[\theta]$, pulls, and empirical acceptance rates.

## How to Run

1. Navigate to the project directory:
```bash
cd /home/parthow/workspace/nudgevest
```

2. Start the local web server:
```bash
python3 app.py
```
*(Or specify custom port: `python3 app.py --port 8000`)*

3. Open your browser:
```
http://localhost:8000
```
