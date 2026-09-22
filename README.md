# NudgeVest

NudgeVest is a behavioral micro-investing MVP demo built with Streamlit. It simulates everyday purchases, chooses a contextual investing nudge with a learning bandit, models acceptance propensity with XGBoost, and tracks the resulting contributions in a simulated portfolio.

> **Demo only:** NudgeVest uses synthetic transactions, synthetic user behavior, and randomized portfolio returns. It is not financial advice, a brokerage product, or a recommendation engine for real investments.

## What it demonstrates

NudgeVest is designed to make a reinforcement-learning-inspired product loop visible:

1. A synthetic transaction creates context: category, amount, day of week, and hour.
2. An XGBoost classifier estimates how likely the user is to accept each nudge type.
3. A contextual Thompson Sampling bandit samples from Beta posteriors for each context bucket.
4. The Thompson samples are weighted by the XGBoost propensity estimates.
5. A noisy behavioral function produces an observed acceptance or decline.
6. Accepted nudges add a small contribution to the simulated portfolio.
7. A randomized market return is applied to the existing balance on every tick.
8. The same transaction context is evaluated by a random-arm baseline for comparison.

The app is intentionally small and inspectable: the model, bandit, simulator, portfolio, and UI each live in their own module.

## Features

- **Synthetic transaction stream**
  - Categories: coffee, food, transport, shopping, and entertainment
  - Category-appropriate purchase ranges
  - Day-of-week and hour context
- **Five nudge arms**
  - Round-up-and-invest
  - Weekly-savings-invest
  - Skip-this-purchase-invest
  - Streak-bonus-invest
  - No nudge control arm
- **Contextual Thompson Sampling**
  - Beta(alpha, beta) posterior per nudge and discretized context bucket
  - Amount buckets: low, medium, and high
  - Separate posteriors for category, amount bucket, and day of week
- **XGBoost propensity model**
  - Trained once at startup on approximately 2,000 synthetic interactions
  - Uses one-hot encoded category, amount bucket, day of week, and nudge type
  - Shares the same hand-coded behavioral ground truth as the live simulator
- **Portfolio simulation**
  - Nudge-specific contribution amounts
  - Small randomized daily return
  - Running balance, total contributions, and return history
- **Evaluation view**
  - Last 10 simulated transactions and outcomes
  - Portfolio balance over time
  - Rolling 10-interaction acceptance rate
  - Cumulative reward for the learned bandit versus a random-arm baseline
  - Aggregated arm alpha, beta, posterior mean, and observation counts
- **Session persistence**
  - `st.session_state` keeps the simulation state across Streamlit reruns
  - Reset control starts a fresh seeded simulation

## Project structure

```text
.
├── app.py                 # Streamlit UI and simulation loop
├── bandit.py              # Contextual Thompson Sampling implementation
├── portfolio.py           # Simulated investment portfolio
├── propensity_model.py    # XGBoost model and behavior ground truth
├── transactions.py        # Synthetic transaction generator
├── requirements.txt       # Python runtime dependencies
├── pyproject.toml         # uv-managed Python project metadata
└── README.md              # Project documentation
```

## Requirements

- Python 3.13 or a compatible recent Python version
- Streamlit
- NumPy
- pandas
- scikit-learn
- XGBoost
- Plotly

The exact Python dependencies are listed in `requirements.txt`. This project also includes `pyproject.toml` and `uv.lock` for reproducible uv-based setup.

## Run locally

### Option 1: install from `requirements.txt`

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

### Option 2: use uv

```bash
uv sync
uv run streamlit run app.py
```

The application opens on Streamlit's default local port. To choose a port explicitly:

```bash
streamlit run app.py --server.port 8000
```

## Using the app

1. Open the Streamlit page.
2. Choose the number of simulation ticks in the sidebar.
3. Select **Run simulation**.
4. Review the live transaction feed.
5. Compare the learned bandit with the random-arm baseline in the reward chart.
6. Inspect the arm stats table to see posterior learning.
7. Select **Reset simulation** to start again from the seeded initial state.

Each tick represents a new synthetic purchase. The simulation can be run repeatedly; each run continues from the existing session state until the simulation is reset.

## How the hybrid selector works

The selector combines model-based propensity estimates with uncertainty-aware bandit exploration.

For each context and nudge arm:

```text
sampled_score = ThompsonSample(Beta(alpha, beta)) × XGBoostPropensity
```

The arm with the highest sampled score is selected. This has two useful effects:

- Thompson Sampling explores arms that still have uncertain posteriors.
- The propensity model favors arms that appear more likely to be accepted in the current context.

After the observed outcome:

- `alpha` increases for an accepted nudge.
- `beta` increases for a declined nudge.

The posterior table shown in the app aggregates these values across all context buckets visited during the current session.

## Reward comparison and regret gap

For each generated transaction, NudgeVest evaluates two decisions:

1. The learned hybrid bandit chooses an arm using its posteriors and propensity scores.
2. A random baseline chooses an arm uniformly from the same five arms.

Both outcomes are sampled from the same behavioral ground-truth function, with independent observation noise. The chart plots cumulative accepted nudges for both strategies and the headline metric reports:

```text
bandit advantage = cumulative bandit reward - cumulative random reward
```

This is a practical demo comparison rather than a formal regret bound against an oracle policy. A positive gap means the learned strategy has produced more accepted nudges than the random baseline in the current simulation.

## Synthetic behavioral ground truth

The demo deliberately uses a transparent behavior function so the learning loop can be inspected:

- Round-up nudges perform better on small purchases.
- Weekly savings nudges perform better on weekends.
- Skip-purchase nudges perform better for shopping and entertainment.
- Streak bonuses receive a lift on selected weekdays.
- The no-nudge control arm has no acceptance reward.

The XGBoost model is trained on samples from this function, while live decisions add small noise to simulate imperfect observed behavior.

## Portfolio contribution rules

Accepted nudges create the following simulated contributions:

- **Round-up-and-invest:** round the purchase up to the next whole dollar
- **Weekly-savings-invest:** approximately 2% of the purchase, bounded
- **Skip-this-purchase-invest:** approximately 5% of the purchase, bounded
- **Streak-bonus-invest:** a fixed contribution
- **No nudge:** no contribution

The portfolio then applies a small randomized daily return with a positive average drift and market-like volatility. These returns are synthetic and should not be interpreted as a forecast.

## Reproducibility

The initial simulation uses a fixed seed so a fresh reset is repeatable. The propensity model also uses a fixed seed for its synthetic training data and XGBoost configuration. This makes the demo easier to inspect and compare while still preserving randomized outcomes within a run.

## Validation

The project has been checked with:

```bash
python -m py_compile app.py bandit.py propensity_model.py portfolio.py transactions.py
```

The core smoke test trains the XGBoost model, generates transactions, selects and updates bandit arms, samples rewards, and updates the portfolio.

## Limitations

- No real bank, brokerage, ETF, or market data is connected.
- No authentication or multi-user state is implemented.
- The behavioral function is hand-coded and synthetic.
- The baseline is random-arm comparison, not an optimal oracle.
- Portfolio returns are randomized demo values.
- Streamlit session state is local to the running browser session.

## Possible next steps

- Add a configurable user profile with persistent preference history.
- Compare against an oracle policy using the known synthetic ground truth.
- Add confidence intervals around acceptance and reward curves.
- Persist simulation runs for side-by-side experiment comparison.
- Add more contextual features such as transaction frequency and recent streak length.
- Replace synthetic transactions with a privacy-preserving sample dataset.
- Add experiment export to CSV or JSON.

## License

No license has been selected yet. Add a license file before distributing or reusing the project publicly.