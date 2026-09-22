# 🌱 NudgeVest — Behavioral Micro-Investing MVP

[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Powered-FF6600.svg)](https://xgboost.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**NudgeVest** is a behavioral micro-investing platform that uses machine learning and reinforcement learning to encourage everyday saving and investing. By monitoring daily transaction contexts, NudgeVest deploys a **Contextual Multi-Armed Bandit (Thompson Sampling)** blended with an **XGBoost Propensity Model** to personalize micro-deposit "nudges" (e.g. rounding up coffee purchases or weekly deposit challenges), accumulating into a simulated compounding investment portfolio.

---

## 📌 Architecture & How It Works

```mermaid
flowchart TD
    TX[Daily Transaction Generator] -->|Context: Category, Amount, Day, Hour| HYBRID[Hybrid Decision Engine]
    
    subgraph HYBRID[Hybrid Nudge Selector]
        XGB[XGBoost Propensity Model<br/>Offline Supervised Prior]
        TS[Contextual Thompson Sampling<br/>105 Beta Posteriors]
        WEIGHT["Score = θ_arm × Propensity(arm)<br/>Select argmax(Score)"]
        XGB --> WEIGHT
        TS --> WEIGHT
    end

    WEIGHT -->|Selected Nudge| USER[User Experience / Simulated Agent]
    USER -->|Feedback: Accept or Decline| UPDATE[Online Bayesian Update]
    UPDATE -->|Update α, β in Context Bucket| TS

    USER -->|If Accepted: Deposit Principal| PORTFOLIO[Compounding Investment Portfolio]
    PORTFOLIO -->|Market Returns ~ N(0.03%, 0.18%)| GROWTH[Portfolio Balance Growth]
```

### 1. The Contextual Multi-Armed Bandit (`bandit.py`)
- **Discrete Context Bucketing:** Context is mapped into $5 \text{ categories} \times 3 \text{ amount buckets (low/med/high)} \times 7 \text{ days of week} = 105$ localized context buckets.
- **Beta-Bernoulli Conjugate Prior:** Each bucket maintains localized $\text{Beta}(\alpha, \beta)$ posteriors for all arms, initialized to $\text{Beta}(1, 1)$ (uniform prior).
- **Online Adaptation:**
  - $\alpha_{b, a} \leftarrow \alpha_{b, a} + 1$ upon acceptance (reward = 1).
  - $\beta_{b, a} \leftarrow \beta_{b, a} + 1$ upon decline (reward = 0).

### 2. The Hybrid XGBoost Propensity Mechanism (`propensity_model.py`)
At startup, an **XGBoost Binary Classifier** is trained on 2,500 synthetic interactions with ground-truth behavioral economics to predict $P(\text{accept} \mid \text{context}, a)$.

```python
# HYBRID MECHANISM:
# Weight Thompson posterior sample by the XGBoost propensity score.
# Blends offline supervised ML prior with live online Bayesian exploration.
final_scores = theta_samples * propensity_scores
chosen_arm = np.argmax(final_scores)
```

### 3. The 5 Behavioral Nudge Arms
| Arm # | Nudge Type | Economics / Action | Behavioral Sweet Spot |
| :---: | :--- | :--- | :--- |
| **0** | **Round-up-and-invest** | Spare change to next dollar ($0.25 - $1.00) | Small frequent purchases (Coffee, Transit) |
| **1** | **Weekly-savings-invest** | Fixed micro-deposit ($5.00) | Weekend budgeting mindset (Fri-Sun) |
| **2** | **Skip-this-purchase-invest**| 50% of purchase value invested ($3 - $20) | Discretionary items (Shopping, Entertainment) |
| **3** | **Streak-bonus-invest** | Habit gamification deposit ($2.50) | Routine morning/evening hours |
| **4** | **No nudge (control)** | $0.00 | Baseline control arm |

### 4. Compounding Micro-Portfolio (`portfolio.py`)
- Accrues principal contributions whenever a nudge is accepted.
- Simulates daily diversified ETF market growth with positive drift and volatility:
  $$\text{Daily Return} \sim \mathcal{N}(\mu = 0.03\%, \sigma = 0.18\%)$$

---

## 📊 Core Proof Point: Regret & Cumulative Rewards

NudgeVest executes a **parallel random-arm baseline** under the exact same transaction sequence:
- **Cumulative Reward:** The contextual bandit quickly identifies user behavioral affinities, achieving a **+25% to +35% higher acceptance rate** than random exploration.
- **Cumulative Regret:** While the random baseline incurs steady **linear regret**, the hybrid bandit achieves **sub-linear regret**, plateauing as it converges on optimal arms.

---

## 🚀 Quickstart & Local Setup

### Prerequisites
- Python 3.10+
- Recommended: virtual environment

### 1. Clone the Repository
```bash
git clone https://github.com/parthowww/nudgevest.git
cd nudgevest
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Start the Local Web Application
```bash
python3 app.py
```
*(Or specify custom host/port: `python3 app.py --port 8000`)*

### 4. Open in Your Web Browser
Navigate to:
👉 **[http://localhost:8000](http://localhost:8000)**

---

## 🖥️ Web Interface Features

- **⚡ Live Control Panel:** Slide transaction batch sizes (10–250 ticks), toggle XGBoost hybrid weighting, step 1 tick at a time, or run batch simulations.
- **📡 Live Transaction Feed:** View the latest transactions, merchant categories, amounts, selected nudges, and real-time user acceptance outcomes.
- **📈 Interactive Chart.js Visualizations:**
  - *Cumulative Rewards:* Bandit vs. Random baseline over ticks.
  - *Regret Curve:* Visual validation of sub-linear regret vs. optimal oracle.
  - *Portfolio Growth:* Compounding investment balance vs. principal deposits.
  - *Rolling Acceptance Rate:* 20-tick rolling conversion percentage.
- **🧪 Posterior Parameter Table:** Live inspection of Bayesian parameters ($\alpha$, $\beta$, $\mathbb{E}[\theta]$, pulls, conversion rate) per arm.

---

## 🔌 API Reference

The FastAPI backend exposes the following REST endpoints:

| Endpoint | Method | Description |
| :--- | :---: | :--- |
| `/` | `GET` | Serves the single-page interactive dashboard. |
| `/api/state` | `GET` | Returns current KPIs, live feed, arm statistics, and chart series. |
| `/api/simulate` | `POST` | Executes $N$ simulation ticks (`{"ticks": 50, "use_hybrid": true}`). |
| `/api/step` | `POST` | Simulates a single tick for step-by-step observation. |
| `/api/reset` | `POST` | Resets all bandit parameters, baseline, portfolio, and history. |

---

## 📁 File Structure

```
nudgevest/
├── app.py               # FastAPI backend + single-page HTML5/Tailwind/Chart.js UI
├── bandit.py            # Contextual Thompson Sampling bandit (hand-rolled, NumPy)
├── propensity_model.py  # XGBoost propensity classifier & behavioral ground-truth
├── portfolio.py         # Compounding portfolio tracker & ETF return simulator
├── transactions.py      # Synthetic transaction generator
├── requirements.txt     # Pinned Python package dependencies
├── .gitignore           # Standard git exclusion rules
└── README.md            # Documentation and architecture guide
```

---

## 📜 License

MIT License. Feel free to use, modify, and build upon this for research and educational purposes.
