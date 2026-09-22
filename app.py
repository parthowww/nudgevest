#!/usr/bin/env python3
"""
app.py
NudgeVest — Behavioral Micro-Investing MVP Demo
Local Web Application (FastAPI + Modern HTML5/Tailwind/Chart.js Frontend).

Runs locally on localhost (e.g. http://localhost:8000), accessible via any web browser.
No Streamlit required.
"""

import sys
import os
import argparse

# Auto-bootstrap virtual environment if packages are in .venv
try:
    import numpy as np
    import fastapi
    import uvicorn
except ImportError:
    for candidate in [
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".venv", "bin", "python")),
        os.path.expanduser("/home/parthow/.venv/bin/python"),
    ]:
        if os.path.exists(candidate) and sys.executable != candidate:
            os.execv(candidate, [candidate] + sys.argv)
    raise

from typing import Dict, Any, List, Optional
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from transactions import generate_transaction
from propensity_model import (
    PropensityModel,
    ground_truth_acceptance_prob,
    simulate_user_response,
    ARM_NAMES,
    NUM_ARMS,
)
from bandit import ContextualThompsonBandit, RandomArmBaseline
from portfolio import SimulatedPortfolio


# ==============================================================================
# Application State
# ==============================================================================
app = FastAPI(title="NudgeVest", description="Behavioral Micro-Investing Bandit Web App")

# Global models and state
propensity_model: Optional[PropensityModel] = None
bandit: Optional[ContextualThompsonBandit] = None
baseline: Optional[RandomArmBaseline] = None
portfolio: Optional[SimulatedPortfolio] = None
simulation_history: List[Dict[str, Any]] = []

cum_reward_bandit: int = 0
cum_reward_random: int = 0
cum_regret_bandit: float = 0.0
cum_regret_random: float = 0.0
total_ticks: int = 0


def init_app_state(reset_all: bool = False):
    """Initializes or resets the global simulation environment."""
    global propensity_model, bandit, baseline, portfolio
    global simulation_history, cum_reward_bandit, cum_reward_random
    global cum_regret_bandit, cum_regret_random, total_ticks

    if propensity_model is None or not propensity_model.is_trained:
        print("[Startup] Training XGBoost Propensity Model on 2,500 synthetic interactions...")
        propensity_model = PropensityModel()
        propensity_model.train(n_samples=2500, random_seed=42)
        print("[Startup] Propensity model ready.")

    if reset_all or bandit is None:
        bandit = ContextualThompsonBandit(hybrid_weighting=True)
        baseline = RandomArmBaseline()
        portfolio = SimulatedPortfolio(initial_balance=0.0)
        simulation_history = []
        cum_reward_bandit = 0
        cum_reward_random = 0
        cum_regret_bandit = 0.0
        cum_regret_random = 0.0
        total_ticks = 0


# Call initial setup
init_app_state()


def step_simulation(use_hybrid: bool = True) -> Dict[str, Any]:
    """Executes a single simulation tick and updates all tracking state."""
    global total_ticks, cum_reward_bandit, cum_reward_random
    global cum_regret_bandit, cum_regret_random, simulation_history

    total_ticks += 1
    context = generate_transaction(tick=total_ticks)

    # 1. Arm Selection
    bandit.hybrid_weighting = use_hybrid
    chosen_arm_bandit, raw_samples, final_scores = bandit.select_arm(
        context, propensity_model=propensity_model if use_hybrid else None
    )
    chosen_arm_random = baseline.select_arm()

    # 2. Optimal Oracle Arm for Regret Calculation
    all_true_probs = [ground_truth_acceptance_prob(context, a) for a in range(NUM_ARMS)]
    optimal_arm = int(np.argmax(all_true_probs))
    optimal_prob = float(all_true_probs[optimal_arm])

    # 3. User Acceptance Simulation
    accepted_bandit, true_p_bandit = simulate_user_response(context, chosen_arm_bandit)
    accepted_random, true_p_random = simulate_user_response(context, chosen_arm_random)

    # 4. Learning Updates
    bandit.update(context, chosen_arm_bandit, accepted_bandit)
    baseline.update(chosen_arm_random, accepted_random)

    # 5. Portfolio Update
    portfolio_rec = portfolio.step(total_ticks, context, chosen_arm_bandit, bool(accepted_bandit))

    # 6. Regret Tracking
    instant_regret_bandit = max(0.0, optimal_prob - true_p_bandit)
    instant_regret_random = max(0.0, optimal_prob - true_p_random)

    cum_reward_bandit += accepted_bandit
    cum_reward_random += accepted_random
    cum_regret_bandit += instant_regret_bandit
    cum_regret_random += instant_regret_random

    record = {
        "tick": total_ticks,
        "category": context["category"],
        "amount": context["amount"],
        "amount_bucket": context["amount_bucket"],
        "day_name": context["day_name"],
        "hour": context["hour"],
        "merchant": context["merchant"],
        "round_up_amount": context["round_up_amount"],
        "bandit_arm": chosen_arm_bandit,
        "bandit_arm_name": ARM_NAMES[chosen_arm_bandit],
        "bandit_accepted": accepted_bandit,
        "bandit_true_p": round(true_p_bandit, 3),
        "random_arm": chosen_arm_random,
        "random_arm_name": ARM_NAMES[chosen_arm_random],
        "random_accepted": accepted_random,
        "random_true_p": round(true_p_random, 3),
        "cum_reward_bandit": cum_reward_bandit,
        "cum_reward_random": cum_reward_random,
        "cum_regret_bandit": round(cum_regret_bandit, 3),
        "cum_regret_random": round(cum_regret_random, 3),
        "invested_amount": portfolio_rec["invested_amount"],
        "portfolio_balance": portfolio_rec["portfolio_balance"],
        "total_contributions": portfolio_rec["total_contributions"],
        "daily_return_pct": portfolio_rec["daily_return_pct"],
    }
    simulation_history.append(record)
    return record


def get_current_dashboard_payload() -> Dict[str, Any]:
    """Assembles all data needed by the frontend dashboard."""
    bandit_rate = (cum_reward_bandit / total_ticks * 100) if total_ticks > 0 else 0.0
    random_rate = (cum_reward_random / total_ticks * 100) if total_ticks > 0 else 0.0
    rate_lift = bandit_rate - random_rate

    # Downsample points for chart performance if history is large
    step_size = max(1, len(simulation_history) // 150)
    sampled_history = simulation_history[::step_size]
    if simulation_history and (not sampled_history or sampled_history[-1]["tick"] != simulation_history[-1]["tick"]):
        sampled_history.append(simulation_history[-1])

    # Compute rolling acceptance rates
    window = min(20, max(5, total_ticks // 4)) if total_ticks >= 5 else 1
    rolling_ticks = []
    rolling_bandit = []
    rolling_random = []

    if simulation_history:
        acc_b = [r["bandit_accepted"] for r in simulation_history]
        acc_r = [r["random_accepted"] for r in simulation_history]
        for i in range(len(simulation_history)):
            start_idx = max(0, i - window + 1)
            rolling_ticks.append(simulation_history[i]["tick"])
            rolling_bandit.append(round(sum(acc_b[start_idx : i + 1]) / (i - start_idx + 1) * 100, 1))
            rolling_random.append(round(sum(acc_r[start_idx : i + 1]) / (i - start_idx + 1) * 100, 1))

        # Sample rolling rates to match chart points
        rolling_ticks = rolling_ticks[::step_size]
        rolling_bandit = rolling_bandit[::step_size]
        rolling_random = rolling_random[::step_size]

    return {
        "kpis": {
            "total_ticks": total_ticks,
            "portfolio_balance": round(portfolio.balance, 2) if portfolio else 0.0,
            "total_contributions": round(portfolio.total_contributions, 2) if portfolio else 0.0,
            "market_gain": round(portfolio.balance - portfolio.total_contributions, 2) if portfolio else 0.0,
            "cum_reward_bandit": cum_reward_bandit,
            "cum_reward_random": cum_reward_random,
            "bandit_acceptance_rate": round(bandit_rate, 1),
            "random_acceptance_rate": round(random_rate, 1),
            "acceptance_lift": round(rate_lift, 1),
            "cum_regret_bandit": round(cum_regret_bandit, 2),
            "cum_regret_random": round(cum_regret_random, 2),
        },
        "live_feed": simulation_history[-10:][::-1] if simulation_history else [],
        "arm_stats": bandit.get_arm_stats() if bandit else [],
        "charts": {
            "ticks": [r["tick"] for r in sampled_history],
            "reward_bandit": [r["cum_reward_bandit"] for r in sampled_history],
            "reward_random": [r["cum_reward_random"] for r in sampled_history],
            "regret_bandit": [r["cum_regret_bandit"] for r in sampled_history],
            "regret_random": [r["cum_regret_random"] for r in sampled_history],
            "portfolio_balance": [r["portfolio_balance"] for r in sampled_history],
            "total_contributions": [r["total_contributions"] for r in sampled_history],
            "rolling_ticks": rolling_ticks,
            "rolling_bandit": rolling_bandit,
            "rolling_random": rolling_random,
        },
    }


# ==============================================================================
# API Routes
# ==============================================================================
class SimulateRequest(BaseModel):
    ticks: int = 50
    use_hybrid: bool = True


@app.get("/api/state")
def api_get_state():
    return JSONResponse(get_current_dashboard_payload())


@app.post("/api/simulate")
def api_simulate(req: SimulateRequest):
    count = max(1, min(req.ticks, 500))
    for _ in range(count):
        step_simulation(use_hybrid=req.use_hybrid)
    return JSONResponse(get_current_dashboard_payload())


@app.post("/api/step")
def api_step(req: SimulateRequest):
    step_simulation(use_hybrid=req.use_hybrid)
    return JSONResponse(get_current_dashboard_payload())


@app.post("/api/reset")
def api_reset():
    init_app_state(reset_all=True)
    return JSONResponse(get_current_dashboard_payload())


# ==============================================================================
# Modern Single-Page Web Frontend (HTML5 + Tailwind CSS + Chart.js)
# ==============================================================================
HTML_CONTENT = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NudgeVest — Behavioral Micro-Investing MVP</title>
    <!-- Tailwind CSS CDN -->
    <script src="https://cdn.tailwindcss.com"></script>
    <!-- Chart.js CDN -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <!-- Inter Font -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
    <style>
        body { font-family: 'Inter', sans-serif; }
        .tab-active { border-bottom: 2px solid #10B981; color: #10B981; font-weight: 600; }
        .pulse-live { animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite; }
        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: .4; } }
    </style>
</head>
<body class="bg-slate-50 text-slate-800 min-h-screen">

    <!-- Top Navigation Header -->
    <header class="bg-white border-b border-slate-200 sticky top-0 z-50">
        <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
            <div class="flex items-center space-x-3">
                <div class="h-10 w-10 rounded-xl bg-emerald-500 text-white flex items-center justify-center font-extrabold text-xl shadow-sm">
                    🌱
                </div>
                <div>
                    <h1 class="text-lg font-bold text-slate-900 leading-tight">NudgeVest</h1>
                    <p class="text-xs text-slate-500">Contextual Bandit Micro-Investing MVP</p>
                </div>
            </div>
            <div class="flex items-center space-x-3">
                <span class="inline-flex items-center px-2.5 py-1 rounded-full text-xs font-medium bg-emerald-50 text-emerald-700 border border-emerald-200">
                    <span class="w-2 h-2 mr-1.5 bg-emerald-500 rounded-full pulse-live"></span> Localhost:8000
                </span>
                <span class="text-xs text-slate-400">|</span>
                <span class="text-xs text-slate-600 font-mono bg-slate-100 px-2 py-1 rounded">XGBoost + Thompson Sampling</span>
            </div>
        </div>
    </header>

    <!-- Main Container -->
    <main class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6 space-y-6">

        <!-- Control Bar Card -->
        <div class="bg-white rounded-2xl p-5 border border-slate-200/80 shadow-sm flex flex-wrap items-center justify-between gap-4">
            <div class="flex flex-wrap items-center gap-4">
                <div class="flex items-center space-x-2">
                    <label for="tickSlider" class="text-sm font-semibold text-slate-700">Simulate Ticks:</label>
                    <input type="range" id="tickSlider" min="10" max="250" step="10" value="50" class="w-32 accent-emerald-600">
                    <span id="sliderValue" class="text-sm font-mono font-bold bg-slate-100 px-2 py-0.5 rounded text-slate-800">50</span>
                </div>

                <div class="h-6 w-px bg-slate-200 hidden sm:block"></div>

                <label class="flex items-center space-x-2 cursor-pointer text-sm font-medium text-slate-700">
                    <input type="checkbox" id="hybridToggle" checked class="w-4 h-4 text-emerald-600 rounded border-slate-300 focus:ring-emerald-500">
                    <span>Hybrid XGBoost Weighting</span>
                </label>
            </div>

            <div class="flex items-center space-x-2">
                <button id="stepBtn" onclick="stepOneTick()" class="px-3.5 py-2 text-sm font-semibold rounded-xl border border-slate-300 hover:bg-slate-50 text-slate-700 transition flex items-center space-x-1.5">
                    <span>⚡ Step 1</span>
                </button>
                <button id="runBtn" onclick="runSimulation()" class="px-5 py-2 text-sm font-semibold rounded-xl bg-emerald-600 hover:bg-emerald-700 text-white shadow-sm transition flex items-center space-x-2">
                    <span id="runSpinner" class="hidden animate-spin">⏳</span>
                    <span id="runText">▶ Run Simulation</span>
                </button>
                <button id="resetBtn" onclick="resetSimulation()" class="px-3.5 py-2 text-sm font-semibold rounded-xl border border-rose-200 text-rose-600 hover:bg-rose-50 transition">
                    🔄 Reset
                </button>
            </div>
        </div>

        <!-- KPI Metric Cards Grid -->
        <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            <!-- Metric 1: Portfolio Balance -->
            <div class="bg-white rounded-2xl p-5 border border-slate-200/80 shadow-sm">
                <p class="text-xs font-medium uppercase tracking-wider text-slate-500">Portfolio Balance</p>
                <h3 id="kpiBalance" class="text-2xl font-extrabold text-slate-900 mt-1">$0.00</h3>
                <p id="kpiGains" class="text-xs font-semibold text-emerald-600 mt-1">+$0.00 market gain</p>
            </div>

            <!-- Metric 2: Total Invested -->
            <div class="bg-white rounded-2xl p-5 border border-slate-200/80 shadow-sm">
                <p class="text-xs font-medium uppercase tracking-wider text-slate-500">Total Micro-Deposits</p>
                <h3 id="kpiInvested" class="text-2xl font-extrabold text-slate-900 mt-1">$0.00</h3>
                <p id="kpiAccepts" class="text-xs text-slate-500 mt-1">0 accepted nudges</p>
            </div>

            <!-- Metric 3: Bandit Acceptance Rate -->
            <div class="bg-white rounded-2xl p-5 border border-slate-200/80 shadow-sm">
                <p class="text-xs font-medium uppercase tracking-wider text-slate-500">Bandit Acceptance Rate</p>
                <h3 id="kpiBanditRate" class="text-2xl font-extrabold text-emerald-600 mt-1">0.0%</h3>
                <p id="kpiLift" class="text-xs font-semibold text-emerald-700 mt-1">+0.0% vs Random</p>
            </div>

            <!-- Metric 4: Baseline Rate -->
            <div class="bg-white rounded-2xl p-5 border border-slate-200/80 shadow-sm">
                <p class="text-xs font-medium uppercase tracking-wider text-slate-500">Random Baseline Rate</p>
                <h3 id="kpiRandomRate" class="text-2xl font-extrabold text-slate-600 mt-1">0.0%</h3>
                <p id="kpiRegret" class="text-xs text-slate-500 mt-1">Regret: 0.00</p>
            </div>
        </div>

        <!-- SECTION 1: Live Feed Table -->
        <div class="bg-white rounded-2xl p-6 border border-slate-200/80 shadow-sm">
            <div class="flex items-center justify-between mb-4">
                <div>
                    <h2 class="text-base font-bold text-slate-900">1. 📡 Live Transaction & Nudge Feed</h2>
                    <p class="text-xs text-slate-500">Latest simulated user transactions with selected behavioral nudge and live response.</p>
                </div>
                <span id="feedCount" class="text-xs font-mono bg-slate-100 px-2.5 py-1 rounded-full text-slate-600">Showing last 10</span>
            </div>

            <div class="overflow-x-auto">
                <table class="w-full text-left text-sm">
                    <thead>
                        <tr class="border-b border-slate-100 text-xs font-semibold uppercase tracking-wider text-slate-400">
                            <th class="pb-3 px-3">Tick</th>
                            <th class="pb-3 px-3">Merchant / Category</th>
                            <th class="pb-3 px-3">Amount</th>
                            <th class="pb-3 px-3">Time</th>
                            <th class="pb-3 px-3">Nudge Shown</th>
                            <th class="pb-3 px-3">Outcome</th>
                            <th class="pb-3 px-3">Portfolio Impact</th>
                        </tr>
                    </thead>
                    <tbody id="feedTableBody" class="divide-y divide-slate-100">
                        <tr>
                            <td colspan="7" class="py-8 text-center text-sm text-slate-400">
                                No transactions simulated yet. Click <b>▶ Run Simulation</b> to begin.
                            </td>
                        </tr>
                    </tbody>
                </table>
            </div>
        </div>

        <!-- SECTION 2: Analytics & Regret Charts -->
        <div class="bg-white rounded-2xl p-6 border border-slate-200/80 shadow-sm">
            <div class="flex items-center justify-between mb-4">
                <div>
                    <h2 class="text-base font-bold text-slate-900">2. 📊 Performance & Learning Analytics</h2>
                    <p class="text-xs text-slate-500">Core proof point: Contextual bandit achieves higher reward and sub-linear regret vs. linear penalty of random baseline.</p>
                </div>
                <!-- Chart Tabs -->
                <div class="flex space-x-4 border-b border-slate-200 text-sm">
                    <button onclick="switchChartTab('rewards')" id="tabRewards" class="pb-2 tab-active transition">Rewards & Regret</button>
                    <button onclick="switchChartTab('portfolio')" id="tabPortfolio" class="pb-2 text-slate-500 hover:text-slate-700 transition">Portfolio Growth</button>
                    <button onclick="switchChartTab('acceptance')" id="tabAcceptance" class="pb-2 text-slate-500 hover:text-slate-700 transition">Rolling Acceptance</button>
                </div>
            </div>

            <!-- Chart Containers -->
            <div class="grid grid-cols-1 lg:grid-cols-2 gap-6" id="viewRewards">
                <div class="bg-slate-50 p-4 rounded-xl border border-slate-100">
                    <h4 class="text-xs font-bold uppercase tracking-wider text-slate-500 mb-2">Cumulative Rewards (Accepted Nudges)</h4>
                    <div class="h-64"><canvas id="rewardChart"></canvas></div>
                </div>
                <div class="bg-slate-50 p-4 rounded-xl border border-slate-100">
                    <h4 class="text-xs font-bold uppercase tracking-wider text-slate-500 mb-2">Cumulative Regret vs. Optimal Oracle</h4>
                    <div class="h-64"><canvas id="regretChart"></canvas></div>
                </div>
            </div>

            <div class="hidden grid grid-cols-1 gap-6" id="viewPortfolio">
                <div class="bg-slate-50 p-4 rounded-xl border border-slate-100">
                    <h4 class="text-xs font-bold uppercase tracking-wider text-slate-500 mb-2">Portfolio Balance vs. Micro-Deposits ($)</h4>
                    <div class="h-72"><canvas id="portfolioChart"></canvas></div>
                </div>
            </div>

            <div class="hidden grid grid-cols-1 gap-6" id="viewAcceptance">
                <div class="bg-slate-50 p-4 rounded-xl border border-slate-100">
                    <h4 class="text-xs font-bold uppercase tracking-wider text-slate-500 mb-2">Rolling Acceptance Rate (%) [20-Tick Window]</h4>
                    <div class="h-72"><canvas id="rollingChart"></canvas></div>
                </div>
            </div>
        </div>

        <!-- SECTION 3: Arm Stats Table -->
        <div class="bg-white rounded-2xl p-6 border border-slate-200/80 shadow-sm">
            <div class="mb-4">
                <h2 class="text-base font-bold text-slate-900">3. 🧪 Arm Stats & Posterior Parameter Inspection</h2>
                <p class="text-xs text-slate-500">Live Bayesian posterior parameters Beta(α, β) and empirical conversion rates proving active learning.</p>
            </div>

            <div class="overflow-x-auto">
                <table class="w-full text-left text-sm">
                    <thead>
                        <tr class="border-b border-slate-100 text-xs font-semibold uppercase tracking-wider text-slate-400">
                            <th class="pb-3 px-3">Arm #</th>
                            <th class="pb-3 px-3">Nudge Type</th>
                            <th class="pb-3 px-3">Impressions (Pulls)</th>
                            <th class="pb-3 px-3">Acceptances</th>
                            <th class="pb-3 px-3">Empirical Rate</th>
                            <th class="pb-3 px-3">Global Alpha (α)</th>
                            <th class="pb-3 px-3">Global Beta (β)</th>
                            <th class="pb-3 px-3">Posterior Mean E[θ]</th>
                        </tr>
                    </thead>
                    <tbody id="armStatsTableBody" class="divide-y divide-slate-100 font-mono text-xs">
                        <tr>
                            <td colspan="8" class="py-6 text-center text-slate-400">Loading arm statistics...</td>
                        </tr>
                    </tbody>
                </table>
            </div>
        </div>

    </main>

    <!-- Frontend Script -->
    <script>
        let rewardChartInstance = null;
        let regretChartInstance = null;
        let portfolioChartInstance = null;
        let rollingChartInstance = null;

        // Slider sync
        const slider = document.getElementById('tickSlider');
        const sliderValue = document.getElementById('sliderValue');
        slider.addEventListener('input', (e) => {
            sliderValue.innerText = e.target.value;
        });

        // Initialize Charts
        function initCharts() {
            const ctxReward = document.getElementById('rewardChart').getContext('2d');
            rewardChartInstance = new Chart(ctxReward, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [
                        { label: 'Bandit (Hybrid)', data: [], borderColor: '#10B981', backgroundColor: 'rgba(16, 185, 129, 0.1)', borderWidth: 2.5, tension: 0.1, fill: false },
                        { label: 'Random Baseline', data: [], borderColor: '#94A3B8', borderWidth: 2, borderDash: [4, 4], tension: 0.1, fill: false }
                    ]
                },
                options: { responsive: true, maintainAspectRatio: false, scales: { x: { grid: { display: false } } } }
            });

            const ctxRegret = document.getElementById('regretChart').getContext('2d');
            regretChartInstance = new Chart(ctxRegret, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [
                        { label: 'Bandit Regret', data: [], borderColor: '#3B82F6', borderWidth: 2.5, tension: 0.1 },
                        { label: 'Random Regret', data: [], borderColor: '#EF4444', borderWidth: 2, borderDash: [4, 4], tension: 0.1 }
                    ]
                },
                options: { responsive: true, maintainAspectRatio: false, scales: { x: { grid: { display: false } } } }
            });

            const ctxPort = document.getElementById('portfolioChart').getContext('2d');
            portfolioChartInstance = new Chart(ctxPort, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [
                        { label: 'Portfolio Balance ($)', data: [], borderColor: '#059669', backgroundColor: 'rgba(5, 150, 105, 0.1)', borderWidth: 2.5, fill: true },
                        { label: 'Principal Deposits ($)', data: [], borderColor: '#64748B', borderWidth: 2, borderDash: [4, 4] }
                    ]
                },
                options: { responsive: true, maintainAspectRatio: false, scales: { x: { grid: { display: false } } } }
            });

            const ctxRoll = document.getElementById('rollingChart').getContext('2d');
            rollingChartInstance = new Chart(ctxRoll, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [
                        { label: 'Bandit Rolling Acceptance (%)', data: [], borderColor: '#10B981', borderWidth: 2.5 },
                        { label: 'Random Rolling Acceptance (%)', data: [], borderColor: '#94A3B8', borderWidth: 2, borderDash: [4, 4] }
                    ]
                },
                options: { responsive: true, maintainAspectRatio: false, scales: { y: { min: 0, max: 100 } } }
            });
        }

        function switchChartTab(tab) {
            document.getElementById('viewRewards').classList.add('hidden');
            document.getElementById('viewPortfolio').classList.add('hidden');
            document.getElementById('viewAcceptance').classList.add('hidden');

            document.getElementById('tabRewards').className = 'pb-2 text-slate-500 hover:text-slate-700 transition';
            document.getElementById('tabPortfolio').className = 'pb-2 text-slate-500 hover:text-slate-700 transition';
            document.getElementById('tabAcceptance').className = 'pb-2 text-slate-500 hover:text-slate-700 transition';

            if (tab === 'rewards') {
                document.getElementById('viewRewards').classList.remove('hidden');
                document.getElementById('tabRewards').className = 'pb-2 tab-active transition';
            } else if (tab === 'portfolio') {
                document.getElementById('viewPortfolio').classList.remove('hidden');
                document.getElementById('tabPortfolio').className = 'pb-2 tab-active transition';
            } else if (tab === 'acceptance') {
                document.getElementById('viewAcceptance').classList.remove('hidden');
                document.getElementById('tabAcceptance').className = 'pb-2 tab-active transition';
            }
        }

        // Render Dashboard Data
        function renderDashboard(data) {
            // Update KPIs
            const k = data.kpis;
            document.getElementById('kpiBalance').innerText = `$${k.portfolio_balance.toFixed(2)}`;
            document.getElementById('kpiGains').innerText = `${k.market_gain >= 0 ? '+' : ''}$${k.market_gain.toFixed(2)} market gain`;
            document.getElementById('kpiInvested').innerText = `$${k.total_contributions.toFixed(2)}`;
            document.getElementById('kpiAccepts').innerText = `${k.cum_reward_bandit} of ${k.total_ticks} accepted`;
            document.getElementById('kpiBanditRate').innerText = `${k.bandit_acceptance_rate.toFixed(1)}%`;
            document.getElementById('kpiLift').innerText = `${k.acceptance_lift >= 0 ? '+' : ''}${k.acceptance_lift.toFixed(1)}% vs Random`;
            document.getElementById('kpiRandomRate').innerText = `${k.random_acceptance_rate.toFixed(1)}%`;
            document.getElementById('kpiRegret').innerText = `Bandit Regret: ${k.cum_regret_bandit.toFixed(1)} vs ${k.cum_regret_random.toFixed(1)}`;

            // Render Live Feed
            const tbody = document.getElementById('feedTableBody');
            if (!data.live_feed || data.live_feed.length === 0) {
                tbody.innerHTML = `<tr><td colspan="7" class="py-8 text-center text-sm text-slate-400">No transactions simulated yet. Click <b>▶ Run Simulation</b> to begin.</td></tr>`;
            } else {
                tbody.innerHTML = data.live_feed.map(row => {
                    let badge = '';
                    if (row.bandit_arm === 4) {
                        badge = `<span class="px-2 py-0.5 rounded-full text-xs font-medium bg-slate-100 text-slate-600">Control (No Nudge)</span>`;
                    } else if (row.bandit_accepted) {
                        badge = `<span class="px-2 py-0.5 rounded-full text-xs font-semibold bg-emerald-100 text-emerald-800">✓ Accepted</span>`;
                    } else {
                        badge = `<span class="px-2 py-0.5 rounded-full text-xs font-semibold bg-rose-100 text-rose-800">✕ Declined</span>`;
                    }
                    return `
                        <tr class="hover:bg-slate-50/70 transition">
                            <td class="py-3 px-3 font-mono font-bold text-slate-500">#${row.tick}</td>
                            <td class="py-3 px-3 font-medium text-slate-900">${row.merchant} <span class="text-xs text-slate-400">(${row.category})</span></td>
                            <td class="py-3 px-3 font-mono font-semibold">$${row.amount.toFixed(2)}</td>
                            <td class="py-3 px-3 text-xs text-slate-500">${row.day_name} @ ${String(row.hour).padStart(2, '0')}:00</td>
                            <td class="py-3 px-3 text-slate-700 font-medium">${row.bandit_arm_name}</td>
                            <td class="py-3 px-3">${badge}</td>
                            <td class="py-3 px-3 font-mono text-xs ${row.bandit_accepted ? 'font-bold text-emerald-600' : 'text-slate-400'}">${row.bandit_accepted ? `+$${row.invested_amount.toFixed(2)}` : '$0.00'}</td>
                        </tr>
                    `;
                }).join('');
            }

            // Render Arm Stats Table
            const statsBody = document.getElementById('armStatsTableBody');
            statsBody.innerHTML = data.arm_stats.map(s => `
                <tr class="hover:bg-slate-50/70 transition">
                    <td class="py-3 px-3 font-bold text-slate-700">#${s.arm_index}</td>
                    <td class="py-3 px-3 font-sans font-medium text-slate-900">${s.arm_name}</td>
                    <td class="py-3 px-3 font-bold">${s.pulls}</td>
                    <td class="py-3 px-3">${s.acceptances}</td>
                    <td class="py-3 px-3 font-bold ${s.empirical_rate > 0.5 ? 'text-emerald-600' : 'text-slate-700'}">${(s.empirical_rate * 100).toFixed(1)}%</td>
                    <td class="py-3 px-3 text-slate-500">${s.global_alpha.toFixed(1)}</td>
                    <td class="py-3 px-3 text-slate-500">${s.global_beta.toFixed(1)}</td>
                    <td class="py-3 px-3 font-bold text-emerald-700">${s.posterior_mean.toFixed(3)}</td>
                </tr>
            `).join('');

            // Update Charts
            const c = data.charts;
            if (c && c.ticks && c.ticks.length > 0) {
                rewardChartInstance.data.labels = c.ticks;
                rewardChartInstance.data.datasets[0].data = c.reward_bandit;
                rewardChartInstance.data.datasets[1].data = c.reward_random;
                rewardChartInstance.update();

                regretChartInstance.data.labels = c.ticks;
                regretChartInstance.data.datasets[0].data = c.regret_bandit;
                regretChartInstance.data.datasets[1].data = c.regret_random;
                regretChartInstance.update();

                portfolioChartInstance.data.labels = c.ticks;
                portfolioChartInstance.data.datasets[0].data = c.portfolio_balance;
                portfolioChartInstance.data.datasets[1].data = c.total_contributions;
                portfolioChartInstance.update();

                rollingChartInstance.data.labels = c.rolling_ticks;
                rollingChartInstance.data.datasets[0].data = c.rolling_bandit;
                rollingChartInstance.data.datasets[1].data = c.rolling_random;
                rollingChartInstance.update();
            }
        }

        // API Calls
        async function fetchState() {
            try {
                const res = await fetch('/api/state');
                const data = await res.json();
                renderDashboard(data);
            } catch (err) {
                console.error("Fetch state error:", err);
            }
        }

        async function runSimulation() {
            const ticks = parseInt(document.getElementById('tickSlider').value);
            const useHybrid = document.getElementById('hybridToggle').checked;
            const runBtn = document.getElementById('runBtn');
            const runSpinner = document.getElementById('runSpinner');
            const runText = document.getElementById('runText');

            runBtn.disabled = true;
            runSpinner.classList.remove('hidden');
            runText.innerText = "Simulating...";

            try {
                const res = await fetch('/api/simulate', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ ticks: ticks, use_hybrid: useHybrid })
                });
                const data = await res.json();
                renderDashboard(data);
            } catch (err) {
                console.error("Run simulation error:", err);
            } finally {
                runBtn.disabled = false;
                runSpinner.classList.add('hidden');
                runText.innerText = "▶ Run Simulation";
            }
        }

        async function stepOneTick() {
            const useHybrid = document.getElementById('hybridToggle').checked;
            try {
                const res = await fetch('/api/step', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ ticks: 1, use_hybrid: useHybrid })
                });
                const data = await res.json();
                renderDashboard(data);
            } catch (err) {
                console.error("Step error:", err);
            }
        }

        async function resetSimulation() {
            if (!confirm("Are you sure you want to reset the simulation state?")) return;
            try {
                const res = await fetch('/api/reset', { method: 'POST' });
                const data = await res.json();
                renderDashboard(data);
            } catch (err) {
                console.error("Reset error:", err);
            }
        }

        window.addEventListener('DOMContentLoaded', () => {
            initCharts();
            fetchState();
        });
    </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index_page():
    """Serves the complete single-page interactive NudgeVest web app."""
    return HTMLResponse(content=HTML_CONTENT)


# ==============================================================================
# Entry Point
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Run NudgeVest Web Application on localhost.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port to run on (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload on code changes")
    args = parser.parse_args()

    banner = f"""
================================================================================
                    🌱 NUDGEVEST — MICRO-INVESTING WEB APP
================================================================================
  Server running at:  http://localhost:{args.port}
  Network address:    http://{args.host}:{args.port}
================================================================================
    """
    print(banner)

    uvicorn.run("app:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
