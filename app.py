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
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="theme-color" content="#f7f8f5">
  <title>NudgeVest — Saving habits, made visible</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap" rel="stylesheet">
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <style>
    :root{--paper:#f7f8f5;--white:#fff;--ink:#202923;--muted:#68736b;--line:#e3e8e2;--green:#26734d;--green-dark:#1b5b3b;--green-pale:#e6f1e9;--shadow:0 1px 2px rgba(25,45,32,.04)}
    *{box-sizing:border-box}
    body{margin:0;background:var(--paper);color:var(--ink);font-family:"DM Sans",sans-serif;font-size:14px;line-height:1.5;-webkit-font-smoothing:antialiased}
    button,input{font:inherit}button{cursor:pointer}
    button:focus-visible,input:focus-visible{outline:3px solid rgba(38,115,77,.28);outline-offset:2px}
    .topbar{height:68px;background:rgba(255,255,255,.88);border-bottom:1px solid var(--line);display:flex;align-items:center}
    .topbar-inner{width:min(1180px,calc(100% - 48px));margin:auto;display:flex;align-items:center;justify-content:space-between}
    .brand{display:flex;align-items:center;gap:10px;font-family:Manrope,sans-serif;font-size:17px;font-weight:800;letter-spacing:-.04em}
    .brand-mark{width:28px;height:28px;background:var(--green);color:white;border-radius:8px;display:grid;place-items:center;font-size:15px}
    .topmeta{display:flex;align-items:center;gap:10px;color:var(--muted);font-size:12px}
    .status-dot{width:7px;height:7px;border-radius:50%;background:#9ba69d;display:inline-block}
    .wrap{width:min(1180px,calc(100% - 48px));margin:0 auto;padding:38px 0 64px}
    .intro{display:flex;justify-content:space-between;align-items:flex-end;gap:24px;margin-bottom:26px}
    .eyebrow{text-transform:uppercase;letter-spacing:.12em;font-size:10px;font-weight:700;color:var(--green);margin-bottom:8px}
    h1,h2,h3,p{margin-top:0}h1{font-family:Manrope,sans-serif;font-size:clamp(27px,4vw,36px);letter-spacing:-.055em;line-height:1.12;margin-bottom:9px;font-weight:700}
    .intro p{color:var(--muted);font-size:14px;margin-bottom:0}
    .demo-tag{white-space:nowrap;border:1px solid #d6e1d8;background:#f0f5f0;color:var(--green-dark);border-radius:999px;padding:7px 11px;font-size:11px;font-weight:700;letter-spacing:.02em}
    .layout{display:grid;grid-template-columns:minmax(280px,.76fr) minmax(0,1.55fr);gap:16px;align-items:start}
    .panel{background:var(--white);border:1px solid var(--line);border-radius:12px;box-shadow:var(--shadow)}.panel-pad{padding:20px}
    .section-title{font-family:Manrope,sans-serif;font-size:15px;letter-spacing:-.025em;font-weight:700;margin:0}
    .section-note{font-size:12px;color:var(--muted);margin:5px 0 0}
    .controls{position:sticky;top:16px}.control-head{padding-bottom:17px;border-bottom:1px solid var(--line);margin-bottom:17px}
    .field-label{font-size:12px;font-weight:700;display:flex;justify-content:space-between;gap:12px;margin-bottom:8px}
    .field-label span:last-child{color:var(--green);font-variant-numeric:tabular-nums}
    input[type=range]{width:100%;accent-color:var(--green)}.range-ends{display:flex;justify-content:space-between;color:#879188;font-size:10px;margin-top:2px}
    .toggle-row{display:flex;align-items:center;justify-content:space-between;gap:14px;padding:14px 0;border-top:1px solid var(--line);margin-top:16px}
    .toggle-copy strong{font-size:12px;display:block}.toggle-copy small{display:block;font-size:11px;color:var(--muted);margin-top:3px}
    .switch{position:relative;width:36px;height:21px;flex:none}.switch input{opacity:0;width:0;height:0}
    .switch-ui{position:absolute;inset:0;background:#cbd3cc;border-radius:99px;transition:.18s}
    .switch-ui:before{content:"";position:absolute;height:15px;width:15px;left:3px;top:3px;background:white;border-radius:50%;transition:.18s;box-shadow:0 1px 2px #0002}
    .switch input:checked+.switch-ui{background:var(--green)}.switch input:checked+.switch-ui:before{transform:translateX(15px)}
    .actions{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:6px}
    .btn{border:1px solid var(--line);background:white;color:var(--ink);border-radius:8px;min-height:40px;padding:9px 12px;font-weight:700;font-size:12px}
    .btn:hover{background:#f8faf8;border-color:#cbd7cd}.btn-primary{grid-column:1/-1;background:var(--green);border-color:var(--green);color:white;min-height:44px;font-size:13px}
    .btn-primary:hover{background:var(--green-dark);border-color:var(--green-dark)}.btn:disabled{opacity:.6;cursor:wait}
    .reset{background:transparent;border:0;color:var(--muted);font-size:11px;padding:10px 4px;text-decoration:underline;text-underline-offset:3px}
    .control-foot{margin-top:16px;border-top:1px solid var(--line);padding-top:13px;color:var(--muted);font-size:11px}.control-foot strong{color:var(--ink)}
    .metrics{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px}
    .metric{background:var(--white);border:1px solid var(--line);border-radius:11px;padding:17px 18px;min-height:112px;box-shadow:var(--shadow)}
    .metric-label{color:var(--muted);font-size:11px;font-weight:600}
    .metric-value{font-family:Manrope,sans-serif;font-size:25px;letter-spacing:-.05em;font-weight:700;line-height:1.2;margin:10px 0 4px;font-variant-numeric:tabular-nums}
    .metric-sub{font-size:10px;color:var(--muted)}.right-stack{display:grid;gap:14px}
    .panel-heading{display:flex;justify-content:space-between;align-items:flex-start;gap:16px;margin-bottom:17px}
    .chart-tabs{display:flex;gap:4px;padding:3px;border:1px solid var(--line);background:#f8faf8;border-radius:8px;flex:none}
    .chart-tab{background:transparent;border:0;color:var(--muted);font-size:10px;font-weight:700;padding:6px 8px;border-radius:5px}
    .chart-tab.active{background:white;color:var(--green-dark);box-shadow:0 1px 3px #1b36251c}
    .chart-grid{display:grid;grid-template-columns:1fr 1fr;gap:14px}.chart-box h3{font-size:10px;text-transform:uppercase;letter-spacing:.07em;color:var(--muted);margin:0 0 8px}.chart-wrap{height:220px;position:relative}.chart-wrap.tall{height:300px}
    .chart-legend{font-size:11px;color:var(--muted);display:flex;gap:16px;margin-top:12px;align-items:center}
    .legend-dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:6px;background:var(--green)}.legend-dot.gray{background:#9ba69d}
    .activity{margin-top:16px}.table-wrap{overflow:auto}
    table{border-collapse:collapse;width:100%;font-size:11px;text-align:left}
    th{font-weight:700;color:#7b867e;text-transform:uppercase;letter-spacing:.07em;font-size:9px;white-space:nowrap}
    td,th{padding:11px 10px;border-bottom:1px solid #edf0ec}tbody tr:last-child td{border-bottom:0}
    td{white-space:nowrap}.empty{text-align:center;color:var(--muted);white-space:normal;padding:27px!important}
    .merchant{font-weight:700;color:var(--ink)}.subtle{font-size:10px;color:var(--muted)}
    .pill{display:inline-flex;align-items:center;gap:5px;padding:4px 7px;border-radius:99px;font-size:10px;font-weight:700}
    .pill.good{color:var(--green-dark);background:var(--green-pale)}.pill.bad{color:#8e4841;background:#f7ebe9}.pill.neutral{color:#56615a;background:#eff1ef}
    details{border-top:1px solid var(--line);margin-top:14px;padding-top:14px}summary{cursor:pointer;font-size:12px;font-weight:700;color:var(--ink)}
    summary::marker{color:var(--green)}.advanced-note{font-size:11px;color:var(--muted);margin:8px 0 14px}
    .footer-note{color:#879188;font-size:10px;margin:17px 2px 0}
    .sr-only{position:absolute;width:1px;height:1px;padding:0;margin:-1px;overflow:hidden;clip:rect(0,0,0,0);white-space:nowrap;border:0}
    @media(max-width:850px){.layout{grid-template-columns:1fr}.controls{position:static}}
    @media(max-width:600px){.topbar{height:58px}.topbar-inner,.wrap{width:calc(100% - 28px)}.wrap{padding:27px 0 42px}.intro{display:block;margin-bottom:20px}.demo-tag{display:inline-block;margin-top:14px}.metrics{grid-template-columns:1fr 1fr}.metric{padding:14px;min-height:100px}.metric:last-child{grid-column:1/-1}.metric-value{font-size:22px}.panel-pad{padding:16px}.panel-heading{display:block}.chart-tabs{margin-top:13px;width:max-content;max-width:100%}.chart-tab{padding:6px 7px}.chart-grid{grid-template-columns:1fr}.chart-wrap{height:220px}.chart-wrap.tall{height:250px}.topmeta{font-size:10px}}
    @media(prefers-reduced-motion:reduce){*,*::before,*::after{scroll-behavior:auto!important;transition:none!important}}
  </style>
</head>
<body>
  <header class="topbar"><div class="topbar-inner">
    <div class="brand"><span class="brand-mark" aria-hidden="true">n</span><span>NudgeVest</span></div>
    <div class="topmeta"><span class="status-dot" aria-hidden="true"></span><span>Interactive product demo</span></div>
  </div></header>
  <main class="wrap">
    <section class="intro" aria-labelledby="pageTitle">
      <div><div class="eyebrow">Behavioral investing · simulator</div><h1 id="pageTitle">Small choices, made visible.</h1><p>Explore how timely saving prompts can shape a simulated investing habit.</p></div>
      <span class="demo-tag">SIMULATED DATA · NO REAL MONEY</span>
    </section>
    <div class="layout">
      <aside class="panel panel-pad controls" aria-labelledby="controlsTitle">
        <div class="control-head"><h2 id="controlsTitle" class="section-title">Run a simulation</h2><p class="section-note">Choose how many everyday transactions to model.</p></div>
        <label class="field-label" for="tickSlider"><span>Transactions</span><span><b id="sliderValue">50</b> events</span></label>
        <input id="tickSlider" type="range" min="10" max="250" value="50" step="10" aria-label="Number of transactions to simulate">
        <div class="range-ends"><span>10</span><span>250</span></div>
        <div class="toggle-row"><div class="toggle-copy"><strong>Adaptive recommendations</strong><small>Compare contextual learning with random selection</small></div><label class="switch"><input id="hybridToggle" type="checkbox" checked aria-label="Use adaptive recommendations"><span class="switch-ui"></span></label></div>
        <div class="actions">
          <button id="runBtn" class="btn btn-primary" type="button" onclick="runSimulation()"><span id="runSpinner" class="sr-only">Working. </span><span id="runText">Run simulation</span></button>
          <button class="btn" type="button" onclick="stepOneTick()">One step</button>
          <button class="btn" type="button" onclick="switchChartTab('portfolio');document.getElementById('analyticsTitle').scrollIntoView({behavior:'smooth',block:'start'})">View portfolio</button>
        </div>
        <div style="text-align:right"><button class="reset" type="button" onclick="resetSimulation()">Reset this demo</button></div>
        <div class="control-foot"><strong>What you’re seeing</strong><br>Sample transactions, simulated choices and illustrative market movement. Nothing here is connected to an account or a real investment.</div>
      </aside>
      <section class="right-stack" aria-label="Simulation results">
        <div class="metrics" aria-live="polite" aria-atomic="true">
          <article class="metric"><div class="metric-label">Simulated portfolio</div><div id="kpiBalance" class="metric-value">$0.00</div><div id="kpiGains" class="metric-sub">+$0.00 simulated change</div></article>
          <article class="metric"><div class="metric-label">Contributions in demo</div><div id="kpiInvested" class="metric-value">$0.00</div><div class="metric-sub">From accepted simulated nudges</div></article>
          <article class="metric"><div class="metric-label">Prompts accepted</div><div id="kpiAccepts" class="metric-value">0 of 0</div><div id="kpiBanditRate" class="metric-sub">0% acceptance rate</div></article>
        </div>
        <section class="panel panel-pad" aria-labelledby="analyticsTitle">
          <div class="panel-heading"><div><h2 id="analyticsTitle" class="section-title">How the approach learns</h2><p class="section-note">Compare the adaptive strategy with a random baseline.</p></div>
            <div class="chart-tabs" role="tablist" aria-label="Choose a chart">
              <button id="tabRewards" class="chart-tab active" type="button" role="tab" aria-selected="true" onclick="switchChartTab('rewards')">Learning</button>
              <button id="tabPortfolio" class="chart-tab" type="button" role="tab" aria-selected="false" onclick="switchChartTab('portfolio')">Portfolio</button>
              <button id="tabAcceptance" class="chart-tab" type="button" role="tab" aria-selected="false" onclick="switchChartTab('acceptance')">Acceptance</button>
            </div>
          </div>
          <div id="viewRewards" class="chart-grid"><section class="chart-box" aria-label="Accepted prompt comparison"><h3>Accepted prompts</h3><div class="chart-wrap"><canvas id="rewardChart" role="img" aria-label="Cumulative accepted prompts for adaptive and random strategies"></canvas></div></section><section class="chart-box" aria-label="Regret comparison"><h3>Cumulative regret</h3><div class="chart-wrap"><canvas id="regretChart" role="img" aria-label="Cumulative regret for adaptive and random strategies"></canvas></div></section></div>
          <div id="viewPortfolio" class="chart-wrap tall" hidden><canvas id="portfolioChart" role="img" aria-label="Illustrative simulated portfolio balance and contributions"></canvas></div>
          <div id="viewAcceptance" class="chart-wrap tall" hidden><canvas id="rollingChart" role="img" aria-label="Rolling acceptance rates for adaptive and random strategies"></canvas></div>
          <div class="chart-legend"><span><i class="legend-dot"></i>Adaptive</span><span><i class="legend-dot gray"></i>Random baseline</span><span id="kpiLift" class="subtle">0.0% difference</span></div>
          <div id="kpiRandomRate" class="sr-only">0% random baseline rate</div><div id="kpiRegret" class="sr-only">Regret details</div>
        </section>
      </section>
    </div>
    <section class="panel panel-pad activity" aria-labelledby="activityTitle">
      <div class="panel-heading"><div><h2 id="activityTitle" class="section-title">Recent simulated activity</h2><p class="section-note">A clear view of each prompt, response and illustrative contribution.</p></div><span class="subtle" id="activityCount">Latest 10 events</span></div>
      <div class="table-wrap"><table><thead><tr><th scope="col">Event</th><th scope="col">Purchase</th><th scope="col">Amount</th><th scope="col">Prompt</th><th scope="col">Outcome</th><th scope="col">Demo contribution</th></tr></thead><tbody id="feedTableBody"><tr><td colspan="6" class="empty">Choose “Run simulation” to see how the model responds to everyday spending.</td></tr></tbody></table></div>
      <details><summary>Explore strategy performance</summary><p class="advanced-note">Research view: simulated comparisons and posterior estimates. These measures describe this run only and do not predict real-world outcomes.</p>
        <div class="table-wrap"><table><thead><tr><th scope="col">Strategy</th><th scope="col">Selections</th><th scope="col">Acceptances</th><th scope="col">Observed rate</th><th scope="col">Alpha</th><th scope="col">Beta</th><th scope="col">Posterior mean</th></tr></thead><tbody id="armStatsTableBody"><tr><td colspan="7" class="empty">Run the simulation to view strategy statistics.</td></tr></tbody></table></div>
      </details>
    </section>
    <p class="footer-note">NudgeVest is an educational simulation. Portfolio values and market movement are generated for demonstration and are not financial advice or actual investment performance.</p>
  </main>
  <script>
    let rewardChartInstance=null,regretChartInstance=null,portfolioChartInstance=null,rollingChartInstance=null;
    const slider=document.getElementById("tickSlider");
    slider.addEventListener("input",function(){document.getElementById("sliderValue").textContent=slider.value;});
    function switchChartTab(tab){
      const views={rewards:"viewRewards",portfolio:"viewPortfolio",acceptance:"viewAcceptance"};
      const buttons={rewards:"tabRewards",portfolio:"tabPortfolio",acceptance:"tabAcceptance"};
      Object.keys(views).forEach(function(key){document.getElementById(views[key]).hidden=key!==tab;const b=document.getElementById(buttons[key]);b.classList.toggle("active",key===tab);b.setAttribute("aria-selected",key===tab?"true":"false");});
      if(tab==="portfolio"&&portfolioChartInstance)portfolioChartInstance.resize();
      if(tab==="acceptance"&&rollingChartInstance)rollingChartInstance.resize();
      if(tab==="rewards"&&rewardChartInstance){rewardChartInstance.resize();regretChartInstance.resize();}
    }
    function makeLineChart(id,sets,format){
      return new Chart(document.getElementById(id).getContext("2d"),{type:"line",data:{labels:[],datasets:sets.map(function(s){return{label:s.label,data:[],borderColor:s.color,backgroundColor:s.fill||"transparent",borderWidth:2,borderDash:s.dash||[],pointRadius:0,pointHoverRadius:4,tension:.28,fill:!!s.fill};})},options:{responsive:true,maintainAspectRatio:false,interaction:{intersect:false,mode:"index"},plugins:{legend:{display:false},tooltip:{backgroundColor:"#202923",padding:10,titleFont:{family:"DM Sans"},bodyFont:{family:"DM Sans"}}},scales:{x:{grid:{display:false},ticks:{maxTicksLimit:7,color:"#7b867e",font:{size:10}}},y:{beginAtZero:true,grid:{color:"#edf0ec"},ticks:{color:"#7b867e",font:{size:10},callback:format||undefined}}}}});
    }
    function initCharts(){
      rewardChartInstance=makeLineChart("rewardChart",[{label:"Adaptive",color:"#26734d"},{label:"Random baseline",color:"#a8b0a9",dash:[5,4]}]);
      regretChartInstance=makeLineChart("regretChart",[{label:"Adaptive regret",color:"#26734d"},{label:"Random regret",color:"#a8b0a9",dash:[5,4]}]);
      portfolioChartInstance=makeLineChart("portfolioChart",[{label:"Simulated balance",color:"#26734d",fill:"rgba(38,115,77,.08)"},{label:"Contributions",color:"#c89543",dash:[5,4]}],function(v){return "$"+Number(v).toFixed(0);});
      rollingChartInstance=makeLineChart("rollingChart",[{label:"Adaptive",color:"#26734d"},{label:"Random baseline",color:"#a8b0a9",dash:[5,4]}],function(v){return v+"%";});
    }
    function escapeHTML(value){return String(value==null?"":value).replace(/[&<>"']/g,function(c){return{"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c];});}
    function money(v){return "$"+Number(v||0).toFixed(2);}
    function renderDashboard(data){
      const k=data.kpis||{},c=data.charts||{};
      document.getElementById("kpiBalance").textContent=money(k.portfolio_balance);
      const gain=Number(k.market_gain||0);
      document.getElementById("kpiGains").textContent=(gain>=0?"+":"−")+money(Math.abs(gain))+" simulated change";
      document.getElementById("kpiInvested").textContent=money(k.total_contributions);
      document.getElementById("kpiAccepts").textContent=String(k.cum_reward_bandit||0)+" of "+String(k.total_ticks||0);
      document.getElementById("kpiBanditRate").textContent=Number(k.bandit_acceptance_rate||0).toFixed(1)+"% adaptive acceptance";
      document.getElementById("kpiLift").textContent=(Number(k.acceptance_lift||0)>=0?"+":"")+Number(k.acceptance_lift||0).toFixed(1)+" pts vs random";
      document.getElementById("kpiRandomRate").textContent=Number(k.random_acceptance_rate||0).toFixed(1)+"% random baseline rate";
      document.getElementById("kpiRegret").textContent="Adaptive regret "+Number(k.cum_regret_bandit||0).toFixed(1)+" vs random "+Number(k.cum_regret_random||0).toFixed(1);
      const feed=document.getElementById("feedTableBody"),rows=data.live_feed||[];
      document.getElementById("activityCount").textContent=rows.length?("Showing "+rows.length+" most recent events"):"Latest 10 events";
      if(!rows.length){feed.innerHTML='<tr><td colspan="6" class="empty">Choose “Run simulation” to see how the model responds to everyday spending.</td></tr>';}
      else{feed.innerHTML=rows.map(function(row){
        const accepted=!!row.bandit_accepted,control=Number(row.bandit_arm)===4;
        const pill=control?'<span class="pill neutral">No prompt</span>':(accepted?'<span class="pill good">Accepted</span>':'<span class="pill bad">Declined</span>');
        return '<tr><td class="subtle">#'+escapeHTML(row.tick)+'</td><td><span class="merchant">'+escapeHTML(row.merchant)+'</span><br><span class="subtle">'+escapeHTML(row.category)+' · '+escapeHTML(row.day_name)+'</span></td><td>'+money(row.amount)+'</td><td>'+escapeHTML(row.bandit_arm_name)+'</td><td>'+pill+'</td><td>'+(accepted?money(row.invested_amount):"$0.00")+'</td></tr>';
      }).join("");}
      const stats=document.getElementById("armStatsTableBody"),arms=data.arm_stats||[];
      if(arms.length){stats.innerHTML=arms.map(function(s){return '<tr><td>'+escapeHTML(s.arm_name)+'</td><td>'+escapeHTML(s.pulls)+'</td><td>'+escapeHTML(s.acceptances)+'</td><td>'+Number(s.empirical_rate*100).toFixed(1)+'%</td><td>'+Number(s.global_alpha).toFixed(1)+'</td><td>'+Number(s.global_beta).toFixed(1)+'</td><td>'+Number(s.posterior_mean).toFixed(3)+'</td></tr>';}).join("");}
      if(c.ticks&&c.ticks.length){
        rewardChartInstance.data.labels=c.ticks;rewardChartInstance.data.datasets[0].data=c.reward_bandit;rewardChartInstance.data.datasets[1].data=c.reward_random;rewardChartInstance.update("none");
        regretChartInstance.data.labels=c.ticks;regretChartInstance.data.datasets[0].data=c.regret_bandit;regretChartInstance.data.datasets[1].data=c.regret_random;regretChartInstance.update("none");
        portfolioChartInstance.data.labels=c.ticks;portfolioChartInstance.data.datasets[0].data=c.portfolio_balance;portfolioChartInstance.data.datasets[1].data=c.total_contributions;portfolioChartInstance.update("none");
        rollingChartInstance.data.labels=c.rolling_ticks;rollingChartInstance.data.datasets[0].data=c.rolling_bandit;rollingChartInstance.data.datasets[1].data=c.rolling_random;rollingChartInstance.update("none");
      }
    }
    async function apiCall(path,options){
      const res=await fetch(path,options||{});
      if(!res.ok)throw new Error("Request failed ("+res.status+")");
      return res.json();
    }
    async function fetchState(){try{renderDashboard(await apiCall("/api/state"));}catch(e){console.error("Could not load simulation state",e);}}
    async function runSimulation(){
      const btn=document.getElementById("runBtn"),text=document.getElementById("runText"),hybrid=document.getElementById("hybridToggle").checked;
      btn.disabled=true;text.textContent="Running simulation…";
      try{renderDashboard(await apiCall("/api/simulate",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({ticks:Number(slider.value),use_hybrid:hybrid})}));}
      catch(e){text.textContent="Could not run — try again";console.error(e);}
      finally{btn.disabled=false;if(text.textContent==="Running simulation…")text.textContent="Run simulation";else setTimeout(function(){text.textContent="Run simulation";},2200);}
    }
    async function stepOneTick(){
      const btn=document.querySelector('button[onclick="stepOneTick()"]');btn.disabled=true;
      try{renderDashboard(await apiCall("/api/step",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({ticks:1,use_hybrid:document.getElementById("hybridToggle").checked})}));}
      catch(e){console.error("Could not add a simulation step",e);}
      finally{btn.disabled=false;}
    }
    async function resetSimulation(){
      if(!window.confirm("Reset this demo and clear its simulated history?"))return;
      try{renderDashboard(await apiCall("/api/reset",{method:"POST"}));}
      catch(e){console.error("Could not reset simulation",e);}
    }
    window.addEventListener("DOMContentLoaded",function(){initCharts();fetchState();});
  </script>
</body>
</html>"""


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
