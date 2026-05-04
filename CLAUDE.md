# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research framework for comparing Reinforcement Learning approaches to optimal market making against closed-form analytic solutions (Avellaneda-Stoikov) and heuristic baselines. The core research question: do RL agents outperform classical optimal control across different market conditions?

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Common Commands

```bash
# Train all 6 RL agents across all 12 environments
python scripts/train_all_rl_agents.py

# Compare all 17 agents (requires trained models)
python scripts/compare_all_agents.py

# Aggregate results into summary tables
python scripts/aggregate_results.py

# Full pipeline end-to-end
python scripts/run_full_pipeline.py --full

# Generate reports
python scripts/create_evaluation_report.py      # Statistics & rankings
python scripts/create_visualization_report.py  # Heatmaps, HTML plots
python scripts/create_results_summary.py       # Research paper-style tables
```

No formal test suite or linting configuration exists in this repo.

## Architecture

### Core Abstractions

**Environments** (`envs/`) — Gymnasium environments, all inheriting from `base_env.py`:
- State: `[normalized_time, normalized_mid_price, price_change, normalized_inventory]`
- Action: `[spread_factor, skew_factor]` ∈ [-1, 1]²
- Reward: `ΔPnL - inv_penalty × q² × dt`
- Fill model: Poisson intensity λ(δ) = A·exp(-k·δ) — fills decrease exponentially with spread
- Child classes only need to implement `_update_price()` — everything else is in the base

**Agents** (`agents/`) — All implement `act(obs, info) → action` via `BaseAgent`:
- RL agents additionally implement `train()`, `save()`, `load()` (Stable-Baselines3 wrappers)
- Non-RL agents are stateless

**Experiment Runner** (`experiments/runner.py`) — Central orchestrator:
- `run_experiment()` handles training and evaluation uniformly for both RL and non-RL agents
- Saves models to `models/{EnvName}/{AgentName}/model.zip + metadata.json`
- Saves results to `results/{EnvName}/{AgentName}/metrics.json, pnls.npy, inventory.npy`

### 12 Environments (3 processes × 4 variants)

| Process | Variants |
|---------|---------|
| ABM (Arithmetic Brownian Motion) | vanilla, jump, regime, jump+regime |
| GBM (Geometric Brownian Motion) | vanilla, jump, regime, jump+regime |
| OU (Ornstein-Uhlenbeck) | vanilla, jump, regime, jump+regime |

Jumps are additive for ABM/OU, multiplicative (log-normal) for GBM. Regime-switching uses Markov switching between low/high volatility states.

### 17 Agents

- **6 RL agents**: PPO, DeepPPO, LSTMPPO, SAC, TD3, LSTMDeepSAC
- **2 analytic**: ASClosedFormAgent (Avellaneda-Stoikov 2008 exact solution), ASSimpleHeuristicAgent
- **9 heuristics**: FixedSpread, InventoryShift, InventorySpreadScaler, MidPriceFollow, LastLook, MarketOrderOnly, NoiseTraderNormal, NoiseTraderUniform, ZeroIntelligence

### Configuration System

All parameters live in `configs/`:
- `env_configs.yaml` — environment presets for all 12 environments
- `agent_configs.yaml` — hyperparameter presets for all 17 agents
- `training_configs.yaml` — training/evaluation settings

Load with `configs/config_loader.py`.

### Key Scale Difference to Be Aware Of

GBM volatility uses percentage scale (e.g., 0.02) while ABM/OU use absolute scale (e.g., 2.0). These are only comparable when S0 ≈ 100. GBM jumps are multiplicative vs additive for ABM/OU — appropriate for each process type but magnitudes aren't directly comparable.

## Running a Single Experiment Programmatically

```python
from configs.config_loader import load_config
from envs.abm_vanilla import ABMVanillaEnv
from agents.ppo_agent import PPOAgent
from experiments.runner import run_experiment

env_cfg = load_config("configs/env_configs.yaml")["abm_vanilla"]
agent_cfg = load_config("configs/agent_configs.yaml")["ppo_basic"]

run_experiment(
    env_class=ABMVanillaEnv,
    agent_class=PPOAgent,
    env_config=env_cfg,
    agent_config=agent_cfg,
    train=True,
    n_eval_episodes=100,
    save_model=True
)
```

## Metrics Computed

PnL mean/std, Sharpe ratio, VaR (95%, 99%), Expected Shortfall (95%, 99%) — all implemented in `experiments/metrics.py`.

Early stopping with best model restoration is available via `experiments/callbacks.py`.
