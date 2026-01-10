# Market Making with Reinforcement Learning

A comprehensive research framework for evaluating reinforcement learning agents in market-making environments. This project implements various RL algorithms (PPO, SAC, TD3) and heuristic strategies (Avellaneda-Stoikov, fixed spread, inventory-based) across multiple synthetic market environments.

## Overview

This project provides:
- **12 Synthetic Market Environments**: ABM, GBM, and OU processes with vanilla, jump, regime-switching, and combined variants
- **RL Agents**: PPO, DeepPPO, LSTM-PPO, SAC, TD3, and LSTM-SAC implementations
- **Heuristic Agents**: Avellaneda-Stoikov closed-form, fixed spread, inventory-based strategies, and more
- **Advanced Training Features**: Reward-based early stopping with automatic best weight restoration
- **Comprehensive Evaluation**: Automated comparison pipelines with metrics and visualization

## Installation

### Requirements

- Python 3.10+
- PyTorch 1.12+
- Stable-Baselines3 2.0+
- Gymnasium 0.28+

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd market_making_rl
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### Basic Example: Train and Evaluate PPO Agent

```python
from configs.config_loader import load_config
from envs.abm_vanilla import ABMVanillaEnv
from agents.ppo_agent import PPOAgent
from experiments.runner import run_experiment

# Load configurations
env_cfg = load_config("configs/env_configs.yaml")["abm_vanilla"]
agent_cfg = load_config("configs/agent_configs.yaml")["ppo_basic"]

# Run experiment
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

### Run Full Pipeline

Train all RL agents and compare with heuristics:

```bash
# Train all RL agents (may take several hours)
python scripts/train_all_rl_agents.py

# Compare all agents across environments
python scripts/compare_all_agents.py

# Aggregate results into comparison tables
python scripts/aggregate_results.py

# Or run the full pipeline at once:
python scripts/run_full_pipeline.py --full
```

## Report Generation

After running experiments, you can generate comprehensive reports and visualizations using the following scripts:

### Evaluation Report

Generate a detailed evaluation report with statistics and rankings:

```bash
python scripts/create_evaluation_report.py
```

**Output:** `results/EVALUATION_REPORT.md`

This report includes:
- Overall performance statistics
- Best agent per environment (by PnL and Sharpe ratio)
- Performance by agent category (RL, Analytic, Heuristic)
- Performance by environment type (ABM, GBM, OU)
- Risk analysis (VaR, ES)
- Inventory management statistics

### Appendix

Generate appendix tables in matrix format:

```bash
python scripts/create_appendix.py
```

**Output:** `results/APPENDIX.md`

This creates comprehensive matrices showing:
- All metrics (Mean PnL, Sharpe Ratio, Standard Deviation, VaR, ES, Average Inventory)
- Environments as rows, agents as columns
- Agents grouped by category (RL, Analytic, Heuristic)
- Category averages for each metric

### Visualization Report

Generate comprehensive visualizations with confidence intervals:

```bash
python scripts/create_visualization_report.py
```

**Output:** 
- `results/VISUALIZATION_REPORT.md` - Markdown report with embedded figures
- `results/VISUALIZATION_REPORT.html` - Interactive HTML report
- `results/figures/` - All visualization figures

This report includes:
- Heatmaps for all metrics across agent-environment combinations
- Category comparison charts with 95% confidence intervals
- Risk-return scatter plots
- Agent rankings with error bars
- PnL distribution plots (violin/box plots)
- Best agent visualizations
- Radar charts for multi-metric comparison
- Environment difficulty analysis
- Agent consistency analysis

**Options:**
```bash
# Customize confidence level and bootstrap iterations
python scripts/create_visualization_report.py --confidence 0.99 --n-bootstrap 2000

# Generate only markdown or HTML
python scripts/create_visualization_report.py --format markdown
python scripts/create_visualization_report.py --format html
```

### Results and Conclusions Summary

Generate a research paper-style results and conclusions section:

```bash
python scripts/create_results_summary.py
```

**Output:** `results/RESULTS_AND_CONCLUSIONS.md`

This comprehensive report provides:
- Executive summary with key findings
- Statistical comparisons by agent category
- Individual agent performance analysis
- Environment complexity analysis
- Risk-return trade-off analysis
- Statistical patterns and insights
- Critical discussion (RL vs traditional methods, LSTM effectiveness, etc.)
- Conclusions with research contributions and practical implications

### Agent Risk-Return Profiles

Generate individual risk-return plots for each agent:

```bash
python scripts/create_agent_risk_return_plots.py
```

**Output:**
- `results/AGENT_RISK_RETURN_PROFILES.md` - Markdown file with all plots
- `results/figures/agent_risk_return_*.png` - Individual agent plots

Each plot shows:
- Return standard deviation (risk) on x-axis
- Mean PnL (return) on y-axis
- Points color-coded by environment type (ABM=Blue, GBM=Purple, OU=Orange)
- Environment names labeled on each point
- Legend showing environment types

### PnL Distribution Plots

Generate PnL distribution plots for each agent-environment combination:

```bash
python scripts/create_pnl_distributions.py
```

**Output:**
- `results/PNL_DISTRIBUTIONS.md` - Markdown file with all plots organized by environment
- `results/figures/pnl_dist_*.png` - Individual distribution plots

Each plot shows:
- Histogram with KDE overlay of PnL distribution
- **Mean PnL** highlighted with red dashed vertical line
- **Median** highlighted with green dashed vertical line
- **Interquartile Range (IQR)** highlighted with yellow shaded region
- Statistics box showing mean, median, IQR, standard deviation, and sample size

### Quick Report Generation

Generate all reports at once:

```bash
# Generate all reports
python scripts/create_evaluation_report.py
python scripts/create_appendix.py
python scripts/create_visualization_report.py
python scripts/create_results_summary.py
python scripts/create_agent_risk_return_plots.py
python scripts/create_pnl_distributions.py
```

All reports are saved to the `results/` directory and can be directly included in research papers or theses.

## Project Structure

```
market_making_rl/
├── agents/              # Agent implementations (RL + heuristics)
│   ├── ppo_agent.py    # PPO agent
│   ├── lstm_agent.py   # LSTM-PPO agent
│   ├── sac_agent.py    # SAC agent
│   ├── td3_agent.py    # TD3 agent
│   ├── as_agent.py     # Avellaneda-Stoikov agent
│   └── ...             # Other heuristic agents
│
├── envs/               # Market environments
│   ├── base_env.py     # Base environment class
│   ├── abm_vanilla.py  # Arithmetic Brownian Motion
│   ├── gbm_vanilla.py  # Geometric Brownian Motion
│   ├── ou_vanilla.py   # Ornstein-Uhlenbeck
│   └── ...             # Jump and regime-switching variants
│
├── experiments/        # Experiment utilities
│   ├── runner.py       # Main experiment runner
│   ├── callbacks.py    # Early stopping callbacks
│   ├── metrics.py      # Performance metrics
│   └── plotting.py     # Visualization utilities
│
├── configs/            # Configuration files
│   ├── env_configs.yaml      # Environment presets
│   ├── agent_configs.yaml    # Agent hyperparameters
│   ├── training_configs.yaml # Training settings
│   └── README.md       # Detailed config guide
│
├── scripts/            # Automation scripts
│   ├── train_all_rl_agents.py    # Train all RL agents
│   ├── compare_all_agents.py     # Compare all agents
│   ├── aggregate_results.py      # Aggregate results
│   └── run_full_pipeline.py      # Full pipeline runner
│
├── models/             # Saved trained models
├── results/            # Experiment results
└── data/               # Data directory
```

## Key Features

### 1. Reward-Based Early Stopping

The framework includes advanced early stopping that monitors evaluation performance rather than training loss:

- **Monitors**: Mean reward, mean PnL, or custom metrics
- **Automatic Best Weight Restoration**: Saves and restores best model weights automatically
- **Flexible Configuration**: Choose between reward-based or loss-based monitoring

Example configuration:
```yaml
early_stopping:
  enabled: true
  monitor_type: "reward"      # or "loss"
  monitor: "mean_reward"      # or "mean_pnl"
  patience: 6                 # evaluations without improvement
  eval_freq: 10000            # evaluate every N steps
  n_eval_episodes: 10         # episodes per evaluation
```

### 2. Comprehensive Agent Suite

**Reinforcement Learning Agents:**
- PPO (Proximal Policy Optimization)
- DeepPPO (PPO with deep networks)
- LSTM-PPO (Recurrent PPO for temporal dependencies)
- SAC (Soft Actor-Critic)
- TD3 (Twin Delayed DDPG)
- LSTM-SAC (Deep SAC for pattern recognition)

**Heuristic Agents:**
- Avellaneda-Stoikov closed-form optimal
- Fixed spread strategies
- Inventory-based quote shifting
- Trend-following strategies
- Noise traders (for comparison)

### 3. Multiple Market Models

**Price Processes:**
- **ABM** (Arithmetic Brownian Motion): `dS = σ dW`
- **GBM** (Geometric Brownian Motion): `dS = μS dt + σS dW`
- **OU** (Ornstein-Uhlenbeck): `dS = κ(μ - S) dt + σ dW`

**Variants:**
- Vanilla (basic diffusion)
- Jump diffusion (with Poisson jumps)
- Regime-switching (Markov volatility regimes)
- Jump + Regime (combined)

### 4. Automated Evaluation Pipeline

The framework provides scripts for:
- Training all RL agents across environments
- Comparing RL agents with heuristics
- Aggregating results into comparison tables
- Generating visualizations and reports

## Configuration

All experiments are configured via YAML files for reproducibility. See [`configs/README.md`](configs/README.md) for detailed documentation.

**Quick Configuration Guide:**
- **Environment configs**: `configs/env_configs.yaml`
- **Agent configs**: `configs/agent_configs.yaml`
- **Training configs**: `configs/training_configs.yaml`

## Usage Examples

### Example 1: Compare Agents on Single Environment

```python
from configs.config_loader import load_config
from envs.abm_vanilla import ABMVanillaEnv
from agents.ppo_agent import PPOAgent
from agents.as_agent import ASClosedFormAgent
from experiments.runner import run_experiment

env_cfg = load_config("configs/env_configs.yaml")["abm_vanilla"]

# Compare PPO vs AS
for agent_class, agent_key in [(PPOAgent, "ppo_basic"), 
                                (ASClosedFormAgent, "as_closed_form")]:
    agent_cfg = load_config("configs/agent_configs.yaml")[agent_key]
    run_experiment(
        env_class=ABMVanillaEnv,
        agent_class=agent_class,
        env_config=env_cfg,
        agent_config=agent_cfg,
        train=(agent_class == PPOAgent),
        n_eval_episodes=100
    )
```

### Example 2: Test Different Environments

```python
from configs.config_loader import load_config
from envs import ABMVanillaEnv, OUVanillaEnv, GBMVanillaEnv
from agents.as_agent import ASClosedFormAgent
from experiments.runner import run_experiment

envs = [
    (ABMVanillaEnv, "abm_vanilla"),
    (OUVanillaEnv, "ou_vanilla"),
    (GBMVanillaEnv, "gbm_vanilla"),
]

agent_cfg = load_config("configs/agent_configs.yaml")["as_closed_form"]

for env_class, env_key in envs:
    env_cfg = load_config("configs/env_configs.yaml")[env_key]
    if env_key == "gbm_vanilla":
        agent_cfg = load_config("configs/agent_configs.yaml")["as_closed_form_gbm"]
    
    run_experiment(
        env_class=env_class,
        agent_class=ASClosedFormAgent,
        env_config=env_cfg,
        agent_config=agent_cfg,
        train=False,
        n_eval_episodes=100
    )
```

### Example 3: Custom Configuration

```python
from envs.abm_vanilla import ABMVanillaEnv
from agents.ppo_agent import PPOAgent
from experiments.runner import run_experiment

# Custom environment config
env_cfg = {
    "S0": 100.0,
    "T": 1.0,
    "dt": 0.0001,
    "sigma": 2.0,
    "A": 5.0,
    "k": 1.5,
    "base_delta": 1.0,
    "max_inventory": 20,
    "inv_penalty": 0.01,
    "seed": 42
}

# Custom agent config with early stopping
agent_cfg = {
    "total_timesteps": 200000,
    "learning_rate": 0.0003,
    "n_steps": 1024,
    "batch_size": 256,
    "early_stopping": {
        "enabled": True,
        "monitor_type": "reward",
        "monitor": "mean_reward",
        "patience": 10,
        "eval_freq": 10000,
        "n_eval_episodes": 10
    }
}

run_experiment(
    env_class=ABMVanillaEnv,
    agent_class=PPOAgent,
    env_config=env_cfg,
    agent_config=agent_cfg,
    train=True,
    n_eval_episodes=100
)
```

## Scripts

See [`scripts/README.md`](scripts/README.md) for detailed documentation on automation scripts.

**Main Scripts:**
- `scripts/train_all_rl_agents.py` - Train all RL agents
- `scripts/compare_all_agents.py` - Compare all agents
- `scripts/aggregate_results.py` - Aggregate results
- `scripts/run_full_pipeline.py` - Run full pipeline

## Evaluation Metrics

The framework computes comprehensive performance metrics:

- **Return Metrics**: Mean PnL, Sharpe ratio
- **Risk Metrics**: Standard deviation, VaR (95%, 99%), Expected Shortfall
- **Inventory Metrics**: Average inventory level

Results are saved as JSON files and can be aggregated for comparison.

## Advanced Features

### Custom Callbacks

Implement custom training callbacks by extending `BaseCallback`:

```python
from stable_baselines3.common.callbacks import BaseCallback

class CustomCallback(BaseCallback):
    def _on_step(self) -> bool:
        # Your custom logic
        return True  # Continue training
```

### Custom Metrics

Add custom evaluation metrics by modifying `experiments/metrics.py`.

### Environment Customization

Create new market environments by extending `MarketMakingBaseEnv`:

```python
from envs.base_env import MarketMakingBaseEnv

class CustomEnv(MarketMakingBaseEnv):
    def _update_price(self):
        # Implement your price dynamics
        pass
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Make sure you're running from the project root directory
2. **Model Not Found**: Ensure training completed successfully before comparison
3. **Memory Issues**: Reduce `total_timesteps` or batch sizes in configs
4. **Monitor Warning**: Fixed automatically - evaluation envs are wrapped with Monitor

### Getting Help

- Check `configs/README.md` for configuration details
- Check `scripts/README.md` for script usage
- Review example code in this README
- Check `ENVIRONMENT_EVALUATION.md` for environment details

## Performance Notes

- **Training Time**: Each RL agent training takes 10-30 minutes (depending on `total_timesteps`)
- **Evaluation Time**: Much faster (~seconds per agent)
- **Model Size**: Saved models are typically 1-10 MB each
- **Results Size**: Each experiment result is ~100 KB

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{market_making_rl,
  title={Reinforcement Learning vs. Stochastic Control Approaches to Optimal Market Making: A
Controlled Simulation Study},
  author={Cem Vural},
  year={2026},
  url={https://github.com/cemvural00/market_making_rl}
}
```

## License

[All rights reserved]
Feel free to contact via e-mail: cemvural2000@icloud.com

## Acknowledgments

- Stable-Baselines3 team for the RL framework
- Avellaneda & Stoikov for the optimal market-making formulation
