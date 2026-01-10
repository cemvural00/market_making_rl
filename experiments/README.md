# Experiments Directory

This directory contains utilities and infrastructure for running experiments, computing metrics, and managing training callbacks.

## Overview

The `experiments/` directory provides:
- **Experiment Runner**: Main interface for running agent-environment experiments
- **Metrics Computation**: Performance metrics calculation (PnL, Sharpe, VaR, ES)
- **Training Callbacks**: Early stopping and model checkpointing
- **Model Loading**: Utilities for loading trained RL agents
- **Plotting**: Basic visualization utilities

---

## Modules

### 1. `runner.py` - Experiment Runner

Main entry point for running experiments. Provides a unified interface for training and evaluating agents.

**Key Functions:**

#### `run_experiment()`
Generic experiment runner that handles training and evaluation.

```python
from experiments.runner import run_experiment
from envs.abm_vanilla import ABMVanillaEnv
from agents.ppo_agent import PPOAgent

run_experiment(
    env_class=ABMVanillaEnv,
    agent_class=PPOAgent,
    env_config={"S0": 100.0, "T": 1.0, "sigma": 2.0, ...},
    agent_config={"total_timesteps": 200000, "learning_rate": 0.0003, ...},
    train=True,                    # Train the agent
    n_eval_episodes=100,           # Episodes for evaluation
    save_path="results",           # Results directory
    save_model=True,               # Save trained model
    model_save_path="models"       # Model directory
)
```

**Features:**
- Automatic handling of RL vs non-RL agents
- Training only when applicable
- Metrics computation and saving
- Model saving/loading
- Result organization by environment and agent

#### `evaluate_agent()`
Evaluate an agent on an environment for multiple episodes.

```python
from experiments.runner import evaluate_agent

pnls, inventories = evaluate_agent(
    env=env_instance,
    agent=agent_instance,
    n_episodes=100
)
```

**Returns:**
- `pnls`: Array of final PnL values for each episode
- `inventories`: Array of final inventory positions for each episode

**Features:**
- Automatic LSTM memory reset for recurrent agents
- Clean episode boundaries
- Fair evaluation across agent types

---

### 2. `metrics.py` - Performance Metrics

Computes standard performance metrics from PnL arrays.

**Key Functions:**

#### `compute_basic_metrics()`
Calculate comprehensive performance metrics.

```python
from experiments.metrics import compute_basic_metrics

metrics = compute_basic_metrics(pnls)
```

**Returns:**
```python
{
    "mean": float,      # Mean PnL
    "std": float,       # Standard deviation of PnL
    "sharpe": float,    # Sharpe ratio (mean / std)
    "var_95": float,    # Value at Risk (95% confidence)
    "var_99": float,    # Value at Risk (99% confidence)
    "es_95": float,     # Expected Shortfall (95% confidence)
    "es_99": float      # Expected Shortfall (99% confidence)
}
```

**Metrics Explained:**
- **Mean PnL**: Average profit/loss across episodes
- **Standard Deviation**: Risk measure (volatility)
- **Sharpe Ratio**: Risk-adjusted return (mean / std)
- **VaR (95%/99%)**: Maximum expected loss at confidence level
- **ES (95%/99%)**: Expected loss given VaR threshold is exceeded

---

### 3. `callbacks.py` - Training Callbacks

Custom callbacks for Stable-Baselines3 training with early stopping and best model restoration.

**Key Classes:**

#### `RewardBasedEarlyStopping`
Early stopping callback based on evaluation performance metrics.

```python
from experiments.callbacks import RewardBasedEarlyStopping

callback = RewardBasedEarlyStopping(
    eval_env=eval_env,
    monitor="mean_reward",        # or "mean_pnl", "mean_sharpe"
    patience=6,                   # Evaluations without improvement
    eval_freq=10000,              # Evaluate every N steps
    n_eval_episodes=10,           # Episodes per evaluation
    best_model_save_path="./best_model",
    verbose=1
)
```

**Features:**
- Monitors evaluation metrics (reward, PnL, Sharpe)
- Automatic best model saving
- Automatic best weight restoration
- Configurable patience and thresholds
- Detailed logging

**Monitor Options:**
- `"mean_reward"`: Mean episode reward (default)
- `"mean_pnl"`: Mean PnL per episode (extracted from info)
- `"mean_sharpe"`: Mean Sharpe ratio

#### `LossBasedEarlyStopping`
Early stopping callback based on training loss.

```python
from experiments.callbacks import LossBasedEarlyStopping

callback = LossBasedEarlyStopping(
    patience=1000,                # Steps without improvement
    min_delta=0.001,              # Minimum change for improvement
    verbose=1
)
```

#### `EarlyStoppingCallback`
Generic early stopping callback (base class).

---

### 4. `model_loader.py` - Model Loading

Utilities for loading trained RL agent models.

**Key Functions:**

#### `load_trained_agent()`
Load a trained RL agent from disk.

```python
from experiments.model_loader import load_trained_agent
from agents.ppo_agent import PPOAgent
from envs.abm_vanilla import ABMVanillaEnv

# Create environment (required for loading some models)
env = ABMVanillaEnv(**env_config)

# Load agent
agent = load_trained_agent(
    agent_class=PPOAgent,
    model_path="models/ABMVanillaEnv/PPOAgent/model.zip",
    env=env
)
```

**Features:**
- Automatic model format detection
- Environment compatibility checking
- Error handling for missing models

#### `model_exists()`
Check if a trained model exists.

```python
from experiments.model_loader import model_exists

if model_exists("models/ABMVanillaEnv/PPOAgent/model.zip"):
    # Load model
    pass
```

---

### 5. `plotting.py` - Visualization Utilities

Basic plotting utilities for result visualization.

**Key Functions:**

#### `plot_pnl_distribution()`
Generate a simple histogram of PnL distribution.

```python
from experiments.plotting import plot_pnl_distribution

plot_pnl_distribution(
    pnls=pnl_array,
    out_dir="results/ABMVanillaEnv/PPOAgent"
)
```

**Output:** Saves `pnl_distribution.png` to the specified directory.

**Note:** For more advanced visualizations, see the report generation scripts in `scripts/`.

---

## Usage Examples

### Example 1: Train and Evaluate RL Agent

```python
from experiments.runner import run_experiment
from envs.abm_vanilla import ABMVanillaEnv
from agents.ppo_agent import PPOAgent
from configs.config_loader import load_config

# Load configurations
env_cfg = load_config("configs/env_configs.yaml")["abm_vanilla"]
agent_cfg = load_config("configs/agent_configs.yaml")["ppo_basic"]

# Run experiment
results = run_experiment(
    env_class=ABMVanillaEnv,
    agent_class=PPOAgent,
    env_config=env_cfg,
    agent_config=agent_cfg,
    train=True,
    n_eval_episodes=100,
    save_model=True
)

print(f"Mean PnL: {results['mean']}")
print(f"Sharpe Ratio: {results['sharpe']}")
```

### Example 2: Evaluate Pre-trained Agent

```python
from experiments.runner import run_experiment
from experiments.model_loader import load_trained_agent
from envs.abm_vanilla import ABMVanillaEnv
from agents.ppo_agent import PPOAgent

# Load pre-trained agent
env = ABMVanillaEnv(**env_cfg)
agent = load_trained_agent(
    PPOAgent,
    "models/ABMVanillaEnv/PPOAgent/model.zip",
    env=env
)

# Evaluate
results = run_experiment(
    env_class=ABMVanillaEnv,
    agent_class=PPOAgent,
    env_config=env_cfg,
    agent_config={},
    train=False,  # Skip training
    n_eval_episodes=100
)
```

### Example 3: Custom Early Stopping

```python
from experiments.callbacks import RewardBasedEarlyStopping
from stable_baselines3 import PPO

# Create evaluation environment
eval_env = ABMVanillaEnv(**env_cfg)

# Create callback
callback = RewardBasedEarlyStopping(
    eval_env=eval_env,
    monitor="mean_pnl",
    patience=6,
    eval_freq=10000,
    n_eval_episodes=10,
    best_model_save_path="./best_model",
    verbose=1
)

# Train with callback
agent = PPO("MlpPolicy", env, verbose=1)
agent.learn(total_timesteps=200000, callback=callback)
```

### Example 4: Compute Metrics Manually

```python
from experiments.metrics import compute_basic_metrics
import numpy as np

# Simulated PnL data
pnls = np.array([1.5, 2.3, -0.5, 3.1, 0.8, ...])

# Compute metrics
metrics = compute_basic_metrics(pnls)

print(f"Mean: {metrics['mean']:.4f}")
print(f"Std: {metrics['std']:.4f}")
print(f"Sharpe: {metrics['sharpe']:.4f}")
print(f"VaR (95%): {metrics['var_95']:.4f}")
print(f"ES (95%): {metrics['es_95']:.4f}")
```

---

## File Structure

```
experiments/
├── __init__.py           # Package initialization
├── runner.py             # Main experiment runner
├── metrics.py            # Performance metrics computation
├── callbacks.py          # Training callbacks (early stopping)
├── model_loader.py       # Model loading utilities
└── plotting.py           # Basic plotting utilities
```

---

## Integration with Scripts

These modules are used by:
- `scripts/train_all_rl_agents.py` - Training pipeline
- `scripts/compare_all_agents.py` - Evaluation pipeline
- `scripts/run_full_pipeline.py` - Full pipeline orchestration
- All report generation scripts

---

## Best Practices

1. **Use `run_experiment()`** for standard experiments - it handles all the boilerplate
2. **Set `save_model=True`** for RL agents to enable later evaluation
3. **Use early stopping** to prevent overfitting and save training time
4. **Consistent evaluation** - always use the same `n_eval_episodes` for fair comparison
5. **Save results** - `run_experiment()` automatically saves metrics and arrays

---

## Troubleshooting

### Import Errors
Make sure you're running from the project root:
```bash
cd /path/to/market_making_rl
python your_script.py
```

### Model Loading Failures
- Ensure the model file exists
- Check that the agent class matches the model type
- Verify environment compatibility

### Callback Issues
- Ensure evaluation environment matches training environment config
- Check that monitor metric is available (e.g., "mean_pnl" requires info dict)
- Verify patience and eval_freq are reasonable

---

## References

- Main `README.md` for project overview
- `agents/README.md` for agent documentation
- `envs/README.md` for environment documentation
- `configs/README.md` for configuration details
