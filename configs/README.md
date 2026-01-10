# Configuration Files Guide

This directory contains YAML configuration files for environments, agents, and training settings.

## File Structure

```
configs/
├── config_loader.py          # Utilities for loading configs
├── env_configs.yaml          # Environment presets (ABM, GBM, OU)
├── agent_configs.yaml        # Agent presets (PPO, LSTM, AS, heuristics)
├── training_configs.yaml     # Training/evaluation settings
```

## Quick Start

### Option 1: Load from YAML (Recommended for Reproducibility)

```python
from configs.config_loader import load_config
from envs.abm_vanilla import ABMVanillaEnv
from agents.ppo_agent import PPOAgent
from experiments.runner import run_experiment

# Load environment config
env_cfg = load_config("configs/env_configs.yaml")["abm_vanilla"]

# Load agent config
agent_cfg = load_config("configs/agent_configs.yaml")["ppo_basic"]

# Run experiment
run_experiment(
    env_class=ABMVanillaEnv,
    agent_class=PPOAgent,
    env_config=env_cfg,
    agent_config=agent_cfg,
    train=True,
    n_eval_episodes=100
)
```

### Option 2: Use Python Dicts (Quick Testing)

```python
from envs.abm_vanilla import ABMVanillaEnv
from agents.as_agent import ASClosedFormAgent
from experiments.runner import run_experiment

# Direct dict configs
run_experiment(
    env_class=ABMVanillaEnv,
    agent_class=ASClosedFormAgent,
    env_config={"S0": 100, "sigma": 2.0, "dt": 0.01, "T": 1.0},
    agent_config={"gamma": 0.1, "sigma": 2.0, "k": 1.5},
    train=False,
    n_eval_episodes=100
)
```

### Option 3: Mix YAML and Dicts

```python
from configs.config_loader import load_config

# Load base config from YAML, override specific values
env_cfg = load_config("configs/env_configs.yaml")["abm_vanilla"]
env_cfg["seed"] = 999  # Override seed

agent_cfg = load_config("configs/agent_configs.yaml")["ppo_basic"]
agent_cfg["total_timesteps"] = 50000  # Override training steps
```

## Available Config Presets

### Environment Configs (`env_configs.yaml`)

- `abm_vanilla` - Basic Arithmetic Brownian Motion
- `abm_jump` - ABM with jump diffusion
- `abm_regime` - ABM with regime-switching volatility
- `abm_jump_regime` - ABM with jumps + regimes
- `gbm_vanilla` - Basic Geometric Brownian Motion
- `gbm_jump` - GBM with multiplicative jumps
- `gbm_regime` - GBM with regime-switching
- `gbm_jump_regime` - GBM with jumps + regimes
- `ou_vanilla` - Basic Ornstein-Uhlenbeck
- `ou_jump` - OU with jump diffusion
- `ou_regime` - OU with regime-switching
- `ou_jump_regime` - OU with jumps + regimes

### Agent Configs (`agent_configs.yaml`)

**RL Agents:**
- `ppo_basic` - Standard PPO with default MLP
- `ppo_deep` - PPO with deep network (3x256)
- `ppo_fast` - Fast training (for quick tests)
- `lstm_ppo` - LSTM-based PPO
- `lstm_ppo_large` - Larger LSTM network

**Heuristic Agents:**
- `as_closed_form` - Avellaneda-Stoikov closed-form (for ABM/OU)
- `as_closed_form_gbm` - AS agent for GBM environments
- `as_simple_heuristic` - Simple AS-inspired heuristic
- `fixed_spread` - Fixed spread agent
- `fixed_spread_wide` - Wider fixed spread
- `fixed_spread_narrow` - Narrower fixed spread
- `inventory_shift` - Inventory-based quote shifting
- `inventory_shift_aggressive` - More aggressive inventory management
- `inventory_shift_conservative` - More conservative inventory management

**Test Configs:**
- `ppo_test` - Minimal PPO config for quick tests
- `lstm_ppo_test` - Minimal LSTM config for quick tests

### Training Configs (`training_configs.yaml`)

- `evaluation` - Standard evaluation settings
- `evaluation_quick` - Quick evaluation (100 episodes)
- `evaluation_full` - Full evaluation (100 episodes)
- `training` - Basic training settings
- `training_with_save` - Training with model saving
- `experiment_basic` - Basic experiment settings
- `experiment_reproducible` - Reproducible experiment settings
- `experiment_quick_test` - Quick test settings

## Usage Examples

### Example 1: Compare PPO vs AS on ABM

```python
from configs.config_loader import load_config
from envs.abm_vanilla import ABMVanillaEnv
from agents.ppo_agent import PPOAgent
from agents.as_agent import ASClosedFormAgent
from experiments.runner import run_experiment

# Load configs
env_cfg = load_config("configs/env_configs.yaml")["abm_vanilla"]
ppo_cfg = load_config("configs/agent_configs.yaml")["ppo_basic"]
as_cfg = load_config("configs/agent_configs.yaml")["as_closed_form"]

# Run PPO experiment
run_experiment(
    env_class=ABMVanillaEnv,
    agent_class=PPOAgent,
    env_config=env_cfg,
    agent_config=ppo_cfg,
    train=True,
    n_eval_episodes=100
)

# Run AS experiment
run_experiment(
    env_class=ABMVanillaEnv,
    agent_class=ASClosedFormAgent,
    env_config=env_cfg,
    agent_config=as_cfg,
    train=False,
    n_eval_episodes=100
)
```

### Example 2: Test Different Environments

```python
from configs.config_loader import load_config
from envs import ABMVanillaEnv, OUVanillaEnv, GBMVanillaEnv
from agents.as_agent import ASClosedFormAgent
from experiments.runner import run_experiment

envs = {
    "ABM": (ABMVanillaEnv, "abm_vanilla"),
    "OU": (OUVanillaEnv, "ou_vanilla"),
    "GBM": (GBMVanillaEnv, "gbm_vanilla"),
}

agent_cfg = load_config("configs/agent_configs.yaml")["as_closed_form"]

for env_name, (env_class, env_key) in envs.items():
    env_cfg = load_config("configs/env_configs.yaml")[env_key]
    # Adjust agent config for GBM
    if env_name == "GBM":
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

### Example 3: Quick Test with Minimal Config

```python
from configs.config_loader import load_config
from envs.abm_vanilla import ABMVanillaEnv
from agents.ppo_agent import PPOAgent
from experiments.runner import run_experiment

# Quick test configs
env_cfg = load_config("configs/env_configs.yaml")["abm_vanilla"]
agent_cfg = load_config("configs/agent_configs.yaml")["ppo_test"]

run_experiment(
    env_class=ABMVanillaEnv,
    agent_class=PPOAgent,
    env_config=env_cfg,
    agent_config=agent_cfg,
    train=True,
    n_eval_episodes=100
)
```

## Important Notes

1. **Parameter Matching**: For AS agents, ensure agent config parameters match environment parameters:
   - `sigma` must match env.sigma
   - `k` must match env.k
   - `base_delta` must match env.base_delta
   - `max_inventory` must match env.max_inventory

2. **GBM vs ABM/OU**: GBM uses percentage volatility (0.02 = 2%), while ABM/OU use absolute volatility (2.0). Use `as_closed_form_gbm` for GBM environments.

3. **Seeds**: All configs include a default seed (123). Change it for different random runs.

4. **Reproducibility**: Using YAML configs ensures exact reproducibility. Save your final experiment configs for thesis documentation.

## Customizing Configs

You can:
1. Edit the YAML files directly to add new presets
2. Create new YAML files in `hyperparams/` for specific experiments
3. Override values programmatically after loading
4. Use Python dicts for one-off experiments

## Tips

- **For thesis experiments**: Use YAML configs and save them with your results
- **For quick tests**: Use Python dicts or test configs
- **For hyperparameter search**: Create separate YAML files in `hyperparams/`
- **For reproducibility**: Always specify seeds and save configs with results
