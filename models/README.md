# Models Directory

This directory contains all trained reinforcement learning agent models, organized by environment and agent type.

## Directory Structure

```
models/
├── {EnvName}/              # Environment-specific models
│   └── {AgentName}/        # Agent-specific model directory
│       ├── model.zip       # Trained model file (Stable-Baselines3 format)
│       └── metadata.json   # Training metadata and configuration
│
└── ...
```

**Example:**
```
models/
├── ABMVanillaEnv/
│   ├── PPOAgent/
│   │   ├── model.zip
│   │   └── metadata.json
│   ├── DeepPPOAgent/
│   │   ├── model.zip
│   │   └── metadata.json
│   └── ...
├── GBMJumpEnv/
│   └── ...
└── ...
```

---

## Model Files

### `model.zip`
Trained agent model saved in Stable-Baselines3 format (ZIP archive containing PyTorch weights and architecture).

**Format:** Stable-Baselines3 ZIP archive
- Contains PyTorch model weights
- Contains network architecture
- Contains optimizer state (if applicable)
- Can be loaded with `agent.load(path, env=env)`

**File Size:** Typically 1-10 MB per model, depending on:
- Network architecture depth
- Agent type (PPO, SAC, TD3, etc.)
- Network size (hidden layers, LSTM units)

**Compatibility:**
- Models are saved with Stable-Baselines3 version used during training
- Models should be loaded with compatible agent class and environment
- Models are environment-specific (cannot be directly used on different environments)

---

### `metadata.json`
Training metadata containing configuration and training information.

**Structure:**
```json
{
    "env_class": "ABMVanillaEnv",
    "agent_class": "PPOAgent",
    "env_config": {
        "S0": 100.0,
        "T": 1.0,
        "dt": 0.0001,
        "sigma": 2.0,
        "mu": 0.0,
        "A": 5.0,
        "k": 1.5,
        "base_delta": 1.0,
        "max_inventory": 20,
        "inv_penalty": 0.01,
        "seed": 123
    },
    "agent_config": {
        "total_timesteps": 200000,
        "learning_rate": 0.0003,
        "gamma": 1.0,
        "n_steps": 1024,
        "batch_size": 256,
        "early_stopping": {
            "enabled": true,
            "monitor_type": "reward",
            "monitor": "mean_reward",
            "patience": 6,
            "eval_freq": 10000,
            "n_eval_episodes": 10
        },
        ...
    },
    "total_timesteps": 200000,
    "trained_at": "2026-01-10T14:30:00.123456",
    "model_path": "models/ABMVanillaEnv/PPOAgent/model"
}
```

**Fields:**
- `env_class`: Environment class name
- `agent_class`: Agent class name
- `env_config`: Complete environment configuration used during training
- `agent_config`: Complete agent configuration (hyperparameters, early stopping, etc.)
- `total_timesteps`: Total training timesteps
- `trained_at`: ISO timestamp of when training completed
- `model_path`: Relative path to model file (without .zip extension)

**Usage:**
- Reproduce training configuration
- Verify environment compatibility
- Check training parameters
- Track training timestamps

---

## Loading Models

### Using `experiments.model_loader`

Recommended way to load models:

```python
from experiments.model_loader import load_trained_agent
from agents.ppo_agent import PPOAgent
from envs.abm_vanilla import ABMVanillaEnv
from configs.config_loader import load_config

# Load environment config (should match training config)
env_cfg = load_config("configs/env_configs.yaml")["abm_vanilla"]

# Create environment
env = ABMVanillaEnv(**env_cfg)

# Load trained agent
agent = load_trained_agent(
    agent_class=PPOAgent,
    model_path="models/ABMVanillaEnv/PPOAgent/model",
    env=env
)

# Agent is ready to use
obs, info = env.reset()
action = agent.act(obs, info)
```

### Manual Loading

You can also load models manually:

```python
from agents.ppo_agent import PPOAgent
from envs.abm_vanilla import ABMVanillaEnv
from configs.config_loader import load_config

# Load configs
env_cfg = load_config("configs/env_configs.yaml")["abm_vanilla"]
agent_cfg = load_config("configs/agent_configs.yaml")["ppo_basic"]

# Create environment
env = ABMVanillaEnv(**env_cfg)

# Instantiate agent
agent = PPOAgent(env, config=agent_cfg)

# Load model
agent.load("models/ABMVanillaEnv/PPOAgent/model", env=env)

# Agent is ready to use
```

### Checking if Model Exists

```python
from experiments.model_loader import model_exists

if model_exists("ABMVanillaEnv", "PPOAgent", "models"):
    print("Model exists")
    # Load model...
else:
    print("Model not found")
    # Train model...
```

---

## Supported Agents

Only reinforcement learning agents are saved to this directory. Heuristic and analytic agents do not require training or models.

**RL Agents:**
- `PPOAgent` - Proximal Policy Optimization
- `DeepPPOAgent` - PPO with deep networks
- `LSTMPPOAgent` - Recurrent PPO with LSTM
- `SACAgent` - Soft Actor-Critic
- `TD3Agent` - Twin Delayed DDPG
- `LSTMSACAgent` - SAC with deep networks

**Not Saved:**
- Analytic agents (ASClosedFormAgent, ASSimpleHeuristicAgent)
- Heuristic agents (FixedSpreadAgent, etc.)

---

## Model Organization

Models are organized by:
1. **Environment** - Each environment has its own directory
2. **Agent** - Each agent has its own subdirectory within the environment directory

**Total Models:**
- 6 RL agents × 12 environments = 72 possible models
- Models are only created when training is completed successfully

**Environments:**
- `ABMVanillaEnv`, `ABMJumpEnv`, `ABMRegimeEnv`, `ABMJumpRegimeEnv`
- `GBMVanillaEnv`, `GBMJumpEnv`, `GBMRegimeEnv`, `GBMJumpRegimeEnv`
- `OUVanillaEnv`, `OUJumpEnv`, `OURegimeEnv`, `OUJumpRegimeEnv`

---

## Training Models

### Using Training Scripts

The easiest way to train and save models:

```bash
# Train all RL agents on all environments
python scripts/train_all_rl_agents.py

# Train with custom evaluation episodes
python scripts/train_all_rl_agents.py --eval-episodes 200

# Retrain even if model exists
python scripts/train_all_rl_agents.py --no-skip
```

**Output:**
- Models saved to `models/{EnvName}/{AgentName}/`
- Training metadata saved to `metadata.json`
- Initial evaluation results saved to `results/{EnvName}/{AgentName}/`

### Using Experiment Runner

Train individual models:

```python
from experiments.runner import run_experiment
from envs.abm_vanilla import ABMVanillaEnv
from agents.ppo_agent import PPOAgent
from configs.config_loader import load_config

# Load configs
env_cfg = load_config("configs/env_configs.yaml")["abm_vanilla"]
agent_cfg = load_config("configs/agent_configs.yaml")["ppo_basic"]

# Train and save model
run_experiment(
    env_class=ABMVanillaEnv,
    agent_class=PPOAgent,
    env_config=env_cfg,
    agent_config=agent_cfg,
    train=True,
    save_model=True,          # Save model
    model_save_path="models"  # Model directory
)
```

### Using Agent Directly

Train and save manually:

```python
from agents.ppo_agent import PPOAgent
from envs.abm_vanilla import ABMVanillaEnv
from configs.config_loader import load_config

# Load configs
env_cfg = load_config("configs/env_configs.yaml")["abm_vanilla"]
agent_cfg = load_config("configs/agent_configs.yaml")["ppo_basic"]

# Create environment and agent
env = ABMVanillaEnv(**env_cfg)
agent = PPOAgent(env, config=agent_cfg)

# Train
agent.train()

# Save
import os
model_dir = "models/ABMVanillaEnv/PPOAgent"
os.makedirs(model_dir, exist_ok=True)
agent.save(os.path.join(model_dir, "model"))
```

---

## Model Compatibility

### Environment Compatibility

Models are **environment-specific**:
- A model trained on `ABMVanillaEnv` cannot be directly used on `GBMVanillaEnv`
- Environment configuration must match training configuration
- Observation and action spaces must match

**To verify compatibility:**
```python
import json

# Load metadata
with open("models/ABMVanillaEnv/PPOAgent/metadata.json", "r") as f:
    metadata = json.load(f)

# Check environment config
env_config = metadata["env_config"]
print(f"Model trained on: {metadata['env_class']}")
print(f"Environment config: {env_config}")

# Use this config to create compatible environment
```

### Agent Compatibility

Models are **agent-specific**:
- A `PPOAgent` model cannot be loaded with `SACAgent`
- Agent class must match exactly
- Architecture must be compatible

### Version Compatibility

Models are saved with Stable-Baselines3 version used during training:
- Check SB3 version: `import stable_baselines3; print(stable_baselines3.__version__)`
- Models from newer versions may not be compatible with older SB3
- Upgrade SB3 if loading older models fails

---

## Best Practices

1. **Save Metadata**: Always keep `metadata.json` files - they contain training configuration needed for reproducibility
2. **Version Control**: Consider versioning important models or metadata files
3. **Backup Models**: Backup trained models before major changes
4. **Verify Compatibility**: Always check environment configuration matches before loading
5. **Organize by Experiment**: Use consistent naming for model organization
6. **Document Training**: Add notes to metadata or separate documentation for important training runs

---

## Model Sizes

Typical model sizes:
- **PPOAgent**: 1-3 MB
- **DeepPPOAgent**: 2-5 MB
- **LSTMPPOAgent**: 3-7 MB (LSTM increases size)
- **SACAgent**: 2-4 MB
- **TD3Agent**: 2-4 MB
- **LSTMSACAgent**: 3-7 MB

**Total Storage:**
- Full model suite (6 agents × 12 environments): ~150-300 MB
- With metadata files: ~160-310 MB

---

## Troubleshooting

### Model Not Found

**Error:** `FileNotFoundError: Model file not found: model.zip`

**Solutions:**
- Check that model exists: `os.path.exists("models/ABMVanillaEnv/PPOAgent/model.zip")`
- Verify path is correct
- Train the model if it doesn't exist

### Environment Mismatch

**Error:** Model loads but performs poorly or crashes

**Solutions:**
- Verify environment configuration matches `metadata.json`
- Check observation/action space compatibility
- Ensure environment class matches training environment

### Loading Errors

**Error:** `RuntimeError` or `KeyError` when loading

**Solutions:**
- Verify Stable-Baselines3 version compatibility
- Check agent class matches model type
- Ensure environment is properly instantiated before loading

### Memory Issues

**Error:** Out of memory when loading multiple models

**Solutions:**
- Load models one at a time
- Unload previous model before loading next
- Use `del agent; import gc; gc.collect()` to free memory

---

## Cleaning Models

### Delete All Models

```bash
# Use clean script
python scripts/clean_outputs.py --models

# Or manually
rm -rf models/*/
```

### Delete Specific Model

```python
import shutil
import os

model_dir = "models/ABMVanillaEnv/PPOAgent"
if os.path.exists(model_dir):
    shutil.rmtree(model_dir)
    print(f"Deleted {model_dir}")
```

### Delete by Agent or Environment

```python
from scripts.clean_outputs import clean_models

# Delete all PPOAgent models
clean_models(agent_name="PPOAgent")

# Delete all ABMVanillaEnv models
clean_models(env_name="ABMVanillaEnv")
```

---

## Example Workflows

### Workflow 1: Train and Evaluate

```python
# 1. Train model
from experiments.runner import run_experiment
from envs.abm_vanilla import ABMVanillaEnv
from agents.ppo_agent import PPOAgent
from configs.config_loader import load_config

env_cfg = load_config("configs/env_configs.yaml")["abm_vanilla"]
agent_cfg = load_config("configs/agent_configs.yaml")["ppo_basic"]

run_experiment(
    env_class=ABMVanillaEnv,
    agent_class=PPOAgent,
    env_config=env_cfg,
    agent_config=agent_cfg,
    train=True,
    save_model=True,
    n_eval_episodes=100
)

# 2. Later, load and evaluate
from experiments.model_loader import load_trained_agent
from experiments.runner import evaluate_agent

env = ABMVanillaEnv(**env_cfg)
agent = load_trained_agent(PPOAgent, "models/ABMVanillaEnv/PPOAgent/model", env=env)

pnls, inventories = evaluate_agent(env, agent, n_episodes=100)
print(f"Mean PnL: {pnls.mean()}")
```

### Workflow 2: Compare Multiple Models

```python
from experiments.model_loader import load_trained_agent
from envs.abm_vanilla import ABMVanillaEnv
from agents import PPOAgent, SACAgent, TD3Agent
from configs.config_loader import load_config

env_cfg = load_config("configs/env_configs.yaml")["abm_vanilla"]
env = ABMVanillaEnv(**env_cfg)

agents = {
    "PPO": load_trained_agent(PPOAgent, "models/ABMVanillaEnv/PPOAgent/model", env),
    "SAC": load_trained_agent(SACAgent, "models/ABMVanillaEnv/SACAgent/model", env),
    "TD3": load_trained_agent(TD3Agent, "models/ABMVanillaEnv/TD3Agent/model", env)
}

# Evaluate each
from experiments.runner import evaluate_agent

for name, agent in agents.items():
    pnls, _ = evaluate_agent(env, agent, n_episodes=100)
    print(f"{name}: Mean PnL = {pnls.mean():.4f}")
```

---

## References

- Main `README.md` for project overview
- `agents/README.md` for agent documentation
- `experiments/README.md` for model loading utilities
- `scripts/README.md` for training scripts
- Stable-Baselines3 documentation: https://stable-baselines3.readthedocs.io/
