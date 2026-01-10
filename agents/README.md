# Agents Directory

This directory contains implementations of all market-making agents used in the experiments. Agents are divided into three categories: **Reinforcement Learning (RL)**, **Analytic**, and **Heuristic**.

## Base Agent Interface

All agents inherit from `BaseAgent` (`base_agent.py`), which defines the standard interface:

```python
class BaseAgent:
    def act(self, obs, info=None) -> np.ndarray:
        """Compute action from observation. Must be implemented by all agents."""
        raise NotImplementedError
    
    def train(self, env) -> None:
        """Train the agent. Only implemented by RL agents."""
        raise NotImplementedError
    
    def save(self, path) -> None:
        """Save agent parameters. Only implemented by RL agents."""
        raise NotImplementedError
    
    def load(self, path) -> None:
        """Load agent parameters. Only implemented by RL agents."""
        raise NotImplementedError
```

### Observation Space

All agents receive observations in the format:
```python
obs = [norm_time, S_norm, dS, q_norm]
```

Where:
- `norm_time`: Normalized time `t/T` ∈ [0, 1]
- `S_norm`: Normalized midprice `S/S0`
- `dS`: Price change from previous step
- `q_norm`: Normalized inventory `q/max_inventory` ∈ [-1, 1]

### Action Space

All agents output actions in the format:
```python
action = [spread_factor, skew_factor]
```

Where:
- `spread_factor` ∈ [-1, 1]: Controls bid-ask spread width
- `skew_factor` ∈ [-1, 1]: Controls bid-ask skew (inventory management)

The environment interprets these factors to construct actual bid and ask prices.

---

## Reinforcement Learning Agents

RL agents use Stable-Baselines3 and require training before use. They support early stopping, model saving/loading, and automatic best weight restoration.

### 1. PPOAgent (`ppo_agent.py`)

**Algorithm:** Proximal Policy Optimization (PPO)

**Description:** Standard PPO implementation using Stable-Baselines3. Well-suited for continuous action spaces and provides stable learning.

**Usage:**
```python
from agents.ppo_agent import PPOAgent

agent = PPOAgent(env, config={
    "total_timesteps": 200000,
    "learning_rate": 0.0003,
    "gamma": 1.0,
    "n_steps": 1024,
    "batch_size": 256,
    "early_stopping": {
        "enabled": True,
        "monitor_type": "reward",
        "monitor": "mean_reward",
        "patience": 6,
        "eval_freq": 10000,
        "n_eval_episodes": 10
    }
})
```

**Key Features:**
- On-policy learning
- Stable training with clipped objectives
- CPU-optimized (recommended over GPU for MLP policies)

---

### 2. DeepPPOAgent (`deep_ppo_agent.py`)

**Algorithm:** PPO with Deep Networks

**Description:** PPO agent with deeper neural network architectures for increased model capacity and pattern recognition.

**Usage:**
```python
from agents.deep_ppo_agent import DeepPPOAgent

agent = DeepPPOAgent(env, config={
    "total_timesteps": 200000,
    "policy_kwargs": {
        "net_arch": [256, 256, 128],  # Deeper networks
        "activation_fn": torch.nn.ReLU
    }
})
```

**Key Features:**
- Deeper networks than standard PPO
- Better for complex, high-dimensional state spaces
- CPU-optimized

---

### 3. LSTMPPOAgent (`lstm_agent.py`)

**Algorithm:** Recurrent PPO with LSTM

**Description:** PPO agent with LSTM layers for temporal dependency modeling. Uses `RecurrentPPO` from `sb3-contrib`.

**Usage:**
```python
from agents.lstm_agent import LSTMPPOAgent

agent = LSTMPPOAgent(env, config={
    "total_timesteps": 300000,
    "n_steps": 512,  # Smaller for LSTM
    "policy_kwargs": {
        "lstm_hidden_size": 128,
        "n_lstm_layers": 1,
        "net_arch": [64, 64]
    }
})
```

**Key Features:**
- Temporal memory via LSTM
- Effective for regime-switching and jump environments
- Maintains context across timesteps
- CPU-optimized

**Note:** Do not use `lstm_agent_new.py` - it is an incomplete alternative version.

---

### 4. SACAgent (`sac_agent.py`)

**Algorithm:** Soft Actor-Critic (SAC)

**Description:** Off-policy actor-critic algorithm with entropy maximization for exploration.

**Usage:**
```python
from agents.sac_agent import SACAgent

agent = SACAgent(env, config={
    "total_timesteps": 300000,
    "learning_rate": 0.0003,
    "buffer_size": 100000,
    "batch_size": 256,
    "gamma": 0.99,
    "tau": 0.005  # Soft update coefficient
})
```

**Key Features:**
- Off-policy learning (sample efficient)
- Automatic entropy tuning
- Well-suited for continuous action spaces
- CPU-optimized

---

### 5. TD3Agent (`td3_agent.py`)

**Algorithm:** Twin Delayed DDPG (TD3)

**Description:** Deterministic policy gradient algorithm with clipped double Q-learning for stability.

**Usage:**
```python
from agents.td3_agent import TD3Agent

agent = TD3Agent(env, config={
    "total_timesteps": 300000,
    "learning_rate": 0.001,
    "buffer_size": 100000,
    "batch_size": 256
})
```

**Key Features:**
- Off-policy deterministic actor-critic
- Reduced overestimation bias
- Stable training dynamics
- CPU-optimized

---

### 6. LSTMSACAgent (`lstm_sac_agent.py`)

**Algorithm:** SAC with Deep Networks

**Description:** SAC agent with deeper network architectures (note: true RecurrentSAC not available in sb3-contrib, so this uses deep networks instead).

**Usage:**
```python
from agents.lstm_sac_agent import LSTMSACAgent

agent = LSTMSACAgent(env, config={
    "total_timesteps": 300000,
    "policy_kwargs": {
        "net_arch": [256, 256, 128]  # Deeper networks
    }
})
```

**Key Features:**
- Deep networks for pattern recognition
- SAC's off-policy learning benefits
- High model capacity
- CPU-optimized

---

## Analytic Agents

Analytic agents implement mathematically derived optimal or near-optimal solutions.

### 7. ASClosedFormAgent (`as_agent.py`)

**Method:** Avellaneda-Stoikov Closed-Form Optimal Solution

**Description:** Implements the closed-form optimal control solution from Avellaneda & Stoikov (2008) under the assumption of:
- Brownian motion price dynamics
- Exponential utility with risk aversion
- Poisson fill model λ(δ) = A exp(-k δ)

**Usage:**
```python
from agents.as_agent import ASClosedFormAgent

# For ABM/OU environments
agent = ASClosedFormAgent(config={
    "gamma": 0.1,  # Risk aversion parameter
    "sigma": 2.0,  # Price volatility (must match environment)
    "A": 5.0,      # Fill intensity parameter (must match environment)
    "k": 1.5       # Fill decay parameter (must match environment)
})

# For GBM environments
agent = ASClosedFormAgent(config={
    "gamma": 0.1,
    "sigma": 0.2,  # Percentage volatility for GBM
    "A": 5.0,
    "k": 1.5,
    "gbm": True    # Enable GBM-specific adjustments
})
```

**Key Features:**
- Theoretically optimal under model assumptions
- No training required
- Highly sensitive to parameter mismatch
- Best performance in vanilla environments matching assumptions

**Limitations:**
- Requires exact knowledge of environment parameters
- Assumes Brownian motion (struggles with jumps/regimes)
- No adaptive learning

---

### 8. ASSimpleHeuristicAgent (`as_agent.py`)

**Method:** AS-Inspired Heuristic

**Description:** Simple interpretable benchmark that behaves "AS-like" but is not the closed-form optimal solution. Uses:
- Spread depends on time-to-maturity
- Skew depends linearly on inventory

**Usage:**
```python
from agents.as_agent import ASSimpleHeuristicAgent

agent = ASSimpleHeuristicAgent(config={
    "gamma": 0.1,           # Inventory sensitivity
    "min_spread_factor": 0.2,
    "max_spread_factor": 0.8,
    "max_inventory": 20
})
```

**Key Features:**
- Interpretable and simple
- Robust across environments
- No training required
- Good baseline for comparison

---

## Heuristic Agents

Heuristic agents implement rule-based strategies for comparison and benchmarking.

### 9. FixedSpreadAgent (`fixed_spread_agent.py`)

**Strategy:** Fixed bid-ask spread

**Description:** Maintains constant spread throughout the trading period.

**Usage:**
```python
from agents.fixed_spread_agent import FixedSpreadAgent

agent = FixedSpreadAgent(config={
    "spread_factor": 0.5  # Constant spread factor
})
```

---

### 10. InventoryShiftAgent (`inv_shift_agent.py`)

**Strategy:** Inventory-based quote shifting

**Description:** Shifts bid and ask quotes based on inventory position to manage risk.

**Usage:**
```python
from agents.inv_shift_agent import InventoryShiftAgent

agent = InventoryShiftAgent(config={
    "spread_factor": 0.5,
    "inv_sensitivity": 0.1  # Inventory sensitivity
})
```

---

### 11. InventorySpreadScalerAgent (`inv_spread_scaler_agent.py`)

**Strategy:** Inventory-based spread scaling

**Description:** Adjusts spread width based on inventory to penalize large positions.

**Usage:**
```python
from agents.inv_spread_scaler_agent import InventorySpreadScalerAgent

agent = InventorySpreadScalerAgent(config={
    "base_spread_factor": 0.5,
    "inv_sensitivity": 0.1
})
```

---

### 12. MidPriceFollowAgent (`mid_price_follow_agent.py`)

**Strategy:** Mid-price following

**Description:** Quotes follow the mid-price with some spread.

**Usage:**
```python
from agents.mid_price_follow_agent import MidPriceFollowAgent

agent = MidPriceFollowAgent(config={
    "spread_factor": 0.5
})
```

---

### 13. LastLookAgent (`last_look_agent.py`)

**Strategy:** Last-look execution

**Description:** Implements last-look execution logic.

**Usage:**
```python
from agents.last_look_agent import LastLookAgent

agent = LastLookAgent(config={
    "spread_factor": 0.5
})
```

---

### 14. MarketOrderOnlyAgent (`market_order_agent.py`)

**Strategy:** Market orders only

**Description:** Executes trades using market orders only (aggressive trading).

**Usage:**
```python
from agents.market_order_agent import MarketOrderOnlyAgent

agent = MarketOrderOnlyAgent(config={
    "spread_factor": 0.0  # No spread for market orders
})
```

---

### 15. NoiseTraderNormal (`noise_trader_normal.py`)

**Strategy:** Random trading with normal distribution

**Description:** Generates random actions from a normal distribution for comparison.

**Usage:**
```python
from agents.noise_trader_normal import NoiseTraderNormal

agent = NoiseTraderNormal(config={
    "mean": 0.0,
    "std": 0.5
})
```

---

### 16. NoiseTraderUniform (`noise_trader_uniform.py`)

**Strategy:** Random trading with uniform distribution

**Description:** Generates random actions from a uniform distribution for comparison.

**Usage:**
```python
from agents.noise_trader_uniform import NoiseTraderUniform

agent = NoiseTraderUniform(config={
    "low": -0.5,
    "high": 0.5
})
```

---

### 17. ZeroIntelligenceAgent (`zero_intelligence_agent.py`)

**Strategy:** Zero-intelligence trading

**Description:** Minimal intelligence baseline that uses fixed or random actions.

**Usage:**
```python
from agents.zero_intelligence_agent import ZeroIntelligenceAgent

agent = ZeroIntelligenceAgent(config={
    "spread_factor": 0.5,
    "skew_factor": 0.0
})
```

---

## Configuration

All agents accept a `config` dictionary for hyperparameters. RL agents also support early stopping configuration:

```python
config = {
    # Agent-specific hyperparameters
    "learning_rate": 0.0003,
    "gamma": 1.0,
    # ...
    
    # Early stopping (RL agents only)
    "early_stopping": {
        "enabled": True,
        "monitor_type": "reward",  # or "loss"
        "monitor": "mean_reward",  # or "mean_pnl", "mean_sharpe"
        "patience": 6,              # evaluations without improvement
        "eval_freq": 10000,         # evaluate every N steps
        "n_eval_episodes": 10       # episodes per evaluation
    }
}
```

For detailed configuration options, see `configs/agent_configs.yaml`.

## Agent Categories Summary

| Category | Agents | Count | Training Required |
|----------|--------|-------|-------------------|
| **RL** | PPOAgent, DeepPPOAgent, LSTMPPOAgent, SACAgent, TD3Agent, LSTMSACAgent | 6 | Yes |
| **Analytic** | ASClosedFormAgent, ASSimpleHeuristicAgent | 2 | No |
| **Heuristic** | FixedSpreadAgent, InventoryShiftAgent, InventorySpreadScalerAgent, MidPriceFollowAgent, LastLookAgent, MarketOrderOnlyAgent, NoiseTraderNormal, NoiseTraderUniform, ZeroIntelligenceAgent | 9 | No |

**Total: 17 agents**

## Best Practices

1. **RL Agents:** Train before evaluation. Use early stopping to prevent overfitting.
2. **ASClosedFormAgent:** Ensure parameters (sigma, A, k) match environment exactly.
3. **Heuristic Agents:** Use as baselines for comparison with RL methods.
4. **Device Selection:** All agents are optimized for CPU on M1 Mac. GPU/MPS may not provide speedup.

## Examples

See `experiments/runner.py` for example usage patterns, or check the main `README.md` for quick start examples.
