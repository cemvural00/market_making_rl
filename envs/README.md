# Environments Directory

This directory contains implementations of all synthetic market-making environments. All environments inherit from `MarketMakingBaseEnv` and implement Gymnasium's standard interface.

## Base Environment

All environments inherit from `MarketMakingBaseEnv` (`base_env.py`), which provides:

- **Common Market Making Logic:**
  - Bid/ask quote construction from actions
  - Poisson fill model: λ(δ) = A exp(-k δ)
  - Inventory and cash accounting
  - Reward: ΔPnL - inv_penalty × q² × dt
  - Gymnasium `reset()` and `step()` methods

- **Action Space:** `Box([-1, -1], [1, 1])` - `[spread_factor, skew_factor]`
- **Observation Space:** `Box([0, 0, -∞, -1], [1, ∞, ∞, 1])` - `[norm_time, S_norm, dS, q_norm]`

**Child classes only need to implement:** `_update_price()` - the price dynamics.

## Price Process Types

### ABM (Arithmetic Brownian Motion)

Price follows: `dS = μ dt + σ dW`

- **Volatility:** Absolute (same units as price)
- **Suitable for:** Short-term price modeling
- **Characteristics:** Price can become negative

**Environments:**
- `ABMVanillaEnv` - Basic ABM
- `ABMJumpEnv` - ABM with Poisson jumps
- `ABMRegimeEnv` - ABM with regime-switching volatility
- `ABMJumpRegimeEnv` - ABM with jumps + regime-switching

---

### GBM (Geometric Brownian Motion)

Price follows: `dS = μ S dt + σ S dW`

- **Volatility:** Percentage (dimensionless)
- **Suitable for:** Long-term price modeling (stock prices)
- **Characteristics:** Price always positive, multiplicative dynamics

**Important:** For GBM, `sigma` is a **percentage** (e.g., `0.2` = 20% volatility). For `S0=100`, `σ=0.2` ≈ `σ=2.0` in ABM magnitude.

**Environments:**
- `GBMVanillaEnv` - Basic GBM
- `GBMJumpEnv` - GBM with multiplicative (log-normal) jumps
- `GBMRegimeEnv` - GBM with regime-switching volatility
- `GBMJumpRegimeEnv` - GBM with jumps + regime-switching

**Jump Implementation:** Jumps are **multiplicative** (log-normal) for GBM, ensuring price remains positive.

---

### OU (Ornstein-Uhlenbeck)

Price follows: `dS = κ(μ - S) dt + σ dW`

- **Volatility:** Absolute (same units as price)
- **Mean Reversion:** Price reverts to long-term mean `μ`
- **Suitable for:** Mean-reverting processes (interest rates, spreads)
- **Characteristics:** Stationary process with bounded variance

**Environments:**
- `OUVanillaEnv` - Basic OU
- `OUJumpEnv` - OU with additive jumps
- `OURegimeEnv` - OU with regime-switching volatility
- `OUJumpRegimeEnv` - OU with jumps + regime-switching

---

## Environment Variants

### Vanilla (Basic Diffusion)

No additional complexity - just the base stochastic process.

**Files:**
- `abm_vanilla.py` - `ABMVanillaEnv`
- `gbm_vanilla.py` - `GBMVanillaEnv`
- `ou_vanilla.py` - `OUVanillaEnv`

**Usage:**
```python
from envs.abm_vanilla import ABMVanillaEnv

env = ABMVanillaEnv(
    S0=100.0,
    T=1.0,
    dt=0.0001,
    sigma=2.0,      # ABM: absolute volatility
    mu=0.0,         # Drift
    A=5.0,          # Fill intensity parameter
    k=1.5,          # Fill decay parameter
    base_delta=1.0,
    max_inventory=20,
    inv_penalty=0.01,
    seed=42
)
```

---

### Jump Diffusion

Adds Poisson jump component to the price process.

**Files:**
- `abm_jump.py` - `ABMJumpEnv`
- `gbm_jump.py` - `GBMJumpEnv`
- `ou_jump.py` - `OUJumpEnv`

**Jump Parameters:**
- `jump_intensity` (λ): Expected jumps per unit time
- `jump_mean`: Mean jump size
- `jump_std`: Standard deviation of jump size

**Jump Implementation:**
- **ABM/OU:** Additive jumps `S = S + J` where `J ~ N(jump_mean, jump_std²)`
- **GBM:** Multiplicative (log-normal) jumps `S = S × exp(J)` where `J ~ N(jump_mean, jump_std²)`

**Usage:**
```python
from envs.abm_jump import ABMJumpEnv

env = ABMJumpEnv(
    S0=100.0,
    T=1.0,
    dt=0.0001,
    sigma=2.0,
    mu=0.0,
    jump_intensity=0.1,    # λ = 0.1 jumps per unit time
    jump_mean=0.0,         # Mean jump size
    jump_std=5.0,          # Jump volatility
    A=5.0,
    k=1.5,
    # ... other params
)
```

**For GBM:**
```python
from envs.gbm_jump import GBMJumpEnv

env = GBMJumpEnv(
    S0=100.0,
    T=1.0,
    dt=0.0001,
    sigma=0.2,             # 20% volatility (percentage)
    mu=0.0,
    jump_intensity=0.1,
    jump_mean=0.0,         # Mean of log-jump size
    jump_std=0.05,         # Std dev of log-jump size (5% log-volatility)
    # ... other params
)
```

---

### Regime-Switching

Markov chain with 2 volatility regimes (low-vol and high-vol).

**Files:**
- `abm_regime.py` - `ABMRegimeEnv`
- `gbm_regime.py` - `GBMRegimeEnv`
- `ou_regime.py` - `OURegimeEnv`

**Regime Parameters:**
- `sigma_low`: Volatility in regime 0 (low-vol)
- `sigma_high`: Volatility in regime 1 (high-vol)
- `transition_matrix`: 2×2 Markov transition matrix
  ```python
  [[P(stay low), P(switch to high)],
   [P(switch to low), P(stay high)]]
  ```
- `initial_regime`: Starting regime (0 or 1)

**Default Transition Matrix:**
```python
[[0.95, 0.05],   # From low-vol: 95% stay, 5% switch to high
 [0.10, 0.90]]   # From high-vol: 10% switch to low, 90% stay
```

**Usage:**
```python
from envs.abm_regime import ABMRegimeEnv

env = ABMRegimeEnv(
    S0=100.0,
    T=1.0,
    dt=0.0001,
    sigma_low=10.0,        # Low-volatility regime
    sigma_high=40.0,       # High-volatility regime
    mu=0.0,
    transition_matrix=[[0.95, 0.05], [0.10, 0.90]],
    initial_regime=0,      # Start in low-vol regime
    A=5.0,
    k=1.5,
    # ... other params
)
```

---

### Jump + Regime-Switching

Combines both jump diffusion and regime-switching.

**Files:**
- `abm_jump_regime.py` - `ABMJumpRegimeEnv`
- `gbm_jump_regime.py` - `GBMJumpRegimeEnv`
- `ou_jump_regime.py` - `OUJumpRegimeEnv`

**Usage:**
```python
from envs.abm_jump_regime import ABMJumpRegimeEnv

env = ABMJumpRegimeEnv(
    S0=100.0,
    T=1.0,
    dt=0.0001,
    sigma_low=10.0,
    sigma_high=40.0,
    mu=0.0,
    jump_intensity=0.1,
    jump_mean=0.0,
    jump_std=5.0,
    transition_matrix=[[0.95, 0.05], [0.10, 0.90]],
    initial_regime=0,
    A=5.0,
    k=1.5,
    # ... other params
)
```

---

## Environment Parameters

All environments accept these common parameters:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `S0` | float | 100.0 | Initial midprice |
| `T` | float | 1.0 | Trading horizon (time units) |
| `dt` | float | 0.0001 | Time step size |
| `A` | float | 5.0 | Fill intensity parameter (Poisson: λ(δ) = A exp(-k δ)) |
| `k` | float | 1.5 | Fill decay parameter |
| `base_delta` | float | 1.0 | Base half-spread for action interpretation |
| `max_inventory` | int | 20 | Maximum inventory position |
| `inv_penalty` | float | 0.01 | Inventory penalty coefficient (reward = ΔPnL - inv_penalty × q² × dt) |
| `seed` | int | None | Random seed for reproducibility |

---

## Configuration

Environments are typically configured via YAML files. See `configs/env_configs.yaml` for examples:

```yaml
abm_vanilla:
  S0: 100.0
  T: 1.0
  dt: 0.0001
  sigma: 2.0
  mu: 0.0
  A: 5.0
  k: 1.5
  base_delta: 1.0
  max_inventory: 20
  inv_penalty: 0.01
  seed: 123

gbm_vanilla:
  S0: 100.0
  T: 1.0
  dt: 0.0001
  sigma: 0.2        # Percentage volatility (20%)
  mu: 0.0
  A: 5.0
  k: 1.5
  # ... other params

abm_jump:
  # ... base params
  jump_intensity: 0.1
  jump_mean: 0.0
  jump_std: 5.0

abm_regime:
  # ... base params
  sigma_low: 10.0
  sigma_high: 40.0
  transition_matrix:
    - [0.95, 0.05]
    - [0.10, 0.90]
  initial_regime: 0
```

---

## Environment Summary

| Type | Vanilla | Jump | Regime | Jump+Regime | Total |
|------|---------|------|--------|-------------|-------|
| **ABM** | ✅ | ✅ | ✅ | ✅ | 4 |
| **GBM** | ✅ | ✅ | ✅ | ✅ | 4 |
| **OU** | ✅ | ✅ | ✅ | ✅ | 4 |
| **Total** | 3 | 3 | 3 | 3 | **12** |

**All 12 environments are mathematically correct and tested.**

---

## Important Notes

### GBM Jump Implementation

**Fixed:** GBM jump environments now use **multiplicative (log-normal) jumps** instead of additive jumps. This is mathematically correct for geometric processes:

```python
# CORRECT (current implementation):
gbm_diffusion = (mu - 0.5 * sigma**2) * dt + sigma * dW
if jump_occurs:
    log_jump = rng.normal(jump_mean, jump_std)
    S = S * exp(gbm_diffusion + log_jump)
```

Previously, jumps were added additively (incorrect), which has been fixed.

### Volatility Units

- **ABM/OU:** Volatility `sigma` is in **absolute units** (same as price)
- **GBM:** Volatility `sigma` is a **percentage** (dimensionless)
  - For `S0=100`: `σ=0.2` (GBM) ≈ `σ=2.0` (ABM) in magnitude

### Parameter Matching

When using `ASClosedFormAgent`, ensure:
- `sigma` matches environment's volatility
- `A` and `k` match environment's fill model parameters
- Set `gbm=True` for GBM environments

---

## Usage Examples

### Basic Usage

```python
from configs.config_loader import load_config
from envs.abm_vanilla import ABMVanillaEnv

# Load config
env_cfg = load_config("configs/env_configs.yaml")["abm_vanilla"]

# Create environment
env = ABMVanillaEnv(**env_cfg)

# Standard Gymnasium interface
obs, info = env.reset()
for _ in range(1000):
    action = agent.act(obs, info)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
```

### Testing Different Environments

```python
from envs import (
    ABMVanillaEnv, ABMJumpEnv, ABMRegimeEnv, ABMJumpRegimeEnv,
    GBMVanillaEnv, GBMJumpEnv, GBMRegimeEnv, GBMJumpRegimeEnv,
    OUVanillaEnv, OUJumpEnv, OURegimeEnv, OUJumpRegimeEnv
)

envs = [
    (ABMVanillaEnv, "abm_vanilla"),
    (GBMVanillaEnv, "gbm_vanilla"),
    (OUVanillaEnv, "ou_vanilla"),
    # ... etc
]

for env_class, env_key in envs:
    env_cfg = load_config("configs/env_configs.yaml")[env_key]
    env = env_class(**env_cfg)
    # Run experiments...
```

### Custom Configuration

```python
from envs.abm_vanilla import ABMVanillaEnv

# Custom parameters
env = ABMVanillaEnv(
    S0=100.0,
    T=1.0,
    dt=0.0001,
    sigma=2.5,         # Custom volatility
    mu=0.0,
    A=6.0,             # Higher fill intensity
    k=1.8,             # Steeper fill decay
    base_delta=1.0,
    max_inventory=25,  # Larger inventory limit
    inv_penalty=0.015, # Higher inventory penalty
    seed=42
)
```

---

## Environment Details

### Fill Model

All environments use the Poisson fill model:
```
λ(δ) = A × exp(-k × δ)
```

Where:
- `λ(δ)`: Fill intensity at half-spread `δ`
- `A`: Base intensity parameter
- `k`: Decay parameter (larger = faster decay with spread)

The probability of a fill in time step `dt` is approximately `λ(δ) × dt`.

### Reward Function

Reward at each step:
```
reward = ΔPnL - inv_penalty × q² × dt
```

Where:
- `ΔPnL`: Change in profit and loss (from trades and inventory mark-to-market)
- `q`: Current inventory position
- `inv_penalty`: Penalty coefficient (discourages large inventory)

### Observation Space

```
obs = [norm_time, S_norm, dS, q_norm]
```

- `norm_time = t / T`: Normalized time ∈ [0, 1]
- `S_norm = S / S0`: Normalized midprice
- `dS = S - S_prev`: Price change from previous step
- `q_norm = q / max_inventory`: Normalized inventory ∈ [-1, 1]

### Action Space

```
action = [spread_factor, skew_factor]
```

Both factors ∈ [-1, 1] are interpreted as:

- **Spread:** Half-spread `δ = base_delta × (1 + spread_factor)`
- **Skew:** Quote shift `skew = skew_factor × base_delta`

Bid and ask prices:
```
bid = S - δ - skew
ask = S + δ - skew
```

---

## Testing

All environments have been mathematically validated. See `ENVIRONMENT_EVALUATION.md` for detailed analysis and validation results.

## References

For detailed mathematical background and validation, see:
- Main `README.md` for overview
- `ENVIRONMENT_EVALUATION.md` for validation results
- `configs/README.md` for configuration details
