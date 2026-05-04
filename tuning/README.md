# Hyperparameter Tuning Pipeline

Bayesian hyperparameter optimisation for all six RL market-making agents using
[Optuna](https://optuna.org/) (TPE sampler). Results persist in a SQLite
database so sessions can be interrupted and resumed without losing progress.

---

## 1. Overview and Motivation

Default hyperparameters from Stable-Baselines3 are general-purpose starting
points, not calibrated to the specific characteristics of this environment
(10,000-step episodes, quadratic inventory penalty, Poisson fill model). Prior
results (v2 run) show high variance across agents — PPO ranked 13th overall
despite being a strong algorithm, likely due to poor hyperparameter choices
rather than algorithmic weakness.

Systematic search addresses this by exploring the space of configurations
efficiently. Because all data is simulated there is no risk of overfitting to
a real data distribution; the only risk is over-tuning to the simulator's
specific parameters, which is mitigated by tuning each agent × environment pair
independently.

---

## 2. Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Library** | Optuna (TPE) | Bayesian; learns from past trials. ~2× more sample-efficient than random search. No server required. |
| **Objective** | Mean Sharpe ratio | Balances return and risk. Averaged across 5 eval episodes to reduce per-trial variance. |
| **On-policy budget** | 30 trials × 30,000 steps | PPO/LSTM-PPO collect rollouts then update in batches — low gradient work per env step. 30k steps is sufficient for ranking. |
| **Off-policy budget** | 30 trials × 10,000 steps | SAC/TD3 do one gradient update per env step. At 10k steps with `gradient_steps=1` that is ≈10k gradient updates — more total gradient work than PPO gets at 30k steps. Relative config ranking stabilises faster for off-policy methods. |
| **`gradient_steps` (off-policy)** | Fixed at 1 (not tuned) | Tuning `gradient_steps` ∈ [1,4] combined with `train_freq=1` produced up to 120k gradient updates per trial at 30k steps, causing 7–9 min/trial vs PPO's ≈30 s/trial. Fixing at 1 (the SB3 default) eliminates this variance with no meaningful loss of search coverage. |
| **Persistence** | SQLite (`studies.db`) | One file per run; all trials stored. Re-running the same command resumes automatically. |
| **Parallelism** | Sequential (default) | SQLite supports multi-process via Optuna's RDB storage if needed. |
| **Pruning** | Disabled | Mid-trial evaluation overhead is not justified at the search budgets used here. |

---

## 3. Hyperparameter Search Spaces

### PPOAgent

| Parameter | Type | Range |
|-----------|------|-------|
| `learning_rate` | log-uniform | [1e-5, 1e-2] |
| `n_steps` | categorical | 512, 1024, 2048 |
| `batch_size` | categorical | 64, 128, 256, 512 |
| `n_epochs` | int | [2, 10] |
| `gamma` | uniform | [0.9, 1.0] |
| `ent_coef` | log-uniform | [1e-4, 0.1] |
| `clip_range` | categorical | 0.1, 0.2, 0.3 |
| `gae_lambda` | uniform | [0.8, 1.0] |

### DeepPPOAgent

All PPO parameters plus:

| Parameter | Type | Options |
|-----------|------|---------|
| `net_arch` | categorical | `[128,128]`, `[256,256]`, `[256,256,128]`, `[256,256,256]` |

### LSTMPPOAgent

| Parameter | Type | Range / Options |
|-----------|------|----------------|
| `learning_rate` | log-uniform | [1e-5, 1e-2] |
| `n_steps` | categorical | 256, 512, 1024 |
| `batch_size` | categorical | 64, 128, 256 |
| `n_epochs` | int | [2, 10] |
| `gamma` | uniform | [0.9, 1.0] |
| `ent_coef` | log-uniform | [1e-5, 0.05] |
| `clip_range` | categorical | 0.1, 0.2, 0.3 |
| `gae_lambda` | uniform | [0.8, 1.0] |
| `lstm_hidden_size` | categorical | 32, 64, 128, 256 |
| `n_lstm_layers` | int | [1, 2] |
| `net_arch` (post-LSTM MLP) | categorical | `[32,32]`, `[64,64]`, `[128,128]` |

### SACAgent

> **Training budget: 10,000 steps per trial** (vs 30,000 for on-policy agents).
> `gradient_steps` is fixed at 1 — see Design Decisions above.

| Parameter | Type | Range / Options |
|-----------|------|----------------|
| `learning_rate` | log-uniform | [1e-5, 1e-2] |
| `buffer_size` | categorical | 50,000; 100,000; 200,000 |
| `batch_size` | categorical | 64, 128, 256, 512 |
| `tau` | log-uniform | [0.001, 0.05] |
| `gamma` | uniform | [0.9, 1.0] |
| `train_freq` | int | [1, 4] |
| `gradient_steps` | fixed | 1 |

### LSTMSACAgent

> Same budget and `gradient_steps` fix as SACAgent. (LSTMSACAgent is a deep MLP SAC
> with tunable `net_arch` — not a true recurrent agent, as RecurrentSAC is unavailable
> in sb3-contrib.)

All SAC parameters plus:

| Parameter | Type | Options |
|-----------|------|---------|
| `net_arch` | categorical | `[128,128]`, `[256,256]`, `[256,256,128]`, `[256,256,256]` |

### TD3Agent

> **Training budget: 10,000 steps per trial.** `gradient_steps` fixed at 1.
> TD3's twin-critic structure already doubles gradient work vs single-critic agents.

All SAC parameters plus:

| Parameter | Type | Range |
|-----------|------|-------|
| `policy_delay` | int | [1, 4] |
| `target_policy_noise` | uniform | [0.1, 0.4] |
| `target_noise_clip` | uniform | [0.3, 0.7] |

---

## 4. Usage

### Install dependency

```bash
pip install optuna
```

### Tune all 6 agents × 12 environments (72 jobs)

```bash
python scripts/tune_hyperparams.py --run-id tuned_v1
```

### Tune a single agent × environment

```bash
python scripts/tune_hyperparams.py --run-id tuned_v1 --agent TD3Agent --env ABMVanillaEnv
```

### Tune one agent across all environments

```bash
python scripts/tune_hyperparams.py --run-id tuned_v1 --agent TD3Agent
```

### Resume an interrupted session

Re-run the exact same command. Optuna automatically loads the existing study
from `tuning_results/{run_id}/studies.db` and continues from the last trial.

```bash
python scripts/tune_hyperparams.py --run-id tuned_v1  # continues where it stopped
```

### Smoke test (verify plumbing — 2 trials, 500 steps)

```bash
python scripts/tune_hyperparams.py --run-id smoke \
    --n-trials 2 --n-train-steps 500 --n-eval-episodes 2
```

### Export best params to YAML

```bash
python scripts/export_best_params.py --run-id tuned_v1
```

### Retrain with tuned hyperparameters

```bash
python scripts/train_all_rl_agents.py \
    --run-id v3 \
    --tuned-params-dir tuning_results/tuned_v1/best_params
```

---

## 5. Output Structure

```
tuning_results/
└── {run_id}/
    ├── studies.db                          ← Optuna SQLite (all trials)
    ├── tuning_log.csv                      ← One row per trial (live append)
    ├── summary.json                         ← Best result per study
    └── best_params/
        ├── PPOAgent__ABMVanillaEnv.yaml
        ├── PPOAgent__ABMJumpEnv.yaml
        ├── ...
        └── TD3Agent__OUJumpRegimeEnv.yaml
```

### Best params YAML schema

```yaml
agent: TD3Agent
env: ABMVanillaEnv
run_id: tuned_v1
n_trials_completed: 30
best_trial: 17
best_sharpe: 0.2134
best_mean_pnl: 4.82
best_std_pnl: 22.6
search_budget_steps: 30000
params:
  total_timesteps: 200000      # full training budget (not search budget)
  learning_rate: 0.000342
  buffer_size: 100000
  batch_size: 256
  tau: 0.0081
  gamma: 0.987
  train_freq: 1
  gradient_steps: 2
  policy_delay: 2
  target_policy_noise: 0.21
  target_noise_clip: 0.45
  learning_starts: 100
  use_vec_env: false
  verbose: 0
```

Note: `total_timesteps` in the YAML is the **full** training budget (from
`configs/tuning_config.yaml → full_timesteps`), not the 30k search budget.
This value is used by `train_all_rl_agents.py --tuned-params-dir`.

---

## 6. Methodological Note (Thesis)

**Why Bayesian search over simulated data is valid:**
Simulated environments have no held-out test distribution that could be
overfit. Each episode is drawn from the same generative process regardless of
whether it is used for tuning or evaluation. The main risk — overfitting to the
simulator's specific parameter values — is mitigated by:

1. Tuning each agent × environment pair independently (no cross-environment
   generalisation is assumed).
2. Using a separate evaluation environment instance (different random seed)
   from the training environment within each trial.
3. Averaging Sharpe across 5 evaluation episodes per trial to reduce sampling
   noise in the objective.

The 30-trial budget balances thoroughness against compute time. At 30k steps
per trial, the search budget is ≈20% of the final training budget (150–200k
steps). Agents at this budget have not converged, but the relative ordering
of hyperparameter configurations is preserved well enough for Optuna's TPE
sampler to identify promising regions.
