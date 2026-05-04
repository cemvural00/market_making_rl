"""
Optuna objective function for RL hyperparameter tuning.

Objective: mean Sharpe ratio across n_eval_episodes episodes.
  Sharpe = mean(PnL) / std(PnL)

Per trial:
  1. Sample hyperparameters from the agent's search space.
  2. Train the agent for exactly n_train_steps (no early stopping).
  3. Evaluate on n_eval_episodes fresh episodes.
  4. Return mean Sharpe (higher = better).

Failed trials return -999.0 so Optuna deprioritises that parameter region.
"""

import copy
import traceback

import numpy as np

from tuning.search_spaces import get_search_space_fn
from experiments.runner import evaluate_agent


def make_objective(agent_class, env_class, env_config,
                   n_train_steps, n_eval_episodes):
    """
    Create an Optuna objective closure for a given agent × environment pair.

    Parameters
    ----------
    agent_class : class
        RL agent class (e.g. TD3Agent).
    env_class : class
        Environment class (e.g. ABMVanillaEnv).
    env_config : dict
        Keyword arguments for env_class.
    n_train_steps : int
        Timesteps to train per trial (search budget, shorter than full training).
    n_eval_episodes : int
        Episodes to evaluate per trial; Sharpe is averaged across these.

    Returns
    -------
    callable
        objective(trial) -> float  (mean Sharpe, higher is better)
    """
    sampler_fn = get_search_space_fn(agent_class.__name__)

    def objective(trial):
        # ── 1. Sample hyperparameters ──────────────────────────────────────
        params = sampler_fn(trial)
        params["total_timesteps"] = n_train_steps

        # Store search budget as user attr for later export
        trial.set_user_attr("n_train_steps", n_train_steps)

        try:
            # ── 2. Train ───────────────────────────────────────────────────
            env = env_class(**env_config)
            # Deep copy: agents pop keys from config during __init__
            agent = agent_class(env, config=copy.deepcopy(params))
            agent.train()  # trains for exactly n_train_steps; no early stopping

            # ── 3. Evaluate ────────────────────────────────────────────────
            # tune_eval_seed = base + 2000: separate from final eval (base+1000)
            # and from training. All trials for the same env use the same seed
            # so Sharpe comparisons across trials are apples-to-apples.
            tune_eval_seed = env_config.get("seed", 123) + 2000
            eval_env = env_class(**env_config)
            pnls, _ = evaluate_agent(eval_env, agent, n_episodes=n_eval_episodes,
                                     eval_seed=tune_eval_seed)

            mean_pnl = float(np.mean(pnls))
            std_pnl  = float(np.std(pnls))
            sharpe   = mean_pnl / (std_pnl + 1e-12)

            # Store extra stats for inspection / export
            trial.set_user_attr("mean_pnl",     round(mean_pnl, 6))
            trial.set_user_attr("std_pnl",      round(std_pnl, 6))
            trial.set_user_attr("n_eval_episodes", n_eval_episodes)

            return float(sharpe)

        except Exception as exc:
            trial.set_user_attr("error",     str(exc))
            trial.set_user_attr("traceback", traceback.format_exc())
            return -999.0

    return objective


def make_heuristic_objective(agent_class, env_class, env_config,
                              n_eval_episodes, sampler_fn):
    """
    Optuna objective for non-RL (analytic + heuristic) agents.

    Unlike the RL objective there is no training step: each trial is purely an
    evaluation rollout. This makes trials ~100× cheaper than RL trials.

    Parameters
    ----------
    agent_class : class
        Non-RL agent class (e.g. ASClosedFormAgent).
    env_class : class
        Environment class (e.g. ABMVanillaEnv).
    env_config : dict
        Keyword arguments for env_class.
    n_eval_episodes : int
        Episodes per trial; Sharpe is averaged across these.
    sampler_fn : callable
        From heuristic_search_spaces.py — takes (trial, env_config) → dict.

    Returns
    -------
    callable
        objective(trial) -> float  (mean Sharpe, higher is better)
    """
    def objective(trial):
        params = sampler_fn(trial, env_config)

        try:
            tune_eval_seed = env_config.get("seed", 123) + 2000
            env   = env_class(**env_config)
            agent = agent_class(config=params)              # no env in __init__
            pnls, _ = evaluate_agent(env, agent, n_episodes=n_eval_episodes,
                                     eval_seed=tune_eval_seed)

            mean_pnl = float(np.mean(pnls))
            std_pnl  = float(np.std(pnls))
            sharpe   = mean_pnl / (std_pnl + 1e-12)

            trial.set_user_attr("mean_pnl",        round(mean_pnl, 6))
            trial.set_user_attr("std_pnl",         round(std_pnl, 6))
            trial.set_user_attr("n_eval_episodes", n_eval_episodes)

            return float(sharpe)

        except Exception as exc:
            trial.set_user_attr("error",     str(exc))
            trial.set_user_attr("traceback", traceback.format_exc())
            return -999.0

    return objective
