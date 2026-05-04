def evaluate_agent(env, agent, n_episodes=100, eval_seed=None):
    """
    Evaluate an agent for n_episodes and return PnL and terminal inventory arrays.

    Parameters
    ----------
    eval_seed : int or None
        Base seed for evaluation episodes. Episode i uses seed (eval_seed + i),
        making evaluation fully reproducible. When None, each reset draws a fresh
        random seed (old behaviour — non-reproducible).
        Use env_config["seed"] + 1000 for final evaluation so all agents on the
        same environment see identical price paths.
    """
    pnls = []
    qs = []

    for ep in range(n_episodes):
        ep_seed = (eval_seed + ep) if eval_seed is not None else None
        obs, info = env.reset(seed=ep_seed)
        done = False
        last = info

        # Reset LSTM memory at start of each episode (for fair evaluation)
        if hasattr(agent, 'reset_memory'):
            agent.reset_memory()
        elif hasattr(agent, '_last_lstm_state'):
            agent._last_lstm_state = None

        while not done:
            action = agent.act(obs, info)
            obs, reward, term, trunc, info = env.step(action)
            done = term or trunc
            last = info

        pnls.append(last["pnl"])
        qs.append(last["inventory"])

    return np.array(pnls), np.array(qs)

# kodun devamı

import os
import json
import copy
import numpy as np

from experiments.metrics import compute_basic_metrics
from experiments.plotting import plot_pnl_distribution  # optional


def run_experiment(
    env_class,
    agent_class,
    env_config={},
    agent_config={},
    train=True,
    n_eval_episodes=100,
    save_path="results",
    save_model=False,
    model_save_path="models"
):
    """
    Generic experiment runner for the entire thesis.

    Parameters
    ----------
    env_class : class
        A class from envs/*.py (e.g. ABMVanillaEnv)

    agent_class : class
        A class from agents/*.py (e.g. ASClosedFormAgent or PPOAgent)

    env_config : dict
        Keyword args for env_class

    agent_config : dict
        Keyword args for agent_class

    train : bool
        If True, calls agent.train() when available.

    n_eval_episodes : int
        How many episodes to evaluate the agent on.

    save_path : str
        Folder where results are saved.

    save_model : bool
        If True, saves trained model after training (only for RL agents).

    model_save_path : str
        Base directory for saving models.
    """
    from datetime import datetime

    # -----------------------
    # Instantiate environment
    # -----------------------
    env = env_class(**env_config)

    # -----------------------
    # Instantiate agent
    # -----------------------
    # Make a deep copy of agent_config to avoid mutation issues
    # (agents use config.pop() which modifies the dict)
    agent_config_copy = copy.deepcopy(agent_config)
    
    # Store original total_timesteps before it gets popped
    original_total_timesteps = agent_config.get("total_timesteps", "unknown")
    
    try:
        agent = agent_class(env, config=agent_config_copy)
    except TypeError:
        # Some heuristic agents take only config
        agent = agent_class(config=agent_config_copy)

    # -----------------------
    # Training (if applicable)
    # -----------------------
    trained = False
    if train and hasattr(agent, "train"):
        print(f"Training {agent_class.__name__} on {env_class.__name__}...")
        agent.train()
        trained = True

    # -----------------------
    # Save model (if requested and agent was trained)
    # -----------------------
    env_name = env_class.__name__
    agent_name = agent_class.__name__
    
    if save_model and trained and hasattr(agent, "save"):
        model_dir = os.path.join(model_save_path, env_name, agent_name)
        os.makedirs(model_dir, exist_ok=True)
        
        model_path = os.path.join(model_dir, "model")
        agent.save(model_path)
        
        # Save training metadata
        # Use original agent_config (not mutated copy) and get total_timesteps from agent if needed
        total_ts = original_total_timesteps
        if total_ts == "unknown" and hasattr(agent, "total_timesteps"):
            total_ts = agent.total_timesteps
        
        metadata = {
            "env_class": env_name,
            "agent_class": agent_name,
            "env_config": env_config,
            "agent_config": agent_config,  # Use original, not mutated copy
            "total_timesteps": total_ts,
            "trained_at": datetime.now().isoformat(),
            "model_path": model_path
        }
        
        metadata_path = os.path.join(model_dir, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)
        
        print(f"✓ Model saved to: {model_dir}")

    # -----------------------
    # Evaluation
    # -----------------------
    # eval_seed = base_seed + 1000 ensures all agents on the same environment
    # are evaluated on identical price paths (reproducible, separated from training).
    eval_seed = env_config.get("seed", 123) + 1000
    eval_env = env_class(**env_config)
    pnls, qs = evaluate_agent(eval_env, agent, n_eval_episodes, eval_seed=eval_seed)

    metrics = compute_basic_metrics(pnls)
    metrics["avg_inventory"] = abs(qs).mean()

    # -----------------------
    # Prepare save folder
    # -----------------------
    out_dir = os.path.join(save_path, env_name, agent_name)
    os.makedirs(out_dir, exist_ok=True)

    # Save metrics
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump({k: float(v) for k, v in metrics.items()}, f, indent=4)

    # Save raw pnl data
    np.save(os.path.join(out_dir, "pnls.npy"), pnls)
    np.save(os.path.join(out_dir, "inventory.npy"), qs)

    # Optional: save plots
    try:
        plot_pnl_distribution(pnls, out_dir)
    except:
        pass

    print(f"\n✔ Experiment completed for {agent_name} on {env_name}.")
    print("Saved to:", out_dir)
    print("Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    return metrics, pnls, qs
