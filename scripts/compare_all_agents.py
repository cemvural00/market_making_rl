"""
Compare all agents across all environments.

This script evaluates:
- RL agents: PPOAgent, DeepPPOAgent, LSTMPPOAgent (loads trained models)
- Analytic agents: ASClosedFormAgent (closed-form optimal)
- Heuristic agents: ASSimpleHeuristicAgent, FixedSpreadAgent, InventoryShiftAgent,
  InventorySpreadScalerAgent, LastLookAgent, MarketOrderOnlyAgent, MidPriceFollowAgent,
  NoiseTraderNormal, NoiseTraderUniform, ZeroIntelligenceAgent

on all 12 environments.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config_loader import load_config
from experiments.runner import run_experiment
from experiments.model_loader import load_trained_agent, model_exists

# Import all environments
from envs.abm_vanilla import ABMVanillaEnv
from envs.abm_jump import ABMJumpEnv
from envs.abm_regime import ABMRegimeEnv
from envs.abm_jump_regime import ABMJumpRegimeEnv
from envs.gbm_vanilla import GBMVanillaEnv
from envs.gbm_jump import GBMJumpEnv
from envs.gbm_regime import GBMRegimeEnv
from envs.gbm_jump_regime import GBMJumpRegimeEnv
from envs.ou_vanilla import OUVanillaEnv
from envs.ou_jump import OUJumpEnv
from envs.ou_regime import OURegimeEnv
from envs.ou_jump_regime import OUJumpRegimeEnv

# Import RL agents
from agents.ppo_agent import PPOAgent
from agents.deep_ppo_agent import DeepPPOAgent
from agents.lstm_agent import LSTMPPOAgent
from agents.sac_agent import SACAgent
from agents.td3_agent import TD3Agent
from agents.lstm_sac_agent import LSTMSACAgent

# Import all other agents (analytic + heuristic)
from agents.as_agent import ASClosedFormAgent, ASSimpleHeuristicAgent
from agents.fixed_spread_agent import FixedSpreadAgent
from agents.inv_shift_agent import InventoryShiftAgent
from agents.inv_spread_scaler_agent import InventorySpreadScalerAgent
from agents.last_look_agent import LastLookAgent
from agents.market_order_agent import MarketOrderOnlyAgent
from agents.mid_price_follow_agent import MidPriceFollowAgent
from agents.noise_trader_normal import NoiseTraderNormal
from agents.noise_trader_uniform import NoiseTraderUniform
from agents.zero_intelligence_agent import ZeroIntelligenceAgent


# Define all environments
ENVIRONMENTS = {
    "ABMVanillaEnv": (ABMVanillaEnv, "abm_vanilla"),
    "ABMJumpEnv": (ABMJumpEnv, "abm_jump"),
    "ABMRegimeEnv": (ABMRegimeEnv, "abm_regime"),
    "ABMJumpRegimeEnv": (ABMJumpRegimeEnv, "abm_jump_regime"),
    "GBMVanillaEnv": (GBMVanillaEnv, "gbm_vanilla"),
    "GBMJumpEnv": (GBMJumpEnv, "gbm_jump"),
    "GBMRegimeEnv": (GBMRegimeEnv, "gbm_regime"),
    "GBMJumpRegimeEnv": (GBMJumpRegimeEnv, "gbm_jump_regime"),
    "OUVanillaEnv": (OUVanillaEnv, "ou_vanilla"),
    "OUJumpEnv": (OUJumpEnv, "ou_jump"),
    "OURegimeEnv": (OURegimeEnv, "ou_regime"),
    "OUJumpRegimeEnv": (OUJumpRegimeEnv, "ou_jump_regime"),
}

# Define RL agents (require trained models)
RL_AGENTS = {
    "PPOAgent": (PPOAgent, "ppo_basic"),
    "DeepPPOAgent": (DeepPPOAgent, "ppo_deep"),
    "LSTMPPOAgent": (LSTMPPOAgent, "lstm_ppo"),
    "SACAgent": (SACAgent, "sac_basic"),
    "TD3Agent": (TD3Agent, "td3_basic"),
    "LSTMSACAgent": (LSTMSACAgent, "lstm_sac"),
}

# Define all other agents (analytic + heuristic, no training needed)
HEURISTIC_AGENTS = {
    "ASClosedFormAgent": (ASClosedFormAgent, "as_closed_form"),
    "ASSimpleHeuristicAgent": (ASSimpleHeuristicAgent, "as_simple_heuristic"),
    "FixedSpreadAgent": (FixedSpreadAgent, "fixed_spread"),
    "InventoryShiftAgent": (InventoryShiftAgent, "inventory_shift"),
    "InventorySpreadScalerAgent": (InventorySpreadScalerAgent, "inventory_spread_scaler"),
    "LastLookAgent": (LastLookAgent, "last_look"),
    "MarketOrderOnlyAgent": (MarketOrderOnlyAgent, "market_order_only"),
    "MidPriceFollowAgent": (MidPriceFollowAgent, "mid_price_follow"),
    "NoiseTraderNormal": (NoiseTraderNormal, "noise_trader_normal"),
    "NoiseTraderUniform": (NoiseTraderUniform, None),  # Uses defaults, no config needed
    "ZeroIntelligenceAgent": (ZeroIntelligenceAgent, None),  # Uses defaults, no config needed
}

# Special configs for GBM environments (AS agent needs different config)
GBM_AGENT_CONFIGS = {
    "ASClosedFormAgent": "as_closed_form_gbm",
}


def get_agent_config(agent_name, env_config_key, all_agent_configs):
    """Get appropriate agent config based on environment type."""
    # Check if this is a GBM environment and agent needs special config
    if "gbm" in env_config_key.lower() and agent_name in GBM_AGENT_CONFIGS:
        config_key = GBM_AGENT_CONFIGS[agent_name]
        return all_agent_configs.get(config_key, {})
    
    # Default: use standard config key
    if agent_name in RL_AGENTS:
        config_key = RL_AGENTS[agent_name][1]
        return all_agent_configs.get(config_key, {}) if config_key else {}
    elif agent_name in HEURISTIC_AGENTS:
        config_key = HEURISTIC_AGENTS[agent_name][1]
        # Some agents don't need configs (use defaults)
        if config_key is None:
            return {}
        return all_agent_configs.get(config_key, {})
    else:
        return {}


def compare_single(env_class, env_config_key, agent_class, agent_name,
                   agent_config, n_eval_episodes=100, skip_if_exists=True,
                   results_base="results", models_base="models"):
    """
    Compare a single agent on a single environment.

    Parameters
    ----------
    env_class : class
        Environment class
    env_config_key : str
        Key in env_configs.yaml
    agent_class : class
        Agent class
    agent_name : str
        Agent class name
    agent_config : dict
        Agent configuration
    n_eval_episodes : int
        Number of episodes for evaluation
    skip_if_exists : bool
        Skip if results already exist
    results_base : str
        Root directory for results (e.g. "results" or "results/v2")
    models_base : str
        Root directory for models (e.g. "models" or "models/v2")
    """
    env_name = env_class.__name__

    # Check if results already exist
    if skip_if_exists:
        results_dir = os.path.join(results_base, env_name, agent_name)
        metrics_file = os.path.join(results_dir, "metrics.json")
        if os.path.exists(metrics_file):
            print(f"⏭️  Skipping {agent_name} on {env_name} (results already exist)")
            return True

    try:
        # Load environment config
        all_env_configs = load_config("configs/env_configs.yaml")
        env_config = all_env_configs.get(env_config_key, {})

        # For RL agents, load trained model
        if agent_name in RL_AGENTS:
            # Check if model exists under the versioned path
            if not model_exists(env_name, agent_name, model_save_path=models_base):
                print(f"⚠️  Model not found for {agent_name} on {env_name}. Skipping.")
                return False

            # Create environment
            env = env_class(**env_config)

            # Load trained agent
            model_path = os.path.join(models_base, env_name, agent_name, "model")
            agent = load_trained_agent(agent_class, env, model_path, agent_config)

            # Run evaluation (no training)
            run_experiment(
                env_class=env_class,
                agent_class=agent_class,
                env_config=env_config,
                agent_config=agent_config,
                train=False,  # Already trained
                n_eval_episodes=n_eval_episodes,
                save_path=results_base,
                save_model=False
            )
        else:
            # Heuristic agent - instantiate and evaluate
            run_experiment(
                env_class=env_class,
                agent_class=agent_class,
                env_config=env_config,
                agent_config=agent_config,
                train=False,  # No training needed
                n_eval_episodes=n_eval_episodes,
                save_path=results_base,
                save_model=False
            )

        return True

    except Exception as e:
        print(f"❌ Error comparing {agent_name} on {env_name}: {e}")
        import traceback
        traceback.print_exc()
        return False


def compare_all(n_eval_episodes=100, skip_if_exists=True, rl_only=False,
                heuristic_only=False, run_id=""):
    """
    Compare all agents on all environments.

    Parameters
    ----------
    n_eval_episodes : int
        Number of episodes for evaluation
    skip_if_exists : bool
        Skip if results already exist
    rl_only : bool
        Only compare RL agents
    heuristic_only : bool
        Only compare heuristic agents
    run_id : str
        Optional run identifier matching the one used during training.
        When set, loads models from models/{run_id}/ and saves results
        to results/{run_id}/.
    """
    results_base = f"results/{run_id}" if run_id else "results"
    models_base = f"models/{run_id}" if run_id else "models"

    # Determine which agents to use
    agents_to_compare = {}
    if not heuristic_only:
        agents_to_compare.update(RL_AGENTS)
    if not rl_only:
        agents_to_compare.update(HEURISTIC_AGENTS)

    total = len(ENVIRONMENTS) * len(agents_to_compare)
    current = 0
    successful = 0
    failed = 0
    skipped = 0

    print(f"\n{'='*60}")
    print(f"COMPARING ALL AGENTS ON ALL ENVIRONMENTS")
    print(f"Total combinations: {total}")
    print(f"RL agents: {len(RL_AGENTS) if not heuristic_only else 0}")
    print(f"Heuristic agents: {len(HEURISTIC_AGENTS) if not rl_only else 0}")
    if run_id:
        print(f"Run ID: {run_id}  →  results: {results_base}/  models: {models_base}/")
    print(f"{'='*60}\n")

    # Load all configs once
    all_env_configs = load_config("configs/env_configs.yaml")
    all_agent_configs = load_config("configs/agent_configs.yaml")

    for env_name, (env_class, env_config_key) in ENVIRONMENTS.items():
        for agent_name, (agent_class, _) in agents_to_compare.items():
            current += 1
            print(f"\n[{current}/{total}] Comparing: {agent_name} on {env_name}")

            # Check if should skip (before calling compare_single)
            if skip_if_exists:
                results_dir = os.path.join(results_base, env_name, agent_name)
                metrics_file = os.path.join(results_dir, "metrics.json")
                if os.path.exists(metrics_file):
                    skipped += 1
                    print(f"⏭️  Skipped (results already exist)")
                    continue

            # Get agent config
            agent_config = get_agent_config(agent_name, env_config_key, all_agent_configs)

            # Compare
            success = compare_single(
                env_class, env_config_key,
                agent_class, agent_name,
                agent_config,
                n_eval_episodes=n_eval_episodes,
                skip_if_exists=False,  # Already checked above
                results_base=results_base,
                models_base=models_base,
            )

            if success:
                successful += 1
            else:
                failed += 1

    # Summary
    print(f"\n{'='*60}")
    print(f"COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"Total: {total}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Skipped: {skipped}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare all agents on all environments")
    parser.add_argument("--no-skip", action="store_true",
                       help="Re-run even if results exist")
    parser.add_argument("--eval-episodes", type=int, default=100,
                       help="Number of evaluation episodes")
    parser.add_argument("--rl-only", action="store_true",
                       help="Only compare RL agents")
    parser.add_argument("--heuristic-only", action="store_true",
                       help="Only compare heuristic agents")
    parser.add_argument("--run-id", type=str, default="",
                       help="Run identifier matching the one used during training "
                            "(e.g. 'v2'). Loads models from models/<run-id>/ and "
                            "saves results to results/<run-id>/.")

    args = parser.parse_args()

    compare_all(
        n_eval_episodes=args.eval_episodes,
        skip_if_exists=not args.no_skip,
        rl_only=args.rl_only,
        heuristic_only=args.heuristic_only,
        run_id=args.run_id,
    )
