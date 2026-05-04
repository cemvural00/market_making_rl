"""
Train all RL agents on all environments and save models.

This script trains:
- PPOAgent
- DeepPPOAgent
- LSTMPPOAgent

on all 12 environments:
- ABM: vanilla, jump, regime, jump_regime
- GBM: vanilla, jump, regime, jump_regime
- OU: vanilla, jump, regime, jump_regime
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config_loader import load_config
from experiments.runner import run_experiment

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

# Import all RL agents
from agents.ppo_agent import PPOAgent
from agents.deep_ppo_agent import DeepPPOAgent
from agents.lstm_agent import LSTMPPOAgent
from agents.sac_agent import SACAgent
from agents.td3_agent import TD3Agent
from agents.lstm_sac_agent import LSTMSACAgent


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

# Define all RL agents with their config keys
RL_AGENTS = {
    "PPOAgent": (PPOAgent, "ppo_basic"),
    "DeepPPOAgent": (DeepPPOAgent, "ppo_deep"),
    "LSTMPPOAgent": (LSTMPPOAgent, "lstm_ppo"),
    "SACAgent": (SACAgent, "sac_basic"),
    "TD3Agent": (TD3Agent, "td3_basic"),
    "LSTMSACAgent": (LSTMSACAgent, "lstm_sac"),
}


def train_single(env_class, env_config_key, agent_class, agent_config_key,
                 n_eval_episodes=100, skip_if_exists=True,
                 results_base="results", models_base="models",
                 tuned_params_dir=None):
    """
    Train a single agent on a single environment.

    Parameters
    ----------
    env_class : class
        Environment class
    env_config_key : str
        Key in env_configs.yaml
    agent_class : class
        Agent class
    agent_config_key : str
        Key in agent_configs.yaml
    n_eval_episodes : int
        Number of episodes for evaluation after training
    skip_if_exists : bool
        Skip training if model already exists
    results_base : str
        Root directory for saving results (e.g. "results" or "results/v2")
    models_base : str
        Root directory for saving models (e.g. "models" or "models/v2")
    tuned_params_dir : str or None
        Path to a directory containing tuned YAML files
        (e.g. "tuning_results/tuned_v1/best_params").
        When set, loads "{AgentName}__{EnvName}.yaml" from this directory
        and overlays its "params" block onto the default agent config.
    """
    env_name = env_class.__name__
    agent_name = agent_class.__name__

    # Check if model already exists
    if skip_if_exists:
        from experiments.model_loader import model_exists
        if model_exists(env_name, agent_name, model_save_path=models_base):
            print(f"⏭️  Skipping {agent_name} on {env_name} (model already exists)")
            return True

    try:
        # Load configs
        all_env_configs = load_config("configs/env_configs.yaml")
        all_agent_configs = load_config("configs/agent_configs.yaml")

        env_config = all_env_configs.get(env_config_key, {})
        agent_config = all_agent_configs.get(agent_config_key, {})

        # Overlay tuned hyperparameters when --tuned-params-dir is provided
        if tuned_params_dir:
            import yaml
            tuned_yaml = os.path.join(tuned_params_dir, f"{agent_name}__{env_name}.yaml")
            if os.path.exists(tuned_yaml):
                with open(tuned_yaml) as f:
                    tuned = yaml.safe_load(f)
                agent_config.update(tuned.get("params", {}))
                print(f"  ↳ Loaded tuned params from: {tuned_yaml}")
            else:
                print(f"  ⚠  No tuned params found at {tuned_yaml}, using defaults")

        # Run training experiment
        print(f"\n{'='*60}")
        print(f"Training: {agent_name} on {env_name}")
        print(f"{'='*60}")

        run_experiment(
            env_class=env_class,
            agent_class=agent_class,
            env_config=env_config,
            agent_config=agent_config,
            train=True,
            n_eval_episodes=n_eval_episodes,
            save_path=results_base,
            save_model=True,
            model_save_path=models_base
        )

        return True

    except Exception as e:
        print(f"❌ Error training {agent_name} on {env_name}: {e}")
        import traceback
        traceback.print_exc()
        return False


def train_all(skip_if_exists=True, n_eval_episodes=100, run_id="",
              tuned_params_dir=None):
    """
    Train all RL agents on all environments.

    Parameters
    ----------
    skip_if_exists : bool
        Skip training if model already exists
    n_eval_episodes : int
        Number of episodes for evaluation after training
    run_id : str
        Optional run identifier. When set, outputs go to
        results/{run_id}/ and models/{run_id}/ so previous
        runs are never overwritten.
    tuned_params_dir : str or None
        Path to a tuning best_params directory (see train_single).
        When set, tuned hyperparameters override YAML defaults for each combo.
    """
    results_base = f"results/{run_id}" if run_id else "results"
    models_base = f"models/{run_id}" if run_id else "models"

    total = len(ENVIRONMENTS) * len(RL_AGENTS)
    current = 0
    successful = 0
    failed = 0
    skipped = 0

    print(f"\n{'='*60}")
    print(f"TRAINING ALL RL AGENTS ON ALL ENVIRONMENTS")
    print(f"Total combinations: {total}")
    if run_id:
        print(f"Run ID: {run_id}  →  results: {results_base}/  models: {models_base}/")
    if tuned_params_dir:
        print(f"Tuned params: {tuned_params_dir}/")
    print(f"{'='*60}\n")

    for env_name, (env_class, env_config_key) in ENVIRONMENTS.items():
        for agent_name, (agent_class, agent_config_key) in RL_AGENTS.items():
            current += 1
            print(f"\n[{current}/{total}] Processing: {agent_name} on {env_name}")

            # Check if should skip
            if skip_if_exists:
                from experiments.model_loader import model_exists
                if model_exists(env_name, agent_name, model_save_path=models_base):
                    skipped += 1
                    print(f"⏭️  Skipped (model exists)")
                    continue

            # Train
            success = train_single(
                env_class, env_config_key,
                agent_class, agent_config_key,
                n_eval_episodes=n_eval_episodes,
                skip_if_exists=False,  # Already checked above
                results_base=results_base,
                models_base=models_base,
                tuned_params_dir=tuned_params_dir,
            )

            if success:
                successful += 1
            else:
                failed += 1

    # Summary
    print(f"\n{'='*60}")
    print(f"TRAINING SUMMARY")
    print(f"{'='*60}")
    print(f"Total: {total}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Skipped: {skipped}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train all RL agents on all environments")
    parser.add_argument("--no-skip", action="store_true",
                       help="Retrain even if model exists")
    parser.add_argument("--eval-episodes", type=int, default=100,
                       help="Number of evaluation episodes after training")
    parser.add_argument("--run-id", type=str, default="",
                       help="Run identifier (e.g. 'v2'). Outputs go to "
                            "results/<run-id>/ and models/<run-id>/ so old "
                            "runs are never overwritten.")
    parser.add_argument("--tuned-params-dir", type=str, default=None,
                       help="Path to a tuning best_params directory "
                            "(e.g. tuning_results/tuned_v1/best_params). "
                            "When set, tuned hyperparameters override YAML "
                            "defaults for each agent x environment combo.")

    args = parser.parse_args()

    train_all(
        skip_if_exists=not args.no_skip,
        n_eval_episodes=args.eval_episodes,
        run_id=args.run_id,
        tuned_params_dir=args.tuned_params_dir,
    )
