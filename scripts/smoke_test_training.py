"""
Smoke test: train all 6 RL agents on multiple vanilla environments with minimal timesteps.
Purpose: verify the 16D observation space doesn't break any agent's training loop.
"""
import argparse
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("[1/9] Imports loaded")

from configs.config_loader import load_config
print("[2/9] Config loader ready")

from experiments.runner import run_experiment
print("[3/9] Experiment runner ready")

from envs.abm_vanilla import ABMVanillaEnv
from envs.gbm_vanilla import GBMVanillaEnv
from envs.ou_vanilla import OUVanillaEnv
print("[4/9] Env classes loaded")

from agents.ppo_agent import PPOAgent
from agents.deep_ppo_agent import DeepPPOAgent
from agents.lstm_agent import LSTMPPOAgent
from agents.sac_agent import SACAgent
from agents.td3_agent import TD3Agent
from agents.lstm_sac_agent import LSTMSACAgent
print("[5/9] All agent classes loaded")

# PPO needs >= n_steps (128), SAC/TD3 need >= learning_starts (100)
SMOKE_TIMESTEPS = 300
DEFAULT_ENVS = ["abm_vanilla", "gbm_vanilla", "ou_vanilla"]
ENV_CLASS_MAP = {
    "abm_vanilla": ABMVanillaEnv,
    "gbm_vanilla": GBMVanillaEnv,
    "ou_vanilla": OUVanillaEnv,
}

AGENTS = [
    ("PPOAgent",      PPOAgent,     {
        "total_timesteps": SMOKE_TIMESTEPS,
        "n_steps": 128, "batch_size": 64,
        "n_epochs": 2, "verbose": 1,
        "learning_rate": 0.0003, "gamma": 1.0,
    }),
    ("DeepPPOAgent",  DeepPPOAgent, {
        "total_timesteps": SMOKE_TIMESTEPS,
        "n_steps": 128, "batch_size": 64,
        "n_epochs": 2, "verbose": 1,
        "learning_rate": 0.0003, "gamma": 1.0,
        "policy_kwargs": {"net_arch": [64, 64], "activation_fn": "ReLU"},
    }),
    ("LSTMPPOAgent",  LSTMPPOAgent, {
        "total_timesteps": SMOKE_TIMESTEPS,
        "n_steps": 128, "batch_size": 64,
        "n_epochs": 2, "verbose": 1,
        "learning_rate": 0.0003, "gamma": 1.0,
        "policy_kwargs": {
            "lstm_hidden_size": 32, "n_lstm_layers": 1,
            "shared_lstm": False, "net_arch": [16, 16], "activation_fn": "ReLU",
        },
    }),
    ("SACAgent",      SACAgent,     {
        "total_timesteps": SMOKE_TIMESTEPS,
        "learning_starts": 100, "batch_size": 64,
        "buffer_size": 5000, "verbose": 1,
        "learning_rate": 0.0003, "gamma": 1.0,
    }),
    ("TD3Agent",      TD3Agent,     {
        "total_timesteps": SMOKE_TIMESTEPS,
        "learning_starts": 100, "batch_size": 64,
        "buffer_size": 5000, "verbose": 1,
        "learning_rate": 0.0003, "gamma": 1.0,
    }),
    ("LSTMSACAgent",  LSTMSACAgent, {
        "total_timesteps": SMOKE_TIMESTEPS,
        "learning_starts": 100, "batch_size": 64,
        "buffer_size": 5000, "verbose": 1,
        "learning_rate": 0.0003, "gamma": 1.0,
        "policy_kwargs": {"net_arch": [64, 64], "activation_fn": "ReLU"},
    }),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Smoke test for RL agents across multiple environments."
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        default=DEFAULT_ENVS,
        help=(
            "List of environment config keys to run. "
            "Default: all three vanilla envs: abm_vanilla gbm_vanilla ou_vanilla."
        ),
    )
    return parser.parse_args()


def normalize_env_names(env_names):
    normalized = [name.strip().lower() for name in env_names if name.strip()]
    invalid = [name for name in normalized if name not in ENV_CLASS_MAP]
    if invalid:
        raise ValueError(
            "Unknown env config(s): {}. "
            "Valid values: {}".format(
                ", ".join(invalid), ", ".join(sorted(ENV_CLASS_MAP.keys()))
            )
        )
    return normalized


args = parse_args()
selected_envs = normalize_env_names(args.configs)

env_configs = load_config("configs/env_configs.yaml")
print(f"[6/9] Loaded env configs: {', '.join(selected_envs)}")

# Quick obs-shape sanity check before training
for env_name in selected_envs:
    env_cls = ENV_CLASS_MAP[env_name]
    env_cfg = env_configs.get(env_name, {})
    if not env_cfg:
        raise ValueError(f"No configuration found for env '{env_name}'")
    env_check = env_cls(**env_cfg)
    obs, _ = env_check.reset()
    print(f"[7/9] Obs shape check for {env_name}: {obs.shape}  (expected (16,))")
    assert obs.shape == (16,), f"Wrong obs shape for {env_name}: {obs.shape}"

results = []

total_runs = len(selected_envs) * len(AGENTS)
print(
    f"\n[8/9] Starting training loop — "
    f"{len(selected_envs)} env(s) × {len(AGENTS)} agents × {SMOKE_TIMESTEPS} timesteps"
)
print("\n" + "=" * 65)
print(
    f"SMOKE TEST — {', '.join(selected_envs)}; "
    f"16D obs, {SMOKE_TIMESTEPS} timesteps/agent"
)
print("=" * 65)

t_total = time.time()
run_index = 1
for env_name in selected_envs:
    env_cls = ENV_CLASS_MAP[env_name]
    env_cfg = env_configs[env_name]

    for name, cls, cfg in AGENTS:
        print(f"\n{'─'*65}")
        print(f"[{run_index}/{total_runs}] ▶  {env_name} / {name}")
        print(f"    total_timesteps={cfg['total_timesteps']}  batch_size={cfg['batch_size']}")
        print(f"{'─'*65}")
        sys.stdout.flush()

        t0 = time.time()
        try:
            print(f"    → instantiating env + agent...")
            sys.stdout.flush()
            metrics, pnls, qs = run_experiment(
                env_class=env_cls,
                agent_class=cls,
                env_config=env_cfg,
                agent_config=cfg,
                train=True,
                n_eval_episodes=5,
                save_path="results/smoke_test",
                save_model=False,
            )
            elapsed = time.time() - t0
            results.append(
                {
                    "env": env_name,
                    "agent": name,
                    "status": "PASS",
                    "elapsed": elapsed,
                    "metrics": metrics,
                }
            )
            print(f"\n✅  [{run_index}/{total_runs}] {env_name} / {name} PASSED in {elapsed:.1f}s")
            print(f"    mean_pnl   = {metrics.get('mean', float('nan')):.4f}")
            print(f"    std_pnl    = {metrics.get('std', float('nan')):.4f}")
            print(f"    sharpe     = {metrics.get('sharpe', float('nan')):.4f}")
            print(f"    avg_|inv|  = {metrics.get('avg_inventory', float('nan')):.4f}")
        except Exception as e:
            elapsed = time.time() - t0
            results.append(
                {
                    "env": env_name,
                    "agent": name,
                    "status": "FAIL",
                    "elapsed": elapsed,
                    "error": str(e),
                }
            )
            print(f"\n❌  [{run_index}/{total_runs}] {env_name} / {name} FAILED in {elapsed:.1f}s")
            print(f"    Error: {e}")
            traceback.print_exc()

        sys.stdout.flush()
        run_index += 1


total_elapsed = time.time() - t_total

print("\n" + "=" * 65)
print("SUMMARY")
print("=" * 65)
for r in results:
    tag = "✅ PASS" if r["status"] == "PASS" else "❌ FAIL"
    extra = f"  →  {r['error']}" if r["status"] == "FAIL" else ""
    print(f"  {tag}  {r['env']:<12s}  {r['agent']:<20s}  {r['elapsed']:.1f}s{extra}")

n_pass = sum(1 for r in results if r["status"] == "PASS")
n_fail = sum(1 for r in results if r["status"] == "FAIL")
print(f"\n  {n_pass}/{len(results)} runs passed   total={total_elapsed:.1f}s")
print("=" * 65 + "\n")
