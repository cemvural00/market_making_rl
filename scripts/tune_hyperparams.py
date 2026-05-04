"""
Hyperparameter tuning for RL market-making agents using Optuna (TPE sampler).

Objective
---------
Mean Sharpe ratio (mean_pnl / std_pnl) averaged across n_eval_episodes
per trial. All results persist in a SQLite database — re-running the same
command resumes from the last completed trial.

Usage
-----
# Tune all 6 agents x 12 environments (72 jobs, fully resumable)
python scripts/tune_hyperparams.py --run-id tuned_v1

# Tune one agent x one environment
python scripts/tune_hyperparams.py --run-id tuned_v1 --agent TD3Agent --env ABMVanillaEnv

# Tune one agent across all environments
python scripts/tune_hyperparams.py --run-id tuned_v1 --agent TD3Agent

# Smoke test (2 trials, 500 steps — verifies plumbing only)
python scripts/tune_hyperparams.py --run-id smoke --n-trials 2 --n-train-steps 500 --n-eval-episodes 2

# Export best params to YAML after tuning
python scripts/tune_hyperparams.py --run-id tuned_v1 --export-after
"""

import sys
import os
import argparse
import csv
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)  # suppress per-trial spam

from configs.config_loader import load_yaml_config
from tuning.storage import get_db_path, get_or_create_study, count_completed_trials
from tuning.objective import make_objective
from tuning.export import export_best_params

# ── Environments ──────────────────────────────────────────────────────────────
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

# ── RL Agents ─────────────────────────────────────────────────────────────────
from agents.ppo_agent import PPOAgent
from agents.deep_ppo_agent import DeepPPOAgent
from agents.lstm_agent import LSTMPPOAgent
from agents.sac_agent import SACAgent
from agents.td3_agent import TD3Agent
from agents.lstm_sac_agent import LSTMSACAgent

ALL_ENVS = {
    "ABMVanillaEnv":    (ABMVanillaEnv,    "abm_vanilla"),
    "ABMJumpEnv":       (ABMJumpEnv,       "abm_jump"),
    "ABMRegimeEnv":     (ABMRegimeEnv,     "abm_regime"),
    "ABMJumpRegimeEnv": (ABMJumpRegimeEnv, "abm_jump_regime"),
    "GBMVanillaEnv":    (GBMVanillaEnv,    "gbm_vanilla"),
    "GBMJumpEnv":       (GBMJumpEnv,       "gbm_jump"),
    "GBMRegimeEnv":     (GBMRegimeEnv,     "gbm_regime"),
    "GBMJumpRegimeEnv": (GBMJumpRegimeEnv, "gbm_jump_regime"),
    "OUVanillaEnv":     (OUVanillaEnv,     "ou_vanilla"),
    "OUJumpEnv":        (OUJumpEnv,        "ou_jump"),
    "OURegimeEnv":      (OURegimeEnv,      "ou_regime"),
    "OUJumpRegimeEnv":  (OUJumpRegimeEnv,  "ou_jump_regime"),
}

ALL_AGENTS = {
    "PPOAgent":     PPOAgent,
    "DeepPPOAgent": DeepPPOAgent,
    "LSTMPPOAgent": LSTMPPOAgent,
    "SACAgent":     SACAgent,
    "TD3Agent":     TD3Agent,
    "LSTMSACAgent": LSTMSACAgent,
}

# Off-policy agents use a shorter training budget per trial (see tuning_config.yaml).
_OFF_POLICY_AGENTS = {"SACAgent", "TD3Agent", "LSTMSACAgent"}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _append_csv(log_path, row):
    """Append one row to the trial log CSV (writes header on first call)."""
    write_header = not os.path.exists(log_path)
    with open(log_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# ── Core tuning function ──────────────────────────────────────────────────────

def tune_one(agent_name, agent_class, env_name, env_class, env_config_key,
             n_trials, n_train_steps, n_eval_episodes,
             tuning_results_dir, run_id):
    """
    Run Optuna tuning for one agent × environment combination.

    If a study for this pair already exists in the database, resumes from
    the last completed trial. Returns the study object.
    """
    all_env_configs = load_yaml_config("configs/env_configs.yaml")
    env_config = all_env_configs.get(env_config_key, {})

    db_path = get_db_path(tuning_results_dir, run_id)
    study   = get_or_create_study(db_path, agent_name, env_name)

    already_done = count_completed_trials(study)
    remaining    = n_trials - already_done

    print(f"\n{'─'*65}")
    print(f"  Agent  : {agent_name}")
    print(f"  Env    : {env_name}")
    print(f"  Trials : {already_done} done / {n_trials} total  →  {remaining} remaining")
    print(f"  Budget : {n_train_steps:,} train steps × {n_eval_episodes} eval episodes")
    print(f"{'─'*65}")

    if remaining <= 0:
        print("  ✓ Already complete — skipping")
        return study

    log_path  = os.path.join(tuning_results_dir, run_id, "tuning_log.csv")
    objective = make_objective(agent_class, env_class, env_config,
                               n_train_steps, n_eval_episodes)

    def _trial_callback(study, trial):
        """Print one-line progress + append to CSV after each trial."""
        if trial.state != optuna.trial.TrialState.COMPLETE:
            return

        done  = count_completed_trials(study)
        sh    = trial.value if trial.value is not None else float("nan")
        best  = study.best_value if study.best_trial else sh

        # Short param summary for the console
        SHOW = ("learning_rate", "batch_size", "n_steps",
                "buffer_size", "tau", "gamma")
        short = {k: (f"{v:.4g}" if isinstance(v, float) else v)
                 for k, v in trial.params.items() if k in SHOW}
        param_str = "  ".join(f"{k}={v}" for k, v in short.items())

        flag = "★" if sh >= best - 1e-9 else " "
        print(f"  {flag}[{done:>2}/{n_trials}] trial={trial.number:<3}  "
              f"sharpe={sh:+.4f}  best={best:+.4f}  | {param_str}")

        # Append row to CSV log
        row = {
            "run_id":          run_id,
            "agent":           agent_name,
            "env":             env_name,
            "trial":           trial.number,
            "state":           trial.state.name,
            "sharpe":          round(sh, 6),
            "mean_pnl":        round(trial.user_attrs.get("mean_pnl", float("nan")), 4),
            "std_pnl":         round(trial.user_attrs.get("std_pnl",  float("nan")), 4),
            "n_train_steps":   n_train_steps,
            "n_eval_episodes": n_eval_episodes,
            **{f"p_{k}": v for k, v in trial.params.items()},
        }
        _append_csv(log_path, row)

    study.optimize(
        objective,
        n_trials=remaining,
        callbacks=[_trial_callback],
        show_progress_bar=False,
    )

    done = count_completed_trials(study)
    print(f"\n  ✓ Done: {done}/{n_trials} trials completed")
    if study.best_trial:
        bt = study.best_trial
        print(f"  Best → trial #{bt.number}  "
              f"Sharpe={study.best_value:.4f}  "
              f"mean_pnl={bt.user_attrs.get('mean_pnl', float('nan')):.3f}")

    return study


# ── CLI entry point ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Optuna hyperparameter tuning for RL market-making agents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--run-id", required=True,
                        help="Run identifier (e.g. 'tuned_v1'). "
                             "Results go to tuning_results/{run-id}/.")
    parser.add_argument("--agent", default=None,
                        help=f"Tune only this agent. Choices: {list(ALL_AGENTS)}")
    parser.add_argument("--env", default=None,
                        help=f"Tune only this environment. Choices: {list(ALL_ENVS)}")
    parser.add_argument("--n-trials", type=int, default=None,
                        help="Optuna trials per combo (default: from tuning_config.yaml).")
    parser.add_argument("--n-train-steps", type=int, default=None,
                        help="Training timesteps per trial (search budget).")
    parser.add_argument("--n-eval-episodes", type=int, default=None,
                        help="Eval episodes per trial (Sharpe averaged across these).")
    parser.add_argument("--tuning-results-dir", default=None,
                        help="Root directory for tuning outputs.")
    parser.add_argument("--export-after", action="store_true",
                        help="Export best-param YAML files after all tuning completes.")
    args = parser.parse_args()

    # ── Load defaults from config ──────────────────────────────────────────
    tuning_cfg             = load_yaml_config("configs/tuning_config.yaml")["default"]
    n_trials               = args.n_trials        or tuning_cfg["n_trials"]
    n_train_steps          = args.n_train_steps   or tuning_cfg["n_train_steps"]
    n_train_steps_offpol   = tuning_cfg.get("n_train_steps_offpolicy", n_train_steps)
    n_eval_episodes        = args.n_eval_episodes or tuning_cfg["n_eval_episodes"]
    tuning_results_dir     = args.tuning_results_dir or tuning_cfg["tuning_results_dir"]

    # ── Select agents ──────────────────────────────────────────────────────
    if args.agent:
        if args.agent not in ALL_AGENTS:
            print(f"Unknown agent '{args.agent}'. Available: {list(ALL_AGENTS)}")
            sys.exit(1)
        agents_to_run = {args.agent: ALL_AGENTS[args.agent]}
    else:
        agents_to_run = ALL_AGENTS

    # ── Select environments ────────────────────────────────────────────────
    if args.env:
        if args.env not in ALL_ENVS:
            print(f"Unknown env '{args.env}'. Available: {list(ALL_ENVS)}")
            sys.exit(1)
        envs_to_run = {args.env: ALL_ENVS[args.env]}
    else:
        envs_to_run = ALL_ENVS

    total_jobs = len(agents_to_run) * len(envs_to_run)
    job = 0

    print(f"\n{'='*65}")
    print(f"HYPERPARAMETER TUNING  run_id={args.run_id}")
    print(f"  Agents      : {list(agents_to_run)}")
    print(f"  Environments: {list(envs_to_run)}")
    print(f"  Budget (on-policy) : {n_trials} trials × {n_train_steps:,} steps × {n_eval_episodes} eval ep")
    print(f"  Budget (off-policy): {n_trials} trials × {n_train_steps_offpol:,} steps × {n_eval_episodes} eval ep")
    print(f"  Total jobs  : {total_jobs}")
    print(f"  DB          : tuning_results/{args.run_id}/studies.db")
    print(f"  (Re-run same command to resume from last completed trial)")
    print(f"{'='*65}")

    t_start = time.time()

    for agent_name, agent_class in agents_to_run.items():
        steps = n_train_steps_offpol if agent_name in _OFF_POLICY_AGENTS else n_train_steps
        for env_name, (env_class, env_config_key) in envs_to_run.items():
            job += 1
            print(f"\n[{job}/{total_jobs}]", end="")
            tune_one(
                agent_name=agent_name,
                agent_class=agent_class,
                env_name=env_name,
                env_class=env_class,
                env_config_key=env_config_key,
                n_trials=n_trials,
                n_train_steps=steps,
                n_eval_episodes=n_eval_episodes,
                tuning_results_dir=tuning_results_dir,
                run_id=args.run_id,
            )

    elapsed = time.time() - t_start
    print(f"\n{'='*65}")
    print(f"TUNING COMPLETE  ({elapsed / 60:.1f} min total)")
    print(f"Results: tuning_results/{args.run_id}/")
    print(f"{'='*65}")

    if args.export_after:
        print("\nExporting best params to YAML...")
        full_ts = load_yaml_config("configs/tuning_config.yaml").get("full_timesteps", {})
        export_best_params(
            run_id=args.run_id,
            tuning_results_dir=tuning_results_dir,
            full_timesteps_map=full_ts,
        )


if __name__ == "__main__":
    main()
