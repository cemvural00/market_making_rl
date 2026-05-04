"""
Export best hyperparameters from completed Optuna studies.

For each completed agent × environment study, extracts the best trial's
parameters and writes them to:

    {run_dir}/best_params/{AgentName}__{EnvName}.yaml

Also writes a consolidated summary.json with best Sharpe + trial index for
every study, and a tuning_log.csv is maintained live during tuning.
"""

import os
import json

import yaml
import optuna

from tuning.storage import get_db_path, list_studies
from tuning.search_spaces import reconstruct_policy_kwargs


# Full training timesteps defaults (overridden by full_timesteps_map argument)
_DEFAULT_FULL_TS = {
    "PPOAgent":     150_000,
    "DeepPPOAgent": 200_000,
    "LSTMPPOAgent": 200_000,
    "SACAgent":     200_000,
    "LSTMSACAgent": 200_000,
    "TD3Agent":     200_000,
}


def _params_to_agent_config(agent_name, trial_params, full_timesteps_map):
    """
    Convert raw Optuna trial.params to an agent-ready config dict.

    Optuna stores primitive values for categorical choices (e.g. the string
    key for net_arch). This function reconstructs nested structures like
    policy_kwargs and sets total_timesteps to the full training budget.

    Parameters
    ----------
    agent_name : str
    trial_params : dict
        Copy of trial.params (will be modified in-place).
    full_timesteps_map : dict
        Maps agent name → full training budget.

    Returns
    -------
    dict  Agent config ready for agent_class(env, config=...).
    """
    params = dict(trial_params)  # work on a copy

    # Inject full training budget (not the short search budget)
    ts_map = {**_DEFAULT_FULL_TS, **(full_timesteps_map or {})}
    params["total_timesteps"] = ts_map.get(agent_name, 200_000)

    # Reconstruct policy_kwargs for agents that use them
    policy_kwargs = reconstruct_policy_kwargs(agent_name, params)
    if policy_kwargs is not None:
        params["policy_kwargs"] = policy_kwargs

    # Restore non-tuned fixed defaults that agents expect
    params.setdefault("use_vec_env", False)
    params.setdefault("verbose", 0)
    if agent_name in ("SACAgent", "LSTMSACAgent"):
        params.setdefault("ent_coef", "auto")
        params.setdefault("learning_starts", 100)
    if agent_name == "TD3Agent":
        params.setdefault("learning_starts", 100)

    return params


def export_best_params(run_id, tuning_results_dir="tuning_results",
                       full_timesteps_map=None, verbose=True):
    """
    Read all studies from the run's SQLite database and save best params.

    Parameters
    ----------
    run_id : str
        Tuning run identifier (e.g. "tuned_v1").
    tuning_results_dir : str
        Root directory for tuning outputs.
    full_timesteps_map : dict or None
        Maps agent class name → full training timesteps.
        When provided, overrides the defaults in _DEFAULT_FULL_TS.
    verbose : bool
        Print progress per study.

    Returns
    -------
    dict
        Summary keyed by study name: best_sharpe, best_trial, n_trials, etc.
    """
    db_path = get_db_path(tuning_results_dir, run_id)
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"No studies database found at: {db_path}\n"
                                f"Run tune_hyperparams.py first.")

    run_dir = os.path.join(tuning_results_dir, run_id)
    best_params_dir = os.path.join(run_dir, "best_params")
    os.makedirs(best_params_dir, exist_ok=True)

    storage_url = f"sqlite:///{db_path}"
    study_names = list_studies(db_path)

    summary = {}
    exported = 0
    skipped = 0

    if verbose:
        print(f"\nExporting best params from: {db_path}")
        print(f"Studies found: {len(study_names)}\n")

    for sname in sorted(study_names):
        try:
            study = optuna.load_study(study_name=sname, storage=storage_url)
        except Exception as exc:
            if verbose:
                print(f"  ⚠  Could not load study {sname}: {exc}")
            skipped += 1
            continue

        completed = [t for t in study.trials
                     if t.state == optuna.trial.TrialState.COMPLETE]
        if not completed:
            if verbose:
                print(f"  ⏭  {sname}: no completed trials")
            skipped += 1
            continue

        best = study.best_trial

        # Parse agent / env from study name (separator: __)
        parts = sname.split("__", 1)
        agent_name = parts[0]
        env_name   = parts[1] if len(parts) > 1 else "Unknown"

        # Build agent-ready config
        agent_config = _params_to_agent_config(
            agent_name, dict(best.params), full_timesteps_map
        )

        output = {
            "agent":              agent_name,
            "env":                env_name,
            "run_id":             run_id,
            "n_trials_completed": len(completed),
            "best_trial":         best.number,
            "best_sharpe":        round(best.value, 6),
            "best_mean_pnl":      round(best.user_attrs.get("mean_pnl", float("nan")), 4),
            "best_std_pnl":       round(best.user_attrs.get("std_pnl", float("nan")), 4),
            "search_budget_steps": best.user_attrs.get("n_train_steps", "unknown"),
            "params":             agent_config,
        }

        yaml_path = os.path.join(best_params_dir, f"{agent_name}__{env_name}.yaml")
        with open(yaml_path, "w") as f:
            yaml.dump(output, f, default_flow_style=False, sort_keys=False)

        summary[sname] = {
            "agent":              agent_name,
            "env":                env_name,
            "n_trials_completed": len(completed),
            "best_trial":         best.number,
            "best_sharpe":        round(best.value, 6),
            "best_mean_pnl":      round(best.user_attrs.get("mean_pnl", float("nan")), 4),
            "yaml_path":          yaml_path,
        }
        exported += 1

        if verbose:
            print(f"  ✓  {agent_name} / {env_name}: "
                  f"trial={best.number}  sharpe={best.value:.4f}  "
                  f"n_trials={len(completed)}")

    # Write consolidated summary
    summary_path = os.path.join(run_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    if verbose:
        print(f"\nExported  : {exported}")
        print(f"Skipped   : {skipped}")
        print(f"Best params: {best_params_dir}/")
        print(f"Summary    : {summary_path}")

    return summary
