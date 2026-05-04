"""
Optuna SQLite storage helpers.

Each agent × environment pair gets its own named study within a single
SQLite file per run. Studies use load_if_exists=True, so re-running with
the same --run-id automatically resumes from the last completed trial.

Study naming convention: "{AgentName}__{EnvName}"  (double underscore)
"""

import os
import optuna


def get_db_path(tuning_results_dir, run_id):
    """Return path to the SQLite database for this run (creates directory if needed)."""
    run_dir = os.path.join(tuning_results_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)
    return os.path.join(run_dir, "studies.db")


def study_name(agent_name, env_name):
    """Canonical study name for an agent × environment combination."""
    return f"{agent_name}__{env_name}"


def get_or_create_study(db_path, agent_name, env_name, direction="maximize"):
    """
    Load an existing Optuna study or create a new one.

    Uses SQLite storage so all trials persist on disk. Re-running with the
    same db_path and study name continues from the last completed trial.

    Parameters
    ----------
    db_path : str
        Path to the SQLite file (created automatically if absent).
    agent_name : str
        Agent class name (e.g. "TD3Agent").
    env_name : str
        Environment class name (e.g. "ABMVanillaEnv").
    direction : str
        "maximize" (Sharpe, default) or "minimize".

    Returns
    -------
    optuna.Study
    """
    storage_url = f"sqlite:///{db_path}"
    name = study_name(agent_name, env_name)

    study = optuna.create_study(
        study_name=name,
        storage=storage_url,
        direction=direction,
        load_if_exists=True,                         # ← resume if already started
        sampler=optuna.samplers.TPESampler(seed=42), # reproducible Bayesian search
    )
    return study


def count_completed_trials(study):
    """Return the number of successfully completed trials in a study."""
    return sum(
        1 for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
    )


def list_studies(db_path):
    """Return the names of all studies stored in the given SQLite database."""
    if not os.path.exists(db_path):
        return []
    return optuna.get_all_study_names(storage=f"sqlite:///{db_path}")
