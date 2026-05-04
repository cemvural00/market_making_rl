"""
Export best hyperparameters from completed Optuna studies to YAML files.

Run this after tune_hyperparams.py to save the best configs for retraining:

    python scripts/export_best_params.py --run-id tuned_v1

Output
------
tuning_results/{run_id}/best_params/{AgentName}__{EnvName}.yaml
tuning_results/{run_id}/summary.json
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config_loader import load_yaml_config
from tuning.export import export_best_params


def main():
    parser = argparse.ArgumentParser(
        description="Export best Optuna trial params to YAML files for retraining"
    )
    parser.add_argument("--run-id", required=True,
                        help="Tuning run identifier (e.g. 'tuned_v1').")
    parser.add_argument("--tuning-results-dir", default="tuning_results",
                        help="Root directory for tuning outputs (default: tuning_results).")
    args = parser.parse_args()

    full_ts = load_yaml_config("configs/tuning_config.yaml").get("full_timesteps", {})
    export_best_params(
        run_id=args.run_id,
        tuning_results_dir=args.tuning_results_dir,
        full_timesteps_map=full_ts,
        verbose=True,
    )


if __name__ == "__main__":
    main()
