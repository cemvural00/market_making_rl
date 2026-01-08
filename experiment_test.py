from envs.abm_vanilla import ABMVanillaEnv
from agents.as_agent import ASClosedFormAgent
from experiments.runner import run_experiment

metrics, pnls, qs = run_experiment(
    env_class=ABMVanillaEnv,
    agent_class=ASClosedFormAgent,
    env_config={"S0":100, "sigma":2.0, "dt":0.01, "T":1.0, "A":5.0, "k":1.5},
    agent_config={"gamma":0.1, "sigma":2.0, "k":1.5, "base_delta":1.0},
    train=False,
    n_eval_episodes=2000,
    save_path="results"
)
