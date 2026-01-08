# # # from agents.as_agent import ASAnalyticalAgent
# # # from envs.abm_vanilla import ABMVanillaEnv
# # #
# # # env = ABMVanillaEnv(S0=100, sigma=2.0, seed=42)
# # # agent = ASAnalyticalAgent(config={"gamma": 0.1})
# # #
# # # obs, info = env.reset()
# # # done = False
# # #
# # # while not done:
# # #     action = agent.act(obs, info)
# # #     obs, reward, term, trunc, info = env.step(action)
# # #     done = term or trunc
# # #
# # # print("Final PnL:", info["pnl"])
# #
# # from envs.abm_vanilla import ABMVanillaEnv
# # from agents.as_agent import ASClosedFormAgent
# #
# # env = ABMVanillaEnv(
# #     S0=100,
# #     sigma=2.0,
# #     dt=0.01,
# #     T=1.0,
# #     A=5.0,
# #     k=1.5,
# #     base_delta=1.0,
# #     max_inventory=20,
# #     seed=42,
# # )
# #
# # as_agent = ASClosedFormAgent(config={
# #     "gamma": 0.1,
# #     "sigma": 2.0,      # match env.sigma
# #     "k": 1.5,          # match env.k
# #     "base_delta": 1.0, # match env.base_delta
# #     "max_inventory": 20,
# #     "T": 1.0
# # })
# #
# # obs, info = env.reset()
# # done = False
# # while not done:
# #     action = as_agent.act(obs, info)
# #     obs, reward, term, trunc, info = env.step(action)
# #     done = term or trunc
# #
# # print("AS final PnL:", info["pnl"], "inventory:", info["inventory"])
#
# from agents.lstm_agent import LSTMPPOAgent
# from envs.abm_vanilla import ABMVanillaEnv
# from experiments.runner import run_experiment
#
# env_config = {"S0":100, "sigma":2.0, "dt":0.01, "T":1.0, "A":5.0, "k":1.5}
# agent_config = {
#     "total_timesteps": 40, #400k idi azalttım
#     "learning_rate": 3e-4,
#     "gamma": 1.0,
#     "n_steps": 512,
#     "batch_size": 128,
#     "policy_kwargs": {
#         "lstm_hidden_size": 128,
#         "n_lstm_layers": 1,
#         "net_arch": [64, 64],
#     },
#     "verbose": 1,
# }
#
# run_experiment(
#     env_class=ABMVanillaEnv,
#     agent_class=LSTMPPOAgent,
#     env_config=env_config,
#     agent_config=agent_config,
#     train=True,
#     n_eval_episodes=200
# )

from agents.lstm_agent import LSTMPPOAgent
from envs.abm_vanilla import ABMVanillaEnv
from experiments.runner import run_experiment
from configs.config_loader import load_yaml_config

env_cfg = load_yaml_config("configs/hyperparams/abm_env.yaml")
agent_cfg = load_yaml_config("configs/hyperparams/lstm_ppo.yaml")

run_experiment(
    env_class=ABMVanillaEnv,
    agent_class=LSTMPPOAgent,
    env_config=env_cfg,
    agent_config=agent_cfg,
    train=True,
    n_eval_episodes=20
)
