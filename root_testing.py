# # # # from envs.abm_vanilla import ABMVanillaEnv
# # # #
# # # # env = ABMVanillaEnv(
# # # #     S0=100,
# # # #     sigma=2.0,
# # # #     mu=0.0,
# # # #     T=1.0,
# # # #     dt=0.01,
# # # #     A=5,
# # # #     k=1.5,
# # # #     base_delta=1.0,
# # # #     max_inventory=20,
# # # # )
# # # #
# # # # obs, info = env.reset(seed=42)
# # # # print("Initial obs:", obs)
# # # #
# # # # for _ in range(5):
# # # #     action = env.action_space.sample()
# # # #     obs, reward, terminated, truncated, info = env.step(action)
# # # #     print(obs, reward)
# # # from envs.abm_jump import ABMJumpEnv
# # #
# # # env = ABMJumpEnv(
# # #     S0=100,
# # #     sigma=2.0,
# # #     mu=0.0,
# # #     jump_intensity=0.2,
# # #     jump_mean=0.0,
# # #     jump_std=4.0,
# # #     dt=0.01,
# # #     T=1.0,
# # #     seed=42
# # # )
# # #
# # # obs, info = env.reset()
# # # print("Initial:", obs)
# # #
# # # for _ in range(10):
# # #     action = env.action_space.sample()
# # #     obs, reward, term, trunc, info = env.step(action)
# # #     print(obs, reward)
# # from envs.abm_regime import ABMRegimeEnv
# #
# # env = ABMRegimeEnv(
# #     S0=100,
# #     sigma_low=1.0,
# #     sigma_high=5.0,
# #     mu=0.0,
# #     dt=0.01,
# #     T=1.0,
# #     initial_regime=0,
# #     seed=123
# # )
# #
# # obs, info = env.reset()
# # print("Initial:", obs)
# #
# # for _ in range(10):
# #     action = env.action_space.sample()
# #     obs, reward, term, trunc, info = env.step(action)
# #     print(obs, reward)
#
# from envs.abm_jump_regime import ABMJumpRegimeEnv
#
# env = ABMJumpRegimeEnv(
#     S0=100,
#     sigma_low=1.0,
#     sigma_high=5.0,
#     mu=0.0,
#     jump_intensity=0.2,
#     jump_mean=0.0,
#     jump_std=4.0,
#     dt=0.01,
#     T=1.0,
#     seed=42,
# )
#
# obs, info = env.reset()
# print("Initial obs:", obs)
#
# for _ in range(10):
#     action = env.action_space.sample()
#     obs, reward, term, trunc, info = env.step(action)
#     print(obs, reward)

from agents.base_agent import BaseAgent

try:
    agent = BaseAgent()
    agent.act([0,0,0,0])
except NotImplementedError:
    print("BaseAgent is correctly abstract.")
