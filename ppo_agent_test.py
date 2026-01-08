from envs.abm_vanilla import ABMVanillaEnv
from agents.as_agent import ASClosedFormAgent
from agents.ppo_agent import PPOAgent
import numpy as np


# 1) Create env
env_train = ABMVanillaEnv(
    S0=100,
    sigma=2.0,
    dt=0.01,
    T=1.0,
    A=5.0,
    k=1.5,
    base_delta=1.0,
    max_inventory=20,
    seed=123,
)

# 2) Create PPO agent
ppo_config = {
    "total_timesteps": 200_000,
    "use_vec_env": False,       # set True to wrap in DummyVecEnv
    "learning_rate": 3e-4,
    "gamma": 1.0,
    "n_steps": 1024,
    "batch_size": 256,
    "verbose": 1,
}

ppo_agent = PPOAgent(env_train, config=ppo_config)

# 3) Train
ppo_agent.train()

# 4) Evaluate vs ASClosedFormAgent on a fresh env
env_eval = ABMVanillaEnv(
    S0=100,
    sigma=2.0,
    dt=0.01,
    T=1.0,
    A=5.0,
    k=1.5,
    base_delta=1.0,
    max_inventory=20,
    seed=999,
)

as_agent = ASClosedFormAgent(config={
    "gamma": 0.1,
    "sigma": 2.0,
    "k": 1.5,
    "base_delta": 1.0,
    "max_inventory": 20,
    "T": 1.0,
})


def rollout(env, agent, n_episodes=100):
    pnls = []
    qs = []
    for _ in range(n_episodes):
        obs, info = env.reset()
        done = False
        last_info = info
        while not done:
            action = agent.act(obs, info)
            obs, reward, term, trunc, info = env.step(action)
            done = term or trunc
            last_info = info
        pnls.append(last_info["pnl"])
        qs.append(last_info["inventory"])
    return np.array(pnls), np.array(qs)


rl_pnls, rl_qs = rollout(env_eval, ppo_agent, n_episodes=100)
as_pnls, as_qs = rollout(env_eval, as_agent, n_episodes=100)

print("RL   mean PnL:", rl_pnls.mean(), "std:", rl_pnls.std())
print("AS   mean PnL:", as_pnls.mean(), "std:", as_pnls.std())
