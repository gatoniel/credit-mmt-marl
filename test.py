import numpy as np
from pettingzoo.test import parallel_api_test

from credit_mmt_marl.SimpleCreditEnv import SimpleCreditEnvV0

env = SimpleCreditEnvV0()

# parallel_api_test(env, num_cycles=1_000_000)

RES = "resources"
ACC = "accounts"


def policy(obs, agent):
    # sell half of capital
    if agent.endswith("0"):
        buy_ = np.array([1, 0])
    else:
        buy_ = np.array([0, 1])

    buy = max(np.max(obs[ACC] / 2), 1)

    return {
        "buy capital": [buy / 2],
        "buy capital price": 2 * buy_,
        "buy goods": [buy],
        "buy goods price": buy_,
        "sell capital": [obs[RES][1] / 10],
        "sell capital price": 2 * np.ones_like(obs[ACC]),
        "sell goods": [obs[RES][0]],
        "sell goods price": np.ones_like(obs[ACC]),
        "sell traded capital": [
            0.0,
        ],
        "use capital": [obs[RES][3]],
        "use goods": [obs[RES][0]],
    }


observations, infos = env.reset()

obs = [observations]

while env.agents:
    print("\n\n", env.timestep)
    actions = {
        agent: policy(observations[agent], agent) for agent in env.agents
    }

    observations, rewards, terminations, truncations, infos = env.step(actions)
    obs.append(observations)
    print(observations)

env.close()
