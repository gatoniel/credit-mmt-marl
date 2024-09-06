from pettingzoo.test import parallel_api_test

from credit_mmt_marl.SimpleCreditEnv import SimpleCreditEnvV0

env = SimpleCreditEnvV0()

parallel_api_test(env, num_cycles=1_000_000)

print(env.action_space("Player 1").sample())
