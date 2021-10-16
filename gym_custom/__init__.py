import gym_custom.envs
from gym.envs.registration import register

register(
    id = "CustomAnt-v0",
    entry_point = "gym_custom.envs.custom_ant_env:CustomAntEnv",
    max_episode_steps = 1000,
    reward_threshold = 6000.0,
)
register(
    id = "TestAnt-v0",
    entry_point = "gym_custom.envs.test_ant_env:TestAntEnv",
    max_episode_steps = 1000,
    reward_threshold = 6000.0,
)
register(
    id = "AblationAnt-v0",
    entry_point = "gym_custom.envs.ablation_ant_env:AblationAntEnv",
    max_episode_steps = 1000,
    reward_threshold = 6000.0,
)

# --------------
# import gym_custom.envs
# from .core import custom_make
# from .registration import register