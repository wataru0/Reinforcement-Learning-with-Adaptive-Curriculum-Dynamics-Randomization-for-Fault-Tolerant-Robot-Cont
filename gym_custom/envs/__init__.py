from gym_custom.envs.custom_ant_env import CustomAntEnv
from gym_custom.envs.test_ant_env import TestAntEnv
from gym_custom.envs.ablation_ant_env import AblationAntEnv
# ------------
# from ..registration import register
# register(
#     id="CustomAnt-v0",
#     entry_point="gym_custom.envs.custom_ant_env:CustomAntEnv",
#     max_episode_steps=1000,
#     reward_threshold=6000.0,
#     xml="custom_terrain_ant.xml",
# )