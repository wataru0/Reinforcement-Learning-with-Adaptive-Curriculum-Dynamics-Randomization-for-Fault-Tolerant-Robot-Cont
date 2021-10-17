import gym

from rl4robot.agents import RandomActionAgent
from rl4robot.common.evaluating_loop import EvaluatingLoop
from rl4robot.envs import GymEnv


def main():
    env_id = "Walker2d-v2"
    num_episodes = 100

    # 環境を作成
    gym_env = gym.make(env_id)
    env = GymEnv(gym_env)

    # エージェントを作成
    agent = RandomActionAgent(env.spec.action_range)

    # 評価
    loop = EvaluatingLoop(env, agent, num_episodes)
    result = loop.run()

    # 結果の出力
    print("---- result ----")
    episode_return_mean = result.episode_return_mean()
    print(
        f"average episode return (n = {num_episodes}): {episode_return_mean}"
    )
    print("----------------")


if __name__ == "__main__":
    main()
