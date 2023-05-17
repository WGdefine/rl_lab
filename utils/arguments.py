import argparse


def get_args():
    parser = argparse.ArgumentParser()
    # set env and algo
    parser.add_argument("--env_name", default="classic_MountainCar-v0", type=str)
    parser.add_argument("--max_episodes", default=1000, type=int)
    parser.add_argument(
        "--algo", default="dqn", type=str, help="dqn/ppo/a2c/ddpg/ac/ddqn/duelingq/sac"
    )
    parser.add_argument("--runner_type", default="train", type=str, help="train/test")
    parser.add_argument("--reload_param", default=True)
    
    args = parser.parse_args()

    return args
