import argparse


def get_args():
    parser = argparse.ArgumentParser()
    # set env and algo
    parser.add_argument("--env_name", default="classic_CartPole-v0", type=str)
    parser.add_argument(
        "--algo", default="sac", type=str, help="dqn/ppo/a2c/ddpg/ac/ddqn/duelingq/sac"
    )
    parser.add_argument("--runner_type", default="train", type=str, help="train/test")
    parser.add_argument("--reload_param", default=True)
    
    args = parser.parse_args()

    return args
