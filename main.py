import argparse
from utils.arguments import *
from runner.runner import Runner
import json
from env.chooseenv import *

if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--param_load", default="code")
    # load_args = parser.parse_args()
    # if load_args.param_load == "code":
    #     args=get_args()
    #     print("args:{}".format(args))
    # else:
    #     with open(load_args.param_load, "r") as file:
    #         t_args = argparse.Namespace()
    #         t_args.__dict__.update(json.load(file))
    #         args = parser.parse_args(namespace=t_args)

    args = get_args()
    runner = Runner(args)
    runner.run()
