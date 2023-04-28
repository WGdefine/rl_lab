import argparse
from utils.arguments import *
from runner.runner import Runner

if __name__ == "__main__":
    args=get_args()
    runner=Runner(args)
    runner.run()
