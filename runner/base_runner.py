from env.chooseenv import *

class BaseRunner:
    def __init__(self, args) -> None:
        self.env=make(args.env_name)
        self.run_type=args.run_type

    def run(self):
        pass

    def save(self):
        pass