import importlib
from abc import abstractmethod
import json
from utils.utils import *
class BaseWrapper:
    def __init__(self, args) -> None:
        self.class_literal = args.class_literal
        env_import = importlib.import_module("env.source."+self.class_literal)
        # env_param_path="config/param/env/" + self.env_name + ".json"
        # with open(env_param_path, 'r') as file:
        #     self.env_param_json = json.load(file)[self.env_name]
        # self.env_param = json_to_args(self.env_param_json)
        self.env = getattr(env_import, self.class_literal[0].upper()+self.class_literal[1:].lower())(args)
        self.n_agents = args.n_player

    @abstractmethod
    def reset(self):
        raise NotImplementedError

    @abstractmethod
    def is_terminal(self):
        raise NotImplementedError

    @abstractmethod
    def step(self, action):
        raise NotImplementedError
    
    @abstractmethod
    def set_seed(self, seed):
        raise NotImplementedError
    
    @abstractmethod
    def get_obs_space(self):
        raise NotImplementedError
    
    @abstractmethod
    def get_action_space(self):
        raise NotImplementedError

if __name__ == "__main__":
    test = BaseWrapper([])