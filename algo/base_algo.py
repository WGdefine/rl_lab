import importlib
from abc import abstractmethod
import json
from utils.utils import *
class BaseAlgo:
    def __init__(self, args) -> None:
        #args传入环境参数，algo_param传入算法参数
        self.algo_name = args.algo
        algo_param_path = "config/param/algo/" + self.algo_name + ".json"
        with open(algo_param_path, 'r') as file:
            self.algo_param_json = json.load(file)[self.algo_name]
        self.algo_param = json_to_args(self.algo_param_json)
    
    @abstractmethod
    def choose_action(self, observation):
        raise NotImplementedError
    
    @abstractmethod
    def learn(self):
        raise NotImplementedError
