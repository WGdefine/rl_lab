import torch
from abc import abstractmethod
import importlib
import torch

class BaseAgent(object):
    def __init__(self, args):
        self.args = args
        self.agent = list()
        # self.device = "cuda" if torch.cuda.is_available() else "cpu"


    # inference
    def choose_action_to_env(self, observation, train=True):
        observation_copy = observation.copy()
        observation = observation_copy["obs"]
        # print("agent obs:{}".format(observation))
        agent_id = observation_copy["controlled_player_index"]
        action_from_algo = self.agent[agent_id].choose_action(observation)
        action_to_env = self.action_from_algo_to_env(action_from_algo)
        return action_to_env

    # update algo
    def learn(self, **kwargs):
        writter = kwargs.get("writter", None)
        for idx, agent in enumerate(self.agent):
            training_results = agent.learn()
            if writter is not None:
                for tag, value in training_results.items():
                    writter.add_scalar(
                        f"Training/agent {idx} {tag}",
                        value,
                        global_step=kwargs.get("epoch"),
                    )

    #TODO:实现agent网络参数的保存和加载
    # @abstractmethod
    def save(self, episode):
        '''保存网络参数'''
        for agent in self.agent:
            agent.save(episode)
        # raise NotImplementedError

    # @abstractmethod
    def load(self, file):
        '''加载网络参数'''
        for agent in self.agent:
            agent.load(file)
        # raise NotImplementedError

    def ini_agents(self, args):
        agent_file_name = str("algo." + str(args.algo))
        agent_file_import = importlib.import_module(agent_file_name)
        agent_class_name = args.algo.upper()

        # 实例化agent
        agent = getattr(agent_file_import, agent_class_name)(args)
        return agent

if __name__ == "__main__":
    test=BaseAgent([])