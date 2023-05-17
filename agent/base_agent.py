import torch
from abc import abstractmethod

class BaseAgent(object):
    def __init__(self, args):
        self.args = args
        self.agent = list()


    # inference
    def choose_action_to_env(self, observation, train=True):
        observation_copy = observation.copy()
        observation = observation_copy["obs"]
        agent_id = observation_copy["controlled_player_index"]
        action_from_algo = self.agent[agent_id].choose_action(observation)
        action_to_env = self.action_from_algo_to_env(action_from_algo)
        return action_to_env

    # update algo
    def learn(self):
        for agent in self.agent:
            agent.learn()

    #TODO:实现agent网络参数的保存和加载
    @abstractmethod
    def save(self):
        '''保存网络参数'''
        # for agent in self.agent:
        #     agent.save(save_path, episode)

    @abstractmethod
    def load(self):
        '''加载网络参数'''
        # for agent in self.agent:
        #     agent.load(file)

if __name__ == "__main__":
    test=BaseAgent([])