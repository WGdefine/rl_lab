import importlib
from agent.base_agent import BaseAgent
import os
import argparse
import torch

class MultiRLAgents(BaseAgent):
    def __init__(self, args):
        super(MultiRLAgents, self).__init__(args)
        self.args = args
        self.algo = self.ini_agents(args)
        self.set_agent()
        self.n_agents = self.algo.n_agents
        self.use_single_network = self.algo.use_single_network

    def set_agent(self):
        self.agent = self.algo.agents

    def action_from_algo_to_env(self, joint_action, id):
        """
        :param joint_action:
        :return: wrapped joint action: one-hot
        """

        joint_action_ = None
        action_a = joint_action["action"]
        if not self.args.action_continuous:  # discrete action space
            each = [0] * self.args.action_space[id]
            each[action_a] = 1
            joint_action_ = each
        else:
            joint_action_ = action_a  # continuous action space

        return joint_action_

    def choose_action_to_env(self, observation, train=True):
        obs_copy = observation.copy()
        obs = obs_copy["obs"]
        agent_id = obs_copy["controlled_player_index"]
        action_from_algo = self.agent[agent_id].choose_action(obs, train)
        action_to_env = self.action_from_algo_to_env(
            {"action": action_from_algo}, agent_id
        )
        return action_to_env

    def learn(self):
        self.algo.learn()

    # def save(self):
    #     for agent_id in range(self.n_agents):
    #         if self.use_single_network:
    #             policy_model = self.agent[agent_id].policy.model
    #             torch.save(policy_model.state_dict(), str(self.save_dir) + "/model_agent" + str(agent_id) + ".pt")
    #         else:
    #             policy_actor = self.agent[agent_id].policy.actor
    #             torch.save(policy_actor.state_dict(), str(self.save_dir) + "/actor_agent" + str(agent_id) + ".pt")
    #             policy_critic = self.agent[agent_id].policy.critic
    #             torch.save(policy_critic.state_dict(), str(self.save_dir) + "/critic_agent" + str(agent_id) + ".pt")
    
    # def load(self):
    #     for agent_id in range(self.num_agents):
    #         if self.use_single_network:
    #             policy_model_state_dict = torch.load(str(self.model_dir) + '/model_agent' + str(agent_id) + '.pt')
    #             self.policy[agent_id].model.load_state_dict(policy_model_state_dict)
    #         else:
    #             policy_actor_state_dict = torch.load(str(self.model_dir) + '/actor_agent' + str(agent_id) + '.pt')
    #             self.policy[agent_id].actor.load_state_dict(policy_actor_state_dict)
    #             policy_critic_state_dict = torch.load(str(self.model_dir) + '/critic_agent' + str(agent_id) + '.pt')
    #             self.policy[agent_id].critic.load_state_dict(policy_critic_state_dict)

if __name__=="__main__":
    param=argparse.ArgumentParser()
    param.add_argument("--algo", default="msac", type=str)
    arg=param.parse_args()
    print("param:{}".format(arg))
