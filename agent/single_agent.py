import importlib
from pathlib import Path
import sys
import torch

base_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(base_dir))
from agent.base_agent import BaseAgent


def ini_agents(args):
    agent_file_name = str("algo." + str(args.algo) + "." + str(args.algo))
    agent_file_import = importlib.import_module(agent_file_name)
    agent_class_name = args.algo.upper()

    # 实例化agent
    agent = getattr(agent_file_import, agent_class_name)(args)
    return agent


class SingleRLAgent(BaseAgent):
    def __init__(self, args):
        super(SingleRLAgent, self).__init__(args)
        self.args = args
        self.algo = ini_agents(args)
        self.set_agent()
        self.n_agents = len(self.agent)

    def set_agent(self):
        self.agent.append(self.algo)

    def action_from_algo_to_env(self, joint_action):
        """
        :param joint_action:
        :return: wrapped joint action: one-hot
        """

        joint_action_ = []
        for a in range(self.args.n_player):
            action_a = joint_action["action"]
            if not self.args.action_continuous:  # discrete action space
                each = [0] * self.args.action_space
                each[action_a] = 1
                joint_action_.append(each)
            else:
                joint_action_.append(action_a)  # continuous action space

        return joint_action_
    
    def save(self):
        for agent_id in range(self.n_agents):
            if self.use_single_network:
                policy_model = self.agent[agent_id].policy.model
                torch.save(policy_model.state_dict(), str(self.save_dir) + "/model_agent" + str(agent_id) + ".pt")
            else:
                policy_actor = self.agent[agent_id].policy.actor
                torch.save(policy_actor.state_dict(), str(self.save_dir) + "/actor_agent" + str(agent_id) + ".pt")
                policy_critic = self.agent[agent_id].policy.critic
                torch.save(policy_critic.state_dict(), str(self.save_dir) + "/critic_agent" + str(agent_id) + ".pt")
    
    def load(self):
        for agent_id in range(self.n_agents):
            if self.use_single_network:
                policy_model_state_dict = torch.load(str(self.model_dir) + '/model_agent' + str(agent_id) + '.pt')
                self.policy[agent_id].model.load_state_dict(policy_model_state_dict)
            else:
                policy_actor_state_dict = torch.load(str(self.model_dir) + '/actor_agent' + str(agent_id) + '.pt')
                self.policy[agent_id].actor.load_state_dict(policy_actor_state_dict)
                policy_critic_state_dict = torch.load(str(self.model_dir) + '/critic_agent' + str(agent_id) + '.pt')
                self.policy[agent_id].critic.load_state_dict(policy_critic_state_dict)
