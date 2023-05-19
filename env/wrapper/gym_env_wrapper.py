#测试
import numpy as np
from env.wrapper.base_wrapper import BaseWrapper
class Gym_envWrapper(BaseWrapper):
    def __init__(self, args) -> None:
        super().__init__(args)
        self.max_step = args.max_step
        self.step_cnt = 0
        self.done = False
        self.is_act_continuous = args.is_act_continuous

    def reset(self):
        self.step_cnt = 0
        self.done = False
        state, info = self.env.reset()
        observation = self.get_all_obs(state)
        return observation

    def is_terminal(self):
        if self.step_cnt > self.max_step:
            return True
        return self.done

    def step(self, action):
        self.step_cnt += 1
        action = self.action_decode(action)
        # print("env step:{}".format(self.env.step(action)))
        next_state, reward, self.done, truncated,  info = self.env.step(action)
        observation = self.get_all_obs(next_state)
        return observation, reward, self.done, info
    
    def set_seed(self, seed):
        self.env.set_seed(seed)
    
    def get_obs_space(self):
        return self.env.get_obs_space()
    
    def get_action_space(self):
        return self.env.get_action_space()
    
    def get_all_obs(self, state):
        all_obs = []
        all_state = [state]*self.n_agents
        for i in range(len(all_state)):
            each = {"obs": all_state[i], "controlled_player_index": i}
            all_obs.append(each)
        return all_obs
    
    def action_decode(self, joint_action):

        if not self.is_act_continuous:
            return joint_action[0][0].index(1)
        else:
            return joint_action[0][0]

if __name__ == "__main__":
    test = [0,2,1,3]
    print([test]*1)