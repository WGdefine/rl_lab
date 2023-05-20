from env.wrapper.base_wrapper import *

class FootballWrapper(BaseWrapper):
    def __init__(self, args) -> None:
        super(FootballWrapper, self).__init__(args)
        self.max_step = args.max_step
        self.current_step = 0
        self.terminal = False
        self.is_max_step = args.is_max_step

    def step(self, action):
        self.current_step += 1
        action = self.action_decode(action)
        env_obs, reward, done, info = self.env(action)
        agent_obs = self.get_all_obs(env_obs)
        if self.is_max_step and self.current_step>self.max_step:
            self.terminal = True
        else:
            self.terminal = done
        return agent_obs, reward, done, info
    
    def reset(self):
        self.current_step = 0
        reset_obs = self.env.reset()
        return self.get_all_obs(reset_obs)
    
    def is_terminal(self):
        return self.terminal

    def set_seed(self, seed):
        pass

    def get_all_obs(self, state):
        '''从环境原生obs编码到输入agent的obs'''
        pass

    def get_action_space(self):
        return self.env.get_action_space()
    
    def get_obs_space(self):
        return self.env.get_obs_space()
    
    def action_decode(self, action):
        '''从agent返回action编码到环境输入action'''
        pass