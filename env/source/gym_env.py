import gymnasium as gym
class Gym_env:
    def __init__(self, args) -> None:
        self.core = gym.make(args.game_name)
        self.seed = 0

    def reset(self):
        return self.core.reset(seed=self.seed)

    def step(self, action):
        return self.core.step(action)
    
    def set_seed(self, seed):
        self.seed = seed
    
    def get_obs_space(self):
        return self.core.observation_space
    
    def get_action_space(self):
        return self.core.action_space
    
    
    
