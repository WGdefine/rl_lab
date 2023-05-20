import gfootball.env as football_env
import numpy as np
import json
import os
import argparse
import copy

class Football:
    def __init__(self, args) -> None:
        self.agent_nums = args.agent_nums
        self.env_core = football_env.create_environment(
            env_name=args.game_name, stacked=False,
            representation='raw',
            logdir='/tmp/rllib_test',
            write_goal_dumps=False, write_full_episode_dumps=True, render=False,
            dump_frequency=0,
            number_of_left_players_agent_controls=self.agent_nums[0],
            number_of_right_players_agent_controls=self.agent_nums[1]
        )

    def step(self, action):
        return self.env_core.step(action)

    def reset(self):
        return self.env_core.reset()

    def get_obs_space(self):
        return self.env_core.observation()[0].keys()
    
    def get_action_space(self):
        return self.env_core.action_space
    
    def get_current_obs(self):
        return self.env_core.observation()
    
    # def get_sorted_next_state(self, next_state):
    #     left_team = next_state[:self.agent_nums[0]]
    #     right_team = next_state[self.agent_nums[0]:]
    #     left_team = sorted(left_team, key=lambda keys: keys['active'])
    #     right_team = sorted(right_team, key=lambda keys: keys['active'])

    #     new_state = []
    #     index = 0
    #     for item in left_team:
    #         each = copy.deepcopy(item)
    #         each["controlled_player_index"] = index
    #         new_state.append(each)
    #         index += 1

    #     for item in right_team:
    #         each = copy.deepcopy(item)
    #         each["controlled_player_index"] = index
    #         new_state.append(each)
    #         index += 1

    #     return new_state
    
# def json_to_args(json_list):
#     parser = argparse.ArgumentParser()
#     t_args = argparse.Namespace()
#     t_args.__dict__.update(json_list)
#     args = parser.parse_args(namespace=t_args)
#     return args

# if __name__ == "__main__":
#     config_path = "config/param/env/config.json"
#     with open(config_path, 'r') as file:
#         config_json = json.load(file)["football_11_vs_11_stochastic"]
#     args = json_to_args(config_json)
#     test = Football(args)
#     print("football obs space:{}".format((test.get_obs_space())))
#     print("football action space:{}".format(test.get_action_space()))