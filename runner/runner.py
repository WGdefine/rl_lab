from runner.base_runner import BaseRunner
import random
import sys
import numpy as np
from itertools import count

class Runner(BaseRunner):
    def __init__(self, args) -> None:
        super(Runner, self).__init__(args)

    def get_joint_action_eval(self, all_observation):
        joint_action = []
        agents_id_list = self.agents.agent
        for i in range(len(agents_id_list)):
            # agent_id = agents_id_list[i]
            a_obs = all_observation[i]
            each = self.agents.choose_action_to_env(a_obs)
            joint_action.append(each)
        return joint_action

    def run(self):

        # multi_part_agent_ids, actions_space = self.get_players_and_action_space_list()

        for i_epoch in range(1, self.train_config.max_episodes + 1):
            self.env.set_seed(random.randint(0, sys.maxsize))
            state = self.env.reset()
            step = 0
            Gt = 0
            while not self.env.is_terminal():
                step += 1
                joint_act = self.get_joint_action_eval(
                    state
                )
                next_state, reward, done, info = self.env.step(
                    joint_act
                )
                self.insert_memory(state, next_state, reward, np.float32(done))

                state = next_state
                if self.marl:
                    reward = sum(reward)
                Gt += reward
                if not self.learn_terminal and self.runner_type == 'train':
                    if step % self.learn_freq == 0:
                        self.agents.learn(writter=self.writter, epoch=i_epoch)

            if self.learn_terminal and self.runner_type == 'train':
                self.agents.learn(writter=self.writter, epoch=i_epoch)
            print("i_epoch: ", i_epoch, "Gt: ", "%.2f" % Gt)
            # reward_tag = "reward"
            # self.writer.add_scalars(reward_tag, global_step=i_epoch,
            #                         tag_scalar_dict={'return': Gt})
            self.writter.add_scalar("rollout/Rewards", Gt, global_step=i_epoch)

            if i_epoch % self.train_config.save_interval == 0:
                self.agents.save(i_epoch)

            if i_epoch % self.train_config.evaluate_rate == 0 and i_epoch > 1:
                Gt_real = self.evaluate(i_epoch)
                # self.writer.add_scalars('Eval/rewards', global_step=i_epoch,
                #                         tag_scalar_dict={'return': Gt_real})
                self.writter.add_scalar(
                    "rollout/Eval_Reward", Gt_real, global_step=i_epoch
                )

    # def collect(self, step):
    #     return super().collect(step)
    
    def evaluate(self, i_epoch):
        # multi_part_agent_ids, actions_space = self.get_players_and_action_space_list()

        record = []
        for _ in range(10):
            self.env.set_seed(random.randint(0, sys.maxsize))
            state = self.env.reset()
            Gt_real = 0
            for t in count():
                joint_act = self.get_joint_action_eval(
                    state
                )
                next_state, reward, done, _, = self.env.step(joint_act)
                state = next_state
                Gt_real += reward
                if done:
                    record.append(Gt_real)
                    break
        print(
            "===============",
            "i_epoch: ",
            i_epoch,
            "Gt_real: ",
            "%.2f" % np.mean(record),
        )
        return np.mean(record)
    
    
