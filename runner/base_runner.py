from env.chooseenv import *
from tensorboardX import SummaryWriter
from utils.log_path import *
from utils.utils import *
from agent.muti_agent import *
from agent.single_agent import *
import random
# import torch

class BaseRunner:
    def __init__(self, args) -> None:
        #TODO:修改训练参数输入
        self.env = make(args.env_name)
        self.run_type = args.runner_type
        self.n_agents = self.env.n_agents
        self.algo_name = args.algo
        self.runner_type = args.runner_type
        # self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # self.EnvSetting = EnvSettingDefault(
        #     env_name = args.env_name,
        #     obs_space = self.env.get_obs_space(),
        #     action_space = self.env.get_action_space(),
        #     n_player = self.n_agents
        # )
        self.train_config_path = "config/param/train/"
        # print(self.train_config_path+args.env_name+"_"+args.algo+".json")
        if os.path.exists(self.train_config_path+args.env_name+"_"+args.algo+".json"):
            with open(self.train_config_path+args.env_name+"_"+args.algo+".json", 'r') as file:
                self.train_config_json = json.load(file)
            self.train_config = json_to_args(self.train_config_json)
        else:
            self.train_config = self.init_default_train()

        self.run_dir, self.log_dir = make_logpath(args.env_name, args.algo)
        self.writter = SummaryWriter(str(self.run_dir))
        # self.config_dir = os.path.join(os.getcwd(), "config/param")
        # file_name = args.env_name+"_"+args.algo

        # config_dict = self.load_param(self.algo_name, args.reload_config, file_name)
        # self.param = get_paras_from_dict(config_dict)

        self.marl = self.train_config.marl
        self.learn_terminal = self.train_config.learn_terminal
        self.learn_freq = self.train_config.learn_freq

        #TODO:需要设计Agent传入参数
        if self.marl:
            self.agents = MultiRLAgents(self.train_config)
        else:
            self.agents = SingleRLAgent(self.train_config)

        # self.buffer = []
        torch.manual_seed(self.train_config.seed_nn)
        np.random.seed(self.train_config.seed_np)
        random.seed(self.train_config.seed_random)

    def init_default_train(self):
        pass

    def run(self):
        raise NotImplementedError
    
    # def train(self):
    #     pass

    # def collect(self, step):
    #     raise NotImplementedError
    
    def insert_memory(self, states, state_next, reward, done):
        
        for agent_index, agent_i in enumerate(self.agents.agent):
            agent_i.memory.insert("states", agent_index, states[agent_index]["obs"])
            agent_i.memory.insert(
                "states_next", agent_index, state_next[agent_index]["obs"]
            )
            agent_i.memory.insert("rewards", agent_index, reward)
            agent_i.memory.insert("dones", agent_index, np.array(done, dtype=bool))

    def save(self):
        self.agents.save()

    def restore(self):
        self.agents.load()

    def log_train(self, train_infos, total_num_steps): 
        for agent_id in range(self.n_agents):
            for k, v in train_infos[agent_id].items():
                agent_k = "agent%i/" % agent_id + k
                self.writter.add_scalars(agent_k, {agent_k: v}, total_num_steps)

    def log_env(self, env_infos, total_num_steps):
        for k, v in env_infos.items():
            if len(v) > 0:
                self.writter.add_scalars(k, {k: np.mean(v)}, total_num_steps)

    # def load_param(self, algo, reload_config, file_name):
    #     #TODO:需要修改参数读取路径
    #     if (
    #         not reload_config
    #         and not os.path.exists(os.path.join(self.log_dir, file_name + ".json"))
    #     ) or (
    #         reload_config
    #         and not os.path.exists(os.path.join(self.config_dir, file_name + ".json"))
    #         and not os.path.exists(os.path.join(self.log_dir, file_name + ".json"))
    #     ):
    #         paras = RunnerSetting(
    #             algo=algo,
    #             hyperparameters=globals()[str(algo).upper() + "Default"](),
    #             envparameters=self.EnvSetting,
    #             trainingparameters=TrainingDefault(),
    #             seedparameters=SeedSetting(),
    #         )
    #         save_new_paras(paras, self.log_dir, file_name)
    #         config_dict = load_config(self.log_dir, file_name)

    #     elif not reload_config and os.path.exists(
    #         os.path.join(self.log_dir, file_name + ".json")
    #     ):
    #         config_dict = load_config(self.log_dir, file_name)

    #     elif (
    #         reload_config
    #         and not os.path.exists(os.path.join(self.config_dir, file_name + ".json"))
    #         and os.path.exists(os.path.join(self.log_dir, file_name + ".json"))
    #     ):
    #         config_dict = load_config(self.log_dir, file_name)

    #     else:
    #         config_dict = load_config(self.config_dir, file_name)
        
    #     return config_dict

# if __name__ == "__main__":
#     import os
#     config_dir = os.path.join(os.getcwd(), "config/param")
#     print(config_dir)
#     x=0 if True else 1