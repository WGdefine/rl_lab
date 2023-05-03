from env.chooseenv import *
from config.hyperparam import *
from tensorboardX import SummaryWriter
from utils.log_path import *
from utils.utils import *

class BaseRunner:
    def __init__(self, args) -> None:
        self.env = make(args.env_name)
        self.run_type = args.run_type
        self.n_agents = self.env.n_agents
        self.algo_name = args.algo
        self.runner_type = args.runner_type
        self.EnvSetting = EnvSettingDefault(
            env_name = args.env_name,
            obs_space = self.env.get_obs_space(),
            action_space = self.env.get_action_space(),
            n_player = self.n_agents
        )
        self.run_dir, self.log_dir = make_logpath(args.env_name, args.algo)
        self.writer = SummaryWriter(str(self.run_dir))
        self.config_dir = os.path.join(os.getcwd(), "config/param")
        file_name = args.algo + '_' + args.env_name
        config_dict = self.load_param(self.algo_name, args.reload_config, file_name)

        # self.agents = []

        self.buffer = []

    def run(self):
        raise NotImplementedError

    def collect(self, step):
        raise NotImplementedError
    
    def insert(self, data):
        raise NotImplementedError

    def save(self):
        pass

    def load_param(self, algo, reload_config, file_name):
        if (
            not reload_config
            and not os.path.exists(os.path.join(self.log_dir, file_name + ".json"))
        ) or (
            reload_config
            and not os.path.exists(os.path.join(self.config_dir, file_name + ".json"))
            and not os.path.exists(os.path.join(self.log_dir, file_name + ".json"))
        ):
            paras = RunnerSetting(
                algo=algo,
                hyperparameters=globals()[str(algo).upper() + "Default"](),
                envparameters=self.EnvSetting,
                trainingparameters=TrainingDefault(),
                seedparameters=SeedSetting(),
            )
            save_new_paras(paras, self.log_dir, file_name)
            config_dict = load_config(self.log_dir, file_name)

        elif not reload_config and os.path.exists(
            os.path.join(self.log_dir, file_name + ".json")
        ):
            config_dict = load_config(self.log_dir, file_name)

        elif (
            reload_config
            and not os.path.exists(os.path.join(self.config_dir, file_name + ".json"))
            and os.path.exists(os.path.join(self.log_dir, file_name + ".json"))
        ):
            config_dict = load_config(self.log_dir, file_name)

        else:
            config_dict = load_config(self.config_dir, file_name)
        
        return config_dict

# if __name__ == "__main__":
#     import os
#     config_dir = os.path.join(os.getcwd(), "config/param")
#     print(config_dir)