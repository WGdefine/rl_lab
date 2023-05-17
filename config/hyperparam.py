import attr
import json

@attr.define
class HyperparamSettings:
    use_network: bool = True
    marl: bool = False

@attr.define
class EnvSettingDefault:
    env_name: str = "classic_CartPole-v0"
    obs_space: int = 100
    action_space: int = 100
    obs_continuous: bool = True
    action_continuous: bool = False
    n_player: int = 1

@attr.define
class TrainingDefault:
    max_episodes: int = 1000
    evaluate_rate: int = 50
    render: bool = False
    save_interval: int = 100

@attr.define
class SeedSetting:
    seed_nn: int = 1
    seed_np: int = 1
    seed_random: int = 1

@attr.define
class RunnerSetting:
    hyperparameters: HyperparamSettings = attr.field()
    envparameters: EnvSettingDefault = attr.field()
    trainingparameters: TrainingDefault = attr.field()
    seedparameters: SeedSetting = attr.field()
    algo: str = "DQN"