import attrs
import json

@attrs.define
class HyperparamSettings:
    use_network: bool = True
    marl: bool = False

@attrs.define
class EnvSettingDefault:
    env_name: str = "classic_CartPole-v0"
    obs_space: int = 100
    action_space: int = 100
    obs_continuous: bool = True
    action_continuous: bool = False
    n_player: int = 1

@attrs.define
class TrainingDefault:
    max_episodes: int = 1000
    evaluate_rate: int = 50
    render: bool = False
    save_interval: int = 100

@attrs.define
class SeedSetting:
    seed_nn: int = 1
    seed_np: int = 1
    seed_random: int = 1

@attrs.define
class RunnerSetting:
    hyperparameters: HyperparamSettings = attrs.field()
    envparameters: EnvSettingDefault = attrs.field()
    trainingparameters: TrainingDefault = attrs.field()
    seedparameters: SeedSetting = attrs.field()
    algo: str = "DQN"