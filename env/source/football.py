import gfootball.env as football_env
import numpy as np
import json

class Football():
    def __init__(self) -> None:
        self.env_core = football_env()