# -*- coding:utf-8  -*-
# 作者：zruizhi   
# 创建时间： 2020/9/11 11:17 上午   
# 描述：选择运行环境，需要维护env/__ini__.py && config.json（存储环境默认参数）

import json
# import env
import os
import importlib
from utils.utils import *


def make(env_type, conf=None, seed=None):
    # file_path = os.path.join(os.path.dirname(__file__), 'config.json')
    file_path = "config/param/env/config.json"
    if not conf:
        with open(file_path) as f:
            conf = json.load(f)[env_type]
    class_literal = conf['class_literal']
    env_wrapper_name = str("env.wrapper."+str(class_literal)+"_wrapper")
    env_wrapper_import = importlib.import_module(env_wrapper_name)
    param = json_to_args(conf)
    return getattr(env_wrapper_import, class_literal[0].upper()+class_literal[1:].lower()+"Wrapper")(param)


if __name__ == "__main__":
    make("classic_MountainCar-v0")
