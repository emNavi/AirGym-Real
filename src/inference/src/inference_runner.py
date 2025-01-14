#!/home/emnavi/miniconda3/envs/inference/bin/python

import numpy as np
import os
import yaml
import argparse

import sys
script_dir = os.path.dirname(os.path.realpath(__file__))
airgym_dir = os.path.abspath(os.path.join(script_dir, "../../.."))
sys.path.insert(0, airgym_dir)
yaml_path = os.path.join(script_dir, 'X152b_inference.yaml')

from airgym.envs import *
from argparse import Namespace

from airgym.lib.torch_runner import Runner
from airgym.lib.utils import env_configurations, vecenv

from src.inference.src.players import *
from airgym.lib.utils.vecenv import AirGymRLGPUEnv

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

# register the inference environemnt
from src.inference.src.envs.inference import inference
# task_registry.register("inference", inference, X152bPx4Cfg())
# task_registry.register("inference", inference, X152bAvoidConfig())
task_registry.register("inference", inference, X152bTrackingConfig())

env_configurations.register('X152b', {'env_creator': lambda **kwargs : task_registry.make_env('inference',args=Namespace(**kwargs)),
        'vecenv_type': 'AirGym-RLGPU'})

vecenv.register('AirGym-RLGPU',
                lambda config_name, num_actors, **kwargs: AirGymRLGPUEnv(config_name, num_actors, **kwargs))

def get_args():
    from src.inference.src.utils import gymutil

    custom_parameters = [
        {"name": "--seed", "type": int, "default": 0, "required": False, "help":  "Random seed, if larger than 0 will overwrite the value in yaml config."},
        
        {"name": "--tf", "required": False, "help": "run tensorflow runner", "action": 'store_true'},
        {"name": "--train", "required": False, "help": "train network", "action": 'store_true'},
        {"name": "--play", "required": True, "help": "play(test) network", "action": 'store_true'},
        {"name": "--checkpoint", "type": str, "required": False, "help": "path to checkpoint"},
        {"name": "--file", "type": str, "default": yaml_path, "required": False, "help": "path to config"},
        {"name": "--num_envs", "type": int, "default": "1", "help": "Number of environments to create. Overrides config file if provided."},

        {"name": "--sigma", "type": float, "required": False, "help": "sets new sigma value in case if 'fixed_sigma: True' in yaml config"},
        {"name": "--track",  "action": 'store_true', "help": "if toggled, this experiment will be tracked with Weights and Biases"},
        {"name": "--wandb-project-name", "type": str, "default": "rl_games", "help": "the wandb's project name"},
        {"name": "--wandb-entity", "type": str, "default": None, "help": "the entity (team) of wandb's project"},

        {"name": "--task", "type": str, "default": None, "help": "Override task from config file if provided."},
        {"name": "--experiment_name", "type": str,  "help": "Name of the experiment to run or load. Overrides config file if provided."},

        {"name": "--headless", "action": "store_true", "default": False, "help": "Force display off at all times"},
        {"name": "--horovod", "action": "store_true", "default": False, "help": "Use horovod for multi-gpu training"},
        {"name": "--rl_device", "type": str, "default": "cpu", "help": 'Device used by the RL algorithm, (cpu, gpu, cuda:0, cuda:1 etc..)'},

        {"name": "--ctl_mode", "required": True, "type": str, "help": 'Specify the control mode and the options are: pos, vel, atti, rate, prop'},
        ]
        
    # parse arguments
    args = gymutil.parse_arguments(
        description="RL Policy",
        custom_parameters=custom_parameters)

    # name allignment
    args.sim_device_id = args.compute_device_id
    args.sim_device = args.sim_device_type
    if args.sim_device=='cuda':
        args.sim_device += f":{args.sim_device_id}"
    return args

def ros_get_args():
    custom_parameters = [
        {"name": "--seed", "type": int, "default": 0, "required": False, "help":  "Random seed, if larger than 0 will overwrite the value in yaml config."},
        {"name": "--tf", "required": False, "help": "run tensorflow runner", "action": 'store_true'},
        {"name": "--train", "required": False, "help": "train network", "action": 'store_true'},
        {"name": "--play", "required": False, "help": "play(test) network", "action": 'store_true'},
        {"name": "--checkpoint", "type": str, "required": False, "help": "path to checkpoint"},
        {"name": "--file", "type": str, "default": yaml_path, "required": False, "help": "path to config"},
        {"name": "--num_envs", "type": int, "default": "1", "help": "Number of environments to create. Overrides config file if provided."},
        {"name": "--sigma", "type": float, "required": False, "help": "sets new sigma value in case if 'fixed_sigma: True' in yaml config"},
        {"name": "--track",  "action": 'store_true', "help": "if toggled, this experiment will be tracked with Weights and Biases"},
        {"name": "--wandb-project-name", "type": str, "default": "rl_games", "help": "the wandb's project name"},
        {"name": "--wandb-entity", "type": str, "default": None, "help": "the entity (team) of wandb's project"},
        {"name": "--task", "type": str, "default": None, "help": "Override task from config file if provided."},
        {"name": "--experiment_name", "type": str,  "help": "Name of the experiment to run or load. Overrides config file if provided."},
        {"name": "--headless", "action": "store_true", "default": False, "help": "Force display off at all times"},
        {"name": "--horovod", "action": "store_true", "default": False, "help": "Use horovod for multi-gpu training"},
        {"name": "--rl_device", "type": str, "default": "cpu", "help": 'Device used by the RL algorithm, (cpu, gpu, cuda:0, cuda:1 etc..)'},
        {"name": "--ctl_mode", "required": True, "type": str, "help": 'Specify the control mode and the options are: pos, vel, atti, rate, prop'},
    ]

    # 从ROS参数服务器获取参数
    ros_params = {}
    for param in custom_parameters:
        name = param["name"].lstrip('--')
        default_value = param.get("default", None)
        ros_params[name] = rospy.get_param(f'{name}', default_value)

    # 构建命令行参数
    args_list = []
    for param in custom_parameters:
        name = param["name"].lstrip('--')
        if ros_params[name] is not None:
            if isinstance(ros_params[name], bool):
                if ros_params[name]:
                    args_list.append(param["name"])
            else:
                args_list.extend([param["name"], str(ros_params[name])])

    # 使用argparse解析参数
    parser = argparse.ArgumentParser()
    for param in custom_parameters:
        kwargs = {k: v for k, v in param.items() if k != "name"}
        parser.add_argument(param["name"], **kwargs)

    args = parser.parse_args(args_list)

    args.physics_engine = None
    args.sim_device = 'cpu'
    args.use_gpu = False
    args.subscenes = 0
    args.use_gpu_pipeline = False
    args.num_threads = 1
    return args


def update_config(config, args):

    if args['task'] is not None:
        config['params']['config']['env_name'] = args['task']
    if args['experiment_name'] is not None:
        config['params']['config']['name'] = args['experiment_name']

    config['params']['config']['env_config']['physics_engine'] = args['physics_engine']
    config['params']['config']['env_config']['sim_device'] = args['sim_device']
    config['params']['config']['env_config']['headless'] = args['headless']
    config['params']['config']['env_config']['use_gpu'] = args['use_gpu']
    config['params']['config']['env_config']['subscenes'] = args['subscenes']
    config['params']['config']['env_config']['use_gpu_pipeline'] = args['use_gpu_pipeline']
    config['params']['config']['env_config']['num_threads'] = args['num_threads']
    config['params']['config']['env_config']['ctl_mode'] = args['ctl_mode']

    if args['num_envs'] > 0:
        config['params']['config']['num_actors'] = args['num_envs']
        config['params']['config']['env_config']['num_envs'] = args['num_envs']

    if args['seed'] > 0:
        config['params']['seed'] = args['seed']
        config['params']['config']['env_config']['seed'] = args['seed']

    return config

def _restore(agent, args):
    if 'checkpoint' in args and args['checkpoint'] is not None and args['checkpoint'] !='':
        agent.restore(args['checkpoint'])

class InferenceRunner(Runner):
    def __init__(self):
        super().__init__()

    def run_play(self, args):
        print('Started to play')
        player = CpuPlayerContinuous(params=self.params)
        _restore(player, args)
        player.run()

def main(args):
    config_name = args['file']
    args['play'] = True

    print('Loading config:', config_name)
    with open(config_name, 'r') as stream:
        config = yaml.safe_load(stream)
    
        config = update_config(config, args)

        inference_runner = InferenceRunner()
        try:
            inference_runner.load(config)
        except yaml.YAMLError as exc:
            print(exc)

    print(config)

    inference_runner.run(args)

if __name__ == '__main__':
    try:
        args = vars(ros_get_args())
        main(args)
    except (SystemExit, ConnectionRefusedError):
        args = vars(get_args())
        main(args)