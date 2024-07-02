import argparse
import os
import sys
import pickle
import time
import datetime
import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
import torch
import torch.nn as nn
from torch.distributions.uniform import Uniform
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
from torch.distributions import Normal
from stable_baselines3.common.policies import MultiInputActorCriticPolicy
sys.path.append(os.getcwd())
from rfc_utils.config import Config
from agent_envs.humanoid_env2 import HumanoidTemplate
from rl_rfc.core.policy_gaussian import PolicyGaussian
from rl_rfc.core.critic import Value
from rl_rfc.agents import AgentPPO
from rfc_models.mlp import MLP





parser = argparse.ArgumentParser()
parser.add_argument('--cfg', default=None)
parser.add_argument('--render', action='store_true', default=False)
parser.add_argument('--test', action='store_true', default=False)
parser.add_argument('--num_threads', type=int, default=20)
parser.add_argument('--gpu_index', type=int, default=0)
parser.add_argument('--iter', type=int, default=0)
parser.add_argument('--show_noise', action='store_true', default=False)
args = parser.parse_args()
if args.render:
    args.num_threads = 1
cfg = Config(args.cfg, args.test, create_dirs=not (args.render or args.iter > 0))

def make_env():
    def thunk():
        env = HumanoidTemplate(cfg)
        env.seed(cfg.seed)
        return env
    return thunk




if __name__ == "__main__":
    

    
    
    pass













