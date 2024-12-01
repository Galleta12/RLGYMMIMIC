import argparse
import os
import sys
import pickle
import time
import subprocess
import shutil
import torch
import numpy as np
from mujoco.glfw import glfw
sys.path.append(os.getcwd())
from networks_models.common import MLP,Value
from networks_models.policy_net import PolicyGaussian
from networks_models.discriminator_net import Discriminator
from utils.torch import *
from rl_algorithms.ppo import PPO
from reward_function import reward_func
from utils.zfilter import ZFilter
from rfc_utils.logger import create_logger
from rfc_utils.config import Config
from utils.vectorizedEnv import VectorizedEnv
from agent_envs.humanoid_amp_env_2 import HumanoidTemplate
from some_math.math_utils import *
from some_math.transformation import quaternion_multiply
from agent_envs.humanoid_replay import HumanoidReplay


parser = argparse.ArgumentParser()
parser.add_argument('--cfg', default=None)
parser.add_argument('--vis_model_file', default='mocap_v2_vis')
parser.add_argument('--iter', type=int, default=-1)
parser.add_argument('--focus', action='store_true', default=True)
parser.add_argument('--hide_expert', action='store_true', default=False)
parser.add_argument('--preview', action='store_true', default=False)
parser.add_argument('--record', action='store_true', default=False)
parser.add_argument('--record_expert', action='store_true', default=False)
parser.add_argument('--azimuth', type=float, default=45)


args = parser.parse_args()

cfg = Config(args.cfg, False, create_dirs=False)
print("cfg", cfg.vis_model_file)
cfg.env_start_first = True
logger = create_logger(os.path.join(cfg.log_dir, 'log_eval.txt'))

"""make and seed env"""
dtype = torch.float64
torch.set_default_dtype(dtype)
torch.manual_seed(cfg.seed)
torch.set_grad_enabled(False)

env = HumanoidTemplate(cfg,isExpert=False,render_mode = 'human')

env.seed(cfg.seed)


state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
#for the policy states
running_state = ZFilter((state_dim,), clip=5)
#for the amp features
running_state_amp_features = ZFilter((env.amp_features_size,), clip=5)
#for the amp states
running_state_amp = ZFilter((env.amp_features_size,), clip=5)
#for the amp next states
running_next_state_amp = ZFilter((env.amp_features_size,), clip=5)


amp_feature_size = env.amp_features_size


"""define actor and critic"""
policy_net = PolicyGaussian(MLP(state_dim, cfg.policy_hsize, cfg.policy_htype), action_dim, log_std=cfg.log_std, fix_std=cfg.fix_std)
value_net = Value(MLP(state_dim, cfg.value_hsize, cfg.value_htype))
"""define the disc net"""
disc_net = Discriminator(net=MLP(amp_feature_size*2,cfg.value_hsize, cfg.value_htype),amp_reward_coef=0.5) 
cp_path = '%s/iter_%04d.p' % (cfg.model_dir, args.iter)
print(cp_path)
logger.info('loading model from checkpoint: %s' % cp_path)
print("after", cp_path)
model_cp = pickle.load(open(cp_path, "rb"))
policy_net.load_state_dict(model_cp['policy_dict'])
value_net.load_state_dict(model_cp['value_dict'])
disc_net.load_state_dict(model_cp['disc_dict'])
running_state = model_cp['running_state']
running_state_amp_features = model_cp['running_state_amp_features']
running_state_amp = model_cp['running_state_amp']
running_next_state_amp = model_cp['running_next_state_amp']
print(running_state)


def visualize_dynamics():
    state, _ = env.reset()
    if running_state is not None:
        print("runni state no none")
        state = running_state.normalize(state, update=False)
    for t in range(4000):
        #print(t)
        state_var = tensor(state, dtype=dtype).unsqueeze(0)
        action = policy_net.select_action(state_var, mean_action=True)[0].cpu().numpy()
        next_state, reward, done, _,_ = env.step(action)
        env.render()
        if running_state is not None:
            next_state = running_state.normalize(next_state, update=False)
        if done:
            print('done')
            env.reset()
        state = next_state
        
        
        
visualize_dynamics()