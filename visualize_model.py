import argparse
import os
import sys
import pickle
import time
import subprocess
import shutil
import torch
import numpy as np


sys.path.append(os.getcwd())
from rfc_utils.torch import *
from rfc_utils.rfc_math import *
from rl_rfc.core.policy_gaussian import PolicyGaussian
from rl_rfc.core.critic import Value
from rfc_models.mlp import MLP
from agent_envs.humanoid_env import HumanoidTemplate
from rfc_utils.config import Config
from rfc_utils.logger import create_logger
from common.viewer import MyViewer


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
cfg.env_start_first = True
logger = create_logger(os.path.join(cfg.log_dir, 'log_eval.txt'))

"""make and seed env"""
dtype = torch.float64
torch.set_default_dtype(dtype)
torch.manual_seed(cfg.seed)
torch.set_grad_enabled(False)
env = HumanoidTemplate(cfg)
env.seed(cfg.seed)
#actuators = env.model.actuator_names
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

"""load learner policy"""
policy_net = PolicyGaussian(MLP(state_dim, cfg.policy_hsize, cfg.policy_htype), action_dim, log_std=cfg.log_std, fix_std=cfg.fix_std)
value_net = Value(MLP(state_dim, cfg.value_hsize, cfg.value_htype))
cp_path = '%s/iter_%04d.p' % (cfg.model_dir, args.iter)
print(cp_path)
logger.info('loading model from checkpoint: %s' % cp_path)
print("after", cp_path)
model_cp = pickle.load(open(cp_path, "rb"))
policy_net.load_state_dict(model_cp['policy_dict'])
value_net.load_state_dict(model_cp['value_dict'])
running_state = model_cp['running_state']



def data_generator():
    print("calling data generator")
    while True:
        poses = {'pred': [], 'gt': []}
        state = env.reset()
        if running_state is not None:
            print("runni state no none")
            state = running_state(state, update=False)
        for t in range(1000):
            epos = env.get_expert_attr('qpos', env.get_expert_index(t)).copy()
            if env.expert['meta']['cyclic']:
                init_pos = env.expert['init_pos']
                cycle_h = env.expert['cycle_relheading']
                cycle_pos = env.expert['cycle_pos']
                epos[:3] = quat_mul_vec(cycle_h, epos[:3] - init_pos) + cycle_pos
                epos[3:7] = quaternion_multiply(cycle_h, epos[3:7])
            poses['gt'].append(epos)
            poses['pred'].append(env.data.qpos.copy())
            state_var = tensor(state, dtype=dtype).unsqueeze(0)
            action = policy_net.select_action(state_var, mean_action=True)[0].cpu().numpy()
            next_state, reward, done, _ = env.step(action)
            if running_state is not None:
                next_state = running_state(next_state, update=False)
            if done:
                break
            state = next_state
        poses['gt'] = np.vstack(poses['gt'])
        poses['pred'] = np.vstack(poses['pred'])
        num_fr = poses['pred'].shape[0]
        yield poses


def visualize():
    pass



data_gen = data_generator()
data = next(data_gen)



