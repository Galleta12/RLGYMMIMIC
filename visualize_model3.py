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
from utils.torch import *
from rl_algorithms.ppo import PPO
from reward_function import reward_func
from utils.zfilter import ZFilter
from rfc_utils.logger import create_logger
from rfc_utils.config import Config
from utils.vectorizedEnv import VectorizedEnv
from agent_envs.humanoid_env2 import HumanoidTemplate
from some_math.math_utils import *
from some_math.transformation import quaternion_multiply
from agent_envs.humanoid_replay import HumanoidReplay


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

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
parser.add_argument('--dynamic', type=str2bool, nargs='?', const=True, default=True, help="Enable or disable dynamic mode.")

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

if not args.dynamic:
    env = HumanoidTemplate(cfg)
else:
    env = HumanoidTemplate(cfg,render_mode = 'human')
    
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


#num_fr = 0

def data_generator():
    global num_fr 
    print("calling data generator")
    while True:
        poses = {'pred': [], 'gt': []}
        state,_ = env.reset()
        if running_state is not None:
            print("runni state no none")
            state = running_state.normalize(state, update=False)
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
            next_state, reward, done,_, _ = env.step(action)
            if running_state is not None:
                next_state = running_state.normalize(next_state, update=False)
            if done:
                break
            state = next_state
        poses['gt'] = np.vstack(poses['gt'])
        poses['pred'] = np.vstack(poses['pred'])
        num_fr = poses['pred'].shape[0]
   
        yield poses





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
            #print('done')
            env.reset()
        state = next_state




def update_pose(env_replay:HumanoidReplay,data,fr):
    #first q values pred
    env_replay.data.qpos[:env.model.nq] = data['pred'][fr]
    #from the nq index onwards is the ground truth
    env_replay.data.qpos[env.model.nq:] = data['gt'][fr]
    #increment it so we avoid collision
    env_replay.data.qpos[env.model.nq] += 1.0
    #
    if args.hide_expert:
        env_replay.data.qpos[env.model.nq + 2] = 100.0
    
    env_replay.forward()
        






def visualize():
    
    fr = 0  
    env_replay = HumanoidReplay(cfg.vis_model_file,15,render_mode='human')
    #viewer = MyViewer(env_replay.model,env_replay.data) 
    #change color of expert
    ngeom = len(env.model.geom_rgba) - 1
    env_replay.model.geom_rgba[ngeom + 21: ngeom * 2 + 21] = np.array([0.7, 0.0, 0.0, 1])
    
    for _ in range(1000):
    #while not glfw.window_should_close(viewer.window):
        update_pose(env_replay,data,fr)
            
        #viewer.render_frame()
        env_replay.render()
        fr = (fr+1) % num_fr
        
        
        #glfw.swap_buffers(viewer.window)
        #glfw.poll_events()
    #glfw.terminate()


#print(num_fr)

#data_gen = data_generator()
#data = next(data_gen)
if not args.dynamic:
    data_gen = data_generator()
    data = next(data_gen)
    visualize()
else:
    visualize_dynamics()

