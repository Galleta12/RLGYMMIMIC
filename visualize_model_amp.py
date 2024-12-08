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
from agent_envs.humanoid_amp_env_2_test import HumanoidTemplate
from some_math.math_utils import *
from some_math.transformation import quaternion_multiply,quaternion_from_euler
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
    env = HumanoidTemplate(cfg,isExpert=False)
else:
    env = HumanoidTemplate(cfg,isExpert=False,render_mode = 'human')

#env = HumanoidTemplate(cfg,isExpert=False,render_mode = 'human')

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
disc_net = Discriminator(net=MLP(amp_feature_size*2,cfg.value_hsize, cfg.value_htype)) 
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




def data_generator():
    global num_fr 
    print("calling data generator")
    while True:
        poses = {'pred': [], 'target': []}
        state,_ = env.reset()
        if running_state is not None:
            print("runni state no none")
            state = running_state.normalize(state, update=False)
        for t in range(1000):
            target_pos = env.get_target_position()
            poses['target'].append(target_pos)
            poses['pred'].append(env.data.qpos.copy())
            state_var = tensor(state, dtype=dtype).unsqueeze(0)
            action = policy_net.select_action(state_var, mean_action=True)[0].cpu().numpy()
            next_state, reward, done,_, _ = env.step(action)
            if running_state is not None:
                next_state = running_state.normalize(next_state, update=False)
            if done:
                break
            state = next_state
        poses['pred'] = np.vstack(poses['pred'])
        poses['target'] = np.vstack(poses['target'])
        num_fr = poses['pred'].shape[0]
   
        yield poses




def update_pose(env_replay:HumanoidReplay,data,fr):
    # Predicted pose
    env_replay.data.qpos[:env.model.nq] = data['pred'][fr]

    
    print("rot poss", env_replay.data.qpos[:3])
    # Target position
    target_pos = data['target'][fr]
    
    #target_pos = np.append(target_pos,[0,0,0,0])
    # Get predicted root position
    pred_pos = data['pred'][fr][:3]

    #env_replay.data.qpos[env.model.nq:] = target_pos
    
    
    # Compute direction vector from predicted root to target
    direction = target_pos - pred_pos
    direction[2] = 0  # Ignore Z-axis for 2D orientation on the plane
    direction = direction / np.linalg.norm(direction)  # Normalize the direction vector

    # Calculate quaternion to face the target direction
    angle = np.arctan2(direction[1], direction[0])  # Calculate yaw angle
    target_orientation = quaternion_from_euler(0, 0, angle)  # Convert to quaternion

    
     # Set the target's full pose (position + orientation)
    env_replay.data.qpos[env.model.nq:env.model.nq + 3] = target_pos  # Set target position
    env_replay.data.qpos[env.model.nq + 3:env.model.nq + 7] = target_orientation  # Set target orientation

    # Set the target's Z-axis position to match the prediction
    env_replay.data.qpos[env.model.nq + 2] = 0.8  # Align Z-axis with prediction
    
    env_replay.forward()
        


def visualize():
    
    fr = 0  
    env_replay = HumanoidReplay(cfg.vis_model_file,15,render_mode='human')
    #viewer = MyViewer(env_replay.model,env_replay.data) 
    #change color of expert
    ngeom = len(env.model.geom_rgba) - 1
    env_replay.model.geom_rgba[ngeom + 21: ngeom * 2 + 21] = np.array([0.7, 0.0, 0.0, 1])
    
    target_pos = data['target'][fr]
    #target_pos[2] = 2.0
    
    print("target pos", target_pos)
    #env_replay.data.qpos[env.model.nq:env.model.nq + 3] = target_pos
    
    
    
    for _ in range(1000):
    #while not glfw.window_should_close(viewer.window):
        update_pose(env_replay,data,fr)
            
        #viewer.render_frame()
        env_replay.render()
        fr = (fr+1) % num_fr
        
        
        #glfw.swap_buffers(viewer.window)
        #glfw.poll_events()
    #glfw.terminate()



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
        
        
        
if not args.dynamic:
    data_gen = data_generator()
    data = next(data_gen)
    visualize()
else:
    visualize_dynamics()