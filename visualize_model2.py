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
from utils.memory import MemoryManager,TrajBatch
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


# env = HumanoidTemplate(cfg,render_mode = 'human')
# env.seed(cfg.seed)
expert_reward = reward_func[cfg.reward_id]

env = VectorizedEnv(env_object=HumanoidTemplate,num_envs=2,custom_reward=expert_reward,cfg=cfg,render_mode ='human')



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



memory = MemoryManager(num_envs=2)

def visualize_dynamics():
    state,_ ,norm_state= env.reset()
    
        
    state = [running_state(c,update=False) for c in state]
    
    for t in range(4000):
        #print(t)
        state_vars = torch.tensor(state)
        
        mean_actions = np.ones(2)
        
        actions = policy_net.select_action(state_vars, mean_action=mean_actions).cpu().numpy()
        
        next_states, rewards_env, terminateds, truncateds, infos,norm_next_states, loggers= env.step(actions)
        
        memory.append(state, actions, rewards_env, actions, terminateds, truncateds, norm_next_states,terminateds, terminateds, infos)
        
        next_state = [running_state(c,update=False) for c in next_states]
        
        state = next_state
        
        
        env.render()
        
        for index in memory.done_indices():
                env_memory = memory[index]
                env_memory.reset()    
                state[index], infos, norm_states = env.reset(index)
    env.close()








visualize_dynamics()

