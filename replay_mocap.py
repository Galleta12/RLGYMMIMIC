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
from rfc_utils.config import Config
from agent_envs.humanoid_replay_mocap import HumanoidReplayMocap


parser = argparse.ArgumentParser()
parser.add_argument('--cfg', default=None)

args = parser.parse_args()

cfg = Config(args.cfg, False, create_dirs=False)

env_mocap = HumanoidReplayMocap(cfg,render_mode = 'human')

print('expert trail', env_mocap.cfg.env_expert_trail_steps)
print('env start first',env_mocap.cfg.env_start_first)
print('start ind', env_mocap.start_ind)
print('expert lenght', env_mocap.expert['len'])
print('meta:', env_mocap.expert['meta'])
print('meta cyclic:', env_mocap.expert['meta']['cyclic'])


def visualize():
    obs= env_mocap.reset()
    print('expert trail', env_mocap.cfg.env_expert_trail_steps)
    print('env start first',env_mocap.cfg.env_start_first)
    print('start ind', env_mocap.start_ind)
    
    for i in range(2000):
        #random action
        action = env_mocap.action_space.sample()    
        observation, reward, done, info = env_mocap.step(action)
        env_mocap.render()
        if done:
            print('done')
            #break
            obs= env_mocap.reset()


visualize()