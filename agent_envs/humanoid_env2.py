import os
import math
import numpy as np
import mujoco as mj
from typing import Any, Dict, Optional, Tuple, Union
import gymnasium as gym
from gymnasium.utils import EzPickle
import sys
import os
import pickle
import time
from scipy.linalg import cho_solve, cho_factor
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
# from rfc_utils.rfc_mujoco import get_body_qposaddr
# from rfc_utils.tools import get_expert
from tools.tools import get_expert, get_body_qposaddr
from some_math.math_utils import *
from some_math.transformation import quaternion_from_euler
from common.mujoco_envs import MujocoEnv
from agent_envs.pd_controllers import stable_pd_controller
from agent_envs.humanoid_template2 import HumanoidBase
from reward_function import world_rfc_implicit_reward


DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 1,
    "distance": 4.0,
    "lookat": np.array((0.0, 0.0, 2.0)),
    "elevation": -20.0,
}

class HumanoidTemplate(HumanoidBase):
    
    def __init__(self, cfg, frame_skip: int = 15, 
                 default_camera_config: Dict[str, Union[float, int]] = DEFAULT_CAMERA_CONFIG, **kwargs):
        # Call the parent class constructor
        super().__init__(cfg, frame_skip, default_camera_config, **kwargs)
    
    
    def compute_torque(self, ctrl):
        cfg = self.cfg
        dt = self.model.opt.timestep
        ctrl_joint = ctrl[:self.ndof] * cfg.a_scale
        qpos = self.data.qpos.copy()
        qvel = self.data.qvel.copy()
        base_pos = cfg.a_ref
        target_pos = base_pos + ctrl_joint

        tau = stable_pd_controller(self.data,self.model,target_pos,qpos,qvel,cfg,dt)
        return tau
    

    def do_simulation(self, action, n_frames):
       
        cfg = self.cfg
        for i in range(n_frames):
            ctrl = action
            
            torque = self.compute_torque(ctrl)
            
            torque = np.clip(torque, -cfg.torque_lim, cfg.torque_lim)
            self.data.ctrl[:] = torque

            """ Residual Force Control (RFC) """
            if cfg.residual_force:
                vf = ctrl[-self.vf_dim:].copy()
                if cfg.residual_force_mode == 'implicit':
                    self.rfc_implicit(vf)
            mj.mj_step(self.model, self.data)

    def step(self, a):
        cfg = self.cfg
        # record prev state
        self.prev_qpos = self.data.qpos.copy()
        self.prev_qvel = self.data.qvel.copy()
        self.prev_bquat = self.bquat.copy()
        # do simulation
        self.do_simulation(a, self.frame_skip)
        self.cur_t += 1
        self.bquat = self.get_body_quat()
        self.update_expert()
        
        reward = 1.0
        #reward = world_rfc_implicit_reward(self,None,a,None)
        
        
        
        fail = self.expert is not None and self.data.qpos[2] < self.expert['height_lb'] - 0.1
        cyclic = self.expert['meta']['cyclic']
        end =  (cyclic and self.cur_t >= cfg.env_episode_len) or (not cyclic and self.cur_t + self.start_ind >= self.expert['len'])
        done = fail or end
        obs = self.get_obs()
        return obs, reward, done,False, {'fail': fail, 'end': end}
    
    def reset_model(self):
        cfg = self.cfg
        if self.expert is not None:
            #ind = 0 if self.cfg.env_start_first else self.np_random.randint(self.expert['len'])
            ind = 0 if self.cfg.env_start_first else self.np_random.integers(self.expert['len'])
            self.start_ind = ind
            init_pose = self.expert['qpos'][ind, :].copy()
            init_vel = self.expert['qvel'][ind, :].copy()
            init_pose[7:] += self.np_random.normal(loc=0.0, scale=cfg.env_init_noise, size=self.model.nq - 7)
            self.set_state(init_pose, init_vel)
            self.bquat = self.get_body_quat()
            self.update_expert()
        else:
            init_pose = self.data.qpos
            init_pose[2] += 1.0
            self.set_state(init_pose, self.data.qvel)
        return self.get_obs()
    
   

