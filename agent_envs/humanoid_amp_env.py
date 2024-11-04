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

from tools.tools import get_expert, get_body_qposaddr,amp_obs_feature
from tools.datasetAmp import AMPDataset
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
        
         
        self.ampDataset = AMPDataset(self)
        self.time_horizon = 10
        self.amp_features_size = self.get_amp_size() 
        
        
        self.target_speed = np.random.uniform(1, 5)
        self.target_direction = self.set_random_direction()
        self.target_direction_local = self.convert_to_local(self.target_direction)
    
    def set_random_direction(self):
        
        """Generates a random unit direction vector on the x-y plane."""
        angle = np.random.uniform(0, 2 * np.pi)
        return np.array([np.cos(angle), np.sin(angle), 0])
    
    def convert_to_local(self, direction):
        """Converts a global direction vector to the character's local frame."""
        hq = get_heading_q(self.data.qpos[3:7]) # Orientation quaternion of the character
        #conver to local
        return transform_vec(direction, hq)    
    
    def get_amp_size(self):
        amp_features = self.get_amp_features(self.data.qpos.copy(), self.data.qvel.copy())
        return amp_features.shape[0]
        
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
        
        self.do_simulation(a, self.frame_skip)
        
        
        #this is to keep track of how many steps are done
        #and it is reseted on the main reset function.
        self.cur_t += 1
        #print('current step', self.cur_t)
        
        # Update target speed every 50 steps
        # if self.cur_t % 50 == 0:
        #     self.target_speed = np.random.uniform(1, 5)
            #self.target_direction = self.set_random_direction()
        
        
        self.target_direction_local = self.convert_to_local(self.target_direction)
        
        # print('target speed', self.target_speed)
        # print('target direction ', self.target_direction)
        # print('target direction local', self.target_direction_local)
        
        
        self.bquat = self.get_body_quat()
        amp_features = self.get_amp_features(self.data.qpos.copy(), self.data.qvel.copy())
        amp_state_arr,amp_next_state_arr,amp_state,amp_next_state = self.sample_amp_features(1)
        
        #print("step index", self.cur_t)
        

        
        #self.data.qpos[:],self.data.qvel[:] = amp_state[0]['original_qpos'], amp_state[0]["original_qvel"]
        
        #mj.mj_forward(self.model, self.data)
        
            
        reward = 1.0

        
        fail = self.data.qpos[2] < self.ampDataset.height_lb - 0.1
        
        end =   ( self.cur_t >= self.time_horizon)
        
        done = fail or end
        #done = end
        
        # if done:
        #     print('done',done)
        
        obs = self.get_obs()
        
        
       
        
        return obs, reward, done, False,{'fail': fail, 'end': end,
                                         'amp_features':amp_features,
                                         'amp_data_state':amp_state_arr,
                                         'amp_data_next_state':amp_next_state_arr}
    
    
    
    def get_obs(self):
        
        
        
        
        #print('new obs')    
        
        obs = self.get_full_obs()
        
        #79 dim
        obs = np.concatenate([obs, self.target_direction_local, [self.target_speed]])
        
        return obs

    
    
    
    
    def get_initial_amp_features(self):
        amp_features = amp_obs_feature(self,None,self.data.qpos.copy(),self.data.qvel.copy())
        amp_features_concat = np.concatenate([
            amp_features['root_linear_velocity'],
            amp_features['root_angular_velocity'],
            amp_features['local_joint_rotations'],
            amp_features['local_joint_velocities'],
            amp_features['end_effector_positions']
        ])
        
        return amp_features_concat
    
    
    def get_amp_features(self,qpos,qvel):
        amp_features = amp_obs_feature(self,None,qpos,qvel)
        amp_features_concat = np.concatenate([
            amp_features['root_linear_velocity'],
            amp_features['root_angular_velocity'],
            amp_features['local_joint_rotations'],
            amp_features['local_joint_velocities'],
            amp_features['end_effector_positions']
        ])
        
        return amp_features_concat
    
    
    def sample_amp_features(self, num):
        amp_state, amp_next_state = self.ampDataset.sample_state_next_state_batch(num)
        
        amp_state_arrays = []
        amp_next_state_arrays = []
        
        
        
        for i in range(num):
            # Convert each dictionary of AMP features into concatenated NumPy arrays
            amp_state_array = np.concatenate([
                amp_state[i]['root_linear_velocity'],
                amp_state[i]['root_angular_velocity'],
                amp_state[i]['local_joint_rotations'],
                amp_state[i]['local_joint_velocities'],
                amp_state[i]['end_effector_positions']
            ])
            
            amp_next_state_array = np.concatenate([
                amp_next_state[i]['root_linear_velocity'],
                amp_next_state[i]['root_angular_velocity'],
                amp_next_state[i]['local_joint_rotations'],
                amp_next_state[i]['local_joint_velocities'],
                amp_next_state[i]['end_effector_positions']
            ])
            
            
            # Append each concatenated array to the respective list
            amp_state_arrays.append(amp_state_array)
            amp_next_state_arrays.append(amp_next_state_array)
        
        
                
         # Convert lists to arrays for consistency
        return np.array(amp_state_arrays), np.array(amp_next_state_arrays),amp_state,amp_next_state
             
    
    
    def random_sample(self):
        clip = self.ampDataset.sample_clip()
        clip_name = clip['clip_name']
        clip_expert = clip['expert'] 
        qpos =  clip_expert['qpos']
        qvel = clip_expert['qvel']
            
        # Sample an index from the chosen clip
        idx = np.random.randint(clip_expert['len'])
        
        init_pose = qpos[idx]
        init_vel = qvel[idx]
        
        # print('reset', clip_name)
        # print('reset len', clip_expert['len'])
        # print('reset idx', idx)
        
        
        return init_pose,init_vel
    
    
    
    def get_com_velocity(self):
        """Retrieve the center of mass linear velocity of the character."""
        com_velocity = self.data.subtree_linvel[0].copy()  # Linear COM velocity for the entire body (index 0 is the root)
        return com_velocity
    
    
    def reset_model(self):
        cfg = self.cfg

        self.target_speed = np.random.uniform(1, 5)
        #self.target_direction = self.set_random_direction()
        #sample a random amp dataset reference
        
        init_pose,init_vel = self.random_sample()
        
        #print('reset')
        init_pose[7:] += self.np_random.normal(loc=0.0, scale=cfg.env_init_noise, size=self.model.nq - 7)
        self.set_state(init_pose, init_vel)
        self.bquat = self.get_body_quat()
         
        return self.get_obs()
     