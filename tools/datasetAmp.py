import os
import math
import numpy as np
from utils.torch import *
import numpy as np
import mujoco as mj
import pickle
from typing import Any, Dict, Optional, Tuple, Union,List
from agent_envs.humanoid_template2 import HumanoidBase
from some_math.math_utils import *
import sys
sys.path.append(os.getcwd())
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from tools.tools import get_expert


class AMPDataset:
    
    def __init__(self,env:HumanoidBase):
        
        
        self.env = env
        #for now hardcoded
        # Define motion files (more clips can be added here)
        self.motion_files = {
            "run_01": 'data/motion/run_01_test.p',
            "walk_01": 'data/motion/walk_01_test.p',
            "walk_90_left": 'data/motion/walk-90-left.p',
            "walk_90_right": 'data/motion/walk-90-right.p'
        }
        
        self.motion_clips = []
        self.clip_lengths = []
        self.clip_weights = []  # Weights for each motion clip
        self.total_length = 0
        
        self.preloaded_states_per_clip = {}  
        self.preloaded_next_states_per_clip = {}  
        
        self.preloaded_qpos_state = {}  
        self.preloaded_qvel_state = {}  
        
        # You can set different weights for each clip here
        self.default_weights = {
            "run_01": 1.0,
            "walk_01": 0.8,
            "walk_90_left": 0.5,
            "walk_90_right": 0.5
        }
        
        self.load_motion_clips()
    
    def load_motion_clips(self):
        """Loads the motion clips and their metadata (like lengths and weights)."""
        for name, path in self.motion_files.items():
            expert_qpos, expert_meta = pickle.load(open(path, "rb"))
            clip_length = expert_qpos.shape[0]
            
            
            
            
            self.motion_clips.append({
                'clip_name': name,
                'qpos': expert_qpos,
                'meta': expert_meta
            })
            self.clip_lengths.append(clip_length)
            
            
            # Assign the custom weight if specified, otherwise use 1.0
            weight = self.default_weights.get(name, 1.0)
            self.clip_weights.append(weight)
            
            self.total_length += clip_length
            # Save the old state of the agent
            old_state_qpos = self.env.data.qpos.copy()
            old_state_qvel = self.env.data.qvel.copy()

            # Preload data for this clip
            self.preload_data(name, expert_qpos)
            
            # Restore the old state
            self.env.data.qpos[:] = old_state_qpos
            self.env.data.qvel[:] = old_state_qvel
            mj.mj_forward(self.env.model, self.env.data)
                
    
    def preload_data(self, clip_name, expert_qpos):
        """
        Preload the states and next states for a specific clip.
        """
        preloaded_states = []
        preloaded_next_states = []
        
        preloaded_qpos = []
        preloaded_qvel = []
        
        for i in range(expert_qpos.shape[0]):
            qpos_current = expert_qpos[i]
            
            # For the last frame, use the same qpos for both current and next state
            if i == expert_qpos.shape[0] - 1:
                qpos_next = expert_qpos[i]
            else:
                qpos_next = expert_qpos[i + 1]  # Next state
            
            # Set the environment to the current state
            self.env.data.qpos[:] = qpos_current
            mj.mj_forward(self.env.model, self.env.data)
            current_obs_features = self.extract_obs_features(qpos_current)
            
            # Set the environment to the next state
            self.env.data.qpos[:] = qpos_next
            mj.mj_forward(self.env.model, self.env.data)
            next_obs_features = self.extract_obs_features(qpos_next)
            
            # Store the current and next observation
            preloaded_states.append(np.array(current_obs_features))
            preloaded_next_states.append(np.array(next_obs_features))
            
            #set it back to the current qpos
            self.env.data.qpos[:] = qpos_current
            mj.mj_forward(self.env.model, self.env.data)
          
            preloaded_qpos.append(qpos_current)
            preloaded_qvel.append(self.env.qvel[:])
            

        
        # Save the preloaded states and next states for this specific clip
        self.preloaded_states_per_clip[clip_name] = preloaded_states
        self.preloaded_next_states_per_clip[clip_name] = preloaded_next_states
        
        self.preloaded_qpos_state[clip_name] = preloaded_qpos
        self.preloaded_qvel_state[clip_name] = preloaded_qvel
        
    def sample_state_next_state(self):
        """
        Samples a pair of state and next state from preloaded data based on weights.
        """
        # First, sample a clip based on weights
        clip_name = self.sample_clip()
        
        # Retrieve the list of preloaded states and next states for this clip
        states = self.preloaded_states_per_clip[clip_name]
        next_states = self.preloaded_next_states_per_clip[clip_name]
        qpos_states = self.preloaded_qpos_state[clip_name]
        qvel_states = self.preloaded_qvel_state[clip_name]
        # Now, sample a random state and next state from within the selected clip
        state_idx = np.random.randint(len(states))
        state = states[state_idx]
        next_state = next_states[state_idx]
        
        qpos = qpos_states[state_idx]
        qvel = qvel_states[state_idx]
        
        return clip_name,state, next_state,qpos,qvel  
        
    
    def sample_clip(self) -> Dict:
        """Randomly samples a clip based on weights and returns it."""
        clip_idx = np.random.choice(len(self.motion_clips), p=self._get_normalized_weights())
        return self.motion_clips[clip_idx]

    def sample_frame(self, clip: Dict) -> np.ndarray:
        """Samples a specific frame from a given clip."""
        qpos = clip['qpos']
        frame_idx = np.random.randint(qpos.shape[0])  # Random frame index
        return self.extract_obs_features(qpos, frame_idx)

    def _get_normalized_weights(self) -> List[float]:
        """Returns normalized weights for clip selection."""
        total_weight = sum(self.clip_weights)
        return [w / total_weight for w in self.clip_weights]
    
    def extract_obs_features(self, old_qpos: np.ndarray) -> np.ndarray:
        """
        Extracts observation features for a given frame.
        Features include:
        - Linear velocity and angular velocity of the root in local coordiantes
        - Local rotation of each joint
        - Local velocity of each joint
        - 3D positions of the end-effectors in local coordinates
        """
        # Set the environment to the given qpos
        self.env.data.qpos[:] = old_qpos
        mj.mj_forward(self.env.model, self.env.data)

        
        # Extract root velocities (linear and angular)
        root_linear_velocity = self.env.data.qvel[:3]
        root_angular_velocity = self.env.data.qvel[3:6]
    
        # Transform root velocities to the character's local coordinate frame
        # Linear velocity to local coordinates using the root quaternion (global to local)
        root_linear_velocity = transform_vec(root_linear_velocity[:3], old_qpos[3:7]).ravel()

        # Angular velocity to heading coordinates using the heading quaternion
        hq = get_heading_q(old_qpos[3:7])
        root_angular_velocity = transform_vec(root_angular_velocity, hq).ravel()
        
        # Extract local rotations and velocities of each joint
        local_joint_rotations = self.env.data.qpos[7:]  # Joint rotations
        local_joint_velocities = self.env.data.qvel[6:]  # Joint velocities

        # Get end-effector positions
        end_effector_positions = self.env.get_ee_pos(transform='root')

        # Concatenate all features into a single observation array
        obs_features = np.concatenate([
            root_linear_velocity,
            root_angular_velocity,
            local_joint_rotations,
            local_joint_velocities,
            end_effector_positions
        ])
        
        
        return obs_features
    
    
 