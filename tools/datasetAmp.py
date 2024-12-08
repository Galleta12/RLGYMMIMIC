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
    
    def __init__(self, env: HumanoidBase):
        self.env = env
        # Define motion files (more clips can be added here)
        self.motion_files = {
            "run_01": 'data/motion/walk_exagerated.p',
            "run_wide_leg": 'data/motion/run_wide_leg.p',
            "run_left": 'data/motion/run_left.p',
            "run_right": 'data/motion/run_right.p',
            "run_circle": 'data/motion/run_circle.p',
            "run_ver_left": 'data/motion/run_ver_left.p',
            "run_ver_right": 'data/motion/run_ver_right.p'
        }
        
        # self.motion_files = {
        #     "run_01": 'data/motion/run_01_test.p',
        #     "walk_01": 'data/motion/walk_01_test.p',
        #     "walk_90_left": 'data/motion/walk-90-left.p',
        #     "walk_90_right": 'data/motion/walk-90-right.p'
        # }
        
        self.motion_clips = []
        self.clip_lengths = []
        self.clip_weights = []  # Weights for each motion clip
        self.total_length = 0
        
        # Preloaded AMP features per clip
        self.preloaded_states_per_clip = {}  
        self.preloaded_next_states_per_clip = {}  
        
        # You can set different weights for each clip here
        self.default_weights = {
            "run_01": 1.0,
            "walk_01": 0.8,
            "walk_90_left": 0.5,
            "walk_90_right": 0.5
        }
        
        self.load_motion_clips()
        self.height_lb = self.get_minimum_height()
    
    def get_minimum_height(self) -> float:
        """
        Returns the overall minimum height_lb across all motion clips.
        """
        min_height = float('inf')  # Start with an infinitely large number
        for clip in self.motion_clips:
            expert = clip['expert']
            min_height = min(min_height, expert['height_lb'])
        
        return min_height
    
    
    def load_motion_clips(self):
        """Loads the motion clips and their metadata (like lengths and weights)."""
        for name, path in self.motion_files.items():
            expert_qpos, expert_meta = pickle.load(open(path, "rb"))
            clip_length = expert_qpos.shape[0]
            
            # Get expert data with AMP features
            expert = get_expert(expert_qpos, expert_meta, self.env)
            
            self.motion_clips.append({
                'clip_name': name,
                'qpos': expert_qpos,
                'meta': expert_meta,
                'expert': expert
            })
            self.clip_lengths.append(clip_length)
            
            # Assign the custom weight if specified, otherwise use 1.0
            weight = self.default_weights.get(name, 1.0)
            self.clip_weights.append(weight)
            
            self.total_length += clip_length
            
            # Preload AMP-specific features for this clip
            self.preload_amp_data(name, expert)

    def preload_amp_data(self, clip_name: str, expert: Dict):
        """Preload AMP-specific data for a given clip."""
        preloaded_states = []
        preloaded_next_states = []
        
        for i in range(len(expert['qpos']) - 1):
            # Collect AMP features for current and next frames
            current_features = {
                'root_linear_velocity': expert['linear_local_root_amp'][i],
                'root_angular_velocity': expert['linear_angular_root_amp'][i],
                'local_joint_rotations': expert['local_rotation_amp'][i],
                'local_joint_velocities': expert['local_vel_amp'][i],
                'end_effector_positions': expert['local_ee_pos_amp'][i],
                'original_qpos': expert['qpos'][i],
                'original_qvel': expert['qvel'][i]
            }
            
            next_features = {
                'root_linear_velocity': expert['linear_local_root_amp'][i + 1],
                'root_angular_velocity': expert['linear_angular_root_amp'][i + 1],
                'local_joint_rotations': expert['local_rotation_amp'][i + 1],
                'local_joint_velocities': expert['local_vel_amp'][i + 1],
                'end_effector_positions': expert['local_ee_pos_amp'][i + 1],
                'original_qpos': expert['qpos'][i],
                'original_qvel': expert['qvel'][i]
            }
            
            preloaded_states.append(current_features)
            preloaded_next_states.append(next_features)
        
        # Handle the last frame (set next to itself)
        last_state_features = {
            'root_linear_velocity': expert['linear_local_root_amp'][-1],
            'root_angular_velocity': expert['linear_angular_root_amp'][-1],
            'local_joint_rotations': expert['local_rotation_amp'][-1],
            'local_joint_velocities': expert['local_vel_amp'][-1],
            'end_effector_positions': expert['local_ee_pos_amp'][-1],
            'original_qpos': expert['qpos'][-1],
            'original_qvel': expert['qvel'][-1]
        }
        preloaded_states.append(last_state_features)
        preloaded_next_states.append(last_state_features)
        
        # Store in the dictionary
        self.preloaded_states_per_clip[clip_name] = preloaded_states
        self.preloaded_next_states_per_clip[clip_name] = preloaded_next_states

    def sample_clip(self) -> Dict:
        """Randomly samples a clip based on weights and returns it."""
        clip_idx = np.random.choice(len(self.motion_clips), p=self._get_normalized_weights())
        return self.motion_clips[clip_idx]

    def sample_state_next_state_batch(self, batch_size: int):
        """Samples a batch of state and next state pairs from preloaded AMP data."""
        states_batch = []
        next_states_batch = []
        
        for _ in range(batch_size):
            # Sample a clip based on weights
            clip = self.sample_clip()
            clip_name = clip['clip_name']
            preloaded_states = self.preloaded_states_per_clip[clip_name]
            preloaded_next_states = self.preloaded_next_states_per_clip[clip_name]
            
            # Sample an index from the chosen clip
            idx = np.random.randint(len(preloaded_states))
            
            # Append the sampled state and next state to the batch
            states_batch.append(preloaded_states[idx])
            next_states_batch.append(preloaded_next_states[idx])
        
        return states_batch, next_states_batch
        
    
    def _get_normalized_weights(self) -> List[float]:
        """Returns normalized weights for clip selection."""
        total_weight = sum(self.clip_weights)
        return [w / total_weight for w in self.clip_weights]