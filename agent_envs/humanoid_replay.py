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
from rfc_utils.rfc_mujoco import get_body_qposaddr
from rfc_utils.tools import get_expert
from rfc_utils.rfc_math import *
from rfc_utils.transformation import quaternion_from_euler
from common.mujoco_envs import MujocoEnv

DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 1,
    "distance": 4.0,
    "lookat": np.array((0.0, 0.0, 2.0)),
    "elevation": -20.0,
}

class HumanoidReplay(MujocoEnv):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 30,
    }

    def __init__(self, model, 
                 frame_skip: int = 1,   
                 default_camera_config: Dict[str, Union[float, int]] = DEFAULT_CAMERA_CONFIG,**kwargs):
        

        print("this is the model_path",model)
   

        MujocoEnv.__init__(
            self, os.path.abspath(f"assets/mujoco_models/{model}"),  frame_skip,  
            default_camera_config=DEFAULT_CAMERA_CONFIG, observation_space=None,**kwargs
        )
        self.metadata['render_fps'] = int(np.round(self.metadata['render_fps'] / frame_skip))
    
        #self.metadata['render_fps'] = int(np.round(1.0 / self.dt))
 
    def forward(self):
        mj.mj_forward(self.model, self.data)

    
    def reset_model(self):
        c = 0
        self.set_state(
            self.np_random.uniform(low=-c, high=c, size=self.model.nq),
            self.np_random.uniform(low=-c, high=c, size=self.model.nv)
        )
        return None