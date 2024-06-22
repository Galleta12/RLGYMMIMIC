from os import path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray
import mujoco
import gymnasium as gym
from gymnasium import error, logger, spaces
from gymnasium.spaces import Space
from gymnasium.utils import seeding
from pathlib import Path

DEFAULT_SIZE = 500
DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 1,
    "distance": 4.0,
    "lookat": np.array((0.0, 0.0, 2.0)),
    "elevation": -20.0,
}

class MujocoEnv:
    """Superclass for MuJoCo environments."""

    def __init__(self, fullpath, frame_skip):
        if not path.exists(fullpath):
            # try the default assets path
            fullpath = path.join(Path(__file__).parent.parent.parent.parent, 'assets/mujoco_models', path.basename(fullpath))
            if not path.exists(fullpath):
                raise IOError("File %s does not exist" % fullpath)
        self.frame_skip = frame_skip
        self.model = mujoco.MjModel.from_xml_path(fullpath)
        self.data = mujoco.MjData(self.model)
        self.viewer = None
        self._viewers = {}
        self.obs_dim = None
        self.action_space = None
        self.observation_space = None
        self.np_random = None
        self.cur_t = 0  # number of steps taken

        self.metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second': int(np.round(1.0 / self.dt))
        }

        self.init_qpos = self.data.qpos.ravel().copy()
        self.init_qvel = self.data.qvel.ravel().copy()
        self.prev_qpos = None
        self.prev_qvel = None
        self.seed()

        from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer

        self.mujoco_renderer = MujocoRenderer(
            self.model, self.data, DEFAULT_CAMERA_CONFIG
        )
    
    
    def set_spaces(self):
        observation, _reward, done, _info = self.step(np.zeros(self.model.nu))
        self.obs_dim = observation.size
        bounds = self.model.actuator_ctrlrange.copy()
        low = bounds[:, 0]
        high = bounds[:, 1]
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        high = np.inf * np.ones(self.obs_dim)
        low = -high
        self.observation_space = spaces.Box(low, high, dtype=np.float32)
    

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
        
    
    
        # ----------------------------
    def step(self, action):
        """
        Step the environment forward.
        """
        raise NotImplementedError
    
    def reset_model(self):
        """
        Reset the robot degrees of freedom (qpos and qvel).
        Implement this in each subclass.
        """
        raise NotImplementedError

    
    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.cur_t = 0
        ob = self.reset_model()
        old_viewer = self.viewer
        for mode, v in self._viewers.items():
            self.viewer = v
            self.viewer_setup(mode)
        self.viewer = old_viewer
        return ob
    
    def set_state(self, qpos, qvel):
        assert qpos.shape == (self.model.nq,) and qvel.shape == (self.model.nv,)
        
        self.data.qpos[:] = np.copy(qpos)
        self.data.qvel[:] = np.copy(qvel)
        if self.model.na == 0:
            self.data.act[:] = None
        mujoco.mj_forward(self.model, self.data)
    
    @property
    def dt(self):
        return self.model.opt.timestep * self.frame_skip
    
    def do_simulation(self, ctrl, n_frames):
        self.data.ctrl[:] = ctrl

        mujoco.mj_step(self.model, self.data, nstep=n_frames)

        # As of MuJoCo 2.0, force-related quantities like cacc are not computed
        # unless there's a force sensor in the model.
        # See https://github.com/openai/gym/issues/1541
        mujoco.mj_rnePostConstraint(self.model, self.data)
    

    def render(self):
        return self.mujoco_renderer.render(
            self.render_mode, self.camera_id, self.camera_name
        )

    def close(self):
        if self.mujoco_renderer is not None:
            self.mujoco_renderer.close()

    def get_body_com(self, body_name):
        return self.data.body(body_name).xpos

    def state_vector(self):
        return np.concatenate([
            self.data.qpos.flat,
            self.data.qvel.flat
        ])
