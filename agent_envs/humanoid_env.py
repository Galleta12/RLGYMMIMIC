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
DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 1,
    "distance": 4.0,
    "lookat": np.array((0.0, 0.0, 2.0)),
    "elevation": -20.0,
}

class HumanoidTemplate(MujocoEnv):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 30,
    }

    def __init__(self, cfg, 
                 frame_skip: int = 15,   
                 default_camera_config: Dict[str, Union[float, int]] = DEFAULT_CAMERA_CONFIG,**kwargs):
        

        print("this is the model_path",cfg.mujoco_model_file)
   

        MujocoEnv.__init__(
            self, os.path.abspath(f"assets/mujoco_models/{cfg.mujoco_model_file}"), 15,  
            default_camera_config=DEFAULT_CAMERA_CONFIG, observation_space=None,**kwargs
        )
    
        self.metadata['render_fps'] = int(np.round(1.0 / self.dt))
        
        self.cfg = cfg
        self.ee_name = ['lfoot', 'rfoot', 'lwrist', 'rwrist', 'head']
        self.end_reward = 0.0
        self.start_ind = 0
        self.body_names = self.set_body_names()
        self.body_qposaddr = get_body_qposaddr(self.model)
        self.bquat = self.get_body_quat()
        self.prev_bquat = None
        self.expert = None
        self.load_expert()
        self.set_spaces()
    
    def set_body_names(self):
        body_names = [mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_BODY, i) for i in range(self.model.nbody)]
        return body_names
      
    
    def load_expert(self):
        expert_qpos, expert_meta = pickle.load(open(self.cfg.expert_traj_file, "rb"))
        # print(expert_meta)
        self.expert = get_expert(expert_qpos, expert_meta, self)

    def set_spaces(self):
        cfg = self.cfg
        self.ndof = self.model.actuator_ctrlrange.shape[0]
        self.vf_dim = 0
        if cfg.residual_force:
            #print("is residual force:,", cfg.residual_force)
            if cfg.residual_force_mode == 'implicit':
                self.vf_dim = 6
    
        self.action_dim = self.ndof + self.vf_dim
        self.action_space = gym.spaces.Box(low=-np.ones(self.action_dim), high=np.ones(self.action_dim), dtype=np.float32)
        self.obs_dim = self.get_obs().size
        high = np.inf * np.ones(self.obs_dim)
        low = -high
        self.observation_space = gym.spaces.Box(low, high, dtype=np.float32)
    
    def get_obs(self):
        
        obs = self.get_full_obs()
        return obs

    def get_full_obs(self):
        data = self.data
        qpos = data.qpos.copy()
        qvel = data.qvel.copy()
        # transform velocity
        if self.cfg.obs_coord == 'root':
            qvel[:3] = transform_vec(qvel[:3], qpos[3:7]).ravel()
        elif self.cfg.obs_coord =='heading':
            hq = get_heading_q(qpos[3:7])
            qvel[:3] = transform_vec(qvel[:3], hq).ravel()
            
        obs = []
        # pos
        obs.append(qpos[2:])
            
        obs.append(qvel)
        # phase
        if self.cfg.obs_phase:
            
            phase = self.get_phase()
            obs.append(np.array([phase]))
        obs = np.concatenate(obs)
        return obs
    
    def get_ee_pos(self, transform):
        data = self.data
        #ee_name = ['lfoot', 'rfoot', 'lwrist', 'rwrist', 'head']
        ee_pos = []
        root_pos = data.qpos[:3]
        root_q = data.qpos[3:7].copy()
        for name in self.ee_name:
            bone_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, name)
            bone_vec = self.data.xpos[bone_id]
            if transform is not None:
                
                bone_vec = bone_vec - root_pos
                if self.cfg.obs_coord == 'root':
                    bone_vec = transform_vec(bone_vec, root_q)
                elif self.cfg.obs_coord == 'heading':
                    hq = get_heading_q(root_q)
                    bone_vec = transform_vec(bone_vec, hq)
                    
            ee_pos.append(bone_vec)
        return np.concatenate(ee_pos)

    def get_body_quat(self):
        qpos = self.data.qpos.copy()
        body_quat = [qpos[3:7]]
        for body in self.body_names[1:]:
            if body == 'root' or not body in self.body_qposaddr:
                continue
            start, end = self.body_qposaddr[body]
            euler = np.zeros(3)
            euler[:end - start] = qpos[start:end]
            quat = quaternion_from_euler(euler[0], euler[1], euler[2])
            body_quat.append(quat)
        body_quat = np.concatenate(body_quat)
        return body_quat

    def get_com(self):
        return self.data.subtree_com[0, :].copy()
    
    def compute_desired_accel(self, qpos_err, qvel_err, k_p, k_d):
        dt = self.model.opt.timestep
        nv = self.model.nv
        M = np.zeros((nv, nv))
        mj.mj_fullM(self.model, M, self.data.qM)
        C = self.data.qfrc_bias.copy()
        K_p = np.diag(k_p)
        K_d = np.diag(k_d)
        q_accel = cho_solve(cho_factor(M + K_d * dt, overwrite_a=True, check_finite=False),
                            -C[:, None] - K_p.dot(qpos_err[:, None]) - K_d.dot(qvel_err[:, None]), overwrite_b=True, check_finite=False)
        return q_accel.squeeze()
    
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
        # k_p = np.zeros(qvel.shape[0])
        # k_d = np.zeros(qvel.shape[0])
        # k_p[6:] = cfg.jkp
        # k_d[6:] = cfg.jkd
        # qpos_err = np.concatenate((np.zeros(6), qpos[7:] + qvel[6:]*dt - target_pos))
        # qvel_err = qvel
        # q_accel = self.compute_desired_accel(qpos_err, qvel_err, k_p, k_d)
        # qvel_err += q_accel * dt
        # torque = -cfg.jkp * qpos_err[6:] - cfg.jkd * qvel_err[6:]
        # return torque

    """ RFC-Implicit """
    def rfc_implicit(self, vf):
        vf *= self.cfg.residual_force_scale
        hq = get_heading_q(self.data.qpos[3:7])
        vf[:3] = quat_mul_vec(hq, vf[:3])
        self.data.qfrc_applied[:vf.shape[0]] = vf

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
    
        fail = self.expert is not None and self.data.qpos[2] < self.expert['height_lb'] - 0.1
        cyclic = self.expert['meta']['cyclic']
        end =  (cyclic and self.cur_t >= cfg.env_episode_len) or (not cyclic and self.cur_t + self.start_ind >= self.expert['len'] + cfg.env_expert_trail_steps)
        done = fail or end
        obs = self.get_obs()
        return obs, reward, done, {'fail': fail, 'end': end}
    
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
    
    def update_expert(self):
        expert = self.expert
        if expert['meta']['cyclic']:
            if self.cur_t == 0:
                expert['cycle_relheading'] = np.array([1, 0, 0, 0])
                expert['cycle_pos'] = expert['init_pos'].copy()
            elif self.get_expert_index(self.cur_t) == 0:
                expert['cycle_relheading'] = quat_mul(get_heading_q(self.data.qpos[3:7]),
                                                              quat_inverse_no_norm(expert['init_heading']))
                expert['cycle_pos'] = np.concatenate((self.data.qpos[:2], expert['init_pos'][[2]]))
    
    def get_phase(self):
        ind = self.get_expert_index(self.cur_t)
        return ind / self.expert['len']

    def get_expert_index(self, t):
        #print("expert idx ", self.expert['meta']['cyclic'])
        return (self.start_ind + t) % self.expert['len'] \
                if self.expert['meta']['cyclic'] else min(self.start_ind + t, self.expert['len'] - 1)

    def get_expert_offset(self, t):
        if self.expert['meta']['cyclic']:
            n = (self.start_ind + t) // self.expert['len']
            offset = self.expert['meta']['cycle_offset'] * n
        else:
            offset = np.zeros(2)
        return offset

    def get_expert_attr(self, attr, ind):
        return self.expert[attr][ind, :]

