import numpy as np
import math
from rfc_utils.rfc_math import *

def world_rfc_implicit_reward(env, state, action, info):
    # reward coefficients
    cfg = env.cfg
    ws = cfg.reward_weights
    w_p, w_v, w_e, w_c, w_vf = ws.get('w_p', 0.6), ws.get('w_v', 0.1), ws.get('w_e', 0.2), ws.get('w_c', 0.1), ws.get('w_vf', 0.0)
    k_p, k_v, k_e, k_c, k_vf = ws.get('k_p', 2), ws.get('k_v', 0.005), ws.get('k_e', 20), ws.get('k_c', 1000), ws.get('k_vf', 1)
    v_ord = ws.get('v_ord', 2)
    # data from env
    t = env.cur_t
    ind = env.get_expert_index(t)
    prev_bquat = env.prev_bquat
    # learner
    cur_ee = env.get_ee_pos(None)
    cur_bquat = env.get_body_quat()
    cur_bangvel = get_angvel_fd(prev_bquat, cur_bquat, env.dt)
    cur_com = env.get_com()
    # expert
    e_qpos = env.get_expert_attr('qpos', ind)
    e_rpos = e_qpos[:3]
    e_ee = env.get_expert_attr('ee_wpos', ind).copy()
    e_com = env.get_expert_attr('com', ind).copy()
    e_bquat = env.get_expert_attr('bquat', ind)
    e_bangvel = env.get_expert_attr('bangvel', ind)
    expert = env.expert
    if expert['meta']['cyclic']:
        init_pos = expert['init_pos']
        cycle_h = expert['cycle_relheading']
        cycle_pos = expert['cycle_pos']
        orig_rpos = e_rpos.copy()
        e_rpos = quat_mul_vec(cycle_h, e_rpos - init_pos) + cycle_pos
        e_com = quat_mul_vec(cycle_h, e_com - orig_rpos) + e_rpos
        for i in range(e_ee.shape[0] // 3):
            e_ee[3*i: 3*i+3] = quat_mul_vec(cycle_h, e_ee[3*i: 3*i+3] - orig_rpos) + e_rpos

    if not expert['meta']['cyclic'] and env.start_ind + t >= expert['len']:
        e_bangvel = np.zeros_like(e_bangvel)
    # pose reward
    pose_diff = multi_quat_norm(multi_quat_diff(cur_bquat, e_bquat))
    pose_diff[1:] *= cfg.b_diffw
    pose_dist = np.linalg.norm(pose_diff)
    pose_reward = math.exp(-k_p * (pose_dist ** 2))
    # velocity reward
    vel_dist = np.linalg.norm(cur_bangvel - e_bangvel, ord=v_ord)
    vel_reward = math.exp(-k_v * (vel_dist ** 2))
    # ee reward
    ee_dist = np.linalg.norm(cur_ee - e_ee)
    ee_reward = math.exp(-k_e * (ee_dist ** 2))
    # com reward
    com_dist = np.linalg.norm(cur_com - e_com)
    com_reward = math.exp(-k_c * (com_dist ** 2))
    # residual force reward
    if w_vf > 0.0:
        vf = action[-env.vf_dim:]
        vf_reward = math.exp(-k_vf * (np.linalg.norm(vf) ** 2))
    else:
        vf_reward = 0.0
    # overall reward
    reward = w_p * pose_reward + w_v * vel_reward + w_e * ee_reward + w_c * com_reward + w_vf * vf_reward
    reward /= w_p + w_v + w_e + w_c + w_vf
    return reward, np.array([pose_reward, vel_reward, ee_reward, com_reward, vf_reward])



def world_rfc_implicit_reward_gym(env,  action):
    # reward coefficients
    cfg = env.cfg
    ws = cfg.reward_weights
    w_p, w_v, w_e, w_c, w_vf = ws.get('w_p', 0.6), ws.get('w_v', 0.1), ws.get('w_e', 0.2), ws.get('w_c', 0.1), ws.get('w_vf', 0.0)
    k_p, k_v, k_e, k_c, k_vf = ws.get('k_p', 2), ws.get('k_v', 0.005), ws.get('k_e', 20), ws.get('k_c', 1000), ws.get('k_vf', 1)
    v_ord = ws.get('v_ord', 2)
    # data from env
    t = env.cur_t
    ind = env.get_expert_index(t)
    prev_bquat = env.prev_bquat
    # learner
    cur_ee = env.get_ee_pos(None)
    cur_bquat = env.get_body_quat()
    cur_bangvel = get_angvel_fd(prev_bquat, cur_bquat, env.dt)
    cur_com = env.get_com()
    # expert
    e_qpos = env.get_expert_attr('qpos', ind)
    e_rpos = e_qpos[:3]
    e_ee = env.get_expert_attr('ee_wpos', ind).copy()
    e_com = env.get_expert_attr('com', ind).copy()
    e_bquat = env.get_expert_attr('bquat', ind)
    e_bangvel = env.get_expert_attr('bangvel', ind)
    expert = env.expert
    if expert['meta']['cyclic']:
        init_pos = expert['init_pos']
        cycle_h = expert['cycle_relheading']
        cycle_pos = expert['cycle_pos']
        orig_rpos = e_rpos.copy()
        e_rpos = quat_mul_vec(cycle_h, e_rpos - init_pos) + cycle_pos
        e_com = quat_mul_vec(cycle_h, e_com - orig_rpos) + e_rpos
        for i in range(e_ee.shape[0] // 3):
            e_ee[3*i: 3*i+3] = quat_mul_vec(cycle_h, e_ee[3*i: 3*i+3] - orig_rpos) + e_rpos

    if not expert['meta']['cyclic'] and env.start_ind + t >= expert['len']:
        e_bangvel = np.zeros_like(e_bangvel)
    # pose reward
    pose_diff = multi_quat_norm(multi_quat_diff(cur_bquat, e_bquat))
    pose_diff[1:] *= cfg.b_diffw
    pose_dist = np.linalg.norm(pose_diff)
    pose_reward = math.exp(-k_p * (pose_dist ** 2))
    # velocity reward
    vel_dist = np.linalg.norm(cur_bangvel - e_bangvel, ord=v_ord)
    vel_reward = math.exp(-k_v * (vel_dist ** 2))
    # ee reward
    ee_dist = np.linalg.norm(cur_ee - e_ee)
    ee_reward = math.exp(-k_e * (ee_dist ** 2))
    # com reward
    com_dist = np.linalg.norm(cur_com - e_com)
    com_reward = math.exp(-k_c * (com_dist ** 2))
    # residual force reward
    if w_vf > 0.0:
        vf = action[-env.vf_dim:]
        vf_reward = math.exp(-k_vf * (np.linalg.norm(vf) ** 2))
    else:
        vf_reward = 0.0
    # overall reward
    reward = w_p * pose_reward + w_v * vel_reward + w_e * ee_reward + w_c * com_reward + w_vf * vf_reward
    reward /= w_p + w_v + w_e + w_c + w_vf
    return reward, np.array([pose_reward, vel_reward, ee_reward, com_reward, vf_reward])












reward_func = { 
    'world_rfc_implicit': world_rfc_implicit_reward,
}