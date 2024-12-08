import numpy as np
import math
from some_math.math_utils import *

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
        #print('reward cycle')
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
        #print('residual reward')
        vf = action[-env.vf_dim:]
        vf_reward = math.exp(-k_vf * (np.linalg.norm(vf) ** 2))
    else:
        vf_reward = 0.0
    # overall reward
    reward = w_p * pose_reward + w_v * vel_reward + w_e * ee_reward + w_c * com_reward + w_vf * vf_reward
    reward /= w_p + w_v + w_e + w_c + w_vf
    
    reward_info ={ 'pose_reward': pose_reward,
                  'vel_reward':vel_reward,
                  'com_reward':com_reward,
                  'vf_reward': vf_reward} 
    
    
    return reward, np.array([pose_reward, vel_reward, ee_reward, com_reward, vf_reward])
    #return reward, reward_info


def world_reward(env, state, action, info):
    # reward coefficients
    #print("world reward")
    cfg = env.cfg
    ws = cfg.reward_weights
    w_p, w_v, w_e, w_c = ws.get('w_p', 0.6), ws.get('w_v', 0.1), ws.get('w_e', 0.2), ws.get('w_c', 0.1)
    k_p, k_v, k_e, k_c = ws.get('k_p', 2), ws.get('k_v', 0.005), ws.get('k_e', 20), ws.get('k_c', 1000)
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
    
    # overall reward
    reward = w_p * pose_reward + w_v * vel_reward + w_e * ee_reward + w_c * com_reward
    reward /= w_p + w_v + w_e + w_c
    return reward, np.array([pose_reward, vel_reward, ee_reward, com_reward])

def local_rfc_implicit_reward(env, state, action, info):
    # reward coefficients
    cfg = env.cfg
    ws = cfg.reward_weights
    w_p, w_v, w_e, w_rp, w_rv, w_vf = ws.get('w_p', 0.5), ws.get('w_v', 0.0), ws.get('w_e', 0.2), ws.get('w_rp', 0.1), ws.get('w_rv', 0.1), ws.get('w_vf', 0.1)
    k_p, k_v, k_e, k_vf = ws.get('k_p', 2), ws.get('k_v', 0.005), ws.get('k_e', 20), ws.get('k_vf', 1)
    k_rh, k_rq, k_rl, k_ra = ws.get('k_rh', 300), ws.get('k_rq', 300), ws.get('k_rl', 5.0), ws.get('k_ra', 0.5)
    v_ord = ws.get('v_ord', 2)
    
    # data from env
    t = env.cur_t
    ind = env.get_expert_index(t)
    prev_bquat = env.prev_bquat
    prev_qpos = env.prev_qpos
    # learner
    cur_qpos = env.data.qpos.copy()
    cur_qvel = get_qvel_fd_new(prev_qpos, cur_qpos, env.dt, cfg.obs_coord)
    cur_rlinv_local = cur_qvel[:3]
    cur_rangv = cur_qvel[3:6]
    cur_rq_rmh = de_heading(cur_qpos[3:7])
    cur_ee = env.get_ee_pos(cfg.obs_coord)
    cur_bquat = env.get_body_quat()
    cur_bangvel = get_angvel_fd(prev_bquat, cur_bquat, env.dt)
    # expert
    e_qpos = env.get_expert_attr('qpos', ind)
    e_rlinv_local = env.get_expert_attr('rlinv_local', ind)
    e_rangv = env.get_expert_attr('rangv', ind)
    e_rq_rmh = env.get_expert_attr('rq_rmh', ind)
    e_ee = env.get_expert_attr('ee_pos', ind)
    e_bquat = env.get_expert_attr('bquat', ind)
    e_bangvel = env.get_expert_attr('bangvel', ind)
    # pose reward
    pose_diff = multi_quat_norm(multi_quat_diff(cur_bquat[4:], e_bquat[4:]))    # ignore root
    pose_diff *= cfg.b_diffw
    pose_dist = np.linalg.norm(pose_diff)
    pose_reward = math.exp(-k_p * (pose_dist ** 2))
    # velocity reward
    vel_dist = np.linalg.norm(cur_bangvel[3:] - e_bangvel[3:], ord=v_ord)  # ignore root
    vel_reward = math.exp(-k_v * (vel_dist ** 2))
    # ee reward
    ee_dist = np.linalg.norm(cur_ee - e_ee)
    ee_reward = math.exp(-k_e * (ee_dist ** 2))
    # root position reward
    root_height_dist = cur_qpos[2] - e_qpos[2]
    root_quat_dist = multi_quat_norm(multi_quat_diff(cur_rq_rmh, e_rq_rmh))[0]
    root_pose_reward = math.exp(-k_rh * (root_height_dist ** 2) - k_rq * (root_quat_dist ** 2))
    # root velocity reward
    root_linv_dist = np.linalg.norm(cur_rlinv_local - e_rlinv_local)
    root_angv_dist = np.linalg.norm(cur_rangv - e_rangv)
    root_vel_reward = math.exp(-k_rl * (root_linv_dist ** 2) - k_ra * (root_angv_dist ** 2))
    # residual force reward
    if w_vf > 0.0:
        vf = action[-env.vf_dim:]
        vf_reward = math.exp(-k_vf * (np.linalg.norm(vf) ** 2))
    else:
        vf_reward = 0.0
    # overall reward
    reward = w_p * pose_reward + w_v * vel_reward + w_e * ee_reward + w_rp * root_pose_reward + w_rv * root_vel_reward + w_vf * vf_reward
    reward /= w_p + w_v + w_e + w_rp + w_rv + w_vf
    return reward, np.array([pose_reward, vel_reward, ee_reward, root_pose_reward, root_vel_reward, vf_reward])











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




def reward_direction(env,state, action, info):
    
    # Retrieve target direction (d*) and speed (v*) from environment or config
    d_star = env.target_direction_local 
    v_star = env.target_speed      
    
    x_com_velocity = env.get_com_velocity()
    
    # Project COM velocity onto target direction (d*)
    projected_velocity = np.dot(d_star, x_com_velocity)  # d* · ẋ_com

    # Compute the reward as per the given formula
    reward = math.exp(-0.25 * ((v_star - projected_velocity) ** 2))

    w_g = 0.5
    
    return w_g * reward, np.array([w_g * reward])

def reward_location(env, state, action, info):
    
 
    
    # Retrieve the target position (x*) in the global frame 
    x_star_global = env.get_target_position() # Target position in global coordinates
    x_root_t_global = env.data.qpos[:3]  # Agent's root position in global coordinates

    # Compute d* as a unit vector pointing from the agent's root position to the target
    direction_to_target = x_star_global[:2] - x_root_t_global[:2]  # Only consider horizontal plane [x, y]
    distance_to_target = np.linalg.norm(direction_to_target)
    d_star = direction_to_target / (distance_to_target + 1e-8)  # Normalize and prevent division by zero

    # Calculate the first term (distance-based reward)
    distance_reward = 0.7 * math.exp(-0.5 * (distance_to_target ** 2))

    # Calculate the second term (velocity-based reward)
    v_star = 1.0  # Minimum target speed threshold
    x_com_velocity = env.get_com_velocity()[:2]  # COM velocity in the horizontal plane [x, y]
    projected_velocity = np.dot(d_star, x_com_velocity)  # Project COM velocity onto d_star direction
    velocity_reward = 0.3 * math.exp(-((max(0, v_star - projected_velocity)) ** 2))

    
    
    # Overall reward
    reward = distance_reward + velocity_reward

    #w_g = 0.5
    
    
    
    #print('rewards', distance_reward,'vel', velocity_reward)
    return reward, np.array([distance_reward, velocity_reward])


def compute_reward_amp(env,action, style_reward,task_reward):
    
    cfg = env.cfg
    ws = cfg.reward_weights
    w_task, w_style, w_vf = ws.get('w_task', 0.5), ws.get('w_style', 0.5), ws.get('w_vf', 0.0)
    k_vf =  ws.get('k_vf', 1)
    
    # residual force reward
    if w_vf > 0.0:
   
        
        vf = action[-env.vf_dim:]
        vf_reward = math.exp(-k_vf * (np.linalg.norm(vf) ** 2))
    else:
        vf_reward = 0.0
    reward =  w_task * task_reward + w_style * style_reward + w_vf * vf_reward
    reward /= w_task +w_style + w_vf
    
    return reward,vf_reward
    
    




    



reward_func = { 
    'world_rfc_implicit': world_rfc_implicit_reward,
    'local_rfc_implicit': local_rfc_implicit_reward,
    'world_reward': world_reward,
    'reward_direction': reward_direction,
    'reward_location': reward_location,
}