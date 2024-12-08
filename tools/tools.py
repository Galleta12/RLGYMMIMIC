import os
import sys
import mujoco as mj
sys.path.append(os.getcwd())
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from some_math.math_utils import *
#from agent_envs.humanoid_env import HumanoidTemplate

def get_expert(expert_qpos, expert_meta,env):
    # Save the old state
    old_state_qpos = env.data.qpos.copy()
    old_state_qvel = env.data.qvel.copy()

    expert = {'qpos': expert_qpos, 'meta': expert_meta}
    feat_keys = {'qvel', 'rlinv', 'rlinv_local', 'rangv', 'rq_rmh',
                 'com', 'head_pos', 'ee_pos', 'ee_wpos', 'bquat', 'bangvel',
                 'linear_local_root_amp','linear_angular_root_amp','local_rotation_amp','local_vel_amp',
                 'local_ee_pos_amp','xpos'}
    for key in feat_keys:
        expert[key] = []

    for i in range(expert_qpos.shape[0]):
        qpos = expert_qpos[i]
        env.data.qpos[:] = qpos
        mj.mj_forward(env.model, env.data)  # Advance the simulation to the current state
        
        rq_rmh = de_heading(qpos[3:7])
        ee_pos = env.get_ee_pos(env.cfg.obs_coord)
        ee_wpos = env.get_ee_pos(None)
        bquat = env.get_body_quat()
        com = env.get_com()
        
        xpos = np.array([env.data.xpos[j].copy() for j in range(env.model.nbody)])  # Copy is important!
        expert['xpos'].append(xpos)
        
        # print('exper ss',xpos.shape)
        
        
        
        #if not using the standard humanoid model
        if not env.cfg.use_standard_model:
            head_pos = env.get_body_com('head').copy()
        else:
            head_pos = env.get_body_com('neck').copy()
        #calculated after the first frame  
        if i > 0:
            prev_qpos = expert_qpos[i - 1]
            qvel = get_qvel_fd_new(prev_qpos, qpos, env.dt)
            qvel = qvel.clip(-10.0, 10.0)
            rlinv = qvel[:3].copy()
            #transform the relative linear v
            if env.cfg.obs_coord =='root':
                rlinv_local = transform_vec(qvel[:3].copy(), qpos[3:7])
            elif env.cfg.obs_coord =='heading':
                hq = get_heading_q(qpos[3:7])
                rlinv_local = transform_vec(qvel[:3].copy(), hq)
                
            rangv = qvel[3:6].copy()
            expert['qvel'].append(qvel)
            expert['rlinv'].append(rlinv)
            expert['rlinv_local'].append(rlinv_local)
            expert['rangv'].append(rangv)
        
        expert['ee_pos'].append(ee_pos)
        expert['ee_wpos'].append(ee_wpos)
        expert['bquat'].append(bquat)
        expert['com'].append(com)
        expert['head_pos'].append(head_pos)
        expert['rq_rmh'].append(rq_rmh)
        
        

    expert['qvel'].insert(0, expert['qvel'][0].copy())
    expert['rlinv'].insert(0, expert['rlinv'][0].copy())
    expert['rlinv_local'].insert(0, expert['rlinv_local'][0].copy())
    expert['rangv'].insert(0, expert['rangv'][0].copy())

    #for the amp obs features
    for i in range(expert_qpos.shape[0]):
        qpos = expert_qpos[i]
        qvel = expert['qvel'][i]
        # AMP-Specific Features
        amp_obs = amp_obs_feature(env, expert, qpos, qvel)
        expert['linear_local_root_amp'].append(amp_obs['root_linear_velocity'])
        expert['linear_angular_root_amp'].append(amp_obs['root_angular_velocity'])
        expert['local_rotation_amp'].append(amp_obs['local_joint_rotations'])
        expert['local_vel_amp'].append(amp_obs['local_joint_velocities'])
        expert['local_ee_pos_amp'].append(amp_obs['end_effector_positions'])
        
    
    
    # Get expert body quaternions
    for i in range(1, expert_qpos.shape[0]):
        bangvel = get_angvel_fd(expert['bquat'][i - 1], expert['bquat'][i], env.dt)
        expert['bangvel'].append(bangvel)
    expert['bangvel'].insert(0, expert['bangvel'][0].copy())

    for key in feat_keys:
        
        if key == 'xpos':
            expert[key] = np.array(expert[key])  # This keeps the shape as (T, nbody, 3)
        else:
            expert[key] = np.vstack(expert[key])  # For other keys
    expert['len'] = expert['qpos'].shape[0]
    expert['height_lb'] = expert['qpos'][:, 2].min()
    expert['head_height_lb'] = expert['head_pos'][:, 2].min()
    if expert_meta.get('cyclic', False):
        expert['init_heading'] = get_heading_q(expert_qpos[0, 3:7])
        expert['init_pos'] = expert_qpos[0, :3].copy()

    # Restore the old state
    env.data.qpos[:] = old_state_qpos
    env.data.qvel[:] = old_state_qvel
    mj.mj_forward(env.model, env.data)
    
    return expert



def amp_obs_feature(env,expert,qpos,qvel):
    # Extract root velocities (linear and angular)
    root_linear_velocity = qvel[:3]
    root_angular_velocity = qvel[3:6]
    
    # Transform root velocities to the character's local coordinate frame
    # Linear velocity to local coordinates using the root quaternion (global to local)
    root_linear_velocity = transform_vec(root_linear_velocity[:3], qpos[3:7]).ravel()
    # Angular velocity to heading coordinates using the heading quaternion
    hq = get_heading_q(qpos[3:7])
    root_angular_velocity = transform_vec(root_angular_velocity, hq).ravel()
    
    # Extract local rotations and velocities of each joint
    local_joint_rotations = qpos[7:]  # Joint rotations
    local_joint_velocities = qvel[6:]  # Joint velocities

    # Get end-effector positions
    end_effector_positions = env.get_ee_pos(transform='root')

    obs_features = {
        'root_linear_velocity': root_linear_velocity,
        'root_angular_velocity': root_angular_velocity,
        'local_joint_rotations': local_joint_rotations,
        'local_joint_velocities': local_joint_velocities,
        'end_effector_positions': end_effector_positions
    }

    return obs_features
    



def get_body_qposaddr(model):
    body_qposaddr = dict()
    for i in range(model.nbody):
        body_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, i)
        if body_name is None:
            continue
        start_joint = model.body_jntadr[i]
        if start_joint < 0:
            continue
        end_joint = start_joint + model.body_jntnum[i]
        start_qposaddr = model.jnt_qposadr[start_joint]
        if end_joint < len(model.jnt_qposadr):
            end_qposaddr = model.jnt_qposadr[end_joint]
        else:
            end_qposaddr = model.nq
        body_qposaddr[body_name] = (start_qposaddr, end_qposaddr)
    return body_qposaddr