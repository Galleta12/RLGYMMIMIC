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
                 'com', 'head_pos', 'ee_pos', 'ee_wpos', 'bquat', 'bangvel'}
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
        
        #if not using the standard humanoid model
        if not env.cfg.use_standard_model:
            head_pos = env.get_body_com('head').copy()
        else:
            head_pos = env.get_body_com('neck').copy()
            
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

    # Get expert body quaternions
    for i in range(1, expert_qpos.shape[0]):
        bangvel = get_angvel_fd(expert['bquat'][i - 1], expert['bquat'][i], env.dt)
        expert['bangvel'].append(bangvel)
    expert['bangvel'].insert(0, expert['bangvel'][0].copy())

    for key in feat_keys:
        expert[key] = np.vstack(expert[key])
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