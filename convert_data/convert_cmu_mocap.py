import os
import sys
import numpy as np
import mujoco as mj
import math
from mujoco.glfw import glfw
sys.path.append(os.getcwd())


from rfc_utils.transformation import quaternion_from_euler
from rfc_scripts.mocap.pose import load_amc_file, interpolated_traj
from rfc_utils.rfc_mujoco import get_body_qposaddr
from common.viewer import MyViewer
import pickle
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--render', action='store_true', default=False)
parser.add_argument('--amc_id', type=str, default=None)
parser.add_argument('--out_id', type=str, default=None)
parser.add_argument('--model_file', type=str, default="mocap_v2")
parser.add_argument('--mocap_fr', type=int, default=120)
parser.add_argument('--scale', type=float, default=0.45)
parser.add_argument('--dt', type=float, default=1/30.0)
parser.add_argument('--offset_z', type=float, default=0.0)
parser.add_argument('--fix_feet', action='store_true', default=False)
parser.add_argument('--fix_angle', action='store_true', default=False)
args = parser.parse_args()


model_file = f'assets/mujoco_models/{args.model_file}.xml'
model = mj.MjModel.from_xml_path(model_file)
data = mj.MjData(model) 
body_qposaddr = get_body_qposaddr(model)
amc_file = f'data/amc/{args.amc_id}.amc'
cyclic = False
cycle_offset = 0.0
offset_z = 0.0




def convert_amc_file():

    def get_qpos(pose):
        qpos = np.zeros_like(data.qpos)
        for bone_name, ind2 in body_qposaddr.items():
            ind1 = bone_addr[bone_name]
            if bone_name == 'root':
                trans = pose[ind1[0]:ind1[0] + 3].copy()
                trans[1], trans[2] = -trans[2], trans[1]
                angles = pose[ind1[0] + 3:ind1[1]].copy()
                quat = quaternion_from_euler(angles[0], angles[1], angles[2])
                quat[2], quat[3] = -quat[3], quat[2]
                qpos[ind2[0]:ind2[0] + 3] = trans
                qpos[ind2[0] + 3:ind2[1]] = quat
            else:
                qpos[ind2[0]:ind2[1]] = pose[ind1[0]:ind1[1]]
        return qpos

    scale = 1 / args.scale * 0.0254
    poses, bone_addr = load_amc_file(amc_file, scale)
    if args.fix_feet:
        poses[:, bone_addr['lfoot'][0] + 2] = poses[:, bone_addr['lfoot'][0] + 2].clip(np.deg2rad(-10.0), np.deg2rad(10.0))
        poses[:, bone_addr['rfoot'][0] + 2] = poses[:, bone_addr['rfoot'][0] + 2].clip(np.deg2rad(-10.0), np.deg2rad(10.0))
    poses_samp = interpolated_traj(poses, args.dt, mocap_fr=args.mocap_fr)
    expert_traj = []
    for cur_pose in poses_samp:
        cur_qpos = get_qpos(cur_pose)
        expert_traj.append(cur_qpos)
    expert_traj = np.vstack(expert_traj)
    expert_traj[:, 2] += args.offset_z
    if args.fix_angle:
        expert_angles = expert_traj[:, 7:]
        while np.any(expert_angles > np.pi):
            expert_angles[expert_angles > np.pi] -= 2 * np.pi
        while np.any(expert_angles < -np.pi):
            expert_angles[expert_angles < -np.pi] += 2 * np.pi
    return expert_traj


def visualize():
    global g_offset, select_start, select_end
    viewer = MyViewer(model, data)
    viewer.cam.azimuth = 45
    viewer.cam.elevation = -8.0
    viewer.cam.distance = 5.0
    viewer.cam.lookat[2] = 1.0
    t = 0
    fr = 0
   
    
    
    #this needs a lot of improvment
    
    while not glfw.window_should_close(viewer.window):
       

    
        # if t >= math.floor(T):
        #     fr = (fr+1) % expert_traj.shape[0]
        #     t = 0
        fr = (fr+1) % expert_traj.shape[0]
        data.qpos[:] = expert_traj[fr]
        data.qpos[2] += offset_z
        
        mj.mj_forward(model,data)
            
        viewer.cam.lookat[:2] = data.qpos[:2]
        viewer.render_frame()
        t += 1
        glfw.swap_buffers(viewer.window)
        glfw.poll_events()
    glfw.terminate()
    
    select_start = g_offset + select_start
    select_end = g_offset + select_end
    return select_start, select_end
 

expert_traj = convert_amc_file()
print("trajectory shape: ",expert_traj.shape)
select_start = 0
select_end = expert_traj.shape[0]
g_offset = 0

if args.render:
    visualize()
print('expert traj shape:', expert_traj.shape)

meta = {'dt': args.dt, 'mocap_fr': args.mocap_fr, 'scale': args.scale, 'offset_z': args.offset_z,
        'cyclic': cyclic, 'cycle_offset': cycle_offset,
        'select_start': select_start, 'select_end': select_end,
        'fix_feet': args.fix_feet, 'fix_angle': args.fix_angle}


print(meta)

"""save the expert trajectory"""
expert_traj_file = f'data/motion/{args.out_id}.p'
os.makedirs(os.path.dirname(expert_traj_file), exist_ok=True)
pickle.dump((expert_traj, meta), open(expert_traj_file, 'wb'))