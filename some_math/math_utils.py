from some_math.quaternion import *
import numpy as np
import math
#code similiar from https://github.com/Khrylx/RFC/tree/main



#get  direction of yaw
#remeber the first element is w of the quat
def get_heading_q(q):
    hq = q.copy()
    hq[1] = 0
    hq[2] = 0
    hq /= np.linalg.norm(hq)
    
    return hq

def get_heading(q):
    hq = q.copy()
    hq[1] = 0
    hq[2] = 0
    #ensure quat is positive
    hp = np.where(hq[3]<0,-1*hp,hp)
    #normalize quaternion
    hq /= np.linalg.norm(hq)
    #restun angle
    return 2 * math.acos(hq[0])


def de_heading(q):
    #normalize quaternon
    
    return quat_mul(quat_inverse_no_norm(get_heading_q(q)), q)

def multi_quat_diff(nq1, nq0):
    """return the relative quaternions q1-q0 of N joints"""

    nq_diff = np.zeros_like(nq0)
    for i in range(nq1.shape[0] // 4):
        ind = slice(4*i, 4*i + 4)
        q1 = nq1[ind]
        q0 = nq0[ind]
        #normalize for the inverse
        #since it only works with te inverse
        
        nq_diff[ind] = quat_mul(q1, quat_inverse_no_norm(q0))
    return nq_diff


def transform_vec(v, q):
    
    trans = quaternion_matrix(q)[:3, :3]
    
    #the transpose since is the inverse
    # we want to transform to the new defined space
    v = trans.T.dot(v[:, None]).ravel()
    return v


#get angular velcity
def get_angvel_fd(prev_bquat, cur_bquat, dt):
    q_diff = multi_quat_diff(cur_bquat, prev_bquat)
    n_joint = q_diff.shape[0] // 4
    body_angvel = np.zeros(n_joint * 3)
    for i in range(n_joint):
        #body_angvel[3*i: 3*i + 3] = rotation_from_quaternion(q_diff[4*i: 4*i + 4]) / dt
        angle,axis =quat_angle_axis(q_diff[4*i: 4*i + 4])
        angular = (axis*angle)/dt
        body_angvel[3*i: 3*i + 3] = angular 
    return body_angvel



def get_qvel_fd_new(cur_qpos, next_qpos, dt):
    #traslational velocity
    v = (next_qpos[:3] - cur_qpos[:3]) / dt
    #next_pos_norm = quat_normalize(next_qpos[3:7])
    #current_qpos_norm = quat_normalize(cur_qpos[3:7])
    #quaternion from current to next post
    qrel = quat_mul(next_qpos[3:7], quat_inverse_no_norm(cur_qpos[3:7]))
    
    angle,axis = quat_angle_axis(qrel)
    #ensure that they are within range
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle < -np.pi:
        angle += 2 * np.pi
    #get the angular velocity
    rv = (axis * angle) / dt
    
    rv = transform_vec(rv, cur_qpos[3:7])   # angular velocity is in root coord
    
    diff = next_qpos[7:] - cur_qpos[7:]
    while np.any(diff > np.pi):
        diff[diff > np.pi] -= 2 * np.pi
    while np.any(diff < -np.pi):
        diff[diff < -np.pi] += 2 * np.pi
    #angular velocities
    qvel = diff / dt
    qvel = np.concatenate((v, rv, qvel))
    
    return qvel

def multi_quat_norm(nq):
    """return the scalar rotation of a N joints"""

    nq_norm = np.arccos(np.clip(abs(nq[::4]), -1.0, 1.0))
    return nq_norm



def quat_mul_vec(q, v):
    old_shape = v.shape
    v = v.reshape(-1, 3)
    v = v.dot(quaternion_matrix(q)[:3, :3].T)
    return v.reshape(old_shape)
