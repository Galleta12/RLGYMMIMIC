
#slightly modified code from insactor https://github.com/jiawei-ren/insactor
#code similar to rfc code  https://github.com/Khrylx/RFC
import numpy as np
import math
_EPS = np.finfo(float).eps * 4.0


def quat_imaginary(x):
    """
    imaginary components of the quaternion
    """
    return np.array(x[1:4], dtype=np.float64)

# Passive rotation
def quat_rotate(rot, vec):
    """
    Rotate a 3D vector with the 3D rotation
    """
    other_q = np.concatenate([np.array([0.0], dtype=np.float64), vec])
    rotated_vec_q = quat_mul(quat_mul(rot, other_q), quat_conjugate(rot))
      
    # Extract the imaginary part, which represents the rotated vector
    return quat_imaginary(rotated_vec_q)

def quat_mul(a, b):
    """
    Quaternion multiplication with the real part first.
    
    Parameters:
    a, b (np.array): Quaternions to be multiplied, with format [w, x, y, z].
    
    Returns:
    np.array: Result of quaternion multiplication, with format [w, x, y, z].
    """
    w1, x1, y1, z1 = a[0], a[1], a[2], a[3]
    w2, x2, y2, z2 = b[0], b[1], b[2], b[3]
    
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2

    return np.array([w, x, y, z], dtype=np.float64)

def quat_pos(x):
    """
    Ensure the real part (w component) of the quaternion is positive.
    
    Parameters:
    x (np.array): Input quaternion in the format [w, x, y, z].
    
    Returns:
    np.array: Quaternion with positive real part.
    """
    q = np.array(x, dtype=np.float64)  # Copy input to avoid modifying the original
    if q[0] < 0:  # Check if the real part (w) is negative
        q = -q  # Flip the signs of quaternions with negative real parts
    return q

def quat_abs(x):
    """
    Quaternion norm (unit quaternion represents a 3D rotation, which has norm of 1)
    """
    return np.linalg.norm(x)

def quat_unit(x):
    """
    Normalized quaternion with norm of 1
    """
    norm = quat_abs(x)
    if norm < _EPS:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return np.array(x / norm, dtype=np.float64)

def quat_conjugate(x):
    """
    Quaternion with its imaginary part negated
    """
    return np.array([x[0], -x[1], -x[2], -x[3]], dtype=np.float64)

def quat_normalize(q):
    """
    Construct 3D rotation from quaternion (the quaternion needs not to be normalized).
    """
    return quat_unit(quat_pos(q))

def quat_inverse(x):
    """
    The inverse of the rotation
    """
    return quat_conjugate(x)

def quat_mul_norm(x, y):
    """
    Combine two sets of 3D rotations together using the quaternion multiplication operator.
    """
    return quat_normalize(quat_mul(x, y))

def quat_angle_axis(x):
    if 1 - abs(x[0]) < 1e-8:
        axis = np.array([1.0, 0.0, 0.0])
        angle = 0.0
    else:
        s = math.sqrt(1 - x[0]*x[0])
        axis = x[1:4] / s
        angle = 2 * math.acos(x[0])
    return angle, axis

def quat_identity():
    """
    Construct 3D identity rotation.
    """
    return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

def quat_diff_theta(q0, q1):
    """
    Different in rotation angle between quaternions.
    """
    return quat_angle_axis(quat_mul_norm(q0, quat_inverse(q1)))[0]

def quat_inverse_no_norm(q):
    """
    The inverse of the quaternion without normalizing.
    """
    q_conj = quat_conjugate(q)
    q_norm_sq = np.sum(q * q)
    return np.array(q_conj / q_norm_sq, dtype=np.float64)




#code gotted from the transformation.py file check it for more
#information
def quaternion_matrix(quaternion):
    """Return homogeneous rotation matrix from quaternion.
    """
    q = np.array(quaternion, dtype=np.float64, copy=True)
    n = np.dot(q, q)
    if n < _EPS:
        return np.identity(4, dtype=np.float64)
    q *= np.sqrt(2.0 / n)
    q = np.outer(q, q)
    return np.array([
        [1.0-q[2, 2]-q[3, 3],     q[1, 2]-q[3, 0],     q[1, 3]+q[2, 0], 0.0],
        [    q[1, 2]+q[3, 0], 1.0-q[1, 1]-q[3, 3],     q[2, 3]-q[1, 0], 0.0],
        [    q[1, 3]-q[2, 0],     q[2, 3]+q[1, 0], 1.0-q[1, 1]-q[2, 2], 0.0],
        [                0.0,                 0.0,                 0.0, 1.0]], dtype=np.float64)


# def quat_imaginary(x):
#     """
#     imaginary components of the quaternion
#     """
#     return x[..., 1:4]


# #passive rotation
# def quat_rotate(rot, vec):
#     """
#     Rotate a 3D vector with the 3D rotation
#     """
#     other_q = np.concatenate([np.zeros_like(vec[..., :1]), vec], axis=-1)
#     rotated_vec_q = quat_mul(quat_mul(rot, other_q), quat_conjugate(rot))
      
#     # Extract the imaginary part, which represents the rotated vector
#     return quat_imaginary(rotated_vec_q)

# def quat_mul(a, b):
#     """
#     Quaternion multiplication with the real part first.
    
#     Parameters:
#     a, b (np.array): Quaternions to be multiplied, with format [w, x, y, z].
    
#     Returns:
#     np.array: Result of quaternion multiplication, with format [w, x, y, z].
#     """
    
#     w1, x1, y1, z1 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
#     w2, x2, y2, z2 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    
    
#     w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
#     x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
#     y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
#     z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2

#     return np.stack([x, y, z, w], axis=-1)

# def quat_pos(x):
#     """
#     Ensure the real part (w component) of the quaternion is positive.
    
#     Parameters:
#     x (np.array): Input quaternion(s) in the format [x, y, z, w].
    
#     Returns:
#     np.array: Quaternion(s) with positive real part.
#     """
#     q = np.copy(x)  # Copy input to avoid modifying the original
#     negative_real_part = (q[..., -1] < 0)  # Check if the real part (w) is negative
#     q[negative_real_part] = -q[negative_real_part]  # Flip the signs of quaternions with negative real parts
#     return q


# def quat_abs(x):
#     """
#     quaternion norm (unit quaternion represents a 3D rotation, which has norm of 1)
#     """
#     #x = jp.safe_norm(x, axis=-1)
    
#     x = np.linalg.norm(x, axis=-1)  # Use linalg.norm instead of safe_norm
    
#     return x

# def quat_unit(x):
#     """
#     normalized quaternion with norm of 1
#     """
    
#     norm = quat_abs(x)[..., None]
#     norm = np.clip(norm, a_min=1e-9, a_max=1e9)
    
    
#     return x / norm

# def quat_conjugate(x):
#     """
#     quaternion with its imaginary part negated
#     """
#     return np.concatenate([x[..., :1], -x[..., 1:]], axis=-1)

# def quat_normalize(q):
#     """
#     Construct 3D rotation from quaternion (the quaternion needs not to be normalized).
#     """
#     q = quat_unit(quat_pos(q))  # normalized to positive and unit quaternion
#     return q

# def quat_inverse(x):
#     """
#     The inverse of the rotation
#     """
#     return quat_conjugate(x)

# def quat_mul_norm(x, y):
#     """
#     Combine two set of 3D rotations together using \**\* operator. The shape needs to be
#     broadcastable
#     """
#     return quat_normalize(quat_mul(x, y))

# def quat_angle_axis(x):
#     """
#     The (angle, axis) representation of the rotation. The axis is normalized to unit length.
#     The angle is guaranteed to be between [0, pi].??
#     """
#     # Extract the real part (w component)
#     w = x[..., 0]
    
#     # Compute the angle, ensuring the values are clipped between -1 and 1
#     angle = 2 * np.arccos(np.clip(w, -1.0, 1.0))
    
#     # Extract the axis components (x, y, z)
#     axis = x[..., 1:]
    
#     # Normalize the axis
#     norm = np.linalg.norm(axis, axis=-1, keepdims=True)
#     axis = axis / np.clip(norm, 1e-9, 1e9)
    
#     return angle, axis


# def quat_identity(shape):
#     """
#     Construct 3D identity rotation given shape
#     """
#     w = np.ones(shape + (1,))  # Real part
#     xyz = np.zeros(shape + (3,))  # Imaginary parts
#     q = np.concatenate([w, xyz], axis=-1)  # Concatenate along the last axis
#     return quat_normalize(q)


# def quat_identity_like(x):
#     """
#     Construct identity 3D rotation with the same shape
#     """
#     return quat_identity(list(x.shape[:-1]))


# #different in rotation angle between quat
# def quat_diff_theta(q0, q1):
#     return quat_angle_axis(quat_mul_norm(q0, quat_inverse(q1)))[0]


# def quat_inverse_no_norm(q):
#     """
#     The inverse of the quaternion without normalizing.
#     """
#     q_conj = quat_conjugate(q)
#     q_norm_sq = np.sum(q * q, axis=-1, keepdims=True)
#     return q_conj / q_norm_sq