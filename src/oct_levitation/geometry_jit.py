import numpy as np
import numba
import scipy.spatial.transform as scitf

from scipy.linalg import block_diag
from functools import partial

from geometry_msgs.msg import TransformStamped, Transform

EPSILON_TOLERANCE = 1e-15 # for numerical stability
CLOSE_CHECK_TOLERANCE = 1e-3
IDENTITY_QUATERNION = np.array([0.0, 0.0, 0.0, 1.0]) # Identity quaternion

@numba.njit(cache=True)
def numba_cross(v: np.ndarray, w: np.ndarray) -> np.ndarray:
    return np.cross(v, w)

numba_cross(np.zeros(3), np.zeros(3)) # Force compilation for expected argument type signature in import

@numba.njit(cache=True)
def check_if_unit_quaternion(q: np.ndarray):
    """
    Check if the quaternion is a unit quaternion.
    """
    return (np.abs(np.linalg.norm(q) - 1.0) <= CLOSE_CHECK_TOLERANCE)

@numba.njit(cache=True)
def check_if_unit_quaternion_raise_error(q: np.ndarray):
    """
    Check if the quaternion is a unit quaternion.
    """
    if not check_if_unit_quaternion(q):
        raise ValueError(f"Quaternion must be a unit quaternion.") # Constant message in order to support jit

def numpy_quaternion_from_tf_msg(tf_msg: Transform):
    rotation = np.array([
        tf_msg.rotation.x, tf_msg.rotation.y, tf_msg.rotation.z, tf_msg.rotation.w
    ])
    return rotation

def numpy_translation_from_tf_msg(tf_msg: Transform):
    translation = np.array([
        tf_msg.translation.x, tf_msg.translation.y, tf_msg.translation.z
    ])
    return translation

def numpy_arrays_from_tf_msg(tf_msg: Transform):
    """
    Parameters
    ----------
        tf_msg: geometry_msgs/Transform type object.
    
    Returns
    -------
        Tuple(np.ndarray, np.ndarray): The quaternion [x, y, z, w] and the translation [x, y, z] as arrays
    """
    translation = numpy_translation_from_tf_msg(tf_msg)
    rotation = numpy_quaternion_from_tf_msg(tf_msg)

    return rotation, translation

@numba.njit(cache=True)
def get_skew_symmetric_matrix(v: np.ndarray) -> np.ndarray:
    """
    Parameters
    ----------
        v (np.ndarray) : 3x1 array
    """
    assert v.size == 3, "Input vector must have 3 elements."
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])

get_skew_symmetric_matrix(np.zeros(3)) # Force compilation for expected argument type signature in import

@numba.njit(cache=True)
def get_homogeneous_vector(v: np.ndarray) -> np.ndarray:
    """
    Parameters
    ----------
        v: 3x1 array
    
    Returns
    -------
        v_h: 4x1 array
    """
    assert v.size == 3, "Input vector must have 3 elements."
    v_new = np.zeros(4)
    v_new[:3] = v
    v_new[3] = 1.0
    return v_new

get_homogeneous_vector(np.array([0.0, 0.0, 0.0])) # Force compilation for expected argument type signature in import

@numba.njit(cache=True)
def get_non_homoegeneous_vector(v_h: np.ndarray) -> np.ndarray:
    """
    Parameters
    ----------
        v_h: 4x1 array
    
    Returns
    -------
        v: 3x1 array
    """
    assert v_h.size == 4, "Input vector must have 4 elements."
    return v_h[:3]

get_non_homoegeneous_vector(np.array([0.0, 0.0, 0.0, 0.0])) # Force compilation for expected argument type signature in import

#############################################
# Quaternion Related Functions
#############################################

@numba.njit(cache=True)
def rotation_matrix_from_quaternion(q: np.ndarray) -> np.ndarray:
    """
    Parameters
    ----------
        q: Quaternion in the form [x, y, z, w]
    
    Returns
    -------
        R: 3x3 rotation matrix correspoding to q.
    """
    check_if_unit_quaternion_raise_error(q)
    q = q/(np.linalg.norm(q) + EPSILON_TOLERANCE)
    qx = get_skew_symmetric_matrix(q[:3])
    R = (2*q[3]**2 - 1)*np.eye(3) + 2*q[3]*qx + 2*np.outer(q[:3], q[:3])
    return R

rotation_matrix_from_quaternion(IDENTITY_QUATERNION) # Force compilation for expected argument type signature in import

@numba.njit(cache=True)
def rotation_matrix_from_euler_xyz(euler: np.ndarray) -> np.ndarray:
    """
    Parameters
    ----------
        euler: 3x1 array of XYZ extrinsic euler angles in the form [roll, pitch, yaw]
    
    Returns
    -------
        R: 3x3 rotation matrix from the oriented frame to the reference frame.
    """
    assert euler.size == 3, "Euler angles must have 3 elements."
    x = euler[0]
    y = euler[1]
    z = euler[2]
    return np.array([
            [  np.cos(y) * np.cos(z),                                     -np.cos(y) * np.sin(z),                                       np.sin(y)                         ],
            [  np.cos(x) * np.sin(z) + np.sin(x) * np.sin(y) * np.cos(z),  np.cos(x) * np.cos(z) - np.sin(x) * np.sin(y) * np.sin(z),  -np.sin(x) * np.cos(y)  ],
            [  np.sin(x) * np.sin(z) - np.cos(x) * np.sin(y) * np.cos(z),  np.sin(x) * np.cos(z) + np.cos(x) * np.sin(y) * np.sin(z),   np.cos(x) * np.cos(y)  ]
        ])

rotation_matrix_from_euler_xyz(np.zeros(3)) # Force compilation for expected argument type signature in import

def invert_quaternion(q: np.ndarray) -> np.ndarray:
    """
    Computes and returns the complex conjugate and therefore the inverse quaterion
    representing the inverse rotation.
    """
    return np.concatenate([-q[:3], [q[3]]])

@numba.njit(cache=True)
def get_normal_vector_from_quaternion(q: np.ndarray) -> np.ndarray:
    """
    Parameters
    ----------
        q: Quaternion in the form [x, y, z, w]

    Returns
    -------
        v: 3x1 normal vector of the local frame expressed in the world frame.
    """
    R = rotation_matrix_from_quaternion(q)
    return R[:, 2]

get_normal_vector_from_quaternion(IDENTITY_QUATERNION) # Force compilation for expected argument type signature in import

@numba.njit(cache=True)
def get_final_rotation_matrix_from_sequence_of_quaternions(*quaternions: np.ndarray) -> np.ndarray:
    R = np.eye(3)
    for quaternion in quaternions:
        R = R @ rotation_matrix_from_quaternion(quaternion)
    return R

get_final_rotation_matrix_from_sequence_of_quaternions(IDENTITY_QUATERNION, IDENTITY_QUATERNION, IDENTITY_QUATERNION) # Force compilation for expected argument type signature in import

@numba.njit(cache=True)
def get_normal_angles_from_quaternion(q: np.ndarray) -> np.ndarray:
    """
    Jasan's implementation used for the pendulum but JIT compiled with better semantics.
    Parameters
    ----------
        q: Quaternion in the form [x, y, z, w]

    Returns
    -------
        v: 2 element array containing the angles of the local frame normal
           with the world XZ and YZ planes respectively. They seem to correspond to
           ZYX extrinsic euler angles for zero yaw.
    """
    n = get_normal_vector_from_quaternion(q)
    angle_y = np.arctan2(n[0], n[2])
    angle_x = np.arcsin(-n[1])
    return np.array([angle_x, angle_y])

get_normal_angles_from_quaternion(IDENTITY_QUATERNION) # Force compilation for expected argument type signature in import

@numba.njit(cache=True)
def transformation_matrix_from_quaternion(q: np.ndarray, p: np.ndarray) -> np.ndarray:
    """
    Parameters
    ----------
        q: 4x1 array of Quaternion in the form [x, y, z, w]
        p: 3x1 array of translation in the form [x, y, z]
    """
    q = q/(np.linalg.norm(q) + EPSILON_TOLERANCE)
    R = rotation_matrix_from_quaternion(q)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p
    return T

transformation_matrix_from_quaternion(IDENTITY_QUATERNION, np.zeros(3)) # Force compilation for expected argument type signature in import

@numba.njit(cache=True)
def transform_vector_from_quaternion(q: np.ndarray, p: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Full translationa and rotation of a vector given a quaternion and the translation
    vector.

    Parameters:

        q: 4x1 array of Quaternion in the form [x, y, z, w]
        p: 3x1 array of translation in the form [x, y, z]
        v: 3x1 array of vector to be transformed
    
    Returns:
        v_tf: 3x1 transformed vector 
    """
    v_homo = get_homogeneous_vector(v)
    v_tf_homo = transformation_matrix_from_quaternion(q, p).dot(v_homo)
    return get_non_homoegeneous_vector(v_tf_homo)

transform_vector_from_quaternion(IDENTITY_QUATERNION, np.zeros(3), np.zeros(3))

def transform_vector_from_transform_stamped(msg: TransformStamped, v: np.ndarray) -> np.ndarray:
    """
    Full translationa and rotation of a vector given the ROS TransformStamped message.

    Parameters:

        msg (TransformStamped): The ROS transform message
        v: 3x1 array of vector to be transformed
    
    Returns:
        v_tf: 3x1 transformed vector 
    """
    q = np.array([msg.transform.rotation.x,
                  msg.transform.rotation.y,
                  msg.transform.rotation.z,
                  msg.transform.rotation.w])
    p = np.array([msg.transform.translation.x,
                  msg.transform.translation.y,
                  msg.transform.translation.z])
    return transform_vector_from_quaternion(q, p, v)

@numba.njit(cache=True)
def invert_transformation_matrix(T: np.ndarray) -> np.ndarray:
    """
    Takes the inverse of the 4x4 homogeneous transformation matrix.

    Parameters
    ----------
        T: 4x4 transformation matrix
    """
    R = T[:3, :3]
    p = T[:3, 3]
    T_inv = np.eye(4)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ p
    return T_inv

invert_transformation_matrix(np.eye(4)) # Force compilation for expected argument type signature in import

def transformation_matrix_from_compose_transforms(*transforms: Transform) -> Transform:
    """
    Applies transforms in the left-> right sequence supplied and gives the final transform from the
    final frame to the initial frame.
    """
    T = np.eye(4)
    for transform in transforms:
        T_tf = transformation_matrix_from_quaternion(
            numpy_quaternion_from_tf_msg(transform),
            numpy_translation_from_tf_msg(transform)
        )
        T = T @ T_tf
    
    return T

@numba.njit(cache=True)
def get_left_quaternion_matrix(q: np.ndarray) -> np.ndarray:
    """
    Parameters
    ----------
        q: Quaternion in the form [x, y, z, w]
    """
    check_if_unit_quaternion_raise_error(q)
    q = q/(np.linalg.norm(q) + EPSILON_TOLERANCE)
    q_left = np.zeros((4, 4))
    q_left[0, 0] = q[3]
    q_left[0, 1] = -q[0]
    q_left[0, 2] = -q[1]
    q_left[0, 3] = -q[2]
    q_left[1, 0] = q[0]
    q_left[2, 0] = q[1]
    q_left[3, 0] = q[2]
    q_left[1, 1] = q[3]
    q_left[1, 2] = -q[2]
    q_left[1, 3] = q[1]
    q_left[2, 1] = q[2]
    q_left[2, 2] = q[3]
    q_left[2, 3] = -q[0]
    q_left[3, 1] = -q[1]
    q_left[3, 2] = q[0]
    q_left[3, 3] = q[3]
    return q_left

get_left_quaternion_matrix(IDENTITY_QUATERNION) # Force compilation for expected argument type signature in import

@numba.njit(cache=True)
def get_right_quaternion_matrix(q: np.ndarray) -> np.ndarray:
    """
    Parameters
    ----------
        q: Quaternion in the form [x, y, z, w]
    """
    check_if_unit_quaternion_raise_error(q)
    q = q/(np.linalg.norm(q) + EPSILON_TOLERANCE)
    q_right = np.zeros((4, 4))
    q_right[0, 0] = q[3]
    q_right[0, 1] = -q[0]
    q_right[0, 2] = -q[1]
    q_right[0, 3] = -q[2]
    q_right[1, 0] = q[0]
    q_right[2, 0] = q[1]
    q_right[3, 0] = q[2]

    q_right[1, 1] = q[3]
    q_right[1, 2] = q[2]
    q_right[1, 3] = -q[1]
    q_right[2, 1] = -q[2]
    q_right[2, 2] = q[3]
    q_right[2, 3] = q[0]
    q_right[3, 1] = q[1]
    q_right[3, 2] = -q[0]
    q_right[3, 3] = q[3]
    return q_right

get_right_quaternion_matrix(IDENTITY_QUATERNION) # Force compilation for expected argument type signature in import

@numba.njit(cache=True)
def rotate_vector_from_quaternion(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Parameters
    ----------
        q: Quaternion in the form [x, y, z, w]
        v: 3x1 array
    
    Returns
    -------
        v_rot: 3x1 rotated array
    """
    # Using quaternion algebra to avoid ambiguities.
    check_if_unit_quaternion(q)
    v_mag = np.linalg.norm(v)
    v = v/(v_mag + EPSILON_TOLERANCE) # normalizing
    v_aug = np.vstack((np.array([[0]]), v.reshape(3, 1))) # Tuple of arrays needed for numba
    Ml = get_left_quaternion_matrix(q)
    q_T = np.array([-q[0], -q[1], -q[2], q[3]])
    Mr = get_right_quaternion_matrix(q_T)
    v_rot = (Ml @ Mr @ v_aug).flatten()
    return v_rot[1:]*v_mag

rotate_vector_from_quaternion(IDENTITY_QUATERNION, np.array([0.0, 0.0, 1.0])) # Force compilation for expected argument type signature in import

#############################################
# Euler Angle Related Functions
#############################################

@numba.njit(cache=True)
def rotation_matrix_from_euler_xyz(euler: np.ndarray) -> np.ndarray:
    """
    Parameters
    ----------
        euler: 3x1 array in the form [roll, pitch, yaw]
    
    Returns
    -------
        R: 3x3 rotation matrix from the oriented frame to the reference frame.
    """
    assert euler.size == 3, "Euler angles must have 3 elements."
    x = euler[0]
    y = euler[1]
    z = euler[2]
    return np.array([
            [  np.cos(y) * np.cos(z),                                     -np.cos(y) * np.sin(z),                                       np.sin(y)                         ],
            [  np.cos(x) * np.sin(z) + np.sin(x) * np.sin(y) * np.cos(z),  np.cos(x) * np.cos(z) - np.sin(x) * np.sin(y) * np.sin(z),  -np.sin(x) * np.cos(y)  ],
            [  np.sin(x) * np.sin(z) - np.cos(x) * np.sin(y) * np.cos(z),  np.sin(x) * np.cos(z) + np.cos(x) * np.sin(y) * np.sin(z),   np.cos(x) * np.cos(y)  ]
        ])

# Force compilation
rotation_matrix_from_euler_xyz(np.zeros(3))

@numba.njit(cache=True)
def transformation_matrix_from_euler_xyz(euler: np.ndarray, p: np.ndarray) -> np.ndarray:
    """
    Parameters
    ----------
        euler: 3x1 array in the form [roll, pitch, yaw]
        p: 3x1 array of translation in the form [x, y, z]
    
    Returns
    -------
        T: 4x4 transformation matrix from the oriented frame to the reference frame.
    """
    R = rotation_matrix_from_euler_xyz(euler)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p
    return T

transformation_matrix_from_euler_xyz(np.zeros(3), np.zeros(3)) # Force compilation for expected argument type signature in import

@numba.njit(cache=True)
def local_angular_velocity_to_euler_xyz_rate_map_matrix(euler: np.ndarray) -> np.ndarray:
    """
    Parameters
    ----------
        euler: 3x1 array in the form [roll, pitch, yaw]
    
    Returns
    -------
        E_exyz_inv: 3x3 matrix that maps the body frame angular velocity to the Euler XYZ rates.
    """
    # If you are referring to the robot dynamics lecture notes, the expression there relates the
    # world frame angular velocity, this map here is for the local frame angular velocity to the
    # Euler XYZ rates.
    y = euler[1]
    z = euler[2]
    return np.array([
            [  np.cos(z)/(np.cos(y) + EPSILON_TOLERANCE),  -np.sin(z)/(np.cos(y) + EPSILON_TOLERANCE),  0  ],
            [  np.sin(z),             np.cos(z),            0  ],
            [ -np.cos(z)*np.tan(y),   np.sin(z)*np.tan(y),  1  ]
        ])

local_angular_velocity_to_euler_xyz_rate_map_matrix(np.zeros(3)) # Force compilation for expected argument type signature in import

@numba.njit(cache=True)
def euler_xyz_rate_to_local_angular_velocity_map_matrix(euler: np.ndarray) -> np.ndarray:
    """
    Parameters
    ----------
        euler: 3x1 array in the form [roll, pitch, yaw]
    
    Returns
    -------
        E_exyz: 3x3 matrix that maps the Euler XYZ rates to the body frame angular velocity.
    """
    y = euler[1]
    z = euler[2]
    return np.array([
            [  np.cos(z)*np.cos(y),  np.sin(z),  0  ],
            [ -np.sin(z)*np.cos(y),  np.cos(z),  0  ],
            [  np.sin(y),            0,          1  ]
        ])

@numba.njit(cache=True)
def local_angular_velocities_from_euler_xyz_rate(euler: np.ndarray, euler_rate: np.ndarray) -> np.ndarray:
    return euler_xyz_rate_to_local_angular_velocity_map_matrix(euler) @ euler_rate

local_angular_velocities_from_euler_xyz_rate(np.zeros(3), np.zeros(3)) # Force compilation for expected argument type signature in import

@numba.njit(cache=True)
def quaternion_from_rotation_matrix(R):
    """
    Convert a 3x3 rotation matrix to quaternion representation
    Source: tsc_utils.rotations

    Args:
        R (np.ndarray): rotation matrix (3x3 numpy array)
    Returns:
        np.ndarray: quaternion as 4d numpy array [x,y,z,w]
    """
    w2_a = 1 + R[0, 0] + R[1, 1] + R[2, 2]
    w2_b = 1 + R[0, 0] - R[1, 1] - R[2, 2]
    w2_c = 1 - R[0, 0] + R[1, 1] - R[2, 2]
    w2_d = 1 - R[0, 0] - R[1, 1] + R[2, 2]

    w2_m = max([w2_a, w2_b, w2_c, w2_d])

    if w2_m == w2_a:
        w = 0.5 * np.sqrt(w2_a)
        x = (R[2, 1] - R[1, 2]) / (4 * w)
        y = (R[0, 2] - R[2, 0]) / (4 * w)
        z = (R[1, 0] - R[0, 1]) / (4 * w)
    elif w2_m == w2_b:
        x = 0.5 * np.sqrt(w2_b)
        y = (R[0, 1] + R[1, 0]) / (4 * x)
        z = (R[0, 2] + R[2, 0]) / (4 * x)
        w = (R[2, 1] - R[1, 2]) / (4 * x)
    elif w2_m == w2_c:
        y = 0.5 * np.sqrt(w2_c)
        x = (R[1, 0] + R[0, 1]) / (4 * y)
        z = (R[1, 2] + R[2, 1]) / (4 * y)
        w = (R[0, 2] - R[2, 0]) / (4 * y)
    else:
        z = 0.5 * np.sqrt(w2_d)
        x = (R[2, 0] + R[0, 2]) / (4 * z)
        y = (R[2, 1] + R[1, 2]) / (4 * z)
        w = (R[1, 0] - R[0, 1]) / (4 * z)

    return np.array([x, y, z, w])

quaternion_from_rotation_matrix(np.eye(3)) # Force compilation for expected argument type signature in import

@numba.njit(cache=True)
def euler_xyz_from_rotation_matrix(R: np.ndarray) -> np.ndarray:
    """
    Parameters
    ----------
        R: 3x3 rotation matrix from the oriented frame to the reference frame.

    Returns
    -------
        euler: 3x1 array in the form [roll, pitch, yaw] for XYZ extrinsic euler angles.
    """
    euler = np.zeros(3)
    euler[0] = np.arctan2(-R[1, 2], R[2, 2])
    euler[1] = np.arcsin(R[0, 2])
    euler[2] = np.arctan2(-R[0, 1], R[0, 0])
    return euler

euler_xyz_from_rotation_matrix(np.eye(3)) # Force compilation for expected argument type signature in import

@numba.njit(cache=True)
def euler_xyz_from_quaternion(q: np.ndarray) -> np.ndarray:
    """
    This function converts quaternion to XYZ extrinsic euler angles through an interconversion to the rotation matrix.
    I am sure there are better ways, but this is a straightforward method I chose to stick to (and a lot of other libraries do too).
    
    Parameters
    ----------
        q: Quaternion in the form [x, y, z, w]

    Returns
    -------
        euler: 3x1 array in the form [roll, pitch, yaw] (in rad)
    """
    return euler_xyz_from_rotation_matrix(rotation_matrix_from_quaternion(q))

euler_xyz_from_quaternion(IDENTITY_QUATERNION) # Force compilation for expected argument type signature in import

@numba.njit(cache=True)
def quaternion_from_euler_xyz(euler: np.ndarray) -> np.ndarray:
    """
    This function concerts XYZ extrinsic euler angles to quaternion through an interconversion to the rotation matrix.
    I am sure there are better ways, but this is a straightforward method I chose to stick to (and a lot of other libraries do too).

    Parameters:
        euler (np.ndarray): The euler angles in the form [roll, pitch, yaw]
    
    Returns:
        q: Quaternion in the form [x, y, z, w]
    """
    return quaternion_from_rotation_matrix(rotation_matrix_from_euler_xyz(euler))

quaternion_from_euler_xyz(np.zeros(3)) # Force compilation for expected argument type signature in import

@numba.njit(cache=True)
def angle_residual(a: float, b: float):
    """
    Computes the smaller arc's angle residual between a and b by converting it to the 
    range [-pi, pi].

    Parameters
    ----------
        a (float) : angle in rad
        b (float) : angle in rad
    
    Returns
    -------
        residual (float) : a - b shifted to the range [-pi, pi]
    """
    residual = a - b
    residual = residual % (2*np.pi) # First force to the range [0, 2*pi]
    if residual > np.pi:
        residual -= 2*np.pi
    return residual

angle_residual(0.0, 0.0) # Force compilation for expected argument type signature in import
#############################################
# Reduced Attitude Representation Realted
# Functions
#############################################

@numba.njit(cache=True)
def angular_velocity_body_frame_from_rotation_matrix(R: np.ndarray, R_dot: np.ndarray) -> np.ndarray:
    """
    Parameters
    ----------
        R: 3x3 rotation matrix from the local frame to the reference frame.
        R_dot: 3x3 time derivative of the rotation matrix.
    
    Returns
    -------
        omega: 3x1 angular velocity of local frame w.r.t refrence frame expressed in the local frame.
    """
    omega = np.zeros(3)
    omega_skew = R.T @ R_dot # Should be this for body fixed frame resolved angular velocity.
    omega[0] = (omega_skew[2, 1] - omega_skew[1, 2])/2
    omega[1] = (omega_skew[0, 2] - omega_skew[2, 0])/2
    omega[2] = (omega_skew[1, 0] - omega_skew[0, 1])/2
    return omega

angular_velocity_body_frame_from_rotation_matrix(np.eye(3), np.zeros((3, 3))) # Force compilation for expected argument type signature at import

@numba.njit(cache=True)
def angular_velocity_world_frame_from_rotation_matrix(R: np.ndarray, R_dot: np.ndarray) -> np.ndarray:
    """
    Parameters
    ----------
        R: 3x3 rotation matrix from the local frame to the reference frame.
        R_dot: 3x3 time derivative of the rotation matrix.
    
    Returns
    -------
        omega: 3x1 angular velocity of local frame w.r.t refrence frame expressed in the reference frame.
    """
    omega = np.zeros(3)
    omega_skew = R_dot @ R.T # Should be this for reference frame resolved angular velocity.
    omega[0] = (omega_skew[2, 1] - omega_skew[1, 2])/2
    omega[1] = (omega_skew[0, 2] - omega_skew[2, 0])/2
    omega[2] = (omega_skew[1, 0] - omega_skew[0, 1])/2
    return omega

angular_velocity_world_frame_from_rotation_matrix(np.eye(3), np.zeros((3, 3))) # Force compilation for expected argument type signature in import

@numba.njit(cache=True)
def angular_velocity_world_frame_from_quaternion(q: np.ndarray, q_dot: np.ndarray) -> np.ndarray:
    """
    Parameters
    ----------
        q: Quaternion in the form [x, y, z, w]
        q_dot: Time derivative of the quaternion, can be calculated however you want, first order approximation, etc.
    
    Returns
    -------
        omega: 3 element angular velocity of local frame w.r.t refrence frame expressed in the reference frame.
    """
    left_component_matrix = np.array([
        [ q[3], -q[2], q[1]],
        [ q[2], q[3], -q[0]],
        [ -q[1], q[0], q[3]],
        [ -q[0], -q[1], -q[2]]
    ])

    omega = 2*left_component_matrix.T @ q_dot
    return omega.flatten()

angular_velocity_world_frame_from_quaternion(IDENTITY_QUATERNION, np.array([0.0, 0.0, 0.0, 0.0])) # Force compilation for expected argument type signature in import

@numba.njit(cache=True)
def inertial_reduced_attitude_from_quaternion(q: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    This function computes the inertial reduced attitude representation from the current pose
    quaternion for a body fixed vector b expressed in the body frame.
    """
    R = rotation_matrix_from_quaternion(q)
    b = b/np.linalg.norm(b, 2) # Reduced attitude should always be a unit vector representing a direction
    Lambda = R @ b
    return Lambda

inertial_reduced_attitude_from_quaternion(IDENTITY_QUATERNION, np.array([0.0, 0.0, 1.0])) # Force compilation for expected argument type signature in import

z_axis_inertial_attitude_from_quaternion = partial(inertial_reduced_attitude_from_quaternion, b=np.array([0.0, 0.0, 1.0]))

@numba.njit(cache=True)
def inertial_reduced_attitude_from_rotation_matrix(R: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    This function computes the inertial reduced attitude representation from the current pose
    rotation matrix for a body fixed vector b expressed in the body frame.

    Parameters
    ----------
        R: 3x3 rotation matrix from the local frame to the inertial reference frame.
        b: 3x1 array representing the body fixed vector to represent the reduced attitude.

    Returns
    -------
        Lambda: 3x1 array representing the inertial reduced attitude.
    """
    b = b/np.linalg.norm(b, 2)
    Lambda = R @ b
    return Lambda

inertial_reduced_attitude_from_rotation_matrix(np.eye(3), np.array([0.0, 0.0, 1.0]) * 0.45) # Force compilation for expected argument type signature in import

#############################################
# Magnetic Interaction Matrix Calculations
#############################################

@numba.njit(cache=True)
def magnetic_interaction_grad5_to_force(dipole_moment: np.ndarray) -> np.ndarray:
    M_F = np.array([
                [ dipole_moment[0],  dipole_moment[1], dipole_moment[2], 0.0,              0.0 ],
                [ 0.0,              dipole_moment[0],  0.0,              dipole_moment[1], dipole_moment[2]],
                [-dipole_moment[2],  0.0,              dipole_moment[0], -dipole_moment[2], dipole_moment[1]]
            ])
    return M_F

magnetic_interaction_grad5_to_force(np.array([1.0, 0.0, 0.0])) # Force compilation for expected argument type signature in import

@numba.njit(cache=True)
def magnetic_interaction_field_to_local_torque_from_rotmat(local_dipole_moment: np.ndarray,
                                                     R: np.ndarray) -> np.ndarray:
    """
    The dipole moment must be expressed in the local frame for this to work.
    """
    return get_skew_symmetric_matrix(local_dipole_moment) @ R.T

magnetic_interaction_field_to_local_torque_from_rotmat(np.array([1.0, 0.0, 0.0]),
                                                    np.eye(3)) # Force compilation for expected argument type signature in import

@numba.njit(cache=True)
def magnetic_interaction_field_to_local_torque(dipole_strength: float,
                                               local_dipole_axis: np.ndarray,
                                               dipole_quaternion: np.ndarray) -> np.ndarray:
    return dipole_strength * get_skew_symmetric_matrix(local_dipole_axis) @ (rotation_matrix_from_quaternion(dipole_quaternion).T)


magnetic_interaction_field_to_local_torque(1.0,
                                           np.array([0.0, 0.0, 1.0]),
                                           IDENTITY_QUATERNION) # Force compilation for expected argument type signature in import

@numba.njit(cache=True)
def magnetic_interaction_inertial_force_torque(local_dipole_moment: np.ndarray,
                                              quaternion: np.ndarray,
                                              remove_z_torque: bool) -> np.ndarray:
    """
    Args:
        local_dipole_moment (float): The dipole moment vector of the dipole in the local frame. 
        eg: [0.0, 0.0, -1.0] * strength for a north down dipole aligned with local frame z axis.
    
    Returns:
        np.ndarray: The magnetic interaction matrix of the dipole mapping world frame fields and gradients to world frame forces
        and world frame torques.
    """
    R = rotation_matrix_from_quaternion(quaternion)
    dipole_moment = R @ local_dipole_moment
    M_F = magnetic_interaction_grad5_to_force(dipole_moment)
    M_Tau = get_skew_symmetric_matrix(dipole_moment)
    M = np.zeros((6, 8))
    
    M[:3, :3] = M_Tau
    M[3:, 3:] = M_F

    if remove_z_torque:
        M = np.vstack((M[:2], M[3:]))

    return M

magnetic_interaction_inertial_force_torque(np.array([0.0, 0.0, -0.45]), IDENTITY_QUATERNION, True) # Force compilation for expected argument type signature in import

@numba.njit(cache=True)
def magnetic_interaction_force_local_torque(local_dipole_moment: np.ndarray,
                                            quaternion: np.ndarray,
                                            remove_z_torque: bool = True) -> np.ndarray:
    """
    
    Args:
        local_dipole_moment (float): The dipole moment vector of the dipole in the local frame. 
        eg: [0.0, 0.0, -1.0] * strength for a north down dipole aligned with local frame z axis.
    
    Returns:
        np.ndarray: The magnetic interaction matrix of the dipole mapping world frame fields and gradients to world frame forces
        and local frame torques.
    """
    R = rotation_matrix_from_quaternion(quaternion)
    dipole_moment = R @ local_dipole_moment
    M_F = magnetic_interaction_grad5_to_force(dipole_moment)
    M_Tau = get_skew_symmetric_matrix(local_dipole_moment) @ R.T
    M = np.zeros((6, 8))
    
    M[:3, :3] = M_Tau
    M[3:, 3:] = M_F

    if remove_z_torque:
        M = np.vstack((M[:2], M[3:]))
    return M

magnetic_interaction_force_local_torque(np.array([0.0, 0.0, -0.45]), IDENTITY_QUATERNION) # Force compilation for expected argument type signature in import 

@numba.njit(cache=True)
def magnetic_interaction_force_local_torque_from_rotmat(local_dipole_moment: np.ndarray,
                                            R: np.ndarray,
                                            remove_z_torque: bool) -> np.ndarray:
    """
    
    Args:
        local_dipole_moment (float): The dipole moment vector of the dipole in the local frame. 
        eg: [0.0, 0.0, -1.0] * strength for a north down dipole aligned with local frame z axis.
    
    Returns:
        np.ndarray: The magnetic interaction matrix of the dipole mapping world frame fields and gradients to world frame forces
        and local frame torques.
    """
    dipole_moment = R @ local_dipole_moment
    M_F = magnetic_interaction_grad5_to_force(dipole_moment)
    M_Tau = get_skew_symmetric_matrix(local_dipole_moment) @ R.T
    M = np.zeros((6, 8))
    
    M[:3, :3] = M_Tau
    M[3:, 3:] = M_F

    if remove_z_torque:
        M = np.vstack((M[:2], M[3:]))
    return M

magnetic_interaction_force_local_torque_from_rotmat(np.array([0.0, 0.0, -0.45]), np.eye(3), True) # Force compilation for expected argument type signature in import