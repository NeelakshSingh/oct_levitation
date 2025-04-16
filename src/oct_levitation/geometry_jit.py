import numpy as np
import numba
import scipy.spatial.transform as scitf

from scipy.linalg import block_diag
from functools import partial

from geometry_msgs.msg import TransformStamped, Transform

EPSILON_TOLERANCE = 1e-15 # for numerical stability
CLOSE_CHECK_TOLERANCE = 1e-3
IDENTITY_QUATERNION = np.array([0.0, 0.0, 0.0, 1.0]) # Identity quaternion

@numba.njit
def check_if_unit_quaternion(q: np.ndarray):
    """
    Check if the quaternion is a unit quaternion.
    """
    return (np.abs(np.linalg.norm(q) - 1.0) <= CLOSE_CHECK_TOLERANCE)

@numba.njit
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

@numba.njit
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

get_skew_symmetric_matrix(np.zeros(3)) # Force compilation on import

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
    return np.concatenate((v, [1]))

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

#############################################
# Quaternion Related Functions
#############################################

@numba.njit
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

rotation_matrix_from_quaternion(IDENTITY_QUATERNION) # Force compilation on import

def invert_quaternion(q: np.ndarray) -> np.ndarray:
    """
    Computes and returns the complex conjugate and therefore the inverse quaterion
    representing the inverse rotation.
    """
    return np.concatenate([-q[:3], [q[3]]])

@numba.njit
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

get_normal_vector_from_quaternion(IDENTITY_QUATERNION) # Force compilation on import

@numba.njit
def get_final_rotation_matrix_from_sequence_of_quaternions(*quaternions: np.ndarray) -> np.ndarray:
    R = np.eye(3)
    for quaternion in quaternions:
        R = R @ rotation_matrix_from_quaternion(quaternion)
    return R

get_final_rotation_matrix_from_sequence_of_quaternions(IDENTITY_QUATERNION, IDENTITY_QUATERNION, IDENTITY_QUATERNION) # Force compilation on import

@numba.njit
def get_normal_alpha_beta_from_quaternion(q: np.ndarray) -> np.ndarray:
    """
    Parameters
    ----------
        q: Quaternion in the form [x, y, z, w]

    Returns
    -------
        v: 2 element array containing the angles of the local frame normal
           with the world XZ and YZ planes. First element is the angle alpha
           with the YZ plane and the second element is the angle beta with the
           new XZ plane obtained after the first YZ rotation. This seems to be
           similar to YXZ intrinsic euler rotations.
    """
    n = get_normal_vector_from_quaternion(q)
    alpha = np.arctan2(n[0], n[2])
    beta = np.arcsin(-n[1])
    return np.array([alpha, beta])

get_normal_alpha_beta_from_quaternion(IDENTITY_QUATERNION) # Force compilation on import

@numba.njit
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

transformation_matrix_from_quaternion(IDENTITY_QUATERNION, np.zeros(3)) # Force compilation on import

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

@numba.njit
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

    # return np.block([[q[3], -q[:3].reshape(1, 3)],
                    #  [q[:3].reshape(3, 1), q[3]*np.eye(3) + get_skew_symmetric_matrix(q[:3])]])

get_left_quaternion_matrix(IDENTITY_QUATERNION) # Force compilation on import

@numba.njit
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
    # return np.block([[q[3], -q[:3].reshape(1, 3)],
    #                  [q[:3].reshape(3, 1), q[3]*np.eye(3) - get_skew_symmetric_matrix(q[:3])]])

get_right_quaternion_matrix(IDENTITY_QUATERNION) # Force compilation on import

@numba.njit
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

rotate_vector_from_quaternion(IDENTITY_QUATERNION, np.array([0.0, 0.0, 1.0])) # Force compilation on import

def quaternion_from_rotation_matrix(R: np.ndarray) -> np.ndarray:
    """
    Parameters
    ----------
        R: 3x3 rotation matrix
    
    Returns
    -------
        q: Quaternion in the form [x, y, z, w]
    """
    scipy_rotation = scitf.Rotation.from_matrix(R)
    q = scipy_rotation.as_quat() # Always returned in scalar last from scipy 1.10
    return q

#############################################
# Euler Angle Related Functions
#############################################

@numba.njit
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

def local_angular_velocities_from_euler_xyz_rate(euler: np.ndarray, euler_rate: np.ndarray) -> np.ndarray:
    return euler_xyz_rate_to_local_angular_velocity_map_matrix(euler) @ euler_rate

def euler_xyz_from_rotation_matrix(R: np.ndarray) -> np.ndarray:
    """
    Parameters
    ----------
        R: 3x3 rotation matrix from the oriented frame to the reference frame.

    Returns
    -------
        euler: 3x1 array in the form [roll, pitch, yaw]
    """
    scipy_rotation = scitf.Rotation.from_matrix(R)
    euler = scipy_rotation.as_euler('XYZ') # caps for intrinsic
    return euler

def euler_xyz_from_quaternion(q: np.ndarray) -> np.ndarray:
    """
    Wrapper for scipy's Rotation class to convert a quaternion to euler angles.
    If you are wondering why I made this instead of directly using scipy's methods,
    well it is because it is super easy to mix the intrinsic and extrinsic notations
    and get the wrong results with scipy.
    
    Parameters
    ----------
        q: Quaternion in the form [x, y, z, w]

    Returns
    -------
        euler: 3x1 array in the form [roll, pitch, yaw] (in rad)
    """
    scipy_rotation = scitf.Rotation.from_quat(q)
    euler = scipy_rotation.as_euler('XYZ') # caps for intrinsic
    return euler

def euler_zyx_from_quaternion(q: np.ndarray) -> np.ndarray:
    """
    Wrapper for scipy's Rotation class to convert a quaternion to euler angles.
    If you are wondering why I made this instead of directly using scipy's methods,
    well it is because it is super easy to mix the intrinsic and extrinsic notations
    and get the wrong results with scipy.
    
    Parameters
    ----------
        q: Quaternion in the form [x, y, z, w]

    Returns
    -------
        euler: 3x1 array in the form [roll, pitch, yaw] (in rad)
    """
    scipy_rotation = scitf.Rotation.from_quat(q)
    euler = scipy_rotation.as_euler('ZYX') # caps for intrinsic
    # Change euler to r, p, y form
    euler = np.array([euler[2], euler[1], euler[0]])
    return euler

def quaternion_from_euler_xyz(euler: np.ndarray) -> np.ndarray:
    """
    Wrapper for scipy's Rotation class for converting XYZ tait-bryan intrinsic angles
    to quaternion.

    Parameters:
        euler (np.ndarray): The euler angles in the form [roll, pitch, yaw]
    
    Returns:
        q: Quaternion in the form [x, y, z, w]
    """
    scipy_rotation = scitf.Rotation.from_euler('XYZ', euler)
    q = scipy_rotation.as_quat()
    return q

def quaternion_from_euler_zyx(euler: np.ndarray) -> np.ndarray:
    """
    Wrapper for scipy's Rotation class for converting XYZ tait-bryan intrinsic angles
    to quaternion.

    Parameters:
        euler (np.ndarray): The euler angles in the form [roll, pitch, yaw]
    
    Returns:
        q: Quaternion in the form [x, y, z, w]
    """
    # Change euler to r, p, y form
    euler = np.array([euler[2], euler[1], euler[0]])
    scipy_rotation = scitf.Rotation.from_euler('ZYX', euler)
    q = scipy_rotation.as_quat()
    return q

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

#############################################
# Reduced Attitude Representation Realted
# Functions
#############################################

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

def angular_velocity_ref_frame_from_rotation_matrix(R: np.ndarray, R_dot: np.ndarray) -> np.ndarray:
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

def angular_velocity_ref_frame_from_quaternion(q: np.ndarray, q_dot: np.ndarray) -> np.ndarray:
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

def inertial_reduced_attitude_from_quaternion(q: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    This function computes the inertial reduced attitude representation from the current pose
    quaternion for a body fixed vector b expressed in the body frame.
    """
    R = rotation_matrix_from_quaternion(q)
    b = b/np.linalg.norm(b, 2) # Reduced attitude should always be a unit vector representing a direction
    Lambda = R @ b
    return Lambda

z_axis_inertial_attitude_from_quaternion = partial(inertial_reduced_attitude_from_quaternion, b=np.array([0, 0, 1]))

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

#############################################
# Magnetic Interaction Matrix Calculations
#############################################

@numba.njit
def magnetic_interaction_grad5_to_force(dipole_moment: np.ndarray) -> np.ndarray:
    M_F = np.array([
                [ dipole_moment[0],  dipole_moment[1], dipole_moment[2], 0.0,              0.0 ],
                [ 0.0,              dipole_moment[0],  0.0,              dipole_moment[1], dipole_moment[2]],
                [-dipole_moment[2],  0.0,              dipole_moment[0], -dipole_moment[2], dipole_moment[1]]
            ])
    return M_F

magnetic_interaction_grad5_to_force(np.array([1.0, 0.0, 0.0])) # Force compilation on import

@numba.njit
def magnetic_interaction_field_to_torque(dipole_moment: np.ndarray) -> np.ndarray:
    return get_skew_symmetric_matrix(dipole_moment)

magnetic_interaction_field_to_torque(np.array([1.0, 0.0, 0.0])) # Force compilation on import

def magnetic_interaction_field_to_local_torque(dipole_strength: float,
                                               dipole_axis: np.ndarray,
                                               dipole_quaternion: np.ndarray) -> np.ndarray:
    return dipole_strength * get_skew_symmetric_matrix(dipole_axis) @ (rotation_matrix_from_quaternion(dipole_quaternion).T)

def magnetic_interaction_matrix_from_dipole_moment(dipole_moment: np.ndarray,
                                                   full_mat: float = True,
                                                   torque_first: bool = True) -> np.ndarray:
    """
    This function returns the magnetic interaction matrix of a dipole.
    This is purely defined by the orientation of the dipole and its strength.
    Args:
        dipole_moment (float): The dipole moment vector of the dipole.
        full_mat (float): Whether to return the full magnetic interaction matrix. 
                          If False, it returns the tuple (M_F, M_Tau) for the force and
                          torque magnetization matrices respectively. Defaults to False.
        torque_first (bool): Whether to return the torque block first or the force block first\
                             when full_mat is set to True.
                             If True, then [[M_Tau], [M_F]] is returned and vice versa.
    
    Returns:
        np.ndarray: The magnetic interaction matrix of the dipole
    """
    M_F = magnetic_interaction_grad5_to_force(dipole_moment)
    M_Tau = magnetic_interaction_field_to_torque(dipole_moment)
    if full_mat:
        if torque_first:
            return block_diag(M_Tau, M_F)
        else:
            return block_diag(M_F, M_Tau)
    else:
        return M_F, M_Tau

def magnetic_interaction_matrix_from_quaternion(dipole_quaternion: np.ndarray,
                                    dipole_strength:float,
                                    full_mat: float = True,
                                    torque_first: bool = True,
                                    torque_in_local_frame: bool = False,
                                    dipole_axis: np.ndarray = np.array([0, 0, 1])) -> np.ndarray:
    """
    This function returns the magnetic interaction matrix of a dipole.
    This is purely defined by the orientation of the dipole and its strength.

    Args:
        dipole_quaternion (np.ndarray): Quaternion of the form [qx, qy, qz, qw].
        dipole_strength (float): The strength of the dipole.
        full_mat (float): Whether to return the full magnetic interaction matrix. 
                          If False, it returns the tuple (M_F, M_Tau) for the force and
                          torque magnetization matrices respectively. Defaults to False.
        torque_first (bool): Whether to return the torque block first or the force block first\
                             when full_mat is set to True.
                             If True, then [[M_Tau], [M_F]] is returned and vice versa.
        dipole_axis (np.ndarray): The axis of the dipole according to vicon in home position. 
                                  Defaults to [0, 0, 1].
    
    Returns:
        np.ndarray: The magnetic interaction matrix of the dipole
    """
    dipole_axis = dipole_axis/np.linalg.norm(dipole_axis, 2)
    R_OH = rotation_matrix_from_quaternion(dipole_quaternion)
    dipole_axis = R_OH.dot(dipole_axis)
    dipole_axis = dipole_axis/np.linalg.norm(dipole_axis, 2)
    dipole_moment = dipole_strength*dipole_axis

    M_F = magnetic_interaction_grad5_to_force(dipole_moment)

    if torque_in_local_frame:
        M_Tau = magnetic_interaction_field_to_local_torque(dipole_strength=dipole_strength,
                                                           dipole_axis=dipole_axis,
                                                           dipole_quaternion=dipole_quaternion)
    else:
        M_Tau = magnetic_interaction_field_to_torque(dipole_moment)
    if full_mat:
        if torque_first:
            return block_diag(M_Tau, M_F)
        else:
            return block_diag(M_F, M_Tau)
    else:
        return M_F, M_Tau

def get_magnetic_interaction_matrix(dipole_tf: TransformStamped,
                                    dipole_strength:float,
                                    full_mat: float = True,
                                    torque_first: bool = True,
                                    dipole_axis: np.ndarray = np.array([0, 0, 1])):
    """
    This function returns the magnetic interaction matrix of a dipole.
    This is purely defined by the orientation of the dipole and its strength.

    Args:
        dipole_tf (TransformStamped): The transform of the dipole in the world frame.
        dipole_strength (float): The strength of the dipole.
        full_mat (float): Whether to return the full magnetic interaction matrix. 
                          If False, it returns the tuple (M_F, M_Tau) for the force and
                          torque magnetization matrices respectively. Defaults to False.
        torque_first (bool): Whether to return the torque block first or the force block first\
                             when full_mat is set to True.
                             If True, then [[M_Tau], [M_F]] is returned and vice versa.
        dipole_axis (np.ndarray): The axis of the dipole according to vicon in home position. 
                                  Defaults to [0, 0, 1].
    
    Returns:
        np.ndarray: The magnetic interaction matrix of the dipole
    """
    dipole_axis = dipole_axis/np.linalg.norm(dipole_axis, 2)
    dipole_quaternion = np.array([dipole_tf.transform.rotation.x,
                                  dipole_tf.transform.rotation.y,
                                  dipole_tf.transform.rotation.z,
                                  dipole_tf.transform.rotation.w])
    return magnetic_interaction_matrix_from_quaternion(dipole_quaternion,
                                                       dipole_strength=dipole_strength,
                                                       full_mat=full_mat,
                                                       torque_first=torque_first,
                                                       dipole_axis=dipole_axis)

@numba.njit
def get_full_magnetic_interaction_torque_first_jit(dipole_axis: np.ndarray,
                                                   dipole_quaternion: np.ndarray,
                                                   dipole_strength: np.float64) -> np.ndarray:
    """
    This function is a faster jit version to get the magnetic interaction matrix for the world frame forces and torques.
    """
    dipole_axis = dipole_axis/np.linalg.norm(dipole_axis, 2)
    dipole_moment = dipole_strength * rotate_vector_from_quaternion(dipole_quaternion, dipole_axis)
    M_Tau = magnetic_interaction_field_to_torque(dipole_moment)

    M_F = magnetic_interaction_grad5_to_force(dipole_moment)
    M = np.zeros((6, 8))

    M[:3, :3] = M_Tau
    M[3:, 3:] = M_F

    return M

get_full_magnetic_interaction_torque_first_jit(np.array([0.0, 0.0, 1.0]),
                                                np.array([0.0, 0.0, 0.0, 1.0]),
                                                1.0) # Force compilation on import