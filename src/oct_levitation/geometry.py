import numpy as np
import scipy.spatial.transform as scitf

EPSILON_TOLERANCE = 1e-12 # for numerical stability

def get_skew_symmetric_matrix(v: np.ndarray) -> np.ndarray:
    """
    Args:
        v: 3x1 array
    """
    assert v.size == 3, "Input vector must have 3 elements."
    v = v.flatten()
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])

def rotation_matrix_from_quaternion(q: np.ndarray) -> np.ndarray:
    """
    Args:
        q: Quaternion in the form [x, y, z, w]
    """
    q = q/(np.linalg.norm(q) + EPSILON_TOLERANCE)
    qx = get_skew_symmetric_matrix(q[:3])
    R = (2*q[3]**2 - 1)*np.eye(3) + 2*q[3]*qx + 2*np.outer(q[:3], q[:3])
    return R

def get_left_quaternion_matrix(q: np.ndarray) -> np.ndarray:
    """
    Args:
        q: Quaternion in the form [x, y, z, w]
    """
    q = q/(np.linalg.norm(q) + EPSILON_TOLERANCE)
    return np.block([[q[3], -q[:3].reshape(1, 3)],
                     [q[:3].reshape(3, 1), q[3]*np.eye(3) + get_skew_symmetric_matrix(q[:3])]])

def get_right_quaternion_matrix(q: np.ndarray) -> np.ndarray:
    """
    Args:
        q: Quaternion in the form [x, y, z, w]
    """
    q = q/(np.linalg.norm(q) + EPSILON_TOLERANCE)
    return np.block([[q[3], -q[:3].reshape(1, 3)],
                     [q[:3].reshape(3, 1), q[3]*np.eye(3) - get_skew_symmetric_matrix(q[:3])]])

def rotate_vector_from_quaternion(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Args:
        q: Quaternion in the form [x, y, z, w]
        v: 3x1 array
    
    Returns:
        v_rot: 3x1 rotated array
    """
    # Using quaternion algebra to avoid ambiguities.
    if not np.isclose(np.linalg.norm(q), 1.0):
        raise ValueError("Quaternion must be normalized.")
    v_mag = np.linalg.norm(v)
    v = v/(v_mag + EPSILON_TOLERANCE) # normalizing
    v_aug = np.vstack((0, v.reshape(3, 1)))
    Ml = get_left_quaternion_matrix(q)
    q_T = np.array([-q[0], -q[1], -q[2], q[3]])
    Mr = get_right_quaternion_matrix(q_T)
    v_rot = (Ml @ Mr @ v_aug).flatten()
    return v_rot[1:]*v_mag

def rotation_matrix_from_euler_xyz(euler: np.ndarray) -> np.ndarray:
    """
    Args:
        euler: 3x1 array in the form [roll, pitch, yaw]
    
    Returns:
        R: 3x3 rotation matrix from the oriented frame to the reference frame.
    """
    x = euler[0]
    y = euler[1]
    z = euler[2]
    return np.array([
            [  np.cos(y) * np.cos(z),                                     -np.cos(y) * np.sin(z),                                       np.sin(y)                         ],
            [  np.cos(x) * np.sin(z) + np.sin(x) * np.sin(y) * np.cos(z),  np.cos(x) * np.cos(z) - np.sin(x) * np.sin(y) * np.sin(z),  -np.sin(x) * np.cos(y)  ],
            [  np.sin(x) * np.sin(z) - np.cos(x) * np.sin(y) * np.cos(z),  np.sin(x) * np.cos(z) + np.cos(x) * np.sin(y) * np.sin(z),   np.cos(x) * np.cos(y)  ]
        ])

def local_angular_velocity_to_euler_xyz_rate_map_matrix(euler: np.ndarray) -> np.ndarray:
    """
    Args:
        euler: 3x1 array in the form [roll, pitch, yaw]
    
    Returns:
        E_exyz_inv: 3x3 matrix that maps the body frame angular velocity to the Euler XYZ rates.
    """
    y = euler[1]
    z = euler[2]
    return np.array([
            [  np.cos(z)/(np.cos(y) + EPSILON_TOLERANCE),  -np.sin(z)/(np.cos(y) + EPSILON_TOLERANCE),  0  ],
            [  np.sin(z),             np.cos(z),            0  ],
            [ -np.cos(z)*np.tan(y),   np.sin(z)*np.tan(y),  1  ]
        ])

def euler_xyz_rate_to_local_angular_velocity_map_matrix(euler: np.ndarray) -> np.ndarray:
    """
    Args:
        euler: 3x1 array in the form [roll, pitch, yaw]
    
    Returns:
        E_exyz: 3x3 matrix that maps the Euler XYZ rates to the body frame angular velocity.
    """
    y = euler[1]
    z = euler[2]
    return np.array([
            [  np.cos(z)*np.cos(y),  np.sin(z),  0  ],
            [ -np.sin(z)*np.cos(y),  np.cos(z),  0  ],
            [  np.sin(y),            0,          1  ]
        ])

def euler_xyz_from_rotation_matrix(R: np.ndarray) -> np.ndarray:
    """
    Args:
        R: 3x3 rotation matrix from the oriented frame to the reference frame.

    Returns:
        euler: 3x1 array in the form [roll, pitch, yaw]
    """
    scipy_rotation = scitf.Rotation.from_matrix(R)
    euler = scipy_rotation.as_euler('XYZ') # caps for intrinsic
    return euler

def euler_xyz_from_quaternion(q: np.ndarray) -> np.ndarray:
    """
    Source: 
    Wrapper for scipy's Rotation class to convert a quaternion to euler angles.
    If you are wondering why I made this instead of directly using scipy's methods,
    well it is because it is super easy to mix the intrinsic and extrinsic notations
    and get the wrong results with scipy.
    Args:
        q: Quaternion in the form [x, y, z, w]

    Returns:
        euler: 3x1 array in the form [roll, pitch, yaw] (in rad)
    """
    scipy_rotation = scitf.Rotation.from_quat(q)
    euler = scipy_rotation.as_euler('XYZ') # caps for intrinsic
    return euler