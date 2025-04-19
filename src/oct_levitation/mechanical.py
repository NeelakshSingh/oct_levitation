import numpy as np
import numpy.typing as np_t
import oct_levitation.geometry as geometry
import tf.transformations as tr
import tf2_ros
import rospy
import os

from dataclasses import dataclass
from mag_manip import mag_manip
from geometry_msgs.msg import Transform, TransformStamped
from oct_levitation.common import Constants
from typing import List, Dict, Union, Tuple


##################################
### Rigid Body Properties: Mass and Material
##################################

@dataclass
class PrincipleAxesAndMomentsOfInertia:
    """
    Principle axes and moments of inertia of a rigid body at the center of mass. All directions
    should be with respect to the body-fixed frame used for the rest of the rigid body properties.
    Principle axes are the directions along which the individual moments of inertia are maximum.

    Moments are in kg*m^2. Directions are in meters.

    Parameters
    ----------
    Ix (np_t.NDArray) : X-axis of the principle axis of inertia.
    Iy (np_t.NDArray) : Y-axis of the principle axis of inertia.
    Iz (np_t.NDArray) : Z-axis of the principle axis of inertia.
    Px (float) : Principle moment of inertia about the X-axis.
    Py (float) : Principle moment of inertia about the Y-axis.
    Pz (float) : Principle moment of inertia about the Z-axis.
    """
    Ix: np_t.NDArray
    Iy: np_t.NDArray
    Iz: np_t.NDArray
    Px: float
    Py: float
    Pz: float

    def __post_init__(self):
        self.Ix = self.Ix/np.linalg.norm(self.Ix, 2)
        self.Iy = self.Iy/np.linalg.norm(self.Iy, 2)
        self.Iz = self.Iz/np.linalg.norm(self.Iz, 2)

@dataclass
class MassProperties:
    """
    Mass properties of a rigid body.

    Parameters
    ----------
    m (float) : Mass of the rigid body in kg.
    I_bf (np_t.NDArray) : Inertia tensor of the rigid body with respect to the body-fixed frame.
    I_com (np_t.NDArray) : Inertia tensor of the rigid body with respect to the center of mass for axes 
                           aligned with the body-fixed frame.
    com_position (np_t.NDArray) : Position of the center of mass with respect to the body-fixed frame.
    com_inertia_properties (PrincipleAxesAndMomentsOfInertia) : Principle axes and moments of inertia of 
                           the rigid body with respect to the center of mass. All directions are with respect
                            to the body-fixed frame.
    """
    m: float
    I_bf: np_t.NDArray
    I_com: np_t.NDArray
    com_position: np_t.NDArray
    com_inertia_properties: PrincipleAxesAndMomentsOfInertia

@dataclass
class MaterialProperties:
    """
    Material properties of the body, used for magnetic properties and mass calculations.

    Parameters
    ----------
    density (float) : Density of the material in kg/m^3.
    Br (float) : Remanence of the material in Tesla [T].
    """
    density: float
    Br: float

####################################
### Shapes and Geometric Properties To Store Magnets
####################################

@dataclass
class ShapePropertiesInterface:
    _volume = None # No type annotation to keep this variable out of init and private.

    @property
    def volume(self):
        if self._volume is None:
            raise NotImplementedError(f"Volume calculation not implemented for the shape {self.__name__}")
        return self._volume

@dataclass
class CylindricalRingShape(ShapePropertiesInterface):
    """
    Geometric properties of a cylindrical ring.

    Parameters
    ----------
    t (float) : Thickness (height) of the ring in meters.
    Ri (float) : Inner radius of the ring in meters.
    Ro (float) : Outer radius of the ring in meters.
    """
    t: float
    Ri: float
    Ro: float

    @property
    def volume(self):
        if self._volume is None:
            self._volume = np.pi*(self.Ro**2 - self.Ri**2)*self.t
        return self._volume

@dataclass
class CylindricalShape(CylindricalRingShape):
    """
    Geometric properties of a cylindrical shape.

    Parameters
    ----------
    t (float) : Thickness (height) of the cylinder in meters.
    R (float) : Radius of the cylinder in meters.
    """
    def __init__(self, t: float, R:float):
        object.__setattr__(self, 't', t)
        object.__setattr__(self, 'Ri', 0.0)
        object.__setattr__(self, 'Ro', R)

@dataclass
class PermanentMagnet:
    """
    Represents a permanent magnet with defined geometry, material properties.

    Attributes:
        geometry (ShapePropertiesInterface): The geometric properties of the magnet.
        material (MaterialProperties): The material properties, including remanent magnetization.
    """
    geometry: ShapePropertiesInterface
    material: MaterialProperties
    magnetization_axis: np_t.NDArray

    def get_dipole_strength(self):
        return self.material.Br*self.geometry.volume/Constants.mu_0 # kg*m^2/s


##################################
### Magnetic Dipole and Rigid Body
##################################

@dataclass
class MagneticDipole:
    """
    Represents a magnetic dipole, which can be associated with one or more permanent magnets.

    Attributes:
        name (str): The name of the dipole.
        strength (float): The magnetic dipole moment strength (A·m²).
        axis (np_t.NDArray): A 3D unit vector representing the dipole's orientation.
        transform (Transform): The pose of the dipole w.r.t to the body fixed frame it is attached to.
        frame_name (str): The name of the coordinate frame in which the dipole is defined.
        magnet_stack (Dict[Transform, PermanentMagnet]): A map of magnet transforms w.r.t dipole center frame and the magnet properties themselves.
    """
    name: str
    axis: np_t.NDArray
    transform: Transform
    frame_name: str
    magnet_stack: List[Tuple[Transform, PermanentMagnet]]
    _strength: float = None
    
    @property
    def strength(self):
        # NOTE: Just a naive addition for now, because it considers all magnets to be aligned. 
        # They won't stick to each other otherwise anyways.
        if self._strength is None:
            self._strength = 0.0
            for magnet_tf, magnet in self.magnet_stack:
                self._strength += magnet.get_dipole_strength()
        return self._strength


@dataclass
class SingleDipoleRigidBody:
    """
    Interface for a dipole rigid body and its required mechanical properties.

    Parameters
    ----------
    material_properties (MaterialProperties) : Material properties of the rigid body.
    mass_properties (MassProperties) : Mass properties of the rigid body.
    geometric_properties (ShapePropertiesInterface) : Geometric properties of the rigid body.
    dipole_strength (float) : Dipole strength of the rigid body in A*m^2.
    mframe (float) : Mass of any external mechanical attachment to the rigid body in kg, for example, vicon markers.
    dipole_axis (np_t.NDArray) : Axis of the dipole (S->N) in the body-fixed frame.

    Methods
    -------
    get_gravitational_torque(q: np_t.NDArray, g: np_t.NDArray) -> np_t.NDArray:
        This function takes the current orientation of the rigid body and returns the torque due to gravity.
        It is useful for gravity compensation in control.
    """
    mass_properties: MassProperties
    dipole: MagneticDipole
    mframe: float
    pose_frame: str

    def get_gravitational_torque(self, msg: geometry.TransformStamped, g: np_t.NDArray = np.array([0, 0, -Constants.g])) -> np_t.NDArray:
        """
        This function takes the current orientation of the rigid body and returns the torque due to gravity.
        It is useful for gravity compensation in control.

        Parameters
        ----------
        msg (TransformStamped): The ROS transform message of the rigid body.
        g (np_t.NDArray) : Gravitational acceleration vector in the world frame. Defaults to downward
                           earth's gravitation along z-axis.

        Returns
        -------
        np_t.NDArray : Torque due to gravity in the world frame.
        """
        com_world = geometry.transform_vector_from_transform_stamped(msg, self.mass_properties.com_position)
        return np.cross(com_world, self.mass_properties.m*g)
    
    def get_gravitational_force(self, g: np_t.NDArray = np.array([0, 0, -Constants.g])) -> np_t.NDArray:
        """
        This function returns the gravitational force acting on the rigid body. 
        It is useful for gravity compensation in control.

        Parameters
        ----------
        g (np_t.NDArray) : Gravitational acceleration vector in the world frame. Defaults to downward
                            earth's gravitation along z-axis.
        
        Returns
        -------
        np_t.NDArray : Gravitational force acting on the rigid body in the world frame.
        """
        return self.mass_properties.m*g
    
    def get_gravitational_wrench(self, msg: geometry.TransformStamped, g: np_t.NDArray = np.array([0, 0, -Constants.g])) -> np_t.NDArray:
        """
        This function returns the gravitational wrench acting on the rigid body. 
        It is useful for gravity compensation in control.

        Parameters
        ----------
        msg (TransformStamped): The ROS transform message of the rigid body.
        g (numpy.NDArray) : Gravitational acceleration vector in the world frame. Defaults to downward
                           earth's gravitation along z-axis.
        
        Returns
        -------
        numpy.NDArray : Gravitational wrench [Force, Torque] acting on the rigid body in the world frame.
        """
        return np.hstack((self.get_gravitational_force(g), self.get_gravitational_torque(msg, g)))

@dataclass
class MultiDipoleRigidBody:
    """
    Represents a rigid body with multiple attached magnetic dipoles.

    Attributes
    ----------
    mass_properties : MassProperties
        The mass properties of the rigid body, including mass and inertia.
    mframe : float
        The reference frame mass multiplier for scaling calculations.
    pose_frame : str
        The name of the reference frame in which the rigid body's pose is defined.
    dipole_list : List[MagneticDipole]
        A list of magnetic dipoles attached to the rigid body.
    """
    name: str
    mass_properties: MassProperties
    pose_frame: str
    dipole_list: List[MagneticDipole]

    # Default arguments
    com_wrench_topic_prefix: str = "com_wrench" # object_name/wrench
    dipole_wrench_topic_prefix: str = "wrench" # dipole_name/wrench

    def __post_init__(self):
        # initialize topic names
        self.com_wrench_topic = os.path.join(self.name, self.com_wrench_topic_prefix)

        self.dipole_wrench_topic_list = [os.path.join(self.name, dipole.name, self.dipole_wrench_topic_prefix)
                                         for dipole in self.dipole_list]

    def get_gravitational_force(self, g: np_t.NDArray = np.array([0, 0, -Constants.g])) -> np_t.NDArray:
        """
        Compute the gravitational force acting on the rigid body.

        Parameters
        ----------
        g : np_t.NDArray, optional
            Gravitational acceleration vector in the world frame. Defaults to Earth's gravity 
            along the negative z-axis.

        Returns
        -------
        np_t.NDArray
            Gravitational force acting on the rigid body in the world frame.
        """
        return self.mass_properties.m*g
    
    def get_dipole_transforms(self, tf_buffer: tf2_ros.Buffer, world_frame: str = "vicon/world") -> List[TransformStamped]:
        dipole_tf_list = []
        for dipole in self.dipole_list:
            dipole_tf: TransformStamped = tf_buffer.lookup_transform(world_frame,
                                                                     dipole.frame_name,
                                                                     rospy.Time())
            dipole_tf_list.append(dipole_tf)

        return dipole_tf_list
    
    def get_vicon_frame_transform(self, tf_buffer: tf2_ros.Buffer, world_frame: str = "vicon/world") -> TransformStamped:
        return tf_buffer.lookup_transform(world_frame, self.pose_frame, rospy.Time())
    
    def get_magnetic_interaction_matrices_from_world_frame(self,tf_buffer: tf2_ros.Buffer,
                                                           world_frame: str = "vicon/world", 
                                                           full_mat: bool = False, 
                                                           torque_first: bool = False) -> List[Union[np_t.ArrayLike, List[np_t.ArrayLike]]]:
        """
        Compute the magnetic interaction matrices for each attached dipole.

        Parameters
        ----------
        world_frame : str
            Name of the world frame to be passed to lookup transform

        Returns
        -------
        List[Union[np_t.ArrayLike, List[np_t.ArrayLike]]]
            List of each magnetic interaction matrix function return values, order corresponding to dipole list
        """
        mag_list = []
        for dipole in self.dipole_list:
            dipole_tf: TransformStamped = tf_buffer.lookup_transform(world_frame,
                                                                     dipole.frame_name,
                                                                     rospy.Time())

            dipole_magnetization = geometry.get_magnetic_interaction_matrix(dipole_tf=dipole_tf,
                                                                            dipole_strength=dipole.strength,
                                                                            dipole_axis=dipole.axis,
                                                                            full_mat=full_mat,
                                                                            torque_first=torque_first)
            mag_list.append(dipole_magnetization)
        
        return mag_list
    
    def get_current_wrench_matrices_from_world_frame_mpem(self,tf_buffer: tf2_ros.Buffer,
                                                          mpem_model: mag_manip.ForwardModelMPEM,
                                                           world_frame: str = "vicon/world", 
                                                           full_mat: bool = False, 
                                                           torque_first: bool = False) -> List[Union[np_t.ArrayLike, List[np_t.ArrayLike]]]:
        """
        Compute the magnetic interaction matrices for each attached dipole.

        Parameters
        ----------
        world_frame : str
            Name of the world frame to be passed to lookup transform

        Returns
        -------
        List[Union[np_t.ArrayLike, List[np_t.ArrayLike]]]
            List of each magnetic interaction matrix function return values, order corresponding to dipole list
        """
        MA_list = []
        for dipole in self.dipole_list:
            dipole_tf: TransformStamped = tf_buffer.lookup_transform(world_frame,
                                                                     dipole.frame_name,
                                                                     rospy.Time())

            dipole_magnetization = geometry.get_magnetic_interaction_matrix(dipole_tf=dipole_tf,
                                                                            dipole_strength=dipole.strength,
                                                                            dipole_axis=dipole.axis,
                                                                            full_mat=full_mat,
                                                                            torque_first=torque_first)
            
            A = mpem_model.getActuationMatrix(np.array([
                dipole_tf.transform.translation.x,
                dipole_tf.transform.translation.y,
                dipole_tf.transform.translation.z
            ]))

            MA_list.append( dipole_magnetization @ A )
        
        return MA_list