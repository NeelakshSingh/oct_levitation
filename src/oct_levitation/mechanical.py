import numpy as np
import numpy.typing as np_t
from dataclasses import dataclass
from oct_levitation.common import Constants
from oct_levitation.geometry import rotate_vector_from_quaternion

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
class RigidBodyDipoleInterface:
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
    material_properties: MaterialProperties
    mass_properties: MassProperties
    geometric_properties: CylindricalRingShape
    dipole_strength: float
    mframe: float
    dipole_axis: np_t.NDArray

    def get_gravitational_torque(self, q: np_t.NDArray, g: np_t.NDArray = np.array([0, 0, -Constants.g])) -> np_t.NDArray:
        """
        This function takes the current orientation of the rigid body and returns the torque due to gravity.
        It is useful for gravity compensation in control.

        Parameters
        ----------
        q (np_t.NDArray) : Current orientation of the rigid body in quaternion form w.r.t the world frame.
        g (np_t.NDArray) : Gravitational acceleration vector in the world frame. Defaults to downward
                           earth's gravitation along z-axis.

        Returns
        -------
        np_t.NDArray : Torque due to gravity in the world frame.
        """
        com_world = rotate_vector_from_quaternion(q, self.mass_properties.com_position)
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
    
    def get_gravitational_wrench(self, q: np_t.NDArray, g: np_t.NDArray = np.array([0, 0, -Constants.g])) -> np_t.NDArray:
        """
        This function returns the gravitational wrench acting on the rigid body. 
        It is useful for gravity compensation in control.

        Parameters
        ----------
        q (np_t.NDArray) : Current orientation of the rigid body in quaternion form w.r.t the world frame.
        g (np_t.NDArray) : Gravitational acceleration vector in the world frame. Defaults to downward
                           earth's gravitation along z-axis.
        
        Returns
        -------
        np_t.NDArray : Gravitational wrench [Force, Torque] acting on the rigid body in the world frame.
        """
        return np.hstack((self.get_gravitational_force(g), self.get_gravitational_torque(q, g)))

@dataclass
class TrackingMetadata:
    """
    Metadata for tracking a rigid body through an external tracking system.

    Parameters
    ----------
    pose_frame (str) : Frame name in which the pose of the rigid body is published
    """
    pose_frame: str

##############################################
# DEFINED RIGID BODY DIPOLES #
##############################################

@dataclass
class NarrowRingMagnetS1(RigidBodyDipoleInterface):
    material_properties: MaterialProperties = MaterialProperties(7.5e3, 1.36)
    geometric_properties: CylindricalRingShape = CylindricalRingShape(4.96e-3, (5.11e-3)/2, (9.95e-3)/2)
    dipole_strength: float = material_properties.Br*geometric_properties.volume/Constants.mu_0 # kg*m^2/s
    mframe: float = 2.9e-3
    # Computed using SolidWorks
    mass_properties: MassProperties = MassProperties(geometric_properties.volume*material_properties.density + mframe,
                                                        np.array([[492.29, -74.08, 9.38],
                                                                  [-74.28, 807.43, 5.19],
                                                                  [9.38, 5.19, 1251.91]])*1e-9,
                                                        np.array([[486.01, -78.53, 3.83],
                                                                  [-78.53, 795.57, 2.12],
                                                                  [3.83, 2.12, 1241.41]])*1e-9,
                                                        np.array([1.27, 0.70, 0.87])*1e-3,
                                                        PrincipleAxesAndMomentsOfInertia(
                                                            Ix=np.array([0.97, -0.23, 0.00]),
                                                            Iy=np.array([0.23, 0.97, 0.01]),
                                                            Iz=np.array([-0.01, -0.01, 1.00]),
                                                            Px=467.21e-9,
                                                            Py=814.33e-9,
                                                            Pz=1241.44e-9
                                                        ))
    tracking_data: TrackingMetadata = TrackingMetadata("vicon/small_ring_S1/Origin")
    dipole_axis: np_t.NDArray = np.array([0, 0, 1]) # Default dipole axis is along the z-axis

@dataclass
class NarrowRingMagnetSymmetricSquareS1(RigidBodyDipoleInterface):
    material_properties: MaterialProperties = MaterialProperties(7.5e3, 1.36)
    geometric_properties: CylindricalRingShape = CylindricalRingShape(4.96e-3, (5.11e-3)/2, (9.95e-3)/2)
    dipole_strength: float = material_properties.Br*geometric_properties.volume/Constants.mu_0 # kg*m^2/s
    mframe: float = 22.54e-3 # Mass of the square frame in kg
    # Computed using SolidWorks
    mass_properties: MassProperties = MassProperties(24.67e-3,
                                                     np.diag([6918.86, 7712.23, 14125.84])*1e-9,
                                                     np.diag([6794.76, 7588.13, 14125.84])*1e-9,
                                                     np.array([0.00, 0.00, 2.24])*1e-3,
                                                     PrincipleAxesAndMomentsOfInertia(
                                                            Ix=np.array([-1.00, 0.00, 0.00]),
                                                            Iy=np.array([0.00, 1.00, 0.00]),
                                                            Iz=np.array([0.00, 0.00, -1.00]),
                                                            Px=6794.76e-9,
                                                            Py=7588.13e-9,
                                                            Pz=14125.84e-9
                                                     ))
    tracking_data: TrackingMetadata = TrackingMetadata("vicon/small_ring_square5_frame_S1/Origin")
    dipole_axis: np_t.NDArray = np.array([0, 0, 1])

@dataclass
class NarrowRingMagnetSymmetricXFrameS1(RigidBodyDipoleInterface):
    material_properties: MaterialProperties = MaterialProperties(7.5e3, 1.36)
    geometric_properties: CylindricalRingShape = CylindricalRingShape(4.96e-3, (5.11e-3)/2, (9.95e-3)/2)
    dipole_strength: float = material_properties.Br*geometric_properties.volume/Constants.mu_0 # kg*m^2/s
    mframe: float = 10.35e-3 # Mass of the X frame in kg
    # Computed using SolidWorks
    mass_properties: MassProperties = MassProperties(1.24800000e-02,
                                                     np.array([[4.06596000e-06, 0.00000000e+00, 0.00000000e+00],
                                                               [0.00000000e+00, 2.15403000e-06, 0.00000000e+00],
                                                               [0.00000000e+00, 0.00000000e+00, 5.86103000e-06]]),
                                                     np.array([[3.98796000e-06, 0.00000000e+00, 0.00000000e+00],
                                                               [0.00000000e+00, 2.07603000e-06, 0.00000000e+00],
                                                               [0.00000000e+00, 0.00000000e+00, 5.86103000e-06]]),
                                                     np.array([0.00000000, 0.00000000, 0.00250000])*1e-3,
                                                     PrincipleAxesAndMomentsOfInertia(
                                                            Ix=np.array([0.00000000, -1.00000000, 0.00000000]),
                                                            Iy=np.array([-1.00000000, 0.00000000, 0.00000000]),
                                                            Iz=np.array([0.00000000, 0.00000000, -1.00000000]),
                                                            Px=2.07603000e-06,
                                                            Py=3.98796000e-06,
                                                            Pz=5.86103000e-06
                                                    ))

    tracking_data: TrackingMetadata = TrackingMetadata("vicon/small_ring_X_frame_S1/Origin")
    dipole_axis: np_t.NDArray = np.array([0, 0, 1])