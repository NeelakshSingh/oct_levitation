from oct_levitation.mechanical import *
from geometry_msgs.msg import Vector3, Quaternion
from typing import Dict

##############################################
# UTILITIES #
##############################################

# This way of registration is actually not ideal. I would rather come up with a way to initialize the file
# from a json, yaml, or xml file. Because one we have too many rigid bodies all of them will take up RAM in
# the dictionary. Works for now though.
REGISTERED_BODIES: Dict[str, MultiDipoleRigidBody] = {}

UNIT_QUATERNION = Quaternion(0.0, 0.0, 0.0, 1.0)
X_FLIP_QUATERNION = Quaternion(*geometry.quaternion_from_euler_xyz(np.array([np.pi, 0, 0])))

def register_rigid_body(rigid_body: MultiDipoleRigidBody):
    """
    Register a rigid body in the global dictionary.
    """
    if rigid_body.name in REGISTERED_BODIES:
        raise ValueError(f"Rigid body '{rigid_body.name}' is already registered.")
    REGISTERED_BODIES[rigid_body.name] = rigid_body

def list_registered_bodies() -> List[str]:
    """
    List all registered rigid body names.
    
    Returns:
        List[str]: A list of registered rigid body names.
    """
    return list(REGISTERED_BODIES.keys())

##############################################
# MATERIALS #
##############################################

N52Material = MaterialProperties(
    density = 7500,
    Br = 1.47
)

##############################################
# PERMANENT MAGNETS #
##############################################

DiscMagnet10x5_N52 = PermanentMagnet(
    geometry=CylindricalShape(
        t=5e-3, R=5e-3
    ),
    material=N52Material,
    magnetization_axis=np.array([0, 0, 1])
)

##############################################
# RIGID BODIES #
##############################################

GreentecRingDo80Di67MD10_2N52 = MultiDipoleRigidBody(
    name="greentec_do80_di67_Md10_2N52",
    mass_properties = MassProperties(39.3e-3, # The new print is lighter so its mass stayed the same despite heavier magnets.
                                     PrincipleAxesAndMomentsOfInertia( # Changed manually to match the vicon frame, will use this for control.
                                         Ix=np.array([1.0, 0.0, 0.0]),
                                         Iy=np.array([0.0, 1.0, 0.0]),
                                         Iz=np.array([0.0, 0.0, 1.0]),
                                         Px=2.00067e-05,
                                         Py=2.00067e-05,
                                         Pz=3.6221940020475127e-05 # This one is without including magnets and the markers since I don't use it. Recalculate if you need it.
                                     )),
    pose_frame = "vicon/greentec_do80_di67_Md10/Origin",
    dipole_list = [
        MagneticDipole(
            name="CenterDiscDipole",
            axis=np.array([0.0, 0.0, -1.0]), # South pole up dipole, set as a property for now. If required, one can calculate it from the individual magnets.
            transform=Transform(Vector3(0.0, 0.0, 0.0), UNIT_QUATERNION),
            frame_name="vicon/greentec_do80_di67_Md10/Origin",
            magnet_stack=[
                (Transform(Vector3(0.0, 0.0, 3e-3), X_FLIP_QUATERNION), DiscMagnet10x5_N52), # Because these are attached north down.
                (Transform(Vector3(0.0, 0.0, -3e-3), X_FLIP_QUATERNION), DiscMagnet10x5_N52),
            ]
        )
    ]
)

register_rigid_body(GreentecRingDo80Di67MD10_2N52)

BronzefillRing27gm_2N52 = MultiDipoleRigidBody(
    name="bronzefill_ring_27gms_2N52",
    mass_properties = MassProperties(32.4e-3,
                                     PrincipleAxesAndMomentsOfInertia( # Changed manually to match the vicon frame, will use this for control.
                                         Ix=np.array([1.0, 0.0, 0.0]),
                                         Iy=np.array([0.0, 1.0, 0.0]),
                                         Iz=np.array([0.0, 0.0, 1.0]),
                                         Px=6.211e-06,
                                         Py=5.637e-06,
                                         Pz=1.145e-05 # This one is without including magnets and the markers since I don't use it. Recalculate if you need it.
                                     )),
    pose_frame = "vicon/bronzefill_ring_27gms_3_markers/Origin",
    dipole_list = [
        MagneticDipole(
            name="CenterDiscDipole",
            axis=np.array([0.0, 0.0, -1.0]), # South pole up dipole, set as a property for now. If required, one can calculate it from the individual magnets.
            transform=Transform(Vector3(0.0, 0.0, 0.0), UNIT_QUATERNION),
            frame_name="vicon/bronzefill_ring_27gms_3_markers/Origin",
            magnet_stack=[
                (Transform(Vector3(0.0, 0.0, 3e-3), X_FLIP_QUATERNION), DiscMagnet10x5_N52), # Because these are attached north down.
                (Transform(Vector3(0.0, 0.0, -3e-3), X_FLIP_QUATERNION), DiscMagnet10x5_N52),
            ]
        )
    ]
)

register_rigid_body(BronzefillRing27gm_2N52)

BronzefillRing27gm_4N52 = MultiDipoleRigidBody(
    name="bronzefill_ring_27gms_4N52",
    mass_properties = MassProperties(32.4e-3,
                                     PrincipleAxesAndMomentsOfInertia( # Changed manually to match the vicon frame, will use this for control.
                                         Ix=np.array([1.0, 0.0, 0.0]),
                                         Iy=np.array([0.0, 1.0, 0.0]),
                                         Iz=np.array([0.0, 0.0, 1.0]),
                                         Px=6.211e-06,
                                         Py=5.637e-06,
                                         Pz=1.145e-05 # This one is without including magnets and the markers since I don't use it. Recalculate if you need it.
                                     )),
    pose_frame = "vicon/bronzefill_ring_27gms_3_markers/Origin",
    dipole_list = [
        MagneticDipole(
            name="CenterDiscDipole",
            axis=np.array([0.0, 0.0, -1.0]), # South pole up dipole, set as a property for now. If required, one can calculate it from the individual magnets.
            transform=Transform(Vector3(0.0, 0.0, 0.0), UNIT_QUATERNION),
            frame_name="vicon/bronzefill_ring_27gms_3_markers/Origin",
            magnet_stack=[
                (Transform(Vector3(0.0, 0.0, 8e-3), X_FLIP_QUATERNION), DiscMagnet10x5_N52), # Because these are attached north down.
                (Transform(Vector3(0.0, 0.0, 3e-3), X_FLIP_QUATERNION), DiscMagnet10x5_N52), # Because these are attached north down.
                (Transform(Vector3(0.0, 0.0, -3e-3), X_FLIP_QUATERNION), DiscMagnet10x5_N52), # Because these are attached north down.
                (Transform(Vector3(0.0, 0.0, -8e-3), X_FLIP_QUATERNION), DiscMagnet10x5_N52),
            ]
        )
    ]
)

register_rigid_body(BronzefillRing27gm_4N52)