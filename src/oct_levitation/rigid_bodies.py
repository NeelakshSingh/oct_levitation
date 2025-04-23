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
ZERO_TRANSLATION = Vector3()

def register_rigid_body(rigid_body: MultiDipoleRigidBody):
    """
    Register a rigid body in the global dictionary.
    """
    REGISTERED_BODIES[rigid_body.name] = rigid_body

##############################################
# MATERIALS #
##############################################

N35Material = MaterialProperties(
    density = 7500,
    Br = 1.17
)

N45Material = MaterialProperties(
    density = 7500,
    Br = 1.35
)

N50Material = MaterialProperties(
    density=7500,
    Br=1.42
)

N52Material = MaterialProperties(
    density = 7500,
    Br = 1.47
)

##############################################
# PERMANENT MAGNETS #
##############################################

HKCMDiscMagnet10x3_N35 = PermanentMagnet(
    geometry=CylindricalRingShape(
        t=3e-3, Ri=0.0, Ro=5e-3
    ),
    material=N35Material,
    magnetization_axis=np.array([0, 0, 1])
)

RingMagnet10x5x5_N35 = PermanentMagnet(
    geometry=CylindricalRingShape(
        t=5e-3, Ri=5e-3/2, Ro=5e-3
    ),
    material=N35Material,
    magnetization_axis=np.array([0, 0, 1])
)

RingMagnet8x3x4_N45 = PermanentMagnet(
    geometry=CylindricalRingShape(
        t=4e-3, Ri=(3e-3)/2, Ro=4e-3
    ),
    material=N45Material,
    magnetization_axis=np.array([0, 0, 1])
)

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

Onyx80x22DiscCenterRingDipole = MultiDipoleRigidBody(
    name="onyx_disc_80x22",
    mass_properties = MassProperties(8.51500000e-02, # TODO: Replace this with the parameteric inertia calculation
                                     np.array([[3.37872600e-05, -1.45300000e-07, -6.66800000e-08],
                                               [-1.45300000e-07, 3.02408200e-05, -1.11150000e-07],
                                               [-6.66800000e-08, -1.11150000e-07, 5.90235300e-05]]),
                                     np.array([[3.37844000e-05, -1.45960000e-07, -6.58400000e-08],
                                               [-1.45960000e-07, 3.02386600e-05, -1.09760000e-07],
                                               [-6.58400000e-08, -1.09760000e-07, 5.90220300e-05]]),
                                     np.array([-0.00007000, -0.00011000, 0.00014000]),
                                     PrincipleAxesAndMomentsOfInertia(
                                         Ix=np.array([0.04000000, -1.00000000, 0.00000000]),
                                         Iy=np.array([1.00000000, 0.04000000, 0.00000000]),
                                         Iz=np.array([0.00000000, 0.00000000, 1.00000000]),
                                         Px=3.02322700e-05,
                                         Py=3.37902000e-05,
                                         Pz=5.90226200e-05
                                    )),
    pose_frame = "vicon/onyx_disc_80x22/Origin",
    dipole_list = [
        MagneticDipole(
            name="CenterRingDipole",
            axis=np.array([0.0, 0.0, -1.0]), # South pole up dipole, set as a property for now. If required, one can calculate it from the individual magnets.
            transform=Transform(Vector3(0.0, 0.0, 0.0), UNIT_QUATERNION),
            frame_name="vicon/onyx_disc_80x22/Origin",
            magnet_stack=[
                (Transform(Vector3(0.0, 0.0, 3e-3), Quaternion(*geometry.quaternion_from_euler_xyz(np.array([np.pi, 0, 0])))), RingMagnet10x5x5_N35), # Because these are attached north down, axis is along north fashion.
                (Transform(Vector3(0.0, 0.0, 8e-3), Quaternion(*geometry.quaternion_from_euler_xyz(np.array([np.pi, 0, 0])))), RingMagnet10x5x5_N35),
                (Transform(Vector3(0.0, 0.0, 11e-3), Quaternion(*geometry.quaternion_from_euler_xyz(np.array([np.pi, 0, 0])))), RingMagnet10x5x5_N35),
                (Transform(Vector3(0.0, 0.0, -3e-3), Quaternion(*geometry.quaternion_from_euler_xyz(np.array([np.pi, 0, 0])))), RingMagnet10x5x5_N35),
                (Transform(Vector3(0.0, 0.0, -8e-3), Quaternion(*geometry.quaternion_from_euler_xyz(np.array([np.pi, 0, 0])))), RingMagnet10x5x5_N35),
                (Transform(Vector3(0.0, 0.0, -11e-3), Quaternion(*geometry.quaternion_from_euler_xyz(np.array([np.pi, 0, 0])))), RingMagnet10x5x5_N35),
            ]
        )
    ]
)

register_rigid_body(Onyx80x22DiscCenterRingDipole)

Onyx80x7_5DiscCenterRingDipole = MultiDipoleRigidBody(
    name="onyx_disc_80x7_5",
    mass_properties = MassProperties(4.97200000e-02,
                                     np.array([[1.72934900e-05, -1.45310000e-07, -4.92400000e-08],
                                     [-1.45310000e-07, 1.53558800e-05, -8.20600000e-08],
                                     [-4.92400000e-08, -8.20600000e-08, 3.00703000e-05]]),
                                     np.array([[1.72891100e-05, -1.46440000e-07, -4.79400000e-08],
                                     [-1.46440000e-07, 1.53527100e-05, -7.99000000e-08],
                                     [-4.79400000e-08, -7.99000000e-08, 3.00677300e-05]]),
                                     np.array([-0.00012000, -0.00019000, 0.00022000]),
                                     PrincipleAxesAndMomentsOfInertia(
                                         Ix=np.array([0.07000000, -1.00000000, 0.01000000]),
                                         Iy=np.array([-1.00000000, -0.07000000, 0.00000000]),
                                         Iz=np.array([0.00000000, -0.01000000, -1.00000000]),
                                         Px=1.53413000e-05,
                                         Py=1.72998900e-05,
                                         Pz=3.00683500e-05
                                    )),
    pose_frame = "vicon/onyx_disc_80x7_5/Origin",
    dipole_list = [
        MagneticDipole(
            name="CenterRingDipole",
            axis=np.array([0.0, 0.0, -1.0]), # South pole up dipole, set as a property for now. If required, one can calculate it from the individual magnets.
            transform=Transform(Vector3(0.0, 0.0, 0.0), UNIT_QUATERNION),
            frame_name="vicon/onyx_disc_80x7_5/Origin",
            magnet_stack=[
                (Transform(Vector3(0.0, 0.0, 3e-3), Quaternion(*geometry.quaternion_from_euler_xyz(np.array([np.pi, 0, 0])))), RingMagnet10x5x5_N35), # Because these are attached north down, axis is along north fashion.
                (Transform(Vector3(0.0, 0.0, 8e-3), Quaternion(*geometry.quaternion_from_euler_xyz(np.array([np.pi, 0, 0])))), RingMagnet10x5x5_N35),
                (Transform(Vector3(0.0, 0.0, 11e-3), Quaternion(*geometry.quaternion_from_euler_xyz(np.array([np.pi, 0, 0])))), RingMagnet10x5x5_N35),
                (Transform(Vector3(0.0, 0.0, -3e-3), Quaternion(*geometry.quaternion_from_euler_xyz(np.array([np.pi, 0, 0])))), RingMagnet10x5x5_N35),
                (Transform(Vector3(0.0, 0.0, -8e-3), Quaternion(*geometry.quaternion_from_euler_xyz(np.array([np.pi, 0, 0])))), RingMagnet10x5x5_N35),
                (Transform(Vector3(0.0, 0.0, -11e-3), Quaternion(*geometry.quaternion_from_euler_xyz(np.array([np.pi, 0, 0])))), RingMagnet10x5x5_N35),
            ]
        )
    ]
)

register_rigid_body(Onyx80x7_5DiscCenterRingDipole)

Onyx80x22DiscCenterRingDipoleI40 = MultiDipoleRigidBody(
    name="onyx_disc_80x22_I40",
    mass_properties = MassProperties(4.23800000e-02,
                                     np.array([[1.39912560e-04, -2.79978900e-05, 1.06775840e-04],
                                               [-2.79978900e-05, 2.38473280e-04, -3.05975000e-05],
                                               [1.06775840e-04, -3.05975000e-05, 1.18070670e-04]]),
                                     np.array([[1.46556700e-05, -6.48700000e-08, 1.46640000e-07],
                                               [-6.48700000e-08, 2.42591800e-05, 1.08130000e-07],
                                               [1.46640000e-07, 1.08130000e-07, 1.30259300e-05]]),
                                     np.array([0.04784000, -0.01378000, 0.05259000]),
                                     PrincipleAxesAndMomentsOfInertia(
                                         Ix=np.array([0.09000000, 0.01000000, 1.00000000]),
                                         Iy=np.array([1.00000000, -0.01000000, -0.09000000]),
                                         Iz=np.array([0.01000000, 1.00000000, -0.01000000]),
                                         Px=1.30119200e-05,
                                         Py=1.46681800e-05,
                                         Pz=2.42606800e-05
                                    )),
    pose_frame = "vicon/onyx_disc_80x22/Origin",
    dipole_list = [
        MagneticDipole(
            name="CenterRingDipole",
            axis=np.array([0.0, 0.0, -1.0]), # South pole up dipole, set as a property for now. If required, one can calculate it from the individual magnets.
            transform=Transform(Vector3(0.0, 0.0, 0.0), UNIT_QUATERNION),
            frame_name="vicon/onyx_disc_80x22/Origin",
            magnet_stack=[
                (Transform(Vector3(0.0, 0.0, 3e-3), Quaternion(*geometry.quaternion_from_euler_xyz(np.array([np.pi, 0, 0])))), RingMagnet10x5x5_N35), # Because these are attached north down, axis is along north fashion.
                (Transform(Vector3(0.0, 0.0, 8e-3), Quaternion(*geometry.quaternion_from_euler_xyz(np.array([np.pi, 0, 0])))), RingMagnet10x5x5_N35),
                (Transform(Vector3(0.0, 0.0, 11e-3), Quaternion(*geometry.quaternion_from_euler_xyz(np.array([np.pi, 0, 0])))), RingMagnet10x5x5_N35),
                (Transform(Vector3(0.0, 0.0, -3e-3), Quaternion(*geometry.quaternion_from_euler_xyz(np.array([np.pi, 0, 0])))), RingMagnet10x5x5_N35),
                (Transform(Vector3(0.0, 0.0, -8e-3), Quaternion(*geometry.quaternion_from_euler_xyz(np.array([np.pi, 0, 0])))), RingMagnet10x5x5_N35),
                (Transform(Vector3(0.0, 0.0, -11e-3), Quaternion(*geometry.quaternion_from_euler_xyz(np.array([np.pi, 0, 0])))), RingMagnet10x5x5_N35),
            ]
        )
    ]
)

register_rigid_body(Onyx80x22DiscCenterRingDipoleI40)

Onyx80x22DiscCenterDiscDipoleI40_N52 = MultiDipoleRigidBody(
    name="onyx_disc_80x22_I40_N52",
    mass_properties = MassProperties(0.04679786466911065,
                                     np.array([[ 1.40133453e-04, -2.79978900e-05,  1.06775840e-04], # HOW CAN THE INERTIA BE HIGHER FOR THIS DISC THAN THE ONE WITH 100% INFILL?
                                               [-2.79978900e-05,  2.38694173e-04, -3.05975000e-05],
                                               [ 1.06775840e-04, -3.05975000e-05,  1.18291563e-04]]),
                                     np.array([[ 1.48765632e-05, -6.48700000e-08,  1.46640000e-07],
                                               [-6.48700000e-08,  2.44800732e-05,  1.08130000e-07],
                                               [ 1.46640000e-07,  1.08130000e-07,  1.32468232e-05]]),
                                     np.array([0.04784000, -0.01378000, 0.05259000]),
                                     PrincipleAxesAndMomentsOfInertia(
                                         Ix=np.array([0.09000000, 0.01000000, 1.00000000]),
                                         Iy=np.array([1.00000000, -0.01000000, -0.09000000]),
                                         Iz=np.array([0.01000000, 1.00000000, -0.01000000]),
                                         Px=1.3232813233455532e-05,
                                         Py=1.4889073233455532e-05,
                                         Pz=2.4481573233455534e-05
                                    )),
    pose_frame = "vicon/onyx_disc_80x22/Origin",
    dipole_list = [
        MagneticDipole(
            name="CenterDiscDipole",
            axis=np.array([0.0, 0.0, -1.0]), # South pole up dipole, set as a property for now. If required, one can calculate it from the individual magnets.
            transform=Transform(Vector3(0.0, 0.0, 0.0), UNIT_QUATERNION),
            frame_name="vicon/onyx_disc_80x22/Origin",
            magnet_stack=[
                (Transform(Vector3(0.0, 0.0, 3e-3), Quaternion(*geometry.quaternion_from_euler_xyz(np.array([np.pi, 0, 0])))), DiscMagnet10x5_N52), # Because these are attached north down, axis is along north fashion.
                (Transform(Vector3(0.0, 0.0, 8e-3), Quaternion(*geometry.quaternion_from_euler_xyz(np.array([np.pi, 0, 0])))), DiscMagnet10x5_N52),
                (Transform(Vector3(0.0, 0.0, 11e-3), Quaternion(*geometry.quaternion_from_euler_xyz(np.array([np.pi, 0, 0])))), DiscMagnet10x5_N52),
                (Transform(Vector3(0.0, 0.0, -3e-3), Quaternion(*geometry.quaternion_from_euler_xyz(np.array([np.pi, 0, 0])))), DiscMagnet10x5_N52),
                (Transform(Vector3(0.0, 0.0, -8e-3), Quaternion(*geometry.quaternion_from_euler_xyz(np.array([np.pi, 0, 0])))), DiscMagnet10x5_N52),
                (Transform(Vector3(0.0, 0.0, -11e-3), Quaternion(*geometry.quaternion_from_euler_xyz(np.array([np.pi, 0, 0])))), DiscMagnet10x5_N52),
            ]
        )
    ]
)

register_rigid_body(Onyx80x22DiscCenterDiscDipoleI40_N52)

# Onyx80x22DiscCenterDiscDipoleI20 = MultiDipoleRigidBody(
#     name="onyx_disc_80x22_I20",
#     mass_properties = MassProperties(0.04679786466911065,
#                                      np.array([[ 1.40133453e-04, -2.79978900e-05,  1.06775840e-04],
#                                                [-2.79978900e-05,  2.38694173e-04, -3.05975000e-05],
#                                                [ 1.06775840e-04, -3.05975000e-05,  1.18291563e-04]]),
#                                      np.array([[ 1.48765632e-05, -6.48700000e-08,  1.46640000e-07],
#                                                [-6.48700000e-08,  2.44800732e-05,  1.08130000e-07],
#                                                [ 1.46640000e-07,  1.08130000e-07,  1.32468232e-05]]),
#                                      np.array([0.04784000, -0.01378000, 0.05259000]),
#                                      PrincipleAxesAndMomentsOfInertia(
#                                          Ix=np.array([0.09000000, 0.01000000, 1.00000000]),
#                                          Iy=np.array([1.00000000, -0.01000000, -0.09000000]),
#                                          Iz=np.array([0.01000000, 1.00000000, -0.01000000]),
#                                          Px=1.3232813233455532e-05,
#                                          Py=1.4889073233455532e-05,
#                                          Pz=2.4481573233455534e-05
#                                     )),
#     pose_frame = "vicon/onyx_disc_80x22/Origin",
#     dipole_list = [
#         MagneticDipole(
#             name="CenterDiscDipole",
#             axis=np.array([0.0, 0.0, -1.0]), # South pole up dipole, set as a property for now. If required, one can calculate it from the individual magnets.
#             transform=Transform(Vector3(0.0, 0.0, 0.0), UNIT_QUATERNION),
#             frame_name="vicon/onyx_disc_80x22/Origin",
#             magnet_stack=[
#                 (Transform(Vector3(0.0, 0.0, 3e-3), Quaternion(*geometry.quaternion_from_euler_xyz(np.array([np.pi, 0, 0])))), RingMagnet10x5x5_N35), # Because these are attached north down, axis is along north fashion.
#                 (Transform(Vector3(0.0, 0.0, 8e-3), Quaternion(*geometry.quaternion_from_euler_xyz(np.array([np.pi, 0, 0])))), RingMagnet10x5x5_N35),
#                 (Transform(Vector3(0.0, 0.0, 11e-3), Quaternion(*geometry.quaternion_from_euler_xyz(np.array([np.pi, 0, 0])))), RingMagnet10x5x5_N35),
#                 (Transform(Vector3(0.0, 0.0, -3e-3), Quaternion(*geometry.quaternion_from_euler_xyz(np.array([np.pi, 0, 0])))), RingMagnet10x5x5_N35),
#                 (Transform(Vector3(0.0, 0.0, -8e-3), Quaternion(*geometry.quaternion_from_euler_xyz(np.array([np.pi, 0, 0])))), RingMagnet10x5x5_N35),
#                 (Transform(Vector3(0.0, 0.0, -11e-3), Quaternion(*geometry.quaternion_from_euler_xyz(np.array([np.pi, 0, 0])))), RingMagnet10x5x5_N35),
#             ]
#         )
#     ]
# )

# register_rigid_body(Onyx80x22DiscCenterDiscDipoleI20)