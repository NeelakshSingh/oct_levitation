from oct_levitation.mechanical import *
from geometry_msgs.msg import Vector3, Quaternion

##############################################
# UTILITIES #
##############################################

UNIT_QUATERNION = Quaternion(0.0, 0.0, 0.0, 1.0)
ZERO_TRANSLATION = Vector3()

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

##############################################
# PERMANENT MAGNETS #
##############################################

HKCMDiscMagnet10x3 = PermanentMagnet(
    geometry=CylindricalRingShape(
        t=3e-3, Ri=0.0, Ro=5e-3
    ),
    material=N35Material,
    magnetization_axis=np.array([0, 0, 1])
)

RingMagnet10x5x5 = PermanentMagnet(
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

##############################################
# RIGID BODIES #
##############################################

# TwoDipoleDisc100x15_6HKCM10x3 = MultiDipoleRigidBody(
#     name="two_dipole_disc_100x15_hkcm10x3",
#     mass_properties = MassProperties(1.46710000e-01,
#                                     np.array([[8.33202800e-05, 0.00000000e+00, 0.00000000e+00],
#                                               [0.00000000e+00, 1.02598910e-04, 0.00000000e+00],
#                                               [0.00000000e+00, 0.00000000e+00, 1.79951710e-04]]),
#                                     np.array([[8.33202800e-05, 0.00000000e+00, 0.00000000e+00],
#                                               [0.00000000e+00, 1.02598910e-04, 0.00000000e+00],
#                                               [0.00000000e+00, 0.00000000e+00, 1.79951710e-04]]),
#                                     np.array([0.00000000, 0.00000000, 0.00000000]),
#                                     PrincipleAxesAndMomentsOfInertia(
#                                         Ix=np.array([1.00000000, 0.00000000, 0.00000000]),
#                                         Iy=np.array([0.00000000, -1.00000000, 0.00000000]),
#                                         Iz=np.array([0.00000000, 0.00000000, -1.00000000]),
#                                         Px=8.33202800e-05,
#                                         Py=1.02598910e-04,
#                                         Pz=1.79951710e-04
#                                     )),
#     pose_frame = "vicon/two_dipole_disc_100x15_hkcm10x3/Origin",
#     dipole_list = [
#         MagneticDipole(
#             name="DipolePosX",
#             strength=HKCMDiscMagnet10x3.get_dipole_strength()*6, # Built by symmetric stacking of 6 magnets
#             axis=np.array([0.0, 0.0, 1.0]),
#             transform=Transform(Vector3(30e-3, 0.0, 0.0), UNIT_QUATERNION),
#             frame_name="vicon/two_dipole_disc_100x15_hkcm10x3/DipolePosX"
#         ),
#         MagneticDipole(
#             name="DipoleNegX",
#             strength=HKCMDiscMagnet10x3.get_dipole_strength()*6, # Built by symmetric stacking of 6 magnets
#             axis=np.array([0.0, 0.0, 1.0]),
#             transform=Transform(Vector3(-30e-3, 0.0, 0.0), UNIT_QUATERNION),
#             frame_name="vicon/two_dipole_disc_100x15_hkcm10x3/DipoleNegX"
#         )
#     ]
# )

# TwoDipoleDisc80x15_6HKCM10x3 = MultiDipoleRigidBody(
#     name="two_dipole_disc_80x15",
#     mass_properties = MassProperties(0.096,
#                                      np.array([[3.56773200e-05, 0.00000000e+00, 0.00000000e+00],
#                                                [0.00000000e+00, 5.32428400e-05, 0.00000000e+00],
#                                                [0.00000000e+00, 0.00000000e+00, 8.46661400e-05]]),
#                                     np.array([[3.56773200e-05, 0.00000000e+00, 0.00000000e+00],
#                                               [0.00000000e+00, 5.32428400e-05, 0.00000000e+00],
#                                               [0.00000000e+00, 0.00000000e+00, 8.46661400e-05]]),
#                                     np.array([0.00000000, 0.00000000, 0.00000000]),
#                                     PrincipleAxesAndMomentsOfInertia(
#                                         Ix=np.array([1.00000000, 0.00000000, 0.00000000]),
#                                         Iy=np.array([0.00000000, -1.00000000, 0.00000000]),
#                                         Iz=np.array([0.00000000, 0.00000000, -1.00000000]),
#                                         Px=3.56773200e-05,
#                                         Py=5.32428400e-05,
#                                         Pz=8.46661400e-05
#                                     )),
#     pose_frame = "vicon/two_dipole_disc_80x15/Origin",
#     dipole_list = [
#         MagneticDipole(
#             name="DipolePosX",
#             strength=HKCMDiscMagnet10x3.get_dipole_strength()*6, # Built by symmetric stacking of 6 magnets
#             axis=np.array([0.0, 0.0, -1.0]), # South pole up dipole
#             transform=Transform(Vector3(30e-3, 0.0, 0.0), UNIT_QUATERNION),
#             frame_name="vicon/two_dipole_disc_80x15/DipolePosX"
#         ),
#         MagneticDipole(
#             name="DipoleNegX",
#             strength=HKCMDiscMagnet10x3.get_dipole_strength()*6, # Built by symmetric stacking of 6 magnets
#             axis=np.array([0.0, 0.0, -1.0]), # South pole up dipole
#             transform=Transform(Vector3(-30e-3, 0.0, 0.0), UNIT_QUATERNION),
#             frame_name="vicon/two_dipole_disc_80x15/DipoleNegX"
#         )
#     ]
# )

# POMDiscCenterRingDipole = MultiDipoleRigidBody(
#     name="pom_disc_80x22",
#     mass_properties = MassProperties(1.57330000e-01,
#                                     np.array([[6.93584300e-05, 4.17310000e-07, 2.95500000e-07],
#                                               [4.17310000e-07, 6.21933200e-05, -4.92530000e-07],
#                                               [2.95500000e-07, -4.92530000e-07, 1.16048330e-04]]),
#                                     np.array([[6.93334500e-05, 4.20260000e-07, 2.89540000e-07],
#                                               [4.20260000e-07, 6.21714900e-05, -4.82600000e-07],
#                                               [2.89540000e-07, -4.82600000e-07, 1.16041640e-04]]),
#                                     np.array([0.00011000, -0.00018000, 0.00036000]),
#                                     PrincipleAxesAndMomentsOfInertia(
#                                         Ix=np.array([-0.06000000, -1.00000000, 0.01000000]),
#                                         Iy=np.array([1.00000000, -0.06000000, 0.01000000]),
#                                         Iz=np.array([-0.01000000, 0.01000000, 1.00000000]),
#                                         Px=6.21429000e-05,
#                                         Py=6.93558800e-05,
#                                         Pz=1.16047800e-04
#                                     )),
#     pose_frame = "vicon/pom_disc_80x22/Origin",
#     dipole_list = [
#         MagneticDipole(
#             name="CenterRingDipole",
#             strength=RingMagnet10x5x5.get_dipole_strength()*6, # Built by symmetric stacking of 6 magnets
#             axis=np.array([0.0, 0.0, -1.0]), # South pole up dipole
#             transform=Transform(Vector3(0.0, 0.0, 0.0), UNIT_QUATERNION),
#             frame_name="vicon/pom_disc_80x22/Origin"
#         )
#     ]
# )

Onyx80x22DiscCenterRingDipole = MultiDipoleRigidBody(
    name="onyx_disc_80x22",
    mass_properties = MassProperties(8.51500000e-02,
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
                (Transform(Vector3(0.0, 0.0, 3e-3), Quaternion(*geometry.quaternion_from_euler_xyz(np.array([np.pi, 0, 0])))), RingMagnet10x5x5), # Because these are attached north down, axis is along north fashion.
                (Transform(Vector3(0.0, 0.0, 8e-3), Quaternion(*geometry.quaternion_from_euler_xyz(np.array([np.pi, 0, 0])))), RingMagnet10x5x5),
                (Transform(Vector3(0.0, 0.0, 11e-3), Quaternion(*geometry.quaternion_from_euler_xyz(np.array([np.pi, 0, 0])))), RingMagnet10x5x5),
                (Transform(Vector3(0.0, 0.0, -3e-3), Quaternion(*geometry.quaternion_from_euler_xyz(np.array([np.pi, 0, 0])))), RingMagnet10x5x5),
                (Transform(Vector3(0.0, 0.0, -8e-3), Quaternion(*geometry.quaternion_from_euler_xyz(np.array([np.pi, 0, 0])))), RingMagnet10x5x5),
                (Transform(Vector3(0.0, 0.0, -11e-3), Quaternion(*geometry.quaternion_from_euler_xyz(np.array([np.pi, 0, 0])))), RingMagnet10x5x5),
            ]
        )
    ]
)