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

##############################################
# PERMANENT MAGNETS #
##############################################

HKCMDiscMagnet10x3 = PermanentMagnet(
    geometry=CylindricalRingShape(
        t=3e-3, Ri=0.0, Ro=5e-3
    ),
    material=N35Material
)

##############################################
# RIGID BODIES #
##############################################

TwoDipoleDisc100x15_6HKCM10x3 = MultiDipoleRigidBody(
    name="two_dipole_disc_100x15_hkcm10x3",
    mass_properties = MassProperties(1.46710000e-01,
                                    np.array([[8.33202800e-05, 0.00000000e+00, 0.00000000e+00],
                                              [0.00000000e+00, 1.02598910e-04, 0.00000000e+00],
                                              [0.00000000e+00, 0.00000000e+00, 1.79951710e-04]]),
                                    np.array([[8.33202800e-05, 0.00000000e+00, 0.00000000e+00],
                                              [0.00000000e+00, 1.02598910e-04, 0.00000000e+00],
                                              [0.00000000e+00, 0.00000000e+00, 1.79951710e-04]]),
                                    np.array([0.00000000, 0.00000000, 0.00000000]),
                                    PrincipleAxesAndMomentsOfInertia(
                                        Ix=np.array([1.00000000, 0.00000000, 0.00000000]),
                                        Iy=np.array([0.00000000, -1.00000000, 0.00000000]),
                                        Iz=np.array([0.00000000, 0.00000000, -1.00000000]),
                                        Px=8.33202800e-05,
                                        Py=1.02598910e-04,
                                        Pz=1.79951710e-04
                                    )),
    pose_frame = "vicon/two_dipole_disc_100x15_hkcm10x3/Origin",
    dipole_list = [
        MagneticDipole(
            name="DipolePosX",
            strength=HKCMDiscMagnet10x3.get_dipole_strength()*6, # Built by symmetric stacking of 6 magnets
            axis=np.array([0.0, 0.0, 1.0]),
            transform=Transform(Vector3(30e-3, 0.0, 0.0), UNIT_QUATERNION),
            frame_name="vicon/two_dipole_disc_100x15_hkcm10x3/DipolePosX"
        ),
        MagneticDipole(
            name="DipoleNegX",
            strength=HKCMDiscMagnet10x3.get_dipole_strength()*6, # Built by symmetric stacking of 6 magnets
            axis=np.array([0.0, 0.0, 1.0]),
            transform=Transform(Vector3(-30e-3, 0.0, 0.0), UNIT_QUATERNION),
            frame_name="vicon/two_dipole_disc_100x15_hkcm10x3/DipoleNegX"
        )
    ]
)

TwoDipoleDisc80x15_6HKCM10x3 = MultiDipoleRigidBody(
    name="two_dipole_disc_80x15",
    mass_properties = MassProperties(0.0937,
                                     np.array([[3.56773200e-05, 0.00000000e+00, 0.00000000e+00],
                                               [0.00000000e+00, 5.32428400e-05, 0.00000000e+00],
                                               [0.00000000e+00, 0.00000000e+00, 8.46661400e-05]]),
                                    np.array([[3.56773200e-05, 0.00000000e+00, 0.00000000e+00],
                                              [0.00000000e+00, 5.32428400e-05, 0.00000000e+00],
                                              [0.00000000e+00, 0.00000000e+00, 8.46661400e-05]]),
                                    np.array([0.00000000, 0.00000000, 0.00000000]),
                                    PrincipleAxesAndMomentsOfInertia(
                                        Ix=np.array([1.00000000, 0.00000000, 0.00000000]),
                                        Iy=np.array([0.00000000, -1.00000000, 0.00000000]),
                                        Iz=np.array([0.00000000, 0.00000000, -1.00000000]),
                                        Px=3.56773200e-05,
                                        Py=5.32428400e-05,
                                        Pz=8.46661400e-05
                                    )),
    pose_frame = "vicon/two_dipole_disc_80x15/Origin",
    dipole_list = [
        MagneticDipole(
            name="DipolePosX",
            strength=HKCMDiscMagnet10x3.get_dipole_strength()*6, # Built by symmetric stacking of 6 magnets
            axis=np.array([0.0, 0.0, -1.0]), # South pole up dipole
            transform=Transform(Vector3(30e-3, 0.0, 0.0), UNIT_QUATERNION),
            frame_name="vicon/two_dipole_disc_80x15/DipolePosX"
        ),
        MagneticDipole(
            name="DipoleNegX",
            strength=HKCMDiscMagnet10x3.get_dipole_strength()*6, # Built by symmetric stacking of 6 magnets
            axis=np.array([0.0, 0.0, -1.0]), # South pole up dipole
            transform=Transform(Vector3(-30e-3, 0.0, 0.0), UNIT_QUATERNION),
            frame_name="vicon/two_dipole_disc_80x15/DipoleNegX"
        )
    ]
)