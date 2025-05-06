#ifndef OCT_LEVITATION_COMMON_H
#define OCT_LEVITATION_COMMON_H

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <mag_manip/types.h>

namespace oct_levitation {
// TYPEDEFS
typedef Eigen::Vector4d QuaternionVec;
#define MU_0 1.25663706e-6 // Permeability of free space in m kg s^-2 A^-2

// UTILITY FUNCTIONS

double getRingVolume(double ri, double ro, double height) {
    return M_PI * height * (std::pow(ro, 2) - std::pow(ri, 2));
}

double getRingLateralInertia(double ri, double ro, double height, double density) {
    double mass = getRingVolume(ri, ro, height) * density;
    double inertia = (1.0 / 12.0) * mass * ( 3 * (std::pow(ri, 2) + std::pow(ro, 2)) + std::pow(height, 2));
    return inertia;
}

double getRingLongitudinalInertia(double ri, double ro, double height, double density) {
    double mass = getRingVolume(ri, ro, height) * density;
    double inertia = (1.0 / 2.0) * mass * (std::pow(ri, 2) + std::pow(ro, 2));
    return inertia;
}

// STRUCTS
struct RingObject {
    double inner_radius;
    double outer_radius;
    double height;
    double density;
    double mass;
    double Ixxyy;
    double Izz;
    double volume;

    RingObject(double inner_radius, double outer_radius, double height, double density)
        : inner_radius(inner_radius), 
          outer_radius(outer_radius), 
          height(height), 
          density(density) {
        Ixxyy = getRingLateralInertia(inner_radius, outer_radius, height, density);
        Izz = getRingLongitudinalInertia(inner_radius, outer_radius, height, density);
        volume = getRingVolume(inner_radius, outer_radius, height);
        mass = volume * density;
    };
};

struct RingMagnet {
    RingObject geometry;
    double remanence;

    RingMagnet(double inner_radius, double outer_radius, double height, double density, double remanence)
        : geometry(inner_radius, outer_radius, height, density), 
          remanence(remanence) {
        // Constructor body can be empty as all initialization is done in the member initializer list
    }

    double getStrength() const {
        return remanence * geometry.volume / MU_0;
    }
};

struct RingMagnetsRingObject {
    RingMagnet magnet;
    RingObject geometry;
    mag_manip::DipoleVec dipole_axis;
    u_int num_magnets;
    std::vector<double> magnet_z_positions;
    double mass;
    double Ixxyy;
    double Izz;

    RingMagnetsRingObject(const RingMagnet& magnet, 
                          double inner_radius,
                          double outer_radius,
                          double height,
                          double density,
                          int num_magnets,
                          const bool north_pole_down,
                          const std::vector<double>& magnet_z_positions)
        : magnet(magnet), 
         geometry(inner_radius, outer_radius, height, density),
         num_magnets(num_magnets), 
         magnet_z_positions(magnet_z_positions) {

            if(magnet_z_positions.size() != num_magnets) {
                throw std::invalid_argument("Number of z positions must match the number of magnets.");
            }
         
            mass = num_magnets * magnet.geometry.mass + geometry.mass;
            double Ixxyy_magnets = getRingLateralInertia(
                magnet.geometry.inner_radius, 
                magnet.geometry.outer_radius, 
                geometry.height * num_magnets, // HARD ASSUMPTION ON VERTICAL STACKING
                magnet.geometry.density
            );
            double Izz_magnets = getRingLongitudinalInertia(
                magnet.geometry.inner_radius, 
                magnet.geometry.outer_radius, 
                geometry.height * num_magnets, // HARD ASSUMPTION ON VERTICAL STACKING
                magnet.geometry.density
            );

            // Assuming that the centers of magnets and discs are aligned
            // and that the magnets are stacked vertically
            Ixxyy = Ixxyy_magnets + geometry.Ixxyy;
            Izz = Izz_magnets + geometry.Izz;

            if (north_pole_down) {
                dipole_axis = Eigen::Vector3d(0.0, 0.0, -1.0);
            }
            else {
                dipole_axis = Eigen::Vector3d(0.0, 0.0, 1.0);
            }
        }
    
    mag_manip::DipoleVec getLocalDipoleMoment() const {
        return dipole_axis * magnet.getStrength() * num_magnets;
    }
};

} // namespace oct_levitation

#endif // OCT_LEVITATION_COMMON_H