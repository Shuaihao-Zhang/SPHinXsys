#pragma once

#include <array>

namespace SPH
{
namespace lammps
{
inline std::array<double, 2> rigidSurfaceVelocity2d(
    const std::array<double, 2> &center_velocity, double omega_z,
    const std::array<double, 2> &relative_position)
{
    return {center_velocity[0] - omega_z * relative_position[1],
            center_velocity[1] + omega_z * relative_position[0]};
}

inline std::array<double, 3> crossProduct3d(const std::array<double, 3> &left,
                                           const std::array<double, 3> &right)
{
    return {left[1] * right[2] - left[2] * right[1],
            left[2] * right[0] - left[0] * right[2],
            left[0] * right[1] - left[1] * right[0]};
}

inline std::array<double, 3> rigidSurfaceVelocity3d(
    const std::array<double, 3> &center_velocity,
    const std::array<double, 3> &angular_velocity,
    const std::array<double, 3> &relative_position)
{
    const std::array<double, 3> rotational_velocity =
        crossProduct3d(angular_velocity, relative_position);
    return {center_velocity[0] + rotational_velocity[0],
            center_velocity[1] + rotational_velocity[1],
            center_velocity[2] + rotational_velocity[2]};
}

inline double hydrodynamicTorque2d(const std::array<double, 2> &relative_position,
                                   const std::array<double, 2> &force)
{
    return relative_position[0] * force[1] - relative_position[1] * force[0];
}

inline std::array<double, 3> hydrodynamicTorque3d(
    const std::array<double, 3> &relative_position,
    const std::array<double, 3> &force)
{
    return crossProduct3d(relative_position, force);
}
} // namespace lammps
} // namespace SPH
