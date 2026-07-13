#pragma once

#include "sphinxsys.h"
#include "lammps_instance.h"
#include "lammps_dem_adapter_common.h"
#include "lammps_io.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace SPH;

namespace LammpsGranularCollapse2D
{
using SPH::lammps_examples::LammpsInstance;
using SPH::lammps_examples::CouplingAdvanceResult;
using SPH::lammps_examples::CouplingStepPlan;
using SPH::lammps_examples::LammpsTimeIntegrator;
using SPH::lammps_examples::makeCouplingStepPlan;
using SPH::lammps_examples::extract_atom_vector3_by_consecutive_id;
using SPH::lammps_examples::validate_consecutive_atom_ids;
using SPH::lammps_examples::VtpPvdWriter;
using SPH::lammps_examples::VtpScalarPointField;
using SPH::lammps_examples::VtpVectorPointField2d;
using SPH::lammps_examples::write_dem_discs_to_vtp;
using SPH::lammps_examples::write_dem_points_fields_to_vtp;

// Keep these DEM parameters aligned with the underwater collapse case.
inline constexpr Real kTankLengthX = 0.16;
inline constexpr Real kTankHeightY = 0.14;
inline constexpr Real kGrainRadius = 0.005;
inline constexpr Real kGrainDiameter = 2.0 * kGrainRadius;
inline constexpr int kColumnCols = 4;
inline constexpr int kColumnRows = 6;
inline constexpr int kParticleCount = kColumnCols * kColumnRows;
inline constexpr Real kInitialWallClearance = 0.1 * (kGrainDiameter / 10.0);
inline constexpr Real kColumnSpacingX = 2.05 * kGrainRadius;
inline constexpr Real kColumnSpacingY = 2.02 * kGrainRadius;
inline constexpr double kDemDt = 1.0e-5;
inline constexpr Real kEndTime = 0.12;
inline constexpr Real kVtpOutputInterval = 0.01;

inline constexpr Real kGrainDensity = 2500.0;
inline constexpr Real kOutOfPlaneDepth = 1.0;
inline constexpr Real kGravity = 9.81;
inline constexpr Real kBottomWallY = 0.0;
inline constexpr Real kLeftWallX = 0.0;
inline constexpr Real kRightWallX = kTankLengthX;
inline constexpr Real kContactNormalStiffness = 5.0e4;
inline constexpr Real kContactRestitution = 0.2;
inline constexpr Real kContactTangentialStiffness = 4.0e4;
inline constexpr Real kContactTangentialDamping = 0.0;
inline constexpr Real kContactFriction = 0.5;

inline constexpr Real kMassRelativeTolerance = 1.0e-12;
inline constexpr Real kMinimumRightFrontDisplacement = 1.0e-4;
inline constexpr Real kMinimumMaxSpeed = 1.0e-3;

struct DEMParticleState
{
    int id = 0;
    Vec2d center = Vec2d::Zero();
    Vec2d velocity = Vec2d::Zero();
    Vec2d net_force = Vec2d::Zero();
    Vec2d contact_force_estimate = Vec2d::Zero();
    Vec2d acceleration = Vec2d::Zero();
    Real omega_z = 0.0;
};

inline std::vector<Vec2d> initial_dem_centers()
{
    std::vector<Vec2d> centers;
    centers.reserve(kParticleCount);
    for (int row = 0; row < kColumnRows; ++row)
    {
        for (int col = 0; col < kColumnCols; ++col)
        {
            const Real stagger = (row % 2 == 0) ? 0.0 : 0.30 * kGrainRadius;
            const Real lean = 0.08 * kGrainRadius * static_cast<Real>(row) / static_cast<Real>(kColumnRows - 1);
            const Real x = kLeftWallX + kInitialWallClearance + kGrainRadius +
                           static_cast<Real>(col) * kColumnSpacingX + stagger + lean;
            const Real y = kBottomWallY + kInitialWallClearance + kGrainRadius +
                           static_cast<Real>(row) * kColumnSpacingY;
            centers.emplace_back(x, y);
        }
    }
    return centers;
}

inline Real grain_area()
{
    return Pi * kGrainRadius * kGrainRadius;
}

inline Real grain_mass()
{
    return kGrainDensity * grain_area() * kOutOfPlaneDepth;
}

inline Real grain_weight()
{
    return grain_mass() * kGravity;
}

inline Vec2d gravity_force()
{
    return Vec2d(0.0, -grain_weight());
}

inline Real lammps_sphere_volume()
{
    return (4.0 / 3.0) * Pi * kGrainRadius * kGrainRadius * kGrainRadius;
}

inline Real lammps_equivalent_sphere_density()
{
    return grain_mass() / lammps_sphere_volume();
}

template <size_t N>
inline Vec2d to_vec2d(const std::array<double, N> &values, int index)
{
    return Vec2d(values[3 * index], values[3 * index + 1]);
}

inline Real initial_right_front_x()
{
    const std::vector<Vec2d> centers = initial_dem_centers();
    Real right_front = -std::numeric_limits<Real>::max();
    for (const Vec2d &center : centers)
    {
        right_front = std::max(right_front, center[0] + kGrainRadius);
    }
    return right_front;
}

class LammpsDEMOnlyColumnAdapter
{
  public:
    LammpsDEMOnlyColumnAdapter()
    {
        const std::vector<Vec2d> centers = initial_dem_centers();
        std::ostringstream cmds;
        cmds << std::setprecision(17)
             << "dimension 2\n"
             << "units si\n"
             << "atom_style sphere\n"
             << "atom_modify map array\n"
             << "boundary f f p\n"
             << "newton off\n"
             << "comm_modify vel yes\n"
             << "region box block -0.02 " << kTankLengthX + 0.02
             << " -0.10 " << kTankHeightY + 0.08
             << " -0.001 0.001 units box\n"
             << "create_box 1 box\n";
        for (const Vec2d &center : centers)
        {
            cmds << "create_atoms 1 single "
                 << center[0] << ' ' << center[1] << " 0.0 units box\n";
        }
        cmds << "set type 1 diameter " << kGrainDiameter
             << " density " << lammps_equivalent_sphere_density() << "\n"
             << "velocity all set 0.0 0.0 0.0 units box\n"
             << "pair_style granular\n"
             << "pair_coeff * * hooke " << kContactNormalStiffness << ' '
             << kContactRestitution
             << " tangential linear_history " << kContactTangentialStiffness << ' '
             << kContactTangentialDamping << ' ' << kContactFriction
             << " damping coeff_restitution\n"
             << "neighbor 0.01 bin\n"
             << "neigh_modify delay 0 every 1 check yes\n"
             << "fix int all nve/sphere\n"
             << "fix grav all gravity " << kGravity << " vector 0.0 -1.0 0.0\n"
             << "fix xwall all wall/gran granular hooke " << kContactNormalStiffness << ' '
             << kContactRestitution
             << " tangential linear_history " << kContactTangentialStiffness << ' '
             << kContactTangentialDamping << ' ' << kContactFriction
             << " damping coeff_restitution xplane " << kLeftWallX << ' ' << kRightWallX << " contacts\n"
             << "fix floor all wall/gran granular hooke " << kContactNormalStiffness << ' '
             << kContactRestitution
             << " tangential linear_history " << kContactTangentialStiffness << ' '
             << kContactTangentialDamping << ' ' << kContactFriction
             << " damping coeff_restitution yplane " << kBottomWallY << " NULL contacts\n"
             << "timestep " << kDemDt << "\n"
             << "thermo 1000000\n";

        lammps_.commands_string(cmds.str(), "LAMMPS DEM-only initialization commands");
        lammps_.command("run 0 post no", "LAMMPS initial force evaluation");
    }

    CouplingStepPlan planCouplingStep(Real duration_limit) const
    {
        return makeCouplingStepPlan(duration_limit, kDemDt);
    }

    CouplingAdvanceResult advance(const CouplingStepPlan &plan)
    {
        return time_integrator_.advance(plan);
    }

    CouplingAdvanceResult advance(const CouplingStepPlan &plan, Real driver_time_before)
    {
        return time_integrator_.advance(plan, driver_time_before);
    }

    int runForDuration(Real duration)
    {
        return advance(planCouplingStep(duration)).dem_steps;
    }

    Real particleMass() const
    {
        auto *rmass = static_cast<double *>(lammps_extract_atom(lammps_.get(), "rmass"));
        lammps_.throw_if_error("extract atom rmass");
        if (rmass == nullptr)
        {
            throw std::runtime_error("lammps_extract_atom returned null for rmass");
        }
        return static_cast<Real>(rmass[0]);
    }

    std::vector<DEMParticleState> pullStates() const
    {
        validate_consecutive_atom_ids(lammps_, kParticleCount);
        std::array<double, 3 * kParticleCount> x{};
        std::array<double, 3 * kParticleCount> v{};
        std::array<double, 3 * kParticleCount> f{};
        std::array<double, 3 * kParticleCount> omega{};

        extract_atom_vector3_by_consecutive_id(lammps_, "x", kParticleCount, x.data());
        extract_atom_vector3_by_consecutive_id(lammps_, "v", kParticleCount, v.data());
        extract_atom_vector3_by_consecutive_id(lammps_, "f", kParticleCount, f.data());
        extract_atom_vector3_by_consecutive_id(lammps_, "omega", kParticleCount, omega.data());

        std::vector<DEMParticleState> states;
        states.reserve(kParticleCount);
        for (int i = 0; i < kParticleCount; ++i)
        {
            DEMParticleState state;
            state.id = i + 1;
            state.center = to_vec2d(x, i);
            state.velocity = to_vec2d(v, i);
            state.net_force = to_vec2d(f, i);
            state.contact_force_estimate = state.net_force - gravity_force();
            state.acceleration = state.net_force / grain_mass();
            state.omega_z = static_cast<Real>(omega[3 * i + 2]);
            states.push_back(state);
        }
        return states;
    }

    int version() const { return lammps_.version(); }

  private:
    LammpsInstance lammps_;
    LammpsTimeIntegrator time_integrator_{lammps_, kDemDt};
};

inline bool is_finite(const Vec2d &value)
{
    return std::isfinite(value[0]) && std::isfinite(value[1]);
}

inline bool all_finite(const std::vector<DEMParticleState> &states)
{
    for (const DEMParticleState &state : states)
    {
        if (!is_finite(state.center) || !is_finite(state.velocity) ||
            !is_finite(state.net_force) || !is_finite(state.contact_force_estimate) ||
            !is_finite(state.acceleration) || !std::isfinite(state.omega_z))
        {
            return false;
        }
    }
    return true;
}

inline Real right_front_x(const std::vector<DEMParticleState> &states)
{
    Real right_front = -std::numeric_limits<Real>::max();
    for (const DEMParticleState &state : states)
    {
        right_front = std::max(right_front, state.center[0] + kGrainRadius);
    }
    return right_front;
}

inline Real max_speed(const std::vector<DEMParticleState> &states)
{
    Real value = 0.0;
    for (const DEMParticleState &state : states)
    {
        value = std::max(value, state.velocity.norm());
    }
    return value;
}

inline Real max_force_norm(const std::vector<Vec2d> &forces)
{
    Real value = 0.0;
    for (const Vec2d &force : forces)
    {
        value = std::max(value, force.norm());
    }
    return value;
}

inline std::vector<Vec2d> centers_from_states(const std::vector<DEMParticleState> &states)
{
    std::vector<Vec2d> centers;
    centers.reserve(states.size());
    for (const DEMParticleState &state : states)
    {
        centers.push_back(state.center);
    }
    return centers;
}

inline std::vector<Vec2d> velocities_from_states(const std::vector<DEMParticleState> &states)
{
    std::vector<Vec2d> velocities;
    velocities.reserve(states.size());
    for (const DEMParticleState &state : states)
    {
        velocities.push_back(state.velocity);
    }
    return velocities;
}

inline std::vector<Vec2d> net_forces_from_states(const std::vector<DEMParticleState> &states)
{
    std::vector<Vec2d> forces;
    forces.reserve(states.size());
    for (const DEMParticleState &state : states)
    {
        forces.push_back(state.net_force);
    }
    return forces;
}

inline std::vector<Vec2d> contact_forces_from_states(const std::vector<DEMParticleState> &states)
{
    std::vector<Vec2d> forces;
    forces.reserve(states.size());
    for (const DEMParticleState &state : states)
    {
        forces.push_back(state.contact_force_estimate);
    }
    return forces;
}

inline std::vector<Real> ids_as_scalars()
{
    std::vector<Real> ids;
    ids.reserve(kParticleCount);
    for (int i = 0; i < kParticleCount; ++i)
    {
        ids.push_back(static_cast<Real>(i + 1));
    }
    return ids;
}

inline std::vector<Real> scalar_filled(Real value)
{
    return std::vector<Real>(kParticleCount, value);
}

inline std::filesystem::path write_dem_force_vtp(int iteration,
                                                 Real time,
                                                 const std::vector<DEMParticleState> &states)
{
    return write_dem_points_fields_to_vtp(
        iteration,
        time,
        "DEM_Forces",
        centers_from_states(states),
        {{"NetForceFromLAMMPS", net_forces_from_states(states)},
         {"ContactForceEstimate", contact_forces_from_states(states)},
         {"Velocity", velocities_from_states(states)}},
        {{"ParticleId", ids_as_scalars()},
         {"Radius", scalar_filled(kGrainRadius)},
         {"MassPerUnitDepth", scalar_filled(grain_mass())}},
        "ContactForceEstimate");
}

inline void write_motion_csv_header(std::ofstream &csv)
{
    csv << "frame,time_s,lammps_step,particle_id,"
           "x_m,y_m,vx_m_per_s,vy_m_per_s,omega_z_rad_per_s,"
           "right_front_x_m\n";
}

inline void write_force_csv_header(std::ofstream &csv)
{
    csv << "frame,time_s,particle_id,"
           "Fx_net_N,Fy_net_N,net_force_norm_N,"
           "Fx_contact_estimate_N,Fy_contact_estimate_N,contact_force_norm_N,"
           "grain_weight_N\n";
}

inline void write_motion_csv_samples(std::ofstream &csv,
                                     int frame,
                                     Real time,
                                     int lammps_step,
                                     const std::vector<DEMParticleState> &states)
{
    const Real right_front = right_front_x(states);
    for (const DEMParticleState &state : states)
    {
        csv << frame << ','
            << time << ','
            << lammps_step << ','
            << state.id << ','
            << state.center[0] << ','
            << state.center[1] << ','
            << state.velocity[0] << ','
            << state.velocity[1] << ','
            << state.omega_z << ','
            << right_front << '\n';
    }
}

inline void write_force_csv_samples(std::ofstream &csv,
                                    int frame,
                                    Real time,
                                    const std::vector<DEMParticleState> &states)
{
    for (const DEMParticleState &state : states)
    {
        csv << frame << ','
            << time << ','
            << state.id << ','
            << state.net_force[0] << ','
            << state.net_force[1] << ','
            << state.net_force.norm() << ','
            << state.contact_force_estimate[0] << ','
            << state.contact_force_estimate[1] << ','
            << state.contact_force_estimate.norm() << ','
            << grain_weight() << '\n';
    }
}
} // namespace LammpsGranularCollapse2D
