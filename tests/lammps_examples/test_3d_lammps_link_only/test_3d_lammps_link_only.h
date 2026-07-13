#pragma once

#include "lammps_instance.h"
#include "lammps_dem_adapter_common.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace LammpsLinkOnly
{
using SPH::lammps_examples::LammpsInstance;
using SPH::lammps_examples::CouplingAdvanceResult;
using SPH::lammps_examples::ExternalForceBuffer;
using SPH::lammps_examples::LammpsTimeIntegrator;
using SPH::lammps_examples::ParticleForce;
using SPH::lammps_examples::makeCouplingStepPlan;
using SPH::lammps_examples::extract_atom_vector3_by_consecutive_id;
using SPH::lammps_examples::validate_consecutive_atom_ids;
//----------------------------------------------------------------------
//	Single-particle DEM setup.
//----------------------------------------------------------------------
inline constexpr double kGravity = 9.81;
inline constexpr double kDiameter = 0.02;
inline constexpr double kDensity = 2500.0;
inline constexpr double kInitialZ = 1.0;
inline constexpr double kTimeStep = 1.0e-5;
inline constexpr int kTotalSteps = 1000;
inline constexpr int kSampleEvery = 100;
inline constexpr double kErrorTolerance = 1.0e-9;

struct DEMState
{
    std::array<double, 3> center{0.0, 0.0, 0.0};
    std::array<double, 3> velocity{0.0, 0.0, 0.0};
};

struct MotionSample
{
    int number_of_iterations = 0;
    int lammps_step = 0;
    double time = 0.0;
    DEMState dem_state;
    double z_freefall = 0.0;
    double vz_freefall = 0.0;
    double z_minus_freefall = 0.0;
    double vz_minus_freefall = 0.0;
};

inline double sphere_mass()
{
    const double radius = 0.5 * kDiameter;
    return kDensity * (4.0 / 3.0) * 3.1415926535897932384626433832795 * radius * radius * radius;
}

class LammpsDEMAdapter
{
  public:
    LammpsDEMAdapter()
    {
        std::ostringstream cmds;
        cmds << std::setprecision(17)
             << "units si\n"
             << "atom_style sphere\n"
             << "atom_modify map array\n"
             << "boundary f f f\n"
             << "newton off\n"
             << "comm_modify vel yes\n"
             << "region box block -0.1 0.1 -0.1 0.1 -1.0 2.0 units box\n"
             << "create_box 1 box\n"
             << "create_atoms 1 single 0.0 0.0 " << kInitialZ << " units box\n"
             << "set atom 1 diameter " << kDiameter << " density " << kDensity << "\n"
             << "velocity all set 0.0 0.0 0.0 units box\n"
             << "pair_style zero 0.1\n"
             << "pair_coeff * *\n"
             << "neighbor 0.01 bin\n"
             << "neigh_modify delay 0 every 1 check yes\n"
             << "fix int all nve/sphere\n"
             << "fix grav all gravity " << kGravity << " vector 0.0 0.0 -1.0\n"
             << "fix ext all external pf/callback 1 1\n"
             << "timestep " << kTimeStep << "\n"
             << "thermo_style custom step atoms zlo zhi\n"
             << "thermo 100\n";

        lammps_.commands_string(cmds.str(), "LAMMPS initialization commands");
        external_force_.registerFix(lammps_, "ext");
    }

    CouplingAdvanceResult runSubsteps(int steps)
    {
        if (steps <= 0)
        {
            throw std::invalid_argument("LAMMPS substep count must be positive");
        }
        return time_integrator_.advance(
            makeCouplingStepPlan(static_cast<double>(steps) * kTimeStep, kTimeStep));
    }

    DEMState pullState() const
    {
        validate_consecutive_atom_ids(lammps_, 1);
        DEMState state;
        extract_atom_vector3_by_consecutive_id(lammps_, "x", 1, state.center.data());
        extract_atom_vector3_by_consecutive_id(lammps_, "v", 1, state.velocity.data());
        return state;
    }

    int version() const { return lammps_.version(); }

    const ExternalForceBuffer &externalForce() const { return external_force_; }

  private:
    ExternalForceBuffer external_force_;
    LammpsInstance lammps_;
    LammpsTimeIntegrator time_integrator_{lammps_, kTimeStep};
};

inline MotionSample make_motion_sample(int number_of_iterations,
                                       int lammps_step,
                                       const DEMState &dem_state,
                                       double acceleration_z)
{
    MotionSample sample;
    sample.number_of_iterations = number_of_iterations;
    sample.lammps_step = lammps_step;
    sample.time = static_cast<double>(lammps_step) * kTimeStep;
    sample.dem_state = dem_state;
    sample.z_freefall = kInitialZ + 0.5 * acceleration_z * sample.time * sample.time;
    sample.vz_freefall = acceleration_z * sample.time;
    sample.z_minus_freefall = sample.dem_state.center[2] - sample.z_freefall;
    sample.vz_minus_freefall = sample.dem_state.velocity[2] - sample.vz_freefall;
    return sample;
}

inline void write_motion_csv_header(std::ofstream &csv)
{
    csv << "number_of_iterations,lammps_step,time_s,"
           "x_m,y_m,z_m,vx_m_per_s,vy_m_per_s,vz_m_per_s,"
           "z_freefall_m,vz_freefall_m_per_s,z_minus_freefall_m,vz_minus_freefall_m_per_s\n";
}

inline void write_motion_csv_sample(std::ofstream &csv, const MotionSample &sample)
{
    csv << sample.number_of_iterations << ','
        << sample.lammps_step << ','
        << sample.time << ','
        << sample.dem_state.center[0] << ','
        << sample.dem_state.center[1] << ','
        << sample.dem_state.center[2] << ','
        << sample.dem_state.velocity[0] << ','
        << sample.dem_state.velocity[1] << ','
        << sample.dem_state.velocity[2] << ','
        << sample.z_freefall << ','
        << sample.vz_freefall << ','
        << sample.z_minus_freefall << ','
        << sample.vz_minus_freefall << '\n';
}
} // namespace LammpsLinkOnly
