#pragma once

#include "lammps_instance.h"

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

struct ExternalForce
{
    std::array<double, 3> force{0.0, 0.0, 0.0};
    int callback_calls = 0;
    int atom1_updates = 0;
};

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

#if defined(LAMMPS_BIGBIG)
using tagint_c = int64_t;
#else
using tagint_c = int;
#endif

extern "C" void external_force_callback(void *ptr,
                                         int64_t /*timestep*/,
                                         int nlocal,
                                         tagint_c *ids,
                                         double ** /*x*/,
                                         double **fexternal)
{
    auto *external = static_cast<ExternalForce *>(ptr);
    ++external->callback_calls;

    for (int i = 0; i < nlocal; ++i)
    {
        fexternal[i][0] = 0.0;
        fexternal[i][1] = 0.0;
        fexternal[i][2] = 0.0;

        if (ids[i] == 1)
        {
            fexternal[i][0] = external->force[0];
            fexternal[i][1] = external->force[1];
            fexternal[i][2] = external->force[2];
            ++external->atom1_updates;
        }
    }
}

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
        lammps_set_fix_external_callback(lammps_.get(), "ext", &external_force_callback, &external_force_);
        lammps_.throw_if_error("lammps_set_fix_external_callback");
    }

    void runSubsteps(int steps)
    {
        if (steps <= 0)
        {
            return;
        }
        lammps_.command("run " + std::to_string(steps) + " post no", "LAMMPS run chunk");
    }

    DEMState pullState() const
    {
        DEMState state;
        lammps_gather_atoms(lammps_.get(), "x", 1, 3, state.center.data());
        lammps_.throw_if_error("gather atom positions");
        lammps_gather_atoms(lammps_.get(), "v", 1, 3, state.velocity.data());
        lammps_.throw_if_error("gather atom velocities");
        return state;
    }

    int version() const { return lammps_.version(); }

    const ExternalForce &externalForce() const { return external_force_; }

  private:
    LammpsInstance lammps_;
    ExternalForce external_force_;
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
