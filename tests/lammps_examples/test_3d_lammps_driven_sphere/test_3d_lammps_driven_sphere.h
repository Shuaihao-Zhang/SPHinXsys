#pragma once

#include "sphinxsys.h"
#include "library.h"

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
#include <vector>

using namespace SPH;

namespace LammpsDrivenSphere
{
//----------------------------------------------------------------------
//	Single-particle DEM setup and SPHinXsys proxy boundary geometry.
//----------------------------------------------------------------------
inline constexpr Real kGravity = 9.81;
inline constexpr Real kDiameter = 0.02;
inline constexpr Real kRadius = 0.5 * kDiameter;
inline constexpr Real kDensity = 2500.0;
inline constexpr Real kTimeStep = 1.0e-5;
inline constexpr int kTotalSteps = 1000;
inline constexpr int kSampleEvery = 100;
inline constexpr Real kErrorTolerance = 1.0e-9;

inline const Vec3d kInitialCenter(0.0, 0.0, 1.0);
inline const BoundingBoxd kSystemDomainBounds(Vec3d(-0.05, -0.05, 0.90),
                                              Vec3d(0.05, 0.05, 1.05));

struct ExternalForce
{
    std::array<double, 3> force{0.0, 0.0, 0.0};
    int callback_calls = 0;
    int atom1_updates = 0;
};

struct DEMState
{
    Vec3d center = Vec3d::Zero();
    Vec3d velocity = Vec3d::Zero();
    Vec3d omega = Vec3d::Zero();
};

struct MotionSample
{
    int number_of_iterations = 0;
    int lammps_step = 0;
    Real time = 0.0;
    DEMState dem_state;
    Vec3d sph_geometric_center = Vec3d::Zero();
    Real center_error = 0.0;
    Real z_freefall = 0.0;
    Real vz_freefall = 0.0;
    Real z_minus_freefall = 0.0;
    Real vz_minus_freefall = 0.0;
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

inline Vec3d to_vec3d(const std::array<double, 3> &values)
{
    return Vec3d(values[0], values[1], values[2]);
}

class LammpsInstance
{
  public:
    LammpsInstance()
    {
        const char *args[] = {"liblammps", "-log", "none", "-screen", "none", "-nocite", nullptr};
        auto **argv = const_cast<char **>(args);
        const int argc = static_cast<int>((sizeof(args) / sizeof(char *)) - 1);

        handle_ = lammps_open_no_mpi(argc, argv, nullptr);
        if (handle_ == nullptr)
        {
            throw std::runtime_error("lammps_open_no_mpi returned null");
        }
    }

    LammpsInstance(const LammpsInstance &) = delete;
    LammpsInstance &operator=(const LammpsInstance &) = delete;

    ~LammpsInstance()
    {
        if (handle_ != nullptr)
        {
            lammps_close(handle_);
        }
    }

    void *get() const { return handle_; }

    int version() const { return lammps_version(handle_); }

    void command(const std::string &cmd, const std::string &context) const
    {
        lammps_command(handle_, cmd.c_str());
        throw_if_error(context);
    }

    void commands_string(const std::string &cmds, const std::string &context) const
    {
        lammps_commands_string(handle_, cmds.c_str());
        throw_if_error(context);
    }

    void throw_if_error(const std::string &context) const
    {
        if (lammps_has_error(handle_) == 0)
        {
            return;
        }

        char buffer[4096] = {};
        lammps_get_last_error_message(handle_, buffer, static_cast<int>(sizeof(buffer)));
        std::ostringstream msg;
        msg << context << " failed: " << buffer;
        throw std::runtime_error(msg.str());
    }

  private:
    void *handle_ = nullptr;
};

inline Real sphere_mass()
{
    return kDensity * (4.0 / 3.0) * Pi * kRadius * kRadius * kRadius;
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
             << "create_atoms 1 single "
             << kInitialCenter[0] << ' '
             << kInitialCenter[1] << ' '
             << kInitialCenter[2] << " units box\n"
             << "set atom 1 diameter " << kDiameter
             << " density " << kDensity << "\n"
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
        std::array<double, 3> x{};
        std::array<double, 3> v{};
        std::array<double, 3> omega{};

        lammps_gather_atoms(lammps_.get(), "x", 1, 3, x.data());
        lammps_.throw_if_error("gather atom positions");
        lammps_gather_atoms(lammps_.get(), "v", 1, 3, v.data());
        lammps_.throw_if_error("gather atom velocities");
        lammps_gather_atoms(lammps_.get(), "omega", 1, 3, omega.data());
        lammps_.throw_if_error("gather atom angular velocities");

        return DEMState{to_vec3d(x), to_vec3d(v), to_vec3d(omega)};
    }

    int version() const { return lammps_.version(); }

    const ExternalForce &externalForce() const { return external_force_; }

  private:
    LammpsInstance lammps_;
    ExternalForce external_force_;
};

//----------------------------------------------------------------------
//	SPHinXsys moving boundary driven by the LAMMPS particle state.
//----------------------------------------------------------------------
class DrivenSphereBoundary
{
  public:
    DrivenSphereBoundary(SolidBody &sphere_body, const Vec3d &initial_center)
        : sphere_body_(sphere_body), particles_(sphere_body.getBaseParticles()),
          pos_(particles_.ParticlePositions()),
          vel_(particles_.registerStateVariableData<Vecd>("Velocity")),
          initial_center_(initial_center)
    {
        if (particles_.TotalRealParticles() == 0)
        {
            throw std::runtime_error("SPHinXsys sphere body generated zero particles");
        }

        const Vec3d centroid_offset = initial_center_ - geometricCenter();
        for (UnsignedInt i = 0; i != particles_.TotalRealParticles(); ++i)
        {
            pos_[i] += centroid_offset;
        }

        relative_positions_.reserve(particles_.TotalRealParticles());
        for (UnsignedInt i = 0; i != particles_.TotalRealParticles(); ++i)
        {
            relative_positions_.push_back(pos_[i] - initial_center_);
            vel_[i] = Vec3d::Zero();
        }
    }

    void update(const DEMState &state)
    {
        for (UnsignedInt i = 0; i != particles_.TotalRealParticles(); ++i)
        {
            pos_[i] = state.center + relative_positions_[i];
            vel_[i] = state.velocity;
        }
        sphere_body_.setNewlyUpdated();
    }

    Vec3d geometricCenter() const
    {
        Vec3d center = Vec3d::Zero();
        for (UnsignedInt i = 0; i != particles_.TotalRealParticles(); ++i)
        {
            center += pos_[i];
        }
        return center / static_cast<Real>(particles_.TotalRealParticles());
    }

    UnsignedInt particleCount() const { return particles_.TotalRealParticles(); }

  private:
    SolidBody &sphere_body_;
    BaseParticles &particles_;
    Vecd *pos_;
    Vecd *vel_;
    Vec3d initial_center_;
    std::vector<Vec3d> relative_positions_;
};

inline MotionSample make_motion_sample(int number_of_iterations,
                                       int lammps_step,
                                       const DEMState &dem_state,
                                       const Vec3d &sph_geometric_center)
{
    MotionSample sample;
    sample.number_of_iterations = number_of_iterations;
    sample.lammps_step = lammps_step;
    sample.time = static_cast<Real>(lammps_step) * kTimeStep;
    sample.dem_state = dem_state;
    sample.sph_geometric_center = sph_geometric_center;
    sample.center_error = (sph_geometric_center - dem_state.center).norm();
    sample.z_freefall = kInitialCenter[2] - 0.5 * kGravity * sample.time * sample.time;
    sample.vz_freefall = -kGravity * sample.time;
    sample.z_minus_freefall = dem_state.center[2] - sample.z_freefall;
    sample.vz_minus_freefall = dem_state.velocity[2] - sample.vz_freefall;
    return sample;
}

inline void write_motion_csv_header(std::ofstream &csv)
{
    csv << "number_of_iterations,lammps_step,time_s,"
           "lammps_x_m,lammps_y_m,lammps_z_m,"
           "lammps_vx_m_per_s,lammps_vy_m_per_s,lammps_vz_m_per_s,"
           "lammps_omega_x_rad_per_s,lammps_omega_y_rad_per_s,lammps_omega_z_rad_per_s,"
           "sph_center_x_m,sph_center_y_m,sph_center_z_m,"
           "center_error_m,z_freefall_m,vz_freefall_m_per_s,z_minus_freefall_m,vz_minus_freefall_m_per_s\n";
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
        << sample.dem_state.omega[0] << ','
        << sample.dem_state.omega[1] << ','
        << sample.dem_state.omega[2] << ','
        << sample.sph_geometric_center[0] << ','
        << sample.sph_geometric_center[1] << ','
        << sample.sph_geometric_center[2] << ','
        << sample.center_error << ','
        << sample.z_freefall << ','
        << sample.vz_freefall << ','
        << sample.z_minus_freefall << ','
        << sample.vz_minus_freefall << '\n';
}
} // namespace LammpsDrivenSphere
