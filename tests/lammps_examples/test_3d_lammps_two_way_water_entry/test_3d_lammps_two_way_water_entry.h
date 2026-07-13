#pragma once

#include "sphinxsys.h"
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
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace SPH;

namespace LammpsTwoWayWaterEntry
{
using SPH::lammps_examples::LammpsInstance;
//----------------------------------------------------------------------
//	Basic geometry parameters and numerical setup.
//----------------------------------------------------------------------
inline constexpr Real kTankLengthX = 0.12;
inline constexpr Real kTankLengthY = 0.12;
inline constexpr Real kTankHeightZ = 0.16;
inline constexpr Real kWaterHeight = 0.12;
inline constexpr Real kSphereRadius = 0.01;
inline constexpr Real kSphereDiameter = 2.0 * kSphereRadius;
inline constexpr Real kParticleSpacing = kSphereDiameter / 10.0;
inline constexpr Real kBoundaryWidth = 4.0 * kParticleSpacing;
inline constexpr Real kInitialClearance = 0.010;
inline constexpr Real kEndTime = 0.06;
inline constexpr Real kVtpOutputInterval = 0.01;
inline constexpr double kDemMaxDt = 1.0e-5;
inline constexpr int kRelaxationSteps = 1000;
inline constexpr int kRelaxationOutputInterval = 200;
inline constexpr Real kForceRelaxationAlpha = 0.3;
inline constexpr Real kForceCapWeightFactor = 5.0;
inline constexpr Real kWallXMin = 0.0;
inline constexpr Real kWallXMax = kTankLengthX;
inline constexpr Real kWallYMin = 0.0;
inline constexpr Real kWallYMax = kTankLengthY;
inline constexpr Real kBottomWallZ = 0.0;
inline constexpr Real kContactNormalStiffness = 5.0e4;
inline constexpr Real kContactRestitution = 0.2;
inline constexpr Real kContactTangentialStiffness = 4.0e4;
inline constexpr Real kContactTangentialDamping = 0.0;
inline constexpr Real kContactFriction = 0.5;
//----------------------------------------------------------------------
//	Material parameters.
//----------------------------------------------------------------------
inline constexpr Real kWaterDensity = 1000.0;
inline constexpr Real kSphereDensity = 2500.0;
inline constexpr Real kGravity = 9.81;
inline constexpr Real kDynamicViscosity = 1.0e-3;
inline const Real kCharacteristicVelocity = 2.0 * std::sqrt(kGravity * kWaterHeight);
inline const Real kSoundSpeed = 10.0 * kCharacteristicVelocity;
//----------------------------------------------------------------------
//	Validation parameters.
//----------------------------------------------------------------------
inline constexpr Real kPreEntryClearanceForStats = 4.0 * kParticleSpacing;
inline constexpr Real kPostEntryDepthForStats = 0.5 * kParticleSpacing;
inline constexpr Real kCenterErrorTolerance = 1.0e-10;
inline constexpr Real kPreEntryForceNormTolerance = 1.0e-3;
inline constexpr Real kMinimumPostEntryFz = 1.0e-2;
inline constexpr Real kMinimumTrajectoryDifference = 1.0e-5;
inline const std::string kRelaxedSphereReloadBodyName = "FixedSphereBoundary";
//----------------------------------------------------------------------
//	Geometric shapes used in this case.
//----------------------------------------------------------------------
inline const Vec3d kInitialCenter(0.5 * kTankLengthX,
                                  0.5 * kTankLengthY,
                                  kWaterHeight + kSphereRadius + kInitialClearance);
inline const BoundingBoxd kSystemDomainBounds(
    Vec3d(-kBoundaryWidth, -kBoundaryWidth, -kBoundaryWidth),
    Vec3d(kTankLengthX + kBoundaryWidth, kTankLengthY + kBoundaryWidth, kTankHeightZ + kBoundaryWidth));

class WaterBlock : public ComplexShape
{
  public:
    explicit WaterBlock(const std::string &shape_name) : ComplexShape(shape_name)
    {
        add<GeometricShapeBox>(
            Transform(Vec3d(0.5 * kTankLengthX, 0.5 * kTankLengthY, 0.5 * kWaterHeight)),
            Vec3d(0.5 * kTankLengthX, 0.5 * kTankLengthY, 0.5 * kWaterHeight),
            "WaterBox");
    }
};

class WallBoundary : public ComplexShape
{
  public:
    explicit WallBoundary(const std::string &shape_name) : ComplexShape(shape_name)
    {
        add<GeometricShapeBox>(
            Transform(Vec3d(0.5 * kTankLengthX, 0.5 * kTankLengthY, 0.5 * kTankHeightZ)),
            Vec3d(0.5 * kTankLengthX + kBoundaryWidth,
                  0.5 * kTankLengthY + kBoundaryWidth,
                  0.5 * kTankHeightZ + kBoundaryWidth),
            "OuterTank");
        subtract<GeometricShapeBox>(
            Transform(Vec3d(0.5 * kTankLengthX, 0.5 * kTankLengthY, 0.5 * kTankHeightZ + kBoundaryWidth)),
            Vec3d(0.5 * kTankLengthX,
                  0.5 * kTankLengthY,
                  0.5 * kTankHeightZ + kBoundaryWidth),
            "InnerTankVoid");
    }
};
//----------------------------------------------------------------------
//	Initial hydrostatic field for the water body.
//----------------------------------------------------------------------
class HydrostaticPressureField : public fluid_dynamics::FluidInitialCondition
{
  public:
    explicit HydrostaticPressureField(SPHBody &sph_body)
        : fluid_dynamics::FluidInitialCondition(sph_body),
          rho_(particles_->getVariableDataByName<Real>("Density")),
          mass_(particles_->getVariableDataByName<Real>("Mass")),
          pressure_(particles_->registerStateVariableData<Real>("Pressure")) {}

    void update(size_t index_i, Real = 0.0)
    {
        const Real pressure = kWaterDensity * kGravity * SMAX(Real(0), kWaterHeight - pos_[index_i][2]);
        pressure_[index_i] = pressure;
        rho_[index_i] = kWaterDensity + pressure / (kSoundSpeed * kSoundSpeed);
        mass_[index_i] = rho_[index_i] * particles_->ParticleVolume(index_i);
        vel_[index_i] = Vecd::Zero();
    }

  private:
    Real *rho_;
    Real *mass_;
    Real *pressure_;
};
//----------------------------------------------------------------------
//	LAMMPS one-particle DEM adapter.
//----------------------------------------------------------------------
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
             << "region box block -0.02 " << kTankLengthX + 0.02
             << " -0.02 " << kTankLengthY + 0.02
             << " -0.05 " << kTankHeightZ + 0.08 << " units box\n"
             << "create_box 1 box\n"
             << "create_atoms 1 single "
             << kInitialCenter[0] << ' '
             << kInitialCenter[1] << ' '
             << kInitialCenter[2] << " units box\n"
             << "set atom 1 diameter " << kSphereDiameter
             << " density " << kSphereDensity << "\n"
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
             << "fix grav all gravity " << kGravity << " vector 0.0 0.0 -1.0\n"
             << "fix ext all external pf/callback 1 1\n"
             << "fix xwall all wall/gran granular hooke " << kContactNormalStiffness << ' '
             << kContactRestitution
             << " tangential linear_history " << kContactTangentialStiffness << ' '
             << kContactTangentialDamping << ' ' << kContactFriction
             << " damping coeff_restitution xplane " << kWallXMin << ' ' << kWallXMax << " contacts\n"
             << "fix ywall all wall/gran granular hooke " << kContactNormalStiffness << ' '
             << kContactRestitution
             << " tangential linear_history " << kContactTangentialStiffness << ' '
             << kContactTangentialDamping << ' ' << kContactFriction
             << " damping coeff_restitution yplane " << kWallYMin << ' ' << kWallYMax << " contacts\n"
             << "fix floor all wall/gran granular hooke " << kContactNormalStiffness << ' '
             << kContactRestitution
             << " tangential linear_history " << kContactTangentialStiffness << ' '
             << kContactTangentialDamping << ' ' << kContactFriction
             << " damping coeff_restitution zplane " << kBottomWallZ << " NULL contacts\n"
             << "timestep " << kDemMaxDt << "\n"
             << "thermo 1000000\n";

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

    void setTimestep(double dem_timestep)
    {
        std::ostringstream cmd;
        cmd << std::setprecision(17) << "timestep " << dem_timestep;
        lammps_.command(cmd.str(), "LAMMPS timestep update");
    }

    // Synchronize LAMMPS to one SPH acoustic step. Most DEM substeps use
    // kDemMaxDt, and a final short step removes the remaining time mismatch.
    int runForDuration(Real acoustic_step)
    {
        if (acoustic_step <= TinyReal)
        {
            return 0;
        }

        const int full_steps = static_cast<int>(std::floor(acoustic_step / kDemMaxDt));
        const Real remainder = acoustic_step - static_cast<Real>(full_steps) * kDemMaxDt;
        int executed_steps = 0;

        // Run most of the acoustic interval with the nominal DEM timestep.
        if (full_steps > 0)
        {
            setTimestep(kDemMaxDt);
            runSubsteps(full_steps);
            executed_steps += full_steps;
        }

        // Use one short tail step so the LAMMPS time exactly matches the SPH acoustic step.
        if (remainder > TinyReal)
        {
            setTimestep(remainder);
            runSubsteps(1);
            setTimestep(kDemMaxDt);
            executed_steps += 1;
        }

        return executed_steps;
    }

    void setExternalForce(const Vec3d &force)
    {
        // This value is consumed by fix external pf/callback during LAMMPS run.
        external_force_.force[0] = force[0];
        external_force_.force[1] = force[1];
        external_force_.force[2] = force[2];
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
//	two-way moving boundary driven by the LAMMPS particle state.
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
//----------------------------------------------------------------------
//	Diagnostics and small case-local utilities.
//----------------------------------------------------------------------
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

struct ForceSample
{
    int number_of_iterations = 0;
    Real time = 0.0;
    Vec3d raw_force = Vec3d::Zero();
    Vec3d applied_force = Vec3d::Zero();
    Real raw_force_norm = 0.0;
    Real applied_force_norm = 0.0;
    Real sphere_bottom_z = 0.0;
    bool pre_entry_stat = false;
    bool post_entry_stat = false;
    bool capped = false;
};

struct ForceStats
{
    Vec3d pre_entry_sum = Vec3d::Zero();
    Real pre_entry_norm_sum = 0.0;
    int pre_entry_count = 0;
    Real max_raw_fz = -std::numeric_limits<Real>::max();
    Real max_applied_fz = -std::numeric_limits<Real>::max();
    int post_entry_count = 0;
    int cap_count = 0;

    void add(const ForceSample &sample)
    {
        if (sample.pre_entry_stat)
        {
            pre_entry_sum += sample.applied_force;
            pre_entry_norm_sum += sample.applied_force_norm;
            ++pre_entry_count;
        }
        if (sample.post_entry_stat)
        {
            max_raw_fz = std::max(max_raw_fz, sample.raw_force[2]);
            max_applied_fz = std::max(max_applied_fz, sample.applied_force[2]);
            ++post_entry_count;
        }
        if (sample.capped)
        {
            ++cap_count;
        }
    }

    Vec3d pre_entry_mean_force() const
    {
        if (pre_entry_count == 0)
        {
            return Vec3d::Zero();
        }
        return pre_entry_sum / static_cast<Real>(pre_entry_count);
    }

    Real pre_entry_mean_force_norm() const
    {
        if (pre_entry_count == 0)
        {
            return 0.0;
        }
        return pre_entry_norm_sum / static_cast<Real>(pre_entry_count);
    }
};

inline std::filesystem::path reload_particle_file()
{
    return std::filesystem::path(IO::getEnvironment().ReloadFolder()) / "Reload.xml";
}

inline bool reload_particle_file_exists()
{
    return std::filesystem::exists(reload_particle_file());
}

inline std::filesystem::path fixed_sphere_reload_file()
{
    return std::filesystem::current_path().parent_path().parent_path() /
           "test_3d_fixed_sphere_hydro_force" / "bin" / "reload" / "Reload.xml";
}

inline void import_fixed_sphere_reload_if_available()
{
    const std::filesystem::path local_reload = reload_particle_file();
    if (std::filesystem::exists(local_reload))
    {
        return;
    }

    const std::filesystem::path fixed_reload = fixed_sphere_reload_file();
    if (!std::filesystem::exists(fixed_reload))
    {
        return;
    }

    std::filesystem::create_directories(local_reload.parent_path());
    std::filesystem::copy_file(fixed_reload, local_reload, std::filesystem::copy_options::overwrite_existing);
}

inline Real sphere_volume()
{
    return (4.0 / 3.0) * Pi * kSphereRadius * kSphereRadius * kSphereRadius;
}

inline Real sphere_mass()
{
    return kSphereDensity * sphere_volume();
}

inline Real sphere_weight()
{
    return sphere_mass() * kGravity;
}

inline Real water_entry_time()
{
    return std::sqrt(2.0 * kInitialClearance / kGravity);
}

inline Real bottom_wall_overlap(const Vec3d &center)
{
    return SMAX(Real(0), kBottomWallZ - (center[2] - kSphereRadius));
}

inline Real tank_wall_overlap(const Vec3d &center)
{
    const Real xlo_overlap = SMAX(Real(0), kWallXMin - (center[0] - kSphereRadius));
    const Real xhi_overlap = SMAX(Real(0), (center[0] + kSphereRadius) - kWallXMax);
    const Real ylo_overlap = SMAX(Real(0), kWallYMin - (center[1] - kSphereRadius));
    const Real yhi_overlap = SMAX(Real(0), (center[1] + kSphereRadius) - kWallYMax);
    return SMAX(bottom_wall_overlap(center), SMAX(SMAX(xlo_overlap, xhi_overlap), SMAX(ylo_overlap, yhi_overlap)));
}

inline bool is_finite(const Vec3d &value)
{
    return std::isfinite(value[0]) && std::isfinite(value[1]) && std::isfinite(value[2]);
}

inline std::string generate_sphere_boundary_particles(SPHSystem &sph_system, SolidBody &sphere_boundary)
{
    if (!sph_system.RunParticleRelaxation() && sph_system.ReloadParticles() && reload_particle_file_exists())
    {
        sphere_boundary.generateParticles<BaseParticles, Reload>(kRelaxedSphereReloadBodyName);
        return "Reload";
    }

    sphere_boundary.defineBodyLevelSetShape().writeLevelSet();
    sphere_boundary.generateParticles<BaseParticles, Lattice>();
    return "Lattice";
}

inline int run_sphere_particle_relaxation(SolidBody &sphere_boundary)
{
    InnerRelation sphere_inner(sphere_boundary);

    using namespace relax_dynamics;
    SimpleDynamics<RandomizeParticlePosition> random_sphere_particles(sphere_boundary);
    RelaxationStepInner relaxation_step_inner(sphere_inner);
    BodyStatesRecordingToVtp write_sphere_state(sphere_boundary);
    ReloadParticleIO write_particle_reload_files(sphere_boundary, kRelaxedSphereReloadBodyName);

    random_sphere_particles.exec(0.25);
    relaxation_step_inner.SurfaceBounding().exec();
    write_sphere_state.writeToFile(0);

    int ite_p = 0;
    while (ite_p < kRelaxationSteps)
    {
        relaxation_step_inner.exec();
        ite_p += 1;
        if (ite_p % kRelaxationOutputInterval == 0)
        {
            std::cout << std::fixed << std::setprecision(9)
                      << "Relaxation steps for the LAMMPS-driven sphere boundary N = "
                      << ite_p << "\n";
            write_sphere_state.writeToFile(ite_p);
        }
    }

    write_particle_reload_files.writeToFile(0);
    const std::filesystem::path reload_path = reload_particle_file();
    if (!std::filesystem::exists(reload_path))
    {
        throw std::runtime_error("particle relaxation did not write reload/Reload.xml");
    }

    std::cout << std::setprecision(17);
    std::cout << "SPHinXsys LAMMPS two-way water-entry sphere particle relaxation\n";
    std::cout << "sphere_particles: " << sphere_boundary.getBaseParticles().TotalRealParticles() << '\n';
    std::cout << "particle_spacing_m: " << kParticleSpacing << '\n';
    std::cout << "relaxation_steps: " << kRelaxationSteps << '\n';
    std::cout << "reload_body_name: " << kRelaxedSphereReloadBodyName << '\n';
    std::cout << "reload_file: " << std::filesystem::absolute(reload_path).string() << '\n';
    std::cout << "VTP_output_folder: " << std::filesystem::absolute(IO::getEnvironment().OutputFolder()).string() << '\n';
    std::cout << "status: RELAXATION_PASS\n";
    return 0;
}

inline void update_water_sphere_configuration(FluidBody &water_block,
                                              SolidBody &sphere_boundary,
                                              ComplexRelation &water_complex,
                                              ContactRelation &sphere_contact)
{
    water_block.updateCellLinkedList();
    sphere_boundary.updateCellLinkedList();
    water_complex.updateConfiguration();
    sphere_contact.updateConfiguration();
}

inline Vec3d sum_sphere_hydro_force(SPHBody &sphere)
{
    BaseParticles &particles = sphere.getBaseParticles();
    Vecd *pressure_force = particles.getVariableDataByName<Vecd>("PressureForceFromFluid");
    Vecd *viscous_force = particles.getVariableDataByName<Vecd>("ViscousForceFromFluid");

    Vec3d total = Vec3d::Zero();
    for (UnsignedInt i = 0; i != particles.TotalRealParticles(); ++i)
    {
        const Vecd particle_force = pressure_force[i] + viscous_force[i];
        total += Vec3d(particle_force[0], particle_force[1], particle_force[2]);
    }
    return total;
}

struct ForceApplication
{
    Vec3d applied_force = Vec3d::Zero();
    bool capped = false;
};

inline ForceApplication relax_and_cap_force(const Vec3d &raw_force, const Vec3d &previous_applied_force)
{
    ForceApplication result;
    Vec3d relaxed_force =
        kForceRelaxationAlpha * raw_force + (1.0 - kForceRelaxationAlpha) * previous_applied_force;

    const Real cap = kForceCapWeightFactor * sphere_weight();
    const Real relaxed_norm = relaxed_force.norm();
    if (relaxed_norm > cap && relaxed_norm > TinyReal)
    {
        relaxed_force *= cap / relaxed_norm;
        result.capped = true;
    }

    result.applied_force = relaxed_force;
    return result;
}

inline MotionSample make_motion_sample(int number_of_iterations,
                                       int lammps_step,
                                       Real time,
                                       const DEMState &dem_state,
                                       const Vec3d &sph_geometric_center)
{
    MotionSample sample;
    sample.number_of_iterations = number_of_iterations;
    sample.lammps_step = lammps_step;
    sample.time = time;
    sample.dem_state = dem_state;
    sample.sph_geometric_center = sph_geometric_center;
    sample.center_error = (sph_geometric_center - dem_state.center).norm();
    sample.z_freefall = kInitialCenter[2] - 0.5 * kGravity * sample.time * sample.time;
    sample.vz_freefall = -kGravity * sample.time;
    sample.z_minus_freefall = dem_state.center[2] - sample.z_freefall;
    sample.vz_minus_freefall = dem_state.velocity[2] - sample.vz_freefall;
    return sample;
}

inline ForceSample make_force_sample(int number_of_iterations,
                                     Real time,
                                     const DEMState &dem_state,
                                     const Vec3d &raw_force,
                                     const Vec3d &applied_force,
                                     bool capped)
{
    ForceSample sample;
    sample.number_of_iterations = number_of_iterations;
    sample.time = time;
    sample.raw_force = raw_force;
    sample.applied_force = applied_force;
    sample.raw_force_norm = raw_force.norm();
    sample.applied_force_norm = applied_force.norm();
    sample.sphere_bottom_z = dem_state.center[2] - kSphereRadius;
    sample.pre_entry_stat = sample.sphere_bottom_z > kWaterHeight + kPreEntryClearanceForStats;
    sample.post_entry_stat = sample.sphere_bottom_z < kWaterHeight - kPostEntryDepthForStats;
    sample.capped = capped;
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

inline void write_force_csv_header(std::ofstream &csv)
{
    csv << "number_of_iterations,time_s,"
           "Fx_raw_N,Fy_raw_N,Fz_raw_N,raw_force_norm_N,"
           "Fx_applied_N,Fy_applied_N,Fz_applied_N,applied_force_norm_N,"
           "sphere_weight_N,force_cap_N,capped,"
           "sphere_bottom_z_m,geometric_entry,pre_entry_stat,post_entry_stat\n";
}

inline void write_force_csv_sample(std::ofstream &csv, const ForceSample &sample)
{
    const bool geometric_entry = sample.sphere_bottom_z <= kWaterHeight;
    csv << sample.number_of_iterations << ','
        << sample.time << ','
        << sample.raw_force[0] << ','
        << sample.raw_force[1] << ','
        << sample.raw_force[2] << ','
        << sample.raw_force_norm << ','
        << sample.applied_force[0] << ','
        << sample.applied_force[1] << ','
        << sample.applied_force[2] << ','
        << sample.applied_force_norm << ','
        << sphere_weight() << ','
        << kForceCapWeightFactor * sphere_weight() << ','
        << (sample.capped ? 1 : 0) << ','
        << sample.sphere_bottom_z << ','
        << (geometric_entry ? 1 : 0) << ','
        << (sample.pre_entry_stat ? 1 : 0) << ','
        << (sample.post_entry_stat ? 1 : 0) << '\n';
}
} // namespace LammpsTwoWayWaterEntry
