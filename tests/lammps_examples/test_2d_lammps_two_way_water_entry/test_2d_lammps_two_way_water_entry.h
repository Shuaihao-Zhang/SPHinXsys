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
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace SPH;

namespace LammpsTwoWayWaterEntry2D
{
//----------------------------------------------------------------------
//	Basic geometry parameters and numerical setup.
//----------------------------------------------------------------------
inline constexpr Real kTankLengthX = 0.12;
inline constexpr Real kTankHeightY = 0.16;
inline constexpr Real kWaterHeight = 0.12;
inline constexpr Real kCylinderRadius = 0.01;
inline constexpr Real kCylinderDiameter = 2.0 * kCylinderRadius;
inline constexpr Real kParticleSpacing = kCylinderDiameter / 10.0;
inline constexpr Real kBoundaryWidth = 4.0 * kParticleSpacing;
inline constexpr Real kInitialClearance = 0.010;
inline constexpr Real kEndTime = 1.0;
inline constexpr Real kVtpOutputInterval = 0.01;
inline constexpr double kDemMaxDt = 1.0e-5;
inline constexpr int kRelaxationSteps = 1000;
inline constexpr int kRelaxationOutputInterval = 200;
inline constexpr Real kForceRelaxationAlpha = 0.3;
inline constexpr Real kForceCapWeightFactor = 5.0;
//----------------------------------------------------------------------
//	Material parameters.
//----------------------------------------------------------------------
inline constexpr Real kWaterDensity = 1000.0;
inline constexpr Real kCylinderDensity = 2500.0;
inline constexpr Real kGravity = 9.81;
inline constexpr Real kDynamicViscosity = 1.0e-3;
inline const Real kCharacteristicVelocity = 2.0 * std::sqrt(kGravity * kWaterHeight);
inline const Real kSoundSpeed = 10.0 * kCharacteristicVelocity;
//----------------------------------------------------------------------
//	Validation parameters.
//----------------------------------------------------------------------
inline constexpr Real kPreEntryClearanceForStats = 4.0 * kParticleSpacing;
inline constexpr Real kPostEntryDepthForStats = 0.0;
inline constexpr Real kCenterErrorTolerance = 1.0e-10;
inline constexpr Real kPreEntryForceNormTolerance = 1.0e-3;
inline constexpr Real kMinimumPostEntryFy = 1.0e-2;
inline constexpr Real kMinimumTrajectoryDifference = 1.0e-5;
inline constexpr Real kMassRelativeTolerance = 1.0e-12;
inline constexpr Real kMinimumFinalTopSubmergence = 2.0 * kParticleSpacing;
inline const std::string kRelaxedCylinderReloadBodyName = "LammpsTwoWayWaterEntryCylinder";
//----------------------------------------------------------------------
//	Geometric shapes used in this case.
//----------------------------------------------------------------------
inline const Vec2d kInitialCenter(0.5 * kTankLengthX,
                                  kWaterHeight + kCylinderRadius + kInitialClearance);
inline const BoundingBoxd kSystemDomainBounds(
    Vec2d(-kBoundaryWidth, -kBoundaryWidth),
    Vec2d(kTankLengthX + kBoundaryWidth, kTankHeightY + kBoundaryWidth));

class WaterBlock : public ComplexShape
{
  public:
    explicit WaterBlock(const std::string &shape_name) : ComplexShape(shape_name)
    {
        add<GeometricShapeBox>(
            Transform(Vec2d(0.5 * kTankLengthX, 0.5 * kWaterHeight)),
            Vec2d(0.5 * kTankLengthX, 0.5 * kWaterHeight),
            "WaterBox");
    }
};

class WallBoundary : public ComplexShape
{
  public:
    explicit WallBoundary(const std::string &shape_name) : ComplexShape(shape_name)
    {
        add<GeometricShapeBox>(
            Transform(Vec2d(0.5 * kTankLengthX, 0.5 * kTankHeightY)),
            Vec2d(0.5 * kTankLengthX + kBoundaryWidth,
                  0.5 * kTankHeightY + kBoundaryWidth),
            "OuterTank");
        subtract<GeometricShapeBox>(
            Transform(Vec2d(0.5 * kTankLengthX, 0.5 * kTankHeightY + kBoundaryWidth)),
            Vec2d(0.5 * kTankLengthX,
                  0.5 * kTankHeightY + kBoundaryWidth),
            "InnerTankVoid");
    }
};

class CylinderBoundaryShape : public MultiPolygonShape
{
  public:
    explicit CylinderBoundaryShape(const std::string &shape_name) : MultiPolygonShape(shape_name)
    {
        multi_polygon_.addCircle(kInitialCenter, kCylinderRadius, 100, GeometricOps::add);
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
        const Real pressure = kWaterDensity * kGravity * SMAX(Real(0), kWaterHeight - pos_[index_i][1]);
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
//	LAMMPS one-disc DEM adapter. LAMMPS remains 3-component internally,
//	but this example maps SPHinXsys (x, y) to LAMMPS (x, y, z = 0).
//----------------------------------------------------------------------
struct ExternalForce
{
    std::array<double, 3> force{0.0, 0.0, 0.0};
    int callback_calls = 0;
    int atom1_updates = 0;
};

struct DEMState
{
    Vec2d center = Vec2d::Zero();
    Vec2d velocity = Vec2d::Zero();
    Vec2d acceleration = Vec2d::Zero();
    Real omega_z = 0.0;
};

inline Real cylinder_area()
{
    return Pi * kCylinderRadius * kCylinderRadius;
}

inline Real cylinder_mass()
{
    return kCylinderDensity * cylinder_area();
}

inline Real cylinder_weight()
{
    return cylinder_mass() * kGravity;
}

inline Real lammps_sphere_volume()
{
    return (4.0 / 3.0) * Pi * kCylinderRadius * kCylinderRadius * kCylinderRadius;
}

inline Real lammps_equivalent_sphere_density()
{
    return cylinder_mass() / lammps_sphere_volume();
}

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
            fexternal[i][2] = 0.0;
            ++external->atom1_updates;
        }
    }
}

inline Vec2d to_vec2d(const std::array<double, 3> &values)
{
    return Vec2d(values[0], values[1]);
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

class LammpsDEMAdapter
{
  public:
    LammpsDEMAdapter()
    {
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
             << " -0.05 " << kTankHeightY + 0.08
             << " -0.001 0.001 units box\n"
             << "create_box 1 box\n"
             << "create_atoms 1 single "
             << kInitialCenter[0] << ' '
             << kInitialCenter[1] << " 0.0 units box\n"
             << "set atom 1 diameter " << kCylinderDiameter
             << " density " << lammps_equivalent_sphere_density() << "\n"
             << "velocity all set 0.0 0.0 0.0 units box\n"
             << "pair_style zero 0.1\n"
             << "pair_coeff * *\n"
             << "neighbor 0.01 bin\n"
             << "neigh_modify delay 0 every 1 check yes\n"
             << "fix int all nve/sphere\n"
             << "fix grav all gravity " << kGravity << " vector 0.0 -1.0 0.0\n"
             << "fix ext all external pf/callback 1 1\n"
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

    // Keep LAMMPS and SPHinXsys synchronized over one SPH acoustic step.
    int runForDuration(Real acoustic_step)
    {
        if (acoustic_step <= TinyReal)
        {
            return 0;
        }

        const int full_steps = static_cast<int>(std::floor(acoustic_step / kDemMaxDt));
        const Real remainder = acoustic_step - static_cast<Real>(full_steps) * kDemMaxDt;
        int executed_steps = 0;

        if (full_steps > 0)
        {
            setTimestep(kDemMaxDt);
            runSubsteps(full_steps);
            executed_steps += full_steps;
        }

        if (remainder > TinyReal)
        {
            setTimestep(remainder);
            runSubsteps(1);
            setTimestep(kDemMaxDt);
            executed_steps += 1;
        }

        return executed_steps;
    }

    void setExternalForce(const Vec2d &force)
    {
        // This is the hydrodynamic force only. LAMMPS supplies gravity through fix gravity.
        external_force_.force[0] = force[0];
        external_force_.force[1] = force[1];
        external_force_.force[2] = 0.0;
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

        const Vec2d acceleration(
            external_force_.force[0] / cylinder_mass(),
            external_force_.force[1] / cylinder_mass() - kGravity);

        return DEMState{to_vec2d(x), to_vec2d(v), acceleration, static_cast<Real>(omega[2])};
    }

    int version() const { return lammps_.version(); }

    const ExternalForce &externalForce() const { return external_force_; }

  private:
    LammpsInstance lammps_;
    ExternalForce external_force_;
};
//----------------------------------------------------------------------
//	Two-way moving boundary driven by the LAMMPS disc state.
//----------------------------------------------------------------------
class DrivenCylinderBoundary
{
  public:
    DrivenCylinderBoundary(SolidBody &cylinder_body, const Vec2d &initial_center)
        : cylinder_body_(cylinder_body), particles_(cylinder_body.getBaseParticles()),
          pos_(particles_.ParticlePositions()),
          vel_(particles_.registerStateVariableData<Vecd>("Velocity")),
          acc_(particles_.registerStateVariableData<Vecd>("Acceleration")),
          initial_center_(initial_center)
    {
        if (particles_.TotalRealParticles() == 0)
        {
            throw std::runtime_error("SPHinXsys cylinder body generated zero particles");
        }

        const Vec2d centroid_offset = initial_center_ - geometricCenter();
        for (UnsignedInt i = 0; i != particles_.TotalRealParticles(); ++i)
        {
            pos_[i] += centroid_offset;
        }

        relative_positions_.reserve(particles_.TotalRealParticles());
        for (UnsignedInt i = 0; i != particles_.TotalRealParticles(); ++i)
        {
            relative_positions_.push_back(pos_[i] - initial_center_);
            vel_[i] = Vec2d::Zero();
            acc_[i] = Vec2d::Zero();
        }
    }

    void update(const DEMState &state)
    {
        for (UnsignedInt i = 0; i != particles_.TotalRealParticles(); ++i)
        {
            pos_[i] = state.center + relative_positions_[i];
            vel_[i] = state.velocity;
            acc_[i] = state.acceleration;
        }
        cylinder_body_.setNewlyUpdated();
    }

    Vec2d geometricCenter() const
    {
        Vec2d center = Vec2d::Zero();
        for (UnsignedInt i = 0; i != particles_.TotalRealParticles(); ++i)
        {
            center += pos_[i];
        }
        return center / static_cast<Real>(particles_.TotalRealParticles());
    }

    UnsignedInt particleCount() const { return particles_.TotalRealParticles(); }

  private:
    SolidBody &cylinder_body_;
    BaseParticles &particles_;
    Vecd *pos_;
    Vecd *vel_;
    Vecd *acc_;
    Vec2d initial_center_;
    std::vector<Vec2d> relative_positions_;
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
    Vec2d sph_geometric_center = Vec2d::Zero();
    Real center_error = 0.0;
    Real y_freefall = 0.0;
    Real vy_freefall = 0.0;
    Real y_minus_freefall = 0.0;
    Real vy_minus_freefall = 0.0;
};

struct ForceSample
{
    int number_of_iterations = 0;
    Real time = 0.0;
    Vec2d raw_force = Vec2d::Zero();
    Vec2d applied_force = Vec2d::Zero();
    Real raw_force_norm = 0.0;
    Real applied_force_norm = 0.0;
    Real cylinder_bottom_y = 0.0;
    bool pre_entry_stat = false;
    bool post_entry_stat = false;
    bool capped = false;
};

struct ForceStats
{
    Vec2d pre_entry_sum = Vec2d::Zero();
    Real pre_entry_norm_sum = 0.0;
    int pre_entry_count = 0;
    Real max_raw_fy = -std::numeric_limits<Real>::max();
    Real max_applied_fy = -std::numeric_limits<Real>::max();
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
            max_raw_fy = std::max(max_raw_fy, sample.raw_force[1]);
            max_applied_fy = std::max(max_applied_fy, sample.applied_force[1]);
            ++post_entry_count;
        }
        if (sample.capped)
        {
            ++cap_count;
        }
    }

    Vec2d pre_entry_mean_force() const
    {
        if (pre_entry_count == 0)
        {
            return Vec2d::Zero();
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

inline Real water_entry_time()
{
    return std::sqrt(2.0 * kInitialClearance / kGravity);
}

inline bool is_finite(const Vec2d &value)
{
    return std::isfinite(value[0]) && std::isfinite(value[1]);
}

inline std::string generate_cylinder_boundary_particles(SPHSystem &sph_system, SolidBody &cylinder_boundary)
{
    cylinder_boundary.defineAdaptationRatios(1.15, 1.0);
    cylinder_boundary.defineBodyLevelSetShape().writeLevelSet();

    if (!sph_system.RunParticleRelaxation() && sph_system.ReloadParticles() && reload_particle_file_exists())
    {
        cylinder_boundary.generateParticles<BaseParticles, Reload>(kRelaxedCylinderReloadBodyName);
        return "Reload";
    }

    cylinder_boundary.generateParticles<BaseParticles, Lattice>();
    return "Lattice";
}

inline int run_cylinder_particle_relaxation(SolidBody &cylinder_boundary)
{
    InnerRelation cylinder_inner(cylinder_boundary);

    using namespace relax_dynamics;
    SimpleDynamics<RandomizeParticlePosition> random_cylinder_particles(cylinder_boundary);
    RelaxationStepInner relaxation_step_inner(cylinder_inner);
    BodyStatesRecordingToVtp write_cylinder_state(cylinder_boundary);
    ReloadParticleIO write_particle_reload_files(cylinder_boundary, kRelaxedCylinderReloadBodyName);

    random_cylinder_particles.exec(0.25);
    relaxation_step_inner.SurfaceBounding().exec();
    write_cylinder_state.writeToFile(0);

    int ite_p = 0;
    while (ite_p < kRelaxationSteps)
    {
        relaxation_step_inner.exec();
        ite_p += 1;
        if (ite_p % kRelaxationOutputInterval == 0)
        {
            std::cout << std::fixed << std::setprecision(9)
                      << "Relaxation steps for the LAMMPS-driven cylinder boundary N = "
                      << ite_p << "\n";
            write_cylinder_state.writeToFile(ite_p);
        }
    }

    write_particle_reload_files.writeToFile(0);
    const std::filesystem::path reload_path = reload_particle_file();
    if (!std::filesystem::exists(reload_path))
    {
        throw std::runtime_error("particle relaxation did not write reload/Reload.xml");
    }

    std::cout << std::setprecision(17);
    std::cout << "SPHinXsys 2D LAMMPS two-way water-entry cylinder particle relaxation\n";
    std::cout << "cylinder_particles: " << cylinder_boundary.getBaseParticles().TotalRealParticles() << '\n';
    std::cout << "particle_spacing_m: " << kParticleSpacing << '\n';
    std::cout << "relaxation_steps: " << kRelaxationSteps << '\n';
    std::cout << "reload_body_name: " << kRelaxedCylinderReloadBodyName << '\n';
    std::cout << "reload_file: " << std::filesystem::absolute(reload_path).string() << '\n';
    std::cout << "VTP_output_folder: " << std::filesystem::absolute(IO::getEnvironment().OutputFolder()).string() << '\n';
    std::cout << "status: RELAXATION_PASS\n";
    return 0;
}

inline void update_water_cylinder_configuration(FluidBody &water_block,
                                                SolidBody &cylinder_boundary,
                                                ComplexRelation &water_complex,
                                                ContactRelation &cylinder_contact)
{
    water_block.updateCellLinkedList();
    cylinder_boundary.updateCellLinkedList();
    water_complex.updateConfiguration();
    cylinder_contact.updateConfiguration();
}

inline Vec2d sum_cylinder_hydro_force(SPHBody &cylinder)
{
    BaseParticles &particles = cylinder.getBaseParticles();
    Vecd *pressure_force = particles.getVariableDataByName<Vecd>("PressureForceFromFluid");
    Vecd *viscous_force = particles.getVariableDataByName<Vecd>("ViscousForceFromFluid");

    Vec2d total = Vec2d::Zero();
    for (UnsignedInt i = 0; i != particles.TotalRealParticles(); ++i)
    {
        const Vecd particle_force = pressure_force[i] + viscous_force[i];
        total += Vec2d(particle_force[0], particle_force[1]);
    }
    return total;
}

struct ForceApplication
{
    Vec2d applied_force = Vec2d::Zero();
    bool capped = false;
};

inline ForceApplication relax_and_cap_force(const Vec2d &raw_force, const Vec2d &previous_applied_force)
{
    ForceApplication result;
    Vec2d relaxed_force =
        kForceRelaxationAlpha * raw_force + (1.0 - kForceRelaxationAlpha) * previous_applied_force;

    const Real cap = kForceCapWeightFactor * cylinder_weight();
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
                                       const Vec2d &sph_geometric_center)
{
    MotionSample sample;
    sample.number_of_iterations = number_of_iterations;
    sample.lammps_step = lammps_step;
    sample.time = time;
    sample.dem_state = dem_state;
    sample.sph_geometric_center = sph_geometric_center;
    sample.center_error = (sph_geometric_center - dem_state.center).norm();
    sample.y_freefall = kInitialCenter[1] - 0.5 * kGravity * sample.time * sample.time;
    sample.vy_freefall = -kGravity * sample.time;
    sample.y_minus_freefall = dem_state.center[1] - sample.y_freefall;
    sample.vy_minus_freefall = dem_state.velocity[1] - sample.vy_freefall;
    return sample;
}

inline ForceSample make_force_sample(int number_of_iterations,
                                     Real time,
                                     const DEMState &dem_state,
                                     const Vec2d &raw_force,
                                     const Vec2d &applied_force,
                                     bool capped)
{
    ForceSample sample;
    sample.number_of_iterations = number_of_iterations;
    sample.time = time;
    sample.raw_force = raw_force;
    sample.applied_force = applied_force;
    sample.raw_force_norm = raw_force.norm();
    sample.applied_force_norm = applied_force.norm();
    sample.cylinder_bottom_y = dem_state.center[1] - kCylinderRadius;
    sample.pre_entry_stat = sample.cylinder_bottom_y > kWaterHeight + kPreEntryClearanceForStats;
    sample.post_entry_stat = sample.cylinder_bottom_y < kWaterHeight - kPostEntryDepthForStats;
    sample.capped = capped;
    return sample;
}

inline void write_motion_csv_header(std::ofstream &csv)
{
    csv << "number_of_iterations,lammps_step,time_s,"
           "lammps_x_m,lammps_y_m,lammps_vx_m_per_s,lammps_vy_m_per_s,"
           "lammps_ax_m_per_s2,lammps_ay_m_per_s2,lammps_omega_z_rad_per_s,"
           "sph_center_x_m,sph_center_y_m,center_error_m,"
           "y_freefall_m,vy_freefall_m_per_s,y_minus_freefall_m,vy_minus_freefall_m_per_s\n";
}

inline void write_motion_csv_sample(std::ofstream &csv, const MotionSample &sample)
{
    csv << sample.number_of_iterations << ','
        << sample.lammps_step << ','
        << sample.time << ','
        << sample.dem_state.center[0] << ','
        << sample.dem_state.center[1] << ','
        << sample.dem_state.velocity[0] << ','
        << sample.dem_state.velocity[1] << ','
        << sample.dem_state.acceleration[0] << ','
        << sample.dem_state.acceleration[1] << ','
        << sample.dem_state.omega_z << ','
        << sample.sph_geometric_center[0] << ','
        << sample.sph_geometric_center[1] << ','
        << sample.center_error << ','
        << sample.y_freefall << ','
        << sample.vy_freefall << ','
        << sample.y_minus_freefall << ','
        << sample.vy_minus_freefall << '\n';
}

inline void write_force_csv_header(std::ofstream &csv)
{
    csv << "number_of_iterations,time_s,"
           "Fx_raw_N,Fy_raw_N,raw_force_norm_N,"
           "Fx_applied_N,Fy_applied_N,applied_force_norm_N,"
           "cylinder_weight_N,force_cap_N,capped,"
           "cylinder_bottom_y_m,geometric_entry,pre_entry_stat,post_entry_stat\n";
}

inline void write_force_csv_sample(std::ofstream &csv, const ForceSample &sample)
{
    const bool geometric_entry = sample.cylinder_bottom_y <= kWaterHeight;
    csv << sample.number_of_iterations << ','
        << sample.time << ','
        << sample.raw_force[0] << ','
        << sample.raw_force[1] << ','
        << sample.raw_force_norm << ','
        << sample.applied_force[0] << ','
        << sample.applied_force[1] << ','
        << sample.applied_force_norm << ','
        << cylinder_weight() << ','
        << kForceCapWeightFactor * cylinder_weight() << ','
        << (sample.capped ? 1 : 0) << ','
        << sample.cylinder_bottom_y << ','
        << (geometric_entry ? 1 : 0) << ','
        << (sample.pre_entry_stat ? 1 : 0) << ','
        << (sample.post_entry_stat ? 1 : 0) << '\n';
}
} // namespace LammpsTwoWayWaterEntry2D
