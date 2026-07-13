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
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace SPH;

namespace LammpsOneWayWaterEntry
{
using SPH::lammps_examples::LammpsInstance;
using SPH::lammps_examples::CouplingAdvanceResult;
using SPH::lammps_examples::CouplingStepPlan;
using SPH::lammps_examples::ExternalForceBuffer;
using SPH::lammps_examples::LammpsTimeIntegrator;
using SPH::lammps_examples::makeCouplingStepPlan;
using SPH::lammps_examples::extract_atom_vector3_by_consecutive_id;
using SPH::lammps_examples::validate_consecutive_atom_ids;
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
inline constexpr Real kCenterErrorTolerance = 1.0e-9;
inline constexpr Real kFreeFallErrorTolerance = 1.0e-8;
inline constexpr Real kPreEntryForceNormTolerance = 1.0e-3;
inline constexpr Real kMinimumPostEntryFz = 1.0e-4;
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

    void update(size_t index_i, Real dt = 0.0)
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
struct DEMState
{
    Vec3d center = Vec3d::Zero();
    Vec3d velocity = Vec3d::Zero();
    Vec3d omega = Vec3d::Zero();
};

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
             << "pair_style zero 0.1\n"
             << "pair_coeff * *\n"
             << "neighbor 0.01 bin\n"
             << "neigh_modify delay 0 every 1 check yes\n"
             << "fix int all nve/sphere\n"
             << "fix grav all gravity " << kGravity << " vector 0.0 0.0 -1.0\n"
             << "fix ext all external pf/callback 1 1\n"
             << "timestep " << kDemMaxDt << "\n"
             << "thermo 1000000\n";

        lammps_.commands_string(cmds.str(), "LAMMPS initialization commands");
        external_force_.registerFix(lammps_, "ext");
    }

    CouplingStepPlan planCouplingStep(Real acoustic_limit) const
    {
        return makeCouplingStepPlan(acoustic_limit, kDemMaxDt);
    }

    CouplingAdvanceResult advance(const CouplingStepPlan &plan)
    {
        return time_integrator_.advance(plan);
    }

    CouplingAdvanceResult advance(const CouplingStepPlan &plan, Real driver_time_before)
    {
        return time_integrator_.advance(plan, driver_time_before);
    }

    int runForDuration(Real acoustic_step)
    {
        return advance(planCouplingStep(acoustic_step)).dem_steps;
    }

    DEMState pullState() const
    {
        validate_consecutive_atom_ids(lammps_, 1);
        std::array<double, 3> x{};
        std::array<double, 3> v{};
        std::array<double, 3> omega{};

        extract_atom_vector3_by_consecutive_id(lammps_, "x", 1, x.data());
        extract_atom_vector3_by_consecutive_id(lammps_, "v", 1, v.data());
        extract_atom_vector3_by_consecutive_id(lammps_, "omega", 1, omega.data());

        return DEMState{to_vec3d(x), to_vec3d(v), to_vec3d(omega)};
    }

    int version() const { return lammps_.version(); }

    const ExternalForceBuffer &externalForce() const { return external_force_; }

  private:
    ExternalForceBuffer external_force_;
    LammpsInstance lammps_;
    LammpsTimeIntegrator time_integrator_{lammps_, kDemMaxDt};
};
//----------------------------------------------------------------------
//	One-way moving boundary driven by the LAMMPS particle state.
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
            vel_[i] = state.velocity + state.omega.cross(relative_positions_[i]);
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
    Vec3d force = Vec3d::Zero();
    Real force_norm = 0.0;
    Vec3d hydrodynamic_torque = Vec3d::Zero();
    Real sphere_bottom_z = 0.0;
    bool pre_entry_stat = false;
    bool post_entry_stat = false;
};

struct ForceStats
{
    Vec3d pre_entry_sum = Vec3d::Zero();
    Real pre_entry_norm_sum = 0.0;
    int pre_entry_count = 0;
    Real post_entry_max_fz = -std::numeric_limits<Real>::max();
    int post_entry_count = 0;

    void add(const ForceSample &sample)
    {
        if (sample.pre_entry_stat)
        {
            pre_entry_sum += sample.force;
            pre_entry_norm_sum += sample.force_norm;
            ++pre_entry_count;
        }
        if (sample.post_entry_stat)
        {
            post_entry_max_fz = std::max(post_entry_max_fz, sample.force[2]);
            ++post_entry_count;
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

inline Real water_entry_time()
{
    return std::sqrt(2.0 * kInitialClearance / kGravity);
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
    std::cout << "SPHinXsys LAMMPS one-way water-entry sphere particle relaxation\n";
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

inline Vec3d sum_sphere_hydro_torque(SPHBody &sphere, const Vec3d &center)
{
    BaseParticles &particles = sphere.getBaseParticles();
    Vecd *positions = particles.ParticlePositions();
    Vecd *pressure_force = particles.getVariableDataByName<Vecd>("PressureForceFromFluid");
    Vecd *viscous_force = particles.getVariableDataByName<Vecd>("ViscousForceFromFluid");

    Vec3d torque = Vec3d::Zero();
    for (UnsignedInt i = 0; i != particles.TotalRealParticles(); ++i)
    {
        const Vec3d relative_position = positions[i] - center;
        const Vecd particle_force = pressure_force[i] + viscous_force[i];
        torque += relative_position.cross(
            Vec3d(particle_force[0], particle_force[1], particle_force[2]));
    }
    return torque;
}

inline std::filesystem::path write_dem_load_to_vtp(
    int iteration, Real time, const DEMState &state,
    const Vec3d &hydrodynamic_force, const Vec3d &hydrodynamic_torque)
{
    return SPH::lammps_examples::write_dem_point_fields_to_vtp(
        iteration, time, "DEM_Load", state.center,
        {{"HydrodynamicForceRaw", hydrodynamic_force},
         {"HydrodynamicTorqueRaw", hydrodynamic_torque},
         {"Velocity", state.velocity},
         {"AngularVelocity", state.omega}},
        {{"Radius", kSphereRadius}, {"Mass", sphere_mass()}},
        "HydrodynamicForceRaw");
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
                                      const Vec3d &force,
                                      const Vec3d &hydrodynamic_torque)
{
    ForceSample sample;
    sample.number_of_iterations = number_of_iterations;
    sample.time = time;
    sample.force = force;
    sample.force_norm = force.norm();
    sample.hydrodynamic_torque = hydrodynamic_torque;
    sample.sphere_bottom_z = dem_state.center[2] - kSphereRadius;
    sample.pre_entry_stat = sample.sphere_bottom_z > kWaterHeight + kPreEntryClearanceForStats;
    sample.post_entry_stat = sample.sphere_bottom_z < kWaterHeight - kPostEntryDepthForStats;
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
    csv << "number_of_iterations,time_s,Fx_N,Fy_N,Fz_N,force_norm_N,"
           "Tx_Nm,Ty_Nm,Tz_Nm,torque_norm_Nm,"
           "sphere_bottom_z_m,geometric_entry,pre_entry_stat,post_entry_stat\n";
}

inline void write_force_csv_sample(std::ofstream &csv, const ForceSample &sample)
{
    const bool geometric_entry = sample.sphere_bottom_z <= kWaterHeight;
    csv << sample.number_of_iterations << ','
        << sample.time << ','
        << sample.force[0] << ','
        << sample.force[1] << ','
        << sample.force[2] << ','
        << sample.force_norm << ','
        << sample.hydrodynamic_torque[0] << ','
        << sample.hydrodynamic_torque[1] << ','
        << sample.hydrodynamic_torque[2] << ','
        << sample.hydrodynamic_torque.norm() << ','
        << sample.sphere_bottom_z << ','
        << (geometric_entry ? 1 : 0) << ','
        << (sample.pre_entry_stat ? 1 : 0) << ','
        << (sample.post_entry_stat ? 1 : 0) << '\n';
}
} // namespace LammpsOneWayWaterEntry
