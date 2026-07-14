#pragma once

#include "sphinxsys.h"
#include "lammps_instance.h"
#include "lammps_dem_adapter_common.h"
#include "lammps_io.h"

#include <algorithm>
#include <array>
#include <cmath>
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

namespace LammpsUnderwaterGranularCollapse3D
{
using SPH::lammps_examples::CouplingAdvanceResult;
using SPH::lammps_examples::CouplingStepPlan;
using SPH::lammps_examples::ExternalForceBuffer;
using SPH::lammps_examples::LammpsInstance;
using SPH::lammps_examples::LammpsTimeIntegrator;
using SPH::lammps_examples::ParticleForce;
using SPH::lammps_examples::VtpPvdWriter;
using SPH::lammps_examples::VtpScalarPointField;
using SPH::lammps_examples::VtpVectorPointField3d;
using SPH::lammps_examples::extract_atom_vector3_by_consecutive_id;
using SPH::lammps_examples::lammps_example_output_path;
using SPH::lammps_examples::makeCouplingStepPlan;
using SPH::lammps_examples::validate_consecutive_atom_ids;
using SPH::lammps_examples::write_dem_points_fields_to_vtp;
using SPH::lammps_examples::write_dem_spheres_to_vtp;

//----------------------------------------------------------------------
//  The 2 mm SPH spacing resolves the connected 3D pore space of the compact
//  column while keeping this verification case practical on a local CPU.
//----------------------------------------------------------------------
inline constexpr Real kTankLengthX = 0.16;
inline constexpr Real kTankHeightZ = 0.14;
inline constexpr Real kWaterHeight = 0.11;
inline constexpr Real kGrainRadius = 0.008;
inline constexpr Real kGrainDiameter = 2.0 * kGrainRadius;
inline constexpr int kColumnCountX = 3;
inline constexpr int kColumnCountY = 3;
inline constexpr int kColumnCountZ = 4;
inline constexpr int kParticleCount = kColumnCountX * kColumnCountY * kColumnCountZ;
inline constexpr Real kParticleSpacing = 0.002;
inline constexpr Real kBoundaryWidth = 4.0 * kParticleSpacing;
inline constexpr Real kInitialWallClearance = 0.1 * kParticleSpacing;
inline constexpr Real kInitialWaterGrainGap = 0.5 * kParticleSpacing;
inline constexpr Real kWaterVoidRadius = kGrainRadius + kInitialWaterGrainGap;
inline constexpr Real kColumnSpacingX = 2.05 * kGrainRadius;
inline constexpr Real kColumnSpacingY = 2.05 * kGrainRadius;
inline constexpr Real kColumnSpacingZ = 2.02 * kGrainRadius;
// The three-grain sample spans the full transverse water width. Both outer
// grain surfaces start at the same small clearance from the y-plane walls.
inline constexpr Real kTankLengthY = 2.0 * (kGrainRadius + kInitialWallClearance) +
                                     static_cast<Real>(kColumnCountY - 1) * kColumnSpacingY;
inline constexpr Real kEndTime = 1.0;
inline constexpr Real kFluidRelaxationTime = 0.0;
inline constexpr Real kVtpOutputInterval = 0.02;
inline constexpr int kCsvOutputStride = 25;
inline constexpr double kDemMaxDt = 1.0e-5;
inline constexpr int kRelaxationSteps = 1000;
inline constexpr int kRelaxationOutputInterval = 200;
inline constexpr Real kForceRelaxationAlpha = 0.2;
inline constexpr Real kForceCapWeightFactor = 5.0;

inline constexpr Real kWaterDensity = 1000.0;
inline constexpr Real kGrainDensity = 2500.0;
inline constexpr Real kGravity = 9.81;
inline constexpr Real kDynamicViscosity = 1.0e-3;
inline const Real kCharacteristicVelocity = 2.0 * std::sqrt(kGravity * kWaterHeight);
inline const Real kSoundSpeed = 10.0 * kCharacteristicVelocity;

inline constexpr Real kLeftWallX = 0.0;
inline constexpr Real kRightWallX = kTankLengthX;
inline constexpr Real kFrontWallY = 0.0;
inline constexpr Real kBackWallY = kTankLengthY;
inline constexpr Real kBottomWallZ = 0.0;
inline constexpr Real kContactNormalStiffness = 5.0e4;
inline constexpr Real kContactRestitution = 0.2;
inline constexpr Real kContactTangentialStiffness = 4.0e4;
inline constexpr Real kContactTangentialDamping = 0.0;
inline constexpr Real kContactFriction = 0.5;

inline constexpr Real kCenterErrorTolerance = 1.0e-10;
inline constexpr Real kMassRelativeTolerance = 1.0e-12;
inline constexpr Real kMinimumRightFrontDisplacement = 2.0e-4;
inline constexpr Real kMinimumHydroForceNorm = 1.0e-8;
inline const std::string kRelaxedGranularColumnReloadBodyName = "LammpsUnderwaterGranularColumn3D";

inline const BoundingBoxd kSystemDomainBounds(
    Vec3d(-kBoundaryWidth, -kBoundaryWidth, -kBoundaryWidth),
    Vec3d(kTankLengthX + kBoundaryWidth, kTankLengthY + kBoundaryWidth, kTankHeightZ + kBoundaryWidth));

inline std::vector<Vec3d> initial_dem_centers()
{
    std::vector<Vec3d> centers;
    centers.reserve(kParticleCount);
    const Real y_origin = kFrontWallY + kInitialWallClearance + kGrainRadius;
    for (int layer = 0; layer < kColumnCountZ; ++layer)
    {
        for (int row = 0; row < kColumnCountY; ++row)
        {
            for (int column = 0; column < kColumnCountX; ++column)
            {
                const Real lean = 0.14 * kGrainRadius *
                                  static_cast<Real>(layer) / static_cast<Real>(kColumnCountZ - 1);
                const Real x = kLeftWallX + kInitialWallClearance + kGrainRadius +
                               static_cast<Real>(column) * kColumnSpacingX + lean;
                const Real y = y_origin + static_cast<Real>(row) * kColumnSpacingY;
                const Real z = kBottomWallZ + kInitialWallClearance + kGrainRadius +
                               static_cast<Real>(layer) * kColumnSpacingZ;
                centers.emplace_back(x, y, z);
            }
        }
    }
    return centers;
}

inline void verify_initial_transverse_confinement()
{
    const std::vector<Vec3d> centers = initial_dem_centers();
    Real min_surface_y = std::numeric_limits<Real>::max();
    Real max_surface_y = -std::numeric_limits<Real>::max();
    for (const Vec3d &center : centers)
    {
        min_surface_y = std::min(min_surface_y, center[1] - kGrainRadius);
        max_surface_y = std::max(max_surface_y, center[1] + kGrainRadius);
    }

    const Real front_clearance = min_surface_y - kFrontWallY;
    const Real back_clearance = kBackWallY - max_surface_y;
    const Real tolerance = 100.0 * TinyReal;
    if (std::abs(front_clearance - kInitialWallClearance) > tolerance ||
        std::abs(back_clearance - kInitialWallClearance) > tolerance)
    {
        throw std::runtime_error("the 3D DEM sample is not transversely confined by both y-plane walls");
    }
}

class WaterBlock : public ComplexShape
{
  public:
    explicit WaterBlock(const std::string &shape_name) : ComplexShape(shape_name)
    {
        add<GeometricShapeBox>(
            Transform(Vec3d(0.5 * kTankLengthX, 0.5 * kTankLengthY, 0.5 * kWaterHeight)),
            Vec3d(0.5 * kTankLengthX, 0.5 * kTankLengthY, 0.5 * kWaterHeight),
            "WaterBox");
        int grain_id = 1;
        for (const Vec3d &center : initial_dem_centers())
        {
            // Start with a half-dp dry shell so lattice water particles do not
            // overlap the LAMMPS-driven proxy surface at t = 0.
            subtract<GeometricShapeBall>(center, kWaterVoidRadius,
                                         "InitialGrainVoid_" + std::to_string(grain_id));
            ++grain_id;
        }
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
            Transform(Vec3d(0.5 * kTankLengthX, 0.5 * kTankLengthY,
                            0.5 * kTankHeightZ + kBoundaryWidth)),
            Vec3d(0.5 * kTankLengthX, 0.5 * kTankLengthY,
                  0.5 * kTankHeightZ + kBoundaryWidth),
            "InnerTankVoid");
    }
};

class GranularColumnShape : public ComplexShape
{
  public:
    explicit GranularColumnShape(const std::string &shape_name) : ComplexShape(shape_name)
    {
        int grain_id = 1;
        for (const Vec3d &center : initial_dem_centers())
        {
            add<GeometricShapeBall>(center, kGrainRadius, "Grain_" + std::to_string(grain_id));
            ++grain_id;
        }
    }
};

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

struct DEMParticleState
{
    int id = 0;
    Vec3d center = Vec3d::Zero();
    Vec3d velocity = Vec3d::Zero();
    Vec3d acceleration = Vec3d::Zero();
    Vec3d omega = Vec3d::Zero();
};

inline Real grain_volume()
{
    return (4.0 / 3.0) * Pi * kGrainRadius * kGrainRadius * kGrainRadius;
}

inline Real grain_mass()
{
    return kGrainDensity * grain_volume();
}

inline Real grain_weight()
{
    return grain_mass() * kGravity;
}

template <size_t N>
inline Vec3d to_vec3d(const std::array<double, N> &values, int index)
{
    return Vec3d(values[3 * index], values[3 * index + 1], values[3 * index + 2]);
}

inline Real initial_right_front_x()
{
    Real right_front = -std::numeric_limits<Real>::max();
    for (const Vec3d &center : initial_dem_centers())
    {
        right_front = std::max(right_front, center[0] + kGrainRadius);
    }
    return right_front;
}

class LammpsGranularColumnAdapter
{
  public:
    LammpsGranularColumnAdapter()
    {
        std::ostringstream cmds;
        cmds << std::setprecision(17)
             << "dimension 3\n"
             << "units si\n"
             << "atom_style sphere\n"
             << "atom_modify map array\n"
             << "boundary f f f\n"
             << "newton off\n"
             << "comm_modify vel yes\n"
             << "region box block -0.02 " << kTankLengthX + 0.02
             << " -0.02 " << kTankLengthY + 0.02
             << " -0.05 " << kTankHeightZ + 0.08 << " units box\n"
             << "create_box 1 box\n";
        for (const Vec3d &center : initial_dem_centers())
        {
            cmds << "create_atoms 1 single "
                 << center[0] << ' ' << center[1] << ' ' << center[2] << " units box\n";
        }
        cmds << "set type 1 diameter " << kGrainDiameter
             << " density " << kGrainDensity << "\n"
             << "velocity all set 0.0 0.0 0.0 units box\n"
             << "pair_style granular\n"
             << "pair_coeff * * hooke " << kContactNormalStiffness << ' '
             << kContactRestitution
             << " tangential linear_history " << kContactTangentialStiffness << ' '
             << kContactTangentialDamping << ' ' << kContactFriction
             << " damping coeff_restitution\n"
             << "neighbor 0.02 bin\n"
             << "neigh_modify delay 0 every 1 check yes\n"
             << "fix int all nve/sphere\n"
             << "fix grav all gravity " << kGravity << " vector 0.0 0.0 -1.0\n"
             << "fix ext all external pf/callback 1 1\n"
             << "fix xwall all wall/gran granular hooke " << kContactNormalStiffness << ' '
             << kContactRestitution
             << " tangential linear_history " << kContactTangentialStiffness << ' '
             << kContactTangentialDamping << ' ' << kContactFriction
             << " damping coeff_restitution xplane " << kLeftWallX << ' ' << kRightWallX << " contacts\n"
             << "fix ywall all wall/gran granular hooke " << kContactNormalStiffness << ' '
             << kContactRestitution
             << " tangential linear_history " << kContactTangentialStiffness << ' '
             << kContactTangentialDamping << ' ' << kContactFriction
             << " damping coeff_restitution yplane " << kFrontWallY << ' ' << kBackWallY << " contacts\n"
             << "fix floor all wall/gran granular hooke " << kContactNormalStiffness << ' '
             << kContactRestitution
             << " tangential linear_history " << kContactTangentialStiffness << ' '
             << kContactTangentialDamping << ' ' << kContactFriction
             << " damping coeff_restitution zplane " << kBottomWallZ << " NULL contacts\n"
             << "timestep " << kDemMaxDt << "\n"
             << "thermo 1000000\n";

        lammps_.commands_string(cmds.str(), "3D granular collapse LAMMPS initialization commands");
        external_force_.registerFix(lammps_, "ext");
    }

    CouplingStepPlan planCouplingStep(Real acoustic_limit) const
    {
        return makeCouplingStepPlan(acoustic_limit, kDemMaxDt);
    }

    CouplingAdvanceResult advance(const CouplingStepPlan &plan, Real driver_time_before)
    {
        return time_integrator_.advance(plan, driver_time_before);
    }

    void setExternalForces(const std::vector<Vec3d> &forces)
    {
        if (static_cast<int>(forces.size()) != kParticleCount)
        {
            throw std::runtime_error("hydrodynamic force vector size does not match 3D DEM particle count");
        }

        std::vector<ParticleForce> particle_forces;
        particle_forces.reserve(kParticleCount);
        for (int id = 1; id <= kParticleCount; ++id)
        {
            const Vec3d &force = forces[id - 1];
            particle_forces.push_back(ParticleForce{id, {force[0], force[1], force[2]}});
        }
        external_force_.setForces(particle_forces);
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
        for (int index = 0; index < kParticleCount; ++index)
        {
            DEMParticleState state;
            state.id = index + 1;
            state.center = to_vec3d(x, index);
            state.velocity = to_vec3d(v, index);
            state.acceleration = to_vec3d(f, index) / grain_mass();
            state.omega = to_vec3d(omega, index);
            states.push_back(state);
        }
        return states;
    }

    int version() const { return lammps_.version(); }
    const ExternalForceBuffer &externalForceTable() const { return external_force_; }

  private:
    ExternalForceBuffer external_force_;
    LammpsInstance lammps_;
    LammpsTimeIntegrator time_integrator_{lammps_, kDemMaxDt};
};

class DrivenGranularColumnBoundary
{
  public:
    DrivenGranularColumnBoundary(SolidBody &column_body, const std::vector<Vec3d> &initial_centers)
        : column_body_(column_body), particles_(column_body.getBaseParticles()),
          pos_(particles_.ParticlePositions()),
          vel_(particles_.registerStateVariableData<Vecd>("Velocity")),
          acc_(particles_.registerStateVariableData<Vecd>("Acceleration")),
          initial_centers_(initial_centers)
    {
        if (particles_.TotalRealParticles() == 0)
        {
            throw std::runtime_error("SPHinXsys 3D granular column body generated zero particles");
        }

        particle_dem_ids_.resize(particles_.TotalRealParticles());
        relative_positions_.resize(particles_.TotalRealParticles(), Vec3d::Zero());
        std::vector<Vec3d> center_sums(kParticleCount, Vec3d::Zero());
        std::vector<int> particle_counts(kParticleCount, 0);
        for (UnsignedInt i = 0; i != particles_.TotalRealParticles(); ++i)
        {
            const int id = nearest_dem_id(pos_[i]);
            particle_dem_ids_[i] = id;
            center_sums[id] += pos_[i];
            ++particle_counts[id];
        }
        for (int id = 0; id < kParticleCount; ++id)
        {
            if (particle_counts[id] == 0)
            {
                throw std::runtime_error("a 3D DEM grain received zero SPHinXsys proxy particles");
            }
            const Vec3d correction = initial_centers_[id] -
                                     center_sums[id] / static_cast<Real>(particle_counts[id]);
            for (UnsignedInt i = 0; i != particles_.TotalRealParticles(); ++i)
            {
                if (particle_dem_ids_[i] == id)
                {
                    pos_[i] += correction;
                }
            }
        }
        for (UnsignedInt i = 0; i != particles_.TotalRealParticles(); ++i)
        {
            const int id = particle_dem_ids_[i];
            relative_positions_[i] = pos_[i] - initial_centers_[id];
            vel_[i] = Vec3d::Zero();
            acc_[i] = Vec3d::Zero();
        }
    }

    void update(const std::vector<DEMParticleState> &states)
    {
        if (static_cast<int>(states.size()) != kParticleCount)
        {
            throw std::runtime_error("SPHinXsys proxy update received an unexpected DEM state count");
        }
        for (UnsignedInt i = 0; i != particles_.TotalRealParticles(); ++i)
        {
            const DEMParticleState &state = states[particle_dem_ids_[i]];
            pos_[i] = state.center + relative_positions_[i];
            vel_[i] = state.velocity + state.omega.cross(relative_positions_[i]);
            acc_[i] = state.acceleration;
        }
        column_body_.setNewlyUpdated();
    }

    std::vector<Vec3d> geometricCenters() const
    {
        std::vector<Vec3d> center_sums(kParticleCount, Vec3d::Zero());
        std::vector<int> particle_counts(kParticleCount, 0);
        for (UnsignedInt i = 0; i != particles_.TotalRealParticles(); ++i)
        {
            const int id = particle_dem_ids_[i];
            center_sums[id] += pos_[i];
            ++particle_counts[id];
        }
        std::vector<Vec3d> centers(kParticleCount, Vec3d::Zero());
        for (int id = 0; id < kParticleCount; ++id)
        {
            centers[id] = center_sums[id] / static_cast<Real>(particle_counts[id]);
        }
        return centers;
    }

    const std::vector<int> &particleDemIds() const { return particle_dem_ids_; }
    UnsignedInt particleCount() const { return particles_.TotalRealParticles(); }

  private:
    int nearest_dem_id(const Vec3d &position) const
    {
        int best_id = 0;
        Real best_distance = std::numeric_limits<Real>::max();
        for (int id = 0; id < static_cast<int>(initial_centers_.size()); ++id)
        {
            const Real distance = (position - initial_centers_[id]).squaredNorm();
            if (distance < best_distance)
            {
                best_distance = distance;
                best_id = id;
            }
        }
        return best_id;
    }

    SolidBody &column_body_;
    BaseParticles &particles_;
    Vecd *pos_;
    Vecd *vel_;
    Vecd *acc_;
    std::vector<Vec3d> initial_centers_;
    std::vector<int> particle_dem_ids_;
    std::vector<Vec3d> relative_positions_;
};

inline void update_water_grain_configuration(FluidBody &water_block,
                                             SolidBody &grain_boundary,
                                             ComplexRelation &water_complex,
                                             ContactRelation &grain_contact)
{
    water_block.updateCellLinkedList();
    grain_boundary.updateCellLinkedList();
    water_complex.updateConfiguration();
    grain_contact.updateConfiguration();
}

inline std::vector<Vec3d> sum_grain_hydro_forces(SPHBody &grain_boundary,
                                                  const std::vector<int> &particle_dem_ids)
{
    BaseParticles &particles = grain_boundary.getBaseParticles();
    Vecd *pressure_force = particles.getVariableDataByName<Vecd>("PressureForceFromFluid");
    Vecd *viscous_force = particles.getVariableDataByName<Vecd>("ViscousForceFromFluid");
    std::vector<Vec3d> forces(kParticleCount, Vec3d::Zero());
    for (UnsignedInt i = 0; i != particles.TotalRealParticles(); ++i)
    {
        forces[particle_dem_ids[i]] += pressure_force[i] + viscous_force[i];
    }
    return forces;
}

inline std::vector<Vec3d> sum_grain_hydro_torques(SPHBody &grain_boundary,
                                                   const std::vector<int> &particle_dem_ids,
                                                   const std::vector<DEMParticleState> &states)
{
    BaseParticles &particles = grain_boundary.getBaseParticles();
    Vecd *positions = particles.ParticlePositions();
    Vecd *pressure_force = particles.getVariableDataByName<Vecd>("PressureForceFromFluid");
    Vecd *viscous_force = particles.getVariableDataByName<Vecd>("ViscousForceFromFluid");
    std::vector<Vec3d> torques(kParticleCount, Vec3d::Zero());
    for (UnsignedInt i = 0; i != particles.TotalRealParticles(); ++i)
    {
        const int id = particle_dem_ids[i];
        torques[id] += (positions[i] - states[id].center).cross(pressure_force[i] + viscous_force[i]);
    }
    return torques;
}

inline Real max_center_sync_error(const std::vector<DEMParticleState> &states,
                                  const std::vector<Vec3d> &proxy_centers)
{
    Real error = 0.0;
    for (int index = 0; index < kParticleCount; ++index)
    {
        error = std::max(error, (states[index].center - proxy_centers[index]).norm());
    }
    return error;
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

inline Real max_hydro_force_norm(const std::vector<Vec3d> &forces)
{
    Real max_norm = 0.0;
    for (const Vec3d &force : forces)
    {
        max_norm = std::max(max_norm, force.norm());
    }
    return max_norm;
}

inline bool is_finite(const Vec3d &value)
{
    return std::isfinite(value[0]) && std::isfinite(value[1]) && std::isfinite(value[2]);
}

inline bool all_finite(const std::vector<DEMParticleState> &states,
                       const std::vector<Vec3d> &forces)
{
    for (const DEMParticleState &state : states)
    {
        if (!is_finite(state.center) || !is_finite(state.velocity) ||
            !is_finite(state.acceleration) || !is_finite(state.omega))
        {
            return false;
        }
    }
    return std::all_of(forces.begin(), forces.end(), [](const Vec3d &force) { return is_finite(force); });
}

inline std::vector<Vec3d> centers_from_states(const std::vector<DEMParticleState> &states)
{
    std::vector<Vec3d> centers;
    centers.reserve(states.size());
    for (const DEMParticleState &state : states)
    {
        centers.push_back(state.center);
    }
    return centers;
}

inline std::vector<Vec3d> velocities_from_states(const std::vector<DEMParticleState> &states)
{
    std::vector<Vec3d> velocities;
    velocities.reserve(states.size());
    for (const DEMParticleState &state : states)
    {
        velocities.push_back(state.velocity);
    }
    return velocities;
}

inline std::vector<Real> ids_as_scalars()
{
    std::vector<Real> ids;
    ids.reserve(kParticleCount);
    for (int index = 0; index < kParticleCount; ++index)
    {
        ids.push_back(static_cast<Real>(index + 1));
    }
    return ids;
}

inline std::vector<Real> scalar_filled(Real value)
{
    return std::vector<Real>(kParticleCount, value);
}

struct ForceApplication
{
    std::vector<Vec3d> applied_forces;
    std::vector<int> capped_flags;

    int cap_count() const
    {
        return std::accumulate(capped_flags.begin(), capped_flags.end(), 0);
    }
};

inline ForceApplication relax_and_cap_forces(const std::vector<Vec3d> &raw_forces,
                                             const std::vector<Vec3d> &previous_applied_forces)
{
    if (static_cast<int>(raw_forces.size()) != kParticleCount ||
        static_cast<int>(previous_applied_forces.size()) != kParticleCount)
    {
        throw std::runtime_error("3D force relaxation received a vector with the wrong DEM particle count");
    }

    ForceApplication result;
    result.applied_forces.resize(kParticleCount, Vec3d::Zero());
    result.capped_flags.resize(kParticleCount, 0);
    const Real cap = kForceCapWeightFactor * grain_weight();
    for (int index = 0; index < kParticleCount; ++index)
    {
        Vec3d relaxed_force = kForceRelaxationAlpha * raw_forces[index] +
                              (1.0 - kForceRelaxationAlpha) * previous_applied_forces[index];
        const Real force_norm = relaxed_force.norm();
        if (force_norm > cap && force_norm > TinyReal)
        {
            relaxed_force *= cap / force_norm;
            result.capped_flags[index] = 1;
        }
        result.applied_forces[index] = relaxed_force;
    }
    return result;
}

inline std::filesystem::path write_dem_force_vtp(
    int iteration,
    Real time,
    const std::vector<DEMParticleState> &states,
    const std::vector<Vec3d> &raw_hydro_forces,
    const std::vector<Vec3d> &applied_hydro_forces,
    const std::vector<Vec3d> &raw_hydrodynamic_torques,
    const std::vector<int> &capped_flags)
{
    std::vector<Real> caps;
    caps.reserve(capped_flags.size());
    for (int capped : capped_flags)
    {
        caps.push_back(static_cast<Real>(capped));
    }
    return write_dem_points_fields_to_vtp(
        iteration, time, "DEM_Forces", centers_from_states(states),
        {{"HydrodynamicForceRaw", raw_hydro_forces},
         {"HydrodynamicForceApplied", applied_hydro_forces},
         {"Velocity", velocities_from_states(states)},
         {"HydrodynamicTorqueRaw", raw_hydrodynamic_torques}},
        {{"ParticleId", ids_as_scalars()},
         {"Radius", scalar_filled(kGrainRadius)},
         {"Mass", scalar_filled(grain_mass())},
         {"Capped", caps}},
        "HydrodynamicForceApplied");
}

inline std::filesystem::path reload_particle_file()
{
    return std::filesystem::path(IO::getEnvironment().ReloadFolder()) / "Reload.xml";
}

inline bool reload_particle_file_exists()
{
    return std::filesystem::exists(reload_particle_file());
}

inline std::string generate_granular_column_boundary_particles(SPHSystem &sph_system,
                                                               SolidBody &grain_boundary)
{
    grain_boundary.defineAdaptationRatios(1.15, 1.0);
    grain_boundary.defineBodyLevelSetShape().writeLevelSet();
    if (!sph_system.RunParticleRelaxation() && sph_system.ReloadParticles() && reload_particle_file_exists())
    {
        grain_boundary.generateParticles<BaseParticles, Reload>(kRelaxedGranularColumnReloadBodyName);
        return "Reload";
    }
    grain_boundary.generateParticles<BaseParticles, Lattice>();
    return "Lattice";
}

inline int run_granular_proxy_particle_relaxation(SolidBody &grain_boundary)
{
    InnerRelation grain_inner(grain_boundary);
    using namespace relax_dynamics;
    SimpleDynamics<RandomizeParticlePosition> random_grain_particles(grain_boundary);
    RelaxationStepInner grain_relaxation_step(grain_inner);
    BodyStatesRecordingToVtp write_relaxed_state(grain_boundary);
    ReloadParticleIO write_particle_reload_files(grain_boundary);

    random_grain_particles.exec(0.25);
    grain_relaxation_step.SurfaceBounding().exec();
    write_relaxed_state.writeToFile(0);
    for (int iteration = 1; iteration <= kRelaxationSteps; ++iteration)
    {
        grain_relaxation_step.exec();
        if (iteration % kRelaxationOutputInterval == 0)
        {
            std::cout << "Relaxation steps for the 3D granular proxy N = " << iteration << '\n';
            write_relaxed_state.writeToFile(iteration);
        }
    }
    write_particle_reload_files.writeToFile(0);
    if (!reload_particle_file_exists())
    {
        throw std::runtime_error("3D granular proxy relaxation did not write reload/Reload.xml");
    }
    std::cout << "status: RELAXATION_PASS\n";
    return 0;
}

inline void write_motion_csv_header(std::ofstream &csv)
{
    csv << "frame,time_s,coupling_active,lammps_step,particle_id,"
           "x_m,y_m,z_m,vx_m_per_s,vy_m_per_s,vz_m_per_s,"
           "omega_x_rad_per_s,omega_y_rad_per_s,omega_z_rad_per_s,"
           "proxy_center_x_m,proxy_center_y_m,proxy_center_z_m,center_error_m\n";
}

inline void write_force_csv_header(std::ofstream &csv)
{
    csv << "frame,time_s,coupling_active,particle_id,"
           "Fx_raw_N,Fy_raw_N,Fz_raw_N,raw_force_norm_N,"
           "Fx_applied_N,Fy_applied_N,Fz_applied_N,applied_force_norm_N,"
           "Tx_raw_Nm,Ty_raw_Nm,Tz_raw_Nm,grain_weight_N,force_cap_N,capped\n";
}

inline void write_motion_csv_samples(std::ofstream &csv,
                                     int frame,
                                     Real time,
                                     int lammps_step,
                                     const std::vector<DEMParticleState> &states,
                                     const std::vector<Vec3d> &proxy_centers)
{
    for (int index = 0; index < kParticleCount; ++index)
    {
        const DEMParticleState &state = states[index];
        const Real center_error = (state.center - proxy_centers[index]).norm();
        csv << frame << ',' << time << ",1," << lammps_step << ',' << state.id << ','
            << state.center[0] << ',' << state.center[1] << ',' << state.center[2] << ','
            << state.velocity[0] << ',' << state.velocity[1] << ',' << state.velocity[2] << ','
            << state.omega[0] << ',' << state.omega[1] << ',' << state.omega[2] << ','
            << proxy_centers[index][0] << ',' << proxy_centers[index][1] << ',' << proxy_centers[index][2] << ','
            << center_error << '\n';
    }
}

inline void write_force_csv_samples(std::ofstream &csv,
                                    int frame,
                                    Real time,
                                    const std::vector<Vec3d> &raw_hydro_forces,
                                    const std::vector<Vec3d> &applied_hydro_forces,
                                    const std::vector<Vec3d> &raw_hydrodynamic_torques,
                                    const std::vector<int> &capped_flags)
{
    const Real cap = kForceCapWeightFactor * grain_weight();
    for (int index = 0; index < kParticleCount; ++index)
    {
        const Vec3d &raw = raw_hydro_forces[index];
        const Vec3d &applied = applied_hydro_forces[index];
        const Vec3d &torque = raw_hydrodynamic_torques[index];
        csv << frame << ',' << time << ",1," << index + 1 << ','
            << raw[0] << ',' << raw[1] << ',' << raw[2] << ',' << raw.norm() << ','
            << applied[0] << ',' << applied[1] << ',' << applied[2] << ',' << applied.norm() << ','
            << torque[0] << ',' << torque[1] << ',' << torque[2] << ','
            << grain_weight() << ',' << cap << ',' << capped_flags[index] << '\n';
    }
}
} // namespace LammpsUnderwaterGranularCollapse3D
