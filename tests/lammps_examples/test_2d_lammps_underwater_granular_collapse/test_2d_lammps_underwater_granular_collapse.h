#pragma once

#include "sphinxsys.h"
#include "lammps_instance.h"
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

namespace LammpsUnderwaterGranularCollapse2D
{
using SPH::lammps_examples::LammpsInstance;
using SPH::lammps_examples::VtpPvdWriter;
using SPH::lammps_examples::VtpScalarPointField;
using SPH::lammps_examples::VtpVectorPointField2d;
using SPH::lammps_examples::lammps_example_output_path;
using SPH::lammps_examples::write_dem_discs_to_vtp;
using SPH::lammps_examples::write_dem_points_fields_to_vtp;

inline constexpr Real kTankLengthX = 0.16;
inline constexpr Real kTankHeightY = 0.14;
inline constexpr Real kWaterHeight = 0.12;
inline constexpr Real kGrainRadius = 0.005;
inline constexpr Real kGrainDiameter = 2.0 * kGrainRadius;
inline constexpr int kColumnCols = 4;
inline constexpr int kColumnRows = 6;
inline constexpr int kParticleCount = kColumnCols * kColumnRows;
inline constexpr Real kParticleSpacing = kGrainDiameter / 5.0;
inline constexpr Real kBoundaryWidth = 4.0 * kParticleSpacing;
inline constexpr Real kInitialWallClearance = 0.1 * kParticleSpacing;
inline constexpr Real kInitialWaterGrainGap = 0.1 * kParticleSpacing;
inline constexpr Real kWaterVoidRadius = kGrainRadius + kInitialWaterGrainGap;
inline constexpr Real kColumnSpacingX = 2.05 * kGrainRadius;
inline constexpr Real kColumnSpacingY = 2.02 * kGrainRadius;
inline constexpr int kCircleShapeResolution = 96;
inline constexpr Real kEndTime = 1.0;
inline constexpr Real kFluidRelaxationTime = 0.0;
inline constexpr Real kVtpOutputInterval = 0.01;
inline constexpr double kDemMaxDt = 1.0e-5;
inline constexpr int kRelaxationSteps = 1000;
inline constexpr int kWaterRelaxationSteps = 1;
inline constexpr int kRelaxationOutputInterval = 200;
inline constexpr Real kForceRelaxationAlpha = 0.3;
inline constexpr Real kForceCapWeightFactor = 5.0;

inline constexpr Real kWaterDensity = 1000.0;
inline constexpr Real kGrainDensity = 2500.0;
inline constexpr Real kGravity = 9.81;
inline constexpr Real kDynamicViscosity = 1.0e-3;
inline const Real kCharacteristicVelocity = 2.0 * std::sqrt(kGravity * kWaterHeight);
inline const Real kSoundSpeed = 10.0 * kCharacteristicVelocity;

inline constexpr Real kBottomWallY = 0.0;
inline constexpr Real kLeftWallX = 0.0;
inline constexpr Real kRightWallX = kTankLengthX;
inline constexpr Real kContactNormalStiffness = 5.0e4;
inline constexpr Real kContactRestitution = 0.2;
inline constexpr Real kContactTangentialStiffness = 4.0e4;
inline constexpr Real kContactTangentialDamping = 0.0;
inline constexpr Real kContactFriction = 0.5;

inline constexpr Real kCenterErrorTolerance = 1.0e-10;
inline constexpr Real kMassRelativeTolerance = 1.0e-12;
inline constexpr Real kMinimumRightFrontDisplacement = 5.0e-4;
inline constexpr Real kMinimumHydroForceNorm = 1.0e-8;
inline const std::string kRelaxedWaterReloadBodyName = "WaterBody";
inline const std::string kRelaxedGranularColumnReloadBodyName = "LammpsUnderwaterGranularColumn";

inline const BoundingBoxd kSystemDomainBounds(
    Vec2d(-kBoundaryWidth, -kBoundaryWidth),
    Vec2d(kTankLengthX + kBoundaryWidth, kTankHeightY + kBoundaryWidth));

inline std::vector<Vecd> create_water_shape()
{
    return {Vecd(0.0, 0.0),
            Vecd(0.0, kWaterHeight),
            Vecd(kTankLengthX, kWaterHeight),
            Vecd(kTankLengthX, 0.0),
            Vecd(0.0, 0.0)};
}

inline std::vector<Vecd> create_outer_wall_shape()
{
    return {Vecd(-kBoundaryWidth, -kBoundaryWidth),
            Vecd(-kBoundaryWidth, kTankHeightY + kBoundaryWidth),
            Vecd(kTankLengthX + kBoundaryWidth, kTankHeightY + kBoundaryWidth),
            Vecd(kTankLengthX + kBoundaryWidth, -kBoundaryWidth),
            Vecd(-kBoundaryWidth, -kBoundaryWidth)};
}

inline std::vector<Vecd> create_inner_wall_shape()
{
    return {Vecd(0.0, 0.0),
            Vecd(0.0, kTankHeightY),
            Vecd(kTankLengthX, kTankHeightY),
            Vecd(kTankLengthX, 0.0),
            Vecd(0.0, 0.0)};
}

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

class WaterBlock : public MultiPolygonShape
{
  public:
    explicit WaterBlock(const std::string &shape_name) : MultiPolygonShape(shape_name)
    {
        multi_polygon_.addPolygon(create_water_shape(), GeometricOps::add);
        for (const Vec2d &center : initial_dem_centers())
        {
            // Remove an initial dry buffer around each DEM grain so the lattice
            // water body does not contain sub-dp slivers between grains or walls.
            multi_polygon_.addCircle(center, kWaterVoidRadius, kCircleShapeResolution, GeometricOps::sub);
        }
    }
};

class WallBoundary : public MultiPolygonShape
{
  public:
    explicit WallBoundary(const std::string &shape_name) : MultiPolygonShape(shape_name)
    {
        multi_polygon_.addPolygon(create_outer_wall_shape(), GeometricOps::add);
        multi_polygon_.addPolygon(create_inner_wall_shape(), GeometricOps::sub);
    }
};

class GranularColumnShape : public MultiPolygonShape
{
  public:
    explicit GranularColumnShape(const std::string &shape_name) : MultiPolygonShape(shape_name)
    {
        for (const Vec2d &center : initial_dem_centers())
        {
            multi_polygon_.addCircle(center, kGrainRadius, kCircleShapeResolution, GeometricOps::add);
        }
    }
};

struct InitialWaterGeometryDiagnostics
{
    Real min_signed_distance_to_dem_surface = std::numeric_limits<Real>::max();
    UnsignedInt water_inside_dem_surface_count = 0;
    UnsignedInt water_within_half_dp_of_dem_surface_count = 0;
    UnsignedInt water_within_one_dp_of_dem_surface_count = 0;
    UnsignedInt water_in_granular_region_count = 0;
};

inline InitialWaterGeometryDiagnostics diagnose_initial_water_geometry(SPHBody &water_block)
{
    InitialWaterGeometryDiagnostics diagnostics;
    BaseParticles &particles = water_block.getBaseParticles();
    Vecd *pos = particles.ParticlePositions();
    const std::vector<Vec2d> centers = initial_dem_centers();
    Real granular_region_right = -std::numeric_limits<Real>::max();
    Real granular_region_top = -std::numeric_limits<Real>::max();
    for (const Vec2d &center : centers)
    {
        granular_region_right = std::max(granular_region_right, center[0] + kGrainRadius + kParticleSpacing);
        granular_region_top = std::max(granular_region_top, center[1] + kGrainRadius + kParticleSpacing);
    }

    for (UnsignedInt i = 0; i != particles.TotalRealParticles(); ++i)
    {
        const Vec2d water_position(pos[i][0], pos[i][1]);
        Real best_signed_distance = std::numeric_limits<Real>::max();
        for (const Vec2d &center : centers)
        {
            const Real signed_distance = (water_position - center).norm() - kGrainRadius;
            best_signed_distance = std::min(best_signed_distance, signed_distance);
        }

        diagnostics.min_signed_distance_to_dem_surface =
            std::min(diagnostics.min_signed_distance_to_dem_surface, best_signed_distance);
        if (best_signed_distance < 0.0)
        {
            ++diagnostics.water_inside_dem_surface_count;
        }
        else
        {
            if (best_signed_distance < 0.5 * kParticleSpacing)
            {
                ++diagnostics.water_within_half_dp_of_dem_surface_count;
            }
            if (best_signed_distance < kParticleSpacing)
            {
                ++diagnostics.water_within_one_dp_of_dem_surface_count;
            }
        }

        if (water_position[0] < granular_region_right && water_position[1] < granular_region_top)
        {
            ++diagnostics.water_in_granular_region_count;
        }
    }

    return diagnostics;
}

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

struct DEMParticleState
{
    int id = 0;
    Vec2d center = Vec2d::Zero();
    Vec2d velocity = Vec2d::Zero();
    Vec2d acceleration = Vec2d::Zero();
    Real omega_z = 0.0;
};

struct ExternalForceTable
{
    std::array<std::array<double, 3>, kParticleCount + 1> force_by_id{};
    int callback_calls = 0;
    int atom_updates = 0;
};

inline Real grain_area()
{
    return Pi * kGrainRadius * kGrainRadius;
}

inline Real grain_mass()
{
    return kGrainDensity * grain_area();
}

inline Real grain_weight()
{
    return grain_mass() * kGravity;
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
    auto *external = static_cast<ExternalForceTable *>(ptr);
    ++external->callback_calls;

    for (int i = 0; i < nlocal; ++i)
    {
        fexternal[i][0] = 0.0;
        fexternal[i][1] = 0.0;
        fexternal[i][2] = 0.0;

        const int id = static_cast<int>(ids[i]);
        if (id >= 1 && id <= kParticleCount)
        {
            fexternal[i][0] = external->force_by_id[id][0];
            fexternal[i][1] = external->force_by_id[id][1];
            fexternal[i][2] = 0.0;
            ++external->atom_updates;
        }
    }
}

class LammpsGranularColumnAdapter
{
  public:
    LammpsGranularColumnAdapter()
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
             << "fix ext all external pf/callback 1 1\n"
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

    void setExternalForces(const std::vector<Vec2d> &forces)
    {
        if (static_cast<int>(forces.size()) != kParticleCount)
        {
            throw std::runtime_error("hydrodynamic force vector size does not match DEM particle count");
        }

        for (int id = 1; id <= kParticleCount; ++id)
        {
            const Vec2d &force = forces[id - 1];
            external_force_.force_by_id[id][0] = force[0];
            external_force_.force_by_id[id][1] = force[1];
            external_force_.force_by_id[id][2] = 0.0;
        }
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
        std::array<double, 3 * kParticleCount> x{};
        std::array<double, 3 * kParticleCount> v{};
        std::array<double, 3 * kParticleCount> f{};
        std::array<double, 3 * kParticleCount> omega{};

        lammps_gather_atoms(lammps_.get(), "x", 1, 3, x.data());
        lammps_.throw_if_error("gather atom positions");
        lammps_gather_atoms(lammps_.get(), "v", 1, 3, v.data());
        lammps_.throw_if_error("gather atom velocities");
        lammps_gather_atoms(lammps_.get(), "f", 1, 3, f.data());
        lammps_.throw_if_error("gather atom forces");
        lammps_gather_atoms(lammps_.get(), "omega", 1, 3, omega.data());
        lammps_.throw_if_error("gather atom angular velocities");

        std::vector<DEMParticleState> states;
        states.reserve(kParticleCount);
        for (int i = 0; i < kParticleCount; ++i)
        {
            DEMParticleState state;
            state.id = i + 1;
            state.center = to_vec2d(x, i);
            state.velocity = to_vec2d(v, i);
            state.acceleration = to_vec2d(f, i) / grain_mass();
            state.omega_z = static_cast<Real>(omega[3 * i + 2]);
            states.push_back(state);
        }
        return states;
    }

    int version() const { return lammps_.version(); }

    const ExternalForceTable &externalForceTable() const { return external_force_; }

  private:
    LammpsInstance lammps_;
    ExternalForceTable external_force_;
};

class DrivenGranularColumnBoundary
{
  public:
    DrivenGranularColumnBoundary(SolidBody &column_body, const std::vector<Vec2d> &initial_centers)
        : column_body_(column_body), particles_(column_body.getBaseParticles()),
          pos_(particles_.ParticlePositions()),
          vel_(particles_.registerStateVariableData<Vecd>("Velocity")),
          acc_(particles_.registerStateVariableData<Vecd>("Acceleration")),
          initial_centers_(initial_centers)
    {
        if (particles_.TotalRealParticles() == 0)
        {
            throw std::runtime_error("SPHinXsys granular column body generated zero particles");
        }

        particle_dem_ids_.resize(particles_.TotalRealParticles());
        relative_positions_.resize(particles_.TotalRealParticles(), Vec2d::Zero());
        particle_counts_.assign(kParticleCount, 0);
        std::vector<Vec2d> center_sums(kParticleCount, Vec2d::Zero());

        for (UnsignedInt i = 0; i != particles_.TotalRealParticles(); ++i)
        {
            const int id = nearest_dem_id(pos_[i]);
            particle_dem_ids_[i] = id;
            center_sums[id] += pos_[i];
            ++particle_counts_[id];
        }

        for (int id = 0; id < kParticleCount; ++id)
        {
            if (particle_counts_[id] == 0)
            {
                throw std::runtime_error("a DEM grain received zero SPHinXsys proxy particles");
            }
            const Vec2d proxy_center = center_sums[id] / static_cast<Real>(particle_counts_[id]);
            const Vec2d offset = initial_centers_[id] - proxy_center;
            for (UnsignedInt i = 0; i != particles_.TotalRealParticles(); ++i)
            {
                if (particle_dem_ids_[i] == id)
                {
                    pos_[i] += offset;
                }
            }
        }

        for (UnsignedInt i = 0; i != particles_.TotalRealParticles(); ++i)
        {
            const int id = particle_dem_ids_[i];
            relative_positions_[i] = pos_[i] - initial_centers_[id];
            vel_[i] = Vec2d::Zero();
            acc_[i] = Vec2d::Zero();
        }
    }

    void update(const std::vector<DEMParticleState> &states)
    {
        for (UnsignedInt i = 0; i != particles_.TotalRealParticles(); ++i)
        {
            const int id = particle_dem_ids_[i];
            pos_[i] = states[id].center + relative_positions_[i];
            vel_[i] = states[id].velocity;
            acc_[i] = states[id].acceleration;
        }
        column_body_.setNewlyUpdated();
    }

    std::vector<Vec2d> geometricCenters() const
    {
        std::vector<Vec2d> center_sums(kParticleCount, Vec2d::Zero());
        std::vector<int> counts(kParticleCount, 0);
        for (UnsignedInt i = 0; i != particles_.TotalRealParticles(); ++i)
        {
            const int id = particle_dem_ids_[i];
            center_sums[id] += pos_[i];
            ++counts[id];
        }

        std::vector<Vec2d> centers(kParticleCount, Vec2d::Zero());
        for (int id = 0; id < kParticleCount; ++id)
        {
            centers[id] = center_sums[id] / static_cast<Real>(counts[id]);
        }
        return centers;
    }

    const std::vector<int> &particleDemIds() const { return particle_dem_ids_; }

    UnsignedInt particleCount() const { return particles_.TotalRealParticles(); }

  private:
    int nearest_dem_id(const Vec2d &position) const
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
    std::vector<Vec2d> initial_centers_;
    std::vector<int> particle_dem_ids_;
    std::vector<Vec2d> relative_positions_;
    std::vector<int> particle_counts_;
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

inline std::vector<Vec2d> sum_grain_hydro_forces(SPHBody &grain_boundary,
                                                 const std::vector<int> &particle_dem_ids)
{
    BaseParticles &particles = grain_boundary.getBaseParticles();
    Vecd *pressure_force = particles.getVariableDataByName<Vecd>("PressureForceFromFluid");
    Vecd *viscous_force = particles.getVariableDataByName<Vecd>("ViscousForceFromFluid");

    std::vector<Vec2d> forces(kParticleCount, Vec2d::Zero());
    for (UnsignedInt i = 0; i != particles.TotalRealParticles(); ++i)
    {
        const Vecd particle_force = pressure_force[i] + viscous_force[i];
        forces[particle_dem_ids[i]] += Vec2d(particle_force[0], particle_force[1]);
    }
    return forces;
}

inline Real max_center_sync_error(const std::vector<DEMParticleState> &states,
                                  const std::vector<Vec2d> &proxy_centers)
{
    Real error = 0.0;
    for (int i = 0; i < kParticleCount; ++i)
    {
        error = std::max(error, (states[i].center - proxy_centers[i]).norm());
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

inline Real max_hydro_force_norm(const std::vector<Vec2d> &forces)
{
    Real max_norm = 0.0;
    for (const Vec2d &force : forces)
    {
        max_norm = std::max(max_norm, force.norm());
    }
    return max_norm;
}

inline bool is_finite(const Vec2d &value)
{
    return std::isfinite(value[0]) && std::isfinite(value[1]);
}

inline bool all_finite(const std::vector<DEMParticleState> &states,
                       const std::vector<Vec2d> &forces)
{
    for (const DEMParticleState &state : states)
    {
        if (!is_finite(state.center) || !is_finite(state.velocity) || !is_finite(state.acceleration) ||
            !std::isfinite(state.omega_z))
        {
            return false;
        }
    }
    for (const Vec2d &force : forces)
    {
        if (!is_finite(force))
        {
            return false;
        }
    }
    return true;
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

inline std::filesystem::path reload_particle_file()
{
    return std::filesystem::path(IO::getEnvironment().ReloadFolder()) / "Reload.xml";
}

inline bool reload_particle_file_exists()
{
    return std::filesystem::exists(reload_particle_file());
}

inline bool reload_particle_body_exists(const std::string &body_name)
{
    if (!reload_particle_file_exists())
    {
        return false;
    }

    std::ifstream reload_file(reload_particle_file());
    if (!reload_file)
    {
        return false;
    }

    std::ostringstream buffer;
    buffer << reload_file.rdbuf();
    return buffer.str().find("name=\"" + body_name + "\"") != std::string::npos;
}

inline std::string generate_water_particles(SPHSystem &sph_system, FluidBody &water_block)
{
    if (!sph_system.RunParticleRelaxation() && sph_system.ReloadParticles() &&
        reload_particle_body_exists(kRelaxedWaterReloadBodyName))
    {
        water_block.generateParticles<BaseParticles, Reload>(kRelaxedWaterReloadBodyName);
        return "Reload";
    }

    water_block.generateParticles<BaseParticles, Lattice>();
    return "Lattice";
}

inline std::string generate_granular_column_boundary_particles(SPHSystem &sph_system,
                                                               SolidBody &grain_boundary)
{
    grain_boundary.defineAdaptationRatios(1.15, 1.0);
    grain_boundary.defineBodyLevelSetShape().writeLevelSet();

    if (!sph_system.RunParticleRelaxation() && sph_system.ReloadParticles() &&
        reload_particle_body_exists(kRelaxedGranularColumnReloadBodyName))
    {
        grain_boundary.generateParticles<BaseParticles, Reload>(kRelaxedGranularColumnReloadBodyName);
        return "Reload";
    }

    grain_boundary.generateParticles<BaseParticles, Lattice>();
    return "Lattice";
}

inline int run_underwater_granular_particle_relaxation(FluidBody &water_block,
                                                       SolidBody &grain_boundary,
                                                       SolidBody &wall_boundary)
{
    (void)wall_boundary;
    InnerRelation water_inner(water_block);
    InnerRelation grain_inner(grain_boundary);

    using namespace relax_dynamics;
    SimpleDynamics<RandomizeParticlePosition> random_water_particles(water_block);
    SimpleDynamics<RandomizeParticlePosition> random_grain_particles(grain_boundary);
    RelaxationStepLevelSetCorrectionInner water_relaxation_step(water_inner);
    RelaxationStepInner grain_relaxation_step(grain_inner);
    BodyStatesRecordingToVtp write_relaxed_state(water_block.getSPHSystem());
    ReloadParticleIO write_particle_reload_files(SPHBodyVector{&water_block, &grain_boundary});

    random_water_particles.exec(0.25);
    random_grain_particles.exec(0.25);
    water_relaxation_step.SurfaceBounding().exec();
    grain_relaxation_step.SurfaceBounding().exec();
    write_relaxed_state.writeToFile(0);

    int ite_p = 0;
    while (ite_p < kRelaxationSteps)
    {
        if (ite_p < kWaterRelaxationSteps)
        {
            water_relaxation_step.exec();
        }
        grain_relaxation_step.exec();
        ite_p += 1;
        if (ite_p % kRelaxationOutputInterval == 0)
        {
            std::cout << std::fixed << std::setprecision(9)
                      << "Relaxation steps for the LAMMPS underwater water-grain system N = "
                      << ite_p << "\n";
            write_relaxed_state.writeToFile(ite_p);
        }
    }

    write_particle_reload_files.writeToFile(0);
    const std::filesystem::path reload_path = reload_particle_file();
    if (!std::filesystem::exists(reload_path) ||
        !reload_particle_body_exists(kRelaxedWaterReloadBodyName) ||
        !reload_particle_body_exists(kRelaxedGranularColumnReloadBodyName))
    {
        throw std::runtime_error("particle relaxation did not write both WaterBody and granular column reload data");
    }

    const InitialWaterGeometryDiagnostics relaxed_water_geometry =
        diagnose_initial_water_geometry(water_block);

    std::cout << std::setprecision(17);
    std::cout << "SPHinXsys 2D LAMMPS underwater water-grain particle relaxation\n";
    std::cout << "water_particles: " << water_block.getBaseParticles().TotalRealParticles() << '\n';
    std::cout << "grain_proxy_particles: " << grain_boundary.getBaseParticles().TotalRealParticles() << '\n';
    std::cout << "dem_particle_count: " << kParticleCount << '\n';
    std::cout << "particle_spacing_m: " << kParticleSpacing << '\n';
    std::cout << "initial_water_grain_gap_m: " << kInitialWaterGrainGap << '\n';
    std::cout << "water_void_radius_m: " << kWaterVoidRadius << '\n';
    std::cout << "relaxed_water_min_signed_distance_to_dem_surface_m: "
              << relaxed_water_geometry.min_signed_distance_to_dem_surface << '\n';
    std::cout << "relaxed_water_inside_dem_surface_count: "
              << relaxed_water_geometry.water_inside_dem_surface_count << '\n';
    std::cout << "relaxed_water_within_half_dp_of_dem_surface_count: "
              << relaxed_water_geometry.water_within_half_dp_of_dem_surface_count << '\n';
    std::cout << "relaxed_water_within_one_dp_of_dem_surface_count: "
              << relaxed_water_geometry.water_within_one_dp_of_dem_surface_count << '\n';
    std::cout << "relaxation_steps: " << kRelaxationSteps << '\n';
    std::cout << "water_relaxation_steps: " << kWaterRelaxationSteps << '\n';
    std::cout << "relaxation_output_interval: " << kRelaxationOutputInterval << '\n';
    std::cout << "reload_water_body_name: " << kRelaxedWaterReloadBodyName << '\n';
    std::cout << "reload_grain_body_name: " << kRelaxedGranularColumnReloadBodyName << '\n';
    std::cout << "reload_file: " << std::filesystem::absolute(reload_path).string() << '\n';
    std::cout << "VTP_output_folder: " << std::filesystem::absolute(IO::getEnvironment().OutputFolder()).string() << '\n';
    std::cout << "status: RELAXATION_PASS\n";
    return 0;
}

struct ForceApplication
{
    std::vector<Vec2d> applied_forces;
    std::vector<int> capped_flags;

    int cap_count() const
    {
        return std::accumulate(capped_flags.begin(), capped_flags.end(), 0);
    }
};

inline ForceApplication relax_and_cap_forces(const std::vector<Vec2d> &raw_forces,
                                             const std::vector<Vec2d> &previous_applied_forces)
{
    if (static_cast<int>(raw_forces.size()) != kParticleCount ||
        static_cast<int>(previous_applied_forces.size()) != kParticleCount)
    {
        throw std::runtime_error("force relaxation received a vector with the wrong DEM particle count");
    }

    ForceApplication result;
    result.applied_forces.resize(kParticleCount, Vec2d::Zero());
    result.capped_flags.resize(kParticleCount, 0);
    const Real cap = kForceCapWeightFactor * grain_weight();

    for (int i = 0; i < kParticleCount; ++i)
    {
        Vec2d relaxed_force =
            kForceRelaxationAlpha * raw_forces[i] +
            (1.0 - kForceRelaxationAlpha) * previous_applied_forces[i];
        const Real relaxed_norm = relaxed_force.norm();
        if (relaxed_norm > cap && relaxed_norm > TinyReal)
        {
            relaxed_force *= cap / relaxed_norm;
            result.capped_flags[i] = 1;
        }
        result.applied_forces[i] = relaxed_force;
    }

    return result;
}

inline std::filesystem::path write_dem_force_vtp(int iteration,
                                                 Real time,
                                                 const std::vector<DEMParticleState> &states,
                                                 const std::vector<Vec2d> &raw_hydro_forces,
                                                 const std::vector<Vec2d> &applied_hydro_forces)
{
    return write_dem_points_fields_to_vtp(
        iteration,
        time,
        "DEM_Forces",
        centers_from_states(states),
        {{"HydrodynamicForceRaw", raw_hydro_forces},
         {"HydrodynamicForceApplied", applied_hydro_forces},
         {"Velocity", velocities_from_states(states)}},
        {{"ParticleId", ids_as_scalars()},
         {"Radius", scalar_filled(kGrainRadius)},
         {"MassPerUnitDepth", scalar_filled(grain_mass())}},
        "HydrodynamicForceApplied");
}

inline void write_motion_csv_header(std::ofstream &csv)
{
    csv << "frame,time_s,relaxation_phase,coupling_active,lammps_step,particle_id,"
           "x_m,y_m,vx_m_per_s,vy_m_per_s,omega_z_rad_per_s,"
           "proxy_center_x_m,proxy_center_y_m,center_error_m\n";
}

inline void write_force_csv_header(std::ofstream &csv)
{
    csv << "frame,time_s,relaxation_phase,coupling_active,particle_id,"
           "Fx_raw_N,Fy_raw_N,raw_force_norm_N,"
           "Fx_applied_N,Fy_applied_N,applied_force_norm_N,"
           "grain_weight_N,force_cap_N,capped\n";
}

inline void write_motion_csv_samples(std::ofstream &csv,
                                     int frame,
                                     Real time,
                                     bool coupling_active,
                                     int lammps_step,
                                     const std::vector<DEMParticleState> &states,
                                     const std::vector<Vec2d> &proxy_centers)
{
    const int relaxation_phase = coupling_active ? 0 : 1;
    for (int i = 0; i < kParticleCount; ++i)
    {
        const Real center_error = (states[i].center - proxy_centers[i]).norm();
        csv << frame << ','
            << time << ','
            << relaxation_phase << ','
            << (coupling_active ? 1 : 0) << ','
            << lammps_step << ','
            << states[i].id << ','
            << states[i].center[0] << ','
            << states[i].center[1] << ','
            << states[i].velocity[0] << ','
            << states[i].velocity[1] << ','
            << states[i].omega_z << ','
            << proxy_centers[i][0] << ','
            << proxy_centers[i][1] << ','
            << center_error << '\n';
    }
}

inline void write_force_csv_samples(std::ofstream &csv,
                                    int frame,
                                    Real time,
                                    bool coupling_active,
                                    const std::vector<Vec2d> &raw_hydro_forces,
                                    const std::vector<Vec2d> &applied_hydro_forces,
                                    const std::vector<int> &capped_flags)
{
    const int relaxation_phase = coupling_active ? 0 : 1;
    for (int i = 0; i < kParticleCount; ++i)
    {
        csv << frame << ','
            << time << ','
            << relaxation_phase << ','
            << (coupling_active ? 1 : 0) << ','
            << i + 1 << ','
            << raw_hydro_forces[i][0] << ','
            << raw_hydro_forces[i][1] << ','
            << raw_hydro_forces[i].norm() << ','
            << applied_hydro_forces[i][0] << ','
            << applied_hydro_forces[i][1] << ','
            << applied_hydro_forces[i].norm() << ','
            << grain_weight() << ','
            << kForceCapWeightFactor * grain_weight() << ','
            << capped_flags[i] << '\n';
    }
}
} // namespace LammpsUnderwaterGranularCollapse2D
