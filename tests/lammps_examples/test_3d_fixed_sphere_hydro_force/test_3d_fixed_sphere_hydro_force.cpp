#include "sphinxsys.h"

#include <algorithm>
#include <cmath>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace SPH;

namespace {

constexpr Real kTankLengthX = 0.12;
constexpr Real kTankLengthY = 0.12;
constexpr Real kTankHeightZ = 0.16;
constexpr Real kWaterHeight = 0.12;
constexpr Real kSphereRadius = 0.01;
constexpr Real kSphereDiameter = 2.0 * kSphereRadius;
constexpr Real kParticleSpacing = kSphereDiameter / 10.0;
constexpr Real kBoundaryWidth = 4.0 * kParticleSpacing;
constexpr Real kWaterDensity = 1000.0;
constexpr Real kGravity = 9.81;
constexpr Real kDynamicViscosity = 1.0e-3;
const Real kCharacteristicVelocity = 2.0 * std::sqrt(kGravity * kWaterHeight);
const Real kSoundSpeed = 10.0 * kCharacteristicVelocity;
constexpr int kSamples = 20;
constexpr Real kSampleInterval = 1.0e-4;
constexpr Real kRelativeErrorTolerance = 0.75;
constexpr Real kLateralForceToleranceRatio = 0.35;
constexpr int kRelaxationSteps = 1000;
constexpr int kRelaxationOutputInterval = 200;

const Vec3d kSphereCenter(0.5 * kTankLengthX, 0.5 * kTankLengthY, 0.06);
const BoundingBoxd kSystemDomainBounds(
    Vec3d(-kBoundaryWidth, -kBoundaryWidth, -kBoundaryWidth),
    Vec3d(kTankLengthX + kBoundaryWidth, kTankLengthY + kBoundaryWidth, kTankHeightZ + kBoundaryWidth));

Real sphere_volume()
{
    return (4.0 / 3.0) * Pi * kSphereRadius * kSphereRadius * kSphereRadius;
}

Real theoretical_buoyancy()
{
    return kWaterDensity * kGravity * sphere_volume();
}

std::filesystem::path reload_particle_file()
{
    return std::filesystem::path(IO::getEnvironment().ReloadFolder()) / "Reload.xml";
}

bool reload_particle_file_exists()
{
    return std::filesystem::exists(reload_particle_file());
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
        subtract<GeometricShapeBall>(kSphereCenter, kSphereRadius, "SphereVoid");
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

struct ForceSample
{
    Real time = 0.0;
    Vec3d force = Vec3d::Zero();
    Real theoretical = 0.0;
    Real relative_error = 0.0;
};

Vec3d sum_sphere_hydro_force(SPHBody &sphere)
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

void write_csv_header(std::ofstream &csv)
{
    csv << "time,Fx,Fy,Fz,theoretical_buoyancy,relative_error\n";
}

void write_csv_sample(std::ofstream &csv, const ForceSample &sample)
{
    csv << sample.time << ','
        << sample.force[0] << ','
        << sample.force[1] << ','
        << sample.force[2] << ','
        << sample.theoretical << ','
        << sample.relative_error << '\n';
}

ForceSample make_sample(Real time, const Vec3d &force)
{
    ForceSample sample;
    sample.time = time;
    sample.force = force;
    sample.theoretical = theoretical_buoyancy();
    sample.relative_error = std::abs(force[2] - sample.theoretical) / sample.theoretical;
    return sample;
}

Vec3d average_force(const std::vector<ForceSample> &samples)
{
    Vec3d sum = Vec3d::Zero();
    for (const ForceSample &sample : samples)
    {
        sum += sample.force;
    }
    return sum / static_cast<Real>(samples.size());
}

std::string generate_sphere_boundary_particles(SPHSystem &sph_system, SolidBody &sphere_boundary)
{
    if (!sph_system.RunParticleRelaxation() && sph_system.ReloadParticles() && reload_particle_file_exists())
    {
        sphere_boundary.generateParticles<BaseParticles, Reload>(sphere_boundary.Name());
        return "Reload";
    }

    sphere_boundary.defineBodyLevelSetShape().writeLevelSet();
    sphere_boundary.generateParticles<BaseParticles, Lattice>();

    return "Lattice";
}

int run_sphere_particle_relaxation(SPHSystem &sph_system, SolidBody &sphere_boundary)
{
    InnerRelation sphere_inner(sphere_boundary);

    using namespace relax_dynamics;
    SimpleDynamics<RandomizeParticlePosition> random_sphere_particles(sphere_boundary);
    RelaxationStepInner relaxation_step_inner(sphere_inner);
    BodyStatesRecordingToVtp write_sphere_state(sphere_boundary);
    ReloadParticleIO write_particle_reload_files(sphere_boundary);

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
                      << "Relaxation steps for the fixed sphere boundary N = "
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
    std::cout << "SPHinXsys fixed sphere particle relaxation\n";
    std::cout << "sphere_particles: " << sphere_boundary.getBaseParticles().TotalRealParticles() << '\n';
    std::cout << "particle_spacing_m: " << kParticleSpacing << '\n';
    std::cout << "relaxation_steps: " << kRelaxationSteps << '\n';
    std::cout << "reload_file: " << std::filesystem::absolute(reload_path).string() << '\n';
    std::cout << "VTP_output_folder: " << std::filesystem::absolute(IO::getEnvironment().OutputFolder()).string() << '\n';
    std::cout << "status: RELAXATION_PASS\n";
    return 0;
}

} // namespace

int main(int ac, char *av[])
{
    try
    {
        SPHSystem sph_system(kSystemDomainBounds, kParticleSpacing);
        sph_system.setRunParticleRelaxation(false);
        sph_system.setReloadParticles(reload_particle_file_exists());
        sph_system.handleCommandlineOptions(ac, av);

        if (sph_system.RunParticleRelaxation())
        {
            sph_system.setReloadParticles(false);
        }
        else if (sph_system.ReloadParticles() && !reload_particle_file_exists())
        {
            std::cout << "WARNING: particle reload was requested, but "
                      << std::filesystem::absolute(reload_particle_file()).string()
                      << " was not found. Falling back to lattice particles.\n";
            sph_system.setReloadParticles(false);
        }

        SolidBody sphere_boundary(
            sph_system, makeShared<GeometricShapeBall>(kSphereCenter, kSphereRadius, "FixedSphereBoundary"));
        sphere_boundary.defineMatterMaterial<Solid>(kWaterDensity);
        const std::string sphere_particle_source = generate_sphere_boundary_particles(sph_system, sphere_boundary);

        if (sph_system.RunParticleRelaxation())
        {
            return run_sphere_particle_relaxation(sph_system, sphere_boundary);
        }

        FluidBody water_block(sph_system, makeShared<WaterBlock>("WaterBody"));
        water_block.defineMatterMaterial<WeaklyCompressibleFluid>(kWaterDensity, kSoundSpeed);
        water_block.addMaterialProperty<Viscosity>(kDynamicViscosity);
        water_block.generateParticles<BaseParticles, Lattice>();

        SolidBody wall_boundary(sph_system, makeShared<WallBoundary>("WallBoundary"));
        wall_boundary.defineMatterMaterial<Solid>();
        wall_boundary.generateParticles<BaseParticles, Lattice>();

        ContactRelation sphere_contact(sphere_boundary, {&water_block});

        SimpleDynamics<HydrostaticPressureField> hydrostatic_pressure(water_block);
        Gravity gravity(Vec3d(0.0, 0.0, -kGravity));
        SimpleDynamics<GravityForce<Gravity>> constant_gravity(water_block, gravity);
        SimpleDynamics<NormalDirectionFromBodyShape> wall_normal_direction(wall_boundary);
        SimpleDynamics<NormalDirectionFromBodyShape> sphere_normal_direction(sphere_boundary);

        InteractionWithUpdate<solid_dynamics::ViscousForceFromFluid> viscous_force_on_sphere(sphere_contact);
        InteractionWithUpdate<solid_dynamics::PressureForceFromFluid<fluid_dynamics::Integration2ndHalfInnerNoRiemann>>
            pressure_force_on_sphere(sphere_contact);

        BodyStatesRecordingToVtp write_real_body_states(sph_system);
        write_real_body_states.addToWrite<Real>(water_block, "Pressure");
        write_real_body_states.addToWrite<Vecd>(sphere_boundary, "PressureForceFromFluid");
        write_real_body_states.addToWrite<Vecd>(sphere_boundary, "ViscousForceFromFluid");
        write_real_body_states.addToWrite<Vecd>(sphere_boundary, "NormalDirection");

        sph_system.initializeSystemCellLinkedLists();
        sph_system.initializeSystemConfigurations();
        hydrostatic_pressure.exec();
        constant_gravity.exec();
        wall_normal_direction.exec();
        sphere_normal_direction.exec();

        const std::filesystem::path csv_path = "sphere_force.csv";
        std::ofstream csv(csv_path);
        if (!csv)
        {
            throw std::runtime_error("could not open sphere_force.csv for writing");
        }
        csv << std::setprecision(17);
        write_csv_header(csv);

        std::vector<ForceSample> samples;
        samples.reserve(kSamples + 1);

        Real &physical_time = *sph_system.getSystemVariableDataByName<Real>("PhysicalTime");
        for (int sample_index = 0; sample_index <= kSamples; ++sample_index)
        {
            physical_time = sample_index * kSampleInterval;
            hydrostatic_pressure.exec();
            constant_gravity.exec();
            sphere_contact.updateConfiguration();
            viscous_force_on_sphere.exec();
            pressure_force_on_sphere.exec();

            ForceSample sample = make_sample(physical_time, sum_sphere_hydro_force(sphere_boundary));
            write_csv_sample(csv, sample);
            samples.push_back(sample);

            if (sample_index == 0 || sample_index == kSamples)
            {
                write_real_body_states.writeToFile(sample_index);
            }
        }
        csv.close();

        const Vec3d mean_force = average_force(samples);
        const Real buoyancy = theoretical_buoyancy();
        const Real relative_error = std::abs(mean_force[2] - buoyancy) / buoyancy;
        const Real lateral_ratio = std::sqrt(mean_force[0] * mean_force[0] + mean_force[1] * mean_force[1]) /
                                   (std::abs(mean_force[2]) + TinyReal);

        std::cout << std::setprecision(17);
        std::cout << "SPHinXsys fixed sphere hydro force example\n";
        std::cout << "CSV: " << std::filesystem::absolute(csv_path).string() << '\n';
        std::cout << "VTP_output_folder: " << std::filesystem::absolute(IO::getEnvironment().OutputFolder()).string() << '\n';
        std::cout << "water_particles: " << water_block.getBaseParticles().TotalRealParticles() << '\n';
        std::cout << "sphere_particles: " << sphere_boundary.getBaseParticles().TotalRealParticles() << '\n';
        std::cout << "sphere_particle_source: " << sphere_particle_source << '\n';
        std::cout << "particle_spacing_m: " << kParticleSpacing << '\n';
        std::cout << "theoretical_buoyancy_N: " << buoyancy << '\n';
        std::cout << "mean_force_N: "
                  << mean_force[0] << ','
                  << mean_force[1] << ','
                  << mean_force[2] << '\n';
        std::cout << "relative_error: " << relative_error << '\n';
        std::cout << "lateral_force_ratio: " << lateral_ratio << '\n';

        if (mean_force[2] <= 0.0)
        {
            std::cerr << "ERROR: mean hydro force on the sphere is not upward.\n";
            return 1;
        }
        if (relative_error > kRelativeErrorTolerance)
        {
            std::cerr << "ERROR: relative buoyancy error exceeded tolerance "
                      << kRelativeErrorTolerance << ".\n";
            return 1;
        }
        if (lateral_ratio > kLateralForceToleranceRatio)
        {
            std::cerr << "ERROR: lateral force ratio exceeded tolerance "
                      << kLateralForceToleranceRatio << ".\n";
            return 1;
        }

        std::cout << "status: PASS\n";
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "ERROR: " << e.what() << '\n';
        return 1;
    }
}
