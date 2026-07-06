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

namespace {

constexpr double kGravity = 9.81;
constexpr double kDiameter = 0.02;
constexpr double kRadius = 0.5 * kDiameter;
constexpr double kDensity = 2500.0;
constexpr double kTimeStep = 1.0e-5;
constexpr int kTotalSteps = 1000;
constexpr int kSampleEvery = 100;
constexpr double kErrorTolerance = 1.0e-9;

const Vec3d kInitialCenter(0.0, 0.0, 1.0);

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

struct Sample
{
    int step = 0;
    double time = 0.0;
    DEMState dem_state;
    Vec3d sph_geometric_center = Vec3d::Zero();
    double center_error = 0.0;
    double z_exact = 0.0;
    double vz_exact = 0.0;
    double z_error = 0.0;
    double vz_error = 0.0;
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

    void commands_string(const char *cmds, const std::string &context) const
    {
        lammps_commands_string(handle_, cmds);
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

Vec3d to_vec3d(const std::array<double, 3> &values)
{
    return Vec3d(values[0], values[1], values[2]);
}

double sphere_mass()
{
    return kDensity * (4.0 / 3.0) * Pi * kRadius * kRadius * kRadius;
}

class LammpsDEMAdapter
{
  public:
    LammpsDEMAdapter()
    {
        lammps_.commands_string(R"lmp(
units si
atom_style sphere
atom_modify map array
boundary f f f
newton off
comm_modify vel yes

region box block -0.1 0.1 -0.1 0.1 -1.0 2.0 units box
create_box 1 box
create_atoms 1 single 0.0 0.0 1.0 units box
set atom 1 diameter 0.02 density 2500.0
velocity all set 0.0 0.0 0.0 units box

pair_style zero 0.1
pair_coeff * *
neighbor 0.01 bin
neigh_modify delay 0 every 1 check yes

fix int all nve/sphere
fix grav all gravity 9.81 vector 0.0 0.0 -1.0
fix ext all external pf/callback 1 1

timestep 1.0e-5
thermo_style custom step atoms zlo zhi
thermo 100
)lmp",
                                "LAMMPS initialization commands");

        lammps_set_fix_external_callback(lammps_.get(), "ext", &external_force_callback, &external_force_);
        lammps_.throw_if_error("lammps_set_fix_external_callback");
    }

    void runSubsteps(int steps)
    {
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

Sample make_sample(int step, const DEMState &dem_state, const Vec3d &sph_geometric_center)
{
    Sample sample;
    sample.step = step;
    sample.time = step * kTimeStep;
    sample.dem_state = dem_state;
    sample.sph_geometric_center = sph_geometric_center;
    sample.center_error = (sph_geometric_center - dem_state.center).norm();
    sample.z_exact = kInitialCenter[2] - 0.5 * kGravity * sample.time * sample.time;
    sample.vz_exact = -kGravity * sample.time;
    sample.z_error = dem_state.center[2] - sample.z_exact;
    sample.vz_error = dem_state.velocity[2] - sample.vz_exact;
    return sample;
}

void write_csv_header(std::ofstream &csv)
{
    csv << "step,time_s,"
           "lammps_x_m,lammps_y_m,lammps_z_m,"
           "lammps_vx_m_per_s,lammps_vy_m_per_s,lammps_vz_m_per_s,"
           "lammps_omega_x_rad_per_s,lammps_omega_y_rad_per_s,lammps_omega_z_rad_per_s,"
           "sph_center_x_m,sph_center_y_m,sph_center_z_m,"
           "center_error_m,z_exact_m,vz_exact_m_per_s,z_error_m,vz_error_m_per_s\n";
}

void write_csv_sample(std::ofstream &csv, const Sample &sample)
{
    csv << sample.step << ','
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
        << sample.z_exact << ','
        << sample.vz_exact << ','
        << sample.z_error << ','
        << sample.vz_error << '\n';
}

} // namespace

int main(int ac, char *av[])
{
    try
    {
        BoundingBoxd system_domain_bounds(Vec3d(-0.05, -0.05, 0.90), Vec3d(0.05, 0.05, 1.05));
        SPHSystem sph_system(system_domain_bounds, kDiameter / 4.0);
        sph_system.setRunParticleRelaxation(false);
        sph_system.setReloadParticles(false);
        sph_system.handleCommandlineOptions(ac, av);

        SolidBody sphere_boundary(
            sph_system, makeShared<GeometricShapeBall>(kInitialCenter, kRadius, "LammpsDrivenSphere"));
        sphere_boundary.defineMatterMaterial<Solid>(kDensity);
        sphere_boundary.generateParticles<BaseParticles, Lattice>();

        DrivenSphereBoundary driven_sphere(sphere_boundary, kInitialCenter);

        BodyStatesRecordingToVtp write_sphere_state(sphere_boundary);
        write_sphere_state.addToWrite<Vecd>(sphere_boundary, "Velocity");

        LammpsDEMAdapter dem_adapter;
        const std::filesystem::path csv_path = "sphere_motion.csv";
        const std::filesystem::path dll_path = "liblammps.dll";
        const std::filesystem::path output_path = std::filesystem::absolute(IO::getEnvironment().OutputFolder());

        std::ofstream csv(csv_path);
        if (!csv)
        {
            throw std::runtime_error("could not open sphere_motion.csv for writing");
        }
        csv << std::setprecision(17);
        write_csv_header(csv);

        double max_center_error = 0.0;
        double max_abs_z_error = 0.0;
        double max_abs_vz_error = 0.0;

        auto record = [&](int step)
        {
            sph_system.svPhysicalTime().setValue(step * kTimeStep);
            const DEMState dem_state = dem_adapter.pullState();
            driven_sphere.update(dem_state);
            write_sphere_state.writeToFile(step);

            Sample sample = make_sample(step, dem_state, driven_sphere.geometricCenter());
            write_csv_sample(csv, sample);

            max_center_error = std::max(max_center_error, std::abs(sample.center_error));
            max_abs_z_error = std::max(max_abs_z_error, std::abs(sample.z_error));
            max_abs_vz_error = std::max(max_abs_vz_error, std::abs(sample.vz_error));
            return sample;
        };

        Sample final_sample = record(0);
        for (int step = kSampleEvery; step <= kTotalSteps; step += kSampleEvery)
        {
            dem_adapter.runSubsteps(kSampleEvery);
            final_sample = record(step);
        }

        csv.close();

        const ExternalForce &external_force = dem_adapter.externalForce();

        std::cout << std::setprecision(17);
        std::cout << "SPHinXsys LAMMPS-driven sphere boundary example\n";
        std::cout << "liblammps_dll_in_working_directory: " << (std::filesystem::exists(dll_path) ? "yes" : "no") << '\n';
        std::cout << "lammps_version: " << dem_adapter.version() << '\n';
        std::cout << "CSV: " << std::filesystem::absolute(csv_path).string() << '\n';
        std::cout << "VTP_output_folder: " << output_path.string() << '\n';
        std::cout << "sph_sphere_particles: " << driven_sphere.particleCount() << '\n';
        std::cout << "mass_kg: " << sphere_mass() << '\n';
        std::cout << "external_force_N: "
                  << external_force.force[0] << ','
                  << external_force.force[1] << ','
                  << external_force.force[2] << '\n';
        std::cout << "callback_calls: " << external_force.callback_calls << '\n';
        std::cout << "atom1_force_updates: " << external_force.atom1_updates << '\n';
        std::cout << "final_step: " << final_sample.step << '\n';
        std::cout << "final_lammps_center_m: "
                  << final_sample.dem_state.center[0] << ','
                  << final_sample.dem_state.center[1] << ','
                  << final_sample.dem_state.center[2] << '\n';
        std::cout << "final_sph_geometric_center_m: "
                  << final_sample.sph_geometric_center[0] << ','
                  << final_sample.sph_geometric_center[1] << ','
                  << final_sample.sph_geometric_center[2] << '\n';
        std::cout << "final_vz_m_per_s: " << final_sample.dem_state.velocity[2] << '\n';
        std::cout << "final_z_exact_m: " << final_sample.z_exact << '\n';
        std::cout << "final_vz_exact_m_per_s: " << final_sample.vz_exact << '\n';
        std::cout << "max_center_error_m: " << max_center_error << '\n';
        std::cout << "max_abs_z_error_m: " << max_abs_z_error << '\n';
        std::cout << "max_abs_vz_error_m_per_s: " << max_abs_vz_error << '\n';

        if (!std::filesystem::exists(dll_path))
        {
            std::cerr << "ERROR: liblammps.dll was not copied next to the executable.\n";
            return 1;
        }

        if (external_force.callback_calls <= 0 || external_force.atom1_updates <= 0)
        {
            std::cerr << "ERROR: fix external callback did not update atom id=1.\n";
            return 1;
        }

        if (max_center_error > kErrorTolerance)
        {
            std::cerr << "ERROR: SPHinXsys geometric center drift exceeded tolerance "
                      << kErrorTolerance << ".\n";
            return 1;
        }

        if (max_abs_z_error > kErrorTolerance || max_abs_vz_error > kErrorTolerance)
        {
            std::cerr << "ERROR: free-fall error exceeded tolerance " << kErrorTolerance << ".\n";
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
