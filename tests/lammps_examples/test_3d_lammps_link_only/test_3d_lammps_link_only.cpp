#include "library.h"

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

namespace {

constexpr double kGravity = 9.81;
constexpr double kDiameter = 0.02;
constexpr double kDensity = 2500.0;
constexpr double kInitialZ = 1.0;
constexpr double kTimeStep = 1.0e-5;
constexpr int kTotalSteps = 1000;
constexpr int kSampleEvery = 100;
constexpr double kErrorTolerance = 1.0e-9;

struct ExternalForce
{
    std::array<double, 3> force{0.0, 0.0, 0.0};
    int callback_calls = 0;
    int atom1_updates = 0;
};

struct Sample
{
    int step = 0;
    double time = 0.0;
    std::array<double, 3> x{0.0, 0.0, 0.0};
    std::array<double, 3> v{0.0, 0.0, 0.0};
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

double sphere_mass()
{
    const double radius = 0.5 * kDiameter;
    return kDensity * (4.0 / 3.0) * 3.1415926535897932384626433832795 * radius * radius * radius;
}

Sample gather_sample(void *lmp, int step, double acceleration_z)
{
    Sample sample;
    sample.step = step;
    sample.time = step * kTimeStep;

    std::array<double, 3> x{};
    std::array<double, 3> v{};
    lammps_gather_atoms(lmp, "x", 1, 3, x.data());
    lammps_gather_atoms(lmp, "v", 1, 3, v.data());

    sample.x = x;
    sample.v = v;
    sample.z_exact = kInitialZ + 0.5 * acceleration_z * sample.time * sample.time;
    sample.vz_exact = acceleration_z * sample.time;
    sample.z_error = sample.x[2] - sample.z_exact;
    sample.vz_error = sample.v[2] - sample.vz_exact;
    return sample;
}

void write_csv_header(std::ofstream &csv)
{
    csv << "step,time_s,x_m,y_m,z_m,vx_m_per_s,vy_m_per_s,vz_m_per_s,"
           "z_exact_m,vz_exact_m_per_s,z_error_m,vz_error_m_per_s\n";
}

void write_csv_sample(std::ofstream &csv, const Sample &sample)
{
    csv << sample.step << ','
        << sample.time << ','
        << sample.x[0] << ','
        << sample.x[1] << ','
        << sample.x[2] << ','
        << sample.v[0] << ','
        << sample.v[1] << ','
        << sample.v[2] << ','
        << sample.z_exact << ','
        << sample.vz_exact << ','
        << sample.z_error << ','
        << sample.vz_error << '\n';
}

} // namespace

int main()
{
    try
    {
        const std::filesystem::path csv_path = "sphere_motion.csv";
        const std::filesystem::path dll_path = "liblammps.dll";
        const double mass = sphere_mass();

        ExternalForce external_force;
        const double acceleration_z = -kGravity + external_force.force[2] / mass;

        LammpsInstance lammps;
        lammps.commands_string(R"lmp(
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

        lammps_set_fix_external_callback(lammps.get(), "ext", &external_force_callback, &external_force);
        lammps.throw_if_error("lammps_set_fix_external_callback");

        std::ofstream csv(csv_path);
        if (!csv)
        {
            throw std::runtime_error("could not open sphere_motion.csv for writing");
        }
        csv << std::setprecision(17);
        write_csv_header(csv);

        double max_abs_z_error = 0.0;
        double max_abs_vz_error = 0.0;

        auto record = [&](int step)
        {
            Sample sample = gather_sample(lammps.get(), step, acceleration_z);
            max_abs_z_error = std::max(max_abs_z_error, std::abs(sample.z_error));
            max_abs_vz_error = std::max(max_abs_vz_error, std::abs(sample.vz_error));
            write_csv_sample(csv, sample);
            return sample;
        };

        record(0);
        Sample final_sample;
        for (int step = kSampleEvery; step <= kTotalSteps; step += kSampleEvery)
        {
            lammps.command("run 100 post no", "LAMMPS run chunk");
            final_sample = record(step);
        }

        csv.close();

        std::cout << std::setprecision(17);
        std::cout << "SPHinXsys LAMMPS link-only example\n";
        std::cout << "liblammps_dll_in_working_directory: " << (std::filesystem::exists(dll_path) ? "yes" : "no") << '\n';
        std::cout << "lammps_version: " << lammps.version() << '\n';
        std::cout << "CSV: " << std::filesystem::absolute(csv_path).string() << '\n';
        std::cout << "mass_kg: " << mass << '\n';
        std::cout << "external_force_N: "
                  << external_force.force[0] << ','
                  << external_force.force[1] << ','
                  << external_force.force[2] << '\n';
        std::cout << "callback_calls: " << external_force.callback_calls << '\n';
        std::cout << "atom1_force_updates: " << external_force.atom1_updates << '\n';
        std::cout << "final_step: " << final_sample.step << '\n';
        std::cout << "final_z_m: " << final_sample.x[2] << '\n';
        std::cout << "final_z_exact_m: " << final_sample.z_exact << '\n';
        std::cout << "final_vz_m_per_s: " << final_sample.v[2] << '\n';
        std::cout << "final_vz_exact_m_per_s: " << final_sample.vz_exact << '\n';
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
