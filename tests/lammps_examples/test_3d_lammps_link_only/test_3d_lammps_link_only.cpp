/* ------------------------------------------------------------------------- *
 *                                SPHinXsys                                  *
 * ------------------------------------------------------------------------- *
 * This example verifies that an SPHinXsys-side executable can link liblammps,
 * create a no-MPI LAMMPS instance, use fix external, and reproduce the
 * single-sphere free-fall solution.
 * ------------------------------------------------------------------------- */
#include "test_3d_lammps_link_only.h"

using namespace LammpsLinkOnly;

int main()
{
    try
    {
        const std::filesystem::path motion_csv_path = "sphere_motion.csv";
        const std::filesystem::path dll_path = "liblammps.dll";
        const double mass = sphere_mass();

        LammpsDEMAdapter dem_adapter;
        const ExternalForce &external_force = dem_adapter.externalForce();
        const double acceleration_z = -kGravity + external_force.force[2] / mass;

        std::ofstream motion_csv(motion_csv_path);
        if (!motion_csv)
        {
            throw std::runtime_error("could not open sphere_motion.csv for writing");
        }
        motion_csv << std::setprecision(17);
        write_motion_csv_header(motion_csv);

        double max_abs_z_minus_freefall = 0.0;
        double max_abs_vz_minus_freefall = 0.0;

        auto record = [&](int number_of_iterations, int lammps_step)
        {
            const DEMState dem_state = dem_adapter.pullState();
            MotionSample sample = make_motion_sample(number_of_iterations, lammps_step, dem_state, acceleration_z);
            max_abs_z_minus_freefall =
                std::max(max_abs_z_minus_freefall, std::abs(sample.z_minus_freefall));
            max_abs_vz_minus_freefall =
                std::max(max_abs_vz_minus_freefall, std::abs(sample.vz_minus_freefall));
            write_motion_csv_sample(motion_csv, sample);
            return sample;
        };

        int number_of_iterations = 0;
        int lammps_step_count = 0;
        MotionSample final_sample = record(number_of_iterations, lammps_step_count);

        while (lammps_step_count < kTotalSteps)
        {
            dem_adapter.runSubsteps(kSampleEvery);
            lammps_step_count += kSampleEvery;
            ++number_of_iterations;
            final_sample = record(number_of_iterations, lammps_step_count);
        }

        motion_csv.close();

        std::cout << std::setprecision(17);
        std::cout << "SPHinXsys LAMMPS link-only example\n";
        std::cout << "liblammps_dll_in_working_directory: " << (std::filesystem::exists(dll_path) ? "yes" : "no") << '\n';
        std::cout << "lammps_version: " << dem_adapter.version() << '\n';
        std::cout << "sphere_motion_csv: " << std::filesystem::absolute(motion_csv_path).string() << '\n';
        std::cout << "sphere_mass_kg: " << mass << '\n';
        std::cout << "external_force_feedback_N: "
                  << external_force.force[0] << ','
                  << external_force.force[1] << ','
                  << external_force.force[2] << '\n';
        std::cout << "callback_calls: " << external_force.callback_calls << '\n';
        std::cout << "atom1_force_updates: " << external_force.atom1_updates << '\n';
        std::cout << "number_of_iterations: " << number_of_iterations << '\n';
        std::cout << "lammps_substeps_executed: " << lammps_step_count << '\n';
        std::cout << "final_z_m: " << final_sample.dem_state.center[2] << '\n';
        std::cout << "final_z_freefall_m: " << final_sample.z_freefall << '\n';
        std::cout << "final_vz_m_per_s: " << final_sample.dem_state.velocity[2] << '\n';
        std::cout << "final_vz_freefall_m_per_s: " << final_sample.vz_freefall << '\n';
        std::cout << "max_abs_z_minus_freefall_m: " << max_abs_z_minus_freefall << '\n';
        std::cout << "max_abs_vz_minus_freefall_m_per_s: " << max_abs_vz_minus_freefall << '\n';

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
        if (max_abs_z_minus_freefall > kErrorTolerance || max_abs_vz_minus_freefall > kErrorTolerance)
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
