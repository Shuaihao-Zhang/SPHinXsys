/* ------------------------------------------------------------------------- *
 *                                SPHinXsys                                  *
 * ------------------------------------------------------------------------- *
 * This example lets LAMMPS own a free-falling DEM sphere and uses the pulled
 * LAMMPS state to drive a SPHinXsys sphere boundary for visualization.
 * ------------------------------------------------------------------------- */
#include "test_3d_lammps_driven_sphere.h"

using namespace SPH;
using namespace LammpsDrivenSphere;

int main(int ac, char *av[])
{
    try
    {
        //----------------------------------------------------------------------
        //	Build up a SPHSystem and create the proxy solid sphere boundary.
        //----------------------------------------------------------------------
        SPHSystem sph_system(kSystemDomainBounds, kDiameter / 4.0);
        sph_system.setRunParticleRelaxation(false);
        sph_system.setReloadParticles(false);
        sph_system.handleCommandlineOptions(ac, av);

        SolidBody sphere_boundary(
            sph_system, makeShared<GeometricShapeBall>(kInitialCenter, kRadius, "LammpsDrivenSphere"));
        sphere_boundary.defineMatterMaterial<Solid>(kDensity);
        sphere_boundary.generateParticles<BaseParticles, Lattice>();

        DrivenSphereBoundary driven_sphere(sphere_boundary, kInitialCenter);

        //----------------------------------------------------------------------
        //	LAMMPS owns the sphere motion; SPHinXsys only mirrors the state.
        //----------------------------------------------------------------------
        LammpsDEMAdapter dem_adapter;

        //----------------------------------------------------------------------
        //	Define the methods for I/O operations and diagnostics.
        //----------------------------------------------------------------------
        BodyStatesRecordingToVtp write_sphere_state(sphere_boundary);
        write_sphere_state.addToWrite<Vecd>(sphere_boundary, "Velocity");

        const std::filesystem::path motion_csv_path = "sphere_motion.csv";
        const std::filesystem::path dll_path = "liblammps.dll";
        const std::filesystem::path output_path = std::filesystem::absolute(IO::getEnvironment().OutputFolder());

        std::ofstream motion_csv(motion_csv_path);
        if (!motion_csv)
        {
            throw std::runtime_error("could not open sphere_motion.csv for writing");
        }
        motion_csv << std::setprecision(17);
        write_motion_csv_header(motion_csv);

        Real max_center_error = 0.0;
        Real max_abs_z_minus_freefall = 0.0;
        Real max_abs_vz_minus_freefall = 0.0;

        auto record = [&](int number_of_iterations, int lammps_step)
        {
            const DEMState dem_state = dem_adapter.pullState();
            driven_sphere.update(dem_state);

            MotionSample sample =
                make_motion_sample(number_of_iterations, lammps_step, dem_state, driven_sphere.geometricCenter());
            sph_system.svPhysicalTime().setValue(sample.time);
            write_sphere_state.writeToFile(number_of_iterations);
            write_motion_csv_sample(motion_csv, sample);

            max_center_error = std::max(max_center_error, sample.center_error);
            max_abs_z_minus_freefall =
                std::max(max_abs_z_minus_freefall, std::abs(sample.z_minus_freefall));
            max_abs_vz_minus_freefall =
                std::max(max_abs_vz_minus_freefall, std::abs(sample.vz_minus_freefall));
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

        const ExternalForce &external_force = dem_adapter.externalForce();

        std::cout << std::setprecision(17);
        std::cout << "SPHinXsys LAMMPS-driven sphere boundary example\n";
        std::cout << "liblammps_dll_in_working_directory: " << (std::filesystem::exists(dll_path) ? "yes" : "no") << '\n';
        std::cout << "lammps_version: " << dem_adapter.version() << '\n';
        std::cout << "sphere_motion_csv: " << std::filesystem::absolute(motion_csv_path).string() << '\n';
        std::cout << "VTP_output_folder: " << output_path.string() << '\n';
        std::cout << "sph_sphere_particles: " << driven_sphere.particleCount() << '\n';
        std::cout << "sphere_mass_kg: " << sphere_mass() << '\n';
        std::cout << "external_force_feedback_N: "
                  << external_force.force[0] << ','
                  << external_force.force[1] << ','
                  << external_force.force[2] << '\n';
        std::cout << "callback_calls: " << external_force.callback_calls << '\n';
        std::cout << "atom1_force_updates: " << external_force.atom1_updates << '\n';
        std::cout << "number_of_iterations: " << number_of_iterations << '\n';
        std::cout << "lammps_substeps_executed: " << lammps_step_count << '\n';
        std::cout << "final_lammps_center_m: "
                  << final_sample.dem_state.center[0] << ','
                  << final_sample.dem_state.center[1] << ','
                  << final_sample.dem_state.center[2] << '\n';
        std::cout << "final_sph_geometric_center_m: "
                  << final_sample.sph_geometric_center[0] << ','
                  << final_sample.sph_geometric_center[1] << ','
                  << final_sample.sph_geometric_center[2] << '\n';
        std::cout << "final_vz_m_per_s: " << final_sample.dem_state.velocity[2] << '\n';
        std::cout << "final_z_freefall_m: " << final_sample.z_freefall << '\n';
        std::cout << "final_vz_freefall_m_per_s: " << final_sample.vz_freefall << '\n';
        std::cout << "max_center_error_m: " << max_center_error << '\n';
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
        if (max_center_error > kErrorTolerance)
        {
            std::cerr << "ERROR: SPHinXsys geometric center drift exceeded tolerance "
                      << kErrorTolerance << ".\n";
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
