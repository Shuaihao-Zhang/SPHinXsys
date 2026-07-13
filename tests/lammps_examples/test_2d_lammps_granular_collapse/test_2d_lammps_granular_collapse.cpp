/* ------------------------------------------------------------------------- *
 *                                SPHinXsys                                  *
 * ------------------------------------------------------------------------- *
 * DEM-only reference for a 2D granular column collapse. This case deliberately
 * does not create SPHinXsys water bodies, does not compute SPH hydrodynamic
 * forces, and does not use LAMMPS fix external. It isolates the LAMMPS granular
 * contact, wall contact, and gravity response used by the underwater collapse
 * example.
 * ------------------------------------------------------------------------- */
#include "test_2d_lammps_granular_collapse.h"

using namespace SPH;
using namespace LammpsGranularCollapse2D;

int main()
{
    try
    {
        IO::initEnvironment();
        IO::initLogger();

        LammpsDEMOnlyColumnAdapter dem_adapter;
        const Real lammps_particle_mass = dem_adapter.particleMass();
        const Real lammps_mass_relative_error =
            std::abs(lammps_particle_mass - grain_mass()) / grain_mass();

        const std::filesystem::path motion_csv_path = "dem_motion.csv";
        const std::filesystem::path force_csv_path = "dem_force.csv";
        const std::filesystem::path runtime_path = SPH::lammps_examples::lammps_runtime_path();
        const std::filesystem::path output_path = std::filesystem::absolute(IO::getEnvironment().OutputFolder());

        VtpPvdWriter dem_discs_pvd("DEM_Discs");
        VtpPvdWriter dem_forces_pvd("DEM_Forces");

        std::ofstream motion_csv(motion_csv_path);
        std::ofstream force_csv(force_csv_path);
        if (!motion_csv)
        {
            throw std::runtime_error("could not open dem_motion.csv for writing");
        }
        if (!force_csv)
        {
            throw std::runtime_error("could not open dem_force.csv for writing");
        }
        motion_csv << std::setprecision(17);
        force_csv << std::setprecision(17);
        write_motion_csv_header(motion_csv);
        write_force_csv_header(force_csv);

        Real physical_time = 0.0;
        int lammps_step_count = 0;
        int output_count = 0;
        Real max_right_front = initial_right_front_x();
        Real max_speed_history = 0.0;
        Real max_net_force_norm = 0.0;
        Real max_contact_force_norm = 0.0;
        bool finite_state = true;

        std::vector<DEMParticleState> states = dem_adapter.pullStates();
        write_motion_csv_samples(motion_csv, output_count, physical_time, lammps_step_count, states);
        write_force_csv_samples(force_csv, output_count, physical_time, states);

        std::filesystem::path latest_dem_discs_vtp =
            write_dem_discs_to_vtp(output_count, physical_time, "DEM_Discs", centers_from_states(states), kGrainRadius);
        std::filesystem::path latest_dem_forces_vtp =
            write_dem_force_vtp(output_count, physical_time, states);
        dem_discs_pvd.add(physical_time, latest_dem_discs_vtp);
        dem_forces_pvd.add(physical_time, latest_dem_forces_vtp);

        finite_state = finite_state && all_finite(states);
        max_right_front = std::max(max_right_front, right_front_x(states));
        max_speed_history = std::max(max_speed_history, max_speed(states));
        max_net_force_norm = std::max(max_net_force_norm, max_force_norm(net_forces_from_states(states)));
        max_contact_force_norm = std::max(max_contact_force_norm, max_force_norm(contact_forces_from_states(states)));

        const int total_output_steps = static_cast<int>(std::ceil(kEndTime / kVtpOutputInterval));
        for (int output_step = 1; output_step <= total_output_steps; ++output_step)
        {
            const Real target_time =
                SMIN(static_cast<Real>(output_step) * kVtpOutputInterval, kEndTime);
            const Real duration = target_time - physical_time;
            if (duration <= TinyReal)
            {
                continue;
            }
            const CouplingStepPlan coupling_plan = dem_adapter.planCouplingStep(duration);
            const CouplingAdvanceResult coupling_advance =
                dem_adapter.advance(coupling_plan, physical_time);
            lammps_step_count += coupling_advance.dem_steps;
            physical_time = coupling_advance.lammps_time;
            ++output_count;

            states = dem_adapter.pullStates();
            write_motion_csv_samples(motion_csv, output_count, physical_time, lammps_step_count, states);
            write_force_csv_samples(force_csv, output_count, physical_time, states);

            latest_dem_discs_vtp =
                write_dem_discs_to_vtp(output_count, physical_time, "DEM_Discs", centers_from_states(states), kGrainRadius);
            latest_dem_forces_vtp =
                write_dem_force_vtp(output_count, physical_time, states);
            dem_discs_pvd.add(physical_time, latest_dem_discs_vtp);
            dem_forces_pvd.add(physical_time, latest_dem_forces_vtp);

            finite_state = finite_state && all_finite(states);
            max_right_front = std::max(max_right_front, right_front_x(states));
            max_speed_history = std::max(max_speed_history, max_speed(states));
            max_net_force_norm = std::max(max_net_force_norm, max_force_norm(net_forces_from_states(states)));
            max_contact_force_norm = std::max(max_contact_force_norm, max_force_norm(contact_forces_from_states(states)));

            std::cout << std::fixed << std::setprecision(6)
                      << "Output=" << output_count
                      << " Time = " << physical_time
                      << " right_front = " << right_front_x(states)
                      << " max_speed = " << max_speed(states)
                      << " max_contact_force = " << max_force_norm(contact_forces_from_states(states)) << "\n";
        }

        motion_csv.close();
        force_csv.close();

        const Real final_right_front = right_front_x(states);
        const Real right_front_displacement = final_right_front - initial_right_front_x();

        std::cout << std::setprecision(17);
        std::cout << "SPHinXsys LAMMPS DEM-only 2D granular collapse example\n";
        std::cout << "lammps_runtime_in_working_directory: " << (std::filesystem::exists(runtime_path) ? "yes" : "no") << '\n';
        std::cout << "lammps_version: " << dem_adapter.version() << '\n';
        std::cout << "dem_motion_csv: " << std::filesystem::absolute(motion_csv_path).string() << '\n';
        std::cout << "dem_force_csv: " << std::filesystem::absolute(force_csv_path).string() << '\n';
        std::cout << "VTP_output_folder: " << output_path.string() << '\n';
        std::cout << "DEM_discs_vtp_latest: " << std::filesystem::absolute(latest_dem_discs_vtp).string() << '\n';
        std::cout << "DEM_forces_vtp_latest: " << std::filesystem::absolute(latest_dem_forces_vtp).string() << '\n';
        std::cout << "DEM_discs_pvd: " << std::filesystem::absolute(dem_discs_pvd.path()).string() << '\n';
        std::cout << "DEM_forces_pvd: " << std::filesystem::absolute(dem_forces_pvd.path()).string() << '\n';
        std::cout << "dem_particle_count: " << kParticleCount << '\n';
        std::cout << "grain_mass_per_unit_depth_kg_per_m: " << grain_mass() << '\n';
        std::cout << "lammps_equivalent_sphere_density_kg_per_m3: " << lammps_equivalent_sphere_density() << '\n';
        std::cout << "lammps_particle_mass_kg: " << lammps_particle_mass << '\n';
        std::cout << "lammps_particle_mass_relative_error: " << lammps_mass_relative_error << '\n';
        std::cout << "grain_weight_per_unit_depth_N_per_m: " << grain_weight() << '\n';
        std::cout << "contact_normal_stiffness_N_per_m: " << kContactNormalStiffness << '\n';
        std::cout << "contact_restitution: " << kContactRestitution << '\n';
        std::cout << "contact_friction: " << kContactFriction << '\n';
        std::cout << "dem_dt_s: " << kDemDt << '\n';
        std::cout << "lammps_substeps_executed: " << lammps_step_count << '\n';
        std::cout << "end_time_s: " << kEndTime << '\n';
        std::cout << "initial_right_front_x_m: " << initial_right_front_x() << '\n';
        std::cout << "final_right_front_x_m: " << final_right_front << '\n';
        std::cout << "right_front_displacement_m: " << right_front_displacement << '\n';
        std::cout << "max_right_front_x_m: " << max_right_front << '\n';
        std::cout << "max_speed_m_per_s: " << max_speed_history << '\n';
        std::cout << "max_net_force_norm_N: " << max_net_force_norm << '\n';
        std::cout << "max_contact_force_norm_N: " << max_contact_force_norm << '\n';
        std::cout << "hydrodynamic_force_feedback_enabled: no\n";
        std::cout << "finite_state: " << (finite_state ? "yes" : "no") << '\n';

        if (!std::filesystem::exists(runtime_path))
        {
            std::cerr << "ERROR: the LAMMPS runtime library was not staged next to the executable.\n";
            return 1;
        }
        if (!finite_state)
        {
            std::cerr << "ERROR: non-finite value detected in DEM-only state history.\n";
            return 1;
        }
        if (lammps_mass_relative_error > kMassRelativeTolerance)
        {
            std::cerr << "ERROR: LAMMPS particle mass does not match the expected 2D unit-depth grain mass.\n";
            return 1;
        }
        if (right_front_displacement < kMinimumRightFrontDisplacement)
        {
            std::cerr << "ERROR: DEM-only granular column did not move right enough to indicate collapse.\n";
            return 1;
        }
        if (max_speed_history < kMinimumMaxSpeed)
        {
            std::cerr << "ERROR: DEM-only particles did not develop a measurable velocity.\n";
            return 1;
        }
        if (!std::filesystem::exists(motion_csv_path) || !std::filesystem::exists(force_csv_path))
        {
            std::cerr << "ERROR: DEM-only CSV output files were not written.\n";
            return 1;
        }
        if (!std::filesystem::exists(latest_dem_discs_vtp) || !std::filesystem::exists(latest_dem_forces_vtp) ||
            !std::filesystem::exists(dem_discs_pvd.path()) || !std::filesystem::exists(dem_forces_pvd.path()))
        {
            std::cerr << "ERROR: DEM-only VTP/PVD output files were not written.\n";
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
