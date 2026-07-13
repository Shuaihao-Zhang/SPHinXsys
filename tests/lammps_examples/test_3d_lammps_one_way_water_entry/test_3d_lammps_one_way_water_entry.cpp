/* ------------------------------------------------------------------------- *
 *                                SPHinXsys                                  *
 * ------------------------------------------------------------------------- *
 * This example tests a one-way LAMMPS-driven sphere entering a water tank.
 * LAMMPS advances the free-fall sphere, while SPHinXsys updates a matching
 * moving solid boundary and computes the hydrodynamic force for diagnostics.
 * ------------------------------------------------------------------------- */
#include "test_3d_lammps_one_way_water_entry.h"

using namespace SPH;
using namespace LammpsOneWayWaterEntry;

int main(int ac, char *av[])
{
    try
    {
        //----------------------------------------------------------------------
        //	Build up a SPHSystem and handle command line options.
        //----------------------------------------------------------------------
        SPHSystem sph_system(kSystemDomainBounds, kParticleSpacing);
        sph_system.setRunParticleRelaxation(false);
        sph_system.setReloadParticles(reload_particle_file_exists() || std::filesystem::exists(fixed_sphere_reload_file()));
        sph_system.handleCommandlineOptions(ac, av);

        if (sph_system.RunParticleRelaxation())
        {
            sph_system.setReloadParticles(false);
            std::cout << "Particle relaxation mode: reload particles disabled; generating sphere particles from lattice.\n";
        }
        else if (sph_system.ReloadParticles())
        {
            import_fixed_sphere_reload_if_available();
            if (!reload_particle_file_exists())
            {
                std::cout << "WARNING: particle reload was requested, but "
                          << std::filesystem::absolute(reload_particle_file()).string()
                          << " was not found. Falling back to lattice particles.\n";
                sph_system.setReloadParticles(false);
            }
        }

        //----------------------------------------------------------------------
        //	Creating bodies with corresponding materials and particles.
        //----------------------------------------------------------------------
        SolidBody sphere_boundary(
            sph_system, makeShared<GeometricShapeBall>(kInitialCenter, kSphereRadius, "LammpsDrivenWaterEntrySphere"));
        sphere_boundary.defineMatterMaterial<Solid>(kSphereDensity);
        const std::string sphere_particle_source = generate_sphere_boundary_particles(sph_system, sphere_boundary);

        if (sph_system.RunParticleRelaxation())
        {
            return run_sphere_particle_relaxation(sphere_boundary);
        }

        DrivenSphereBoundary driven_sphere(sphere_boundary, kInitialCenter);

        FluidBody water_block(sph_system, makeShared<WaterBlock>("WaterBody"));
        water_block.defineMatterMaterial<WeaklyCompressibleFluid>(kWaterDensity, kSoundSpeed);
        water_block.addMaterialProperty<Viscosity>(kDynamicViscosity);
        water_block.generateParticles<BaseParticles, Lattice>();

        SolidBody wall_boundary(sph_system, makeShared<WallBoundary>("WallBoundary"));
        wall_boundary.defineMatterMaterial<Solid>();
        wall_boundary.generateParticles<BaseParticles, Lattice>();

        //----------------------------------------------------------------------
        //	Define body relations.
        //----------------------------------------------------------------------
        InnerRelation water_inner(water_block);
        ContactRelation water_contact(water_block, {&wall_boundary, &sphere_boundary});
        ContactRelation sphere_contact(sphere_boundary, {&water_block});
        ComplexRelation water_complex(water_inner, water_contact);

        //----------------------------------------------------------------------
        //	Define all numerical methods for the fluid dynamics.
        //----------------------------------------------------------------------
        SimpleDynamics<HydrostaticPressureField> hydrostatic_pressure(water_block);
        Gravity gravity(Vec3d(0.0, 0.0, -kGravity));
        SimpleDynamics<GravityForce<Gravity>> constant_gravity(water_block, gravity);
        SimpleDynamics<NormalDirectionFromBodyShape> wall_normal_direction(wall_boundary);
        SimpleDynamics<NormalDirectionFromBodyShape> sphere_normal_direction(sphere_boundary);

        Dynamics1Level<fluid_dynamics::Integration1stHalfWithWallRiemann>
            pressure_relaxation(water_inner, water_contact);
        Dynamics1Level<fluid_dynamics::Integration2ndHalfWithWallNoRiemann>
            density_relaxation(water_inner, water_contact);
        InteractionWithUpdate<fluid_dynamics::DensitySummationComplexFreeSurface>
            update_density_by_summation(water_inner, water_contact);
        InteractionWithUpdate<fluid_dynamics::ViscousForceWithWall>
            viscous_force(water_inner, water_contact);
        ReduceDynamics<fluid_dynamics::AdvectionViscousTimeStep>
            get_fluid_advection_time_step_size(water_block, kCharacteristicVelocity);
        ReduceDynamics<fluid_dynamics::AcousticTimeStep>
            get_fluid_time_step_size(water_block);
        ParticleSorting particle_sorting(water_block);

        //----------------------------------------------------------------------
        //	Coupling between the SPHinXsys moving boundary and the water body.
        //----------------------------------------------------------------------
        InteractionWithUpdate<solid_dynamics::ViscousForceFromFluid>
            viscous_force_on_sphere(sphere_contact);
        InteractionWithUpdate<solid_dynamics::PressureForceFromFluid<decltype(density_relaxation)>>
            pressure_force_on_sphere(sphere_contact);

        //----------------------------------------------------------------------
        //	LAMMPS owns the sphere motion. The external callback stays zero for
        //	this one-way test and no SPH force is sent back to LAMMPS.
        //----------------------------------------------------------------------
        LammpsDEMAdapter dem_adapter;

        //----------------------------------------------------------------------
        //	Define the methods for I/O operations and observations.
        //----------------------------------------------------------------------
        BodyStatesRecordingToVtp write_real_body_states(sph_system);
        write_real_body_states.addToWrite<Real>(water_block, "Pressure");
        write_real_body_states.addToWrite<Vecd>(wall_boundary, "NormalDirection");
        write_real_body_states.addToWrite<Vecd>(sphere_boundary, "Velocity");
        write_real_body_states.addToWrite<Vecd>(sphere_boundary, "PressureForceFromFluid");
        write_real_body_states.addToWrite<Vecd>(sphere_boundary, "ViscousForceFromFluid");
        write_real_body_states.addToWrite<Vecd>(sphere_boundary, "NormalDirection");

        const std::filesystem::path motion_csv_path = "sphere_motion.csv";
        const std::filesystem::path force_csv_path = "sphere_force.csv";
        const std::filesystem::path runtime_path = SPH::lammps_examples::lammps_runtime_path();
        const std::filesystem::path output_path = std::filesystem::absolute(IO::getEnvironment().OutputFolder());

        std::ofstream motion_csv(motion_csv_path);
        std::ofstream force_csv(force_csv_path);
        if (!motion_csv)
        {
            throw std::runtime_error("could not open sphere_motion.csv for writing");
        }
        if (!force_csv)
        {
            throw std::runtime_error("could not open sphere_force.csv for writing");
        }
        motion_csv << std::setprecision(17);
        force_csv << std::setprecision(17);
        write_motion_csv_header(motion_csv);
        write_force_csv_header(force_csv);

        //----------------------------------------------------------------------
        //	Prepare the simulation with cell linked list, configuration and state.
        //----------------------------------------------------------------------
        sph_system.initializeSystemCellLinkedLists();
        sph_system.initializeSystemConfigurations();
        hydrostatic_pressure.exec();
        constant_gravity.exec();
        wall_normal_direction.exec();
        sphere_normal_direction.exec();

        Real &physical_time = *sph_system.getSystemVariableDataByName<Real>("PhysicalTime");
        physical_time = 0.0;

        //----------------------------------------------------------------------
        //	Initial diagnostics before the time stepping starts.
        //----------------------------------------------------------------------
        Real max_center_error = 0.0;
        Real max_abs_z_minus_freefall = 0.0;
        Real max_abs_vz_minus_freefall = 0.0;
        ForceStats force_stats;
        MotionSample final_motion_sample;
        int number_of_iterations = 0;
        int lammps_step_count = 0;
        Real max_lammps_clock_error = 0.0;

        DEMState dem_state = dem_adapter.pullState();
        driven_sphere.update(dem_state);
        update_water_sphere_configuration(water_block, sphere_boundary, water_complex, sphere_contact);
        viscous_force_on_sphere.exec();
        pressure_force_on_sphere.exec();

        Vec3d hydro_force = sum_sphere_hydro_force(sphere_boundary);
        Vec3d hydro_torque = sum_sphere_hydro_torque(sphere_boundary, dem_state.center);
        final_motion_sample =
            make_motion_sample(number_of_iterations, lammps_step_count, physical_time, dem_state, driven_sphere.geometricCenter());
        ForceSample force_sample = make_force_sample(
            number_of_iterations, physical_time, dem_state, hydro_force, hydro_torque);
        write_motion_csv_sample(motion_csv, final_motion_sample);
        write_force_csv_sample(force_csv, force_sample);
        max_center_error = std::max(max_center_error, final_motion_sample.center_error);
        max_abs_z_minus_freefall =
            std::max(max_abs_z_minus_freefall, std::abs(final_motion_sample.z_minus_freefall));
        max_abs_vz_minus_freefall =
            std::max(max_abs_vz_minus_freefall, std::abs(final_motion_sample.vz_minus_freefall));
        force_stats.add(force_sample);
        write_real_body_states.writeToFile(0);
        std::filesystem::path latest_dem_load_vtp =
            write_dem_load_to_vtp(0, physical_time, dem_state, hydro_force, hydro_torque);

        //----------------------------------------------------------------------
        //	Main loop starts here. The outer loop follows the fluid advection
        //	step and updates particle configuration at a lower frequency. The
        //	inner acoustic loop advances pressure/density and synchronizes the
        //	LAMMPS free-fall particle over the same acoustic interval. The
        //	hydrodynamic force is recorded only and is never sent back to LAMMPS.
        //----------------------------------------------------------------------
        TickCount t1 = TickCount::now();
        TimeInterval interval;
        const int screen_output_interval = 20;
        int advection_iterations = 0;
        int output_iteration = 0;
        Real next_output_time = kVtpOutputInterval;

        while (physical_time < kEndTime - TinyReal)
        {
            ++advection_iterations;

            update_water_sphere_configuration(water_block, sphere_boundary, water_complex, sphere_contact);
            Real advection_step = SMIN(get_fluid_advection_time_step_size.exec(), kEndTime - physical_time);

            update_density_by_summation.exec();
            viscous_force.exec();
            viscous_force_on_sphere.exec();

            Real relaxation_time = 0.0;
            while (relaxation_time < advection_step - TinyReal && physical_time < kEndTime - TinyReal)
            {
                const Real acoustic_limit = SMIN(get_fluid_time_step_size.exec(),
                                                 SMIN(advection_step - relaxation_time, kEndTime - physical_time));
                const CouplingStepPlan coupling_plan = dem_adapter.planCouplingStep(acoustic_limit);
                const Real acoustic_step = coupling_plan.coupled_dt;
                pressure_relaxation.exec(acoustic_step);
                pressure_force_on_sphere.exec();
                density_relaxation.exec(acoustic_step);

                hydro_force = sum_sphere_hydro_force(sphere_boundary);
                hydro_torque = sum_sphere_hydro_torque(sphere_boundary, dem_state.center);

                // One-way coupling: LAMMPS advances under gravity only. No SPH
                // hydrodynamic force is pushed through fix external in this case.
                const CouplingAdvanceResult coupling_advance =
                    dem_adapter.advance(coupling_plan, physical_time);
                lammps_step_count += coupling_advance.dem_steps;
                max_lammps_clock_error =
                    std::max(max_lammps_clock_error, static_cast<Real>(coupling_advance.synchronization_error));
                dem_state = dem_adapter.pullState();
                driven_sphere.update(dem_state);

                relaxation_time += acoustic_step;
                physical_time += acoustic_step;
                ++number_of_iterations;

                final_motion_sample =
                    make_motion_sample(number_of_iterations, lammps_step_count, physical_time, dem_state, driven_sphere.geometricCenter());
                force_sample = make_force_sample(
                    number_of_iterations, physical_time, dem_state, hydro_force, hydro_torque);

                write_motion_csv_sample(motion_csv, final_motion_sample);
                write_force_csv_sample(force_csv, force_sample);
                max_center_error = std::max(max_center_error, final_motion_sample.center_error);
                max_abs_z_minus_freefall =
                    std::max(max_abs_z_minus_freefall, std::abs(final_motion_sample.z_minus_freefall));
                max_abs_vz_minus_freefall =
                    std::max(max_abs_vz_minus_freefall, std::abs(final_motion_sample.vz_minus_freefall));
                force_stats.add(force_sample);

                if (number_of_iterations % screen_output_interval == 0)
                {
                    std::cout << std::fixed << std::setprecision(6)
                              << "N=" << number_of_iterations
                              << " Time = " << physical_time
                              << " advection_step = " << advection_step
                              << " acoustic_limit = " << acoustic_limit
                              << " acoustic_step = " << acoustic_step
                              << " dem_steps = " << coupling_advance.dem_steps
                              << " lammps_atime = " << coupling_advance.lammps_time
                              << " driver_time = " << coupling_advance.driver_time
                              << " driver_lammps_offset = " << coupling_advance.driver_lammps_offset
                              << " clock_error = " << coupling_advance.synchronization_error
                              << " z = " << dem_state.center[2]
                              << " vz = " << dem_state.velocity[2]
                              << " Fz = " << hydro_force[2] << "\n";
                }

                if (physical_time + TinyReal >= next_output_time || physical_time + TinyReal >= kEndTime)
                {
                    output_iteration = number_of_iterations;
                    write_real_body_states.writeToFile(output_iteration);
                    latest_dem_load_vtp = write_dem_load_to_vtp(
                        output_iteration, physical_time, dem_state, hydro_force, hydro_torque);
                    next_output_time += kVtpOutputInterval;
                }
            }

            TickCount t2 = TickCount::now();
            if (advection_iterations % 100 == 0)
            {
                particle_sorting.exec();
            }
            update_water_sphere_configuration(water_block, sphere_boundary, water_complex, sphere_contact);
            TickCount t3 = TickCount::now();
            interval += t3 - t2;

        }
        TickCount t4 = TickCount::now();

        motion_csv.close();
        force_csv.close();

        //----------------------------------------------------------------------
        //	Statistics and regression-style checks.
        //----------------------------------------------------------------------
        const ExternalForceBuffer &external_force = dem_adapter.externalForce();
        const std::array<double, 3> external_force_value = external_force.forceForId(1);
        const Vec3d pre_entry_mean_force = force_stats.pre_entry_mean_force();
        const Real pre_entry_mean_force_norm = force_stats.pre_entry_mean_force_norm();
        const Real post_entry_max_fz =
            force_stats.post_entry_count > 0 ? force_stats.post_entry_max_fz : 0.0;
        const TickCount::interval_t tt = t4 - t1 - interval;

        std::cout << std::setprecision(17);
        std::cout << "SPHinXsys one-way LAMMPS-driven water-entry example\n";
        std::cout << "Total wall time for computation: " << tt.seconds() << " seconds.\n";
        std::cout << "lammps_runtime_in_working_directory: " << (std::filesystem::exists(runtime_path) ? "yes" : "no") << '\n';
        std::cout << "lammps_version: " << dem_adapter.version() << '\n';
        std::cout << "sphere_motion_csv: " << std::filesystem::absolute(motion_csv_path).string() << '\n';
        std::cout << "sphere_force_csv: " << std::filesystem::absolute(force_csv_path).string() << '\n';
        std::cout << "VTP_output_folder: " << output_path.string() << '\n';
        std::cout << "DEM_load_vtp_latest: " << std::filesystem::absolute(latest_dem_load_vtp).string() << '\n';
        std::cout << "reload_file: " << std::filesystem::absolute(reload_particle_file()).string() << '\n';
        std::cout << "sphere_particle_source: " << sphere_particle_source << '\n';
        std::cout << "water_particles: " << water_block.getBaseParticles().TotalRealParticles() << '\n';
        std::cout << "sphere_particles: " << driven_sphere.particleCount() << '\n';
        std::cout << "sphere_mass_kg: " << sphere_mass() << '\n';
        std::cout << "dem_max_dt_s: " << kDemMaxDt << '\n';
        std::cout << "advection_iterations: " << advection_iterations << '\n';
        std::cout << "number_of_iterations: " << number_of_iterations << '\n';
        std::cout << "lammps_substeps_executed: " << lammps_step_count << '\n';
        std::cout << "max_lammps_clock_error_s: " << max_lammps_clock_error << '\n';
        std::cout << "hydrodynamic_torque_feedback_enabled: no\n";
        std::cout << "entry_time_s: " << water_entry_time() << '\n';
        std::cout << "end_time_s: " << kEndTime << '\n';
        std::cout << "external_force_feedback_N: "
                  << external_force_value[0] << ','
                  << external_force_value[1] << ','
                  << external_force_value[2] << '\n';
        std::cout << "callback_calls: " << external_force.callbackCalls() << '\n';
        std::cout << "atom1_force_updates: " << external_force.atomUpdates() << '\n';
        std::cout << "final_lammps_center_m: "
                  << final_motion_sample.dem_state.center[0] << ','
                  << final_motion_sample.dem_state.center[1] << ','
                  << final_motion_sample.dem_state.center[2] << '\n';
        std::cout << "final_sph_geometric_center_m: "
                  << final_motion_sample.sph_geometric_center[0] << ','
                  << final_motion_sample.sph_geometric_center[1] << ','
                  << final_motion_sample.sph_geometric_center[2] << '\n';
        std::cout << "final_vz_m_per_s: " << final_motion_sample.dem_state.velocity[2] << '\n';
        std::cout << "final_z_freefall_m: " << final_motion_sample.z_freefall << '\n';
        std::cout << "final_vz_freefall_m_per_s: " << final_motion_sample.vz_freefall << '\n';
        std::cout << "max_center_error_m: " << max_center_error << '\n';
        std::cout << "max_abs_z_minus_freefall_m: " << max_abs_z_minus_freefall << '\n';
        std::cout << "max_abs_vz_minus_freefall_m_per_s: " << max_abs_vz_minus_freefall << '\n';
        std::cout << "pre_entry_samples: " << force_stats.pre_entry_count << '\n';
        std::cout << "pre_entry_mean_force_N: "
                  << pre_entry_mean_force[0] << ','
                  << pre_entry_mean_force[1] << ','
                  << pre_entry_mean_force[2] << '\n';
        std::cout << "pre_entry_mean_force_norm_N: " << pre_entry_mean_force_norm << '\n';
        std::cout << "post_entry_samples: " << force_stats.post_entry_count << '\n';
        std::cout << "post_entry_max_Fz_N: " << post_entry_max_fz << '\n';

        if (!std::filesystem::exists(runtime_path))
        {
            std::cerr << "ERROR: the LAMMPS runtime library was not staged next to the executable.\n";
            return 1;
        }
        if (!std::filesystem::exists(latest_dem_load_vtp))
        {
            std::cerr << "ERROR: the 3D DEM load VTP file was not written.\n";
            return 1;
        }
        if (external_force.callbackCalls() <= 0 || external_force.atomUpdates() <= 0)
        {
            std::cerr << "ERROR: fix external callback did not update atom id=1.\n";
            return 1;
        }
        if (external_force_value[0] != 0.0 || external_force_value[1] != 0.0 ||
            external_force_value[2] != 0.0)
        {
            std::cerr << "ERROR: hydrodynamic force feedback to LAMMPS is not disabled.\n";
            return 1;
        }
        if (max_center_error > kCenterErrorTolerance)
        {
            std::cerr << "ERROR: SPHinXsys geometric center drift exceeded tolerance "
                      << kCenterErrorTolerance << ".\n";
            return 1;
        }
        if (max_abs_z_minus_freefall > kFreeFallErrorTolerance || max_abs_vz_minus_freefall > kFreeFallErrorTolerance)
        {
            std::cerr << "ERROR: LAMMPS free-fall error exceeded tolerance "
                      << kFreeFallErrorTolerance << ".\n";
            return 1;
        }
        if (force_stats.pre_entry_count == 0 || pre_entry_mean_force_norm > kPreEntryForceNormTolerance)
        {
            std::cerr << "ERROR: pre-entry hydrodynamic force is not close to zero.\n";
            return 1;
        }
        if (force_stats.post_entry_count == 0 || post_entry_max_fz < kMinimumPostEntryFz)
        {
            std::cerr << "ERROR: post-entry upward hydrodynamic force response is too small.\n";
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
