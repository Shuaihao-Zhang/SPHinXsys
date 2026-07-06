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
        const std::filesystem::path dll_path = "liblammps.dll";
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
        Real max_abs_z_error = 0.0;
        Real max_abs_vz_error = 0.0;
        ForceStats force_stats;
        MotionSample final_motion_sample;

        DEMState dem_state = dem_adapter.pullState();
        driven_sphere.update(dem_state);
        update_water_sphere_configuration(water_block, sphere_boundary, water_complex, sphere_contact);
        viscous_force_on_sphere.exec();
        pressure_force_on_sphere.exec();

        Vec3d hydro_force = sum_sphere_hydro_force(sphere_boundary);
        final_motion_sample = make_motion_sample(0, dem_state, driven_sphere.geometricCenter());
        ForceSample force_sample = make_force_sample(0, dem_state, hydro_force);
        write_motion_csv_sample(motion_csv, final_motion_sample);
        write_force_csv_sample(force_csv, force_sample);
        max_center_error = std::max(max_center_error, final_motion_sample.center_error);
        max_abs_z_error = std::max(max_abs_z_error, std::abs(final_motion_sample.z_error));
        max_abs_vz_error = std::max(max_abs_vz_error, std::abs(final_motion_sample.vz_error));
        force_stats.add(force_sample);
        write_real_body_states.writeToFile(0);

        //----------------------------------------------------------------------
        //	Main loop starts here. The organization follows the SPH fluid loop,
        //	with the LAMMPS-driven boundary update placed before each SPH step.
        //----------------------------------------------------------------------
        TickCount t1 = TickCount::now();
        TimeInterval interval;
        const int screen_output_interval = 20;
        int number_of_iterations = 0;

        for (int coupling_step = 1; coupling_step <= kTotalCouplingSteps; ++coupling_step)
        {
            ++number_of_iterations;

            dem_adapter.runSubsteps(kLammpsSubstepsPerCouplingStep);
            dem_state = dem_adapter.pullState();
            driven_sphere.update(dem_state);
            update_water_sphere_configuration(water_block, sphere_boundary, water_complex, sphere_contact);

            update_density_by_summation.exec();
            viscous_force.exec();
            viscous_force_on_sphere.exec();

            Real relaxation_time = 0.0;
            while (relaxation_time < kCouplingDt - TinyReal)
            {
                const Real dt = SMIN(get_fluid_time_step_size.exec(), kCouplingDt - relaxation_time);
                pressure_relaxation.exec(dt);
                pressure_force_on_sphere.exec();
                density_relaxation.exec(dt);

                relaxation_time += dt;
                physical_time += dt;
            }
            physical_time = static_cast<Real>(coupling_step) * kCouplingDt;

            // One-way coupling: the force is recorded only and is not pushed to LAMMPS.
            hydro_force = sum_sphere_hydro_force(sphere_boundary);
            final_motion_sample =
                make_motion_sample(coupling_step, dem_state, driven_sphere.geometricCenter());
            force_sample = make_force_sample(coupling_step, dem_state, hydro_force);

            write_motion_csv_sample(motion_csv, final_motion_sample);
            write_force_csv_sample(force_csv, force_sample);
            max_center_error = std::max(max_center_error, final_motion_sample.center_error);
            max_abs_z_error = std::max(max_abs_z_error, std::abs(final_motion_sample.z_error));
            max_abs_vz_error = std::max(max_abs_vz_error, std::abs(final_motion_sample.vz_error));
            force_stats.add(force_sample);

            if (number_of_iterations % screen_output_interval == 0)
            {
                std::cout << std::fixed << std::setprecision(6)
                          << "N=" << number_of_iterations
                          << " Time = " << physical_time
                          << " z = " << dem_state.center[2]
                          << " Fz = " << hydro_force[2] << "\n";
            }

            TickCount t2 = TickCount::now();
            if (number_of_iterations % 100 == 0)
            {
                particle_sorting.exec();
            }
            update_water_sphere_configuration(water_block, sphere_boundary, water_complex, sphere_contact);
            TickCount t3 = TickCount::now();
            interval += t3 - t2;

            if (coupling_step % kVtpOutputEvery == 0 || coupling_step == kTotalCouplingSteps)
            {
                write_real_body_states.writeToFile(coupling_step);
            }
        }
        TickCount t4 = TickCount::now();

        motion_csv.close();
        force_csv.close();

        //----------------------------------------------------------------------
        //	Statistics and regression-style checks.
        //----------------------------------------------------------------------
        const ExternalForce &external_force = dem_adapter.externalForce();
        const Vec3d pre_entry_mean_force = force_stats.pre_entry_mean_force();
        const Real pre_entry_mean_force_norm = force_stats.pre_entry_mean_force_norm();
        const Real post_entry_max_fz =
            force_stats.post_entry_count > 0 ? force_stats.post_entry_max_fz : 0.0;
        const TickCount::interval_t tt = t4 - t1 - interval;

        std::cout << std::setprecision(17);
        std::cout << "SPHinXsys one-way LAMMPS-driven water-entry example\n";
        std::cout << "Total wall time for computation: " << tt.seconds() << " seconds.\n";
        std::cout << "liblammps_dll_in_working_directory: " << (std::filesystem::exists(dll_path) ? "yes" : "no") << '\n';
        std::cout << "lammps_version: " << dem_adapter.version() << '\n';
        std::cout << "sphere_motion_csv: " << std::filesystem::absolute(motion_csv_path).string() << '\n';
        std::cout << "sphere_force_csv: " << std::filesystem::absolute(force_csv_path).string() << '\n';
        std::cout << "VTP_output_folder: " << output_path.string() << '\n';
        std::cout << "reload_file: " << std::filesystem::absolute(reload_particle_file()).string() << '\n';
        std::cout << "sphere_particle_source: " << sphere_particle_source << '\n';
        std::cout << "water_particles: " << water_block.getBaseParticles().TotalRealParticles() << '\n';
        std::cout << "sphere_particles: " << driven_sphere.particleCount() << '\n';
        std::cout << "sphere_mass_kg: " << sphere_mass() << '\n';
        std::cout << "entry_time_s: " << water_entry_time() << '\n';
        std::cout << "end_time_s: " << kTotalCouplingSteps * kCouplingDt << '\n';
        std::cout << "external_force_feedback_N: "
                  << external_force.force[0] << ','
                  << external_force.force[1] << ','
                  << external_force.force[2] << '\n';
        std::cout << "callback_calls: " << external_force.callback_calls << '\n';
        std::cout << "atom1_force_updates: " << external_force.atom1_updates << '\n';
        std::cout << "final_lammps_center_m: "
                  << final_motion_sample.dem_state.center[0] << ','
                  << final_motion_sample.dem_state.center[1] << ','
                  << final_motion_sample.dem_state.center[2] << '\n';
        std::cout << "final_sph_geometric_center_m: "
                  << final_motion_sample.sph_geometric_center[0] << ','
                  << final_motion_sample.sph_geometric_center[1] << ','
                  << final_motion_sample.sph_geometric_center[2] << '\n';
        std::cout << "final_vz_m_per_s: " << final_motion_sample.dem_state.velocity[2] << '\n';
        std::cout << "final_z_exact_m: " << final_motion_sample.z_exact << '\n';
        std::cout << "final_vz_exact_m_per_s: " << final_motion_sample.vz_exact << '\n';
        std::cout << "max_center_error_m: " << max_center_error << '\n';
        std::cout << "max_abs_z_error_m: " << max_abs_z_error << '\n';
        std::cout << "max_abs_vz_error_m_per_s: " << max_abs_vz_error << '\n';
        std::cout << "pre_entry_samples: " << force_stats.pre_entry_count << '\n';
        std::cout << "pre_entry_mean_force_N: "
                  << pre_entry_mean_force[0] << ','
                  << pre_entry_mean_force[1] << ','
                  << pre_entry_mean_force[2] << '\n';
        std::cout << "pre_entry_mean_force_norm_N: " << pre_entry_mean_force_norm << '\n';
        std::cout << "post_entry_samples: " << force_stats.post_entry_count << '\n';
        std::cout << "post_entry_max_Fz_N: " << post_entry_max_fz << '\n';

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
        if (external_force.force[0] != 0.0 || external_force.force[1] != 0.0 || external_force.force[2] != 0.0)
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
        if (max_abs_z_error > kFreeFallErrorTolerance || max_abs_vz_error > kFreeFallErrorTolerance)
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
