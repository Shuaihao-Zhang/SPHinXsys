/* ------------------------------------------------------------------------- *
 *                                SPHinXsys                                  *
 * ------------------------------------------------------------------------- *
 * This example is a first two-way loose-coupling test for an underwater
 * granular column. LAMMPS owns the circular DEM grains, their gravity,
 * grain-grain contact, and grain-wall contact. SPHinXsys owns the water tank,
 * the moving DEM proxy boundary, and the hydrodynamic force calculation.
 *
 * At each SPH acoustic step, SPHinXsys sums the hydrodynamic force on every
 * DEM proxy grain, relaxes/caps that force, passes it to LAMMPS through
 * fix external, then advances LAMMPS over the same acoustic interval using
 * an integer number of nominal DEM substeps. A shortened DEM step is used
 * only when the SPH stability limit is below one nominal DEM timestep.
 * ------------------------------------------------------------------------- */
#include "test_2d_lammps_underwater_granular_collapse.h"

using namespace SPH;
using namespace LammpsUnderwaterGranularCollapse2D;

int main(int ac, char *av[])
{
    try
    {
        //----------------------------------------------------------------------
        //	Build up a SPHSystem and handle command line options.
        //----------------------------------------------------------------------
        SPHSystem sph_system(kSystemDomainBounds, kParticleSpacing);
        sph_system.setRunParticleRelaxation(false);
        sph_system.setReloadParticles(false);
        sph_system.handleCommandlineOptions(ac, av);

        if (sph_system.RunParticleRelaxation())
        {
            sph_system.setReloadParticles(false);
            std::cout << "Particle relaxation mode: reload particles disabled; generating water and granular column particles from lattice.\n";
        }
        else if (sph_system.ReloadParticles() && !reload_particle_file_exists())
        {
            std::cout << "WARNING: particle reload was requested, but "
                      << std::filesystem::absolute(reload_particle_file()).string()
                      << " was not found. Falling back to lattice particles.\n";
            sph_system.setReloadParticles(false);
        }
        else if (sph_system.ReloadParticles())
        {
            if (!reload_particle_body_exists(kRelaxedWaterReloadBodyName))
            {
                std::cout << "WARNING: reload file does not contain " << kRelaxedWaterReloadBodyName
                          << ". WaterBody will use lattice particles. Run with --relax=true to generate"
                          << " a joint water-grain reload file.\n";
            }
            if (!reload_particle_body_exists(kRelaxedGranularColumnReloadBodyName))
            {
                std::cout << "WARNING: reload file does not contain "
                          << kRelaxedGranularColumnReloadBodyName
                          << ". Granular proxy will use lattice particles.\n";
            }
        }

        //----------------------------------------------------------------------
        //	Creating bodies with corresponding materials and particles.
        //----------------------------------------------------------------------
        SolidBody grain_boundary(
            sph_system, makeShared<GranularColumnShape>("LammpsUnderwaterGranularColumn"));
        grain_boundary.defineMatterMaterial<Solid>(kGrainDensity);
        const std::string grain_particle_source =
            generate_granular_column_boundary_particles(sph_system, grain_boundary);

        FluidBody water_block(sph_system, makeShared<WaterBlock>("WaterBody"));
        water_block.defineBodyLevelSetShape();
        water_block.defineMatterMaterial<WeaklyCompressibleFluid>(kWaterDensity, kSoundSpeed);
        water_block.addMaterialProperty<Viscosity>(kDynamicViscosity);
        water_block.generateParticles<BaseParticles, Lattice>();
        //const std::string water_particle_source =
        //    generate_water_particles(sph_system, water_block);
        //const InitialWaterGeometryDiagnostics initial_water_geometry =
        //    diagnose_initial_water_geometry(water_block);

        SolidBody wall_boundary(sph_system, makeShared<WallBoundary>("WallBoundary"));
        wall_boundary.defineMatterMaterial<Solid>();
        wall_boundary.generateParticles<BaseParticles, Lattice>();

        if (sph_system.RunParticleRelaxation())
        {
            return run_underwater_granular_particle_relaxation(water_block, grain_boundary, wall_boundary);
        }

        DrivenGranularColumnBoundary driven_column(grain_boundary, initial_dem_centers());

        //----------------------------------------------------------------------
        //	Define body relations.
        //----------------------------------------------------------------------
        InnerRelation water_inner(water_block);
        ContactRelation water_contact(water_block, {&wall_boundary, &grain_boundary});
        ContactRelation grain_contact(grain_boundary, {&water_block});
        ComplexRelation water_complex(water_inner, water_contact);

        //----------------------------------------------------------------------
        //	Define all numerical methods for the fluid dynamics.
        //----------------------------------------------------------------------
        SimpleDynamics<HydrostaticPressureField> hydrostatic_pressure(water_block);
        Gravity gravity(Vec2d(0.0, -kGravity));
        SimpleDynamics<GravityForce<Gravity>> constant_gravity(water_block, gravity);
        SimpleDynamics<NormalDirectionFromBodyShape> wall_normal_direction(wall_boundary);
        SimpleDynamics<NormalDirectionFromBodyShape> grain_normal_direction(grain_boundary);

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
        //	Fluid-to-DEM force accumulation on the moving grain proxy body.
        //----------------------------------------------------------------------
        InteractionWithUpdate<solid_dynamics::ViscousForceFromFluid>
            viscous_force_on_grains(grain_contact);
        InteractionWithUpdate<solid_dynamics::PressureForceFromFluid<decltype(density_relaxation)>>
            pressure_force_on_grains(grain_contact);

        //----------------------------------------------------------------------
        //	LAMMPS owns DEM motion. The callback receives only the hydrodynamic
        //	force from SPHinXsys; LAMMPS fix gravity and granular wall/contact
        //	forces remain active inside LAMMPS.
        //----------------------------------------------------------------------
        LammpsGranularColumnAdapter dem_adapter;
        const Real lammps_particle_mass = dem_adapter.particleMass();
        const Real lammps_mass_relative_error =
            std::abs(lammps_particle_mass - grain_mass()) / grain_mass();
        std::vector<Vec2d> previous_applied_forces(kParticleCount, Vec2d::Zero());
        std::vector<int> previous_capped_flags(kParticleCount, 0);
        dem_adapter.setExternalForces(previous_applied_forces);

        //----------------------------------------------------------------------
        //	Define the methods for I/O operations and observations.
        //----------------------------------------------------------------------
        BodyStatesRecordingToVtp write_real_body_states(sph_system);
        write_real_body_states.addToWrite<Real>(water_block, "Pressure");
        write_real_body_states.addToWrite<Vecd>(wall_boundary, "NormalDirection");
        write_real_body_states.addToWrite<Vecd>(grain_boundary, "Velocity");
        write_real_body_states.addToWrite<Vecd>(grain_boundary, "Acceleration");
        write_real_body_states.addToWrite<Vecd>(grain_boundary, "PressureForceFromFluid");
        write_real_body_states.addToWrite<Vecd>(grain_boundary, "ViscousForceFromFluid");
        write_real_body_states.addToWrite<Vecd>(grain_boundary, "NormalDirection");

        const std::filesystem::path motion_csv_path = "granular_motion.csv";
        const std::filesystem::path force_csv_path = "granular_force.csv";
        const std::filesystem::path runtime_path = SPH::lammps_examples::lammps_runtime_path();
        const std::filesystem::path output_path = std::filesystem::absolute(IO::getEnvironment().OutputFolder());
        VtpPvdWriter water_body_pvd("WaterBody");
        VtpPvdWriter wall_boundary_pvd("WallBoundary");
        VtpPvdWriter grain_proxy_pvd("LammpsUnderwaterGranularColumn");
        VtpPvdWriter dem_discs_pvd("DEM_Discs");
        VtpPvdWriter dem_forces_pvd("DEM_Forces");
        auto output_file_exists_with_prefix = [](const std::filesystem::path &folder,
                                                 const std::string &prefix) -> bool
        {
            if (!std::filesystem::exists(folder))
            {
                return false;
            }
            for (const auto &entry : std::filesystem::directory_iterator(folder))
            {
                if (entry.is_regular_file() &&
                    entry.path().filename().string().rfind(prefix, 0) == 0)
                {
                    return true;
                }
            }
            return false;
        };

        std::ofstream motion_csv(motion_csv_path);
        std::ofstream force_csv(force_csv_path);
        if (!motion_csv)
        {
            throw std::runtime_error("could not open granular_motion.csv for writing");
        }
        if (!force_csv)
        {
            throw std::runtime_error("could not open granular_force.csv for writing");
        }
        motion_csv << std::setprecision(17);
        force_csv << std::setprecision(17);
        write_motion_csv_header(motion_csv);
        write_force_csv_header(force_csv);

        //----------------------------------------------------------------------
        //	Prepare the simulation with cell linked lists, configuration and state.
        //----------------------------------------------------------------------
        sph_system.initializeSystemCellLinkedLists();
        sph_system.initializeSystemConfigurations();
        hydrostatic_pressure.exec();
        constant_gravity.exec();
        wall_normal_direction.exec();
        grain_normal_direction.exec();

        Real &physical_time = *sph_system.getSystemVariableDataByName<Real>("PhysicalTime");
        physical_time = 0.0;

        //----------------------------------------------------------------------
        //	Initial diagnostics and first explicit hydrodynamic force feedback.
        //----------------------------------------------------------------------
        Real max_center_error = 0.0;
        Real max_raw_hydro_force_norm = 0.0;
        Real max_raw_hydro_force_norm_during_relaxation = 0.0;
        Real max_raw_hydro_force_norm_after_coupling = 0.0;
        Real max_applied_hydro_force_norm = 0.0;
        Real final_right_front = initial_right_front_x();
        bool finite_state = true;
        int total_cap_count = 0;
        long long coupled_particle_force_samples = 0;
        int number_of_iterations = 0;
        int lammps_step_count = 0;
        Real max_lammps_clock_error = 0.0;
        Real driver_lammps_offset = 0.0;
        auto is_coupling_active = [](Real time) -> bool
        {
            return time + TinyReal >= kFluidRelaxationTime;
        };

        std::vector<DEMParticleState> dem_states = dem_adapter.pullStates();
        driven_column.update(dem_states);
        update_water_grain_configuration(water_block, grain_boundary, water_complex, grain_contact);
        viscous_force_on_grains.exec();
        pressure_force_on_grains.exec();

        std::vector<Vec2d> raw_hydro_forces = sum_grain_hydro_forces(grain_boundary, driven_column.particleDemIds());
        std::vector<Real> raw_hydrodynamic_torques =
            sum_grain_hydro_torques(grain_boundary, driven_column.particleDemIds(), dem_states);
        ForceApplication force_application;
        force_application.applied_forces = previous_applied_forces;
        force_application.capped_flags = previous_capped_flags;
        dem_adapter.setExternalForces(previous_applied_forces);

        std::vector<Vec2d> proxy_centers = driven_column.geometricCenters();
        max_center_error = std::max(max_center_error, max_center_sync_error(dem_states, proxy_centers));
        max_raw_hydro_force_norm = std::max(max_raw_hydro_force_norm, max_hydro_force_norm(raw_hydro_forces));
        max_raw_hydro_force_norm_during_relaxation =
            std::max(max_raw_hydro_force_norm_during_relaxation, max_hydro_force_norm(raw_hydro_forces));
        max_applied_hydro_force_norm =
            std::max(max_applied_hydro_force_norm, max_hydro_force_norm(previous_applied_forces));
        finite_state = finite_state && all_finite(dem_states, raw_hydro_forces) &&
                       all_finite(dem_states, previous_applied_forces);

        write_motion_csv_samples(motion_csv, 0, physical_time, is_coupling_active(physical_time),
                                 lammps_step_count, dem_states, proxy_centers);
        write_force_csv_samples(force_csv, 0, physical_time, is_coupling_active(physical_time), raw_hydro_forces,
                                previous_applied_forces, raw_hydrodynamic_torques, previous_capped_flags);
        write_real_body_states.writeToFile(0);
        water_body_pvd.add(physical_time, lammps_example_output_path("WaterBody", 0));
        wall_boundary_pvd.add(physical_time, lammps_example_output_path("WallBoundary", 0));
        grain_proxy_pvd.add(physical_time, lammps_example_output_path("LammpsUnderwaterGranularColumn", 0));
        std::filesystem::path latest_dem_discs_vtp =
            write_dem_discs_to_vtp(0, physical_time, "DEM_Discs", centers_from_states(dem_states), kGrainRadius);
        std::filesystem::path latest_dem_forces_vtp =
            write_dem_force_vtp(0, physical_time, dem_states, raw_hydro_forces,
                                previous_applied_forces, raw_hydrodynamic_torques);
        dem_discs_pvd.add(physical_time, latest_dem_discs_vtp);
        dem_forces_pvd.add(physical_time, latest_dem_forces_vtp);
        int dem_visualization_output_count = 1;

        //----------------------------------------------------------------------
        //	Main loop starts here. The outer loop follows the fluid advection
        //	step. The inner acoustic loop is the two-way loose-coupling point:
        //	SPHinXsys computes per-grain hydrodynamic force, sends it to the
        //	LAMMPS callback, then LAMMPS advances the granular column over the
        //	same acoustic interval with smaller DEM substeps.
        //----------------------------------------------------------------------
        TickCount t1 = TickCount::now();
        TimeInterval interval;
        const int screen_output_interval = 500;
        int advection_iterations = 0;
        int output_iteration = 0;
        Real next_output_time = kVtpOutputInterval;

        while (physical_time < kEndTime - TinyReal)
        {
            ++advection_iterations;

            update_water_grain_configuration(water_block, grain_boundary, water_complex, grain_contact);
            Real advection_step = SMIN(get_fluid_advection_time_step_size.exec(), kEndTime - physical_time);
            update_density_by_summation.exec();
            viscous_force.exec();
            viscous_force_on_grains.exec();

            Real relaxation_time = 0.0;
            while (relaxation_time < advection_step - TinyReal && physical_time < kEndTime - TinyReal)
            {
                const bool coupling_active_for_step = is_coupling_active(physical_time);
                const Real phase_limit_time =
                    coupling_active_for_step ? kEndTime : SMIN(kFluidRelaxationTime, kEndTime);
                const Real acoustic_limit = SMIN(get_fluid_time_step_size.exec(),
                                                 SMIN(advection_step - relaxation_time,
                                                      SMIN(phase_limit_time - physical_time, kEndTime - physical_time)));
                const CouplingStepPlan coupling_plan = dem_adapter.planCouplingStep(acoustic_limit);
                const Real acoustic_step = coupling_plan.coupled_dt;
                pressure_relaxation.exec(acoustic_step);
                pressure_force_on_grains.exec();
                density_relaxation.exec(acoustic_step);

                raw_hydro_forces = sum_grain_hydro_forces(grain_boundary, driven_column.particleDemIds());
                raw_hydrodynamic_torques =
                    sum_grain_hydro_torques(grain_boundary, driven_column.particleDemIds(), dem_states);
                if (coupling_active_for_step)
                {
                    force_application = relax_and_cap_forces(raw_hydro_forces, previous_applied_forces);
                    previous_applied_forces = force_application.applied_forces;
                    previous_capped_flags = force_application.capped_flags;
                    dem_adapter.setExternalForces(previous_applied_forces);
                    total_cap_count += force_application.cap_count();
                    coupled_particle_force_samples += kParticleCount;

                    const CouplingAdvanceResult coupling_advance =
                        dem_adapter.advance(coupling_plan, physical_time);
                    lammps_step_count += coupling_advance.dem_steps;
                    max_lammps_clock_error =
                        std::max(max_lammps_clock_error,
                                 static_cast<Real>(coupling_advance.synchronization_error));
                    driver_lammps_offset =
                        static_cast<Real>(coupling_advance.driver_lammps_offset);
                    dem_states = dem_adapter.pullStates();
                    driven_column.update(dem_states);
                    max_raw_hydro_force_norm_after_coupling =
                        std::max(max_raw_hydro_force_norm_after_coupling, max_hydro_force_norm(raw_hydro_forces));
                }
                else
                {
                    std::fill(previous_applied_forces.begin(), previous_applied_forces.end(), Vec2d::Zero());
                    std::fill(previous_capped_flags.begin(), previous_capped_flags.end(), 0);
                    force_application.applied_forces = previous_applied_forces;
                    force_application.capped_flags = previous_capped_flags;
                    dem_adapter.setExternalForces(previous_applied_forces);
                    driven_column.update(dem_states);
                    max_raw_hydro_force_norm_during_relaxation =
                        std::max(max_raw_hydro_force_norm_during_relaxation, max_hydro_force_norm(raw_hydro_forces));
                }

                relaxation_time += acoustic_step;
                physical_time += acoustic_step;
                ++number_of_iterations;

                proxy_centers = driven_column.geometricCenters();
                const Real center_error = max_center_sync_error(dem_states, proxy_centers);
                max_center_error = std::max(max_center_error, center_error);
                max_raw_hydro_force_norm =
                    std::max(max_raw_hydro_force_norm, max_hydro_force_norm(raw_hydro_forces));
                max_applied_hydro_force_norm =
                    std::max(max_applied_hydro_force_norm, max_hydro_force_norm(previous_applied_forces));
                final_right_front = right_front_x(dem_states);
                finite_state = finite_state && all_finite(dem_states, raw_hydro_forces) &&
                               all_finite(dem_states, previous_applied_forces) &&
                               std::isfinite(center_error);

                const bool write_csv_sample =
                    number_of_iterations % kCsvOutputStride == 0 ||
                    physical_time + TinyReal >= next_output_time ||
                    physical_time + TinyReal >= kEndTime;
                if (write_csv_sample)
                {
                    write_motion_csv_samples(motion_csv, number_of_iterations, physical_time,
                                             coupling_active_for_step, lammps_step_count,
                                             dem_states, proxy_centers);
                    write_force_csv_samples(force_csv, number_of_iterations, physical_time,
                                            coupling_active_for_step, raw_hydro_forces,
                                            previous_applied_forces, raw_hydrodynamic_torques,
                                            previous_capped_flags);
                }

                if (number_of_iterations % screen_output_interval == 0)
                {
                    std::cout << std::fixed << std::setprecision(6)
                              << "N=" << number_of_iterations
                              << " Time = " << physical_time
                              << " phase = " << (coupling_active_for_step ? "two_way" : "fluid_relax")
                              << " advection_step = " << advection_step
                              << " acoustic_limit = " << acoustic_limit
                              << " acoustic_step = " << acoustic_step
                              << " right_front = " << final_right_front
                              << " max_F_raw = " << max_hydro_force_norm(raw_hydro_forces)
                              << " max_F_applied = " << max_hydro_force_norm(previous_applied_forces)
                              << " capped_particles = " << force_application.cap_count() << "\n";
                }

                if (physical_time + TinyReal >= next_output_time || physical_time + TinyReal >= kEndTime)
                {
                    output_iteration = number_of_iterations;
                    write_real_body_states.writeToFile(output_iteration);
                    water_body_pvd.add(physical_time, lammps_example_output_path("WaterBody", output_iteration));
                    grain_proxy_pvd.add(physical_time, lammps_example_output_path("LammpsUnderwaterGranularColumn", output_iteration));
                    latest_dem_discs_vtp =
                        write_dem_discs_to_vtp(output_iteration, physical_time, "DEM_Discs",
                                               centers_from_states(dem_states), kGrainRadius);
                    latest_dem_forces_vtp =
                        write_dem_force_vtp(output_iteration, physical_time, dem_states,
                                            raw_hydro_forces, previous_applied_forces,
                                            raw_hydrodynamic_torques);
                    dem_discs_pvd.add(physical_time, latest_dem_discs_vtp);
                    dem_forces_pvd.add(physical_time, latest_dem_forces_vtp);
                    ++dem_visualization_output_count;
                    next_output_time += kVtpOutputInterval;
                }
            }

            TickCount t2 = TickCount::now();
            if (advection_iterations % 100 == 0)
            {
                particle_sorting.exec();
            }
            update_water_grain_configuration(water_block, grain_boundary, water_complex, grain_contact);
            TickCount t3 = TickCount::now();
            interval += t3 - t2;
        }
        TickCount t4 = TickCount::now();

        motion_csv.close();
        force_csv.close();

        //----------------------------------------------------------------------
        //	Statistics and regression-style checks.
        //----------------------------------------------------------------------
        const ExternalForceBuffer &external_force = dem_adapter.externalForceTable();
        const Real right_front_displacement = final_right_front - initial_right_front_x();
        const TickCount::interval_t tt = t4 - t1 - interval;

        std::cout << std::setprecision(17);
        std::cout << "SPHinXsys 2D two-way LAMMPS underwater granular collapse example\n";
        std::cout << "Total wall time for computation: " << tt.seconds() << " seconds.\n";
        std::cout << "lammps_runtime_in_working_directory: " << (std::filesystem::exists(runtime_path) ? "yes" : "no") << '\n';
        std::cout << "lammps_version: " << dem_adapter.version() << '\n';
        std::cout << "granular_motion_csv: " << std::filesystem::absolute(motion_csv_path).string() << '\n';
        std::cout << "granular_force_csv: " << std::filesystem::absolute(force_csv_path).string() << '\n';
        std::cout << "VTP_output_folder: " << output_path.string() << '\n';
        std::cout << "DEM_discs_vtp_latest: " << std::filesystem::absolute(latest_dem_discs_vtp).string() << '\n';
        std::cout << "DEM_forces_vtp_latest: " << std::filesystem::absolute(latest_dem_forces_vtp).string() << '\n';
        std::cout << "WaterBody_pvd: " << std::filesystem::absolute(water_body_pvd.path()).string() << '\n';
        std::cout << "SPH_grain_proxy_pvd: " << std::filesystem::absolute(grain_proxy_pvd.path()).string() << '\n';
        std::cout << "DEM_discs_pvd: " << std::filesystem::absolute(dem_discs_pvd.path()).string() << '\n';
        std::cout << "DEM_forces_pvd: " << std::filesystem::absolute(dem_forces_pvd.path()).string() << '\n';
        std::cout << "DEM_visualization_output_count: " << dem_visualization_output_count << '\n';
        std::cout << "water_particles: " << water_block.getBaseParticles().TotalRealParticles() << '\n';
        std::cout << "wall_particles: " << wall_boundary.getBaseParticles().TotalRealParticles() << '\n';
        std::cout << "grain_proxy_particles: " << driven_column.particleCount() << '\n';
        //std::cout << "water_particle_source: " << water_particle_source << '\n';
        std::cout << "grain_particle_source: " << grain_particle_source << '\n';
        std::cout << "initial_wall_clearance_m: " << kInitialWallClearance << '\n';
        std::cout << "initial_water_grain_gap_m: " << kInitialWaterGrainGap << '\n';
        std::cout << "water_void_radius_m: " << kWaterVoidRadius << '\n';
        //std::cout << "initial_water_min_signed_distance_to_dem_surface_m: "
        //          << initial_water_geometry.min_signed_distance_to_dem_surface << '\n';
        //std::cout << "initial_water_inside_dem_surface_count: "
        //          << initial_water_geometry.water_inside_dem_surface_count << '\n';
        //std::cout << "initial_water_within_half_dp_of_dem_surface_count: "
        //          << initial_water_geometry.water_within_half_dp_of_dem_surface_count << '\n';
        //std::cout << "initial_water_within_one_dp_of_dem_surface_count: "
        //          << initial_water_geometry.water_within_one_dp_of_dem_surface_count << '\n';
        //std::cout << "initial_water_in_granular_region_count: "
        //          << initial_water_geometry.water_in_granular_region_count << '\n';
        std::cout << "dem_particle_count: " << kParticleCount << '\n';
        std::cout << "grain_mass_per_unit_depth_kg_per_m: " << grain_mass() << '\n';
        std::cout << "lammps_equivalent_sphere_density_kg_per_m3: " << lammps_equivalent_sphere_density() << '\n';
        std::cout << "lammps_particle_mass_kg: " << lammps_particle_mass << '\n';
        std::cout << "lammps_particle_mass_relative_error: " << lammps_mass_relative_error << '\n';
        std::cout << "grain_weight_per_unit_depth_N_per_m: " << grain_weight() << '\n';
        std::cout << "force_relaxation_alpha: " << kForceRelaxationAlpha << '\n';
        std::cout << "force_cap_N_per_m_per_particle: " << kForceCapWeightFactor * grain_weight() << '\n';
        std::cout << "force_cap_trigger_count: " << total_cap_count << '\n';
        const Real force_cap_fraction = coupled_particle_force_samples > 0
                                            ? static_cast<Real>(total_cap_count) /
                                                  static_cast<Real>(coupled_particle_force_samples)
                                            : 0.0;
        std::cout << "force_cap_fraction: " << force_cap_fraction << '\n';
        std::cout << "momentum_conservation_validation: "
                  << (total_cap_count == 0 ? "eligible" : "not_eligible_capped_2d_case") << '\n';
        std::cout << "csv_output_stride: " << kCsvOutputStride << '\n';
        std::cout << "out_of_plane_depth_m: " << kOutOfPlaneDepth << '\n';
        std::cout << "fluid_relaxation_time_s: " << kFluidRelaxationTime << '\n';
        std::cout << "dem_motion_enabled_during_fluid_relaxation: no\n";
        std::cout << "hydrodynamic_feedback_enabled_during_fluid_relaxation: no\n";
        std::cout << "dem_max_dt_s: " << kDemMaxDt << '\n';
        std::cout << "advection_iterations: " << advection_iterations << '\n';
        std::cout << "number_of_iterations: " << number_of_iterations << '\n';
        std::cout << "lammps_substeps_executed: " << lammps_step_count << '\n';
        std::cout << "driver_lammps_offset_s: " << driver_lammps_offset << '\n';
        std::cout << "max_lammps_clock_error_s: " << max_lammps_clock_error << '\n';
        std::cout << "hydrodynamic_torque_feedback_enabled: no\n";
        std::cout << "end_time_s: " << kEndTime << '\n';
        std::cout << "initial_right_front_x_m: " << initial_right_front_x() << '\n';
        std::cout << "final_right_front_x_m: " << final_right_front << '\n';
        std::cout << "right_front_displacement_m: " << right_front_displacement << '\n';
        std::cout << "max_center_sync_error_m: " << max_center_error << '\n';
        std::cout << "max_raw_hydro_force_norm_N: " << max_raw_hydro_force_norm << '\n';
        std::cout << "max_raw_hydro_force_norm_during_relaxation_N: "
                  << max_raw_hydro_force_norm_during_relaxation << '\n';
        std::cout << "max_raw_hydro_force_norm_after_coupling_N: "
                  << max_raw_hydro_force_norm_after_coupling << '\n';
        std::cout << "max_applied_hydro_force_norm_N: " << max_applied_hydro_force_norm << '\n';
        std::cout << "external_force_feedback_includes_gravity: no\n";
        std::cout << "callback_calls: " << external_force.callbackCalls() << '\n';
        std::cout << "atom_force_updates: " << external_force.atomUpdates() << '\n';
        std::cout << "finite_state: " << (finite_state ? "yes" : "no") << '\n';

        if (!std::filesystem::exists(runtime_path))
        {
            std::cerr << "ERROR: the LAMMPS runtime library was not staged next to the executable.\n";
            return 1;
        }
        if (external_force.callbackCalls() <= 0 || external_force.atomUpdates() <= 0)
        {
            std::cerr << "ERROR: fix external callback did not update DEM particles.\n";
            return 1;
        }
        if (!finite_state)
        {
            std::cerr << "ERROR: non-finite value detected in DEM state or force history.\n";
            return 1;
        }
        if (lammps_mass_relative_error > kMassRelativeTolerance)
        {
            std::cerr << "ERROR: LAMMPS particle mass does not match the SPHinXsys 2D unit-depth grain mass.\n";
            return 1;
        }
        if (max_center_error > kCenterErrorTolerance)
        {
            std::cerr << "ERROR: SPHinXsys DEM proxy center drift exceeded tolerance "
                      << kCenterErrorTolerance << ".\n";
            return 1;
        }
        if (right_front_displacement < kMinimumRightFrontDisplacement)
        {
            std::cerr << "ERROR: granular column did not move to the right enough to indicate collapse.\n";
            return 1;
        }
        if (max_applied_hydro_force_norm < kMinimumHydroForceNorm)
        {
            std::cerr << "ERROR: hydrodynamic force feedback remained effectively zero.\n";
            return 1;
        }
        if (!std::filesystem::exists(motion_csv_path) || !std::filesystem::exists(force_csv_path))
        {
            std::cerr << "ERROR: granular CSV output files were not written.\n";
            return 1;
        }
        if (!output_file_exists_with_prefix(output_path, "DEM_Discs_ite_"))
        {
            std::cerr << "ERROR: DEM disc VTP visualization files were not written.\n";
            return 1;
        }
        if (!output_file_exists_with_prefix(output_path, "DEM_Forces_ite_"))
        {
            std::cerr << "ERROR: DEM force VTP visualization files were not written.\n";
            return 1;
        }
        if (!std::filesystem::exists(latest_dem_discs_vtp) || !std::filesystem::exists(latest_dem_forces_vtp))
        {
            std::cerr << "ERROR: latest DEM visualization VTP file is missing.\n";
            return 1;
        }
        if (!std::filesystem::exists(water_body_pvd.path()) ||
            !std::filesystem::exists(grain_proxy_pvd.path()) ||
            !std::filesystem::exists(dem_discs_pvd.path()) ||
            !std::filesystem::exists(dem_forces_pvd.path()))
        {
            std::cerr << "ERROR: PVD time-series files were not written.\n";
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
