/* ------------------------------------------------------------------------- *
 *                                SPHinXsys                                  *
 * ------------------------------------------------------------------------- *
 * This case is a coarse 3D two-way loose-coupling validation for an
 * underwater granular column. SPHinXsys owns the water dynamics and
 * evaluates per-grain hydrodynamic loads. LAMMPS owns every DEM grain,
 * including gravity, granular contacts, wall contacts, and DEM subcycling.
 * ------------------------------------------------------------------------- */
#include "test_3d_lammps_underwater_granular_collapse.h"

using namespace SPH;
using namespace LammpsUnderwaterGranularCollapse3D;

namespace
{
bool output_file_exists_with_prefix(const std::filesystem::path &folder, const std::string &prefix)
{
    if (!std::filesystem::exists(folder))
    {
        return false;
    }
    for (const auto &entry : std::filesystem::directory_iterator(folder))
    {
        if (entry.is_regular_file() && entry.path().filename().string().rfind(prefix, 0) == 0)
        {
            return true;
        }
    }
    return false;
}
} // namespace

int main(int ac, char *av[])
{
    try
    {
        //----------------------------------------------------------------------
        //  Build the SPH system. Particle relaxation is opt-in because the
        //  default integration path intentionally keeps the water lattice and
        //  initial dry shells identical to the 2D reference case.
        //----------------------------------------------------------------------
        SPHSystem sph_system(kSystemDomainBounds, kParticleSpacing);
        sph_system.setRunParticleRelaxation(false);
        sph_system.setReloadParticles(false);
        sph_system.handleCommandlineOptions(ac, av);
        verify_initial_transverse_confinement();
        if (sph_system.RunParticleRelaxation())
        {
            sph_system.setReloadParticles(false);
            std::cout << "Particle relaxation mode: generating the 3D granular proxy from lattice particles.\n";
        }

        //----------------------------------------------------------------------
        //  Create the SPHinXsys proxy body before water. The proxy is used for
        //  fluid-solid interaction only; its motion is overwritten from the
        //  corresponding LAMMPS particle state after every coupling advance.
        //----------------------------------------------------------------------
        SolidBody grain_boundary(
            sph_system, makeShared<GranularColumnShape>(kRelaxedGranularColumnReloadBodyName));
        grain_boundary.defineMatterMaterial<Solid>(kGrainDensity);
        const std::string grain_particle_source =
            generate_granular_column_boundary_particles(sph_system, grain_boundary);
        if (sph_system.RunParticleRelaxation())
        {
            return run_granular_proxy_particle_relaxation(grain_boundary);
        }
        DrivenGranularColumnBoundary driven_column(grain_boundary, initial_dem_centers());

        FluidBody water_block(sph_system, makeShared<WaterBlock>("WaterBody"));
        water_block.defineBodyLevelSetShape();
        water_block.defineMatterMaterial<WeaklyCompressibleFluid>(kWaterDensity, kSoundSpeed);
        water_block.addMaterialProperty<Viscosity>(kDynamicViscosity);
        water_block.generateParticles<BaseParticles, Lattice>();

        SolidBody wall_boundary(sph_system, makeShared<WallBoundary>("WallBoundary"));
        wall_boundary.defineMatterMaterial<Solid>();
        wall_boundary.generateParticles<BaseParticles, Lattice>();

        //----------------------------------------------------------------------
        //  Define SPH relations and fluid dynamics. The grain contact relation
        //  is also the SPHinXsys-side receiver for pressure and viscous loads.
        //----------------------------------------------------------------------
        InnerRelation water_inner(water_block);
        ContactRelation water_contact(water_block, {&wall_boundary, &grain_boundary});
        ContactRelation grain_contact(grain_boundary, {&water_block});
        ComplexRelation water_complex(water_inner, water_contact);

        SimpleDynamics<HydrostaticPressureField> hydrostatic_pressure(water_block);
        Gravity gravity(Vec3d(0.0, 0.0, -kGravity));
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
        ReduceDynamics<fluid_dynamics::AcousticTimeStep> get_fluid_time_step_size(water_block);
        ParticleSorting particle_sorting(water_block);
        InteractionWithUpdate<solid_dynamics::ViscousForceFromFluid>
            viscous_force_on_grains(grain_contact);
        InteractionWithUpdate<solid_dynamics::PressureForceFromFluid<decltype(density_relaxation)>>
            pressure_force_on_grains(grain_contact);

        //----------------------------------------------------------------------
        //  LAMMPS is the DEM owner. The external callback contains only the
        //  SPHinXsys hydrodynamic force: LAMMPS fix gravity remains active and
        //  therefore gravity is never added to the callback force.
        //----------------------------------------------------------------------
        LammpsGranularColumnAdapter dem_adapter;
        const Real lammps_particle_mass = dem_adapter.particleMass();
        const Real lammps_mass_relative_error =
            std::abs(lammps_particle_mass - grain_mass()) / grain_mass();
        std::vector<Vec3d> previous_applied_forces(kParticleCount, Vec3d::Zero());
        std::vector<int> previous_capped_flags(kParticleCount, 0);
        dem_adapter.setExternalForces(previous_applied_forces);

        //----------------------------------------------------------------------
        //  Configure SPH and DEM output. DEM_Spheres is the visual geometry;
        //  DEM_Forces is a point data set suitable for ParaView Glyph arrows.
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
        VtpPvdWriter grain_proxy_pvd(kRelaxedGranularColumnReloadBodyName);
        VtpPvdWriter dem_spheres_pvd("DEM_Spheres");
        VtpPvdWriter dem_forces_pvd("DEM_Forces");

        std::ofstream motion_csv(motion_csv_path);
        std::ofstream force_csv(force_csv_path);
        if (!motion_csv || !force_csv)
        {
            throw std::runtime_error("could not open 3D granular collapse CSV output files");
        }
        motion_csv << std::setprecision(17);
        force_csv << std::setprecision(17);
        write_motion_csv_header(motion_csv);
        write_force_csv_header(force_csv);

        sph_system.initializeSystemCellLinkedLists();
        sph_system.initializeSystemConfigurations();
        hydrostatic_pressure.exec();
        constant_gravity.exec();
        wall_normal_direction.exec();
        grain_normal_direction.exec();

        Real &physical_time = *sph_system.getSystemVariableDataByName<Real>("PhysicalTime");
        physical_time = 0.0;
        int number_of_iterations = 0;
        int lammps_step_count = 0;
        int advection_iterations = 0;
        int total_cap_count = 0;
        long long coupled_particle_force_samples = 0;
        Real max_center_error = 0.0;
        Real max_raw_hydro_force_norm = 0.0;
        Real max_applied_hydro_force_norm = 0.0;
        Real max_lammps_clock_error = 0.0;
        Real final_right_front = initial_right_front_x();
        bool finite_state = true;

        std::vector<DEMParticleState> dem_states = dem_adapter.pullStates();
        driven_column.update(dem_states);
        update_water_grain_configuration(water_block, grain_boundary, water_complex, grain_contact);
        viscous_force_on_grains.exec();
        pressure_force_on_grains.exec();
        std::vector<Vec3d> raw_hydro_forces =
            sum_grain_hydro_forces(grain_boundary, driven_column.particleDemIds());
        std::vector<Vec3d> raw_hydrodynamic_torques =
            sum_grain_hydro_torques(grain_boundary, driven_column.particleDemIds(), dem_states);
        std::vector<Vec3d> proxy_centers = driven_column.geometricCenters();

        write_motion_csv_samples(motion_csv, 0, physical_time, lammps_step_count, dem_states, proxy_centers);
        write_force_csv_samples(force_csv, 0, physical_time, raw_hydro_forces, previous_applied_forces,
                                raw_hydrodynamic_torques, previous_capped_flags);
        write_real_body_states.writeToFile(0);
        water_body_pvd.add(physical_time, lammps_example_output_path("WaterBody", 0));
        wall_boundary_pvd.add(physical_time, lammps_example_output_path("WallBoundary", 0));
        grain_proxy_pvd.add(physical_time, lammps_example_output_path(kRelaxedGranularColumnReloadBodyName, 0));
        std::filesystem::path latest_dem_spheres_vtp = write_dem_spheres_to_vtp(
            0, physical_time, "DEM_Spheres", centers_from_states(dem_states), kGrainRadius);
        std::filesystem::path latest_dem_forces_vtp = write_dem_force_vtp(
            0, physical_time, dem_states, raw_hydro_forces, previous_applied_forces,
            raw_hydrodynamic_torques, previous_capped_flags);
        dem_spheres_pvd.add(physical_time, latest_dem_spheres_vtp);
        dem_forces_pvd.add(physical_time, latest_dem_forces_vtp);
        int visualization_output_count = 1;

        //----------------------------------------------------------------------
        //  The outer loop follows the SPH advection step. Each inner acoustic
        //  step is one explicit loose-coupling exchange: SPH advances and
        //  sums per-grain hydrodynamic loads, the loads are relaxed/capped and
        //  passed through fix external, then LAMMPS advances over exactly the
        //  aligned acoustic interval with one or more DEM substeps.
        //----------------------------------------------------------------------
        TickCount t1 = TickCount::now();
        TimeInterval interval;
        const int screen_output_interval = 100;
        int output_iteration = 0;
        Real next_output_time = kVtpOutputInterval;
        while (physical_time < kEndTime - TinyReal)
        {
            ++advection_iterations;
            update_water_grain_configuration(water_block, grain_boundary, water_complex, grain_contact);
            const Real advection_step = SMIN(get_fluid_advection_time_step_size.exec(), kEndTime - physical_time);
            update_density_by_summation.exec();
            viscous_force.exec();
            viscous_force_on_grains.exec();

            Real relaxation_time = 0.0;
            while (relaxation_time < advection_step - TinyReal && physical_time < kEndTime - TinyReal)
            {
                const Real acoustic_limit = SMIN(get_fluid_time_step_size.exec(),
                                                 SMIN(advection_step - relaxation_time, kEndTime - physical_time));
                const CouplingStepPlan coupling_plan = dem_adapter.planCouplingStep(acoustic_limit);
                const Real acoustic_step = coupling_plan.coupled_dt;

                pressure_relaxation.exec(acoustic_step);
                pressure_force_on_grains.exec();
                density_relaxation.exec(acoustic_step);
                raw_hydro_forces = sum_grain_hydro_forces(grain_boundary, driven_column.particleDemIds());
                raw_hydrodynamic_torques =
                    sum_grain_hydro_torques(grain_boundary, driven_column.particleDemIds(), dem_states);

                const ForceApplication force_application =
                    relax_and_cap_forces(raw_hydro_forces, previous_applied_forces);
                previous_applied_forces = force_application.applied_forces;
                previous_capped_flags = force_application.capped_flags;
                total_cap_count += force_application.cap_count();
                coupled_particle_force_samples += kParticleCount;

                // LAMMPS reads this array on every DEM substep. It receives no
                // added gravity and no torque feedback in this first 3D case.
                dem_adapter.setExternalForces(previous_applied_forces);
                const CouplingAdvanceResult coupling_advance =
                    dem_adapter.advance(coupling_plan, physical_time);
                lammps_step_count += coupling_advance.dem_steps;
                max_lammps_clock_error = std::max(
                    max_lammps_clock_error, static_cast<Real>(coupling_advance.synchronization_error));
                dem_states = dem_adapter.pullStates();
                driven_column.update(dem_states);

                relaxation_time += acoustic_step;
                physical_time += acoustic_step;
                ++number_of_iterations;
                proxy_centers = driven_column.geometricCenters();
                const Real center_error = max_center_sync_error(dem_states, proxy_centers);
                max_center_error = std::max(max_center_error, center_error);
                max_raw_hydro_force_norm = std::max(max_raw_hydro_force_norm, max_hydro_force_norm(raw_hydro_forces));
                max_applied_hydro_force_norm =
                    std::max(max_applied_hydro_force_norm, max_hydro_force_norm(previous_applied_forces));
                final_right_front = right_front_x(dem_states);
                finite_state = finite_state && all_finite(dem_states, raw_hydro_forces) &&
                               all_finite(dem_states, previous_applied_forces) && std::isfinite(center_error);

                const bool write_csv = number_of_iterations % kCsvOutputStride == 0 ||
                                       physical_time + TinyReal >= next_output_time ||
                                       physical_time + TinyReal >= kEndTime;
                if (write_csv)
                {
                    write_motion_csv_samples(motion_csv, number_of_iterations, physical_time,
                                             lammps_step_count, dem_states, proxy_centers);
                    write_force_csv_samples(force_csv, number_of_iterations, physical_time,
                                            raw_hydro_forces, previous_applied_forces,
                                            raw_hydrodynamic_torques, previous_capped_flags);
                }

                if (number_of_iterations % screen_output_interval == 0)
                {
                    std::cout << std::fixed << std::setprecision(6)
                              << "N=" << number_of_iterations
                              << " Time=" << physical_time
                              << " advection_step=" << advection_step
                              << " acoustic_step=" << acoustic_step
                              << " dem_steps=" << coupling_advance.dem_steps
                              << " right_front=" << final_right_front
                              << " max_F_raw=" << max_hydro_force_norm(raw_hydro_forces)
                              << " max_F_applied=" << max_hydro_force_norm(previous_applied_forces)
                              << " capped_particles=" << force_application.cap_count() << '\n';
                }

                if (physical_time + TinyReal >= next_output_time || physical_time + TinyReal >= kEndTime)
                {
                    output_iteration = number_of_iterations;
                    write_real_body_states.writeToFile(output_iteration);
                    water_body_pvd.add(physical_time, lammps_example_output_path("WaterBody", output_iteration));
                    wall_boundary_pvd.add(physical_time, lammps_example_output_path("WallBoundary", output_iteration));
                    grain_proxy_pvd.add(
                        physical_time, lammps_example_output_path(kRelaxedGranularColumnReloadBodyName, output_iteration));
                    latest_dem_spheres_vtp = write_dem_spheres_to_vtp(
                        output_iteration, physical_time, "DEM_Spheres", centers_from_states(dem_states), kGrainRadius);
                    latest_dem_forces_vtp = write_dem_force_vtp(
                        output_iteration, physical_time, dem_states, raw_hydro_forces, previous_applied_forces,
                        raw_hydrodynamic_torques, previous_capped_flags);
                    dem_spheres_pvd.add(physical_time, latest_dem_spheres_vtp);
                    dem_forces_pvd.add(physical_time, latest_dem_forces_vtp);
                    ++visualization_output_count;
                    next_output_time += kVtpOutputInterval;
                }
            }

            TickCount t2 = TickCount::now();
            if (advection_iterations % 100 == 0)
            {
                particle_sorting.exec();
            }
            update_water_grain_configuration(water_block, grain_boundary, water_complex, grain_contact);
            interval += TickCount::now() - t2;
        }
        TickCount t4 = TickCount::now();
        motion_csv.close();
        force_csv.close();

        const ExternalForceBuffer &external_force = dem_adapter.externalForceTable();
        const Real right_front_displacement = final_right_front - initial_right_front_x();
        const Real force_cap_fraction = coupled_particle_force_samples > 0
                                            ? static_cast<Real>(total_cap_count) /
                                                  static_cast<Real>(coupled_particle_force_samples)
                                            : 0.0;
        const TickCount::interval_t total_compute_time = t4 - t1 - interval;

        std::cout << std::setprecision(17);
        std::cout << "SPHinXsys 3D two-way LAMMPS underwater granular collapse example\n";
        std::cout << "Total wall time for computation: " << total_compute_time.seconds() << " seconds.\n";
        std::cout << "lammps_runtime_in_working_directory: "
                  << (std::filesystem::exists(runtime_path) ? "yes" : "no") << '\n';
        std::cout << "lammps_version: " << dem_adapter.version() << '\n';
        std::cout << "granular_motion_csv: " << std::filesystem::absolute(motion_csv_path).string() << '\n';
        std::cout << "granular_force_csv: " << std::filesystem::absolute(force_csv_path).string() << '\n';
        std::cout << "VTP_output_folder: " << output_path.string() << '\n';
        std::cout << "WaterBody_pvd: " << std::filesystem::absolute(water_body_pvd.path()).string() << '\n';
        std::cout << "WallBoundary_pvd: " << std::filesystem::absolute(wall_boundary_pvd.path()).string() << '\n';
        std::cout << "SPH_grain_proxy_pvd: " << std::filesystem::absolute(grain_proxy_pvd.path()).string() << '\n';
        std::cout << "DEM_spheres_pvd: " << std::filesystem::absolute(dem_spheres_pvd.path()).string() << '\n';
        std::cout << "DEM_forces_pvd: " << std::filesystem::absolute(dem_forces_pvd.path()).string() << '\n';
        std::cout << "DEM_spheres_vtp_latest: " << std::filesystem::absolute(latest_dem_spheres_vtp).string() << '\n';
        std::cout << "DEM_forces_vtp_latest: " << std::filesystem::absolute(latest_dem_forces_vtp).string() << '\n';
        std::cout << "visualization_output_count: " << visualization_output_count << '\n';
        std::cout << "water_particles: " << water_block.getBaseParticles().TotalRealParticles() << '\n';
        std::cout << "wall_particles: " << wall_boundary.getBaseParticles().TotalRealParticles() << '\n';
        std::cout << "grain_proxy_particles: " << driven_column.particleCount() << '\n';
        std::cout << "grain_particle_source: " << grain_particle_source << '\n';
        std::cout << "dem_particle_count: " << kParticleCount << '\n';
        std::cout << "particle_spacing_m: " << kParticleSpacing << '\n';
        std::cout << "grain_radius_m: " << kGrainRadius << '\n';
        std::cout << "grain_mass_kg: " << grain_mass() << '\n';
        std::cout << "lammps_particle_mass_kg: " << lammps_particle_mass << '\n';
        std::cout << "lammps_particle_mass_relative_error: " << lammps_mass_relative_error << '\n';
        std::cout << "grain_weight_N: " << grain_weight() << '\n';
        std::cout << "initial_water_grain_gap_m: " << kInitialWaterGrainGap << '\n';
        std::cout << "force_relaxation_alpha: " << kForceRelaxationAlpha << '\n';
        std::cout << "force_cap_N_per_particle: " << kForceCapWeightFactor * grain_weight() << '\n';
        std::cout << "force_cap_trigger_count: " << total_cap_count << '\n';
        std::cout << "force_cap_fraction: " << force_cap_fraction << '\n';
        std::cout << "fluid_relaxation_time_s: " << kFluidRelaxationTime << '\n';
        std::cout << "dem_max_dt_s: " << kDemMaxDt << '\n';
        std::cout << "advection_iterations: " << advection_iterations << '\n';
        std::cout << "number_of_iterations: " << number_of_iterations << '\n';
        std::cout << "lammps_substeps_executed: " << lammps_step_count << '\n';
        std::cout << "max_lammps_clock_error_s: " << max_lammps_clock_error << '\n';
        std::cout << "initial_right_front_x_m: " << initial_right_front_x() << '\n';
        std::cout << "final_right_front_x_m: " << final_right_front << '\n';
        std::cout << "right_front_displacement_m: " << right_front_displacement << '\n';
        std::cout << "max_center_sync_error_m: " << max_center_error << '\n';
        std::cout << "max_raw_hydro_force_norm_N: " << max_raw_hydro_force_norm << '\n';
        std::cout << "max_applied_hydro_force_norm_N: " << max_applied_hydro_force_norm << '\n';
        std::cout << "callback_calls: " << external_force.callbackCalls() << '\n';
        std::cout << "atom_force_updates: " << external_force.atomUpdates() << '\n';
        std::cout << "finite_state: " << (finite_state ? "yes" : "no") << '\n';

        if (!std::filesystem::exists(runtime_path))
        {
            throw std::runtime_error("the LAMMPS runtime library was not staged next to the executable");
        }
        if (external_force.callbackCalls() <= 0 || external_force.atomUpdates() <= 0)
        {
            throw std::runtime_error("fix external callback did not update the 3D DEM particles");
        }
        if (!finite_state)
        {
            throw std::runtime_error("non-finite value detected in DEM state or force history");
        }
        if (lammps_mass_relative_error > kMassRelativeTolerance)
        {
            throw std::runtime_error("LAMMPS particle mass does not match the 3D spherical grain mass");
        }
        if (max_center_error > kCenterErrorTolerance)
        {
            throw std::runtime_error("SPHinXsys DEM proxy center drift exceeded tolerance");
        }
        if (right_front_displacement < kMinimumRightFrontDisplacement)
        {
            throw std::runtime_error("3D granular column did not move to the right enough to indicate collapse");
        }
        if (max_applied_hydro_force_norm < kMinimumHydroForceNorm)
        {
            throw std::runtime_error("3D hydrodynamic force feedback remained effectively zero");
        }
        if (!std::filesystem::exists(motion_csv_path) || !std::filesystem::exists(force_csv_path) ||
            !std::filesystem::exists(latest_dem_spheres_vtp) || !std::filesystem::exists(latest_dem_forces_vtp) ||
            !std::filesystem::exists(water_body_pvd.path()) || !std::filesystem::exists(wall_boundary_pvd.path()) ||
            !std::filesystem::exists(grain_proxy_pvd.path()) || !std::filesystem::exists(dem_spheres_pvd.path()) ||
            !std::filesystem::exists(dem_forces_pvd.path()) ||
            !output_file_exists_with_prefix(output_path, "DEM_Spheres_ite_") ||
            !output_file_exists_with_prefix(output_path, "DEM_Forces_ite_"))
        {
            throw std::runtime_error("one or more 3D granular collapse output files were not written");
        }

        std::cout << "status: PASS\n";
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << "ERROR: " << error.what() << '\n';
        return 1;
    }
}
