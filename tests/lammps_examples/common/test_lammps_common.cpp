#include "lammps_coupling_clock.h"
#include "lammps_external_force.h"
#include "lammps_instance.h"
#include "lammps_kinematics.h"

#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
void require(bool condition, const std::string &message)
{
    if (!condition)
    {
        throw std::runtime_error(message);
    }
}

void require_near(double actual, double expected, double tolerance, const std::string &message)
{
    require(std::abs(actual - expected) <= tolerance,
            message + ": actual=" + std::to_string(actual) +
                ", expected=" + std::to_string(expected));
}
} // namespace

int main()
{
    try
    {
        using namespace SPH::lammps;

        const CouplingStepPlan rounded = makeCouplingStepPlan(5.3e-5, 1.0e-5);
        require(rounded.dem_steps == 5, "5.3e-5 must use five nominal DEM steps");
        require_near(rounded.coupled_dt, 5.0e-5, 1.0e-15,
                     "the coupled step must be rounded down to the DEM grid");
        require_near(rounded.dem_dt, 1.0e-5, 1.0e-15,
                     "normal steps must retain the nominal DEM timestep");
        require(!rounded.shortened_step, "normal aligned steps must not be marked shortened");

        const CouplingStepPlan exact = makeCouplingStepPlan(5.0e-5, 1.0e-5);
        require(exact.dem_steps == 5, "an exact multiple must retain all five steps");
        require_near(exact.coupled_dt, 5.0e-5, 1.0e-15,
                     "an exact multiple must not be shortened by floating-point roundoff");

        const CouplingStepPlan short_step = makeCouplingStepPlan(3.0e-6, 1.0e-5);
        require(short_step.dem_steps == 1, "a sub-DEM acoustic limit must use one safety step");
        require_near(short_step.dem_dt, 3.0e-6, 1.0e-15,
                     "the safety DEM step must equal the smaller SPH stability limit");
        require(short_step.shortened_step, "a sub-DEM step must be marked shortened");

        bool rejected_invalid_dt = false;
        try
        {
            (void)makeCouplingStepPlan(0.0, 1.0e-5);
        }
        catch (const std::invalid_argument &)
        {
            rejected_invalid_dt = true;
        }
        require(rejected_invalid_dt, "a non-positive acoustic limit must be rejected");

        ExternalForceBuffer force_buffer;
        force_buffer.setForces({{7, {1.0, 2.0, 3.0}}, {42, {-4.0, 5.0, -6.0}}});
        require(force_buffer.forceForId(7) == std::array<double, 3>{1.0, 2.0, 3.0},
                "force lookup must use the actual LAMMPS atom ID");
        require(force_buffer.forceForId(2) == std::array<double, 3>{0.0, 0.0, 0.0},
                "unknown atom IDs must receive zero external force");

        const std::array<double, 2> velocity_2d =
            rigidSurfaceVelocity2d({1.0, 2.0}, 3.0, {4.0, 5.0});
        require(velocity_2d == std::array<double, 2>{-14.0, 14.0},
                "2D proxy velocity must include omega cross r");

        const std::array<double, 3> velocity_3d =
            rigidSurfaceVelocity3d({1.0, -1.0, 0.0}, {0.0, 0.0, 2.0}, {3.0, 4.0, 0.0});
        require(velocity_3d == std::array<double, 3>{-7.0, 5.0, 0.0},
                "3D proxy velocity must include omega cross r");

        require_near(hydrodynamicTorque2d({2.0, 3.0}, {5.0, 7.0}), -1.0, 1.0e-15,
                     "2D hydrodynamic torque must use r cross force");
        require(hydrodynamicTorque3d({1.0, 0.0, 0.0}, {0.0, 2.0, 0.0}) ==
                    std::array<double, 3>{0.0, 0.0, 2.0},
                "3D hydrodynamic torque must use r cross force");

        LammpsInstance lammps;
        lammps.commandsString(
            "units si\n"
            "atom_style sphere\n"
            "region box block -1 1 -1 1 -1 1 units box\n"
            "create_box 1 box\n"
            "create_atoms 1 single 0 0 0 units box\n"
            "set atom 1 diameter 0.02 density 2500\n"
            "fix integrator all nve/sphere\n"
            "fix ext all external pf/callback 1 1\n"
            "timestep 1.0e-5\n",
            "common runtime test initialization");
        require(lammps.atomCount() == 1, "LAMMPS atom count must be available through the common instance");

        force_buffer.setForces({{1, {1.0, 2.0, 3.0}}});
        force_buffer.registerFix(lammps, "ext");
        LammpsTimeIntegrator integrator(lammps, 1.0e-5);
        const CouplingAdvanceResult advance = integrator.advance(rounded);
        require(advance.dem_steps == 5, "the runtime integrator must execute the planned step count");
        require_near(advance.lammps_time, 5.0e-5, 1.0e-12,
                     "LAMMPS atime must match the aligned coupling duration");
        require(advance.synchronization_error <= 1.0e-12,
                "LAMMPS and coupling clocks must remain synchronized");
        require(force_buffer.callbackCalls() > 0 && force_buffer.atomUpdates() > 0,
                "the runtime callback must apply a mapped force to atom ID 1");

        bool rejected_driver_discontinuity = false;
        try
        {
            (void)integrator.advance(exact, 9.0e-5);
        }
        catch (const std::runtime_error &)
        {
            rejected_driver_discontinuity = true;
        }
        require(rejected_driver_discontinuity,
                "the coupling clock must reject a discontinuous SPHinXsys driver time");

        const CouplingAdvanceResult shortened_runtime =
            integrator.advance(short_step, advance.driver_time);
        require_near(shortened_runtime.driver_time, 5.3e-5, 1.0e-12,
                     "a shortened runtime step must advance the driver and LAMMPS equally");
        const CouplingAdvanceResult restored_runtime =
            integrator.advance(exact, shortened_runtime.driver_time);
        require_near(restored_runtime.lammps_time, 1.03e-4, 1.0e-12,
                     "the nominal DEM timestep must be restored after a shortened step");

        std::cout << "status: PASS\n";
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << "ERROR: " << error.what() << '\n';
        return 1;
    }
}
