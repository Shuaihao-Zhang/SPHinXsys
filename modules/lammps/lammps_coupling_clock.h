#pragma once

#include "lammps_instance.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace SPH
{
namespace lammps
{
struct CouplingStepPlan
{
    double requested_dt = 0.0;
    double coupled_dt = 0.0;
    double dem_dt = 0.0;
    int dem_steps = 0;
    bool shortened_step = false;
};

inline CouplingStepPlan makeCouplingStepPlan(double acoustic_limit, double nominal_dem_dt)
{
    if (!std::isfinite(acoustic_limit) || acoustic_limit <= 0.0)
    {
        throw std::invalid_argument("the SPH acoustic-step limit must be finite and positive");
    }
    if (!std::isfinite(nominal_dem_dt) || nominal_dem_dt <= 0.0)
    {
        throw std::invalid_argument("the nominal DEM timestep must be finite and positive");
    }

    CouplingStepPlan plan;
    plan.requested_dt = acoustic_limit;

    const double scale = std::max({1.0, std::abs(acoustic_limit), std::abs(nominal_dem_dt)});
    const double tolerance = 64.0 * std::numeric_limits<double>::epsilon() * scale;
    if (acoustic_limit + tolerance < nominal_dem_dt)
    {
        plan.coupled_dt = acoustic_limit;
        plan.dem_dt = acoustic_limit;
        plan.dem_steps = 1;
        plan.shortened_step = true;
        return plan;
    }

    const double ratio = acoustic_limit / nominal_dem_dt;
    const double ratio_tolerance = 64.0 * std::numeric_limits<double>::epsilon() *
                                   std::max(1.0, std::abs(ratio));
    int steps = static_cast<int>(std::floor(ratio + ratio_tolerance));
    steps = std::max(1, steps);
    double coupled_dt = static_cast<double>(steps) * nominal_dem_dt;
    if (coupled_dt > acoustic_limit + tolerance)
    {
        --steps;
        coupled_dt = static_cast<double>(steps) * nominal_dem_dt;
    }
    if (steps <= 0)
    {
        plan.coupled_dt = acoustic_limit;
        plan.dem_dt = acoustic_limit;
        plan.dem_steps = 1;
        plan.shortened_step = true;
        return plan;
    }

    plan.coupled_dt = coupled_dt;
    plan.dem_dt = nominal_dem_dt;
    plan.dem_steps = steps;
    return plan;
}

struct CouplingAdvanceResult
{
    double requested_dt = 0.0;
    double coupled_dt = 0.0;
    double dem_dt = 0.0;
    int dem_steps = 0;
    bool shortened_step = false;
    double expected_time = 0.0;
    double lammps_time = 0.0;
    double driver_time = 0.0;
    double driver_lammps_offset = 0.0;
    double synchronization_error = 0.0;
};

class LammpsTimeIntegrator
{
  public:
    explicit LammpsTimeIntegrator(LammpsInstance &lammps, double nominal_dem_dt,
                                  double absolute_tolerance = 1.0e-12,
                                  double relative_tolerance = 1.0e-12)
        : lammps_(lammps), nominal_dem_dt_(nominal_dem_dt),
          expected_time_(lammps.accumulatedTime()),
          absolute_tolerance_(absolute_tolerance), relative_tolerance_(relative_tolerance)
    {
        if (!std::isfinite(nominal_dem_dt_) || nominal_dem_dt_ <= 0.0)
        {
            throw std::invalid_argument("the nominal DEM timestep must be finite and positive");
        }
    }

    CouplingAdvanceResult advance(double acoustic_limit)
    {
        return advance(makeCouplingStepPlan(acoustic_limit, nominal_dem_dt_));
    }

    CouplingAdvanceResult advance(const CouplingStepPlan &plan)
    {
        const double driver_time_before = driver_clock_initialized_
                                              ? expected_driver_time_
                                              : expected_time_;
        return advance(plan, driver_time_before);
    }

    CouplingAdvanceResult advance(const CouplingStepPlan &plan, double driver_time_before)
    {
        if (plan.dem_steps <= 0 || plan.dem_dt <= 0.0 || plan.coupled_dt <= 0.0)
        {
            throw std::invalid_argument("the coupling-step plan is invalid");
        }
        if (!std::isfinite(driver_time_before))
        {
            throw std::invalid_argument("the SPHinXsys driver time must be finite");
        }

        if (!driver_clock_initialized_)
        {
            driver_lammps_offset_ = driver_time_before - expected_time_;
            expected_driver_time_ = driver_time_before;
            driver_clock_initialized_ = true;
        }
        const double driver_start_error = std::abs(driver_time_before - expected_driver_time_);
        const double driver_start_tolerance = clockTolerance(expected_driver_time_);
        if (driver_start_error > driver_start_tolerance)
        {
            std::ostringstream message;
            message << std::setprecision(17)
                    << "SPHinXsys driver clock discontinuity: expected " << expected_driver_time_
                    << ", received " << driver_time_before
                    << ", error " << driver_start_error
                    << ", tolerance " << driver_start_tolerance;
            throw std::runtime_error(message.str());
        }

        setTimestep(plan.dem_dt);
        lammps_.command("run " + std::to_string(plan.dem_steps) + " post no",
                        "LAMMPS aligned coupling run");
        if (plan.shortened_step)
        {
            setTimestep(nominal_dem_dt_);
        }

        expected_time_ += plan.coupled_dt;
        expected_driver_time_ += plan.coupled_dt;
        const double lammps_time = lammps_.accumulatedTime();
        const double lammps_internal_error = std::abs(lammps_time - expected_time_);
        const double aligned_driver_time = lammps_time + driver_lammps_offset_;
        const double driver_lammps_error = std::abs(aligned_driver_time - expected_driver_time_);
        const double synchronization_error =
            std::max({driver_start_error, lammps_internal_error, driver_lammps_error});
        const double tolerance = std::max(clockTolerance(expected_time_),
                                          clockTolerance(expected_driver_time_));
        if (synchronization_error > tolerance)
        {
            std::ostringstream message;
            message << std::setprecision(17)
                    << "LAMMPS/SPHinXsys clock mismatch: expected LAMMPS time " << expected_time_
                    << ", LAMMPS atime " << lammps_time
                    << ", expected driver time " << expected_driver_time_
                    << ", LAMMPS-to-driver offset " << driver_lammps_offset_
                    << ", error " << synchronization_error
                    << ", tolerance " << tolerance;
            throw std::runtime_error(message.str());
        }

        return CouplingAdvanceResult{
            plan.requested_dt, plan.coupled_dt, plan.dem_dt, plan.dem_steps,
            plan.shortened_step, expected_time_, lammps_time, expected_driver_time_,
            driver_lammps_offset_, synchronization_error};
    }

    double expectedTime() const { return expected_time_; }
    double nominalDemDt() const { return nominal_dem_dt_; }
    double expectedDriverTime() const { return expected_driver_time_; }
    double driverLammpsOffset() const { return driver_lammps_offset_; }

  private:
    void setTimestep(double timestep)
    {
        std::ostringstream command;
        command << std::setprecision(17) << "timestep " << timestep;
        lammps_.command(command.str(), "LAMMPS timestep update");
    }

    double clockTolerance(double time) const
    {
        return absolute_tolerance_ +
               relative_tolerance_ * std::max(1.0, std::abs(time));
    }

    LammpsInstance &lammps_;
    double nominal_dem_dt_;
    double expected_time_;
    double expected_driver_time_ = 0.0;
    double driver_lammps_offset_ = 0.0;
    bool driver_clock_initialized_ = false;
    double absolute_tolerance_;
    double relative_tolerance_;
};
} // namespace lammps
} // namespace SPH
