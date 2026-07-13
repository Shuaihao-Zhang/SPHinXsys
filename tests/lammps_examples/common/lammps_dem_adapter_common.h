#pragma once

#include "lammps_coupling_clock.h"
#include "lammps_external_force.h"
#include "lammps_kinematics.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace SPH
{
namespace lammps_examples
{
using CouplingAdvanceResult = SPH::lammps::CouplingAdvanceResult;
using CouplingStepPlan = SPH::lammps::CouplingStepPlan;
using ExternalForceBuffer = SPH::lammps::ExternalForceBuffer;
using LammpsTimeIntegrator = SPH::lammps::LammpsTimeIntegrator;
using LammpsTagInt = SPH::lammps::LammpsTagInt;
using ParticleForce = SPH::lammps::ParticleForce;
using SPH::lammps::makeCouplingStepPlan;
using SPH::lammps::hydrodynamicTorque2d;
using SPH::lammps::hydrodynamicTorque3d;
using SPH::lammps::rigidSurfaceVelocity2d;
using SPH::lammps::rigidSurfaceVelocity3d;

inline void validate_consecutive_atom_ids(const LammpsInstance &lammps, int expected_count)
{
    const int actual_count = lammps.atomCount();
    if (actual_count != expected_count)
    {
        throw std::runtime_error("LAMMPS atom-count mismatch: expected " +
                                 std::to_string(expected_count) + ", got " +
                                 std::to_string(actual_count));
    }

    const int local_count = lammps.localAtomCount();
    if (local_count != expected_count)
    {
        throw std::runtime_error(
            "the current non-MPI state-exchange path requires every atom to be locally owned");
    }

    auto *ids = static_cast<LammpsTagInt *>(lammps_extract_atom(lammps.get(), "id"));
    lammps.throwIfError("extract local atom IDs");
    if (ids == nullptr)
    {
        throw std::runtime_error("LAMMPS returned a null local atom-ID view");
    }

    std::vector<unsigned char> seen(static_cast<size_t>(expected_count), 0);
    for (int i = 0; i != local_count; ++i)
    {
        const int64_t id = static_cast<int64_t>(ids[i]);
        if (id < 1 || id > expected_count || seen[static_cast<size_t>(id - 1)] != 0)
        {
            throw std::runtime_error(
                "the current serial gather path requires consecutive LAMMPS atom IDs 1..N");
        }
        seen[static_cast<size_t>(id - 1)] = 1;
    }
}

inline void extract_atom_vector3_by_consecutive_id(const LammpsInstance &lammps,
                                                    const std::string &property,
                                                    int expected_count,
                                                    double *values_by_id)
{
    if (values_by_id == nullptr)
    {
        throw std::invalid_argument("the destination atom-property buffer must not be null");
    }

    const int local_count = lammps.localAtomCount();
    if (local_count != expected_count)
    {
        throw std::runtime_error(
            "the current non-MPI state-exchange path requires every atom to be locally owned");
    }

    auto *ids = static_cast<LammpsTagInt *>(lammps_extract_atom(lammps.get(), "id"));
    auto **values = static_cast<double **>(lammps_extract_atom(lammps.get(), property.c_str()));
    lammps.throwIfError("extract local atom property " + property);
    if (ids == nullptr || values == nullptr)
    {
        throw std::runtime_error("LAMMPS returned a null local atom-property view for " + property);
    }

    std::fill(values_by_id, values_by_id + 3 * expected_count, 0.0);
    for (int i = 0; i != local_count; ++i)
    {
        const int64_t id = static_cast<int64_t>(ids[i]);
        if (id < 1 || id > expected_count)
        {
            throw std::runtime_error("LAMMPS atom ID is outside the validated consecutive range");
        }
        const size_t offset = 3 * static_cast<size_t>(id - 1);
        values_by_id[offset] = values[i][0];
        values_by_id[offset + 1] = values[i][1];
        values_by_id[offset + 2] = values[i][2];
    }
}
} // namespace lammps_examples
} // namespace SPH
