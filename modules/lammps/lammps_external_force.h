#pragma once

#include "lammps_instance.h"

#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace SPH
{
namespace lammps
{
struct ParticleForce
{
    int64_t id = 0;
    std::array<double, 3> force{0.0, 0.0, 0.0};
};

class ExternalForceBuffer
{
  public:
    void setForces(const std::vector<ParticleForce> &forces)
    {
        forces_by_id_.clear();
        for (const ParticleForce &particle_force : forces)
        {
            if (particle_force.id <= 0)
            {
                throw std::invalid_argument("LAMMPS atom IDs must be positive");
            }
            const auto inserted = forces_by_id_.emplace(particle_force.id, particle_force.force);
            if (!inserted.second)
            {
                throw std::invalid_argument("duplicate LAMMPS atom ID in external-force buffer");
            }
        }
    }

    std::array<double, 3> forceForId(int64_t id) const
    {
        const auto found = forces_by_id_.find(id);
        return found == forces_by_id_.end()
                   ? std::array<double, 3>{0.0, 0.0, 0.0}
                   : found->second;
    }

    void registerFix(LammpsInstance &lammps, const std::string &fix_id)
    {
        lammps_set_fix_external_callback(lammps.get(), fix_id.c_str(), &ExternalForceBuffer::callback, this);
        lammps.throwIfError("lammps_set_fix_external_callback for " + fix_id);
    }

    int callbackCalls() const { return callback_calls_; }
    int atomUpdates() const { return atom_updates_; }

  private:
    static void callback(void *context, int64_t /*timestep*/, int nlocal,
                         LammpsTagInt *ids, double ** /*positions*/, double **external_force)
    {
        auto *buffer = static_cast<ExternalForceBuffer *>(context);
        ++buffer->callback_calls_;
        for (int i = 0; i != nlocal; ++i)
        {
            const std::array<double, 3> force = buffer->forceForId(static_cast<int64_t>(ids[i]));
            external_force[i][0] = force[0];
            external_force[i][1] = force[1];
            external_force[i][2] = force[2];
            ++buffer->atom_updates_;
        }
    }

    std::unordered_map<int64_t, std::array<double, 3>> forces_by_id_;
    int callback_calls_ = 0;
    int atom_updates_ = 0;
};
} // namespace lammps
} // namespace SPH
