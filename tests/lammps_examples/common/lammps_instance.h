#pragma once

#include "../../../modules/lammps/lammps_instance.h"

#include <filesystem>

namespace SPH
{
namespace lammps_examples
{
using LammpsInstance = SPH::lammps::LammpsInstance;

inline std::filesystem::path lammps_runtime_path()
{
#if defined(SPHINXSYS_LAMMPS_RUNTIME_FILENAME)
    return SPHINXSYS_LAMMPS_RUNTIME_FILENAME;
#else
    return {};
#endif
}
} // namespace lammps_examples
} // namespace SPH
