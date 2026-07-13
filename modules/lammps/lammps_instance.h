#pragma once

#include <lammps/library.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

namespace SPH
{
namespace lammps
{
#if defined(LAMMPS_BIGBIG)
using LammpsTagInt = int64_t;
#else
using LammpsTagInt = int;
#endif

class LammpsInstance
{
  public:
    LammpsInstance()
    {
        std::array<std::string, 6> arguments = {
            "liblammps", "-log", "none", "-screen", "none", "-nocite"};
        std::array<char *, 6> argv{};
        for (size_t i = 0; i != arguments.size(); ++i)
        {
            argv[i] = arguments[i].data();
        }

        handle_ = lammps_open_no_mpi(static_cast<int>(argv.size()), argv.data(), nullptr);
        if (handle_ == nullptr)
        {
            throw std::runtime_error("lammps_open_no_mpi returned null");
        }
    }

    LammpsInstance(const LammpsInstance &) = delete;
    LammpsInstance &operator=(const LammpsInstance &) = delete;

    ~LammpsInstance()
    {
        if (handle_ != nullptr)
        {
            lammps_close(handle_);
        }
    }

    void *get() const { return handle_; }

    int version() const { return lammps_version(handle_); }

    int atomCount() const
    {
        const double atom_count = lammps_get_natoms(handle_);
        throwIfError("lammps_get_natoms");
        if (!std::isfinite(atom_count) || atom_count < 0.0 ||
            atom_count > static_cast<double>(std::numeric_limits<int>::max()) ||
            std::floor(atom_count) != atom_count)
        {
            throw std::runtime_error("LAMMPS returned an invalid atom count");
        }
        return static_cast<int>(atom_count);
    }

    int localAtomCount() const
    {
        auto *local_count = static_cast<int *>(lammps_extract_global(handle_, "nlocal"));
        throwIfError("extract global nlocal");
        if (local_count == nullptr || *local_count < 0)
        {
            throw std::runtime_error("LAMMPS returned an invalid local atom count");
        }
        return *local_count;
    }

    double accumulatedTime() const
    {
        auto *time = static_cast<double *>(lammps_extract_global(handle_, "atime"));
        throwIfError("extract global atime");
        if (time == nullptr || !std::isfinite(*time))
        {
            throw std::runtime_error("LAMMPS returned an invalid accumulated time");
        }
        return *time;
    }

    void command(const std::string &cmd, const std::string &context) const
    {
        lammps_command(handle_, cmd.c_str());
        throwIfError(context);
    }

    void commandsString(const std::string &commands, const std::string &context) const
    {
        lammps_commands_string(handle_, commands.c_str());
        throwIfError(context);
    }

    void commands_string(const std::string &commands, const std::string &context) const
    {
        commandsString(commands, context);
    }

    void throwIfError(const std::string &context) const
    {
        if (lammps_has_error(handle_) == 0)
        {
            return;
        }

        char buffer[4096] = {};
        lammps_get_last_error_message(handle_, buffer, static_cast<int>(sizeof(buffer)));
        std::ostringstream message;
        message << context << " failed: " << buffer;
        throw std::runtime_error(message.str());
    }

    void throw_if_error(const std::string &context) const
    {
        throwIfError(context);
    }

  private:
    void *handle_ = nullptr;
};
} // namespace lammps
} // namespace SPH
