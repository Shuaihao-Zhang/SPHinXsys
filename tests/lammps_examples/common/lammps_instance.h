#pragma once

#include "library.h"

#include <sstream>
#include <stdexcept>
#include <string>

namespace SPH
{
namespace lammps_examples
{
class LammpsInstance
{
  public:
    LammpsInstance()
    {
        const char *args[] = {"liblammps", "-log", "none", "-screen", "none", "-nocite", nullptr};
        auto **argv = const_cast<char **>(args);
        const int argc = static_cast<int>((sizeof(args) / sizeof(char *)) - 1);

        handle_ = lammps_open_no_mpi(argc, argv, nullptr);
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

    void command(const std::string &cmd, const std::string &context) const
    {
        lammps_command(handle_, cmd.c_str());
        throw_if_error(context);
    }

    void commands_string(const std::string &cmds, const std::string &context) const
    {
        lammps_commands_string(handle_, cmds.c_str());
        throw_if_error(context);
    }

    void throw_if_error(const std::string &context) const
    {
        if (lammps_has_error(handle_) == 0)
        {
            return;
        }

        char buffer[4096] = {};
        lammps_get_last_error_message(handle_, buffer, static_cast<int>(sizeof(buffer)));
        std::ostringstream msg;
        msg << context << " failed: " << buffer;
        throw std::runtime_error(msg.str());
    }

  private:
    void *handle_ = nullptr;
};
} // namespace lammps_examples
} // namespace SPH
