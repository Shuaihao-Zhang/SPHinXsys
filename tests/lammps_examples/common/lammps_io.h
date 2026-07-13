#pragma once

#include "sphinxsys.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace SPH
{
namespace lammps_examples
{
struct VtpVectorField2d
{
    std::string name;
    Vec2d value;
};

struct VtpVectorField3d
{
    std::string name;
    Vec3d value;
};

struct VtpScalarField
{
    std::string name;
    Real value = 0.0;
};

inline std::string format_iteration_index(int iteration)
{
    std::ostringstream stream;
    stream << std::setw(10) << std::setfill('0') << iteration;
    return stream.str();
}

inline std::filesystem::path lammps_example_output_path(const std::string &prefix, int iteration)
{
    std::filesystem::path folder(IO::getEnvironment().OutputFolder());
    std::filesystem::create_directories(folder);
    return folder / (prefix + "_ite_" + format_iteration_index(iteration) + ".vtp");
}

inline std::filesystem::path lammps_example_pvd_path(const std::string &prefix)
{
    std::filesystem::path folder(IO::getEnvironment().OutputFolder());
    std::filesystem::create_directories(folder);
    return folder / (prefix + ".pvd");
}

class VtpPvdWriter
{
  public:
    explicit VtpPvdWriter(const std::string &prefix)
        : pvd_path_(lammps_example_pvd_path(prefix)) {}

    const std::filesystem::path &path() const { return pvd_path_; }

    void add(Real time, const std::filesystem::path &vtp_path)
    {
        entries_.push_back({time, vtp_path.filename().string()});
        write();
    }

  private:
    struct Entry
    {
        Real time = 0.0;
        std::string file_name;
    };

    void write() const
    {
        std::ofstream file(pvd_path_);
        if (!file)
        {
            throw std::runtime_error("could not open PVD time-series file for writing");
        }

        file << std::setprecision(17);
        file << "<?xml version=\"1.0\"?>\n"
             << "<VTKFile type=\"Collection\" version=\"0.1\" byte_order=\"LittleEndian\">\n"
             << "  <Collection>\n";
        for (const Entry &entry : entries_)
        {
            std::string file_name = entry.file_name;
            std::replace(file_name.begin(), file_name.end(), '\\', '/');
            file << "    <DataSet timestep=\"" << entry.time
                 << "\" group=\"\" part=\"0\" file=\"" << file_name << "\"/>\n";
        }
        file << "  </Collection>\n"
             << "</VTKFile>\n";
    }

    std::filesystem::path pvd_path_;
    std::vector<Entry> entries_;
};

inline void write_vec3(std::ofstream &file, const Vec2d &value)
{
    file << value[0] << ' ' << value[1] << " 0";
}

inline void write_vec3(std::ofstream &file, const Vec3d &value)
{
    file << value[0] << ' ' << value[1] << ' ' << value[2];
}

inline void write_vtp_time_value(std::ofstream &file, Real time)
{
    file << "    <FieldData>\n"
         << "      <DataArray type=\"Float64\" Name=\"TimeValue\" NumberOfTuples=\"1\" format=\"ascii\">\n"
         << "        " << time << "\n"
         << "      </DataArray>\n"
         << "    </FieldData>\n";
}

inline std::filesystem::path write_dem_cylinder_to_vtp(int iteration,
                                                       Real time,
                                                       const Vec2d &center,
                                                       Real radius,
                                                       const std::string &prefix = "DEM_Cylinder",
                                                       int circle_segments = 128)
{
    const std::filesystem::path file_path = lammps_example_output_path(prefix, iteration);
    std::ofstream file(file_path);
    if (!file)
    {
        throw std::runtime_error("could not open DEM cylinder VTP for writing");
    }

    file << std::setprecision(17);
    file << "<?xml version=\"1.0\"?>\n"
         << "<VTKFile type=\"PolyData\" version=\"0.1\" byte_order=\"LittleEndian\">\n"
         << "  <PolyData>\n";
    write_vtp_time_value(file, time);
    file << "    <Piece NumberOfPoints=\"" << circle_segments + 1
         << "\" NumberOfVerts=\"1\" NumberOfLines=\"1\" NumberOfPolys=\"1\">\n"
         << "      <PointData Scalars=\"Radius\">\n"
         << "        <DataArray type=\"Float64\" Name=\"Radius\" NumberOfComponents=\"1\" format=\"ascii\">\n"
         << "          ";
    for (int i = 0; i <= circle_segments; ++i)
    {
        file << radius << ' ';
    }
    file << "\n"
         << "        </DataArray>\n"
         << "        <DataArray type=\"Float64\" Name=\"DEMCenter\" NumberOfComponents=\"3\" format=\"ascii\">\n"
         << "          ";
    for (int i = 0; i <= circle_segments; ++i)
    {
        write_vec3(file, center);
        file << ' ';
    }
    file << "\n"
         << "        </DataArray>\n"
         << "      </PointData>\n"
         << "      <Points>\n"
         << "        <DataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"ascii\">\n"
         << "          ";
    write_vec3(file, center);
    file << '\n';
    for (int i = 0; i < circle_segments; ++i)
    {
        const Real angle = 2.0 * Pi * static_cast<Real>(i) / static_cast<Real>(circle_segments);
        const Vec2d point(center[0] + radius * std::cos(angle),
                          center[1] + radius * std::sin(angle));
        file << "          ";
        write_vec3(file, point);
        file << '\n';
    }
    file << "        </DataArray>\n"
         << "      </Points>\n"
         << "      <Verts>\n"
         << "        <DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">0</DataArray>\n"
         << "        <DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">1</DataArray>\n"
         << "      </Verts>\n"
         << "      <Lines>\n"
         << "        <DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">\n"
         << "          ";
    for (int i = 1; i <= circle_segments; ++i)
    {
        file << i << ' ';
    }
    file << "1\n"
         << "        </DataArray>\n"
         << "        <DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">"
         << circle_segments + 1 << "</DataArray>\n"
         << "      </Lines>\n"
         << "      <Polys>\n"
         << "        <DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">\n"
         << "          ";
    for (int i = 1; i <= circle_segments; ++i)
    {
        file << i << ' ';
    }
    file << "\n"
         << "        </DataArray>\n"
         << "        <DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">"
         << circle_segments << "</DataArray>\n"
         << "      </Polys>\n"
         << "    </Piece>\n"
         << "  </PolyData>\n"
         << "</VTKFile>\n";

    return file_path;
}

inline std::filesystem::path write_dem_point_fields_to_vtp(
    int iteration,
    Real time,
    const std::string &prefix,
    const Vec2d &point,
    const std::vector<VtpVectorField2d> &vector_fields,
    const std::vector<VtpScalarField> &scalar_fields,
    const std::string &active_vector_name = "")
{
    const std::filesystem::path file_path = lammps_example_output_path(prefix, iteration);
    std::ofstream file(file_path);
    if (!file)
    {
        throw std::runtime_error("could not open DEM point-field VTP for writing");
    }

    file << std::setprecision(17);
    file << "<?xml version=\"1.0\"?>\n"
         << "<VTKFile type=\"PolyData\" version=\"0.1\" byte_order=\"LittleEndian\">\n"
         << "  <PolyData>\n";
    write_vtp_time_value(file, time);
    file << "    <Piece NumberOfPoints=\"1\" NumberOfVerts=\"1\" NumberOfLines=\"0\" NumberOfPolys=\"0\">\n"
         << "      <PointData";
    if (!active_vector_name.empty())
    {
        file << " Vectors=\"" << active_vector_name << "\"";
    }
    file << ">\n";
    for (const VtpVectorField2d &field : vector_fields)
    {
        file << "        <DataArray type=\"Float64\" Name=\"" << field.name
             << "\" NumberOfComponents=\"3\" format=\"ascii\">";
        write_vec3(file, field.value);
        file << "</DataArray>\n";
    }
    for (const VtpScalarField &field : scalar_fields)
    {
        file << "        <DataArray type=\"Float64\" Name=\"" << field.name
             << "\" NumberOfComponents=\"1\" format=\"ascii\">"
             << field.value << "</DataArray>\n";
    }
    file << "      </PointData>\n"
         << "      <Points>\n"
         << "        <DataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"ascii\">";
    write_vec3(file, point);
    file << "</DataArray>\n"
         << "      </Points>\n"
         << "      <Verts>\n"
         << "        <DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">0</DataArray>\n"
         << "        <DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">1</DataArray>\n"
         << "      </Verts>\n"
         << "    </Piece>\n"
         << "  </PolyData>\n"
         << "</VTKFile>\n";

    return file_path;
}

inline std::filesystem::path write_dem_point_fields_to_vtp(
    int iteration,
    Real time,
    const std::string &prefix,
    const Vec3d &point,
    const std::vector<VtpVectorField3d> &vector_fields,
    const std::vector<VtpScalarField> &scalar_fields,
    const std::string &active_vector_name = "")
{
    const std::filesystem::path file_path = lammps_example_output_path(prefix, iteration);
    std::ofstream file(file_path);
    if (!file)
    {
        throw std::runtime_error("could not open 3D DEM point-field VTP for writing");
    }

    file << std::setprecision(17);
    file << "<?xml version=\"1.0\"?>\n"
         << "<VTKFile type=\"PolyData\" version=\"0.1\" byte_order=\"LittleEndian\">\n"
         << "  <PolyData>\n";
    write_vtp_time_value(file, time);
    file << "    <Piece NumberOfPoints=\"1\" NumberOfVerts=\"1\" NumberOfLines=\"0\" NumberOfPolys=\"0\">\n"
         << "      <PointData";
    if (!active_vector_name.empty())
    {
        file << " Vectors=\"" << active_vector_name << "\"";
    }
    file << ">\n";
    for (const VtpVectorField3d &field : vector_fields)
    {
        file << "        <DataArray type=\"Float64\" Name=\"" << field.name
             << "\" NumberOfComponents=\"3\" format=\"ascii\">";
        write_vec3(file, field.value);
        file << "</DataArray>\n";
    }
    for (const VtpScalarField &field : scalar_fields)
    {
        file << "        <DataArray type=\"Float64\" Name=\"" << field.name
             << "\" NumberOfComponents=\"1\" format=\"ascii\">"
             << field.value << "</DataArray>\n";
    }
    file << "      </PointData>\n"
         << "      <Points>\n"
         << "        <DataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"ascii\">";
    write_vec3(file, point);
    file << "</DataArray>\n"
         << "      </Points>\n"
         << "      <Verts>\n"
         << "        <DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">0</DataArray>\n"
         << "        <DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">1</DataArray>\n"
         << "      </Verts>\n"
         << "    </Piece>\n"
         << "  </PolyData>\n"
         << "</VTKFile>\n";

    return file_path;
}

struct VtpVectorPointField2d
{
    std::string name;
    std::vector<Vec2d> values;
};

struct VtpScalarPointField
{
    std::string name;
    std::vector<Real> values;
};

inline std::filesystem::path write_dem_discs_to_vtp(int iteration,
                                                    Real time,
                                                    const std::string &prefix,
                                                    const std::vector<Vec2d> &centers,
                                                    Real radius,
                                                    int circle_segments = 48)
{
    const std::filesystem::path file_path = lammps_example_output_path(prefix, iteration);
    std::ofstream file(file_path);
    if (!file)
    {
        throw std::runtime_error("could not open DEM discs VTP for writing");
    }

    const int number_of_discs = static_cast<int>(centers.size());
    const int points_per_disc = circle_segments + 1;
    const int total_points = number_of_discs * points_per_disc;

    file << std::setprecision(17);
    file << "<?xml version=\"1.0\"?>\n"
         << "<VTKFile type=\"PolyData\" version=\"0.1\" byte_order=\"LittleEndian\">\n"
         << "  <PolyData>\n";
    write_vtp_time_value(file, time);
    file << "    <Piece NumberOfPoints=\"" << total_points
         << "\" NumberOfVerts=\"" << number_of_discs
         << "\" NumberOfLines=\"" << number_of_discs
         << "\" NumberOfPolys=\"" << number_of_discs << "\">\n"
         << "      <PointData Scalars=\"ParticleId\">\n"
         << "        <DataArray type=\"Int32\" Name=\"ParticleId\" NumberOfComponents=\"1\" format=\"ascii\">\n"
         << "          ";
    for (int disc = 0; disc < number_of_discs; ++disc)
    {
        for (int i = 0; i < points_per_disc; ++i)
        {
            file << disc + 1 << ' ';
        }
    }
    file << "\n"
         << "        </DataArray>\n"
         << "        <DataArray type=\"Float64\" Name=\"Radius\" NumberOfComponents=\"1\" format=\"ascii\">\n"
         << "          ";
    for (int disc = 0; disc < number_of_discs; ++disc)
    {
        for (int i = 0; i < points_per_disc; ++i)
        {
            file << radius << ' ';
        }
    }
    file << "\n"
         << "        </DataArray>\n"
         << "      </PointData>\n"
         << "      <Points>\n"
         << "        <DataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"ascii\">\n";
    for (const Vec2d &center : centers)
    {
        file << "          ";
        write_vec3(file, center);
        file << '\n';
        for (int i = 0; i < circle_segments; ++i)
        {
            const Real angle = 2.0 * Pi * static_cast<Real>(i) / static_cast<Real>(circle_segments);
            const Vec2d point(center[0] + radius * std::cos(angle),
                              center[1] + radius * std::sin(angle));
            file << "          ";
            write_vec3(file, point);
            file << '\n';
        }
    }
    file << "        </DataArray>\n"
         << "      </Points>\n"
         << "      <Verts>\n"
         << "        <DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">\n"
         << "          ";
    for (int disc = 0; disc < number_of_discs; ++disc)
    {
        file << disc * points_per_disc << ' ';
    }
    file << "\n"
         << "        </DataArray>\n"
         << "        <DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">\n"
         << "          ";
    for (int disc = 0; disc < number_of_discs; ++disc)
    {
        file << disc + 1 << ' ';
    }
    file << "\n"
         << "        </DataArray>\n"
         << "      </Verts>\n"
         << "      <Lines>\n"
         << "        <DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">\n";
    for (int disc = 0; disc < number_of_discs; ++disc)
    {
        const int base = disc * points_per_disc;
        file << "          ";
        for (int i = 1; i <= circle_segments; ++i)
        {
            file << base + i << ' ';
        }
        file << base + 1 << '\n';
    }
    file << "        </DataArray>\n"
         << "        <DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">\n"
         << "          ";
    for (int disc = 0; disc < number_of_discs; ++disc)
    {
        file << (disc + 1) * (circle_segments + 1) << ' ';
    }
    file << "\n"
         << "        </DataArray>\n"
         << "      </Lines>\n"
         << "      <Polys>\n"
         << "        <DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">\n";
    for (int disc = 0; disc < number_of_discs; ++disc)
    {
        const int base = disc * points_per_disc;
        file << "          ";
        for (int i = 1; i <= circle_segments; ++i)
        {
            file << base + i << ' ';
        }
        file << '\n';
    }
    file << "        </DataArray>\n"
         << "        <DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">\n"
         << "          ";
    for (int disc = 0; disc < number_of_discs; ++disc)
    {
        file << (disc + 1) * circle_segments << ' ';
    }
    file << "\n"
         << "        </DataArray>\n"
         << "      </Polys>\n"
         << "    </Piece>\n"
         << "  </PolyData>\n"
         << "</VTKFile>\n";

    return file_path;
}

inline std::filesystem::path write_dem_points_fields_to_vtp(
    int iteration,
    Real time,
    const std::string &prefix,
    const std::vector<Vec2d> &points,
    const std::vector<VtpVectorPointField2d> &vector_fields,
    const std::vector<VtpScalarPointField> &scalar_fields,
    const std::string &active_vector_name = "")
{
    const std::filesystem::path file_path = lammps_example_output_path(prefix, iteration);
    std::ofstream file(file_path);
    if (!file)
    {
        throw std::runtime_error("could not open DEM multi-point VTP for writing");
    }

    const int number_of_points = static_cast<int>(points.size());
    auto validate_size = [number_of_points](const auto &field)
    {
        if (static_cast<int>(field.values.size()) != number_of_points)
        {
            throw std::runtime_error("DEM VTP point field size does not match point count");
        }
    };
    for (const auto &field : vector_fields)
    {
        validate_size(field);
    }
    for (const auto &field : scalar_fields)
    {
        validate_size(field);
    }

    file << std::setprecision(17);
    file << "<?xml version=\"1.0\"?>\n"
         << "<VTKFile type=\"PolyData\" version=\"0.1\" byte_order=\"LittleEndian\">\n"
         << "  <PolyData>\n";
    write_vtp_time_value(file, time);
    file << "    <Piece NumberOfPoints=\"" << number_of_points
         << "\" NumberOfVerts=\"" << number_of_points
         << "\" NumberOfLines=\"0\" NumberOfPolys=\"0\">\n"
         << "      <PointData";
    if (!active_vector_name.empty())
    {
        file << " Vectors=\"" << active_vector_name << "\"";
    }
    file << ">\n";
    for (const VtpVectorPointField2d &field : vector_fields)
    {
        file << "        <DataArray type=\"Float64\" Name=\"" << field.name
             << "\" NumberOfComponents=\"3\" format=\"ascii\">\n"
             << "          ";
        for (const Vec2d &value : field.values)
        {
            write_vec3(file, value);
            file << ' ';
        }
        file << "\n"
             << "        </DataArray>\n";
    }
    for (const VtpScalarPointField &field : scalar_fields)
    {
        file << "        <DataArray type=\"Float64\" Name=\"" << field.name
             << "\" NumberOfComponents=\"1\" format=\"ascii\">\n"
             << "          ";
        for (Real value : field.values)
        {
            file << value << ' ';
        }
        file << "\n"
             << "        </DataArray>\n";
    }
    file << "      </PointData>\n"
         << "      <Points>\n"
         << "        <DataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"ascii\">\n"
         << "          ";
    for (const Vec2d &point : points)
    {
        write_vec3(file, point);
        file << ' ';
    }
    file << "\n"
         << "        </DataArray>\n"
         << "      </Points>\n"
         << "      <Verts>\n"
         << "        <DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">\n"
         << "          ";
    for (int i = 0; i < number_of_points; ++i)
    {
        file << i << ' ';
    }
    file << "\n"
         << "        </DataArray>\n"
         << "        <DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">\n"
         << "          ";
    for (int i = 0; i < number_of_points; ++i)
    {
        file << i + 1 << ' ';
    }
    file << "\n"
         << "        </DataArray>\n"
         << "      </Verts>\n"
         << "    </Piece>\n"
         << "  </PolyData>\n"
         << "</VTKFile>\n";

    return file_path;
}
} // namespace lammps_examples
} // namespace SPH
