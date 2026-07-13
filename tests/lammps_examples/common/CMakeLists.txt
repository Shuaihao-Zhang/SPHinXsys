add_library(sphinxsys_lammps_examples_common INTERFACE)

target_sources(sphinxsys_lammps_examples_common INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/lammps_io.h
    ${CMAKE_CURRENT_SOURCE_DIR}/lammps_instance.h
    ${CMAKE_CURRENT_SOURCE_DIR}/lammps_dem_adapter_common.h)

target_include_directories(sphinxsys_lammps_examples_common INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR})

target_compile_features(sphinxsys_lammps_examples_common INTERFACE cxx_std_17)
target_link_libraries(sphinxsys_lammps_examples_common INTERFACE LAMMPS::lammps)
