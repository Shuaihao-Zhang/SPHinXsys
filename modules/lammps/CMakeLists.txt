if(NOT SPHINXSYS_WITH_LAMMPS)
    return()
endif()

if(NOT TARGET LAMMPS::lammps)
    message(FATAL_ERROR "SPHINXSYS_WITH_LAMMPS is ON, but LAMMPS::lammps is unavailable")
endif()

add_library(sphinxsys_lammps STATIC lammps_module.cpp)
add_library(SPHinXsys::sphinxsys_lammps ALIAS sphinxsys_lammps)

target_sources(sphinxsys_lammps PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/lammps_instance.h
    ${CMAKE_CURRENT_SOURCE_DIR}/lammps_coupling_clock.h
    ${CMAKE_CURRENT_SOURCE_DIR}/lammps_external_force.h
    ${CMAKE_CURRENT_SOURCE_DIR}/lammps_kinematics.h)
target_include_directories(sphinxsys_lammps PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>)
target_compile_features(sphinxsys_lammps PUBLIC cxx_std_17)
target_link_libraries(sphinxsys_lammps PUBLIC LAMMPS::lammps)
