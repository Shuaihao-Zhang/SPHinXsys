if(NOT SPHINXSYS_WITH_LAMMPS)
    message(STATUS "Skipping LAMMPS examples because SPHINXSYS_WITH_LAMMPS is OFF")
    return()
endif()

get_filename_component(SPH_DEM_ROOT "${SPHINXSYS_PROJECT_DIR}/.." ABSOLUTE)

set(SPHINXSYS_LAMMPS_INCLUDE_DIR
    "${SPH_DEM_ROOT}/lammps_build/includes/lammps"
    CACHE PATH "Path to LAMMPS generated C library headers")
set(SPHINXSYS_LAMMPS_LIBRARY
    "${SPH_DEM_ROOT}/lammps_build/Release/lammps.lib"
    CACHE FILEPATH "Path to LAMMPS import library")
set(SPHINXSYS_LAMMPS_DLL
    "${SPH_DEM_ROOT}/lammps_build/Release/liblammps.dll"
    CACHE FILEPATH "Path to LAMMPS runtime DLL")

if(NOT EXISTS "${SPHINXSYS_LAMMPS_INCLUDE_DIR}/library.h")
    message(FATAL_ERROR "LAMMPS library.h not found: ${SPHINXSYS_LAMMPS_INCLUDE_DIR}/library.h")
endif()

if(NOT EXISTS "${SPHINXSYS_LAMMPS_LIBRARY}")
    message(FATAL_ERROR "LAMMPS import library not found: ${SPHINXSYS_LAMMPS_LIBRARY}")
endif()

if(NOT EXISTS "${SPHINXSYS_LAMMPS_DLL}")
    message(FATAL_ERROR "LAMMPS runtime DLL not found: ${SPHINXSYS_LAMMPS_DLL}")
endif()

add_library(LAMMPS::lammps SHARED IMPORTED)
set_target_properties(LAMMPS::lammps PROPERTIES
    IMPORTED_IMPLIB "${SPHINXSYS_LAMMPS_LIBRARY}"
    IMPORTED_LOCATION "${SPHINXSYS_LAMMPS_DLL}"
    INTERFACE_INCLUDE_DIRECTORIES "${SPHINXSYS_LAMMPS_INCLUDE_DIR}")

add_subdirectory(common)

SUBDIRLIST(SUBDIRS ${CMAKE_CURRENT_SOURCE_DIR})

foreach(subdir ${SUBDIRS})
    if(NOT "${subdir}" STREQUAL "common" AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${subdir}/CMakeLists.txt)
	    add_subdirectory(${subdir})
    endif()
endforeach()
