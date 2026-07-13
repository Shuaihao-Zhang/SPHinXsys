include_guard(GLOBAL)

function(sphinxsys_configure_lammps_dependency)
    if(TARGET LAMMPS::lammps)
        return()
    endif()

    find_package(LAMMPS CONFIG QUIET)
    if(TARGET LAMMPS::lammps)
        message(STATUS "Using LAMMPS package target: LAMMPS::lammps")
        return()
    endif()
    if(TARGET LAMMPS::LAMMPS)
        add_library(LAMMPS::lammps ALIAS LAMMPS::LAMMPS)
        message(STATUS "Using LAMMPS package target: LAMMPS::LAMMPS")
        return()
    endif()

    get_filename_component(_sph_dem_root "${SPHINXSYS_PROJECT_DIR}/.." ABSOLUTE)
    set(SPHINXSYS_LAMMPS_ROOT
        "${_sph_dem_root}/lammps_build"
        CACHE PATH "LAMMPS build or install prefix used by fallback discovery")
    set(SPHINXSYS_LAMMPS_SIZES
        "smallbig"
        CACHE STRING "LAMMPS integer sizes for fallback discovery: smallbig or bigbig")
    set_property(CACHE SPHINXSYS_LAMMPS_SIZES PROPERTY STRINGS smallbig bigbig)

    string(TOLOWER "${SPHINXSYS_LAMMPS_SIZES}" _lammps_sizes)
    if(NOT _lammps_sizes STREQUAL "smallbig" AND NOT _lammps_sizes STREQUAL "bigbig")
        message(FATAL_ERROR
            "SPHINXSYS_LAMMPS_SIZES must be smallbig or bigbig, got: ${SPHINXSYS_LAMMPS_SIZES}")
    endif()

    set(_lammps_path_suffixes include includes lib bin)
    set(_lammps_known_configurations ${CMAKE_CONFIGURATION_TYPES})
    if(CMAKE_BUILD_TYPE)
        list(APPEND _lammps_known_configurations "${CMAKE_BUILD_TYPE}")
    endif()
    list(REMOVE_DUPLICATES _lammps_known_configurations)
    foreach(_configuration IN LISTS _lammps_known_configurations)
        list(APPEND _lammps_path_suffixes
            "${_configuration}"
            "lib/${_configuration}"
            "bin/${_configuration}")
    endforeach()

    find_path(SPHINXSYS_LAMMPS_INCLUDE_ROOT
        NAMES lammps/library.h
        HINTS "${SPHINXSYS_LAMMPS_ROOT}"
        PATH_SUFFIXES ${_lammps_path_suffixes}
        DOC "Include root containing lammps/library.h")

    find_library(SPHINXSYS_LAMMPS_LIBRARY
        NAMES lammps liblammps
        HINTS "${SPHINXSYS_LAMMPS_ROOT}"
        PATH_SUFFIXES ${_lammps_path_suffixes}
        DOC "LAMMPS library or Windows import library")

    if(NOT SPHINXSYS_LAMMPS_INCLUDE_ROOT OR
       NOT EXISTS "${SPHINXSYS_LAMMPS_INCLUDE_ROOT}/lammps/library.h")
        message(FATAL_ERROR
            "LAMMPS lammps/library.h was not found. Install LAMMPS and set LAMMPS_DIR, or set "
            "SPHINXSYS_LAMMPS_ROOT/SPHINXSYS_LAMMPS_INCLUDE_ROOT for fallback discovery.")
    endif()
    if(NOT SPHINXSYS_LAMMPS_LIBRARY OR NOT EXISTS "${SPHINXSYS_LAMMPS_LIBRARY}")
        message(FATAL_ERROR
            "The LAMMPS library was not found. Install LAMMPS and set LAMMPS_DIR, or set "
            "SPHINXSYS_LAMMPS_ROOT/SPHINXSYS_LAMMPS_LIBRARY for fallback discovery.")
    endif()

    if(WIN32)
        find_file(SPHINXSYS_LAMMPS_RUNTIME
            NAMES
                "${CMAKE_SHARED_LIBRARY_PREFIX}lammps${CMAKE_SHARED_LIBRARY_SUFFIX}"
                "liblammps${CMAKE_SHARED_LIBRARY_SUFFIX}"
                "lammps${CMAKE_SHARED_LIBRARY_SUFFIX}"
            HINTS "${SPHINXSYS_LAMMPS_ROOT}"
            PATH_SUFFIXES ${_lammps_path_suffixes}
            DOC "LAMMPS runtime DLL")
        if(NOT SPHINXSYS_LAMMPS_RUNTIME OR NOT EXISTS "${SPHINXSYS_LAMMPS_RUNTIME}")
            message(FATAL_ERROR
                "The LAMMPS runtime DLL was not found. Set SPHINXSYS_LAMMPS_RUNTIME explicitly.")
        endif()

        add_library(LAMMPS::lammps SHARED IMPORTED GLOBAL)
        set_target_properties(LAMMPS::lammps PROPERTIES
            IMPORTED_IMPLIB "${SPHINXSYS_LAMMPS_LIBRARY}"
            IMPORTED_LOCATION "${SPHINXSYS_LAMMPS_RUNTIME}")
    else()
        add_library(LAMMPS::lammps UNKNOWN IMPORTED GLOBAL)
        set_target_properties(LAMMPS::lammps PROPERTIES
            IMPORTED_LOCATION "${SPHINXSYS_LAMMPS_LIBRARY}")
    endif()

    string(TOUPPER "${_lammps_sizes}" _lammps_sizes_definition)
    set_target_properties(LAMMPS::lammps PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${SPHINXSYS_LAMMPS_INCLUDE_ROOT}"
        INTERFACE_COMPILE_DEFINITIONS "LAMMPS_${_lammps_sizes_definition}")

    message(STATUS "Using fallback LAMMPS library: ${SPHINXSYS_LAMMPS_LIBRARY}")
    message(STATUS "LAMMPS integer-size ABI: ${_lammps_sizes}")
    if(CMAKE_CONFIGURATION_TYPES)
        message(STATUS
            "LAMMPS fallback uses the selected library/runtime pair for this build tree. "
            "Use an installed LAMMPS package for configuration-specific imported locations.")
    endif()
endfunction()
