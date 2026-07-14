if(SPHINXSYS_3D)
    add_subdirectory(test_3d_fixed_sphere_hydro_force)
endif()

if(NOT SPHINXSYS_WITH_LAMMPS)
    message(STATUS "Skipping LAMMPS-linked examples because SPHINXSYS_WITH_LAMMPS is OFF")
    return()
endif()

if(NOT TARGET SPHinXsys::sphinxsys_lammps)
    message(FATAL_ERROR "SPHinXsys::sphinxsys_lammps is required by LAMMPS examples")
endif()

function(sphinxsys_configure_lammps_example target_name)
    target_link_libraries(${target_name} PRIVATE SPHinXsys::sphinxsys_lammps)
    target_compile_definitions(${target_name} PRIVATE
        SPHINXSYS_LAMMPS_RUNTIME_FILENAME="$<TARGET_FILE_NAME:LAMMPS::lammps>")
    add_custom_command(TARGET ${target_name} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "$<TARGET_FILE:LAMMPS::lammps>"
                "$<TARGET_FILE_DIR:${target_name}>"
        COMMENT "Staging the LAMMPS runtime library next to ${target_name}")
    if(APPLE)
        set_property(TARGET ${target_name} APPEND PROPERTY
            BUILD_RPATH "@loader_path")
    elseif(UNIX)
        set_property(TARGET ${target_name} APPEND PROPERTY
            BUILD_RPATH "$ORIGIN")
    endif()
endfunction()

add_subdirectory(common)

if(SPHINXSYS_2D)
    add_subdirectory(test_2d_lammps_granular_collapse)
    add_subdirectory(test_2d_lammps_two_way_water_entry)
    add_subdirectory(test_2d_lammps_underwater_granular_collapse)
endif()

if(SPHINXSYS_3D)
    add_subdirectory(test_3d_lammps_driven_sphere)
    add_subdirectory(test_3d_lammps_link_only)
    add_subdirectory(test_3d_lammps_one_way_water_entry)
    add_subdirectory(test_3d_lammps_two_way_water_entry)
    add_subdirectory(test_3d_lammps_underwater_granular_collapse)
endif()
