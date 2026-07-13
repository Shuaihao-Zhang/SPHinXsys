STRING(REGEX REPLACE ".*/(.*)" "\\1" CURRENT_FOLDER ${CMAKE_CURRENT_SOURCE_DIR})
PROJECT("${CURRENT_FOLDER}")

SET(EXECUTABLE_OUTPUT_PATH "${PROJECT_BINARY_DIR}/bin/")

add_executable(${PROJECT_NAME})
target_sources(${PROJECT_NAME} PRIVATE
    test_2d_lammps_underwater_granular_collapse.cpp
    test_2d_lammps_underwater_granular_collapse.h)
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)
target_link_libraries(${PROJECT_NAME} PRIVATE sphinxsys_2d sphinxsys_lammps_examples_common)
sphinxsys_configure_lammps_example(${PROJECT_NAME})
set_target_properties(${PROJECT_NAME} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${EXECUTABLE_OUTPUT_PATH}"
    RUNTIME_OUTPUT_DIRECTORY_DEBUG "${EXECUTABLE_OUTPUT_PATH}"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${EXECUTABLE_OUTPUT_PATH}"
    RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO "${EXECUTABLE_OUTPUT_PATH}"
    RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL "${EXECUTABLE_OUTPUT_PATH}"
    VS_DEBUGGER_WORKING_DIRECTORY "${EXECUTABLE_OUTPUT_PATH}")

add_test(NAME ${PROJECT_NAME}
    COMMAND ${PROJECT_NAME}
    WORKING_DIRECTORY "${EXECUTABLE_OUTPUT_PATH}")

add_test(NAME ${PROJECT_NAME}_particle_relaxation
    COMMAND ${PROJECT_NAME} --relax=true --state_recording=false
    WORKING_DIRECTORY "${EXECUTABLE_OUTPUT_PATH}")

set_tests_properties(${PROJECT_NAME} PROPERTIES
    LABELS "lammps;granular-collapse;2d;long"
    RUN_SERIAL TRUE)
set_tests_properties(${PROJECT_NAME}_particle_relaxation PROPERTIES
    LABELS "lammps;granular-collapse;2d;particle-relaxation;long"
    RUN_SERIAL TRUE)
