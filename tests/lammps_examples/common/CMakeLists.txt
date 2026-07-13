add_library(sphinxsys_lammps_examples_common INTERFACE)

target_sources(sphinxsys_lammps_examples_common INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/lammps_io.h
    ${CMAKE_CURRENT_SOURCE_DIR}/lammps_instance.h
    ${CMAKE_CURRENT_SOURCE_DIR}/lammps_dem_adapter_common.h)

target_include_directories(sphinxsys_lammps_examples_common INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR})

target_compile_features(sphinxsys_lammps_examples_common INTERFACE cxx_std_17)
target_link_libraries(sphinxsys_lammps_examples_common INTERFACE SPHinXsys::sphinxsys_lammps)

add_executable(test_lammps_common test_lammps_common.cpp)
target_compile_features(test_lammps_common PRIVATE cxx_std_17)
sphinxsys_configure_lammps_example(test_lammps_common)
set_target_properties(test_lammps_common PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/bin"
    RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_CURRENT_BINARY_DIR}/bin"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_CURRENT_BINARY_DIR}/bin"
    RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO "${CMAKE_CURRENT_BINARY_DIR}/bin"
    RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL "${CMAKE_CURRENT_BINARY_DIR}/bin")

add_test(NAME test_lammps_common
    COMMAND test_lammps_common
    WORKING_DIRECTORY "$<TARGET_FILE_DIR:test_lammps_common>")
set_tests_properties(test_lammps_common PROPERTIES LABELS "lammps;fast;common")
