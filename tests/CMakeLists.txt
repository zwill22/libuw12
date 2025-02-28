add_library(catch SHARED catch.hpp)
target_link_libraries(catch Catch2::Catch2WithMain)
target_include_directories(catch
        INTERFACE
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
        $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/src>
)

add_executable(test_parallel test_parallel.cpp)
target_link_libraries(test_parallel catch uw12)
catch_discover_tests(test_parallel)

add_executable(test_print test_print.cpp)
target_link_libraries(test_print catch)
catch_discover_tests(test_print)

add_executable(test_linear_algebra_library
        test_linalg_vec.cpp
        test_linalg_mat.cpp
        test_linalg_mat_op.cpp)
target_link_libraries(test_linear_algebra_library catch uw12)
catch_discover_tests(test_linear_algebra_library)

add_executable(test_utils test_utils.cpp)
target_link_libraries(test_utils catch uw12)
catch_discover_tests(test_utils)

add_executable(test_integral_transformations test_integral_transformations.cpp)
target_link_libraries(test_integral_transformations catch uw12)
catch_discover_tests(test_integral_transformations)

add_executable(test_integral_functions test_integral_functions.cpp)
target_link_libraries(test_integral_functions catch uw12)
catch_discover_tests(test_integral_functions)

add_executable(test_base_integrals test_base_integrals.cpp)
target_link_libraries(test_base_integrals catch uw12)
catch_discover_tests(test_base_integrals)

add_executable(test_integrals test_integrals.cpp)
target_link_libraries(test_integrals catch uw12)
catch_discover_tests(test_integrals)

add_executable(test_density_utils test_density_utils.cpp)
target_link_libraries(test_density_utils catch uw12)
catch_discover_tests(test_density_utils)

add_executable(test_numerical_fock test_numerical_fock.cpp)
target_link_libraries(test_numerical_fock catch uw12)
catch_discover_tests(test_numerical_fock)

add_executable(test_two_electron test_two_el.cpp)
target_link_libraries(test_two_electron catch uw12)
catch_discover_tests(test_two_electron)

add_executable(test_four_electron_utils test_four_el_utils.cpp)
target_link_libraries(test_four_electron_utils catch uw12)
catch_discover_tests(test_four_electron_utils)

add_executable(test_four_electron test_four_el.cpp)
target_link_libraries(test_four_electron catch uw12)
catch_discover_tests(test_four_electron)

add_executable(test_ri_utils test_ri_utils.cpp)
target_link_libraries(test_ri_utils catch uw12)
catch_discover_tests(test_ri_utils)

add_executable(test_three_el_direct_utils test_three_el_direct_utils.cpp)
target_link_libraries(test_three_el_direct_utils catch uw12)
catch_discover_tests(test_three_el_direct_utils)

add_executable(test_three_el_indirect_utils test_three_el_indirect_utils.cpp)
target_link_libraries(test_three_el_indirect_utils catch uw12)
catch_discover_tests(test_three_el_indirect_utils)

add_executable(test_three_electron test_three_el.cpp)
target_link_libraries(test_three_electron catch uw12)
catch_discover_tests(test_three_electron)

add_executable(test_uw12 test_uw12.cpp)
target_link_libraries(test_uw12 catch uw12)
catch_discover_tests(test_uw12)

add_executable(test_uw12_water test_water.cpp)
target_link_libraries(test_uw12_water catch uw12)
catch_discover_tests(test_uw12_water)
