add_executable(test_parallel test_parallel.cpp)
target_link_libraries(test_parallel Catch2::Catch2WithMain uw12)
catch_discover_tests(test_parallel)

add_executable(test_print test_print.cpp)
target_link_libraries(test_print Catch2::Catch2WithMain)
catch_discover_tests(test_print)

add_executable(test_linear_algebra_library
        test_linalg_vec.cpp
        test_linalg_mat.cpp
        test_linalg_mat_op.cpp)
target_link_libraries(test_linear_algebra_library Catch2::Catch2WithMain uw12)
catch_discover_tests(test_linear_algebra_library)

add_executable(test_utils test_utils.cpp)
target_link_libraries(test_utils Catch2::Catch2WithMain uw12)
catch_discover_tests(test_utils)

add_executable(test_integral_transformations test_integral_transformations.cpp)
target_link_libraries(test_integral_transformations Catch2::Catch2WithMain uw12)
catch_discover_tests(test_integral_transformations)

add_executable(test_integral_functions test_integral_functions.cpp)
target_link_libraries(test_integral_functions Catch2::Catch2WithMain uw12)
catch_discover_tests(test_integral_functions)

add_executable(test_base_integrals test_base_integrals.cpp)
target_link_libraries(test_base_integrals Catch2::Catch2WithMain uw12)
catch_discover_tests(test_base_integrals)

add_executable(test_integrals test_integrals.cpp)
target_link_libraries(test_integrals Catch2::Catch2WithMain uw12)
catch_discover_tests(test_integrals)

add_executable(test_density_utils test_density_utils.cpp)
target_link_libraries(test_density_utils Catch2::Catch2WithMain uw12)
catch_discover_tests(test_density_utils)

add_executable(test_numerical_fock test_numerical_fock.cpp)
target_link_libraries(test_numerical_fock Catch2::Catch2WithMain uw12)
catch_discover_tests(test_numerical_fock)

add_executable(test_two_electron test_two_el.cpp)
target_link_libraries(test_two_electron Catch2::Catch2WithMain uw12)
catch_discover_tests(test_two_electron)

add_executable(test_four_electron_utils test_four_el_utils.cpp)
target_link_libraries(test_four_electron_utils Catch2::Catch2WithMain uw12)
catch_discover_tests(test_four_electron_utils)

add_executable(test_four_electron test_four_el.cpp)
target_link_libraries(test_four_electron Catch2::Catch2WithMain uw12)
catch_discover_tests(test_four_electron)

add_executable(test_ri_utils test_ri_utils.cpp)
target_link_libraries(test_ri_utils Catch2::Catch2WithMain uw12)
catch_discover_tests(test_ri_utils)
