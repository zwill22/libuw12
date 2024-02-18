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
