add_executable(test_parallel test_parallel.cpp)
target_link_libraries(test_parallel Catch2::Catch2WithMain uw12)
add_test(NAME UNITTEST_parallel COMMAND bin/test_parallel)

add_executable(test_print test_print.cpp)
target_link_libraries(test_print Catch2::Catch2WithMain uw12)
add_test(NAME UNITTEST_print COMMAND bin/test_print)
