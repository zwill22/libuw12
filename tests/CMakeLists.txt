add_executable(test_parallel test_parallel.cpp)
target_link_libraries(test_parallel Catch2::Catch2WithMain uw12)
add_test(NAME UNITTEST_parallel COMMAND bin/test_parallel)
