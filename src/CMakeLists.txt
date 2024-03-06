configure_file(config.hpp.in ${CMAKE_BINARY_DIR}/src/config.hpp)
target_include_directories(uw12 PUBLIC ${CMAKE_BINARY_DIR}/src/)

add_subdirectory(integrals)
add_subdirectory(two_electron)
