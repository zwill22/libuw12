configure_file(config.hpp.in ${CMAKE_BINARY_DIR}/src/config.hpp)
target_include_directories(uw12 PUBLIC ${CMAKE_BINARY_DIR}/src/)

target_sources(uw12 PRIVATE uw12.cpp)

add_subdirectory(integrals)
add_subdirectory(two_electron)
add_subdirectory(four_electron)
add_subdirectory(three_electron)
