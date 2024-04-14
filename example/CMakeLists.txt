cmake_minimum_required(VERSION 3.22)
project(libuw12_example VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(example example.cpp)

find_library(uw12 uw12 NAMES uw12 libuw12 REQUIRED PATHS ${CMAKE_LIBRARY_PATH})
find_path(uw12_include_dirs libuw12.hpp PATHS ${CMAKE_INCLUDE_PATH})

target_link_libraries(example PRIVATE uw12)
target_include_directories(example PRIVATE ${uw12_include_dirs}
        ${uw12_include_dirs}/src
        ${CMAKE_LIBRARY_PATH}/src
)
target_link_directories(example PRIVATE ${CMAKE_LIBRARY_PATH})

if (DEFINED ${CMAKE_UW12_ROOT})
    add_compile_definitions("UW12_ROOT=${CMAKE_UW12_ROOT}")
else ()
    add_compile_definitions("UW12_ROOT=${CMAKE_INCLUDE_PATH}")
endif ()

# TODO Remove having to discover libuw12 dependencies again (PIMPL?)
# Find TBB
find_package(
        TBB
        CONFIG
        COMPONENTS
        tbbmalloc
        tbbmalloc_proxy
)
target_link_libraries(example PRIVATE TBB::tbb TBB::tbbmalloc TBB::tbbmalloc_proxy)

# Find Arma
find_package(Armadillo 9.900.0)
target_include_directories(example PRIVATE ${ARMADILLO_INCLUDE_DIRS})
