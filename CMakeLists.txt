cmake_minimum_required(VERSION 3.22)
project(libuw12 VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(uw12 libuw12.cpp)
target_include_directories(uw12
        PRIVATE
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
        $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/src>
)

# Looking for parallel threading library
set(SEARCH_PARALLEL OFF)
if (PARALLEL MATCHES TBB)
    set(USE_TBB ON)
elseif (PARALLEL MATCHES OPENMP)
    set(USE_OPENMP ON)
elseif (PARALLEL MATCHES SERIAL)
    set(BUILD_SERIAL ON)
elseif (DEFINED PARALLEL)
    message(FATAL_ERROR "Invalid parallel option: ${PARALLEL}"
            "PARALLEL must be one of 'TBB', 'OPENMP', or 'SERIAL'")
else ()
    message(STATUS "No parallel library specified, will search in default order")
    message(STATUS "Parallel threading library may be specified using `PARALLEL' option")
    set(SEARCH_PARALLEL ON)
endif ()

# Search for TBB
if (USE_TBB OR SEARCH_PARALLEL)
    message(STATUS "Looking for TBB implementation...")

    find_package(
            TBB
            CONFIG
            COMPONENTS
            tbbmalloc
            tbbmalloc_proxy
    )

    if (TBB_FOUND)
        message(STATUS "TBB Found")
        message(STATUS "TBB version: ${TBB_VERSION}")
        target_link_libraries(uw12 PUBLIC TBB::tbb TBB::tbbmalloc TBB::tbbmalloc_proxy)

        set(SEARCH_PARALLEL OFF)
        set(PARALLEL_KEY USE_TBB)
    else ()
        if (USE_TBB)
            message(FATAL_ERROR "TBB not found")
        else()
            message(STATUS "TBB not found")
        endif ()
    endif ()
endif ()

# Search for OpenMP
if (USE_OPENMP OR SEARCH_PARALLEL)
    message(STATUS "Looking for OpenMP implementation...")

    find_package(OpenMP)
    if (OpenMP_CXX_FOUND)
      target_link_libraries(uw12 PUBLIC OpenMP::OpenMP_CXX)

      set(SEARCH_PARALLEL OFF)
      set(PARALLEL_KEY USE_OMP)
    else ()
        if (USE_OPENMP)
            message(FATAL_ERROR "OpenMP not found")
        else ()
            message(STATUS "OpenMP not found")
        endif ()
    endif ()
endif ()

# Set serial build
if (BUILD_SERIAL OR SEARCH_PARALLEL)
    message(STATUS "Serial build")
    set(PARALLEL_KEY SERIAL)
endif ()

# Looking for Linear algebra library
set(SEARCH_LINALG OFF)
if (LINALG_LIB MATCHES ARMA)
    set(USE_ARMA ON)
elseif (LINALG_LIB MATCHES EIGEN)
    set(USE_EIGEN ON)
elseif (DEFINED LINALG_LIB)
    message(FATAL_ERROR "Invalid linear algebra library: ${LINALG_LIB}"
            "LINALG_LIB must be one of 'ARMA', or 'EIGEN'")
else ()
    message(STATUS "No linear algebra library specified, will search in default order")
    set(SEARCH_LINALG ON)
endif ()

add_library(linalg INTERFACE)
# Search for TBB
if (USE_ARMA OR SEARCH_LINALG)
    message(STATUS "Looking for Armadillo linear algebra library...")

    find_package(Armadillo 9.900.0)

    if (ARMADILLO_FOUND)
        message(STATUS "Found Armadillo library")
        set(SEARCH_LINALG OFF)
    else ()
        if (USE_ARMA)
            message(FATAL_ERROR "Armadillo not found")
        else ()
            message(STATUS "Armadillo not found")
        endif ()
    endif ()
endif ()

# Link Armadillo and MKL
if (ARMADILLO_FOUND)
    message(STATUS "Armadillo version: " ${ARMADILLO_VERSION_STRING})
    message(STATUS "ARMADILLO_INCLUDE_DIRS : " ${ARMADILLO_INCLUDE_DIRS})

    target_compile_definitions(
            linalg
            INTERFACE
            ARMA_DONT_USE_WRAPPER
            ARMA_USE_LAPACK
            $<$<NOT:$<CONFIG:Debug>>:ARMA_NO_DEBUG>
            $<$<BOOL:${TBB_FOUND}>:ARMA_USE_TBB_ALLOC>
    )

    set(LINALG_KEY USE_ARMA)

    target_include_directories(linalg INTERFACE ${ARMADILLO_INCLUDE_DIRS})

    if (TBB_FOUND)
        target_compile_definitions(linalg INTERFACE ${TBB_DEFINITIONS})
        target_link_libraries(linalg INTERFACE ${TBB_LIBRARIES})
        target_include_directories(linalg INTERFACE ${TBB_INCLUDE_DIRS})
    endif ()

    message(STATUS "Armadillo requires backend BLAS library to function")

    message(STATUS "Searching for MKL, finding library...")
    set(MKL_LINK sdl)
    if (TBB_FOUND)
        set(MKL_THREADING tbb_thread)
    elseif (USE_OMP)
        set(MKL_THREADING intel_thread)
    else ()
        set(MKL_THREADING sequential)
    endif ()

    find_package(MKL CONFIG)
    if (MKL_FOUND)
        message(STATUS "MKL library found")
        message(STATUS "MKL_VERSION: ${MKL_VERSION}")
        message(STATUS "MKL_LIBRARIES: ${MKL_LIBRARIES}")

        target_link_libraries(linalg INTERFACE $<LINK_ONLY:MKL::MKL>)
    else ()
        message(STATUS "MKL not found, looking for alternative BLAS implementation...")

        find_package(BLAS REQUIRED)
        find_package(LAPACK REQUIRED)

        target_link_libraries(linalg INTERFACE BLAS::BLAS LAPACK::LAPACK)
    endif ()
endif ()

# Search for OpenMP
if (USE_EIGEN OR SEARCH_LINALG)
    message(STATUS "Looking for Eigen linear algebra library...")

    find_package(Eigen3 3.3 NO_MODULE)

    if (TARGET Eigen3::Eigen)
        message(STATUS "Eigen3 found: ${Eigen3_DIR}")

        target_link_libraries(linalg INTERFACE Eigen3::Eigen)

        set(SEARCH_LINALG OFF)
        set(LINALG_KEY USE_EIGEN)
    else ()
        if (USE_EIGEN)
            message(FATAL_ERROR "Eigen linear algebra library not found")
        else ()
            message(STATUS "Eigen linear algebra library not found")
        endif ()
    endif ()
endif ()

# Check linear algebra library has been found
if (SEARCH_LINALG)
    message(FATAL_ERROR "No linear algebra library found")
endif ()

target_link_libraries(uw12 PUBLIC linalg)

if (NOT DEFINED USE_CATCH)
    set(USE_CATCH ON)
    message(STATUS "Setting up Catch2 for unit testing, to disable set USE_CATCH=OFF")
endif ()

if (USE_CATCH)
    find_package(Catch2 3)
    if (${Catch2_DIR} MATCHES "Catch2_DIR_NOTFOUND")
        message(STATUS "Catch2 not found -- Cannot build tests")
        set(FOUND_CATCH FALSE)
    else ()
        message(STATUS "Catch2 found")
        message(STATUS "Catch2_DIR: ${Catch2_DIR}")
        set(FOUND_CATCH TRUE)
    endif ()
endif ()

add_subdirectory(src)

if (USE_CATCH)
    if (FOUND_CATCH)
        include(CTest)
        include(Catch)

        enable_testing()

        add_subdirectory(tests)
    endif ()
endif ()
