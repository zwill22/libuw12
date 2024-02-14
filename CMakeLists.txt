cmake_minimum_required(VERSION 3.22)
project(libuw12 VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(BUILD_SHARED_LIBS "Build shared libraries" ON)

add_library(uw12 SHARED libuw12.cpp)

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
        add_compile_definitions(USE_TBB)
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
      target_link_libraries(uw12 PUBLIC OpenMP::OpenMP__CXX)

      set(SEARCH_PARALLEL OFF)
      add_compile_definitions(USE_OMP)
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

    add_library(arma INTERFACE)
    target_compile_definitions(
            arma
            INTERFACE
            ARMA_DONT_USE_WRAPPER
            ARMA_USE_LAPACK
            $<$<NOT:$<CONFIG:Debug>>:ARMA_NO_DEBUG>
            $<$<BOOL:${TBB_FOUND}>:ARMA_USE_TBB_ALLOC>
    )

    add_compile_definitions(USE_ARMA)

    target_include_directories(arma INTERFACE ${ARMADILLO_INCLUDE_DIRS})

    target_link_libraries(uw12 PUBLIC arma)
    if (TBB_FOUND)
        target_compile_definitions(arma INTERFACE ${TBB_DEFINITIONS})
        target_link_libraries(arma INTERFACE ${TBB_LIBRARIES})
        target_include_directories(arma INTERFACE ${TBB_INCLUDE_DIRS})
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

        target_link_libraries(arma INTERFACE $<LINK_ONLY:MKL::MKL>)
    else ()
        message(STATUS "MKL not found, looking for alternative BLAS implementation...")
        message(WARNING "MKL recommended, if possible please link to MKL!")

        find_package(BLAS REQUIRED)

        target_link_libraries(arma INTERFACE BLAS::BLAS)
    endif ()
endif ()

# Search for OpenMP
if (USE_EIGEN OR SEARCH_LINALG)
    message(STATUS "Looking for Eigen linear algebra library...")

    find_package(Eigen3 3.3 NO_MODULE)

    if (TARGET Eigen3::Eigen)
        add_compile_definitions(USE_EIGEN)
        message(STATUS "Eigen3 found: ${Eigen3_DIR}")
        message(WARNING "Eigen linear algebra library not properly tested")

        target_link_libraries(uw12 PUBLIC Eigen3::Eigen)

        set(SEARCH_LINALG OFF)
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

if (USE_CATCH)
    if (FOUND_CATCH)
        include(CTest)
        include(Catch)

        enable_testing()

        add_subdirectory(tests)
    endif ()
endif ()
