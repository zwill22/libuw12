set(DOXYGEN_OUTPUT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/doxygen/)

message(STATUS "Searching for Doxygen")
find_package(Doxygen COMPONENTS dot OPTIONAL_COMPONENTS mscgen dia)
if (DOXYGEN_FOUND)
    message(STATUS "Doxygen Found")
    message(STATUS "Doxygen version: ${DOXYGEN_VERSION}")

    set(DOXYGEN_USE_MATHJAX YES)
    set(DOXYGEN_GENERATE_XML YES)

    doxygen_add_docs(doc ${PROJECT_SOURCE_DIR})
else ()
    message(STATUS "Doxygen not found - cannot build documentation")
endif ()
