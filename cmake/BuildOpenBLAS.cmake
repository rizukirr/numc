include(ExternalProject)
include(ProcessorCount)

set(OPENBLAS_SOURCE_DIR "${CMAKE_SOURCE_DIR}/external/openblas")
set(OPENBLAS_BUILD_DIR  "${CMAKE_BINARY_DIR}/_deps/openblas-build")
set(OPENBLAS_INSTALL_DIR "${CMAKE_BINARY_DIR}/_deps/openblas-install")

# Use half the cores — OpenBLAS builds many arch kernels
ProcessorCount(_NPROC)
if(_NPROC EQUAL 0)
    set(_NPROC 2)
endif()
math(EXPR OPENBLAS_BUILD_JOBS "${_NPROC} / 2")
if(OPENBLAS_BUILD_JOBS LESS 1)
    set(OPENBLAS_BUILD_JOBS 1)
endif()

ExternalProject_Add(openblas_vendored
    SOURCE_DIR      "${OPENBLAS_SOURCE_DIR}"
    BINARY_DIR      "${OPENBLAS_BUILD_DIR}"
    INSTALL_DIR     "${OPENBLAS_INSTALL_DIR}"
    CMAKE_ARGS
        -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
        -DCMAKE_BUILD_TYPE=Release
        -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
        -DUSE_OPENMP=ON
        -DBUILD_SHARED_LIBS=OFF
        -DNO_LAPACK=ON
        -DNO_FORTRAN=ON
        -DDYNAMIC_ARCH=ON
    BUILD_COMMAND     ${CMAKE_COMMAND} --build <BINARY_DIR> -j${OPENBLAS_BUILD_JOBS}
    INSTALL_COMMAND   ${CMAKE_COMMAND} --install <BINARY_DIR>
    BUILD_ALWAYS      FALSE
    BUILD_BYPRODUCTS  "${OPENBLAS_INSTALL_DIR}/lib/libopenblas.a"
)

# Create include dir at configure time so CMake doesn't reject the imported target
file(MAKE_DIRECTORY "${OPENBLAS_INSTALL_DIR}/include/openblas")

# Create imported target
add_library(OpenBLAS::OpenBLAS_Vendored STATIC IMPORTED GLOBAL)
set_target_properties(OpenBLAS::OpenBLAS_Vendored PROPERTIES
    IMPORTED_LOCATION             "${OPENBLAS_INSTALL_DIR}/lib/libopenblas.a"
    INTERFACE_INCLUDE_DIRECTORIES "${OPENBLAS_INSTALL_DIR}/include/openblas"
)
add_dependencies(OpenBLAS::OpenBLAS_Vendored openblas_vendored)
