include(ExternalProject)
include(ProcessorCount)

# "x86_64" = fat binary with runtime CPUID dispatch (all Intel + AMD kernels)
# Override with -DBLIS_CONFIG=auto for faster compile during dev
set(BLIS_CONFIG "x86_64" CACHE STRING "BLIS configuration target (x86_64, auto, amd64, intel64, ...)")

set(BLIS_SOURCE_DIR "${CMAKE_SOURCE_DIR}/external/blis")
set(BLIS_BUILD_DIR  "${CMAKE_BINARY_DIR}/_deps/blis-build")
set(BLIS_INSTALL_DIR "${CMAKE_BINARY_DIR}/_deps/blis-install")

# Use half the cores for BLIS build — x86_64 fat binary compiles many arch
# variants and can be memory-intensive.  Fall back to 2 if detection fails.
ProcessorCount(_NPROC)
if(_NPROC EQUAL 0)
    set(_NPROC 2)
endif()
math(EXPR BLIS_BUILD_JOBS "${_NPROC} / 2")
if(BLIS_BUILD_JOBS LESS 1)
    set(BLIS_BUILD_JOBS 1)
endif()

ExternalProject_Add(blis_vendored
    SOURCE_DIR      "${BLIS_SOURCE_DIR}"
    BINARY_DIR      "${BLIS_BUILD_DIR}"
    INSTALL_DIR     "${BLIS_INSTALL_DIR}"
    CONFIGURE_COMMAND
        "${BLIS_SOURCE_DIR}/configure"
        "--prefix=<INSTALL_DIR>"
        "--enable-threading=openmp"
        "--enable-static"
        "--disable-shared"
        "${BLIS_CONFIG}"
    BUILD_COMMAND     make -j${BLIS_BUILD_JOBS}
    INSTALL_COMMAND   make install
    BUILD_ALWAYS      FALSE
    BUILD_BYPRODUCTS  "${BLIS_INSTALL_DIR}/lib/libblis.a"
)

# Create include dir at configure time so CMake doesn't reject the imported target
file(MAKE_DIRECTORY "${BLIS_INSTALL_DIR}/include/blis")

# Create imported target
add_library(BLIS::BLIS_Vendored STATIC IMPORTED GLOBAL)
set_target_properties(BLIS::BLIS_Vendored PROPERTIES
    IMPORTED_LOCATION             "${BLIS_INSTALL_DIR}/lib/libblis.a"
    INTERFACE_INCLUDE_DIRECTORIES "${BLIS_INSTALL_DIR}/include/blis"
)
add_dependencies(BLIS::BLIS_Vendored blis_vendored)
