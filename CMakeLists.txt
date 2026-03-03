cmake_minimum_required(VERSION 3.25)
project(numc VERSION 0.1.0 LANGUAGES C)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Target standard
set(CMAKE_C_STANDARD 23)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Standalone detection: are we the top-level project?
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(NUMC_STANDALONE ON)
else()
    set(NUMC_STANDALONE OFF)
endif()

# Options
option(NUMC_BUILD_TESTS      "Build numc tests"             ${NUMC_STANDALONE})
option(NUMC_BUILD_BENCHMARKS "Build numc benchmarks"        ${NUMC_STANDALONE})
option(NUMC_BUILD_EXAMPLES   "Build numc examples"          ${NUMC_STANDALONE})
option(NUMC_ENABLE_ASAN      "Enable AddressSanitizer"      OFF)
option(NUMC_USE_BLAS         "Enable BLAS backend (BLIS)"   OFF)
option(NUMC_WERROR           "Treat warnings as errors"     OFF)
option(NUMC_OPTIMIZE_NATIVE  "Enable -march=native"         ON)

# Dependencies (at root scope so install block can see results)
find_package(OpenMP)

if(NUMC_USE_BLAS)
    find_package(BLIS)

    if(BLIS_FOUND)
        set(BLAS_FOUND TRUE)
        set(BLAS_LIBRARIES ${BLIS_LIBRARIES})
        set(BLAS_INCLUDE_DIRS ${BLIS_INCLUDE_DIRS})
        message(STATUS "Found BLIS: ${BLIS_LIBRARIES}")
    else()
        message(WARNING "BLIS not found. Falling back to naive matmul. Install BLIS or disable NUMC_USE_BLAS.")
    endif()
endif()

# Generate version header from template
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/include/numc/version.h.in"
    "${CMAKE_CURRENT_BINARY_DIR}/include/numc/version.h"
    @ONLY
)

# Library (always built)
add_subdirectory(src)

if(NUMC_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

if(NUMC_BUILD_BENCHMARKS)
    add_subdirectory(bench)
endif()

if(NUMC_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Install & export (standalone only)
if(NUMC_STANDALONE)
    include(GNUInstallDirs)
    include(CMakePackageConfigHelpers)

    # Install the library target
    install(TARGETS numc
        EXPORT numcTargets
        ARCHIVE  DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY  DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME  DESTINATION ${CMAKE_INSTALL_BINDIR}
    )

    # Install public headers
    install(DIRECTORY include/numc
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        PATTERN "*.in" EXCLUDE
    )

    # Install generated version header
    install(FILES "${CMAKE_CURRENT_BINARY_DIR}/include/numc/version.h"
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/numc
    )

    # Install export set
    install(EXPORT numcTargets
        FILE        numcTargets.cmake
        NAMESPACE   numc::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/numc
    )

    # Version file
    write_basic_package_version_file(
        "${CMAKE_CURRENT_BINARY_DIR}/numcConfigVersion.cmake"
        COMPATIBILITY SameMajorVersion
    )

    # Config file from template
    configure_package_config_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/numcConfig.cmake.in"
        "${CMAKE_CURRENT_BINARY_DIR}/numcConfig.cmake"
        INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/numc
    )

    install(FILES
        "${CMAKE_CURRENT_BINARY_DIR}/numcConfig.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/numcConfigVersion.cmake"
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/numc
    )

    # pkg-config
    configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/numc.pc.in"
        "${CMAKE_CURRENT_BINARY_DIR}/numc.pc"
        @ONLY
    )
    install(FILES "${CMAKE_CURRENT_BINARY_DIR}/numc.pc"
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig
    )
endif()
