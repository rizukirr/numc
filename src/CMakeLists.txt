add_library(numc
    array_core.c
    array_print.c
    alloc.c
    elemwise/binary.c
    elemwise/unary.c
    elemwise/clip.c
    reduction/reduction.c
    error.c
)
add_library(numc::numc ALIAS numc)

target_include_directories(numc
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/elemwise
        ${CMAKE_CURRENT_SOURCE_DIR}/reduction
)

target_compile_options(numc PRIVATE
    -Wall -Wextra -Wpedantic
    $<$<CONFIG:Debug>:-g -O0 -fno-omit-frame-pointer>
    $<$<CONFIG:Release>:-O3 -march=native>
    $<$<CONFIG:RelWithDebInfo>:-O2 -g -march=native>
)

# Enable debug error context (file/line) in Debug builds
target_compile_definitions(numc PRIVATE
    $<$<CONFIG:Debug>:NUMC_DEBUG_ERROR_CONTEXT=1>
)

# Thin LTO in Release when requested
if(NUMC_ENABLE_LTO)
  target_compile_options(numc PUBLIC $<$<CONFIG:Release>:-flto=thin>)
  target_link_options(numc PUBLIC $<$<CONFIG:Release>:-flto=thin>)
endif()

# Fast-math relaxations (safe defaults that avoid errno/fenv changes)
if(NUMC_ENABLE_FAST_MATH)
  target_compile_options(numc PUBLIC $<$<CONFIG:Release>:-fno-math-errno -ffp-contract=fast>)
endif()

# PGO flags are opted-in and require a manual workflow (see docs)
if(NUMC_ENABLE_PGO)
  # This flag acts as a placeholder; actual PGO workflow is performed externally
  target_compile_options(numc PUBLIC $<$<CONFIG:Release>:-fprofile-generate>)
  target_link_options(numc PUBLIC $<$<CONFIG:Release>:-fprofile-generate>)
endif()

# AddressSanitizer
if(NUMC_ENABLE_ASAN)
    target_compile_options(numc PUBLIC -fsanitize=address)
    target_link_options(numc PUBLIC -fsanitize=address)
endif()

# OpenMP (find_package is at root scope)
if(OpenMP_C_FOUND)
    target_link_libraries(numc PUBLIC OpenMP::OpenMP_C)
endif()

# Math library
target_link_libraries(numc PUBLIC m)
