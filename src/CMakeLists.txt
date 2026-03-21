set(NUMC_CORE_SOURCES
    array_core.c
    array_print.c
    alloc.c
    elemwise/arithmetic.c
    elemwise/pow.c
    elemwise/compare.c
    elemwise/select.c
    elemwise/unary.c
    elemwise/clip.c
    reduction/sum.c
    reduction/mean.c
    reduction/max.c
    reduction/min.c
    reduction/argmax.c
    reduction/argmin.c
    reduction/dot.c
    matmul/matmul.c
    random/random.c
    error.c
)

# Dedicated .S assembly micro-kernels for x86_64 AVX2 packed GEMM
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    list(APPEND NUMC_CORE_SOURCES
        intrinsics/gemm_ukernel_f32_6x16_avx2.S
        intrinsics/gemm_ukernel_f64_6x8_avx2.S
        intrinsics/gemm_ukernel_i32_6x16_avx2.S
    )
endif()

add_library(numc ${NUMC_CORE_SOURCES})

add_library(numc::numc ALIAS numc)

target_include_directories(numc
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/elemwise
        ${CMAKE_CURRENT_SOURCE_DIR}/reduction
        ${CMAKE_CURRENT_SOURCE_DIR}/matmul
        ${CMAKE_CURRENT_SOURCE_DIR}/random
)

set(MARCH_FLAG "-march=native")
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64" OR CMAKE_C_COMPILER_TARGET MATCHES "aarch64")
    set(MARCH_FLAG "-march=armv8-a")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "riscv64")
    set(MARCH_FLAG "")
endif()

target_compile_options(numc PRIVATE
    -Wall -Wextra -Wpedantic -Wno-pass-failed -Wno-overlength-strings
    $<$<BOOL:${NUMC_WERROR}>:-Werror>
    $<$<CONFIG:Debug>:-g -O0 -fno-omit-frame-pointer>
    $<$<CONFIG:Release>:-O3 ${MARCH_FLAG} -fno-math-errno -fno-signed-zeros -fno-trapping-math -funroll-loops>
    $<$<CONFIG:RelWithDebInfo>:-O2 -g ${MARCH_FLAG}>
)

# Visibility and PIC
set_target_properties(numc PROPERTIES
    C_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
    POSITION_INDEPENDENT_CODE ON
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
)

# Enable debug error context (file/line) in Debug builds
target_compile_definitions(numc PRIVATE
    $<$<CONFIG:Debug>:NUMC_DEBUG_ERROR_CONTEXT=1>
)



# AddressSanitizer
if(NUMC_ENABLE_ASAN)
    target_compile_options(numc PUBLIC -fsanitize=address)
    target_link_options(numc PUBLIC -fsanitize=address)
endif()

# OpenMP (find_package is at root scope)
if(OpenMP_C_FOUND)
    target_link_libraries(numc PUBLIC OpenMP::OpenMP_C)
    target_compile_definitions(numc PRIVATE HAVE_OMP=1)
endif()

# Math library
target_link_libraries(numc PUBLIC m)
