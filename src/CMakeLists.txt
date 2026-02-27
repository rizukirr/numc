add_library(numc
    array_core.c
    array_print.c
    alloc.c
    elemwise/binary.c
    elemwise/unary.c
    elemwise/clip.c
    reduction/reduction.c
    matmul/matmul.c
    matmul/blas.c
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
        ${CMAKE_CURRENT_SOURCE_DIR}/matmul
)

target_compile_options(numc PRIVATE
    -Wall -Wextra -Wpedantic
    $<$<CONFIG:Debug>:-g -O0 -fno-omit-frame-pointer>
    $<$<CONFIG:Release>:-O3 -march=native -fno-math-errno -fno-signed-zeros -fno-trapping-math -funroll-loops>
    $<$<CONFIG:RelWithDebInfo>:-O2 -g -march=native>
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
endif()

if(NUMC_USE_BLAS AND BLAS_FOUND)
    target_compile_definitions(numc PRIVATE HAVE_BLAS=1)
    target_link_libraries(numc PRIVATE ${BLAS_LIBRARIES})
    if(BLAS_INCLUDE_DIRS)
        target_include_directories(numc PRIVATE ${BLAS_INCLUDE_DIRS})
    endif()

endif()

# Math library
target_link_libraries(numc PUBLIC m)
