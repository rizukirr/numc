# Tests subdirectory for numc
# No need for cmake_minimum_required or project() when used as subdirectory

# Helper function to add a test
function(add_numc_test test_name source_file)
    add_executable(${test_name} ${source_file})
    target_link_libraries(${test_name} numc m)
    
    # Add AddressSanitizer for debug builds
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        target_compile_options(${test_name} PRIVATE -fsanitize=address -fno-omit-frame-pointer)
        target_link_options(${test_name} PRIVATE -fsanitize=address)
    endif()
    
    # Add test to CTest (suppress known OpenMP thread pool leak in LSAN)
    add_test(NAME ${test_name} COMMAND ${test_name})
    set_tests_properties(${test_name} PROPERTIES
        ENVIRONMENT "LSAN_OPTIONS=suppressions=${CMAKE_CURRENT_SOURCE_DIR}/lsan_suppressions.txt")
endfunction()

# Unit tests
add_numc_test(test_numc_type unit/test_numc_type.c)
add_numc_test(test_create unit/test_create.c)
add_numc_test(test_core unit/test_core.c)
add_numc_test(test_fill unit/test_fill.c)
add_numc_test(test_slice unit/test_slice.c)
add_numc_test(test_reshape unit/test_reshape.c)
add_numc_test(test_copy unit/test_copy.c)
add_numc_test(test_concat unit/test_concat.c)
add_numc_test(test_transpose unit/test_transpose.c)
add_numc_test(test_memory unit/test_memory.c)
add_numc_test(test_add unit/test_add.c)
add_numc_test(test_reduce unit/test_reduce.c)
add_numc_test(test_scalar unit/test_scalar.c)

# Print message
message(STATUS "Configured numc tests")
message(STATUS "  Run tests with: cd build && ctest")
message(STATUS "  Run tests with output: cd build && ctest --verbose")
