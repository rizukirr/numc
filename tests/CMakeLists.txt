# Tests subdirectory for numc
# No need for cmake_minimum_required or project() when used as subdirectory

# Helper function to add a test
function(add_numc_test test_name source_file)
    add_executable(${test_name} ${source_file})
    target_link_libraries(${test_name} numc_src m)
    
    # Add AddressSanitizer for debug builds
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        target_compile_options(${test_name} PRIVATE -fsanitize=address -fno-omit-frame-pointer)
        target_link_options(${test_name} PRIVATE -fsanitize=address)
    endif()
    
    # Add test to CTest
    add_test(NAME ${test_name} COMMAND ${test_name})
endfunction()

# Add individual test files
add_numc_test(test_dtype test_dtype.c)
add_numc_test(test_array_create test_array_create.c)
add_numc_test(test_array_fill test_array_fill.c)
add_numc_test(test_array_slice test_array_slice.c)
add_numc_test(test_array_reshape test_array_reshape.c)
add_numc_test(test_array_copy test_array_copy.c)
add_numc_test(test_memory_alignment test_memory_alignment.c)

# Print message
message(STATUS "Configured numc tests")
message(STATUS "  Run tests with: cd build && ctest")
message(STATUS "  Run tests with output: cd build && ctest --verbose")
