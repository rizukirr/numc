add_executable(test_array_core test_array_core.c)
target_link_libraries(test_array_core PRIVATE numc::numc)
target_include_directories(test_array_core PRIVATE ${PROJECT_SOURCE_DIR}/src)
add_test(NAME test_array_core COMMAND test_array_core)
