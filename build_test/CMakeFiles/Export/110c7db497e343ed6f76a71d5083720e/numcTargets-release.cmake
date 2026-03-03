#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "numc::numc" for configuration "Release"
set_property(TARGET numc::numc APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(numc::numc PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "C"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libnumc.a"
  )

list(APPEND _cmake_import_check_targets numc::numc )
list(APPEND _cmake_import_check_files_for_numc::numc "${_IMPORT_PREFIX}/lib/libnumc.a" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
