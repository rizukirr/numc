# FindBLIS.cmake - Find BLIS library
#
# This module defines:
#   BLIS_FOUND        - True if BLIS is found
#   BLIS_LIBRARIES    - Libraries to link
#   BLIS_INCLUDE_DIRS - Include directories
#
# Search hints:
#   BLIS_ROOT         - Root directory of BLIS installation

find_path(BLIS_INCLUDE_DIR
    NAMES blis/blis.h blis.h
    HINTS
        ${BLIS_ROOT}
        $ENV{BLIS_ROOT}
        /usr/local
        /usr
    PATH_SUFFIXES include include/blis
)

find_library(BLIS_LIBRARY
    NAMES blis
    HINTS
        ${BLIS_ROOT}
        $ENV{BLIS_ROOT}
        /usr/local
        /usr
    PATH_SUFFIXES lib lib64
)

# Handle multithreaded variant
find_library(BLIS_MT_LIBRARY
    NAMES blis-mt
    HINTS
        ${BLIS_ROOT}
        $ENV{BLIS_ROOT}
        /usr/local
        /usr
    PATH_SUFFIXES lib lib64
)

# Prefer multithreaded version if found
if(BLIS_MT_LIBRARY)
    set(BLIS_LIBRARY ${BLIS_MT_LIBRARY})
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(BLIS
    REQUIRED_VARS BLIS_LIBRARY BLIS_INCLUDE_DIR
)

if(BLIS_FOUND)
    set(BLIS_LIBRARIES ${BLIS_LIBRARY})
    set(BLIS_INCLUDE_DIRS ${BLIS_INCLUDE_DIR})

    # Create imported target
    if(NOT TARGET BLIS::BLIS)
        add_library(BLIS::BLIS UNKNOWN IMPORTED)
        set_target_properties(BLIS::BLIS PROPERTIES
            IMPORTED_LOCATION "${BLIS_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${BLIS_INCLUDE_DIR}"
        )
    endif()
endif()

mark_as_advanced(BLIS_INCLUDE_DIR BLIS_LIBRARY BLIS_MT_LIBRARY)
