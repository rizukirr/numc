set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

set(CMAKE_C_COMPILER aarch64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)

set(CMAKE_FIND_ROOT_PATH /usr/aarch64-linux-gnu)
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)

# Static linking for qemu-user
set(CMAKE_EXE_LINKER_FLAGS_INIT "-static")

# Force static OpenMP
set(OpenMP_C_LIB_NAMES "gomp;pthread" CACHE STRING "")
set(OpenMP_gomp_LIBRARY "/usr/aarch64-linux-gnu/lib/libgomp.a" CACHE FILEPATH "")

# QEMU runner for ctest
set(CMAKE_CROSSCOMPILING_EMULATOR "qemu-aarch64;-L;/usr/aarch64-linux-gnu")
