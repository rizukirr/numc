set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR riscv64)

set(CMAKE_C_COMPILER riscv64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER riscv64-linux-gnu-g++)
set(CMAKE_C_FLAGS_INIT "-march=rv64gcv")

set(CMAKE_FIND_ROOT_PATH /usr/riscv64-linux-gnu)
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)

if(EXISTS "/usr/riscv64-linux-gnu/lib/libgomp.a")
  set(CMAKE_EXE_LINKER_FLAGS_INIT "-static")
  set(OpenMP_C_LIB_NAMES "gomp;pthread" CACHE STRING "")
  set(OpenMP_gomp_LIBRARY "/usr/riscv64-linux-gnu/lib/libgomp.a" CACHE FILEPATH "")
endif()

set(CMAKE_CROSSCOMPILING_EMULATOR "qemu-riscv64;-L;/usr/riscv64-linux-gnu")
