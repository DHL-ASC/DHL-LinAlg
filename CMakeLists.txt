cmake_minimum_required(VERSION 3.20)
project(ASC_bla)

set (CMAKE_CXX_STANDARD 17)

include(CheckCXXCompilerFlag)

CHECK_CXX_COMPILER_FLAG("-O3" COMPILER_OPT_OTHREE_SUPPORTED)
if(COMPILER_OPT_OTHREE_SUPPORTED)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")
endif()

CHECK_CXX_COMPILER_FLAG("-march=native" COMPILER_OPT_ARCH_NATIVE_SUPPORTED)
if (COMPILER_OPT_ARCH_NATIVE_SUPPORTED)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
else()
  CHECK_CXX_COMPILER_FLAG("/arch:AVX2" COMPILER_OPT_ARCH_AVXTWO_SUPPORTED)
  if(COMPILER_OPT_ARCH_AVXTWO_SUPPORTED)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /arch:AVX2")
  endif()
endif()

include_directories(src external_dependencies/HPC/src external_dependencies/HPC/concurrentqueue external_dependencies/HPC/taskflow)
link_directories(src external_dependencies/HPC/src external_dependencies/HPC/concurrentqueue external_dependencies/HPC/taskflow)

find_package(Python 3.8 COMPONENTS Interpreter Development REQUIRED)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE PYBIND11_DIR)
list(APPEND CMAKE_PREFIX_PATH "${PYBIND11_DIR}")
find_package(pybind11 CONFIG REQUIRED)


add_executable (test_vector tests/test_vector.cc external_dependencies/HPC/src/simd.h)
add_executable (test_matrix tests/test_matrix.cc external_dependencies/HPC/src/parallelcomputingtf.cc external_dependencies/HPC/src/timer.cc external_dependencies/HPC/src/simd.h)


find_package(LAPACK REQUIRED)
add_executable (test_lapack tests/test_lapack.cc)
target_link_libraries (test_lapack PUBLIC ${LAPACK_LIBRARIES})

# target_sources(test_vector PUBLIC external_dependencies/HPC/src/simd.h)
# target_sources(test_matrix PUBLIC external_dependencies/HPC/src/simd.h)

pybind11_add_module(bla src/bind_bla.cpp external_dependencies/HPC/src/parallelcomputingtf.cc external_dependencies/HPC/src/timer.cc external_dependencies/HPC/src/simd.h)

# target_sources(bla PUBLIC external_dependencies/HPC/src/simd.h)
# target_link_libraries(bla PRIVATE pybind11::module external_dependencies/HPC/src/timer.h external_dependencies/HPC/src/taskmanager.h external_dependencies/HPC/concurrentqueue/concurrentqueue.h)

install (TARGETS bla DESTINATION dhllinalg)
install (FILES src/vector.h DESTINATION dhllinalg/include)
install (FILES src/matrix.h DESTINATION dhllinalg/include)
install (FILES external_dependencies/HPC/src/taskmanager.h DESTINATION dhllinalg/include)
install (FILES external_dependencies/HPC/src/timer.h DESTINATION dhllinalg/include)
# install (FILES external_dependencies/HPC/concurrentqueue/concurrentqueue.h DESTINATION dhllinalg/include)
