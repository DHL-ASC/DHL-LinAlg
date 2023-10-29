cmake_minimum_required(VERSION 3.20)
project(ASC_bla)

set (CMAKE_CXX_STANDARD 17)


include_directories(src external_dependencies/HPC/src external_dependencies/HPC/concurrentqueue)
link_directories(src external_dependencies/HPC/src external_dependencies/HPC/concurrentqueue)

find_package(Python 3.8 COMPONENTS Interpreter Development REQUIRED)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE PYBIND11_DIR)
list(APPEND CMAKE_PREFIX_PATH "${PYBIND11_DIR}")
find_package(pybind11 CONFIG REQUIRED)


add_executable (test_vector tests/test_vector.cc)
add_executable (test_matrix tests/test_matrix.cc external_dependencies/HPC/src/taskmanager.cc external_dependencies/HPC/src/timer.cc)

pybind11_add_module(bla src/bind_bla.cpp external_dependencies/HPC/src/taskmanager.cc external_dependencies/HPC/src/timer.cc)

# target_link_libraries(bla PRIVATE pybind11::module external_dependencies/HPC/src/timer.h external_dependencies/HPC/src/taskmanager.h external_dependencies/HPC/concurrentqueue/concurrentqueue.h)

install (TARGETS bla DESTINATION dhllinalg)
install (FILES src/vector.h DESTINATION dhllinalg/include)
install (FILES src/matrix.h DESTINATION dhllinalg/include)
install (FILES external_dependencies/HPC/src/taskmanager.h DESTINATION dhllinalg/include)
install (FILES external_dependencies/HPC/src/timer.h DESTINATION dhllinalg/include)
# install (FILES external_dependencies/HPC/concurrentqueue/concurrentqueue.h DESTINATION dhllinalg/include)
