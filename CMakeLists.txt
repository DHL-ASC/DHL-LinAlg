cmake_minimum_required(VERSION 3.20)
project(ASC_bla)

set (CMAKE_CXX_STANDARD 17)

include_directories(src)

add_executable (test_vector tests/test_vector.cc)
target_sources (test_vector PUBLIC src/vector.h src/expression.h)





