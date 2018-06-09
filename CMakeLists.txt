cmake_minimum_required(VERSION 3.10)
project(metrics)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -Ofast -fopenmp -Wrestrict")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Wall")

include_directories(/usr/local/include)

link_directories(/usr/local/Cellar/libpng/1.6.34/lib)

add_executable(metrics main.cpp matrix.h)

target_link_libraries(metrics png16)
