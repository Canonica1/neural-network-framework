cmake_minimum_required(VERSION 3.16)
project(neural_network_framework LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Eigen3 QUIET)
add_library(nn INTERFACE)
target_include_directories(nn INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

set(LOCAL_EIGEN_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/eigen)
if (TARGET Eigen3::Eigen)
    target_link_libraries(nn INTERFACE Eigen3::Eigen)
elseif (EXISTS ${LOCAL_EIGEN_DIR}/Eigen/Dense)
    target_include_directories(nn SYSTEM INTERFACE ${LOCAL_EIGEN_DIR})
else()
    message(FATAL_ERROR "Eigen3 is required. Install Eigen3 or put Eigen headers into third_party/eigen")
endif()

if (MSVC)
    target_compile_options(nn INTERFACE /W4)
else()
    target_compile_options(nn INTERFACE -Wall -Wextra -Wpedantic)
endif()

add_executable(demo examples/demo.cpp)
target_link_libraries(demo PRIVATE nn)

enable_testing()
add_subdirectory(tests)
