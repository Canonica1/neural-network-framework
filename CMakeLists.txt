cmake_minimum_required(VERSION 3.16)
project(neural_network_framework LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Eigen3 REQUIRED)
find_package(folly CONFIG REQUIRED)
add_subdirectory(csv-parser)
add_library(nn INTERFACE)
target_include_directories(nn INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(nn INTERFACE Eigen3::Eigen Folly::folly)
target_compile_options(nn INTERFACE -Wall -Wextra -Wpedantic)

add_executable(demo examples/demo.cpp)
target_link_libraries(demo PRIVATE nn)
target_link_libraries(demo PRIVATE csv)

enable_testing()
add_subdirectory(tests)

