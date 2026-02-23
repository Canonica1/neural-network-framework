cmake_minimum_required(VERSION 3.16)
project(neural_network_framework LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_compile_options(-Wall -Wextra -Wpedantic)

include(FetchContent)
FetchContent_Declare(
    folly
    GIT_REPOSITORY https://github.com/facebook/folly.git
    GIT_TAG v2024.06.03.00
)
FetchContent_MakeAvailable(folly)

FetchContent_Declare(
    eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 3.4.0
)
FetchContent_MakeAvailable(eigen)

add_library(nnframework
    block/Block.cpp
)

target_include_directories(nnframework PUBLIC
    ${CMAKE_SOURCE_DIR}/block
    ${eigen_SOURCE_DIR} 
)

target_link_libraries(nnframework PUBLIC
    folly::folly
    Eigen3::Eigen
)

add_executable(main_app
    main.cpp
)

target_link_libraries(main_app PRIVATE nnframework)

add_executable(tests
    tests.cpp
)

target_link_libraries(tests PRIVATE nnframework)
target_link_libraries(nnframework PRIVATE folly)