cmake_minimum_required(VERSION 3.20)

project(GeMV_Research LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

add_compile_options(
    -O3
    -march=native
    -ffast-math
    -flto
    -DNDEBUG
)

find_package(MPI REQUIRED)

add_subdirectory(include)
add_subdirectory(src)
