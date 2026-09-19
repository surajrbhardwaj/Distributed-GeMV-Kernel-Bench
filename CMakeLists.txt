cmake_minimum_required(VERSION 3.20)

project(GeMV_Research LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_compile_options(
    -O3
    -march=native
    -ffast-math
    -flto
    -DNDEBUG
)

find_package(MPI REQUIRED)

# Compile shared implementations into an internal static library
add_library(GeMV_Core STATIC GeMV.cpp)
target_link_libraries(GeMV_Core PUBLIC MPI::MPI_CXX)

add_executable(GeMV_Serial GeMV_Serial.cpp)
target_link_libraries(GeMV_Serial PRIVATE GeMV_Core)

add_executable(GeMV_TensorParallel GeMV_TensorParallel.cpp)
target_link_libraries(GeMV_TensorParallel PRIVATE GeMV_Core MPI::MPI_CXX)

add_executable(GeMV_PipelineParallel GeMV_PipelineParallel.cpp)
target_link_libraries(GeMV_PipelineParallel PRIVATE GeMV_Core MPI::MPI_CXX)

add_executable(CompareResults CompareResults.cpp)