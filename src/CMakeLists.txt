# Core shared static library
add_library(GeMV_Core STATIC GeMV.cpp)

target_link_libraries(GeMV_Core PUBLIC GeMV_Headers MPI::MPI_CXX)

# Executables
add_executable(GeMV_Serial GeMV_Serial.cpp)
target_link_libraries(GeMV_Serial PRIVATE GeMV_Core)

add_executable(GeMV_TensorParallel GeMV_TensorParallel.cpp)
target_link_libraries(GeMV_TensorParallel PRIVATE GeMV_Core)

add_executable(GeMV_PipelineParallel GeMV_PipelineParallel.cpp)
target_link_libraries(GeMV_PipelineParallel PRIVATE GeMV_Core)

add_executable(CompareResults CompareResults.cpp)
