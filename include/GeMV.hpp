#include <mpi.h>
#include <vector>
#include <stdfloat>
#include <cstdint>
#include <atomic>
#include <ostream>

#pragma once

#define NUM_MATRICES 20

#ifndef MPI_FLOAT16
#define MPI_FLOAT16 MPI_UINT16_T
#endif

/// @brief This is meant to create a visual span in perfetto to represent compute times
void trace_begin(const char* name);

/// @brief This is meant to end a visual span in perfetto to represent compute times
void trace_end();

/// @brief represents the current state of our executor threads. working means that it has a task to do, finished means it is done with the task and terminating means that we want to shutdown the program
enum class ExecutionState: uint8_t {
    WORKING,
    FINISHED,
    TERMINATING
};

/// @brief holds the data that is passed to each thread. everything is held as pointers so that we can alter what they point to easily and rerun the threads with those new inputs
struct ThreadData {
    uint32_t xDim;
    uint32_t yDim;
    uint32_t work;
    uint32_t offset;
    std::float16_t* matrixPtr{nullptr};
    std::float16_t* vectorPtr{nullptr};
    std::float16_t* resultPtr{nullptr};
    std::atomic<ExecutionState> flag{ExecutionState::FINISHED};

    ThreadData() {}
    ThreadData(uint32_t xDim, uint32_t yDim, uint32_t work, uint32_t offset, std::float16_t* matrixPtr, std::float16_t* vectorPtr, std::float16_t* resultPtr, ExecutionState flag): xDim(xDim), yDim(yDim), work(work), offset(offset), matrixPtr(matrixPtr), vectorPtr(vectorPtr), resultPtr(resultPtr) {
        this->flag.store(flag);
    }
    ThreadData& operator=(ThreadData&& other) noexcept {
        if(this != &other) {
            xDim = other.xDim;
            yDim = other.yDim;
            work = other.work;
            offset = other.offset;
            matrixPtr = other.matrixPtr;
            vectorPtr = other.vectorPtr;
            resultPtr = other.resultPtr;
            flag.store(other.flag.load(std::memory_order_relaxed), std::memory_order_release);
        }
        return *this;
    }
};

/// @brief Debug helper function to print out ThreadData struct
std::ostream& operator<<(std::ostream& os, const ThreadData& td);

/// @brief Reporting helper function to convert from struct timeval to seconds
double timeval_to_sec(const struct timeval& tv);

/// @brief the kernel function that runs on each thread to do some amount of the matrix multiplication task
/// @param work the number of dot products to be calculated
/// @param offset the number row index after which we must calculate the dot products
/// @param xDim dimension of the column vector
/// @param matrix matrix represented as a 2d array of elements allocated as a contiguious block of memory
/// @param vector column vector represented as an array for cache coherence
/// @param results the output column vector represented as an array for cache coherence
/// @return void
void GeMV_kernel(const uint32_t work, const uint32_t offset, const uint32_t xDim, const std::float16_t* matrix, const std::float16_t* vector, std::float16_t* results);

/// @brief follows a producer consumer pattern blocking until there is work to be done then calling the GeMV_kernel function to do the work
/// @param data the parameters with which we want to run the GeMV_kernel
void threadLoop(ThreadData* data);

/// @brief runs GeMV across multiple threads on one node
/// @param numThreads the number of threads to use
/// @param xDim the height of the vector to multiply with the matrix
/// @param yDim the height of the matrix to multiply with the vector
/// @param mat a pointer to a flattened array representation of the matrix with dimensions yDim x xDim
/// @param vec a pointer to the flattened array reprentation of the vector with dimensions xDim x 1
/// @param result a pointer to the flattened array representation of the result vector with dimensions yDim x 1
void GeMV_Serial(const uint8_t numThreads, const uint32_t xDim, const uint32_t yDim, std::float16_t**& mats, std::float16_t*& vec, std::float16_t*& result);

/// @brief runs GeMV across multiple threads on multiple nodes following a Tensor Parallel distribution approach
/// @param numThreads the number of threads to use
/// @param xDim the height of the vector to multiply with the matrix
/// @param yDim the height of the matrix to multiply with the vector
/// @param mat a pointer to a flattened array representation of the matrix with dimensions yDim x xDim
/// @param vec a pointer to the flattened array reprentation of the vector with dimensions xDim x 1
/// @param result a pointer to the flattened array representation of the result vector with dimensions yDim x 1
/// @param rank a sequential process id (zero indexed)
/// @param workPerProcess a vector of size numProcesses where each entry denotes the number of rows that process is to compute dot products for
/// @param offsets a vector of size numProcesses where each entry denotes the starting row number (inclusive) of that process to compute dot products from
void GeMV_TensorParallel(const uint8_t numThreads, const uint32_t xDim, const uint32_t yDim, std::float16_t**& mats, std::float16_t*& vec, std::float16_t*& result, int rank, const std::vector<int>& workPerProcess, const std::vector<int>& offsets);

/// @brief runs GeMV across multiple threads on multiple nodes following a Pipeline Parallel distribution approach
/// @param numThreads the number of threads to use
/// @param xDim the height of the vector to multiply with the matrix
/// @param yDim the height of the matrix to multiply with the vector
/// @param mat a pointer to a flattened array representation of the matrix with dimensions yDim x xDim
/// @param vec a pointer to the flattened array reprentation of the vector with dimensions xDim x 1
/// @param result a pointer to the flattened array representation of the result vector with dimensions yDim x 1
/// @param rank a sequential process id (zero indexed)
/// @param workPerProcess a vector of size numProcesses where each entry denotes the number of rows that process is to compute dot products for
/// @param offsets a vector of size numProcesses where each entry denotes the starting row number (inclusive) of that process to compute dot products from
void GeMV_PipelineParallel(const uint8_t numThreads, const uint32_t xDim, const uint32_t yDim, std::float16_t**& mats, std::float16_t*& vec, std::float16_t*& result, int rank, const std::vector<int>& workPerProcess, const std::vector<int>& offsets);
