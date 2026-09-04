#include <stdfloat>
#include <cstdint>
#include <atomic>

#pragma once

/// @brief represents the current state of our executor threads. working means that it has a task to do, finished means it is done with the task and terminating means that we want to shutdown the program
enum class ExecutionState: uint8_t {
    WORKING,
    FINISHED,
    TERMINATING
};

/// @brief holds the data that is passed to each thread. everything is held as pointers so that we can alter what they point to easily and rerun the threads with those new inputs
struct ThreadData {
    uint32_t M;
    uint32_t N;
    uint32_t work;
    uint32_t offset;
    std::float16_t* matrixPtr{nullptr};
    std::float16_t* vectorPtr{nullptr};
    std::float16_t* resultPtr{nullptr};
    std::atomic<ExecutionState> flag{ExecutionState::FINISHED};
    
    ThreadData() {}
    ThreadData(uint32_t M, uint32_t N, uint32_t work, uint32_t offset, std::float16_t* matrixPtr, std::float16_t* vectorPtr, std::float16_t* resultPtr, ExecutionState flag): M(M), N(N), work(work), offset(offset), matrixPtr(matrixPtr), vectorPtr(vectorPtr), resultPtr(resultPtr) {
        this->flag.store(flag);
    }
    ThreadData& operator=(ThreadData&& other) noexcept {
        if(this != &other) {
            M = other.M;
            N = other.N;
            work = other.work;
            offset = other.offset;
            matrixPtr = other.matrixPtr;
            vectorPtr = other.vectorPtr;
            resultPtr = other.resultPtr;
            flag.store(other.flag.load());
        }
        return *this;
    }
};

/// @brief the kernel function that runs on each thread to do some amount of the matrix multiplication task
/// @param work the number of dot products to be calculated
/// @param offset the number row index after which we must calculate the dot products
/// @param N dimension of the column vector
/// @param matrix matrix represented as a 2d array of elements allocated as a contiguious block of memory
/// @param vector column vector represented as an array for cache coherence
/// @param results the output column vector represented as an array for cache coherence
/// @return void
void GeMV_kernel(const uint32_t work, const uint32_t offset, const uint32_t N, const std::float16_t* matrix, const std::float16_t* vector, std::float16_t* results);

/// @brief follows a producer consumer pattern blocking until there is work to be done then calling the GeMV_kernel function to do the work
/// @param data the parameters with which we want to run the GeMV_kernel
void threadLoop(ThreadData* data);

/// @brief runs GeMV across multiple threads on one node
/// @param numThreads the number of threads to use
/// @param M the height of the matrix to multiply with the vector
/// @param N the height of the vector to multiply with the matrix
/// @param mat a pointer to a flattened array representation of the matrix with dimensions MxN
/// @param vec a pointer to the flattened array reprentation of the vector with dimensions Nx1
/// @param result a pointer to the flattened array representation of the result vector with dimensions Mx1
void GeMV(uint8_t numThreads, uint32_t M, uint32_t N, std::float16_t* mat, std::float16_t* vec, std::float16_t* result);