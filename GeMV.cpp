#include "GeMV.hpp"
#include <cstdint>
#include <thread>

void GeMV_kernel(const uint32_t work, const uint32_t offset, const uint32_t N, const std::float16_t* matrix, const std::float16_t* vector, std::float16_t* results) {
    std::float32_t sum;
    for(uint32_t w=0; w<work; w++) {
        sum = 0.0f;
        for(uint32_t i=0; i<N; i++) {
            sum += static_cast<std::float32_t>(matrix[(offset+w)*N + i])*static_cast<std::float32_t>(vector[i]);
        }
        results[offset+w] = static_cast<std::float16_t>(sum);
    }
}

void threadLoop(ThreadData* data) {
    while(data->flag.load() != ExecutionState::TERMINATING) {
        while(data->flag.load()==ExecutionState::FINISHED) {data->flag.wait(ExecutionState::FINISHED);}
        if(data->flag.load()==ExecutionState::TERMINATING) {break;}
        GeMV_kernel(data->work, data->offset, data->N, data->matrixPtr, data->vectorPtr, data->resultPtr);
        data->flag.store(ExecutionState::FINISHED);
        data->flag.notify_one();
    }
}

void GeMV(uint8_t numThreads, uint32_t M, uint32_t N, std::float16_t* mat, std::float16_t* vec, std::float16_t* result) {
    std::thread threads[numThreads];
    ThreadData threadParams[numThreads];
    for(uint8_t i=0; i<numThreads; i++) {
        threadParams[i] = {0, 0, 0, 0, nullptr, nullptr, nullptr, ExecutionState::FINISHED};
        threads[i] = std::thread(threadLoop, &threadParams[i]);
    }

    uint32_t workPerThread = M/numThreads;
    for(uint8_t i=0; i<numThreads; i++) {
        threadParams[i] = {M, N, i==numThreads-1 ? M-i*workPerThread : workPerThread, i*workPerThread, mat, vec, result, ExecutionState::WORKING};
        threadParams[i].flag.notify_one();
    }

    for(uint8_t i=0; i<numThreads; i++) {
        threadParams[i].flag.wait(ExecutionState::WORKING);
    }
    
    for(uint8_t i=0; i<numThreads; i++) {
        threadParams[i].flag.store(ExecutionState::TERMINATING);
        threadParams[i].flag.notify_one();
    }
    for(uint8_t i=0; i<numThreads; i++) {
        threads[i].join();
    }
}

/*
#include <iostream>
#include <cstring>
int main() {
    uint8_t numThreads = std::thread::hardware_concurrency()==0 ? 1 : std::thread::hardware_concurrency();

    const uint32_t M = 2; // num matrix rows
    const uint32_t N = 3; // num matrix cols
    std::float16_t* mat = new std::float16_t[M*N]{0.5f16, -0.2f16, 0.1f16, 1.5f16, 0.0f16, -0.5f16};
    std::float16_t* vec = new std::float16_t[N]{1.0f16, 2.0f16, -1.0f16};

    std::float16_t* result = new std::float16_t[M];
    std::memset(result, 0x00, M*sizeof(std::float16_t));

    GeMV(numThreads, M, N, mat, vec, result);

    for(uint8_t i=0; i<M; i++) {
        std::cout << result[i] << std::endl;
    }

    delete[] mat;
    delete[] vec;
    delete[] result;
}
*/