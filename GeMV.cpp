#include "GeMV.hpp"
#include <iostream>
#include <fstream>
#include <stdfloat>
#include <cstdint>
#include <thread>

void GeMV_kernel(const uint32_t work, const uint32_t offset, const uint32_t N, const std::float16_t* __restrict__ matrix, const std::float16_t* __restrict__ vector, std::float16_t* __restrict__  results) {
    std::float32_t sum;
    uint32_t localOffset = offset * N;
    uint32_t endIndex = offset+work;
    for(uint32_t w=offset; w<endIndex; w++) {
        sum = 0.0f;
        for(uint32_t i=0; i<N; i++) {
            sum += static_cast<std::float32_t>(matrix[localOffset + i])*static_cast<std::float32_t>(vector[i]);
        }
        results[w] = static_cast<std::float16_t>(sum);
        localOffset += N;
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

void GeMV(const uint8_t numThreads, uint32_t M, uint32_t N, std::float16_t*& mat, std::float16_t*& vec, std::float16_t*& result) {
    std::ifstream vectorFile("vector.vec", std::ios::binary);
    {
        uint32_t vectorN;
        vectorFile.read(reinterpret_cast<char*>(&vectorN), sizeof(uint32_t));
        if(N!=vectorN) {std::cerr << "vector and matrix dimensions are not compatible" << std::endl; exit(-1);}
        vectorFile.read(reinterpret_cast<char*>(vec), sizeof(std::float16_t)*N);
    }

    std::thread threads[numThreads];
    ThreadData threadParams[numThreads];
    for(uint8_t i=0; i<numThreads; i++) {
        threadParams[i] = {0, 0, 0, 0, nullptr, nullptr, nullptr, ExecutionState::FINISHED};
        threads[i] = std::thread(threadLoop, &threadParams[i]);
    }
    
    uint32_t workPerThread;
    std::ifstream matrixFile;

    for(int i=1; i<=20; i++) {
        // read in the matrix data
        matrixFile = std::ifstream("matrix" + std::to_string(i) + ".mat", std::ios::binary);
        matrixFile.seekg(8, std::ios::beg);
        matrixFile.read(reinterpret_cast<char*>(mat), sizeof(std::float16_t)*M*N);

        workPerThread = M/numThreads;
        for(uint8_t i=0; i<numThreads; i++) {
            threadParams[i] = {M, N, i==numThreads-1 ? M-i*workPerThread : workPerThread, i*workPerThread, mat, vec, result, ExecutionState::WORKING};
            threadParams[i].flag.notify_one();
        }

        for(uint8_t i=0; i<numThreads; i++) {
            while(threadParams[i].flag.load() == ExecutionState::WORKING) {
                threadParams[i].flag.wait(ExecutionState::WORKING);
            }
        }

        std::float16_t* temp = vec;
        vec = result;
        result = temp;
    }

    for(uint8_t i=0; i<numThreads; i++) {
        threadParams[i].flag.store(ExecutionState::TERMINATING);
        threadParams[i].flag.notify_one();
    }
    for(uint8_t i=0; i<numThreads; i++) {
        threads[i].join();
    }
}


#include <cstring>
#include <string>
int main(int argc, char* argv[]) {
    if(argc<3) {std::cerr << "not enough arguments (must provide matrix dimensions (num rows, num cols))" << std::endl; exit(-1);}

    uint32_t M=std::stoul(argv[1]); // num matrix rows
    uint32_t N=std::stoul(argv[2]); // num matrix cols

    uint8_t numThreads = std::min(std::thread::hardware_concurrency()==0 ? 1 : std::thread::hardware_concurrency(), M);
    numThreads = 6;

    std::float16_t* mat = new std::float16_t[M*N];
    std::float16_t* vec = new std::float16_t[N];

    std::float16_t* result = new std::float16_t[M];
    
    GeMV(numThreads, M, N, mat, vec, result);
    
    delete[] mat;
    delete[] vec;
    delete[] result;
}
