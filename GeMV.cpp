#include "GeMV.hpp"
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
        while(threadParams[i].flag.load() == ExecutionState::WORKING) {
            threadParams[i].flag.wait(ExecutionState::WORKING);
        }
    }
    
    for(uint8_t i=0; i<numThreads; i++) {
        threadParams[i].flag.store(ExecutionState::TERMINATING);
        threadParams[i].flag.notify_one();
    }
    for(uint8_t i=0; i<numThreads; i++) {
        threads[i].join();
    }
}


#include <iostream>
#include <cstring>
#include <fstream>
int main(int argc, char* argv[]) {
    if(argc<3) {std::cerr << "not enough arguments (must provide matrix file and vector file)" << std::endl; exit(-1);}

    std::ifstream vectorFile(argv[1], std::ios::binary);
    std::ifstream matrixFile(argv[2], std::ios::binary);

    uint32_t M; // num matrix rows
    uint32_t N; // num matrix cols
    matrixFile.read(reinterpret_cast<char*>(&M), sizeof(uint32_t));
    matrixFile.read(reinterpret_cast<char*>(&N), sizeof(uint32_t));
    {
        uint32_t vectorN;
        vectorFile.read(reinterpret_cast<char*>(&vectorN), sizeof(uint32_t));
        if(N!=vectorN) {std::cerr << "vector and matrix dimensions are not compatible" << std::endl; exit(-1);}
    }

    std::float16_t* mat = new std::float16_t[M*N];
    std::float16_t* vec = new std::float16_t[N];
    matrixFile.read(reinterpret_cast<char*>(mat), sizeof(std::float16_t)*M*N);
    vectorFile.read(reinterpret_cast<char*>(vec), sizeof(std::float16_t)*N);

    std::float16_t* result = new std::float16_t[M]{};
    std::memset(result, 0x00, M*sizeof(std::float16_t));

    uint8_t numThreads = std::min(std::thread::hardware_concurrency()==0 ? 1 : std::thread::hardware_concurrency(), M);

    GeMV(numThreads, M, N, mat, vec, result);

    for(uint32_t i=0; i<M; i++) {
        std::cout << result[i] << std::endl;
    }
    // TODO: use this output to verify the answers for smaller matrices

    delete[] mat;
    delete[] vec;
    delete[] result;
}
