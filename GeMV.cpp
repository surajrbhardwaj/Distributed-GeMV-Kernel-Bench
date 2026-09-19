#include "GeMV.hpp"
#include <iostream>
#include <fstream>
#include <stdfloat>
#include <algorithm>
#include <cstdint>
#include <thread>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>

#define NUM_MATRICES 20

std::ostream& operator<<(std::ostream& os, const ThreadData& td) {
    os << "ThreadData {\n"
       << "  xDim     " << td.xDim << ",\n"
       << "  yDim:    " << td.yDim << ",\n"
       << "  work:      " << td.work << ",\n"
       << "  offset:    " << td.offset << ",\n"
       << "  matrixPtr: " << static_cast<const void*>(td.matrixPtr) << ",\n"
       << "  vectorPtr: " << static_cast<const void*>(td.vectorPtr) << ",\n"
       << "  resultPtr: " << static_cast<const void*>(td.resultPtr) << ",\n"
       << "}";
    return os;
}

void GeMV_kernel(const uint32_t work, const uint32_t offset, const uint32_t xDim, const std::float16_t* __restrict__ matrix, const std::float16_t* __restrict__ vector, std::float16_t* __restrict__  results) {
    std::float32_t sum;
    uint32_t localOffset = offset * xDim;
    uint32_t endIndex = offset+work;
    for(uint32_t w=offset; w<endIndex; w++) {
        sum = 0.0f;
        for(uint32_t i=0; i<xDim; i++) {
            sum += static_cast<std::float32_t>(matrix[localOffset + i])*static_cast<std::float32_t>(vector[i]);
        }
        results[w] = static_cast<std::float16_t>(sum);
        localOffset += xDim;
    }
}

void threadLoop(ThreadData* data) {
    while(data->flag.load() != ExecutionState::TERMINATING) {
        while(data->flag.load()==ExecutionState::FINISHED) {data->flag.wait(ExecutionState::FINISHED);}
        if(data->flag.load()==ExecutionState::TERMINATING) {break;}
        GeMV_kernel(data->work, data->offset, data->xDim, data->matrixPtr, data->vectorPtr, data->resultPtr);
        data->flag.store(ExecutionState::FINISHED);
        data->flag.notify_one();
    }
}

void GeMV(const uint8_t numThreads, const uint32_t xDim, const uint32_t yDim, std::float16_t**& mats, std::float16_t*& vec, std::float16_t*& result) {
    std::ifstream vectorFile("inputs/vector.vec", std::ios::binary);
    vectorFile.read(reinterpret_cast<char*>(vec), sizeof(std::float16_t)*yDim);

    const size_t dataSize = sizeof(std::float16_t) * xDim * yDim;
    for(int i=1; i<=NUM_MATRICES; i++) {
        int fd = open(("inputs/matrix" + std::to_string(i) + ".mat").c_str(), O_RDONLY);
        if (fd == -1) {std::cout << "error opening one of the input files" << std::endl; exit(-1);}
        void* mapped = mmap(nullptr, dataSize, PROT_READ, MAP_SHARED, fd, 0);
        if (mapped == MAP_FAILED) {
            perror("mmap failed");
            exit(-1);
        }
        close(fd);
        madvise(mapped, dataSize, MADV_SEQUENTIAL);
        mats[i-1] = reinterpret_cast<std::float16_t*>(static_cast<char*>(mapped));
    }
    for(int i=0; i<5; i++) {
        madvise(mats[i], dataSize, MADV_WILLNEED);
    }

    std::thread threads[numThreads];
    ThreadData threadParams[numThreads];
    for(uint8_t i=0; i<numThreads; i++) {
        threadParams[i] = {0, 0, 0, 0, nullptr, nullptr, nullptr, ExecutionState::FINISHED};
        threads[i] = std::thread(threadLoop, &threadParams[i]);
    }

    uint32_t workPerThread;

    for(int i=1; i<=NUM_MATRICES; i++) {
        if(i%5==0 && i!=NUM_MATRICES) {
            for(int j=0; j<5; j++) {
                madvise(mats[i+j], dataSize, MADV_WILLNEED);
                madvise(mats[i-j-1], dataSize, MADV_DONTNEED);
            }
        }

        workPerThread = yDim/numThreads;
        if(workPerThread==0) {workPerThread=1;}
        for(uint8_t j=0; j<numThreads; j++) {
            threadParams[j] = {xDim, yDim, (workPerThread*numThreads>yDim ? (j*workPerThread < yDim ? workPerThread : 0) : (j==numThreads-1 ? yDim-j*workPerThread : workPerThread)), j*workPerThread, mats[i-1], vec, result, ExecutionState::WORKING};
            threadParams[j].flag.notify_one();
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

    for(int i=1; i<=NUM_MATRICES; i++) {
        munmap(mats[i-1], dataSize);
    }

    for(uint8_t i=0; i<numThreads; i++) {
        threadParams[i].flag.store(ExecutionState::TERMINATING);
        threadParams[i].flag.notify_one();
    }
    for(uint8_t i=0; i<numThreads; i++) {
        threads[i].join();
    }

    if(NUM_MATRICES%2==0) {
        std::float16_t* temp = vec;
        vec = result;
        result = temp;
    }
}


/*
#include <cstring>
#include <string>
int main(int argc, char* argv[]) {
    if(argc<3) {std::cerr << "not enough arguments (must provide matrix dimensions (num rows, num cols))" << std::endl; exit(-1);}

    uint32_t xDim=std::stoul(argv[1]); // num matrix cols
    uint32_t yDim=std::stoul(argv[2]); // num matrix rows

    uint8_t numThreads = std::min(std::thread::hardware_concurrency()==0 ? 1 : std::thread::hardware_concurrency(), yDim);

    std::float16_t** mats = new std::float16_t*[NUM_MATRICES];
    std::float16_t* vec = new std::float16_t[xDim];

    std::float16_t* result = new std::float16_t[yDim];

    GeMV(numThreads, xDim, yDim, mats, vec, result);

    std::ofstream out("output.vec", std::ios::binary | std::ios::out | std::ios::trunc);
    if (!out.is_open()) {
        std::cerr << "Failed to open output.vec for writing.\n";
        return -1;
    }
    std::size_t bytes_to_write = yDim * sizeof(std::float16_t);
    out.write(reinterpret_cast<const char*>(result), bytes_to_write);
    if(!out) {
        std::cerr << "Error occurred while writing to output.vec.\n";
    }

    delete[] mats;
    delete[] vec;
    delete[] result;
}
*/