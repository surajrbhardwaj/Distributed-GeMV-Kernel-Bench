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

void GeMV_Serial(const uint8_t numThreads, const uint32_t xDim, const uint32_t yDim, std::float16_t**& mats, std::float16_t*& vec, std::float16_t*& result) {
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

void GeMV_TensorParallel(const uint8_t numThreads, const uint32_t xDim, const uint32_t yDim, std::float16_t**& mats, std::float16_t*& vec, std::float16_t*& result, int rank, const std::vector<int>& workPerProcess, const std::vector<int>& offsets) {
    std::ifstream vectorFile("inputs/vector.vec", std::ios::binary);
    vectorFile.read(reinterpret_cast<char*>(vec), sizeof(std::float16_t)*xDim);

    const size_t dataSize = sizeof(std::float16_t) * xDim * yDim;
    std::vector<void*> originalMatrixPointers(NUM_MATRICES, nullptr);
    for(int i=1; i<=NUM_MATRICES; i++) {
        int fd = open(("inputs/matrix" + std::to_string(i) + ".mat").c_str(), O_RDONLY);
        if (fd == -1) {std::cout << "error opening one of the input files" << std::endl; exit(-1);}
        originalMatrixPointers[i-1] = mmap(nullptr, dataSize, PROT_READ, MAP_SHARED, fd, 0);
        if (originalMatrixPointers[i-1] == MAP_FAILED) {
            perror("mmap failed");
            exit(-1);
        }
        close(fd);
        madvise(originalMatrixPointers[i-1], dataSize, MADV_SEQUENTIAL);
        mats[i-1] = reinterpret_cast<std::float16_t*>(static_cast<char*>(originalMatrixPointers[i-1]) + offsets[rank]*xDim*sizeof(std::float16_t));
    }
    for(int i=0; i<5; i++) {
        madvise(originalMatrixPointers[i], dataSize, MADV_WILLNEED);
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
                madvise(originalMatrixPointers[i+j], dataSize, MADV_WILLNEED);
                madvise(originalMatrixPointers[i-j-1], dataSize, MADV_DONTNEED);
            }
        }

        workPerThread = workPerProcess[rank]/numThreads;
        if(workPerThread==0) {workPerThread=1;}
        for(uint8_t j=0; j<numThreads; j++) {
            threadParams[j] = {xDim, yDim, (workPerThread*numThreads>workPerProcess[rank] ? (j*workPerThread < workPerProcess[rank] ? workPerThread : 0) : (j==numThreads-1 ? workPerProcess[rank]-j*workPerThread : workPerThread)), j*workPerThread, mats[i-1], vec, result+offsets[rank], ExecutionState::WORKING};
            threadParams[j].flag.notify_one();
        }

        for(uint8_t i=0; i<numThreads; i++) {
            while(threadParams[i].flag.load() == ExecutionState::WORKING) {
                threadParams[i].flag.wait(ExecutionState::WORKING);
            }
        }

        // gather the results on each node
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, result, workPerProcess.data(), offsets.data(), MPI_UINT16_T, MPI_COMM_WORLD);

        std::float16_t* temp = vec;
        vec = result;
        result = temp;
    }

    for(int i=1; i<=NUM_MATRICES; i++) {
        munmap(originalMatrixPointers[i-1], dataSize);
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

void GeMV_PipelineParallel(const uint8_t numThreads, const uint32_t xDim, const uint32_t yDim, std::float16_t**& mats, std::float16_t*& vec, std::float16_t*& result, int rank, const std::vector<int>& workPerProcess, const std::vector<int>& offsets) {
    // TODO: be sure to pre populate the memory with the matrices
    const size_t dataSize = sizeof(std::float16_t) * xDim * yDim;
    std::vector<void*> originalMatrixPointers(NUM_MATRICES, nullptr);
    for(int i=1; i<=NUM_MATRICES; i++) {
        int fd = open(("inputs/matrix" + std::to_string(i) + ".mat").c_str(), O_RDONLY);
        if (fd == -1) {std::cout << "error opening one of the input files" << std::endl; exit(-1);}
        originalMatrixPointers[i-1] = mmap(nullptr, dataSize, PROT_READ, MAP_SHARED, fd, 0);
        if (originalMatrixPointers[i-1] == MAP_FAILED) {
            perror("mmap failed");
            exit(-1);
        }
        close(fd);
        madvise(originalMatrixPointers[i-1], dataSize, MADV_SEQUENTIAL);
        mats[i-1] = reinterpret_cast<std::float16_t*>(static_cast<char*>(originalMatrixPointers[i-1]) + offsets[rank]*xDim*sizeof(std::float16_t));
    }
    for(int i=0; i<5; i++) {
        madvise(originalMatrixPointers[i], dataSize, MADV_WILLNEED);
    }

    if(rank==0) {
        std::ifstream vectorFile("inputs/vector.vec", std::ios::binary);
        vectorFile.read(reinterpret_cast<char*>(vec), sizeof(std::float16_t)*xDim);
    }
    else {
        // TODO: mpi recv the vector file from process with rank= rank-1
    }

    std::thread threads[numThreads];
    ThreadData threadParams[numThreads];
    for(uint8_t i=0; i<numThreads; i++) {
        threadParams[i] = {0, 0, 0, 0, nullptr, nullptr, nullptr, ExecutionState::FINISHED};
        threads[i] = std::thread(threadLoop, &threadParams[i]);
    }

    uint32_t workPerThread;

    for(int i=offsets[rank]; i<offsets[rank] + workPerProcess[rank]; i++) {
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

    if(NUM_MATRICES%2==0) {
        std::float16_t* temp = vec;
        vec = result;
        result = temp;
    }

    if(rank+1<workPerProcess.size()) {
        // TODO: mpi send fully calculated vector result to process rank+1
    }


    for(int i=1; i<=NUM_MATRICES; i++) {
        munmap(originalMatrixPointers[i-1], dataSize);
    }

    for(uint8_t i=0; i<numThreads; i++) {
        threadParams[i].flag.store(ExecutionState::TERMINATING);
        threadParams[i].flag.notify_one();
    }
    for(uint8_t i=0; i<numThreads; i++) {
        threads[i].join();
    }
}