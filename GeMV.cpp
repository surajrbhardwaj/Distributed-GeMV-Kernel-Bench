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
#include <mpi.h>
#include <vector>
#include <cstring>
#include <fstream>
#include <string>

// TODO: dynamically calculate the number of matrices to preload based on how much ram we have in the system
// TODO: add in a debug build so that i can run thread, address, and UB sanitizers as well as debug and warning flags when compiling

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
    for(int i=0; i<NUM_MATRICES; i++) {
        int fd = open(("inputs/matrix" + std::to_string(i+1) + ".mat").c_str(), O_RDONLY);
        if (fd == -1) {std::cout << "error opening one of the input files" << std::endl; exit(-1);}
        void* mapped = mmap(nullptr, dataSize, PROT_READ, MAP_SHARED, fd, 0);
        if (mapped == MAP_FAILED) {
            perror("mmap failed");
            exit(-1);
        }
        close(fd);
        madvise(mapped, dataSize, MADV_SEQUENTIAL);
        mats[i] = reinterpret_cast<std::float16_t*>(static_cast<char*>(mapped));
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

    for(int i=0; i<NUM_MATRICES; i++) {
        workPerThread = yDim/numThreads;
        if(workPerThread==0) {workPerThread=1;}
        for(uint8_t j=0; j<numThreads; j++) {
            threadParams[j] = {xDim, yDim, (workPerThread*numThreads>yDim ? (j*workPerThread < yDim ? workPerThread : 0) : (j==numThreads-1 ? yDim-j*workPerThread : workPerThread)), j*workPerThread, mats[i], vec, result, ExecutionState::WORKING};
            threadParams[j].flag.notify_one();
        }

        for(uint8_t j=0; j<numThreads; j++) {
            while(threadParams[j].flag.load() == ExecutionState::WORKING) {
                threadParams[j].flag.wait(ExecutionState::WORKING);
            }
        }

        if(i+5<NUM_MATRICES) {
            madvise(mats[i+5], dataSize, MADV_WILLNEED);
        }
        madvise(mats[i], dataSize, MADV_DONTNEED);

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

// TODO: should actually only be calling madvise on the sections that we are actually going to access (not on the whole matrix)
void GeMV_TensorParallel(const uint8_t numThreads, const uint32_t xDim, const uint32_t yDim, std::float16_t**& mats, std::float16_t*& vec, std::float16_t*& result, int rank, const std::vector<int>& workPerProcess, const std::vector<int>& offsets) {
    std::ifstream vectorFile("inputs/vector.vec", std::ios::binary);
    vectorFile.read(reinterpret_cast<char*>(vec), sizeof(std::float16_t)*xDim);

    const size_t dataSize = sizeof(std::float16_t) * xDim * yDim;
    std::vector<void*> originalMatrixPointers(NUM_MATRICES, nullptr);
    for(int i=0; i<NUM_MATRICES; i++) {
        int fd = open(("inputs/matrix" + std::to_string(i+1) + ".mat").c_str(), O_RDONLY);
        if (fd == -1) {std::cout << "error opening one of the input files" << std::endl; exit(-1);}
        originalMatrixPointers[i] = mmap(nullptr, dataSize, PROT_READ, MAP_SHARED, fd, 0);
        if (originalMatrixPointers[i] == MAP_FAILED) {
            perror("mmap failed");
            exit(-1);
        }
        close(fd);
        madvise(originalMatrixPointers[i], dataSize, MADV_SEQUENTIAL);
        if(i<5) {madvise(originalMatrixPointers[i], dataSize, MADV_WILLNEED);}
        mats[i] = reinterpret_cast<std::float16_t*>(static_cast<char*>(originalMatrixPointers[i]) + offsets[rank]*xDim*sizeof(std::float16_t));
    }

    std::thread threads[numThreads];
    ThreadData threadParams[numThreads];
    for(uint8_t i=0; i<numThreads; i++) {
        threadParams[i] = {0, 0, 0, 0, nullptr, nullptr, nullptr, ExecutionState::FINISHED};
        threads[i] = std::thread(threadLoop, &threadParams[i]);
    }

    uint32_t workPerThread;

    for(int i=0; i<NUM_MATRICES; i++) {
        workPerThread = workPerProcess[rank]/numThreads;
        if(workPerThread==0) {workPerThread=1;}
        for(uint8_t j=0; j<numThreads; j++) {
            threadParams[j] = {xDim, yDim, (workPerThread*numThreads>workPerProcess[rank] ? (j*workPerThread < workPerProcess[rank] ? workPerThread : 0) : (j==numThreads-1 ? workPerProcess[rank]-j*workPerThread : workPerThread)), j*workPerThread, mats[i], vec, result+offsets[rank], ExecutionState::WORKING};
            threadParams[j].flag.notify_one();
        }

        for(uint8_t j=0; j<numThreads; j++) {
            while(threadParams[j].flag.load() == ExecutionState::WORKING) {
                threadParams[j].flag.wait(ExecutionState::WORKING);
            }
        }

        if(i+5<NUM_MATRICES) {
            madvise(originalMatrixPointers[i+5], dataSize, MADV_WILLNEED);
        }
        madvise(originalMatrixPointers[i], dataSize, MADV_DONTNEED);

        // gather the results on each node
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, result, workPerProcess.data(), offsets.data(), MPI_FLOAT16, MPI_COMM_WORLD);

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
    const size_t dataSize = sizeof(std::float16_t) * xDim * yDim;
    std::vector<void*> originalMatrixPointers(workPerProcess[rank], nullptr);
    for(int i=0; i<workPerProcess[rank]; i++) {
        int fd = open(("inputs/matrix" + std::to_string(i+offsets[rank]+1) + ".mat").c_str(), O_RDONLY);
        if (fd == -1) {std::cout << "error opening one of the input files" << std::endl; exit(-1);}
        originalMatrixPointers[i] = mmap(nullptr, dataSize, PROT_READ, MAP_SHARED | MAP_POPULATE, fd, 0);
        if (originalMatrixPointers[i] == MAP_FAILED) {
            perror("mmap failed");
            exit(-1);
        }
        close(fd);
        madvise(originalMatrixPointers[i], dataSize, MADV_SEQUENTIAL);
        mats[i] = reinterpret_cast<std::float16_t*>(originalMatrixPointers[i]);
    }

    if(rank==0) {
        std::ifstream vectorFile("inputs/vector.vec", std::ios::binary);
        vectorFile.read(reinterpret_cast<char*>(vec), sizeof(std::float16_t)*xDim);
    }
    else {
        MPI_Recv(vec, yDim, MPI_FLOAT16, rank-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    std::thread threads[numThreads];
    ThreadData threadParams[numThreads];
    for(uint8_t i=0; i<numThreads; i++) {
        threadParams[i] = {0, 0, 0, 0, nullptr, nullptr, nullptr, ExecutionState::FINISHED};
        threads[i] = std::thread(threadLoop, &threadParams[i]);
    }

    uint32_t workPerThread;

    for(int i=0; i<workPerProcess[rank]; i++) {
        workPerThread = yDim/numThreads;
        if(workPerThread==0) {workPerThread=1;}
        for(uint8_t j=0; j<numThreads; j++) {
            threadParams[j] = {xDim, yDim, (workPerThread*numThreads>yDim ? (j*workPerThread < yDim ? workPerThread : 0) : (j==numThreads-1 ? yDim-j*workPerThread : workPerThread)), j*workPerThread, mats[i], vec, result, ExecutionState::WORKING};
            threadParams[j].flag.notify_one();
        }

        for(uint8_t j=0; j<numThreads; j++) {
            while(threadParams[j].flag.load() == ExecutionState::WORKING) {
                threadParams[j].flag.wait(ExecutionState::WORKING);
            }
        }

        if(i+1!=workPerProcess[rank]) {
            std::float16_t* temp = vec;
            vec = result;
            result = temp;
        }
    }

    if(rank+1<workPerProcess.size()) {
        MPI_Ssend(result, static_cast<int>(yDim), MPI_FLOAT16, rank+1, 0, MPI_COMM_WORLD);
    }


    for(int i=0; i<workPerProcess[rank]; i++) {
        munmap(originalMatrixPointers[i], dataSize);
    }

    for(uint8_t i=0; i<numThreads; i++) {
        threadParams[i].flag.store(ExecutionState::TERMINATING);
        threadParams[i].flag.notify_one();
    }
    for(uint8_t i=0; i<numThreads; i++) {
        threads[i].join();
    }
}