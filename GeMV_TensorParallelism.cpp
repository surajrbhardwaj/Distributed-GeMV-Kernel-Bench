#include "GeMV.hpp"
#include <mpi.h>
#include <vector>
#include <cstring>
#include <fstream>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <thread>
#include <iostream>
#include <string>

#define NUM_MATRICES 20

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

int main(int argc, char** argv) {
    int numProcesses;
    int rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcesses);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if(argc<3) {std::cerr << "not enough arguments (must provide matrix dimensions (num rows, num cols))" << std::endl; exit(-1);}

    uint32_t xDim=std::stoul(argv[1]); // num matrix rows
    uint32_t yDim=std::stoul(argv[2]); // num matrix cols

    uint8_t numThreads = std::min(std::thread::hardware_concurrency()==0 ? 1 : std::thread::hardware_concurrency(), yDim);
    numThreads = 2;

    std::vector<int> workPerProcess(numProcesses);
    std::vector<int> offsets(numProcesses);
    offsets[0] = 0;
    {
        int temp = yDim/numProcesses;
        workPerProcess[0] = temp;
        for(int i=1; i<numProcesses; i++) {
            offsets[i] = offsets[i-1]+workPerProcess[i-1];
            if(i==numProcesses-1) {workPerProcess[i] = yDim-offsets[i];}
            else {workPerProcess[i] = temp;}
        }
    }

    std::float16_t** mats = new std::float16_t*[NUM_MATRICES];
    std::float16_t* vec = new std::float16_t[xDim];

    std::float16_t* result = new std::float16_t[yDim];

    GeMV_TensorParallel(numThreads, xDim, yDim, mats, vec, result, rank, workPerProcess, offsets);

    // gather the results on each node
    // MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, result, workPerProcess.data(), offsets.data(), MPI_UINT16_T, MPI_COMM_WORLD);

    MPI_Finalize();

    if(rank==0) {
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
    }

    delete[] mats;
    delete[] vec;
    delete[] result;
}