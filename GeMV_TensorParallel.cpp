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

int main(int argc, char** argv) {
    int numProcesses;
    int rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcesses);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if(argc<3) {std::cerr << "not enough arguments (must provide matrix dimensions (num rows, num cols))" << std::endl; exit(-1);}

    uint32_t xDim=std::stoul(argv[1]); // num matrix cols
    uint32_t yDim=std::stoul(argv[2]); // num matrix rows

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

    MPI_Finalize();

    delete[] mats;
    delete[] vec;
    delete[] result;
}