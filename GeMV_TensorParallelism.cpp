#include "GeMV.hpp"
#include <mpi.h>
#include <vector>
#include <cstring>

#include <iostream>
int main(int argc, char** argv) {
    int numProcesses;
    int rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcesses);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    const uint32_t M = 10; // num matrix rows
    const uint32_t N = 10; // num matrix cols

    // TODO: matrix should be loaded from file
    // TODO: vec should be initialized randomly on the master then distributed to the slaves
    std::float16_t* mat = new std::float16_t[M*N]{ //{0.5f16, -0.2f16, 0.1f16, 1.5f16, 0.0f16, -0.5f16};
        1.0f16,  0.5f16, -1.0f16,  2.0f16,  0.0f16,  1.5f16, -0.5f16,  0.0f16,  1.0f16, -2.0f16,
        0.0f16,  1.0f16,  2.0f16, -1.0f16,  0.5f16,  0.0f16,  1.0f16, -0.5f16,  0.0f16,  1.0f16,
       -1.0f16,  0.0f16,  1.0f16,  0.5f16, -0.5f16,  2.0f16,  0.0f16,  1.0f16, -1.0f16,  0.0f16,
        0.5f16, -1.0f16,  0.0f16,  1.0f16,  1.5f16, -0.5f16,  2.0f16,  0.0f16,  0.5f16, -1.0f16,
        2.0f16,  0.0f16, -0.5f16,  0.0f16,  1.0f16,  1.0f16, -1.0f16,  0.5f16,  0.0f16,  2.0f16,
        0.0f16,  1.5f16,  0.5f16, -2.0f16,  0.0f16,  1.0f16,  0.0f16, -1.0f16,  2.0f16,  0.5f16,
       -0.5f16,  1.0f16,  0.0f16,  1.5f16, -1.0f16,  0.0f16,  0.5f16,  2.0f16, -0.5f16,  1.0f16,
        1.0f16, -0.5f16,  2.0f16,  0.0f16,  0.5f16, -1.0f16,  1.0f16,  0.0f16,  1.5f16, -0.5f16,
        0.0f16,  2.0f16, -1.0f16,  0.5f16,  0.0f16,  1.5f16, -0.5f16,  1.0f16,  0.0f16,  1.0f16,
       -1.5f16,  0.5f16,  1.0f16, -0.5f16,  2.0f16,  0.0f16,  1.5f16, -1.0f16,  0.5f16,  0.0f16
    };
    std::float16_t* vec = new std::float16_t[N]{ //{1.0f16, 2.0f16, -1.0f16};
        2.0f16,
       -1.0f16,
        0.5f16,
        1.0f16,
       -2.0f16,
        0.0f16,
        1.5f16,
        3.0f16,
       -0.5f16,
        2.0f16
    };

    std::float16_t* result = new std::float16_t[M];
    memset(result, 0x00, M*sizeof(std::float16_t));

    std::vector<int> workPerProcess(numProcesses);
    std::vector<int> offsets(numProcesses);
    offsets[0] = 0;
    workPerProcess[0] = M/numProcesses;
    for(int i=1; i<numProcesses; i++) {
        offsets[i] = offsets[i-1]+workPerProcess[i-1];
        if(i==numProcesses-1) {workPerProcess[i] = M-offsets[i];}
        else {workPerProcess[i] = M/numProcesses;}
    }

    GeMV((uint8_t)2, (uint32_t)workPerProcess[rank], (uint32_t)N, mat+offsets[rank]*N, vec, result+offsets[rank]);
    
    // gather the results on each node
    MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, result, workPerProcess.data(), offsets.data(), MPI_UINT16_T, MPI_COMM_WORLD);

    MPI_Finalize();
    
    if(rank==1) {
        for(uint8_t i=0; i<M; i++) {
            std::cout << result[i] << std::endl;
        }
    }

    delete[] mat;
    delete[] vec;
    delete[] result;
}