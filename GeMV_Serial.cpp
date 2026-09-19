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

    GeMV_Serial(numThreads, xDim, yDim, mats, vec, result);

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