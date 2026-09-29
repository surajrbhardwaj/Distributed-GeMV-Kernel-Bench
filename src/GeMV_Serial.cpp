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
#include <sys/resource.h>
#include <chrono>

int main(int argc, char* argv[]) {
    if(argc<3) {std::cerr << "not enough arguments (must provide matrix dimensions (num rows, num cols))" << std::endl; exit(-1);}

    uint32_t xDim=std::stoul(argv[1]); // num matrix cols
    uint32_t yDim=std::stoul(argv[2]); // num matrix rows

    uint8_t numThreads = std::min(std::thread::hardware_concurrency()==0 ? 1 : std::thread::hardware_concurrency(), yDim);
    numThreads = 6;

    std::float16_t** mats = new std::float16_t*[NUM_MATRICES];
    std::float16_t* vec = new std::float16_t[xDim];

    std::float16_t* result = new std::float16_t[yDim];

    auto wall_start = std::chrono::steady_clock::now();
    GeMV_Serial(numThreads, xDim, yDim, mats, vec, result);
    auto wall_end = std::chrono::steady_clock::now();
    double T_wall = std::chrono::duration<double>(wall_end - wall_start).count();

    std::ofstream out("./out/output.vec", std::ios::binary | std::ios::out | std::ios::trunc);
    if (!out.is_open()) {
        std::cerr << "Failed to open output.vec for writing.\n";
        return -1;
    }
    std::size_t bytes_to_write = yDim * sizeof(std::float16_t);
    out.write(reinterpret_cast<const char*>(result), bytes_to_write);
    if(!out) {
        std::cerr << "Error occurred while writing to output.vec.\n";
    }

    struct rusage resourceUsage;
    if(getrusage(RUSAGE_SELF, &resourceUsage)) {
        std::cerr << "Error occurred while getting maximum RAM usage statistic\n";
    }
    std::cout << "Maximum RAM Usage (bytes): " << resourceUsage.ru_maxrss*1000 << std::endl;
    double T_user = timeval_to_sec(resourceUsage.ru_utime);
    double T_sys  = timeval_to_sec(resourceUsage.ru_stime);
    double T_active = T_user + T_sys;

    std::cout << "Wall Clock Time (seconds): " << T_wall << std::endl;
    std::cout << "User Time (seconds): " << T_user << std::endl;
    std::cout << "System Time (seconds): " << T_sys << std::endl;

    double avg_idle_time = T_wall - (T_active / static_cast<int>(numThreads));
    std::cout << "Idle Time (seconds): " << avg_idle_time << std::endl;

    double bandwidth = (xDim*yDim*sizeof(std::float16_t)*NUM_MATRICES)/(T_wall*1e9);
    std::cout << "Achieved Memory Bandwidth (GB/s): " << bandwidth << std::endl;

    delete[] mats;
    delete[] vec;
    delete[] result;
}
