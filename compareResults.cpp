#include <iostream>
#include <stdfloat>
#include <fstream>

int main() {
    std::ifstream f1("output.vec", std::ios::binary);
    std::ifstream f2("oracle.vec", std::ios::binary);

    if (!f1 || !f2) {
        std::cerr << "Failed to open one or both files.\n";
        return -1;
    }

    const std::float16_t errorTolorance=0.005f16;
    std::float16_t v1, v2;
    bool succeeded=true;
    for(int i=0; i<32768; i++) {
        f1.read(reinterpret_cast<char*>(&v1), sizeof(std::float16_t));
        f2.read(reinterpret_cast<char*>(&v2), sizeof(std::float16_t));
        if(v1-v2>errorTolorance) {
            std::cout << "Mismatch at index " << i << ": "
                      << static_cast<float>(v1) << " != "
                      << static_cast<float>(v2) << '\n';
            succeeded = false;
        }
    }
    if(succeeded) {std::cout << "No discrepancies" << std::endl; return 0;}
    return 1;
}