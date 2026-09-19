#include <iostream>
#include <fstream>
#include <cmath>
#include <stdfloat>

int main() {
    std::ifstream f1("output.vec", std::ios::binary);
    std::ifstream f2("oracle.vec", std::ios::binary);

    if (!f1 || !f2) {
        std::cerr << "Failed to open input files.\n";
        return -1;
    }

    constexpr float absTolerance = 0.05f;
    constexpr float relTolerance = 0.01f;
    bool succeeded = true;

    for (int i = 0; i < 32768; i++) {
        std::float16_t v1{}, v2{};
        if (!f1.read(reinterpret_cast<char*>(&v1), sizeof(std::float16_t)) ||
            !f2.read(reinterpret_cast<char*>(&v2), sizeof(std::float16_t))) {
            std::cerr << "Premature EOF or read failure at index " << i << '\n';
            return 1;
        }

        float val1 = static_cast<float>(v1);
        float val2 = static_cast<float>(v2);
        float diff = std::abs(val1 - val2);
        float maxAllowed = std::max(absTolerance, relTolerance * std::abs(val2));

        if (std::isnan(val1) || std::isnan(val2) || diff > maxAllowed) {
            std::cout << "Mismatch at index " << i << ": "
                      << val1 << " != " << val2
                      << " (diff: " << diff << ")\n";
            succeeded = false;
        }
    }

    if (succeeded) {
        std::cout << "No Discrepancies" << std::endl;
        return 0;
    }
    return 1;
}