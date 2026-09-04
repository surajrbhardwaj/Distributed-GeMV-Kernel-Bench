#include <stdfloat>
#include <iostream>
#include <cstdint>
#include <vector>
#include <optional>
#include <thread>
#include <functional>

/// @param work the number of dot products to be calculated
/// @param offset the number row index after which we must calculate the dot products
/// @param N dimension of the column vector
/// @param matrix matrix represented as a 2d array of elements allocated as a contiguious block of memory
/// @param vector column vector represented as an array for cache coherence
/// @param results the output column vector represented as an array for cache coherence
/// @return void
void GeMV_kernel(const uint32_t work, const uint32_t offset, const uint32_t N, const std::float16_t* matrix, const std::float16_t* vector, std::float16_t* results) {
    std::float32_t sum;
    for(uint32_t w=0; w<work; w++) {
        sum = 0.0f;
        for(uint32_t i=0; i<N; i++) {
            sum += static_cast<std::float32_t>(matrix[(offset+w)*N + i])*static_cast<std::float32_t>(vector[i]);
        }
        results[offset+w] = static_cast<std::float16_t>(sum);
    }
}

/*
/// @param matrix matrix represented as a 2d array of elements (each subarray represents a row in the matrix)
/// @param vector column vector represented as an array for cache coherence
/// @return an optional column vector with size N (matrix.size()) returns std::nullopt on invalid input
std::optional<std::vector<std::float16_t>> GeMV_SIMD(const uint8_t numThreads, const std::vector<std::vector<std::float16_t>>& matrix, const std::vector<std::float16_t>& vector) {
    const uint32_t N = matrix.size();
    const uint32_t M = vector.size();
    if(N==0 || matrix[0].size()!=M || numThreads==0 || numThreads>N) {return std::nullopt;}

    uint32_t workPerThread = N/numThreads;
    if(workPerThread==0) {workPerThread = 1;}

    std::vector<std::float16_t> result(N, 0.0f16);

    std::vector<std::thread> threads;
    threads.reserve(numThreads);
    for(uint8_t i=0; i<numThreads-1; i++) {
        threads.push_back(std::thread(GeMV_kernel, workPerThread, i*workPerThread, M, std::cref(matrix), std::cref(vector), std::ref(result)));
    }
    uint32_t offset = (numThreads-1)*workPerThread;
    threads.push_back(std::thread(GeMV_kernel, N-offset, offset, M, std::cref(matrix), std::cref(vector), std::ref(result)));
    
    for(uint8_t i=0; i<numThreads; i++) {
        threads[i].join();
    }
    
    return result;
}

/// @param matrix matrix represented as a 2d array of elements (each subarray represents a row in the matrix)
/// @param vector column vector represented as an array for cache coherence
/// @return an optional column vector with size N (matrix.size()) returns std::nullopt on invalid input
std::optional<std::vector<std::float16_t>> GeMV(const std::vector<std::vector<std::float16_t>>& matrix, const std::vector<std::float16_t>& vector) {
    return GeMV_SIMD(1, matrix, vector);
}
*/

enum class ExecutionState: uint8_t {
    WORKING,
    FINISHED,
    TERMINATING
};

struct ThreadData {
    uint32_t M;
    uint32_t N;
    uint32_t work;
    uint32_t offset;
    std::float16_t* matrixPtr{nullptr};
    std::float16_t* vectorPtr{nullptr};
    std::float16_t* resultPtr{nullptr};
    std::atomic<ExecutionState> flag{ExecutionState::FINISHED};
    
    ThreadData() {}
    ThreadData(uint32_t M, uint32_t N, uint32_t work, uint32_t offset, std::float16_t* matrixPtr, std::float16_t* vectorPtr, std::float16_t* resultPtr, ExecutionState flag): M(M), N(N), work(work), offset(offset), matrixPtr(matrixPtr), vectorPtr(vectorPtr), resultPtr(resultPtr) {
        this->flag.store(flag);
    }
    ThreadData& operator=(ThreadData&& other) noexcept {
        if(this != &other) {
            M = other.M;
            N = other.N;
            work = other.work;
            offset = other.offset;
            matrixPtr = other.matrixPtr;
            vectorPtr = other.vectorPtr;
            resultPtr = other.resultPtr;
            flag.store(other.flag.load());
        }
        return *this;
    }
};

void consumer(ThreadData* data) {
    while(data->flag.load() != ExecutionState::TERMINATING) {
        while(data->flag.load()==ExecutionState::FINISHED) {data->flag.wait(ExecutionState::FINISHED);}
        if(data->flag.load()==ExecutionState::TERMINATING) {break;}
        GeMV_kernel(data->work, data->offset, data->N, data->matrixPtr, data->vectorPtr, data->resultPtr); // need to add the M and N to the params
        data->flag.store(ExecutionState::FINISHED);
        data->flag.notify_one();
    }
}

int main() {
    uint8_t numThreads = std::thread::hardware_concurrency()==0 ? 1 : std::thread::hardware_concurrency();
    // temporary
    numThreads = 2;

    std::thread threads[numThreads];
    ThreadData threadParams[numThreads];
    for(uint8_t i=0; i<numThreads; i++) {
        threadParams[i] = {0, 0, 0, 0, nullptr, nullptr, nullptr, ExecutionState::FINISHED};
        threads[i] = std::thread(consumer, &threadParams[i]);
    }

    const uint32_t M = 2; // num matrix rows
    const uint32_t N = 3; // num matrix cols
    std::float16_t* mat = new std::float16_t[M*N]{0.5f16, -0.2f16, 0.1f16, 1.5f16, 0.0f16, -0.5f16};
    std::float16_t* vec = new std::float16_t[N]{1.0f16, 2.0f16, -1.0f16};

    std::float16_t* result = new std::float16_t[M]{0.0f16, 0.0f16};

    uint32_t workPerThread = M/numThreads;
    for(uint8_t i=0; i<numThreads; i++) {
        threadParams[i].M = M;
        threadParams[i].N = N;
        threadParams[i].work = i==numThreads-1 ? M-i*workPerThread : workPerThread;
        threadParams[i].offset = i*workPerThread;
        threadParams[i].matrixPtr = mat;
        threadParams[i].vectorPtr = vec;
        threadParams[i].resultPtr = result;
        threadParams[i].flag.store(ExecutionState::WORKING);
        threadParams[i].flag.notify_one();
    }

    for(uint8_t i=0; i<numThreads; i++) {
        threadParams[i].flag.wait(ExecutionState::WORKING);
    }
    
    for(uint8_t i=0; i<M; i++) {
        std::cout << result[i] << std::endl;
    }

    for(uint8_t i=0; i<numThreads; i++) {
        threadParams[i].flag.store(ExecutionState::TERMINATING);
        threadParams[i].flag.notify_one();
    }
    for(uint8_t i=0; i<numThreads; i++) {
        threads[i].join();
    }

    delete[] mat;
    delete[] vec;
    delete[] result;

    /*
    std::vector<std::vector<std::float16_t>> mat = {{0.5f16, -0.2f16, 0.1f16}, {1.5f16, 0.0f16, -0.5f16}};
    std::vector<std::float16_t> vec = {1.0f16, 2.0f16, -1.0f16};

    {
    auto res = GeMV(mat, vec);

    if(res.has_value()) {
        for(int i=0; i<res->size(); i++) {
            std::cout << (*res)[i] << std::endl;
        }
    }
    else {
        std::cout << "operation failed (dimensionality issue)" << std::endl;
    }
    }

    {
    auto res = GeMV_SIMD(2, mat, vec);

    if(res.has_value()) {
        for(int i=0; i<res->size(); i++) {
            std::cout << (*res)[i] << std::endl;
        }
    }
    else {
        std::cout << "operation failed (dimensionality issue)" << std::endl;
    }
    }
    */
}