#include "processor.h"
#include "cuda_kernels.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <random>

using namespace std::chrono;

void processCPU(const float* input, float* output, int size) {
    for (int i = 0; i < size; i++) {
        float val = (input[i] - 0.5f) * 2.0f;
        output[i] = std::tanh(val);
    }
}

int main() {
    std::cout << "\n=== GPU Performance Test (Large Workload) ===\n\n";
    
    printGPUInfo();
    
    // LARGER test - this will show GPU advantage!
    const int BATCH_SIZE = 1024;  // Increased from 256
    const int SIGNAL_LEN = 4096;  // Increased from 1024
    const int TEST_SIZE = BATCH_SIZE * SIGNAL_LEN;
    
    std::cout << "Test Configuration:\n";
    std::cout << "  Batch Size: " << BATCH_SIZE << "\n";
    std::cout << "  Signal Length: " << SIGNAL_LEN << "\n";
    std::cout << "  Total Elements: " << TEST_SIZE << "\n\n";
    
    std::vector<float> input(TEST_SIZE);
    std::vector<float> gpu_output(TEST_SIZE);
    std::vector<float> cpu_output(TEST_SIZE);
    
    // Generate test data
    for (int i = 0; i < TEST_SIZE; i++) {
        input[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    
    // Warmup
    processSignalBatchGPU(input.data(), gpu_output.data(), BATCH_SIZE, SIGNAL_LEN);
    
    // GPU timing
    std::cout << "Running GPU benchmark (20 iterations)...\n";
    auto gpu_start = high_resolution_clock::now();
    for (int iter = 0; iter < 20; iter++) {
        processSignalBatchGPU(input.data(), gpu_output.data(), BATCH_SIZE, SIGNAL_LEN);
    }
    auto gpu_end = high_resolution_clock::now();
    auto gpu_ms = duration_cast<milliseconds>(gpu_end - gpu_start).count();
    
    // CPU timing
    std::cout << "Running CPU benchmark (20 iterations)...\n";
    auto cpu_start = high_resolution_clock::now();
    for (int iter = 0; iter < 20; iter++) {
        processCPU(input.data(), cpu_output.data(), TEST_SIZE);
    }
    auto cpu_end = high_resolution_clock::now();
    auto cpu_ms = duration_cast<milliseconds>(cpu_end - cpu_start).count();
    
    double speedup = static_cast<double>(cpu_ms) / gpu_ms;
    double gpu_throughput = (BATCH_SIZE * 20 * 1000.0) / gpu_ms;
    
    std::cout << "\n========================================\n";
    std::cout << "         PERFORMANCE RESULTS\n";
    std::cout << "========================================\n";
    std::cout << "GPU Time:       " << (gpu_ms / 20.0) << " ms/batch\n";
    std::cout << "CPU Time:       " << (cpu_ms / 20.0) << " ms/batch\n";
    std::cout << "GPU Speedup:    " << speedup << "x\n";
    std::cout << "GPU Throughput: " << static_cast<int>(gpu_throughput) << " events/sec\n";
    std::cout << "========================================\n\n";
    
    if (speedup > 1.0) {
        std::cout << "SUCCESS! GPU is " << speedup << "x faster than CPU!\n";
    } else {
        std::cout << "Note: For this workload size, CPU overhead dominates.\n";
        std::cout << "GPU advantage increases with larger batch sizes.\n";
    }
    
    return 0;
}