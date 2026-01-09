#include "processor.h"
#include "cuda_kernels.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <random>

using namespace std::chrono;

// CPU version for comparison
void processCPU(const float* input, float* output, int size) {
    for (int i = 0; i < size; i++) {
        float val = (input[i] - 0.5f) * 2.0f;
        output[i] = std::tanh(val);
    }
}

// Generate synthetic signal data
std::vector<float> generateSignal(int length, const std::string& type = "normal") {
    std::vector<float> signal(length);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.5f, 0.2f);
    
    if (type == "normal") {
        // Normal signal with some noise
        for (int i = 0; i < length; i++) {
            float t = static_cast<float>(i) / length;
            signal[i] = 0.5f + 0.3f * std::sin(2 * 3.14159f * 2 * t) + 
                       0.1f * dist(gen);
        }
    } else if (type == "high_activity") {
        // High activity signal
        for (int i = 0; i < length; i++) {
            float t = static_cast<float>(i) / length;
            signal[i] = 0.5f + 
                       0.2f * std::sin(2 * 3.14159f * 5 * t) +
                       0.2f * std::sin(2 * 3.14159f * 10 * t) +
                       0.2f * dist(gen);
        }
    } else { // anomalous
        // Anomalous signal with spikes
        for (int i = 0; i < length; i++) {
            float t = static_cast<float>(i) / length;
            signal[i] = 0.5f + 0.2f * std::sin(2 * 3.14159f * t);
            if (i % 100 == 0) {
                signal[i] += dist(gen) * 2.0f; // Add spikes
            }
        }
    }
    
    return signal;
}

int main(int argc, char** argv) {
    std::cout << "\n";
    std::cout << "╔════════════════════════════════════════════════╗\n";
    std::cout << "║  High-Performance Data Processing Pipeline    ║\n";
    std::cout << "║  GPU-Accelerated Signal Processing Benchmark  ║\n";
    std::cout << "╚════════════════════════════════════════════════╝\n";
    std::cout << "\n";
    
    // Print GPU information
    printGPUInfo();
    
    // Configuration
    const int SIGNAL_LEN = 1024;
    const int BATCH_SIZE = 256;
    const int NUM_BATCHES = 100;
    const int TOTAL_EVENTS = BATCH_SIZE * NUM_BATCHES;
    
    std::cout << "Benchmark Configuration:\n";
    std::cout << "  Signal Length: " << SIGNAL_LEN << "\n";
    std::cout << "  Batch Size: " << BATCH_SIZE << "\n";
    std::cout << "  Number of Batches: " << NUM_BATCHES << "\n";
    std::cout << "  Total Events: " << TOTAL_EVENTS << "\n";
    std::cout << "\n";
    
    // ========================================
    // TEST 1: GPU vs CPU Speedup
    // ========================================
    std::cout << "╔════════════════════════════════════════════════╗\n";
    std::cout << "║  TEST 1: GPU vs CPU Performance Comparison    ║\n";
    std::cout << "╚════════════════════════════════════════════════╝\n";
    
    const int TEST_SIZE = BATCH_SIZE * SIGNAL_LEN;
    std::vector<float> test_input(TEST_SIZE);
    std::vector<float> gpu_output(TEST_SIZE);
    std::vector<float> cpu_output(TEST_SIZE);
    
    // Generate test data
    for (int i = 0; i < TEST_SIZE; i++) {
        test_input[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    
    // GPU timing
    std::cout << "\nRunning GPU test (10 iterations)..." << std::endl;
    auto gpu_start = high_resolution_clock::now();
    for (int iter = 0; iter < 10; iter++) {
        processSignalBatchGPU(test_input.data(), gpu_output.data(), 
                             BATCH_SIZE, SIGNAL_LEN);
    }
    auto gpu_end = high_resolution_clock::now();
    auto gpu_time = duration_cast<milliseconds>(gpu_end - gpu_start).count();
    
    // CPU timing
    std::cout << "Running CPU test (10 iterations)..." << std::endl;
    auto cpu_start = high_resolution_clock::now();
    for (int iter = 0; iter < 10; iter++) {
        processCPU(test_input.data(), cpu_output.data(), TEST_SIZE);
    }
    auto cpu_end = high_resolution_clock::now();
    auto cpu_time = duration_cast<milliseconds>(cpu_end - cpu_start).count();
    
    double speedup = static_cast<double>(cpu_time) / gpu_time;
    double gpu_throughput = (BATCH_SIZE * 10 * 1000.0) / gpu_time;
    
    std::cout << "\n┌─────────────────────────────────────────┐\n";
    std::cout << "│ Performance Results                     │\n";
    std::cout << "├─────────────────────────────────────────┤\n";
    std::cout << "│ GPU Time:       " << (gpu_time / 10.0) << " ms/batch\n";
    std::cout << "│ CPU Time:       " << (cpu_time / 10.0) << " ms/batch\n";
    std::cout << "│ Speedup:        " << speedup << "x\n";
    std::cout << "│ GPU Throughput: " << static_cast<int>(gpu_throughput) << " events/sec\n";
    std::cout << "└─────────────────────────────────────────┘\n";
    
    // ========================================
    // TEST 2: Full Pipeline Processing
    // ========================================
    std::cout << "\n╔════════════════════════════════════════════════╗\n";
    std::cout << "║  TEST 2: Full Pipeline Processing             ║\n";
    std::cout << "╚════════════════════════════════════════════════╝\n\n";
    
    SignalProcessor processor(BATCH_SIZE, SIGNAL_LEN);
    
    std::cout << "Processing " << TOTAL_EVENTS << " events...\n";
    
    auto pipeline_start = high_resolution_clock::now();
    
    // Process batches with mixed signal types
    for (int batch = 0; batch < NUM_BATCHES; batch++) {
        for (int i = 0; i < BATCH_SIZE; i++) {
            std::string type;
            if (i % 10 == 0) type = "anomalous";
            else if (i % 5 == 0) type = "high_activity";
            else type = "normal";
            
            auto signal = generateSignal(SIGNAL_LEN, type);
            processor.addEvent(signal);
        }
        
        // Progress indicator
        if ((batch + 1) % 20 == 0) {
            std::cout << "  Progress: " << (batch + 1) << "/" << NUM_BATCHES 
                     << " batches (" << ((batch + 1) * 100 / NUM_BATCHES) << "%)\n";
        }
    }
    
    processor.flush();
    
    auto pipeline_end = high_resolution_clock::now();
    auto pipeline_time = duration_cast<milliseconds>(pipeline_end - pipeline_start).count();
    
    // ========================================
    // TEST 3: Results Analysis
    // ========================================
    std::cout << "\n╔════════════════════════════════════════════════╗\n";
    std::cout << "║  TEST 3: Results Analysis                     ║\n";
    std::cout << "╚════════════════════════════════════════════════╝\n";
    
    auto results = processor.getResults();
    int normal_count = 0, high_activity_count = 0, anomalous_count = 0;
    
    for (const auto& result : results) {
        if (result.classification == "NORMAL") normal_count++;
        else if (result.classification == "HIGH_ACTIVITY") high_activity_count++;
        else if (result.classification == "ANOMALOUS") anomalous_count++;
    }
    
    std::cout << "\nClassification Results:\n";
    std::cout << "  Normal:        " << normal_count << " batches\n";
    std::cout << "  High Activity: " << high_activity_count << " batches\n";
    std::cout << "  Anomalous:     " << anomalous_count << " batches\n";
    
    // ========================================
    // Final Summary
    // ========================================
    auto stats = processor.getStats();
    double total_throughput = (TOTAL_EVENTS * 1000.0) / pipeline_time;
    
    std::cout << "\n";
    std::cout << "╔════════════════════════════════════════════════╗\n";
    std::cout << "║           FINAL BENCHMARK RESULTS              ║\n";
    std::cout << "╚════════════════════════════════════════════════╝\n";
    std::cout << "\n";
    std::cout << "✓ Total Events Processed:  " << stats.total_processed << "\n";
    std::cout << "✓ Average Latency:         " << stats.average_latency_ms << " ms/batch\n";
    std::cout << "✓ Pipeline Throughput:     " << static_cast<int>(total_throughput) << " events/sec\n";
    std::cout << "✓ GPU Speedup:             " << speedup << "x vs CPU\n";
    std::cout << "✓ Total Processing Time:   " << (pipeline_time / 1000.0) << " seconds\n";
    std::cout << "\n";
    std::cout << "════════════════════════════════════════════════\n";
    std::cout << "Benchmark completed successfully!\n";
    std::cout << "════════════════════════════════════════════════\n\n";
    
    return 0;
}