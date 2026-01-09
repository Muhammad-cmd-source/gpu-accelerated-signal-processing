#include "processor.h"
#include "cuda_kernels.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>
#include <iostream>

using namespace std::chrono;

SignalProcessor::SignalProcessor(int batch_size, int signal_length)
    : batch_size_(batch_size), signal_length_(signal_length),
      processed_count_(0), total_latency_ms_(0) {
    
    buffer_.reserve(batch_size);
    
    // Initialize Gaussian smoothing kernel
    int kernel_size = 7;
    smoothing_kernel_.resize(kernel_size);
    float sigma = 1.0f;
    float sum = 0.0f;
    
    for (int i = 0; i < kernel_size; i++) {
        int x = i - kernel_size / 2;
        smoothing_kernel_[i] = std::exp(-(x * x) / (2.0f * sigma * sigma));
        sum += smoothing_kernel_[i];
    }
    
    // Normalize kernel
    for (auto& val : smoothing_kernel_) {
        val /= sum;
    }
    
    std::cout << "SignalProcessor initialized:" << std::endl;
    std::cout << "  Batch size: " << batch_size << std::endl;
    std::cout << "  Signal length: " << signal_length << std::endl;
    std::cout << "  Smoothing kernel size: " << kernel_size << std::endl;
}

SignalProcessor::~SignalProcessor() {
    if (processed_count_ > 0) {
        printStatistics();
    }
}

void SignalProcessor::addEvent(const std::vector<float>& signal_data) {
    if (signal_data.size() != static_cast<size_t>(signal_length_)) {
        std::cerr << "Warning: Signal size mismatch. Expected " << signal_length_
                  << ", got " << signal_data.size() << std::endl;
        return;
    }
    
    buffer_.push_back(signal_data);
    
    if (buffer_.size() >= static_cast<size_t>(batch_size_)) {
        processBatch();
    }
}

void SignalProcessor::processBatch() {
    if (buffer_.empty()) return;
    
    auto start_time = high_resolution_clock::now();
    
    // Prepare data for GPU processing
    std::vector<float> input_batch;
    input_batch.reserve(buffer_.size() * signal_length_);
    
    for (const auto& signal : buffer_) {
        input_batch.insert(input_batch.end(), signal.begin(), signal.end());
    }
    
    std::vector<float> output_batch(input_batch.size());
    
    // GPU batch processing
    processSignalBatchGPU(input_batch.data(), output_batch.data(), 
                         buffer_.size(), signal_length_);
    
    // Apply convolution smoothing on first signal (demonstration)
    std::vector<float> smoothed(signal_length_);
    applyConvolutionGPU(output_batch.data(), smoothed.data(),
                       smoothing_kernel_.data(), signal_length_, 
                       smoothing_kernel_.size());
    
    // Extract features from smoothed signal
    int window_size = 64;
    int num_windows = (signal_length_ - window_size) / window_size + 1;
    std::vector<float> features(num_windows * 5);
    extractFeaturesGPU(smoothed.data(), features.data(), 
                      signal_length_, window_size);
    
    // CPU-side numerical analysis
    ProcessingResult result = analyzeFeatures(features, num_windows);
    result.batch_size = buffer_.size();
    
    auto end_time = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(end_time - start_time);
    result.processing_time_us = duration.count();
    
    // Update statistics
    processed_count_ += buffer_.size();
    total_latency_ms_ += duration.count() / 1000.0;
    
    // Store result
    results_.push_back(result);
    if (results_.size() > 100) {
        results_.erase(results_.begin());
    }
    
    buffer_.clear();
    
    // Print progress every 10 batches
    if (results_.size() % 10 == 0) {
        std::cout << "Processed " << processed_count_ << " events, "
                  << "Avg latency: " << (total_latency_ms_ / results_.size())
                  << " ms/batch" << std::endl;
    }
}

ProcessingResult SignalProcessor::analyzeFeatures(const std::vector<float>& features, 
                                                   int num_windows) {
    ProcessingResult result;
    result.timestamp = std::chrono::system_clock::now();
    
    // Statistical analysis of extracted features
    std::vector<float> means, variances, ranges;
    
    for (int i = 0; i < num_windows; i++) {
        means.push_back(features[i * 5 + 0]);
        variances.push_back(features[i * 5 + 1]);
        ranges.push_back(features[i * 5 + 4]);
    }
    
    // Calculate overall statistics
    result.mean_value = std::accumulate(means.begin(), means.end(), 0.0f) / means.size();
    result.variance = std::accumulate(variances.begin(), variances.end(), 0.0f) / variances.size();
    result.max_amplitude = *std::max_element(ranges.begin(), ranges.end());
    
    // Detect anomalies using statistical thresholds
    float mean_variance = result.variance;
    int anomaly_count = 0;
    for (float var : variances) {
        if (var > mean_variance * 3.0f) {
            anomaly_count++;
        }
    }
    result.anomaly_score = static_cast<float>(anomaly_count) / num_windows;
    
    // Classification based on signal characteristics
    if (result.anomaly_score > 0.15f) {
        result.classification = "ANOMALOUS";
    } else if (result.variance > 0.5f) {
        result.classification = "HIGH_ACTIVITY";
    } else if (result.variance < 0.1f) {
        result.classification = "LOW_ACTIVITY";
    } else {
        result.classification = "NORMAL";
    }
    
    return result;
}

void SignalProcessor::flush() {
    if (!buffer_.empty()) {
        processBatch();
    }
}

std::vector<ProcessingResult> SignalProcessor::getResults() const {
    return results_;
}

ProcessingStats SignalProcessor::getStats() const {
    ProcessingStats stats;
    stats.total_processed = processed_count_;
    stats.average_latency_ms = results_.empty() ? 0 : 
        total_latency_ms_ / results_.size();
    stats.throughput_eps = stats.average_latency_ms > 0 ?
        (batch_size_ * 1000.0) / stats.average_latency_ms : 0;
    
    return stats;
}

void SignalProcessor::printStatistics() const {
    auto stats = getStats();
    std::cout << "\n╔════════════════════════════════════╗" << std::endl;
    std::cout << "║   Processing Statistics           ║" << std::endl;
    std::cout << "╚════════════════════════════════════╝" << std::endl;
    std::cout << "Total Events Processed: " << stats.total_processed << std::endl;
    std::cout << "Average Latency: " << stats.average_latency_ms << " ms/batch" << std::endl;
    std::cout << "Throughput: " << static_cast<int>(stats.throughput_eps) << " events/second" << std::endl;
    std::cout << "════════════════════════════════════\n" << std::endl;
}