#ifndef PROCESSOR_H
#define PROCESSOR_H

#include <vector>
#include <string>
#include <chrono>

struct ProcessingResult {
    std::chrono::system_clock::time_point timestamp;
    float mean_value;
    float variance;
    float max_amplitude;
    float anomaly_score;
    std::string classification;
    int batch_size;
    long long processing_time_us;
};

struct ProcessingStats {
    int total_processed;
    double average_latency_ms;
    double throughput_eps;
};

class SignalProcessor {
public:
    SignalProcessor(int batch_size = 256, int signal_length = 1024);
    ~SignalProcessor();
    
    void addEvent(const std::vector<float>& signal_data);
    void processBatch();
    void flush();
    
    std::vector<ProcessingResult> getResults() const;
    ProcessingStats getStats() const;
    void printStatistics() const;
    
private:
    ProcessingResult analyzeFeatures(const std::vector<float>& features, int num_windows);
    
    int batch_size_;
    int signal_length_;
    std::vector<std::vector<float>> buffer_;
    std::vector<float> smoothing_kernel_;
    
    std::vector<ProcessingResult> results_;
    int processed_count_;
    double total_latency_ms_;
};

#endif // PROCESSOR_H