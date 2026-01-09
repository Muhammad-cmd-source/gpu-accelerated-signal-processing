#include "cuda_kernels.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// GPU kernel for batch signal processing
__global__ void batchProcessKernel(const float* input, float* output, 
                                    int batch_size, int signal_len) {
    int batch_idx = blockIdx.y;
    int signal_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (batch_idx < batch_size && signal_idx < signal_len) {
        int idx = batch_idx * signal_len + signal_idx;
        
        // Normalize and apply non-linearity
        float val = input[idx];
        val = (val - 0.5f) * 2.0f;  // Normalize to [-1, 1]
        val = tanhf(val);            // Non-linear activation
        
        output[idx] = val;
    }
}

// GPU kernel for convolution (signal smoothing)
__global__ void convolutionKernel(const float* input, float* output, 
                                   const float* kernel, int signal_len, int kernel_len) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < signal_len) {
        float sum = 0.0f;
        int half_kernel = kernel_len / 2;
        
        for (int k = 0; k < kernel_len; k++) {
            int signal_idx = idx - half_kernel + k;
            if (signal_idx >= 0 && signal_idx < signal_len) {
                sum += input[signal_idx] * kernel[k];
            }
        }
        output[idx] = sum;
    }
}

// GPU kernel for feature extraction
__global__ void featureExtractionKernel(const float* input, float* features, 
                                        int signal_len, int window_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int num_windows = (signal_len - window_size) / window_size + 1;
    
    if (idx < num_windows) {
        int start = idx * window_size;
        float mean = 0.0f;
        float max_val = -1e9f;
        float min_val = 1e9f;
        
        // Calculate statistics for window
        for (int i = 0; i < window_size && (start + i) < signal_len; i++) {
            float val = input[start + i];
            mean += val;
            max_val = fmaxf(max_val, val);
            min_val = fminf(min_val, val);
        }
        mean /= window_size;
        
        // Calculate variance
        float variance = 0.0f;
        for (int i = 0; i < window_size && (start + i) < signal_len; i++) {
            float diff = input[start + i] - mean;
            variance += diff * diff;
        }
        variance /= window_size;
        
        // Store features: mean, variance, max, min, range
        features[idx * 5 + 0] = mean;
        features[idx * 5 + 1] = variance;
        features[idx * 5 + 2] = max_val;
        features[idx * 5 + 3] = min_val;
        features[idx * 5 + 4] = max_val - min_val;
    }
}

// Host function: Process batch of signals on GPU
extern "C" void processSignalBatchGPU(const float* h_input, float* h_output, 
                                      int batch_size, int signal_len) {
    float *d_input, *d_output;
    size_t data_size = batch_size * signal_len * sizeof(float);
    
    // Allocate GPU memory
    CUDA_CHECK(cudaMalloc(&d_input, data_size));
    CUDA_CHECK(cudaMalloc(&d_output, data_size));
    
    // Copy input to GPU
    CUDA_CHECK(cudaMemcpy(d_input, h_input, data_size, cudaMemcpyHostToDevice));
    
    // Configure kernel launch
    dim3 blockDim(256);
    dim3 gridDim((signal_len + blockDim.x - 1) / blockDim.x, batch_size);
    
    // Launch kernel
    batchProcessKernel<<<gridDim, blockDim>>>(d_input, d_output, batch_size, signal_len);
    CUDA_CHECK(cudaGetLastError());
    
    // Copy result back to CPU
    CUDA_CHECK(cudaMemcpy(h_output, d_output, data_size, cudaMemcpyDeviceToHost));
    
    // Free GPU memory
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
}

// Host function: Extract features from signal
extern "C" void extractFeaturesGPU(const float* h_input, float* h_features, 
                                   int signal_len, int window_size) {
    float *d_input, *d_features;
    int num_windows = (signal_len - window_size) / window_size + 1;
    
    CUDA_CHECK(cudaMalloc(&d_input, signal_len * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_features, num_windows * 5 * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_input, h_input, signal_len * sizeof(float), 
                         cudaMemcpyHostToDevice));
    
    int blockSize = 256;
    int gridSize = (num_windows + blockSize - 1) / blockSize;
    
    featureExtractionKernel<<<gridSize, blockSize>>>(d_input, d_features, 
                                                      signal_len, window_size);
    CUDA_CHECK(cudaGetLastError());
    
    CUDA_CHECK(cudaMemcpy(h_features, d_features, num_windows * 5 * sizeof(float), 
                         cudaMemcpyDeviceToHost));
    
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_features));
}

// Host function: Apply convolution smoothing
extern "C" void applyConvolutionGPU(const float* h_input, float* h_output, 
                                    const float* h_kernel, int signal_len, int kernel_len) {
    float *d_input, *d_output, *d_kernel;
    
    CUDA_CHECK(cudaMalloc(&d_input, signal_len * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, signal_len * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_kernel, kernel_len * sizeof(float)));
    
    CUDA_CHECK(cudaMemcpy(d_input, h_input, signal_len * sizeof(float), 
                         cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_kernel, h_kernel, kernel_len * sizeof(float), 
                         cudaMemcpyHostToDevice));
    
    int blockSize = 256;
    int gridSize = (signal_len + blockSize - 1) / blockSize;
    
    convolutionKernel<<<gridSize, blockSize>>>(d_input, d_output, d_kernel, 
                                                signal_len, kernel_len);
    CUDA_CHECK(cudaGetLastError());
    
    CUDA_CHECK(cudaMemcpy(h_output, d_output, signal_len * sizeof(float), 
                         cudaMemcpyDeviceToHost));
    
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_kernel));
}

// Get GPU device information
extern "C" void printGPUInfo() {
    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    
    printf("\n=== GPU Information ===\n");
    printf("Found %d CUDA device(s)\n\n", deviceCount);
    
    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, i));
        
        printf("Device %d: %s\n", i, prop.name);
        printf("  Compute Capability: %d.%d\n", prop.major, prop.minor);
        printf("  Total Memory: %.2f GB\n", prop.totalGlobalMem / 1024.0 / 1024.0 / 1024.0);
        printf("  Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
        printf("  Multiprocessors: %d\n", prop.multiProcessorCount);
        printf("  CUDA Cores: ~%d\n", prop.multiProcessorCount * 
               (prop.major == 6 ? 128 : 64)); // Estimate for Pascal
        printf("  Clock Rate: %.2f MHz\n", prop.clockRate / 1000.0);
        printf("======================\n\n");
    }
}