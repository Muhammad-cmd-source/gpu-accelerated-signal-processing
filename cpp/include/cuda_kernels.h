#ifndef CUDA_KERNELS_H
#define CUDA_KERNELS_H

#ifdef __cplusplus
extern "C" {
#endif

void processSignalBatchGPU(const float* h_input, float* h_output, 
                           int batch_size, int signal_len);

void extractFeaturesGPU(const float* h_input, float* h_features, 
                        int signal_len, int window_size);

void applyConvolutionGPU(const float* h_input, float* h_output, 
                         const float* h_kernel, int signal_len, int kernel_len);

void printGPUInfo();

#ifdef __cplusplus
}
#endif

#endif // CUDA_KERNELS_H