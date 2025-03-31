#include <iostream>
#include <cuda_runtime.h>

int main() {
    // Get the number of CUDA devices
    int deviceCount;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    if (deviceCount == 0) {
        std::cout << "No CUDA devices found!" << std::endl;
        return 1;
    }

    for (int i = 0; i < deviceCount; ++i) {
        cudaDeviceProp deviceProps;
        
        // Get properties of the CUDA device
        cudaGetDeviceProperties(&deviceProps, i);

        // Print GPU name
        std::cout << "GPU " << i << ": " << deviceProps.name << std::endl;

        // Print CUDA compute capability (version)
        std::cout << "CUDA Compute Capability: " 
                  << deviceProps.major << "." << deviceProps.minor << std::endl;
    }

    return 0;
}
