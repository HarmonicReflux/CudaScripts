/**
 *
 * A minimal CUDA program that:
 * - Verifies that `nvcc`, CUDA drivers, and a compatible `gcc` toolchain are correctly configured
 * - Detects all visible Nvidia GPUs
 * - Reports each GPU's name and CUDA compute capability (major.minor)
 *
 * A successful compile and run confirms that the CUDA toolkit is installed and functional.
 * Useful for quick diagnostics on local machines or compute nodes (e.g. Slurm clusters).
 *
 * Author: HarmonicReflux
 */


#include <iostream>
#include <cuda_runtime.h>

int main() {

std::cout 
    << "Detecting Nvidia GPUs and their CUDA compute capability from an nvcc compiled CUDA script.\n"
    << "If this script compiles from source, and devices are detected successfully, "
    << "it means nvcc, and its dependent gcc compiler, as well as CUDA "
    << "are set up correctly."
    << std::endl;
std::cout  << std::endl;

std::cout << "Scanning visible GPUs..." << std::endl;

    // get the number of CUDA devices
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
        
        // get properties of the CUDA device
        cudaGetDeviceProperties(&deviceProps, i);

        // print GPU name
        std::cout << "GPU " << i << ": " << deviceProps.name << std::endl;

        // print CUDA compute capability (version)
        std::cout << "CUDA compute capability is: " 
                  << deviceProps.major << "." << deviceProps.minor << std::endl;
    }

    std::cout << std::endl;
    std::cout << "Script run succesfully. Exiting now."  << std::endl;

    return 0;
}
