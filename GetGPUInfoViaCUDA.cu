/**
 * ***********************************************************************************************************************
 * Job script for CUDA program to verify GPU and CUDA setup
 * -----------------------------------------------------------------------------------------------------------------------
 * Author: HarmonicReflux
 * Purpose: Verify CUDA installation and detect visible Nvidia GPUs
 * Description: This CUDA program verifies that `nvcc`, CUDA drivers, and a compatible `gcc` toolchain are correctly configured,
 *              detects all visible Nvidia GPUs, and reports each GPU's name and CUDA compute capability (major.minor).
 *
 * Resources requested:
 *   - CUDA toolkit, compatible driver, and GCC toolchain (for local machine or compute nodes)
 *
 * Notes:
 *   - The program is useful for quick diagnostics on local machines or Slurm clusters.
 *   - It helps confirm that the CUDA toolkit is installed and functional.
 *
 * Output:
 *   - Reports each GPU's name and CUDA compute capability
 * ***********************************************************************************************************************
 */


#include <iostream>
#include <cuda_runtime.h>

int main() {

std::cout  << std::endl;
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

    std::cout << "Number of CUDA devices: " << deviceCount << std::endl;

    for (int i = 0; i < deviceCount; ++i) {
        cudaDeviceProp deviceProps;

        // get properties of the CUDA device
        cudaGetDeviceProperties(&deviceProps, i);

        // print GPU name
        std::cout << "\nGPU " << i << ": " << deviceProps.name << std::endl;

	// print Memory 
	std::cout  <<  "Memory: " << deviceProps.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;

        // print CUDA compute capability (version)
        std::cout << "CUDA compute capability is: " 
                  << deviceProps.major << "." << deviceProps.minor << std::endl;
    }

    std::cout << std::endl;
    std::cout << "Script run succesfully. Exiting now.\n"  << std::endl;

    return 0;
}
